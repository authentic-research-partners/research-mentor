"""Local embedding model for semantic memory.

Uses Snowflake Arctic Embed v2 (configurable size) via sentence-transformers.
Always runs on CPU — must not compete with vLLM for GPU VRAM.

Arctic Embed v2 uses custom HuggingFace model code (GTE architecture), so loading
requires ``trust_remote_code=True``.  The model's default config enables xformers-based
memory-efficient attention (CUDA-only) and input unpadding (an optimization paired with
xformers).  We override both to standard (eager) attention via ``config_kwargs`` —
mathematically identical output, just without the xformers dependency.  Both flags must
be disabled together: ``unpad_inputs=True`` with ``use_memory_efficient_attention=False``
causes garbage ``position_ids`` in the GTE embedding layer.

Public API is async (``embed_text`` / ``embed_texts``) — runs the synchronous
sentence-transformers encode in a thread pool so it never blocks the event loop.
PyTorch releases the GIL during the forward pass, so threads give real concurrency.

Vector storage and search is handled by sqlite-vec (core dependency).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger
from sqlite_vec import serialize_float32

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def _fix_gte_buffers(model: SentenceTransformer) -> None:
    """Reinitialize GTE model buffers corrupted by weight loading.

    GTE-architecture models (Arctic Embed v2) use ``persistent=False`` buffers for
    position_ids and rotary cos/sin caches.  With transformers 5.x + torch 2.x, these
    buffers contain uninitialized memory after ``from_pretrained`` loads weights.
    Reinitializing them from scratch fixes the garbage values.

    Safe to call on non-GTE models (no-op) and on mocked models in tests.
    """
    try:
        from typing import Any as _Any

        import torch

        auto_model: _Any = model[0].auto_model
        embeddings_layer: _Any = auto_model.embeddings
        rotary = getattr(embeddings_layer, "rotary_emb", None)
        if rotary is None or not isinstance(rotary, torch.nn.Module):
            return

        max_pos: int = auto_model.config.max_position_embeddings
        embeddings_layer.register_buffer(
            "position_ids", torch.arange(max_pos), persistent=False,
        )
        rotary._set_cos_sin_cache(
            seq_len=rotary.max_seq_len_cached,
            device=rotary.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
    except (AttributeError, TypeError, IndexError):
        logger.debug("Rotary-embedding patch skipped — incompatible internal structure")


def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Load a SentenceTransformer model with all HuggingFace noise suppressed."""
    import logging
    import os
    import warnings

    from sentence_transformers import SentenceTransformer

    # Suppress HF warnings, progress bars, and "malicious code" messages
    _hf_env = {
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    _old_env = {k: os.environ.get(k) for k in _hf_env}
    os.environ.update(_hf_env)

    # Also suppress transformers logger (catches "malicious code" warnings)
    tf_logger = logging.getLogger("transformers")
    old_level = tf_logger.level
    tf_logger.setLevel(logging.ERROR)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = SentenceTransformer(
                model_name,
                device="cpu",
                trust_remote_code=True,
                config_kwargs={
                    "use_memory_efficient_attention": False,
                    "unpad_inputs": False,
                },
            )
    finally:
        tf_logger.setLevel(old_level)
        for k, v in _old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    _fix_gte_buffers(model)

    return model


# Lazy singleton — loaded on first use
_model: SentenceTransformer | None = None
_model_name: str | None = None


def _get_model() -> SentenceTransformer:
    """Get or create the singleton embedding model.

    Always loads on CPU to avoid competing with vLLM for GPU VRAM.
    """
    global _model, _model_name
    if _model is not None:
        return _model

    from research_mentor.config import load_config

    config = load_config()
    _model_name = config.embeddings.model

    # Check if the model is already cached locally
    needs_download = False
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(_model_name, "config.json")
        if not isinstance(cached, str):
            needs_download = True
        else:
            logger.info("Loading embedding model: {} (device=cpu)", _model_name)
    except Exception:
        logger.info("Loading embedding model: {} (device=cpu)", _model_name)

    if needs_download:
        import click

        click.echo()
        click.echo("  Downloading embedding model (~1.2 GB, one-time only)…")
        click.echo("  Stored in: ~/.cache/huggingface/")
        click.echo("  Please wait — this may take a few minutes.")
        click.echo()

    _model = _load_sentence_transformer(_model_name)

    if needs_download:
        import click

        click.echo("  Embedding model ready!")
        click.echo()
        click.echo()

    logger.info(
        "Embedding model loaded: {} (dim={})",
        _model_name,
        _model.get_sentence_embedding_dimension(),
    )
    return _model


def _embed_text_sync(text: str) -> bytes:
    """Generate embedding for a single text string (sync, internal)."""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return serialize_float32(vec.tolist())  # type: ignore[no-any-return]


def _embed_texts_sync(texts: list[str]) -> list[bytes]:
    """Generate embeddings for multiple texts (sync, internal)."""
    if not texts:
        return []
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return [serialize_float32(v.tolist()) for v in vecs]


async def embed_text(text: str) -> bytes:
    """Generate embedding for a single text string.

    Runs in a thread pool — safe to call from async code without blocking.
    Returns sqlite-vec compatible BLOB.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _embed_text_sync, text)


async def embed_texts(texts: list[str]) -> list[bytes]:
    """Generate embeddings for multiple texts (batched for efficiency).

    Runs in a thread pool — safe to call from async code without blocking.
    Returns list of sqlite-vec compatible BLOBs.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _embed_texts_sync, texts)


async def rebuild_embedding_tables(dimension: int | None = None) -> dict[str, int]:
    """Drop and recreate all vec0 tables with the given dimension, then re-embed.

    Called when the embedding model changes (dimension mismatch).
    Preserves source data (conversation_memories, artifact_chunks, retraction_watch)
    and re-embeds everything into fresh vec0 tables.

    Returns counts of re-embedded items per table.
    """
    from research_mentor.config import load_config
    from research_mentor.db.connection import get_db

    if dimension is None:
        # Auto-detect from the model itself — works for any HuggingFace model
        model = _get_model()
        dimension = model.get_sentence_embedding_dimension()
        # Persist discovered dimension to config
        from research_mentor.config import update_user_config

        update_user_config({"embeddings": {"dimension": dimension}})
        logger.info("Auto-detected embedding dimension: {}", dimension)

    logger.info("Rebuilding embedding tables with dimension={}", dimension)

    counts: dict[str, int] = {}

    async with get_db() as db:
        # Mark rebuild as in-progress — cleared only on successful completion.
        # If the process crashes mid-rebuild, the flag stays 'in_progress' and
        # the next startup will detect incomplete embeddings and rebuild again.
        await db.execute(
            "INSERT OR REPLACE INTO app_meta (key, value) VALUES (?, ?)",
            ("embeddings_status", "in_progress"),
        )
        await db.commit()

        # --- memory_embeddings ---
        await db.execute("DROP TABLE IF EXISTS memory_embeddings")
        await db.execute(
            f"CREATE VIRTUAL TABLE memory_embeddings USING vec0("
            f"embedding float[{dimension}] distance_metric=cosine)"
        )

        cursor = await db.execute(
            "SELECT rowid, memory_text FROM conversation_memories"
        )
        memories = list(await cursor.fetchall())
        if memories:
            texts = [row[1] for row in memories]
            blobs = await embed_texts(texts)
            for (rowid, _), blob in zip(memories, blobs, strict=False):
                await db.execute(
                    "INSERT INTO memory_embeddings (rowid, embedding) VALUES (?, ?)",
                    (rowid, blob),
                )
            logger.info("Re-embedded {} memories", len(memories))
        counts["memories"] = len(memories)

        # --- artifact_embeddings ---
        await db.execute("DROP TABLE IF EXISTS artifact_embeddings")
        await db.execute(
            f"CREATE VIRTUAL TABLE artifact_embeddings USING vec0("
            f"embedding float[{dimension}] distance_metric=cosine)"
        )

        cursor = await db.execute(
            "SELECT id, chunk_text FROM artifact_chunks"
        )
        chunks = list(await cursor.fetchall())
        if chunks:
            texts = [row[1] for row in chunks]
            blobs = await embed_texts(texts)
            for (chunk_id, _), blob in zip(chunks, blobs, strict=False):
                await db.execute(
                    "INSERT INTO artifact_embeddings (rowid, embedding) VALUES (?, ?)",
                    (chunk_id, blob),
                )
            logger.info("Re-embedded {} artifact chunks", len(chunks))
        counts["artifact_chunks"] = len(chunks)

        # --- retraction_watch_embeddings ---
        await db.execute("DROP TABLE IF EXISTS retraction_watch_embeddings")
        await db.execute(
            f"CREATE VIRTUAL TABLE retraction_watch_embeddings USING vec0("
            f"embedding float[{dimension}] distance_metric=cosine)"
        )

        cursor = await db.execute(
            "SELECT rowid, title FROM retraction_watch WHERE title IS NOT NULL"
        )
        retractions = list(await cursor.fetchall())
        if retractions:
            texts = [row[1] for row in retractions]
            blobs = await embed_texts(texts)
            for (rowid, _), blob in zip(retractions, blobs, strict=False):
                await db.execute(
                    "INSERT INTO retraction_watch_embeddings "
                    "(rowid, embedding) VALUES (?, ?)",
                    (rowid, blob),
                )
            logger.info("Re-embedded {} retraction watch entries", len(retractions))
        counts["retraction_watch"] = len(retractions)

        # Record which model produced these embeddings + mark complete
        model_name = load_config().embeddings.model
        await db.execute(
            "INSERT OR REPLACE INTO app_meta (key, value) VALUES (?, ?)",
            ("embedding_model", model_name),
        )
        await db.execute(
            "INSERT OR REPLACE INTO app_meta (key, value) VALUES (?, ?)",
            ("embeddings_status", "complete"),
        )
        await db.commit()

    logger.info("Embedding tables rebuilt: {}", counts)
    return counts


def reset_model() -> None:
    """Reset the singleton model (for tests)."""
    global _model, _model_name
    _model = None
    _model_name = None
