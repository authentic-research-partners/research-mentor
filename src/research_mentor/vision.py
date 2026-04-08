"""Vision model for image artifact understanding.

Uses Qwen3-VL-2B-Instruct to generate text descriptions of images when
text extraction produces no text (pure images: photos, diagrams, charts, handwritten notes).

Always runs on CPU — must not compete with vLLM for GPU VRAM.
CPU inference is slow (~15-30s per image) but acceptable at upload time.

Same lazy-singleton pattern as embeddings.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Lazy singleton — loaded on first use
_model: Qwen3VLForConditionalGeneration | None = None
_processor: AutoProcessor | None = None


def _model_cache_dir() -> Path | None:
    """Return the HuggingFace cache directory for the vision model, or None."""
    try:
        from huggingface_hub import scan_cache_dir

        from research_mentor.config import load_config

        model_name = load_config().vision.model
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_name:
                return Path(repo.repo_path)
    except Exception:
        pass
    return None


_DOWNLOAD_COMPLETE_TAG = ".research_mentor_download_complete"


def is_local_model_cached() -> bool:
    """Check if the local vision model was fully downloaded.

    Uses a tag file written after successful model load — partial/interrupted
    downloads won't have the tag.
    """
    cache_dir = _model_cache_dir()
    if cache_dir is None:
        return False
    return (cache_dir / _DOWNLOAD_COMPLETE_TAG).exists()


def _mark_download_complete() -> None:
    """Write the tag file after successful model load."""
    cache_dir = _model_cache_dir()
    if cache_dir:
        (cache_dir / _DOWNLOAD_COMPLETE_TAG).touch()


def local_model_cache_path() -> str:
    """Return the path where the model is (or would be) stored."""
    cache_dir = _model_cache_dir()
    if cache_dir:
        return str(cache_dir)
    # Not downloaded yet — return where it would go
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        from research_mentor.config import load_config

        model_name = load_config().vision.model
        # HF stores as models--org--name
        folder = "models--" + model_name.replace("/", "--")
        return str(Path(HF_HUB_CACHE) / folder)
    except Exception:
        return "~/.cache/huggingface/hub"


def local_model_cache_size_bytes() -> int:
    """Return disk usage of the cached model in bytes, or 0 if not cached."""
    cache_dir = _model_cache_dir()
    if not cache_dir:
        return 0
    try:
        return sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
    except Exception:
        return 0


def remove_local_model_cache() -> bool:
    """Remove the cached local vision model. Returns True if removed."""
    global _model, _processor
    cache_dir = _model_cache_dir()
    if not cache_dir:
        return False
    import shutil

    shutil.rmtree(cache_dir, ignore_errors=True)
    _model = None
    _processor = None
    logger.info("Removed cached vision model: {}", cache_dir)
    return True


def _get_model() -> tuple[Qwen3VLForConditionalGeneration, Any]:
    """Get or create the singleton vision model and processor.

    Always loads on CPU with float32 to avoid competing with vLLM for GPU VRAM.
    ~4GB RAM at 2B params.
    """
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from research_mentor.config import load_config

    config = load_config()
    model_name = config.vision.model

    logger.info("Loading vision model: {} (device=cpu, dtype=float32)", model_name)
    _processor = AutoProcessor.from_pretrained(model_name)  # type: ignore[no-untyped-call]
    _model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    _mark_download_complete()
    logger.info("Vision model loaded: {}", model_name)
    return _model, _processor


def describe_image(
    file_path: str | Path,
    *,
    prompt: str | None = None,
) -> str | None:
    """Generate a text description of an image.

    Args:
        file_path: Path to the image file.
        prompt: Pre-built vision prompt (from build_vision_prompt). Falls back
            to a generic prompt if not provided.

    Returns:
        Text description of the image, or None if the file doesn't exist
        or description fails.
    """
    path = Path(file_path)
    if not path.is_file():
        logger.warning("describe_image: file not found: {}", path)
        return None

    from research_mentor.config import load_config

    config = load_config()

    model, processor = _get_model()

    vision_text = prompt or (
        "Examine this image carefully. "
        "First, transcribe any text you can read "
        "(printed, typed, or handwritten). "
        "Then describe any diagrams, charts, graphs, tables, "
        "or visual elements present in the image."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(path)},
                {"type": "text", "text": vision_text},
            ],
        },
    ]

    from PIL import Image

    image = Image.open(path).convert("RGB")

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_input],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cpu")

    import torch

    with torch.no_grad():
        output_ids = model.generate(  # type: ignore[misc]  # Qwen3VL vs base class signature
            **inputs, max_new_tokens=config.vision.max_new_tokens,
        )

    # Decode only the generated tokens (skip the input)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    description: str = processor.decode(generated_ids, skip_special_tokens=True).strip()

    if not description:
        return None

    logger.info("Generated image description ({} chars) for {}", len(description), path.name)
    return description


def reset_model() -> None:
    """Reset the singleton model (for tests)."""
    global _model, _processor
    _model = None
    _processor = None
