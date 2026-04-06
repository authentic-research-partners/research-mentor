"""Retraction Watch data loader and semantic search.

Downloads the daily-updated CSV from Crossref's Retraction Watch GitLab repo,
parses it into the ``retraction_watch`` SQLite table, and computes embeddings
for semantic search via sqlite-vec.

The CSV covers ALL update types: retractions, expressions of concern, and
corrections with human-curated reasons (fabrication, plagiarism, error, etc.).
Shared data source for both the Questioned and Retractions workshops.

Incremental refresh: on each refresh the CSV is diffed against the DB using
a content hash per row. Only new/changed rows are enriched with abstracts
from OpenAlex and re-embedded.  Deleted rows are removed.

Auto-refresh: data is downloaded on first use if the table is empty, and
re-downloaded if older than 7 days. No API key required.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import time as _time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from loguru import logger

from research_mentor.db import crud
from research_mentor.tools.status import ToolStatus, make_error_result, record_tool_usage

# Max CSV download size: 200 MB.  The Retraction Watch CSV is typically ~50 MB;
# anything larger suggests a corrupt download or unexpected server response.
_MAX_CSV_BYTES = 200 * 1024 * 1024

# GitLab raw URL for the Retraction Watch CSV (updated daily by Crossref)
_CSV_URL = (
    "https://gitlab.com/crossref/retraction-watch-data/-/raw/main/retraction_watch.csv"
)

# How often to auto-refresh (days)
_REFRESH_INTERVAL_DAYS = 7

# Maximum rows to import (safety limit — full CSV is ~40K rows)
_MAX_IMPORT_ROWS = 100_000

_TOOL_NAME = "retraction_watch"




# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


# Fields used for content hashing (fixed order).
_HASH_FIELDS = (
    "original_doi", "retraction_doi", "title", "authors", "journal",
    "publisher", "country", "subject", "retraction_date",
    "retraction_nature", "reason", "article_type", "paywalled", "urls",
)

# Maximum DOIs per OpenAlex batch request
_OPENALEX_BATCH_SIZE = 50


def _compute_match_key(row: dict[str, Any]) -> str:
    """Compute a stable identity key for a Retraction Watch row.

    Uses the original DOI when available, falls back to the retraction DOI,
    and finally to a hash of title + journal + retraction date.
    Prefixed to avoid collisions between key types.
    """
    if row.get("original_doi"):
        return f"doi:{row['original_doi'].lower().strip()}"
    if row.get("retraction_doi"):
        return f"rdoi:{row['retraction_doi'].lower().strip()}"
    parts = "|".join(
        (row.get(f) or "") for f in ("title", "journal", "retraction_date")
    )
    return f"hash:{hashlib.sha256(parts.encode()).hexdigest()}"


def _compute_content_hash(row: dict[str, Any]) -> str:
    """SHA-256 hex digest of all CSV-sourced fields (fixed order).

    Used to detect whether a row's content changed across CSV refreshes.
    """
    parts = "|".join((row.get(f) or "") for f in _HASH_FIELDS)
    return hashlib.sha256(parts.encode()).hexdigest()


def _normalize_nature(raw: str) -> str:
    """Normalize the RetractionNature field to a clean category."""
    lower = raw.strip().lower()
    if "retraction" in lower and "expression" not in lower:
        return "retraction"
    if "expression" in lower or "concern" in lower:
        return "expression of concern"
    if "correction" in lower or "erratum" in lower:
        return "correction"
    return lower


def _parse_csv(text: str) -> list[dict[str, Any]]:
    """Parse Retraction Watch CSV into row dicts.

    The CSV has headers like: Title, Subject, Institution, Journal, Publisher,
    Country, Author, URLS, ArticleType, RetractionDate, RetractionDOI,
    OriginalPaperDOI, RetractionNature, Reason, Paywalled.
    """
    now = datetime.now(UTC).isoformat()
    reader = csv.DictReader(io.StringIO(text))

    rows: list[dict[str, Any]] = []
    for i, record in enumerate(reader):
        if i >= _MAX_IMPORT_ROWS:
            break

        title = (record.get("Title") or "").strip()
        if not title:
            continue

        nature_raw = (record.get("RetractionNature") or "").strip()
        if not nature_raw:
            continue

        row = {
            "original_doi": (record.get("OriginalPaperDOI") or "").strip() or None,
            "retraction_doi": (record.get("RetractionDOI") or "").strip() or None,
            "title": title,
            "authors": (record.get("Author") or "").strip() or None,
            "journal": (record.get("Journal") or "").strip() or None,
            "publisher": (record.get("Publisher") or "").strip() or None,
            "country": (record.get("Country") or "").strip() or None,
            "subject": (record.get("Subject") or "").strip() or None,
            "retraction_date": (record.get("RetractionDate") or "").strip() or None,
            "retraction_nature": _normalize_nature(nature_raw),
            "reason": (record.get("Reason") or "").strip() or None,
            "article_type": (record.get("ArticleType") or "").strip() or None,
            "paywalled": (record.get("Paywalled") or "").strip() or None,
            "urls": (record.get("URLS") or "").strip() or None,
            "updated_at": now,
            "abstract": None,  # populated later by OpenAlex enrichment
        }
        row["match_key"] = _compute_match_key(row)
        row["content_hash"] = _compute_content_hash(row)
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Download & refresh
# ---------------------------------------------------------------------------


async def _download_csv() -> str:
    """Download the Retraction Watch CSV from GitLab with size cap.

    Uses streaming with a byte-count guard.  The Content-Length pre-check
    is skipped when Content-Encoding is set (e.g. gzip) because the header
    reports the compressed wire size, not the decompressed size.
    """
    logger.info("Downloading Retraction Watch CSV from GitLab...")
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("GET", _CSV_URL, follow_redirects=True) as resp:
            resp.raise_for_status()
            if not resp.headers.get("content-encoding"):
                cl = resp.headers.get("content-length")
                if cl and int(cl) > _MAX_CSV_BYTES:
                    raise RuntimeError(
                        f"Retraction Watch CSV too large (Content-Length {cl} bytes, "
                        f"max {_MAX_CSV_BYTES}). Possible corrupted download."
                    )
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > _MAX_CSV_BYTES:
                    raise RuntimeError(
                        f"Retraction Watch CSV exceeded {_MAX_CSV_BYTES} bytes "
                        f"during download. Possible corrupted download."
                    )
                chunks.append(chunk)
    content = b"".join(chunks)
    text = content.decode("utf-8")
    logger.info(
        "Downloaded Retraction Watch CSV: {:.1f} MB",
        len(content) / 1_048_576,
    )
    return text


async def _compute_embeddings(rows: list[dict[str, Any]]) -> list[bytes]:
    """Compute embeddings for rows (title + abstract + subject + reason).

    Uses the embeddings.py infrastructure (configurable Snowflake Arctic Embed,
    CPU, async via thread pool). Batched for efficiency.
    """
    from research_mentor.embeddings import embed_texts

    texts = []
    for row in rows:
        parts = [row["title"]]
        if row.get("abstract"):
            parts.append(row["abstract"])
        if row.get("subject"):
            parts.append(row["subject"])
        if row.get("reason"):
            parts.append(row["reason"])
        texts.append(" ".join(parts))

    logger.info("Computing embeddings for {} papers...", len(texts))
    embeddings = await embed_texts(texts)
    logger.info("Embeddings computed for {} papers", len(embeddings))
    return embeddings


async def _fetch_abstracts_from_openalex(
    dois: list[str],
) -> dict[str, str]:
    """Batch-fetch abstracts from OpenAlex by DOI.

    Returns ``{doi_lower: abstract_text}`` for papers where an abstract was
    found.  Best-effort: network/API errors are logged and result in an empty
    dict — abstracts are enrichment data from a third-party API and must not
    block the primary CSV refresh pipeline.
    """
    from research_mentor.tools.academic_search import (
        decode_inverted_abstract,
        get_openalex_limiter,
    )

    cfg_email: str | None = None
    try:
        from research_mentor.config import load_config
        cfg_email = load_config().services.polite_email or None
    except Exception:
        logger.debug("Failed to load polite_email from config")

    result: dict[str, str] = {}

    for batch_start in range(0, len(dois), _OPENALEX_BATCH_SIZE):
        batch = dois[batch_start:batch_start + _OPENALEX_BATCH_SIZE]
        doi_filter = "|".join(batch)

        try:
            params: dict[str, str | int] = {
                "filter": f"doi:{doi_filter}",
                "per_page": _OPENALEX_BATCH_SIZE,
                "select": "doi,abstract_inverted_index",
            }
            if cfg_email:
                params["mailto"] = cfg_email

            async with get_openalex_limiter():
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(
                        "https://api.openalex.org/works", params=params,
                    )
                    resp.raise_for_status()
                    data = resp.json()

            for work in data.get("results", []):
                doi_raw = work.get("doi") or ""
                inverted = work.get("abstract_inverted_index")
                if doi_raw and inverted:
                    doi_clean = doi_raw.replace("https://doi.org/", "").lower()
                    result[doi_clean] = decode_inverted_abstract(inverted)

        except Exception as exc:
            logger.warning(
                "OpenAlex abstract enrichment failed for batch at offset {}: {}",
                batch_start, exc,
            )

    return result


async def _enrich_rows_with_abstracts(rows: list[dict[str, Any]]) -> None:
    """Fetch abstracts from OpenAlex and set ``row["abstract"]`` in place.

    Only rows with a DOI are enriched.  Rows without a DOI or where OpenAlex
    returned no abstract keep ``abstract=None``.
    """
    dois: list[str] = []
    for row in rows:
        doi = row.get("original_doi") or row.get("retraction_doi")
        if doi:
            dois.append(doi.lower().strip())

    if not dois:
        return

    logger.info("Fetching abstracts from OpenAlex for {} DOIs...", len(dois))
    doi_to_abstract = await _fetch_abstracts_from_openalex(dois)

    enriched = 0
    for row in rows:
        doi = row.get("original_doi") or row.get("retraction_doi")
        if doi:
            abstract = doi_to_abstract.get(doi.lower().strip())
            if abstract:
                row["abstract"] = abstract
                enriched += 1

    logger.info(
        "Enriched {}/{} rows with abstracts from OpenAlex",
        enriched, len(rows),
    )


async def refresh_retraction_watch_data(*, force: bool = False) -> int:
    """Download Retraction Watch CSV, diff, enrich, embed, and sync to DB.

    Incremental: only new/changed rows are enriched and re-embedded.
    Deleted rows (present in DB but absent from CSV) are removed.

    Auto-skips if last refresh was < 7 days ago (unless ``force=True``).
    Returns number of rows inserted + updated.
    """
    if not force:
        last_refresh = await crud.get_retraction_watch_last_refresh()
        if last_refresh:
            last_dt = datetime.fromisoformat(last_refresh)
            if datetime.now(UTC) - last_dt < timedelta(days=_REFRESH_INTERVAL_DAYS):
                logger.debug(
                    "Retraction Watch data is fresh (last refresh: {}), skipping",
                    last_refresh,
                )
                return 0

    csv_text = await _download_csv()
    rows = _parse_csv(csv_text)
    if not rows:
        logger.warning("Retraction Watch CSV parsed to 0 rows — aborting")
        return 0

    # ---- Diff against existing data ----
    existing = await crud.get_retraction_watch_content_hashes()
    incoming_keys = {r["match_key"] for r in rows}

    new_rows: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    changed_row_ids: list[int] = []

    for row in rows:
        mk = row["match_key"]
        if mk not in existing:
            new_rows.append(row)
        else:
            old_id, old_hash = existing[mk]
            if old_hash != row["content_hash"]:
                changed_rows.append(row)
                changed_row_ids.append(old_id)

    deleted_row_ids = [
        row_id for mk, (row_id, _) in existing.items() if mk not in incoming_keys
    ]

    if not new_rows and not changed_rows and not deleted_row_ids:
        logger.info("Retraction Watch: no changes detected")
        await crud.set_retraction_watch_last_refresh(datetime.now(UTC).isoformat())
        return 0

    logger.info(
        "Retraction Watch diff: {} new, {} changed, {} deleted, {} unchanged",
        len(new_rows), len(changed_rows), len(deleted_row_ids),
        len(rows) - len(new_rows) - len(changed_rows),
    )

    # ---- Enrich new/changed with abstracts ----
    rows_to_process = new_rows + changed_rows
    await _enrich_rows_with_abstracts(rows_to_process)

    # ---- Compute embeddings for new/changed only ----
    new_embeddings = await _compute_embeddings(new_rows) if new_rows else []
    changed_embeddings = await _compute_embeddings(changed_rows) if changed_rows else []

    # ---- Sync to DB ----
    inserted, updated, deleted = await crud.sync_retraction_watch_incremental(
        new_rows, new_embeddings,
        changed_rows, changed_embeddings, changed_row_ids,
        deleted_row_ids,
    )
    await crud.set_retraction_watch_last_refresh(datetime.now(UTC).isoformat())

    logger.info(
        "Retraction Watch refresh complete: {} inserted, {} updated, {} deleted",
        inserted, updated, deleted,
    )
    return inserted + updated


async def ensure_retraction_watch_data() -> None:
    """Ensure Retraction Watch data is available and fresh.

    Called before any workshop query.  Downloads if table is empty,
    triggers a full re-sync if existing data lacks match_keys (V23
    migration), or refreshes if data is stale (> 7 days).
    """
    count = await crud.get_retraction_watch_count()
    if count == 0:
        logger.info("Retraction Watch table is empty — downloading data")
        await refresh_retraction_watch_data(force=True)
        return

    # First-run migration: V23 added match_key but existing rows lack it.
    # Force a full refresh so every row gets a match_key, content_hash, and
    # abstract from OpenAlex.
    match_key_count = await crud.get_retraction_watch_match_key_count()
    if match_key_count == 0:
        logger.info(
            "Retraction Watch: migrating to incremental refresh "
            "(populating match_key + abstracts)",
        )
        await refresh_retraction_watch_data(force=True)
        return

    last_refresh = await crud.get_retraction_watch_last_refresh()
    if last_refresh:
        last_dt = datetime.fromisoformat(last_refresh)
        if datetime.now(UTC) - last_dt >= timedelta(days=_REFRESH_INTERVAL_DAYS):
            logger.info("Retraction Watch data is stale — refreshing")
            await refresh_retraction_watch_data(force=True)


# ---------------------------------------------------------------------------
# Search functions (with ToolStatus + usage tracking)
# ---------------------------------------------------------------------------


async def search_scrutinized_papers(
    query: str,
    *,
    max_results: int = 10,
    threshold: float = 0.35,
) -> list[dict[str, Any]]:
    """Semantic search for papers under scrutiny (expressions of concern + corrections).

    Used by the Questioned workshop. Returns papers with ToolStatus.
    """
    t0 = _time.monotonic()
    await ensure_retraction_watch_data()

    try:
        from research_mentor.embeddings import embed_text

        query_embedding = await embed_text(query)
        results = await crud.search_retraction_watch_semantic(
            query_embedding,
            nature=["expression of concern", "correction"],
            limit=max_results,
            threshold=threshold,
        )

        elapsed = _time.monotonic() - t0
        for r in results:
            r["source"] = "Retraction Watch"
            r["tool_status"] = ToolStatus.ok("Retraction Watch")

        asyncio.create_task(
            record_tool_usage(
                _TOOL_NAME, query=query, results=results, elapsed=elapsed,
            ),
        )
        return results

    except Exception as exc:
        elapsed = _time.monotonic() - t0
        msg = f"Retraction Watch search failed: {exc}"
        logger.error(msg)
        asyncio.create_task(
            record_tool_usage(
                _TOOL_NAME, query=query, success=False,
                error_category="unavailable", elapsed=elapsed,
            ),
        )
        return make_error_result("Retraction Watch", "unavailable", msg)


async def search_retracted_papers(
    query: str,
    *,
    max_results: int = 10,
    threshold: float = 0.35,
) -> list[dict[str, Any]]:
    """Semantic search for retracted papers.

    Used by the Retractions workshop. Returns papers with ToolStatus.
    """
    t0 = _time.monotonic()
    await ensure_retraction_watch_data()

    try:
        from research_mentor.embeddings import embed_text

        query_embedding = await embed_text(query)
        results = await crud.search_retraction_watch_semantic(
            query_embedding,
            nature="retraction",
            limit=max_results,
            threshold=threshold,
        )

        elapsed = _time.monotonic() - t0
        for r in results:
            r["source"] = "Retraction Watch"
            r["tool_status"] = ToolStatus.ok("Retraction Watch")

        asyncio.create_task(
            record_tool_usage(
                _TOOL_NAME, query=query, results=results, elapsed=elapsed,
            ),
        )
        return results

    except Exception as exc:
        elapsed = _time.monotonic() - t0
        msg = f"Retraction Watch search failed: {exc}"
        logger.error(msg)
        asyncio.create_task(
            record_tool_usage(
                _TOOL_NAME, query=query, success=False,
                error_category="unavailable", elapsed=elapsed,
            ),
        )
        return make_error_result("Retraction Watch", "unavailable", msg)


async def get_retraction_by_doi(doi: str) -> dict[str, Any] | None:
    """Look up a specific retraction by DOI (original or retraction notice)."""
    t0 = _time.monotonic()
    await ensure_retraction_watch_data()

    result = await crud.get_retraction_watch_by_doi(doi)
    elapsed = _time.monotonic() - t0

    asyncio.create_task(
        record_tool_usage(
            _TOOL_NAME, query=doi, success=result is not None, elapsed=elapsed,
        ),
    )
    return result
