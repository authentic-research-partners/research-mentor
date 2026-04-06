"""CORE API client — abstract enrichment via DOI lookup.

CORE (core.ac.uk) aggregates 300M+ metadata records and 40M+ full-text papers
from 12,000+ institutional repositories worldwide. Used for enriching papers
that lack abstracts after discovery (OpenAlex/Semantic Scholar).

API docs: https://api.core.ac.uk/docs/v3
Rate limits: 100 credits/day unauthenticated, 1000/day with free API key.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from research_mentor.config import load_config
from research_mentor.tools.rate_limiter import (
    MonotonicRateLimiter,
    RateLimitedError,
    parse_retry_after,
    retry_on_429,
)
from research_mentor.tools.status import record_tool_usage

_CORE_BASE = "https://api.core.ac.uk/v3"

# ---------------------------------------------------------------------------
# Rate limiter (lazy init)
# ---------------------------------------------------------------------------

_core_limiter: MonotonicRateLimiter | None = None


def _ensure_limiter() -> MonotonicRateLimiter:
    global _core_limiter
    if _core_limiter is None:
        cfg = load_config()
        rl = cfg.rate_limits.core_api
        _core_limiter = MonotonicRateLimiter(rl[0], rl[1])
    return _core_limiter


# ---------------------------------------------------------------------------
# API key (optional)
# ---------------------------------------------------------------------------


def _read_api_key() -> str | None:
    """Read CORE API key from file. Returns None if not configured."""
    cfg = load_config()
    key_path = Path(cfg.services.core_api.api_key_file).expanduser()
    if not key_path.exists():
        return None
    text = key_path.read_text().strip()
    return text if text else None


# ---------------------------------------------------------------------------
# DOI lookup
# ---------------------------------------------------------------------------


async def search_core_by_doi(doi: str) -> dict[str, Any] | None:
    """Look up a paper by DOI via CORE API. Returns enrichment data or None.

    Args:
        doi: The paper DOI (e.g., "10.1038/nature12373").

    Returns:
        Dict with abstract, download_url, authors, year, core_id — or None
        if the paper isn't in CORE or the request fails.
    """
    limiter = _ensure_limiter()
    api_key = _read_api_key()

    # Strip doi.org prefix if present
    clean_doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    import time as _time

    t0 = _time.monotonic()
    result: dict[str, Any] | None = None
    success = False
    try:
        @retry_on_429()
        async def _request() -> httpx.Response:
            async with limiter:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(
                        f"{_CORE_BASE}/search/works/",
                        params={"q": f'doi:"{clean_doi}"', "limit": 1},
                        headers=headers,
                    )
                if resp.status_code == 429:
                    raise RateLimitedError(
                        "CORE API", f"DOI: {clean_doi}",
                        retry_after=parse_retry_after(resp.headers.get("Retry-After")),
                    )
                return resp

        try:
            resp = await _request()
        except RateLimitedError:
            return None

        if resp.status_code in (401, 403):
            logger.debug("CORE API auth error ({}) for DOI: {}", resp.status_code, clean_doi)
            return None

        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            success = True  # API worked, just no results
            return None

        # Pick the result with the richest metadata (has abstract)
        work = results[0]
        for candidate in results:
            if candidate.get("abstract"):
                work = candidate
                break

        abstract = work.get("abstract")
        if not abstract:
            success = True
            return None

        authors = [
            a.get("name", "") for a in (work.get("authors") or []) if a.get("name")
        ]

        success = True
        result = {
            "abstract": abstract,
            "download_url": work.get("downloadUrl"),
            "authors": authors,
            "year": work.get("yearPublished"),
            "core_id": work.get("id"),
            "source": "CORE",
        }
        return result

    except TimeoutError:
        logger.debug("CORE API timeout for DOI: {}", clean_doi)
        return None
    except Exception as e:
        logger.debug("CORE API error for DOI {}: {}", clean_doi, e)
        return None
    finally:
        asyncio.create_task(record_tool_usage(
            "core_api",
            query=clean_doi,
            success=success,
            error_category=None if success else "network",
            elapsed=_time.monotonic() - t0,
        ))
