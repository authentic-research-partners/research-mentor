"""Institution search tool — find institutions and resolve ROR IDs.

Uses the OpenAlex ``/institutions`` endpoint and the ROR API to discover
universities, labs, and facilities.  Shares the OpenAlex rate limiter with
``academic_search``.
"""

from __future__ import annotations

import asyncio
import time as _time
from typing import Any

import httpx
from loguru import logger

from research_mentor.config import load_config
from research_mentor.tools.academic_search import _ensure_limiters
from research_mentor.tools.rate_limiter import MonotonicRateLimiter
from research_mentor.tools.status import make_error_result, record_tool_usage

# ---------------------------------------------------------------------------
# ROR rate limiter (separate from OpenAlex)
# ---------------------------------------------------------------------------

_ror_limiter_initialized = False
_ror_limiter: MonotonicRateLimiter


def _ensure_ror_limiter() -> None:
    """Initialize the ROR API rate limiter from config (once)."""
    global _ror_limiter_initialized, _ror_limiter

    if _ror_limiter_initialized:
        return

    cfg = load_config()
    rl = cfg.rate_limits
    _ror_limiter = MonotonicRateLimiter(rl.ror_api[0], rl.ror_api[1])
    _ror_limiter_initialized = True


# ---------------------------------------------------------------------------
# ROR resolution
# ---------------------------------------------------------------------------


async def resolve_institution_ror(name: str) -> str | None:
    """Resolve an institution name to a ROR ID via the ROR API.

    Args:
        name: Institution name (e.g. "University of Michigan").

    Returns:
        ROR URL (e.g. ``"https://ror.org/00jmfr291"``) or *None* if not found.
    """
    _ensure_ror_limiter()

    try:
        async with _ror_limiter:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.ror.org/v2/organizations",
                    params={"query": name},
                )
                resp.raise_for_status()
                data = resp.json()

        items = data.get("items", [])
        if not items:
            return None

        # Return the top match
        ror_id: str | None = items[0].get("id")
        return ror_id
    except Exception as e:
        logger.debug("ROR resolution failed for '{}': {}", name, e)
        return None


# ---------------------------------------------------------------------------
# Institution search (OpenAlex)
# ---------------------------------------------------------------------------


async def search_institutions(
    query: str,
    *,
    type_filter: str | None = None,
    country: str | None = None,
    max_results: int = 10,
    mailto: str | None = None,
) -> list[dict[str, Any]]:
    """Search OpenAlex for institutions, labs, or facilities.

    Args:
        query: Search query (e.g. "mass spectrometry core facility").
        type_filter: Optional institution type — "education", "facility",
            "healthcare", "company", "archive", "nonprofit", "government",
            "other".
        country: Optional two-letter country code (e.g. "US").
        max_results: Maximum results to return (default 10).
        mailto: Email for polite pool.  Uses config value if *None*.

    Returns:
        List of institution dicts with name, type, country, ror, homepage_url,
        works_count, cited_by_count, openalex_id.
        On error, returns a single-element list with ``error`` and
        ``tool_status`` keys.
    """
    _ensure_limiters()
    if mailto is None:
        mailto = load_config().services.polite_email or None

    # --- build filters ---
    filters: list[str] = [f"display_name.search:{query}"]
    if type_filter:
        filters.append(f"type:{type_filter}")
    if country:
        filters.append(f"country_code:{country.upper()}")

    async def _with_rate_limit() -> list[dict[str, Any]]:
        from research_mentor.tools.academic_search import get_openalex_limiter

        params: dict[str, str | int] = {
            "filter": ",".join(filters),
            "per_page": max_results,
            "select": (
                "id,display_name,ror,type,country_code,"
                "homepage_url,works_count,cited_by_count"
            ),
            "sort": "works_count:desc",
        }
        if mailto:
            params["mailto"] = mailto

        async with get_openalex_limiter():
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    "https://api.openalex.org/institutions", params=params,
                )
                resp.raise_for_status()
                data = resp.json()

        results: list[dict[str, Any]] = []
        for inst in data.get("results", []):
            results.append({
                "name": inst.get("display_name", "Unknown"),
                "type": inst.get("type", ""),
                "country": inst.get("country_code", ""),
                "ror": inst.get("ror", ""),
                "homepage_url": inst.get("homepage_url", ""),
                "works_count": inst.get("works_count", 0),
                "cited_by_count": inst.get("cited_by_count", 0),
                "openalex_id": inst.get("id", ""),
                "source": "OpenAlex",
            })
        return results

    t0 = _time.monotonic()
    try:
        results = await asyncio.wait_for(_with_rate_limit(), timeout=15.0)
    except TimeoutError:
        logger.error("Institution search timeout after 15s for query: {}", query)
        results = make_error_result(
            "OpenAlex", "timeout", "Institution search timed out after 15 seconds.",
        )
    except Exception as e:
        logger.error("Institution search failed: {}", e)
        results = make_error_result("OpenAlex", "network", f"Institution search failed: {e}")
    asyncio.create_task(
        record_tool_usage(
            "institution_search", query=query, results=results, elapsed=_time.monotonic() - t0,
        ),
    )
    return results
