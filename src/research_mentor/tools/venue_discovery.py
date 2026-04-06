"""Venue discovery tools — OpenAlex /sources endpoint.

Provides journal/conference discovery via OpenAlex. Used by the Sharing
& Publication module for venue matching. Curated venue matching lives
in ``sharing.utils.venue_matching`` (not here — this is the API client).
"""

from __future__ import annotations

import asyncio
import time as _time
from typing import Any

import httpx
from loguru import logger

from research_mentor.config import load_config

# Reuse the rate limiter from academic_search (same OpenAlex endpoint family).
from research_mentor.tools.academic_search import get_openalex_limiter
from research_mentor.tools.status import make_error_result, record_tool_usage


async def search_openalex_sources(
    query: str,
    *,
    source_type: str | None = None,
    is_oa: bool | None = None,
    max_results: int = 20,
    mailto: str | None = None,
) -> list[dict[str, Any]]:
    """Search OpenAlex for journals, conferences, and other publication venues.

    Args:
        query: Search query (topic or venue name).
        source_type: Filter by type: 'journal', 'conference', 'repository',
            'ebook platform', or None for all.
        is_oa: Filter for open-access venues only (True), non-OA (False),
            or all (None).
        max_results: Maximum results (default 20).
        mailto: Email for polite pool. If None, uses config value.

    Returns:
        List of venue dicts with display_name, type, is_oa, apc_usd,
        h_index, cited_by_count, subjects, homepage_url, issn, openalex_id.
    """
    limiter = get_openalex_limiter()
    if mailto is None:
        mailto = load_config().services.polite_email or None

    async def _with_rate_limit() -> list[dict[str, Any]]:
        # Build filter string
        filters: list[str] = []
        if query:
            filters.append(f"display_name.search:{query}")
        if source_type:
            filters.append(f"type:{source_type}")
        if is_oa is True:
            filters.append("is_oa:true")
        elif is_oa is False:
            filters.append("is_oa:false")

        params: dict[str, str | int] = {
            "per_page": max_results,
            "select": (
                "id,display_name,type,issn,is_oa,apc_usd,"
                "summary_stats,homepage_url,host_organization_name,"
                "topics,works_count,cited_by_count"
            ),
        }
        if filters:
            params["filter"] = ",".join(filters)
        if mailto:
            params["mailto"] = mailto

        async with limiter:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    "https://api.openalex.org/sources", params=params,
                )
                resp.raise_for_status()
                data = resp.json()

        results: list[dict[str, Any]] = []
        for source in data.get("results", []):
            # Extract summary stats
            stats = source.get("summary_stats", {})
            h_index = stats.get("h_index", 0)
            mean_citedness = stats.get("2yr_mean_citedness", 0.0)

            # Extract top topics/subjects
            topics = source.get("topics", []) or []
            subject_names = [
                t.get("display_name", "")
                for t in topics[:5]
                if t.get("display_name")
            ]

            results.append({
                "display_name": source.get("display_name", "Unknown"),
                "type": source.get("type"),
                "is_oa": source.get("is_oa", False),
                "apc_usd": source.get("apc_usd"),
                "h_index": h_index,
                "mean_citedness_2yr": mean_citedness,
                "cited_by_count": source.get("cited_by_count", 0),
                "works_count": source.get("works_count", 0),
                "subjects": subject_names,
                "homepage_url": source.get("homepage_url"),
                "issn": source.get("issn"),
                "host_organization": source.get("host_organization_name"),
                "openalex_id": source.get("id"),
                "source": "OpenAlex",
            })
        return results

    t0 = _time.monotonic()
    try:
        results = await asyncio.wait_for(_with_rate_limit(), timeout=15.0)
    except TimeoutError:
        logger.error("OpenAlex sources search timeout after 15s for query: {}", query)
        results = make_error_result(
            "OpenAlex", "timeout", "OpenAlex sources search timed out after 15 seconds.",
        )
    except Exception as e:
        logger.error("OpenAlex sources search failed: {}", e)
        results = make_error_result("OpenAlex", "network", f"OpenAlex sources search failed: {e}")

    asyncio.create_task(
        record_tool_usage(
            "openalex_sources", query=query, results=results, elapsed=_time.monotonic() - t0,
        ),
    )

    return results
