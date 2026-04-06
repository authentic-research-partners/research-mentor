"""Researcher search tool — find active researchers by topic via OpenAlex.

Uses the OpenAlex ``/authors`` endpoint to discover researchers working on
specific topics.  Shares the OpenAlex rate limiter with ``academic_search``.
Results are enriched with ORCID profile data (biography, employment,
education) when available.
"""

from __future__ import annotations

import asyncio
import time as _time
from typing import Any

import httpx
from loguru import logger

from research_mentor.config import load_config
from research_mentor.tools.academic_search import _ensure_limiters
from research_mentor.tools.status import make_error_result, record_tool_usage


async def search_researchers(
    topic: str,
    *,
    institution_ror: str | None = None,
    country: str | None = None,
    max_results: int = 10,
    mailto: str | None = None,
) -> list[dict[str, Any]]:
    """Search OpenAlex for researchers working on a topic.

    Args:
        topic: Research topic to search (e.g. "mass spectrometry proteomics").
        institution_ror: Optional ROR ID to filter by institution.
        country: Optional two-letter country code (e.g. "US", "DE").
        max_results: Maximum results to return (default 10).
        mailto: Email for polite pool.  Uses config value if *None*.

    Returns:
        List of researcher dicts with name, affiliation, works_count,
        cited_by_count, h_index, topics, orcid, openalex_id.
        On error, returns a single-element list with ``error`` and
        ``tool_status`` keys.
    """
    _ensure_limiters()
    if mailto is None:
        mailto = load_config().services.polite_email or None

    # --- build filters ---
    filters: list[str] = [f"topics.display_name.search:{topic}"]
    if institution_ror:
        # Accept both full URL and bare ID
        ror_id = institution_ror
        if not ror_id.startswith("https://"):
            ror_id = f"https://ror.org/{ror_id}"
        filters.append(f"last_known_institutions.ror:{ror_id}")
    if country:
        filters.append(
            f"last_known_institutions.country_code:{country.upper()}"
        )

    async def _with_rate_limit() -> list[dict[str, Any]]:
        from research_mentor.tools.academic_search import get_openalex_limiter

        params: dict[str, str | int] = {
            "filter": ",".join(filters),
            "per_page": max_results,
            "select": (
                "id,display_name,last_known_institutions,"
                "works_count,cited_by_count,topics,ids,summary_stats"
            ),
            "sort": "works_count:desc",
        }
        if mailto:
            params["mailto"] = mailto

        async with get_openalex_limiter():
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    "https://api.openalex.org/authors", params=params,
                )
                resp.raise_for_status()
                data = resp.json()

        results: list[dict[str, Any]] = []
        for author in data.get("results", []):
            # Primary institution
            institutions = author.get("last_known_institutions") or []
            affiliation = institutions[0].get("display_name", "") if institutions else ""
            inst_country = institutions[0].get("country_code", "") if institutions else ""

            # Top topics (up to 5)
            raw_topics = author.get("topics") or []
            topic_names = [t.get("display_name", "") for t in raw_topics[:5]]

            # h-index from summary_stats
            summary = author.get("summary_stats") or {}
            h_index = summary.get("h_index", 0)

            # ORCID
            ids = author.get("ids") or {}
            orcid = ids.get("orcid", "")

            results.append({
                "name": author.get("display_name", "Unknown"),
                "affiliation": affiliation,
                "institution_country": inst_country,
                "works_count": author.get("works_count", 0),
                "cited_by_count": author.get("cited_by_count", 0),
                "h_index": h_index,
                "topics": topic_names,
                "orcid": orcid,
                "openalex_id": author.get("id", ""),
                "source": "OpenAlex",
            })
        return results

    t0 = _time.monotonic()
    try:
        results = await asyncio.wait_for(_with_rate_limit(), timeout=15.0)
    except TimeoutError:
        logger.error("Researcher search timeout after 15s for topic: {}", topic)
        results = make_error_result(
            "OpenAlex", "timeout", "Researcher search timed out after 15 seconds.",
        )
    except Exception as e:
        logger.error("Researcher search failed: {}", e)
        results = make_error_result("OpenAlex", "network", f"Researcher search failed: {e}")
    asyncio.create_task(
        record_tool_usage(
            "researcher_search", query=topic, results=results, elapsed=_time.monotonic() - t0,
        ),
    )

    # Enrich with ORCID profiles (non-fatal — individual failures are logged)
    if results and "error" not in results[0]:
        try:
            from research_mentor.tools.orcid import enrich_researchers_with_orcid

            results = await enrich_researchers_with_orcid(results)
        except Exception:
            logger.debug("ORCID enrichment failed for researcher search")

    return results
