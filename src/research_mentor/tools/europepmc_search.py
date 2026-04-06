"""Europe PMC full-text search — biomedical + life science papers.

Searches across 40M+ abstracts and 9M full-text articles. Complements
OpenAlex and Semantic Scholar with full-text search (not just titles/abstracts),
strong European institutional repository coverage, and preprint indexing.

Free, no API key required. Rate limit shared with paper_fetcher's Europe PMC
PDF downloader (separate limiter instance, same config value).
"""

from __future__ import annotations

import asyncio
import time as _time
from typing import Any

import httpx
from loguru import logger

from research_mentor.config import load_config
from research_mentor.tools.rate_limiter import MonotonicRateLimiter
from research_mentor.tools.status import make_error_result, record_tool_usage

_EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# ---------------------------------------------------------------------------
# Rate limiter (lazy init)
# ---------------------------------------------------------------------------

_limiter: MonotonicRateLimiter | None = None


def _ensure_limiter() -> MonotonicRateLimiter:
    global _limiter
    if _limiter is None:
        cfg = load_config()
        rl = cfg.rate_limits.europepmc
        _limiter = MonotonicRateLimiter(rl[0], rl[1])
    return _limiter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def search_europepmc(
    query: str,
    max_results: int = 10,
    year_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Search Europe PMC for biomedical and life science papers.

    Searches full-text when available (9M+ OA articles), otherwise metadata.

    Args:
        query: Search query (e.g., "CRISPR gene therapy delivery").
        max_results: Maximum results to return (default 10).
        year_filter: Year range as "YYYY-YYYY" (e.g., "2020-2025").
            Appended as ``FIRST_PDATE:[YYYY TO YYYY]`` to the query.

    Returns:
        List of paper dicts with title, authors, abstract, url, doi,
        pdf_url, year, citation_count, source, pubmed_id, pmcid.
    """

    async def _with_rate_limit() -> list[dict[str, Any]]:
        # Build query with optional year filter
        search_query = query
        if year_filter:
            parts = year_filter.split("-")
            if len(parts) == 2:
                search_query = f"{query} FIRST_PDATE:[{parts[0]} TO {parts[1]}]"

        params: dict[str, str | int] = {
            "query": search_query,
            "format": "json",
            "resultType": "core",
            "pageSize": max_results,
            "sort": "CITED desc",
        }

        async with _ensure_limiter():
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(_EUROPEPMC_SEARCH, params=params)
                resp.raise_for_status()
                data = resp.json()

        result_list = data.get("resultList", {}).get("result", [])
        results: list[dict[str, Any]] = []

        for item in result_list:
            title = item.get("title", "").strip()
            if not title:
                continue

            # Authors
            author_list = item.get("authorList", {}).get("author", [])
            authors = []
            for a in author_list:
                full = a.get("fullName")
                if full:
                    authors.append(full)
                else:
                    parts = []
                    if a.get("lastName"):
                        parts.append(a["lastName"])
                    if a.get("firstName"):
                        parts.append(a["firstName"])
                    if parts:
                        authors.append(" ".join(parts))

            abstract = (item.get("abstractText") or "").strip()

            # URL — prefer DOI link, fall back to Europe PMC article page
            doi = item.get("doi")
            pmid = item.get("pmid")
            pmcid = item.get("pmcid")
            source_name = item.get("source", "MED")

            if doi:
                url = f"https://doi.org/{doi}"
            elif pmid:
                url = f"https://europepmc.org/article/{source_name}/{pmid}"
            else:
                url = f"https://europepmc.org/search?query={query}"

            # PDF URL from fullTextUrlList
            pdf_url = None
            url_list = item.get("fullTextUrlList", {}).get("fullTextUrl", [])
            for u in url_list:
                if u.get("documentStyle") == "pdf" and u.get("availability") == "Open access":
                    pdf_url = u.get("url")
                    break

            # Year
            year = None
            pub_year = item.get("pubYear")
            if pub_year:
                try:
                    year = int(pub_year)
                except (ValueError, TypeError):
                    pass

            results.append({
                "title": title,
                "authors": authors,
                "abstract": abstract or "No abstract available",
                "url": url,
                "doi": doi,
                "pdf_url": pdf_url,
                "year": year,
                "citation_count": item.get("citedByCount", 0),
                "source": "Europe PMC",
                "pubmed_id": pmid,
                "pmcid": pmcid,
            })

        return results

    t0 = _time.monotonic()
    try:
        results = await asyncio.wait_for(_with_rate_limit(), timeout=15.0)
    except TimeoutError:
        logger.error("Europe PMC search timeout after 15s for query: {}", query)
        results = make_error_result(
            "Europe PMC", "timeout", "Europe PMC search timed out after 15 seconds.",
        )
    except Exception as e:
        logger.error("Europe PMC search failed: {}", e)
        results = make_error_result("Europe PMC", "network", f"Europe PMC search failed: {e}")
    asyncio.create_task(
        record_tool_usage(
            "europepmc_search", query=query, results=results, elapsed=_time.monotonic() - t0,
        ),
    )
    return results
