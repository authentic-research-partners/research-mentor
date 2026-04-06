"""Multi-source academic search with enrichment.

``search_and_enrich()`` is the primary function workshops should call.
It runs parallel discovery (OpenAlex + Semantic Scholar + Europe PMC),
deduplicates by DOI, enriches papers missing abstracts (arXiv + CORE),
adds OA access URLs (Unpaywall), and sorts results by quality tier.

See docs/philosophy/academic-search-strategy.md for the design rationale.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from research_mentor.tools.academic_search import (
    search_openalex,
    search_semantic_scholar,
)
from research_mentor.tools.core_api import search_core_by_doi
from research_mentor.tools.europepmc_search import search_europepmc
from research_mentor.tools.unpaywall import lookup_unpaywall

# Concurrency cap for enrichment calls (avoid hammering APIs)
_ENRICH_SEMAPHORE = asyncio.Semaphore(5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def search_and_enrich(
    query: str,
    max_results: int = 15,
    *,
    mailto: str | None = None,
) -> list[dict[str, Any]]:
    """Search multiple academic sources and enrich results.

    Phase 1 — Discovery (parallel):
        OpenAlex + Semantic Scholar, then DOI deduplication.

    Phase 2 — Enrichment (parallel):
        For papers missing abstracts: arXiv (if has arXiv ID) + CORE (if has DOI).
        For papers missing OA links: Unpaywall DOI lookup.
        All enrichment is non-fatal — failures are logged at debug level.

    Papers without abstracts are dropped after enrichment (title-only
    papers give neither the LLM nor the student useful information).

    Remaining papers are sorted by quality:
        0 = has abstract + OA access URL (student can read immediately)
        1 = has abstract only

    Args:
        query: Search query (e.g., "water droplet optics").
        max_results: Maximum results after deduplication (default 15).
        mailto: Email for OpenAlex polite pool. If None, uses config value.

    Returns:
        List of enriched paper dicts, sorted by quality tier then citation count.
    """
    # Safety review: check for accidental PII leaks before sending externally
    from research_mentor.tools.query_safety_review import review_outgoing_query

    if not await review_outgoing_query(query, "search_and_enrich"):
        from research_mentor.tools.status import make_error_result
        return make_error_result(
            "Academic Search", "safety", "Query blocked by safety review",
        )

    # Over-request from discovery to compensate for losses from dedup + filtering.
    # After dedup and title-only filtering, we may lose 30-50% of results.
    discovery_count = max_results * 2

    # Phase 1: Parallel discovery
    papers = await _discover(query, discovery_count, mailto)

    if not papers:
        return []

    # Deduplicate by DOI (prefer version with abstract)
    papers = _deduplicate_by_doi(papers)

    # Trim before enrichment (avoid enriching papers we'll discard)
    papers = papers[: discovery_count]

    # Phase 2: Parallel enrichment
    papers = await _enrich(papers)

    # Drop title-only papers (enrichment couldn't find an abstract either)
    papers = [
        p for p in papers
        if p.get("abstract", "No abstract available") != "No abstract available"
    ]

    # Sort by quality tier, then citation count
    papers = _sort_by_quality(papers)

    # Final trim to requested count
    return papers[:max_results]


# ---------------------------------------------------------------------------
# Phase 1: Discovery
# ---------------------------------------------------------------------------


async def _discover(
    query: str, max_results: int, mailto: str | None,
) -> list[dict[str, Any]]:
    """Run parallel discovery across OpenAlex + Semantic Scholar + Europe PMC.

    If all fail, return empty list. Partial failures are tolerated.
    """
    openalex_task = asyncio.create_task(
        search_openalex(query, max_results=max_results, mailto=mailto),
    )
    s2_task = asyncio.create_task(
        search_semantic_scholar(query, max_results=max_results),
    )
    epmc_task = asyncio.create_task(
        search_europepmc(query, max_results=max_results),
    )

    results = await asyncio.gather(
        openalex_task, s2_task, epmc_task, return_exceptions=True,
    )

    all_papers: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, Exception):
            logger.debug("Discovery source failed: {}", result)
            continue
        if isinstance(result, list):
            # Filter out error entries
            for paper in result:
                if "error" not in paper:
                    all_papers.append(paper)

    return all_papers


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _deduplicate_by_doi(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate papers by DOI, keeping the version with an abstract.

    Papers without DOI are always kept (can't deduplicate them).
    """
    seen_dois: dict[str, dict[str, Any]] = {}
    no_doi: list[dict[str, Any]] = []

    for paper in papers:
        doi = paper.get("doi")
        if not doi:
            no_doi.append(paper)
            continue

        # Normalize DOI for comparison
        normalized = doi.lower().strip()
        existing = seen_dois.get(normalized)

        if existing is None:
            seen_dois[normalized] = paper
        else:
            # Prefer the version with an abstract
            existing_has_abstract = (
                existing.get("abstract", "No abstract available")
                != "No abstract available"
            )
            new_has_abstract = (
                paper.get("abstract", "No abstract available")
                != "No abstract available"
            )
            if new_has_abstract and not existing_has_abstract:
                seen_dois[normalized] = paper

    return list(seen_dois.values()) + no_doi


# ---------------------------------------------------------------------------
# Phase 2: Enrichment
# ---------------------------------------------------------------------------


async def _enrich(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich papers: fill missing abstracts and add OA access URLs."""
    tasks = [_enrich_single(paper) for paper in papers]
    return await asyncio.gather(*tasks)


async def _enrich_single(paper: dict[str, Any]) -> dict[str, Any]:
    """Enrich a single paper (abstract + access URL). Non-fatal."""
    async with _ENRICH_SEMAPHORE:
        # Abstract enrichment (only if missing)
        abstract = paper.get("abstract", "No abstract available")
        needs_abstract = abstract == "No abstract available"

        if needs_abstract:
            paper = await _enrich_abstract(paper)

        # Access URL enrichment (only if no OA link yet)
        if not _has_oa_link(paper):
            paper = await _enrich_access_url(paper)

    return paper


async def _enrich_abstract(paper: dict[str, Any]) -> dict[str, Any]:
    """Try to fill in a missing abstract from arXiv or CORE."""
    # Try arXiv first (if paper has arXiv ID)
    arxiv_id = paper.get("arxiv_id")
    if arxiv_id:
        try:
            from research_mentor.tools.academic_search import search_arxiv

            results = await search_arxiv(arxiv_id, max_results=1)
            if results and "error" not in results[0]:
                arxiv_paper = results[0]
                if arxiv_paper.get("abstract"):
                    paper["abstract"] = arxiv_paper["abstract"]
                    if not paper.get("pdf_url"):
                        paper["pdf_url"] = arxiv_paper.get("pdf_url")
                    logger.debug("Enriched abstract from arXiv: {}", arxiv_id)
                    return paper
        except Exception:
            logger.debug("arXiv abstract enrichment failed for {}", arxiv_id)

    # Try CORE (if paper has DOI)
    doi = paper.get("doi")
    if doi:
        try:
            core_data = await search_core_by_doi(doi)
            if core_data and core_data.get("abstract"):
                paper["abstract"] = core_data["abstract"]
                if core_data.get("download_url"):
                    paper["download_url"] = core_data["download_url"]
                logger.debug("Enriched abstract from CORE: {}", doi)
                return paper
        except Exception:
            logger.debug("CORE abstract enrichment failed for {}", doi)

    return paper


async def _enrich_access_url(paper: dict[str, Any]) -> dict[str, Any]:
    """Try to find an OA access URL via Unpaywall."""
    doi = paper.get("doi")
    if not doi:
        return paper

    try:
        unpaywall_data = await lookup_unpaywall(doi)
        if unpaywall_data:
            if unpaywall_data.get("pdf_url"):
                paper["unpaywall_url"] = unpaywall_data["pdf_url"]
            elif unpaywall_data.get("oa_url"):
                paper["unpaywall_url"] = unpaywall_data["oa_url"]
            logger.debug("Found OA link via Unpaywall: {}", doi)
    except Exception:
        logger.debug("Unpaywall lookup failed for {}", doi)

    return paper


def _has_oa_link(paper: dict[str, Any]) -> bool:
    """Check if paper already has an OA access link."""
    if paper.get("pdf_url"):
        return True
    if paper.get("download_url"):
        return True
    if paper.get("unpaywall_url"):
        return True
    return False


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def _sort_by_quality(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort papers by quality tier, then by citation count descending.

    Tier 0: has abstract + OA access URL (best — student can read immediately)
    Tier 1: has abstract only
    (Title-only papers are filtered out before sorting.)
    """

    def _quality_key(paper: dict[str, Any]) -> tuple[int, int]:
        abstract = paper.get("abstract", "No abstract available")
        has_abstract = abstract != "No abstract available"
        has_oa = _has_oa_link(paper)

        if has_abstract and has_oa:
            tier = 0
        elif has_abstract:
            tier = 1
        else:
            tier = 2

        # Negative citation count for descending sort within tier
        citations = -(paper.get("citation_count", 0) or 0)
        return (tier, citations)

    return sorted(papers, key=_quality_key)
