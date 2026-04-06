"""Stage 2: Scrutiny Search — find papers under scrutiny via semantic search.

Queries the local Retraction Watch DB for expressions of concern and
corrections matching the student's field, then enriches with OpenAlex
for citation counts and abstracts.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from research_mentor.question_workshop.questioned._config import questioned_config
from research_mentor.question_workshop.questioned.schemas import ScrutinizedPaper
from research_mentor.tools.retraction_watch import search_scrutinized_papers


async def scrutiny_searcher(field: str) -> dict[str, Any]:
    """Search for papers under scrutiny in the given field.

    Returns:
        Dict with ``papers`` (list of ScrutinizedPaper) and ``tool_warnings``.
    """
    cfg = questioned_config()
    logger.info("Stage 2/4: Scrutiny Search — field='{}' max={}", field, cfg.max_papers)

    tool_warnings: list[dict[str, Any]] = []

    # Semantic search against Retraction Watch DB
    raw_results = await search_scrutinized_papers(
        field,
        max_results=cfg.max_papers,
    )

    if not raw_results:
        logger.warning("No scrutinized papers found for field '{}'", field)
        return {"papers": [], "tool_warnings": tool_warnings}

    logger.info("Found {} scrutinized papers for '{}'", len(raw_results), field)

    # Enrich with OpenAlex for citation counts
    papers = []
    for raw in raw_results:
        paper = ScrutinizedPaper(
            doi=raw.get("original_doi"),
            title=raw["title"],
            authors=raw.get("authors"),
            year=_extract_year(raw.get("retraction_date")),
            journal=raw.get("journal"),
            scrutiny_type=raw.get("retraction_nature", "unknown"),
            reason=raw.get("reason"),
            abstract=raw.get("abstract"),
            similarity=raw.get("similarity"),
        )

        # Enrich with OpenAlex (citation count, abstract, publication year)
        if paper.doi:
            enrichment = await _enrich_from_openalex(paper.doi, paper.title)
            if enrichment.get("citation_count") is not None:
                paper.citation_count = enrichment["citation_count"]
            if enrichment.get("abstract") and not paper.abstract:
                paper.abstract = enrichment["abstract"]
            if enrichment.get("year"):
                paper.year = str(enrichment["year"])

        papers.append(paper)

    logger.info("Enriched {} papers with citation data", len(papers))
    return {"papers": papers, "tool_warnings": tool_warnings}


def _extract_year(date_str: str | None) -> str | None:
    """Extract year from a date string like '2024-01-15'."""
    if not date_str:
        return None
    parts = date_str.strip().split("-")
    if parts and len(parts[0]) == 4:
        return parts[0]
    # Try other common formats
    parts = date_str.strip().split("/")
    if parts and len(parts[-1]) == 4:
        return parts[-1]
    return None


async def _enrich_from_openalex(
    doi: str, title: str,
) -> dict[str, Any]:
    """Get citation count + abstract from OpenAlex."""
    try:
        from research_mentor.tools.academic_search import search_openalex

        results = await search_openalex(f"doi:{doi}", max_results=1)
        if results and not any("error" in r for r in results):
            r = results[0]
            return {
                "citation_count": r.get("citation_count"),
                "abstract": r.get("abstract"),
                "year": r.get("year"),
            }

        # Fallback: search by title
        results = await search_openalex(title, max_results=1)
        if results and not any("error" in r for r in results):
            r = results[0]
            return {
                "citation_count": r.get("citation_count"),
                "abstract": r.get("abstract"),
                "year": r.get("year"),
            }
    except Exception:
        logger.debug("Failed to enrich from OpenAlex for {}", doi)
    return {}
