"""Evidence search tool for claim verification.

Thin wrapper around ``research_mentor.tools.search_and_enrich.search_and_enrich``
with claim-specific multi-query logic and cross-query DOI deduplication.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from research_mentor.tools.search_and_enrich import search_and_enrich
from research_mentor.tools.status import ToolStatus


async def search_claim_evidence(
    claim_text: str,
    independent_var: str,
    dependent_var: str,
    max_papers: int = 15,
) -> dict[str, Any]:
    """Search for peer-reviewed evidence related to a claim.

    Generates multiple search queries from the claim variables
    and aggregates + deduplicates results.

    Args:
        claim_text: The normalized claim text.
        independent_var: The independent variable (X).
        dependent_var: The dependent variable (Y).
        max_papers: Maximum papers to return (default: 15).

    Returns:
        Dict with keys: papers, queries_used, total_found.
    """
    logger.info(
        "Searching evidence for claim: X='{}', Y='{}'",
        independent_var, dependent_var,
    )

    queries = _generate_search_queries(independent_var, dependent_var, claim_text)

    all_papers: list[dict[str, Any]] = []
    queries_used: list[str] = []
    tool_statuses: list[ToolStatus] = []

    for query in queries:
        papers = await search_and_enrich(query=query, max_results=5)

        # Separate errors from real papers
        real_papers = [p for p in papers if "error" not in p]
        for p in papers:
            if "tool_status" in p:
                tool_statuses.append(p["tool_status"])

        if real_papers:
            all_papers.extend(real_papers)
            queries_used.append(query)

        logger.debug("Query '{}': found {} papers", query, len(real_papers))

    # Deduplicate by DOI
    seen_dois: set[str] = set()
    unique_papers: list[dict[str, Any]] = []
    for paper in all_papers:
        doi = paper.get("doi")
        if doi and doi in seen_dois:
            continue
        if doi:
            seen_dois.add(doi)
        unique_papers.append(paper)

    formatted_papers = [
        {
            "title": p.get("title", "Untitled"),
            "authors": _format_authors(p.get("authors", [])),
            "year": p.get("year"),
            "journal": p.get("journal"),
            "doi": p.get("doi"),
            "url": p.get("url"),
            "citation_count": p.get("citation_count", 0),
            "abstract": p.get("abstract") or None,
        }
        for p in unique_papers[:max_papers]
    ]

    logger.info(
        "Evidence search complete: {} unique papers from {} queries",
        len(formatted_papers), len(queries_used),
    )

    return {
        "papers": formatted_papers,
        "queries_used": queries_used,
        "total_found": len(unique_papers),
        "tool_statuses": tool_statuses,
    }


def _generate_search_queries(
    independent_var: str,
    dependent_var: str,
    claim_text: str,
) -> list[str]:
    """Generate multiple search queries for comprehensive evidence search.

    Returns queries from most specific to most general.
    """
    queries = [
        # Both variables together
        f"{independent_var} {dependent_var}",
        # Add "effect"
        f"{independent_var} effect {dependent_var}",
        # Clinical/research focus
        f"{independent_var} {dependent_var} study clinical trial",
        # Review/meta-analysis (most reliable)
        f"{independent_var} {dependent_var} systematic review meta-analysis",
    ]

    # Key nouns from claim text as fallback query
    claim_words = claim_text.lower().split()
    key_words = [
        w for w in claim_words
        if len(w) > 4
        and w not in {"which", "their", "would", "could", "should", "about", "these"}
    ][:3]
    if key_words:
        queries.append(" ".join(key_words))

    return queries


def _format_authors(authors: list[str | dict[str, str]]) -> str:
    """Format author list as 'First Author et al.' or full list if short."""
    if not authors:
        return "Unknown"

    names: list[str] = []
    for a in authors:
        if isinstance(a, dict):
            name = a.get("name", "")
        else:
            name = a
        if name:
            names.append(name)

    if not names:
        return "Unknown"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]} et al."
