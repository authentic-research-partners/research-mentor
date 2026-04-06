"""Stage 2 Search Node — paper search via OpenAlex.

Runs on first entry to Stage 2 (no papers yet):
1. Check variables exist
2. Search OpenAlex for relevant papers
3. Format and display results with review mode offer
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from research_mentor.question_workshop.hypothesis.state import HypothesisState
from research_mentor.question_workshop.hypothesis.utils.prompts import STAGE_2_PAPER_SEARCH_OFFER
from research_mentor.tools.search_and_enrich import search_and_enrich


def _format_papers_for_display(
    papers: list[dict[str, Any]],
) -> str:
    """Format papers into a readable markdown list."""
    if not papers:
        return "No relevant papers found."

    lines = [f"**Found {len(papers)} relevant papers:**\n"]
    for i, paper in enumerate(papers[:10], 1):
        title = paper.get("title", "Untitled")
        authors = paper.get("authors", "Unknown")
        year = paper.get("year", "N/A")
        journal = paper.get("journal", "Unknown journal")
        citations = paper.get("citation_count", 0)
        abstract = paper.get("abstract") or "No abstract available"

        lines.append(
            f"**{i}. {title}**\n"
            f"   *{authors}* ({year}) — {journal}\n"
            f"   Citations: {citations}\n"
            f"   {abstract}\n"
        )

    return "\n".join(lines)


async def stage_2_search(state: HypothesisState) -> dict[str, Any]:
    """Stage 2 Search: find papers via OpenAlex on first entry."""
    logger.info("Stage 2 Search: Searching for papers...")

    independent_var = state.get("independent_var", "")
    dependent_var = state.get("dependent_var", "")

    if not independent_var or not dependent_var:
        logger.info("Variables not yet identified — skipping search")
        return {
            "last_ai_response": (
                "Let's discuss the relationship between your variables "
                "before searching for papers."
            ),
        }

    query = f"{independent_var} {dependent_var}".strip()
    logger.info("Searching OpenAlex with query: '{}'", query)

    try:
        papers = await search_and_enrich(query=query, max_results=10)
    except Exception as e:
        logger.exception("OpenAlex search failed: {}", e)
        return {
            "last_ai_response": (
                "I tried to search for papers but encountered an issue. "
                "Let's continue — we can try again later."
            ),
        }

    logger.info("Found {} papers from OpenAlex", len(papers))

    if not papers:
        return {
            "last_ai_response": (
                f"I searched for papers on **{independent_var}** and "
                f"**{dependent_var}** but didn't find relevant results. "
                f"Let's refine our search terms or discuss what you "
                f"already know about the relationship."
            ),
            "papers_found_count": 0,
        }

    formatted = _format_papers_for_display(papers)
    response_text = formatted + STAGE_2_PAPER_SEARCH_OFFER

    return {
        "last_ai_response": response_text,
        "papers_reviewed": papers,
        "papers_found_count": len(papers),
    }
