"""Stage 1: Landscape Explorer — domain guard, literature search, pattern extraction.

Classifies the topic (natural science guard), searches papers via
search_and_enrich, extracts landscape patterns and field maturity.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.gaps.pipeline.schemas import LandscapeResult
from research_mentor.question_workshop.gaps.schemas import (
    DomainClassification,
    LandscapePatterns,
    ModeRecommendation,
)
from research_mentor.question_workshop.gaps.utils.prompts import (
    PHASE_1_LANDSCAPE_EXTRACTION,
    PHASE_1_MODE_RECOMMENDATION,
)
from research_mentor.tools.search_and_enrich import search_and_enrich


async def explore_landscape(
    field_topic: str,
    interest_area: str | None,
    population: str,
    config: Any | None = None,
) -> LandscapeResult:
    """Search literature, extract landscape patterns, assess field maturity.

    Raises ValueError if domain is social_science, refused, or off_topic.
    """
    logger.info("Gaps Pipeline Stage 1: Exploring '{}'", field_topic[:80])

    # --- Domain classification guard ---
    await _classify_domain(field_topic)

    # --- Broad landscape search ---
    papers = await search_and_enrich(query=field_topic, max_results=15)

    if not papers or (len(papers) == 1 and "error" in papers[0]):
        logger.warning("Stage 1: No papers found for '{}'", field_topic)
        return LandscapeResult(
            papers=[],
            domain_summary=f"No published papers found for '{field_topic}'.",
        )

    # --- Targeted search if interest_area provided ---
    if interest_area:
        targeted_query = f"{field_topic} {interest_area}"
        targeted_papers = await search_and_enrich(
            query=targeted_query, max_results=10,
        )
        if targeted_papers and not (
            len(targeted_papers) == 1 and "error" in targeted_papers[0]
        ):
            # Merge, deduplicate by DOI
            existing_dois = {
                p.get("doi") for p in papers if p.get("doi")
            }
            for tp in targeted_papers:
                doi = tp.get("doi")
                if not doi or doi not in existing_dois:
                    papers.append(tp)
                    if doi:
                        existing_dois.add(doi)

    logger.info("Stage 1: {} papers total after merge", len(papers))

    # --- Extract landscape patterns ---
    papers_text = _format_papers(papers)
    patterns = await structured_call(
        LandscapePatterns,
        [
            SystemMessage(content=PHASE_1_LANDSCAPE_EXTRACTION.format(
                papers_text=papers_text,
            )),
            HumanMessage(content="Extract landscape patterns from these papers."),
        ],
        thinking="high",
        temperature=0.0,
    )

    # --- Assess field maturity ---
    mode_rec = await structured_call(
        ModeRecommendation,
        [
            SystemMessage(content=PHASE_1_MODE_RECOMMENDATION.format(
                domain_topic=field_topic,
                field_maturity="unknown",
                consensus_count=len(patterns.consensus),
                debate_count=len(patterns.debates),
                gap_count=len(patterns.gaps),
                frontier_count=len(patterns.frontiers),
            )),
            HumanMessage(content="Recommend a strategic mode for this field."),
        ],
        thinking="medium",
        temperature=0.0,
    )

    logger.info(
        "Stage 1 complete: {} papers, maturity={}, {} consensus, {} debates, "
        "{} gaps, {} frontiers",
        len(papers), mode_rec.field_maturity,
        len(patterns.consensus), len(patterns.debates),
        len(patterns.gaps), len(patterns.frontiers),
    )

    return LandscapeResult(
        papers=papers,
        consensus=patterns.consensus,
        debates=patterns.debates,
        gaps=patterns.gaps,
        frontiers=patterns.frontiers,
        field_maturity=mode_rec.field_maturity,
        domain_summary=patterns.domain_summary,
    )


async def _classify_domain(field_topic: str) -> None:
    """Classify domain and raise ValueError for non-natural-science topics."""
    from research_mentor.question_workshop.domain import (
        get_domain_classification_prompt,
    )

    prompt = get_domain_classification_prompt().format(topic=field_topic)
    classification = await structured_call(
        DomainClassification,
        [
            SystemMessage(content=prompt),
            HumanMessage(content="Classify this topic."),
        ],
        thinking="high",
        temperature=0.0,
    )

    logger.info(
        "Stage 1: Domain classification — type={}, confidence={}",
        classification.domain_type, classification.confidence,
    )

    if classification.domain_type == "social_science":
        from research_mentor.components_registry import get_redirect_phrase

        msg = (
            get_redirect_phrase("gaps", "hypothesis")
            or "That topic is better suited for the Hypothesis workshop."
        )
        raise ValueError(msg)

    if classification.domain_type == "refused":
        raise ValueError(
            "That topic goes beyond typical science and involves "
            "social, political, ethical, or clinical considerations "
            "that we can't address with methodology alone. "
            "Try a different topic focused on natural science."
        )

    if classification.domain_type == "off_topic":
        raise ValueError(
            "That doesn't appear to be a research topic. "
            "Please provide a natural science topic like physics, "
            "chemistry, biology, or earth science."
        )


def _format_papers(papers: list[dict[str, Any]]) -> str:
    """Format papers for LLM extraction."""
    lines = []
    for i, paper in enumerate(papers[:15], 1):
        title = paper.get("title", "Untitled")
        year = paper.get("year", "")
        abstract = paper.get("abstract", "No abstract available")
        citations = paper.get("citation_count", 0)
        lines.append(
            f"[{i}] {title} ({year}, {citations} citations)\n"
            f"Abstract: {abstract}\n"
        )
    return "\n".join(lines)
