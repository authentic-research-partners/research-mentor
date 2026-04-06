"""Stage 1: Phenomenon Mapper — search literature, identify surprising patterns.

Classifies domain, searches for papers, extracts cross-domain signals,
and produces a structured analysis of the phenomenon landscape.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.theory.pipeline.schemas import (
    DomainClassification,
    PhenomenonAnalysis,
    PhenomenonResult,
)
from research_mentor.tools.search_and_enrich import search_and_enrich

DOMAIN_CLASSIFICATION_PROMPT = """\
Classify this research topic and extract a clean academic search query.

Topic: {topic}

Rules:
- domain_type: 'natural_science' if it involves physical/biological/chemical \
phenomena. 'social_science' if it involves human behavior, society, cognition, \
economics. 'off_topic' if it's not a research topic at all.
- search_query: Extract a clean 3-5 word academic search query. Remove ALL filler \
words. Use only core scientific nouns and terms.

Both natural science AND social science are valid. Only reject clearly off-topic input."""

PHENOMENON_ANALYSIS_PROMPT = """\
You are analyzing published research papers and a student's observations about \
a scientific topic. Extract patterns from the combined evidence.

Student's observations:
{observations_text}

Papers found:
{papers_text}

Extract:
1. surprising_findings: Results that contradict expectations or conventional wisdom \
(2-4 items). Look for anomalies, unexpected correlations, counter-intuitive results.
2. cross_domain_connections: Links to other fields mentioned in papers or implied \
by the student's observations (0-3 items).
3. unexplained_phenomena: Patterns that lack established explanation — gaps where \
theory is incomplete (1-3 items).
4. domain_summary: One-sentence overview of this research area based on the papers.
5. key_observations: The student's observations enriched with literature context — \
connect what they noticed to what the literature says (2-5 items)."""


async def map_phenomenon(
    observations_text: str,
    domain: str,
    population: str,
    student_profile_text: str = "",
) -> PhenomenonResult:
    """Stage 1: Map the phenomenon — search, analyze, extract patterns.

    Args:
        observations_text: Student's observations/puzzles (free text).
        domain: Research domain (e.g. "crystal formation").
        population: Student level (e.g. "high_school", "university").
        student_profile_text: Pre-formatted student profile context.

    Returns:
        PhenomenonResult with papers, patterns, and observations.
    """
    logger.info("Theory Stage 1: Mapping phenomenon for '{}'", domain[:80])

    # Step 1: Domain classification + search query extraction
    domain_class = await structured_call(
        DomainClassification,
        [
            SystemMessage(content=DOMAIN_CLASSIFICATION_PROMPT.format(
                topic=f"{domain}. Student observations: {observations_text[:200]}",
            )),
            HumanMessage(content="Classify this topic."),
        ],
        thinking="high",
        temperature=0.0,
    )

    logger.info(
        "Stage 1: Domain={}, query='{}'",
        domain_class.domain_type, domain_class.search_query,
    )

    if domain_class.domain_type == "off_topic":
        logger.warning("Stage 1: Off-topic input rejected")
        return PhenomenonResult(
            papers=[],
            surprising_findings=[],
            cross_domain_connections=[],
            unexplained_phenomena=[],
            domain_summary="Topic was classified as off-topic for research.",
            key_observations=[],
        )

    # Step 2: Primary domain search
    search_query = domain_class.search_query or domain
    papers = await search_and_enrich(query=search_query, max_results=15)

    # Step 3: Cross-domain search if observations mention other fields
    cross_domain_papers = await _search_cross_domain(
        observations_text, domain, search_query,
    )
    if cross_domain_papers:
        papers = papers + cross_domain_papers

    if not papers:
        logger.warning("Stage 1: No papers found for '{}'", search_query)
        return PhenomenonResult(
            papers=[],
            surprising_findings=[],
            cross_domain_connections=[],
            unexplained_phenomena=[],
            domain_summary=f"No published research found for '{domain}'.",
            key_observations=[observations_text[:200]],
        )

    logger.info("Stage 1: Found {} total papers", len(papers))

    # Step 4: Extract patterns from papers + observations
    papers_text = _format_papers(papers)

    analysis = await structured_call(
        PhenomenonAnalysis,
        [
            SystemMessage(content=PHENOMENON_ANALYSIS_PROMPT.format(
                observations_text=observations_text,
                papers_text=papers_text,
            )),
            HumanMessage(content="Analyze patterns from these papers and observations."),
        ],
        thinking="high",
        temperature=0.0,
    )

    logger.info(
        "Stage 1 complete: {} surprising, {} cross-domain, {} unexplained, {} observations",
        len(analysis.surprising_findings),
        len(analysis.cross_domain_connections),
        len(analysis.unexplained_phenomena),
        len(analysis.key_observations),
    )

    return PhenomenonResult(
        papers=papers,
        surprising_findings=analysis.surprising_findings,
        cross_domain_connections=analysis.cross_domain_connections,
        unexplained_phenomena=analysis.unexplained_phenomena,
        domain_summary=analysis.domain_summary,
        key_observations=analysis.key_observations,
    )


async def _search_cross_domain(
    observations_text: str,
    domain: str,
    primary_query: str,
) -> list[dict[str, Any]]:
    """Search for cross-domain connections mentioned in observations.

    Looks for keywords suggesting other fields and runs up to 2
    additional searches in parallel.
    """
    # Use structured_call to detect cross-domain hints rather than keyword matching
    from pydantic import BaseModel
    from pydantic import Field as PydanticField

    class CrossDomainHints(BaseModel):
        """Detect cross-domain connections in student observations."""
        hints: list[str] = PydanticField(
            default_factory=list,
            description=(
                "Other scientific fields referenced or implied in the text "
                "(0-2 items). Only include if clearly mentioned. "
                "Examples: 'biology', 'economics', 'neuroscience'."
            ),
        )

    try:
        result = await structured_call(
            CrossDomainHints,
            [
                SystemMessage(content=(
                    f"The student is studying '{domain}'. Do their observations "
                    f"mention or imply connections to OTHER scientific fields?\n\n"
                    f"Observations: {observations_text[:500]}\n\n"
                    f"Extract 0-2 other fields. Only include clearly referenced fields."
                )),
                HumanMessage(content="Extract cross-domain hints."),
            ],
            thinking="high",
            temperature=0.0,
        )
    except Exception:
        logger.debug("Cross-domain hint detection failed, skipping")
        return []

    if not result.hints:
        return []

    # Run up to 2 cross-domain searches in parallel
    hints = result.hints[:2]
    logger.info("Stage 1: Cross-domain search for: {}", hints)

    tasks = [
        search_and_enrich(
            query=f"{domain} {hint}",
            max_results=10,
        )
        for hint in hints
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_papers: list[dict[str, Any]] = []
    for r in results:
        if isinstance(r, list):
            all_papers.extend(r)

    return all_papers


def _format_papers(papers: list[dict[str, Any]]) -> str:
    """Format papers for LLM extraction."""
    lines = []
    for i, p in enumerate(papers[:15], 1):
        title = p.get("title", "Untitled")
        year = p.get("year", "")
        abstract = p.get("abstract", "No abstract")
        lines.append(f"[{i}] {title} ({year})\nAbstract: {abstract}\n")
    return "\n".join(lines)
