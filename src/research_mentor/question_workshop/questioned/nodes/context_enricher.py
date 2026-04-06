"""Stage 3: Context Enrichment — generate pedagogical context for each paper.

Uses LLM to generate ``what_it_claimed``, ``why_flagged``, and
``resolution_status`` from the paper metadata. This transforms raw
Retraction Watch data into student-readable context.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.questioned._config import questioned_config
from research_mentor.question_workshop.questioned.schemas import (
    EnrichedPaper,
    PaperEnrichment,
    ScrutinizedPaper,
)

_ENRICHMENT_SYSTEM = """\
You are restating facts about a scientific paper under scrutiny. \
Use ONLY the information provided below — do not invent details.

TASK: Produce three fields from the metadata given.

1. **what_it_claimed** — Restate in plain language what the paper \
claimed, using specifics from the title and abstract. Include any \
numbers, methods, or populations mentioned. 1-2 sentences.

2. **why_flagged** — Restate why the paper is flagged, using the \
"Reason" field. Translate codes like "Error in Analysis" into plain \
language (e.g., "the statistical analysis contained errors"). \
Do NOT name individuals. 1-2 sentences.

3. **resolution_status** — One of: "unresolved", "corrected", \
"retracted", "under review". Infer from the scrutiny type: \
expression of concern → "unresolved", correction → "corrected".

CRITICAL: Use the SPECIFIC details from the metadata. If the abstract \
mentions "N=2,400" or "Cronbach's alpha=0.92", include those numbers. \
Never write "This research likely explored..." — you HAVE the facts.

**Student level:** {student_level}"""


def _student_level_desc(demographics: dict[str, Any]) -> str:
    """Describe the student's level for prompt adaptation.

    Delegates to the shared ``build_student_profile_context`` so that
    education level, domain expertise, and professional experience are
    included alongside the population-aware guidance instruction.
    """
    from research_mentor.agent.prompts.shared import build_student_profile_context

    result = build_student_profile_context(demographics)
    return result if result else "Not specified — use accessible language."


async def context_enricher(
    papers: list[ScrutinizedPaper],
    demographics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Enrich scrutinized papers with LLM-generated pedagogical context.

    Returns:
        Dict with ``enriched_papers`` (list of EnrichedPaper).
    """
    cfg = questioned_config()
    student_level = _student_level_desc(demographics or {})
    system = _ENRICHMENT_SYSTEM.format(student_level=student_level)
    logger.info("Stage 3/4: Context Enrichment — {} papers", len(papers))

    enriched: list[EnrichedPaper] = []
    for i, paper in enumerate(papers):
        logger.info("Enriching paper {}/{}: '{}'", i + 1, len(papers), paper.title[:60])

        abstract_line = (
            f"Abstract: {paper.abstract}\n" if paper.abstract else ""
        )
        paper_info = (
            f"Title: {paper.title}\n"
            f"Journal: {paper.journal or 'Unknown'}\n"
            f"Scrutiny type: {paper.scrutiny_type}\n"
            f"Reason: {paper.reason or 'Not specified'}\n"
            f"Year: {paper.year or 'Unknown'}\n"
            f"{abstract_line}"
        )

        enrichment = await structured_call(
            PaperEnrichment,
            [
                SystemMessage(content=system),
                HumanMessage(content=f"Enrich this paper:\n\n{paper_info}"),
            ],
            thinking="medium",
            temperature=cfg.extraction_temperature,
        )

        enriched.append(EnrichedPaper(
            doi=paper.doi,
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            journal=paper.journal,
            scrutiny_type=paper.scrutiny_type,
            reason=paper.reason,
            citation_count=paper.citation_count,
            what_it_claimed=enrichment.what_it_claimed,
            why_flagged=enrichment.why_flagged,
            resolution_status=enrichment.resolution_status,
        ))

    logger.info("Enriched {} papers", len(enriched))
    return {"enriched_papers": enriched}
