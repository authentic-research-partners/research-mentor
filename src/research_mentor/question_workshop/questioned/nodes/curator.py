"""Stage 4: Curation — select and frame the most pedagogically interesting cases.

LLM-based curation that selects 3-5 papers from the enriched list,
prioritizing methodological issues over administrative corrections,
high citation counts, and diverse failure types.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.questioned._config import questioned_config
from research_mentor.question_workshop.questioned.schemas import (
    CuratedCase,
    CurationOutput,
    EnrichedPaper,
    QuestionedResult,
)

_CURATION_SYSTEM = """\
You are curating a set of scientific papers under scrutiny for a student \
learning about scientific skepticism.

Your task: select the {max_curated} most pedagogically interesting cases \
from the list below, then for each one generate:

1. A **pedagogical lesson** (1-2 sentences) — what this case teaches about \
   how science self-corrects, how errors can persist, or why peer review is \
   necessary but not sufficient.

2. **Discussion questions** (2-3 questions) — critical thinking questions that \
   help the student reason about the case. Questions should probe methodology, \
   evidence evaluation, and the scientific process. Each question must be \
   ONLY about its own case — never reference or compare other cases.

**Selection priorities** (in order):
- Methodological issues (fabrication, statistical errors, unreliable methods) \
  are MORE interesting than administrative corrections (authorship disputes, \
  duplicate publication)
- Higher citation count = bigger impact on the field = more interesting
- Diverse failure types — don't select 3 papers all flagged for the same reason
- Clear, understandable failure mode — cases students can reason about

**Also generate a meta_lesson** (2-3 sentences) — an overarching takeaway about \
how science self-corrects. This should connect the individual cases into a \
broader lesson about scientific integrity.

Frame everything around methodology and processes. NEVER name individual \
researchers or assign personal blame. Focus on systemic lessons.

**Student level:** {student_level}
Adapt language complexity and discussion question depth to this level. \
For younger students, prefer concrete, accessible cases with straightforward \
failure modes. For advanced students, include more nuanced methodological \
discussion."""


def _student_level_desc(demographics: dict[str, Any]) -> str:
    """Describe the student's level for prompt adaptation.

    Delegates to the shared ``build_student_profile_context`` so that
    education level, domain expertise, and professional experience are
    included alongside the population-aware guidance instruction.
    """
    from research_mentor.agent.prompts.shared import build_student_profile_context

    result = build_student_profile_context(demographics)
    return result if result else "Not specified — use accessible language."


async def curator(
    enriched_papers: list[EnrichedPaper],
    field: str,
    demographics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Curate enriched papers into pedagogically framed cases.

    Returns:
        Dict with ``questioned_result`` (QuestionedResult).
    """
    cfg = questioned_config()
    max_curated = min(cfg.max_curated, len(enriched_papers))
    logger.info(
        "Stage 4/4: Curation — selecting {} from {} papers",
        max_curated, len(enriched_papers),
    )

    # Format papers for LLM
    paper_descriptions = []
    for i, paper in enumerate(enriched_papers):
        desc = (
            f"[{i}] Title: {paper.title}\n"
            f"    Journal: {paper.journal or 'Unknown'}\n"
            f"    Year: {paper.year or 'Unknown'}\n"
            f"    Scrutiny type: {paper.scrutiny_type}\n"
            f"    Reason: {paper.reason or 'Not specified'}\n"
            f"    Citation count: {paper.citation_count or 'Unknown'}\n"
            f"    What it claimed: {paper.what_it_claimed}\n"
            f"    Why flagged: {paper.why_flagged}\n"
            f"    Status: {paper.resolution_status}"
        )
        paper_descriptions.append(desc)

    papers_text = "\n\n".join(paper_descriptions)
    student_level = _student_level_desc(demographics or {})
    system = _CURATION_SYSTEM.format(
        max_curated=max_curated, student_level=student_level,
    )

    curation = await structured_call(
        CurationOutput,
        [
            SystemMessage(content=system),
            HumanMessage(
                content=f"Field: {field}\n\nPapers:\n\n{papers_text}",
            ),
        ],
        thinking="high",
        temperature=cfg.curation_temperature,
    )

    # Build curated cases from selected indices
    cases: list[CuratedCase] = []
    for idx, (lesson, questions) in enumerate(zip(
        curation.pedagogical_lessons,
        curation.discussion_questions,
        strict=False,
    )):
        if idx >= len(curation.selected_indices):
            break
        paper_idx = curation.selected_indices[idx]
        if paper_idx < 0 or paper_idx >= len(enriched_papers):
            continue

        paper = enriched_papers[paper_idx]
        cases.append(CuratedCase(
            title=paper.title,
            authors=paper.authors,
            journal=paper.journal,
            year=paper.year,
            doi=paper.doi,
            scrutiny_type=paper.scrutiny_type,
            reason=paper.reason,
            citation_count=paper.citation_count,
            what_it_claimed=paper.what_it_claimed,
            why_flagged=paper.why_flagged,
            resolution_status=paper.resolution_status,
            pedagogical_lesson=lesson,
            discussion_questions=questions if questions else ["What can we learn from this case?"],
        ))

    if not cases:
        # Fail loud: a fabricated case from the first paper would present invented
        # pedagogical content as if curated (fail-fast principle: no graceful
        # degradation that hides issues). An empty curation is a real failure to surface.
        raise RuntimeError(
            f"Curation produced no valid cases for field {field!r} from "
            f"{len(enriched_papers)} enriched paper(s) "
            f"(selected_indices={curation.selected_indices})."
        )

    result = QuestionedResult(
        cases=cases,
        field=field,
        meta_lesson=curation.meta_lesson,
    )

    logger.info("Curated {} cases for field '{}'", len(cases), field)
    return {"questioned_result": result}
