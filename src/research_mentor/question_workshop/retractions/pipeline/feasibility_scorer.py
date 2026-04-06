"""Stage 5: Feasibility Scorer — rank questions by feasibility dimensions.

Scores each question on 6 dimensions, ranks by overall_score, returns top 3-5.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.retractions.pipeline.schemas import (
    ImpactedCase,
    ImpactedCases,
    RetractionFeasibility,
    RetractionQuestion,
    RetractionQuestions,
    ScoredRetractionQuestion,
    ScoredRetractionQuestions,
)

_LLM_SEMAPHORE = asyncio.Semaphore(20)

_FEASIBILITY_SYSTEM = """\
You are scoring a research question derived from a retracted paper.

Score each dimension from 0 (worst) to 10 (best):

1. **data_availability** — Can this question be investigated with publicly \
available data, standard lab equipment, or accessible tools?
2. **student_accessibility** — Is this appropriate for the student's level? \
Consider required skills, time commitment, and resources.
3. **scientific_value** — How important is it to re-answer this question? \
Higher if the retraction left a significant gap in the field.
4. **methodology_soundness** — Do the proposed methodology improvements \
actually address the original failure? Higher if the safeguards are well-matched.
5. **integrity_value** — How much does pursuing this question teach about \
research integrity? Higher if the student learns about pre-registration, \
replication, blinding, or other integrity practices.
6. **overall_score** — Holistic recommendation. Not a simple average — weigh \
scientific_value and methodology_soundness most heavily.

Provide brief reasoning (2-3 sentences) explaining the scores.

**Student level:** {population}"""


async def _score_one(
    question: RetractionQuestion,
    population: str,
) -> RetractionFeasibility:
    """Score a single question."""
    async with _LLM_SEMAPHORE:
        return await _score_one_inner(question, population)


async def _score_one_inner(
    question: RetractionQuestion,
    population: str,
) -> RetractionFeasibility:
    """Score a single question (under semaphore)."""
    system = _FEASIBILITY_SYSTEM.format(population=population)
    q_info = (
        f"Question: {question.question}\n"
        f"Reopened gap: {question.reopened_gap}\n"
        f"Avoids original mistakes: {question.avoids_original_mistakes}\n"
        f"Methodology improvements: {', '.join(question.methodology_improvements)}\n"
        f"Why interesting: {question.why_interesting}"
    )

    return await structured_call(
        RetractionFeasibility,
        [
            SystemMessage(content=system),
            HumanMessage(content=f"Score this question:\n\n{q_info}"),
        ],
        thinking="high",
        temperature=0.2,
    )


def _find_source_case(
    doi: str, impacted_cases: ImpactedCases,
) -> ImpactedCase | None:
    """Find the impacted case matching a DOI."""
    for ic in impacted_cases.cases:
        if ic.case.doi == doi:
            return ic
    return None


async def score_feasibility(
    candidates: RetractionQuestions,
    impacted_cases: ImpactedCases,
    population: str,
    max_output: int = 5,
) -> ScoredRetractionQuestions:
    """Stage 5: Score and rank questions by feasibility (parallel).

    Args:
        candidates: Stage 4 output.
        impacted_cases: Stage 3 output (for attaching source case data).
        population: Student level for calibrating accessibility scores.
        max_output: Maximum questions to return (default 5).

    Returns:
        ScoredRetractionQuestions ranked by overall_score (top max_output).
    """
    logger.info("Stage 5/5: Feasibility Scorer — {} questions", len(candidates.questions))

    # Score all questions in parallel
    tasks = [
        _score_one(q, population) for q in candidates.questions
    ]
    scores = await asyncio.gather(*tasks)

    # Build scored questions with source case data
    scored: list[ScoredRetractionQuestion] = []
    for question, feasibility in zip(candidates.questions, scores, strict=True):
        source = _find_source_case(question.source_case_doi, impacted_cases)
        source_case = source.case if source else impacted_cases.cases[0].case
        failure_type = source.analysis.failure_type if source else "unknown"

        scored.append(ScoredRetractionQuestion(
            question=question.question,
            source_case=source_case,
            failure_type=failure_type,
            reopened_gap=question.reopened_gap,
            avoids_original_mistakes=question.avoids_original_mistakes,
            methodology_improvements=question.methodology_improvements,
            why_interesting=question.why_interesting,
            feasibility=feasibility,
        ))

    # Rank by overall_score, return top N
    scored.sort(key=lambda sq: sq.feasibility.overall_score, reverse=True)
    top = scored[:max_output]

    logger.info(
        "Feasibility scoring complete: top scores={}",
        [sq.feasibility.overall_score for sq in top],
    )
    return ScoredRetractionQuestions(questions=top)
