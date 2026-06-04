"""Stage 6: Feasibility Scorer — assess and rank theory-grounded questions.

Scores each question on methodological feasibility, data availability,
student accessibility, scientific value, and framework relevance.
Runs scoring in parallel via asyncio.gather, returns top 3-5.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.theory.pipeline.schemas import (
    PhenomenonResult,
    ScoredTheoryQuestion,
    TheoryFeasibility,
    TheoryQuestion,
    TheoryQuestions,
)

_LLM_SEMAPHORE = asyncio.Semaphore(20)

FEASIBILITY_PROMPT = """\
Assess the feasibility of this theory-grounded research question for a \
{population} student.

Question: {question}
Type: {question_type}
Framework connection: {framework_connection}

Score each dimension (1-10):
1. methodological_feasibility: Can this be investigated with available methods? \
(10=standard lab/survey methods, 1=no known method exists)
2. data_availability: Is evidence or data accessible? \
(10=public datasets exist, 1=requires proprietary data or new experiments)
3. student_accessibility: Appropriate for a {population} student? \
(10=straightforward, 1=requires graduate-level training)
4. scientific_value: Would answering this advance understanding? \
(10=would resolve a major open question, 1=reproduces known results)
5. framework_relevance: How well does it test or extend the framework? \
(10=directly tests a core prediction, 1=tangentially related)
6. overall_score: Overall recommendation considering all factors.

Provide brief reasoning (2-3 sentences) justifying your scores.

Be calibrated — a middle schooler cannot run molecular dynamics, and a question \
that merely confirms well-established theory has low scientific value."""


async def score_feasibility(
    candidates: TheoryQuestions,
    phenomenon: PhenomenonResult,
    population: str = "high_school",
) -> list[ScoredTheoryQuestion]:
    """Stage 6: Score and rank theory questions by feasibility.

    Args:
        candidates: Output from Stage 5.
        phenomenon: Output from Stage 1 (for context).
        population: Student level for calibration.

    Returns:
        Top 3-5 scored questions, ranked by overall_score.
    """
    logger.info(
        "Theory Stage 6: Scoring {} questions", len(candidates.questions),
    )

    # Score all questions in parallel
    tasks = [
        _score_single(q, population)
        for q in candidates.questions
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Pair questions with scores, skip failures
    scored: list[ScoredTheoryQuestion] = []
    for question, result in zip(candidates.questions, results, strict=False):
        if isinstance(result, BaseException):
            logger.warning("Stage 6: Scoring failed for '{}': {}", question.question[:60], result)
            continue
        scored.append(ScoredTheoryQuestion(
            question=question.question,
            question_type=question.question_type,
            framework_connection=question.framework_connection,
            literature_support=question.literature_support,
            why_interesting=question.why_interesting,
            feasibility=result,
        ))

    # Sort by overall score descending, return top 5
    scored.sort(key=lambda q: q.feasibility.overall_score, reverse=True)
    top = scored[:5]

    logger.info(
        "Stage 6 complete: {} scored, returning top {} (scores: {})",
        len(scored), len(top),
        ", ".join(str(q.feasibility.overall_score) for q in top),
    )

    return top


async def _score_single(
    question: TheoryQuestion,
    population: str,
) -> TheoryFeasibility:
    """Score a single question."""
    async with _LLM_SEMAPHORE:
        return await structured_call(
            TheoryFeasibility,
            [
                SystemMessage(content=FEASIBILITY_PROMPT.format(
                    population=population,
                    question=question.question,
                    question_type=question.question_type,
                    framework_connection=question.framework_connection,
                )),
                HumanMessage(content="Assess feasibility."),
            ],
            thinking="high",
            temperature=0.0,
        )
