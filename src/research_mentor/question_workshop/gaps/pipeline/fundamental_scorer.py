"""Stage 4: FUNDAMENTAL Scorer — evaluate and rank candidate questions.

Scores each candidate question on 11 FUNDAMENTAL criteria in parallel,
applies weighted ranking (Deep, Approachable, Literature-based weighted 1.5x),
and returns the top 3-5 questions passing threshold.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.gaps.pipeline.schemas import (
    CandidateQuestion,
    CandidateQuestions,
    LandscapeResult,
    ScoredQuestion,
    ScoredQuestions,
)
from research_mentor.question_workshop.gaps.schemas import FUNDAMENTALScore
from research_mentor.question_workshop.gaps.utils.prompts import (
    PHASE_3_FUNDAMENTAL_EVAL,
)

_LLM_SEMAPHORE = asyncio.Semaphore(20)

# Criteria with 1.5x weight in the overall score
_WEIGHTED_CRITERIA = frozenset({"deep", "approachable", "literature_based"})

_ALL_CRITERIA = [
    "falsifiable", "unaddressed", "necessary", "deep", "approachable",
    "meaningful", "exciting", "narrowed", "theoretically_grounded",
    "articulated", "literature_based",
]


def compute_weighted_score(score: FUNDAMENTALScore) -> float:
    """Compute weighted overall score (Deep, Approachable, Literature-based at 1.5x)."""
    total = 0.0
    weight_sum = 0.0
    for criterion in _ALL_CRITERIA:
        value = getattr(score, criterion, 0.0)
        weight = 1.5 if criterion in _WEIGHTED_CRITERIA else 1.0
        total += value * weight
        weight_sum += weight
    return round(total / weight_sum, 3) if weight_sum > 0 else 0.0


async def score_fundamental(
    candidates: CandidateQuestions,
    landscape: LandscapeResult,
    config: Any | None = None,
) -> ScoredQuestions:
    """Score all candidate questions on FUNDAMENTAL criteria in parallel.

    Returns top 3-5 questions that pass the threshold (0.8).
    If fewer than 3 pass, returns top 3 regardless with notes.
    """
    logger.info(
        "Gaps Pipeline Stage 4: Scoring {} candidate questions",
        len(candidates.questions),
    )

    # Build literature context once
    lit_context = _build_literature_context(landscape)

    # Score all questions in parallel
    tasks = [
        _score_one(candidate, lit_context)
        for candidate in candidates.questions
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect successful scores
    scored: list[ScoredQuestion] = []
    for candidate, result in zip(candidates.questions, results):
        if isinstance(result, BaseException):
            logger.warning(
                "Scoring failed for '{}': {}",
                candidate.question[:60], result,
            )
            continue
        scored.append(result)

    logger.info(
        "Stage 4 complete: {} scored (selection handled by orchestrator)",
        len(scored),
    )

    return ScoredQuestions(questions=scored)


async def _score_one(
    candidate: CandidateQuestion,
    literature_context: str,
) -> ScoredQuestion:
    """Score a single candidate question on FUNDAMENTAL criteria."""
    async with _LLM_SEMAPHORE:
        prompt = PHASE_3_FUNDAMENTAL_EVAL.format(
            question=candidate.question,
            literature_context=literature_context,
            user_reasoning=candidate.why_fundamental or "Not provided",
        )

        score = await structured_call(
            FUNDAMENTALScore,
            [
                SystemMessage(content=prompt),
                HumanMessage(content="Evaluate this research question."),
            ],
            thinking="high",
            temperature=0.0,
        )

        # Compute weighted score (don't rely on LLM's overall)
        weighted = compute_weighted_score(score)

        scores_dict: dict[str, Any] = {
            criterion: getattr(score, criterion) for criterion in _ALL_CRITERIA
        }
        scores_dict["overall_llm"] = score.overall
        scores_dict["overall_weighted"] = weighted
        scores_dict["reasoning"] = score.reasoning
        scores_dict["refinement_suggestions"] = score.refinement_suggestions

        return ScoredQuestion(
            question=candidate.question,
            gap_addressed=candidate.gap_addressed,
            mode_origin=candidate.mode_origin,
            literature_support=candidate.literature_support,
            why_fundamental=candidate.why_fundamental,
            scores=scores_dict,
        )


def _build_literature_context(landscape: LandscapeResult) -> str:
    """Build literature context for scoring from landscape data."""
    parts = []
    if landscape.domain_summary:
        parts.append(f"Domain: {landscape.domain_summary}")
    if landscape.consensus:
        parts.append(f"Consensus: {'; '.join(landscape.consensus[:3])}")
    if landscape.debates:
        parts.append(f"Debates: {'; '.join(landscape.debates[:3])}")
    if landscape.gaps:
        parts.append(f"Known gaps: {'; '.join(landscape.gaps[:3])}")
    if landscape.frontiers:
        parts.append(f"Frontiers: {'; '.join(landscape.frontiers[:2])}")

    # Add top paper references
    paper_refs = []
    for p in landscape.papers[:5]:
        title = p.get("title", "")
        year = p.get("year", "")
        if title:
            paper_refs.append(f"{title} ({year})")
    if paper_refs:
        parts.append(f"Key papers: {'; '.join(paper_refs)}")

    return "\n".join(parts) if parts else "No literature context available"
