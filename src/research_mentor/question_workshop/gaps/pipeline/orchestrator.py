"""Gaps Pipeline orchestrator — 4-stage sequential pipeline.

Produces fundamental research questions by systematically analyzing
literature gaps, contradictions, and untested assumptions:
1. Landscape Explorer — domain guard, search, pattern extraction
2. Gap Analyzer — apply 3-5 strategic modes in parallel
3. Question Generator — synthesize gaps into candidate questions
4. FUNDAMENTAL Scorer — evaluate and rank on 11 criteria

~60-90 seconds per run.  Batch generation uses asyncio.gather.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from research_mentor.question_workshop.citation_utils import format_papers_for_prompt
from research_mentor.question_workshop.gaps.pipeline.fundamental_scorer import (
    score_fundamental,
)
from research_mentor.question_workshop.gaps.pipeline.gap_analyzer import analyze_gaps
from research_mentor.question_workshop.gaps.pipeline.landscape_explorer import (
    explore_landscape,
)
from research_mentor.question_workshop.gaps.pipeline.question_generator import (
    generate_questions,
)
from research_mentor.question_workshop.question_selector import (
    ScoredCandidate,
    select_questions,
)


def _derive_population(demographics: dict[str, Any] | None) -> str:
    """Derive population/level from student demographics."""
    if not demographics:
        return "high_school"
    age = demographics.get("age")
    edu = demographics.get("educationLevel", "")
    if isinstance(age, int):
        if age < 14:
            return "middle_school"
        if age < 18:
            return "high_school"
        if age < 23:
            return "university"
        return "adult"
    edu_lower = str(edu).lower()
    if "middle" in edu_lower:
        return "middle_school"
    if "college" in edu_lower or "university" in edu_lower or "undergrad" in edu_lower:
        return "university"
    if "graduate" in edu_lower or "phd" in edu_lower or "professional" in edu_lower:
        return "adult"
    return "high_school"


def _ensure_mode_diversity(
    selected: list[dict[str, Any]],
    all_candidates: list[ScoredCandidate],
    applied_modes: set[str],
) -> list[dict[str, Any]]:
    """Swap lowest-scored selections with best questions from missing modes.

    If the top-3 selection dropped an entire strategy mode, replace the
    lowest-scored selected question with the best question from that mode.
    Only swaps if the replacement meets a minimum quality bar (0.4).
    """
    if len(selected) < 2 or not applied_modes:
        return selected

    covered_modes = {q.get("mode_origin") for q in selected}
    missing_modes = applied_modes - covered_modes

    if not missing_modes:
        return selected

    # Build lookup: mode → best candidate (by score, not already selected)
    selected_texts = {q.get("question", "") for q in selected}
    mode_best: dict[str, ScoredCandidate] = {}
    for candidate in sorted(all_candidates, key=lambda c: c.overall_score, reverse=True):
        mode = candidate.data.get("mode_origin", "")
        if mode in missing_modes and mode not in mode_best:
            if candidate.question_text not in selected_texts and candidate.overall_score >= 0.4:
                mode_best[mode] = candidate

    if not mode_best:
        return selected

    # Sort selected by score ascending so we replace weakest first
    selected_with_scores = [
        (q, q.get("scores", {}).get("overall_weighted", 0.0))
        for q in selected
    ]
    selected_with_scores.sort(key=lambda x: x[1])

    result = list(selected)
    replacements_made = 0
    for mode, replacement in mode_best.items():
        if replacements_made >= len(selected) - 1:
            break  # Keep at least one original selection
        # Replace the lowest-scored selected question
        weakest = selected_with_scores[replacements_made][0]
        idx = next(i for i, q in enumerate(result) if q is weakest)
        logger.info(
            "Mode diversity: replacing '{}' (mode={}) with '{}' (mode={})",
            weakest.get("question", "")[:60],
            weakest.get("mode_origin"),
            replacement.question_text[:60],
            mode,
        )
        result[idx] = replacement.data
        replacements_made += 1

    return result


async def generate_gaps_questions(
    field_topic: str,
    population: str = "high_school",
    interest_area: str | None = None,
    config: Any | None = None,
    student_demographics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate fundamental research questions through the 4-stage pipeline.

    Args:
        field_topic: Scientific field/topic (free text).
        population: Student level for depth calibration.
        interest_area: Optional focus within the field.
        config: Optional GapsConfig override.
        student_demographics: Optional student profile (population derived if provided).

    Returns:
        Result dict with scored questions, metadata, and paper references.

    Raises:
        ValueError: If domain is social_science, refused, or off_topic.
    """
    if student_demographics:
        population = _derive_population(student_demographics)

    logger.info(
        "Gaps Pipeline: generating field='{}' interest='{}' population={}",
        field_topic[:80], interest_area, population,
    )

    # Stage 1: Landscape Exploration
    logger.info("Stage 1/4: Landscape Exploration")
    landscape = await explore_landscape(field_topic, interest_area, population, config)

    if not landscape.papers:
        return {
            "success": False,
            "rejection_reason": f"No papers found for '{field_topic}'",
            "stage_reached": "landscape_exploration",
        }

    # Stage 2: Gap Analysis
    logger.info("Stage 2/4: Gap Analysis")
    gaps = await analyze_gaps(landscape, population, config)

    if not gaps.all_candidate_questions:
        return {
            "success": False,
            "rejection_reason": "No research gaps found in the literature",
            "stage_reached": "gap_analysis",
        }

    # Stage 3: Question Generation
    logger.info("Stage 3/4: Question Generation")
    candidates = await generate_questions(landscape, gaps, population, config)

    if not candidates.questions:
        return {
            "success": False,
            "rejection_reason": "Could not generate candidate questions from gaps",
            "stage_reached": "question_generation",
        }

    # Stage 4: FUNDAMENTAL Scoring
    logger.info("Stage 4/4: FUNDAMENTAL Scoring")
    scored = await score_fundamental(candidates, landscape, config)

    # Select: quality filter + similarity dedup + max 3
    scored_candidates = [
        ScoredCandidate(
            question_text=q.question,
            overall_score=q.scores.get("overall_weighted", 0.0),
            data=q.model_dump(),
        )
        for q in scored.questions
    ]
    # Gaps uses 0-1 scale, so threshold 0.6 = reasonable quality bar
    selected = select_questions(
        scored_candidates,
        quality_threshold=0.6,
        similarity_threshold=0.85,
        max_results=3,
    )

    # Ensure mode diversity: replace lowest-scored selection with best
    # question from a missing mode, so each applied strategy is represented.
    applied_modes = {s.strategy_name for s in gaps.strategies_applied}
    selected = _ensure_mode_diversity(selected, scored_candidates, applied_modes)

    # Format paper references
    paper_refs = format_papers_for_prompt(landscape.papers, limit=10)

    result: dict[str, Any] = {
        "success": True,
        "field_topic": field_topic,
        "interest_area": interest_area,
        "population": population,
        "questions": selected,
        "field_maturity": landscape.field_maturity,
        "domain_summary": landscape.domain_summary,
        "strategies_applied": [
            s.strategy_name for s in gaps.strategies_applied
        ],
        "papers_found": len(landscape.papers),
        "paper_references": paper_refs,
        "landscape": {
            "consensus": landscape.consensus,
            "debates": landscape.debates,
            "gaps": landscape.gaps,
            "frontiers": landscape.frontiers,
        },
    }

    logger.info(
        "Gaps Pipeline complete: {} questions selected",
        len(selected),
    )

    return result


async def generate_best_of_n(
    field_topic: str,
    population: str = "high_school",
    interest_area: str | None = None,
    candidates: int = 3,
    config: Any | None = None,
    student_demographics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate N pipeline runs in parallel, merge and re-select.

    Each run already applies select_questions internally.  This merges
    the selected questions from all runs and re-applies selection for
    the final quality/similarity filter.
    """
    if student_demographics:
        population = _derive_population(student_demographics)

    logger.info(
        "Gaps Pipeline: generating best of {} (field='{}')",
        candidates, field_topic[:60],
    )

    tasks = [
        generate_gaps_questions(
            field_topic=field_topic,
            population=population,
            interest_area=interest_area,
            config=config,
        )
        for _ in range(candidates)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    good: list[dict[str, Any]] = [
        r for r in results
        if isinstance(r, dict) and r.get("success")
    ]

    if not good:
        for r in results:
            if isinstance(r, BaseException):
                raise r
        for r in results:
            if isinstance(r, dict):
                return r
        raise RuntimeError("All candidates failed")

    # Merge questions from all runs, re-apply selection
    all_questions = [
        ScoredCandidate(
            question_text=q.get("question", ""),
            overall_score=q.get("scores", {}).get("overall_weighted", 0.0),
            data=q,
        )
        for result in good
        for q in result.get("questions", [])
    ]

    selected = select_questions(
        all_questions,
        quality_threshold=0.6,
        similarity_threshold=0.85,
        max_results=3,
    )

    best = good[0].copy()
    best["questions"] = selected

    logger.info(
        "Gaps Pipeline best of {}: {} selected from {} runs",
        candidates, len(selected), len(good),
    )

    return best
