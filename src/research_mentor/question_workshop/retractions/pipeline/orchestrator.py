"""Retractions pipeline orchestrator — 5-stage batch question generation.

Produces 3-5 ranked research questions from gaps re-opened by retracted
papers, grounded in failure analysis and field impact assessment.

Input: field (free text) OR specific paper (DOI/title) + population
Output: ScoredRetractionQuestions (ranked list with feasibility scores)

~60-120 seconds total. No checkpointing (re-run on failure).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from loguru import logger

from research_mentor.config import load_config
from research_mentor.db import crud
from research_mentor.question_workshop.question_selector import (
    ScoredCandidate,
    select_questions,
)
from research_mentor.question_workshop.retractions.pipeline.case_retriever import (
    retrieve_cases,
)
from research_mentor.question_workshop.retractions.pipeline.failure_analyzer import (
    analyze_failures,
)
from research_mentor.question_workshop.retractions.pipeline.feasibility_scorer import (
    score_feasibility,
)
from research_mentor.question_workshop.retractions.pipeline.impact_assessor import (
    assess_impact,
)
from research_mentor.question_workshop.retractions.pipeline.question_generator import (
    generate_questions,
)


def _retractions_config() -> Any:
    """Load Retractions config from TOML (cached at config level)."""
    return load_config().question_workshop.retractions


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


async def generate_retraction_questions(
    query: str,
    project_id: str | None = None,
    student_demographics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full 5-stage Retractions pipeline.

    Args:
        query: Field name (e.g., "psychology") OR DOI/title.
        project_id: Optional project to link results to.
        student_demographics: Optional student profile for adaptation.
            Population/level is derived from demographics automatically.

    Returns:
        Complete result dict (stored in generated_problems table),
        or a rejection dict if no cases are found.
    """
    cfg = _retractions_config()
    population = _derive_population(student_demographics)
    logger.info(
        "Retractions pipeline: starting (query='{}', population='{}')",
        query[:80], population,
    )

    # ===== Stage 1: Case Retrieval =====
    logger.info("Stage 1/5: Case Retrieval")
    try:
        cases = await retrieve_cases(
            query, population,
            max_results=cfg.pipeline_max_browse_results,
        )
    except ValueError as exc:
        logger.warning("Case retrieval failed: {}", exc)
        return {
            "success": False,
            "rejection_reason": str(exc),
            "stage_reached": "case_retrieval",
        }

    # ===== Stage 2: Failure Analysis =====
    logger.info("Stage 2/5: Failure Analysis")
    analyzed = await analyze_failures(cases, population)

    # ===== Stage 3: Impact Assessment =====
    logger.info("Stage 3/5: Impact Assessment")
    impacted = await assess_impact(analyzed)

    # ===== Stage 4: Question Generation =====
    logger.info("Stage 4/5: Question Generation")
    candidates = await generate_questions(impacted, population)

    # ===== Stage 5: Feasibility Scoring =====
    logger.info("Stage 5/5: Feasibility Scoring")
    scored = await score_feasibility(
        candidates, impacted, population,
        max_output=cfg.pipeline_max_output_questions,
    )

    # ===== Select: quality filter + similarity dedup + max 3 =====
    all_questions_data = [
        {
            "question": sq.question,
            "source_case_doi": sq.source_case.doi,
            "source_case_title": sq.source_case.title,
            "failure_type": sq.failure_type,
            "reopened_gap": sq.reopened_gap,
            "avoids_original_mistakes": sq.avoids_original_mistakes,
            "methodology_improvements": sq.methodology_improvements,
            "why_interesting": sq.why_interesting,
            "feasibility": {
                "data_availability": sq.feasibility.data_availability,
                "student_accessibility": sq.feasibility.student_accessibility,
                "scientific_value": sq.feasibility.scientific_value,
                "methodology_soundness": sq.feasibility.methodology_soundness,
                "integrity_value": sq.feasibility.integrity_value,
                "overall_score": sq.feasibility.overall_score,
                "reasoning": sq.feasibility.reasoning,
            },
        }
        for sq in scored.questions
    ]

    scored_candidates = [
        ScoredCandidate(
            question_text=sq.question,
            # Normalize 0-10 → 0-1
            overall_score=sq.feasibility.overall_score / 10.0,
            data=qd,
        )
        for sq, qd in zip(scored.questions, all_questions_data)
    ]
    questions_data = select_questions(
        scored_candidates,
        quality_threshold=0.6,
        similarity_threshold=0.85,
        max_results=3,
    )

    # Build DB record from the best selected question
    best_q = questions_data[0]
    overall_quality = round(best_q["feasibility"]["overall_score"], 1)

    record = {
        "id": uuid.uuid4().hex,
        "project_id": project_id,
        "field": query,
        "problem_type": "retraction_question",
        "workshop_type": "retractions",
        "domains": json.dumps([]),
        "user_suggestion": query,
        "title": f"Research Questions from Retracted Papers in {query.title()}",
        "description": best_q["reopened_gap"],
        "investigation": best_q["question"],
        "core_concepts": json.dumps([]),
        "materials": json.dumps([]),
        "feasibility": (
            "HIGH" if overall_quality >= 7
            else "MEDIUM" if overall_quality >= 4
            else "LOW"
        ),
        "recommended": "YES" if overall_quality >= 6 else "WITH_MODIFICATIONS",
        "safety_level": "SAFE",
        "complexity_score": None,
        "engagement_score": None,
        "overall_quality": overall_quality,
        "metadata": json.dumps({
            "questions": questions_data,
            "cases_found": len(cases.cases),
            "mode": cases.mode,
            "population": population,
            "questions_generated": len(candidates.questions),
            "questions_selected": len(questions_data),
        }),
    }

    stored = await crud.create_generated_problem(**record)

    logger.info(
        "Retractions pipeline complete: query='{}' questions={} top_score={}",
        query[:60], len(questions_data), overall_quality,
    )

    return {
        "success": True,
        **stored,
    }
