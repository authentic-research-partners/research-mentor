"""Theory Question Workshop orchestrator — 6-stage pipeline.

Produces 3-5 ranked theory-grounded research questions:
1. Phenomenon Mapper — search literature, identify surprising patterns
2. Abductive Reasoner — generate and compare candidate explanations
3. Framework Builder — construct a theoretical framework
4. Consilience Tester — test framework across domains (parallel)
5. Question Generator — produce theory-grounded questions
6. Feasibility Scorer — score and rank questions (parallel)

Batch generation uses asyncio.gather (same pattern as Modeling/Phenomenon).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from loguru import logger

from research_mentor.question_workshop.citation_utils import format_papers_for_prompt
from research_mentor.question_workshop.question_selector import (
    ScoredCandidate,
    select_questions,
)
from research_mentor.question_workshop.theory.pipeline.abductive_reasoner import (
    generate_explanations,
)
from research_mentor.question_workshop.theory.pipeline.consilience_tester import (
    test_consilience,
)
from research_mentor.question_workshop.theory.pipeline.feasibility_scorer import (
    score_feasibility,
)
from research_mentor.question_workshop.theory.pipeline.framework_builder import (
    build_framework,
)
from research_mentor.question_workshop.theory.pipeline.phenomenon_mapper import (
    map_phenomenon,
)
from research_mentor.question_workshop.theory.pipeline.question_generator import (
    generate_questions,
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


async def generate_theory_questions(
    observations_text: str,
    domain: str,
    project_id: str | None = None,
    student_demographics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate theory-grounded research questions through the 6-stage pipeline.

    Args:
        observations_text: Student's observations/puzzles (free text).
        domain: Research domain (e.g. "crystal formation").
        project_id: Optional project to link results to.
        student_demographics: Optional student profile for adaptation.

    Returns:
        Complete result dict with questions, framework, consilience, and metadata.
    """
    population = _derive_population(student_demographics)

    logger.info(
        "Question Workshop [theory]: generating domain='{}' population={}",
        domain[:80], population,
    )

    from research_mentor.agent.prompts.shared import build_student_profile_context

    profile_text = build_student_profile_context(student_demographics or {})

    # Stage 1: Phenomenon Mapping
    logger.info("Stage 1/6: Phenomenon Mapping")
    phenomenon = await map_phenomenon(
        observations_text=observations_text,
        domain=domain,
        population=population,
        student_profile_text=profile_text,
    )

    if not phenomenon.papers and not phenomenon.key_observations:
        return {
            "id": uuid.uuid4().hex,
            "project_id": project_id,
            "success": False,
            "rejection_reason": "No papers or observations found for this topic.",
            "domain": domain,
        }

    # Stage 2: Abductive Reasoning
    logger.info("Stage 2/6: Abductive Reasoning")
    abduction = await generate_explanations(
        phenomenon=phenomenon,
        observations_text=observations_text,
        population=population,
        student_profile_text=profile_text,
    )

    # Stage 3: Framework Building
    logger.info("Stage 3/6: Framework Building")
    framework_result = await build_framework(
        abduction=abduction,
        phenomenon=phenomenon,
        population=population,
        student_profile_text=profile_text,
    )

    # Stage 4: Consilience Testing
    logger.info("Stage 4/6: Consilience Testing")
    consilience = await test_consilience(
        framework_result=framework_result,
        phenomenon=phenomenon,
        domain=domain,
        population=population,
        student_profile_text=profile_text,
    )

    # Stage 5: Question Generation
    logger.info("Stage 5/6: Question Generation")
    candidates = await generate_questions(
        framework_result=framework_result,
        consilience=consilience,
        phenomenon=phenomenon,
        population=population,
        student_profile_text=profile_text,
    )

    # Stage 6: Feasibility Scoring
    logger.info("Stage 6/6: Feasibility Scoring")
    scored = await score_feasibility(
        candidates=candidates,
        phenomenon=phenomenon,
        population=population,
    )

    # Format paper references
    paper_refs = format_papers_for_prompt(phenomenon.papers, limit=5) if phenomenon.papers else ""

    # Consilience summary
    consilience_summary = (
        f"Tested {len(consilience.domains_tested)} domains: "
        f"{consilience.successes} strong, {consilience.partial} partial, "
        f"{consilience.failures} weak/contradicts. "
        f"Framework strength: {consilience.framework_strength}."
    )

    # Select: quality filter + similarity dedup + max 3
    all_question_dicts = [
        {
            "question": q.question,
            "question_type": q.question_type,
            "framework_connection": q.framework_connection,
            "literature_support": q.literature_support,
            "why_interesting": q.why_interesting,
            "feasibility": {
                "methodological_feasibility": q.feasibility.methodological_feasibility,
                "data_availability": q.feasibility.data_availability,
                "student_accessibility": q.feasibility.student_accessibility,
                "scientific_value": q.feasibility.scientific_value,
                "framework_relevance": q.feasibility.framework_relevance,
                "overall_score": q.feasibility.overall_score,
                "reasoning": q.feasibility.reasoning,
            },
        }
        for q in scored
    ]

    scored_candidates = [
        ScoredCandidate(
            question_text=q.question,
            # Normalize 1-10 → 0-1
            overall_score=q.feasibility.overall_score / 10.0,
            data=qd,
        )
        for q, qd in zip(scored, all_question_dicts, strict=False)
    ]
    selected = select_questions(
        scored_candidates,
        quality_threshold=0.4,
        similarity_threshold=0.85,
        max_results=3,
    )

    # Overall quality from selected questions
    if selected:
        avg_score = round(
            sum(q["feasibility"]["overall_score"] for q in selected) / len(selected), 1,
        )
    else:
        avg_score = 0.0

    framework = framework_result.framework
    record: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "project_id": project_id,
        "success": True,
        "domain": domain,
        "observations_text": observations_text,
        "overall_quality": avg_score,
        "questions": selected,
        "framework": {
            "framework_type": framework.framework_type,
            "description": framework.description,
            "categories": framework.categories,
            "relationships": framework.relationships,
            "predictions": framework.predictions,
            "historical_analogy": framework.historical_analogy,
        },
        "consilience": {
            "domains_tested": [
                {
                    "domain": t.domain,
                    "fit": t.fit,
                    "what_it_explains": t.what_it_explains,
                    "what_it_fails": t.what_it_fails,
                }
                for t in consilience.domains_tested
            ],
            "framework_strength": consilience.framework_strength,
            "revision_notes": consilience.revision_notes,
        },
        "consilience_summary": consilience_summary,
        "abduction": {
            "best_explanation": abduction.comparison.best_explanation,
            "runner_up": abduction.comparison.runner_up,
            "key_weakness": abduction.comparison.key_weakness,
            "comparison_reasoning": abduction.comparison.comparison_reasoning,
            "candidates_count": len(abduction.candidates),
            "candidates": [
                {
                    "explanation": c.explanation,
                    "supporting_evidence": c.supporting_evidence,
                    "fails_to_explain": c.fails_to_explain,
                    "source_domain": c.source_domain,
                }
                for c in abduction.candidates
            ],
        },
        "phenomenon": {
            "domain_summary": phenomenon.domain_summary,
            "surprising_findings": phenomenon.surprising_findings,
            "unexplained_phenomena": phenomenon.unexplained_phenomena,
            "cross_domain_connections": phenomenon.cross_domain_connections,
        },
        "papers_found": len(phenomenon.papers),
        "paper_references": paper_refs,
        "metadata": json.dumps({
            "population": population,
        }),
    }

    logger.info(
        "Question Workshop [theory] complete: {} questions selected, quality={}, framework={}",
        len(selected), avg_score, framework.framework_type,
    )

    return record


async def generate_best_of_n(
    observations_text: str,
    domain: str,
    project_id: str | None = None,
    candidates: int = 3,
    student_demographics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate N theory pipelines in parallel, merge and re-select.

    Each run already applies select_questions internally.  This merges
    the selected questions from all runs and re-applies selection.
    """
    logger.info(
        "Question Workshop [theory]: generating best of {} (domain='{}')",
        candidates, domain[:60],
    )

    tasks = [
        generate_theory_questions(
            observations_text=observations_text,
            domain=domain,
            project_id=project_id,
            student_demographics=student_demographics,
        )
        for _ in range(candidates)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    good: list[dict[str, Any]] = [
        r for r in results
        if isinstance(r, dict) and r.get("success", False)
    ]

    if not good:
        for r in results:
            if isinstance(r, dict):
                return r
        for r in results:
            if isinstance(r, BaseException):
                raise r
        raise RuntimeError("All candidates failed")

    # Merge questions from all runs, re-apply selection
    all_questions = [
        ScoredCandidate(
            question_text=q.get("question", ""),
            overall_score=q.get("feasibility", {}).get("overall_score", 0) / 10.0,
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
    if selected:
        best["overall_quality"] = round(
            sum(q["feasibility"]["overall_score"] for q in selected) / len(selected), 1,
        )

    logger.info(
        "Question Workshop [theory] best of {}: {} selected from {} runs",
        candidates, len(selected), len(good),
    )

    return best
