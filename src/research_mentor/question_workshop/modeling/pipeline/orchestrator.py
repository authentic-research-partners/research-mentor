"""Modeling Question Workshop orchestrator — multi-approach pipeline.

Produces research questions investigable through computational modeling:
1. System Explorer — search published models, map the landscape
2. Approach Filter — which approaches fit this system? (code + LLM)
3. Question Generator — produce candidates per viable approach (parallel)
4. Feasibility Scorer — assess each candidate
5. Selection — pick best per approach, return 1-N results

When student picks "Any" or a group, generates across multiple approaches.
When student picks a specific approach, generates 3 candidates of that one.
All candidates run in parallel via asyncio.gather (vLLM batches internally).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.citation_utils import format_papers_for_prompt
from research_mentor.question_workshop.modeling.pipeline.feasibility_scorer import (
    score_feasibility,
)
from research_mentor.question_workshop.modeling.pipeline.question_generator import (
    generate_question,
)
from research_mentor.question_workshop.modeling.pipeline.schemas import (
    APPROACH_GROUPS,
    MODELING_APPROACHES,
    ApproachFitnessSet,
)
from research_mentor.question_workshop.modeling.pipeline.system_explorer import (
    explore_system,
)
from research_mentor.question_workshop.question_selector import (
    ScoredCandidate,
    select_questions,
)

_PIPELINE_SEMAPHORE = asyncio.Semaphore(20)


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
    if any(k in edu_lower for k in ("college", "university", "undergrad")):
        return "university"
    if any(k in edu_lower for k in ("graduate", "phd", "professional")):
        return "adult"
    return "high_school"


def _resolve_approaches(approach: str) -> list[str]:
    """Resolve an approach input to a list of specific approach keys.

    - "any" → all 12 approaches
    - group key (e.g., "agents") → approaches in that group
    - specific key (e.g., "agent_based") → just that one
    """
    if approach == "any":
        return list(MODELING_APPROACHES.keys())
    if approach in APPROACH_GROUPS:
        approaches = APPROACH_GROUPS[approach]["approaches"]
        return list(approaches) if isinstance(approaches, list) else [str(approaches)]
    if approach in MODELING_APPROACHES:
        return [approach]
    raise ValueError(
        f"Invalid approach '{approach}'. Must be one of: "
        f"{', '.join(sorted(MODELING_APPROACHES))} or "
        f"{', '.join(sorted(APPROACH_GROUPS))} or 'any'"
    )


async def _filter_approaches(
    system_topic: str,
    approaches: list[str],
) -> list[str]:
    """Filter approaches to those that make sense for the system.

    If only 1 approach, skip filtering (student chose specifically).
    """
    if len(approaches) <= 1:
        return approaches

    approach_descriptions = "\n".join(
        f"- {key}: {MODELING_APPROACHES[key]['label']} — "
        f"{MODELING_APPROACHES[key]['description'][:100]}"
        for key in approaches
        if key in MODELING_APPROACHES
    )

    prompt = (
        f"Which of these modeling approaches can meaningfully model "
        f"'{system_topic}'? Mark each as fits=true or fits=false.\n\n"
        f"Approaches:\n{approach_descriptions}\n\n"
        f"Be strict: DFT cannot model traffic, agent-based cannot model "
        f"electronic structure. Only mark fits=true if the approach is a "
        f"natural fit for this specific system."
    )

    result = await structured_call(
        ApproachFitnessSet,
        [
            SystemMessage(content=prompt),
            HumanMessage(content="Filter approaches."),
        ],
        thinking="medium",
        temperature=0.0,
    )

    viable = [
        r.approach for r in result.results
        if r.fits and r.approach in MODELING_APPROACHES
    ]

    if not viable:
        # If filter rejected everything, keep the most general one
        logger.warning(
            "Approach filter rejected all {} approaches for '{}', "
            "falling back to numerical_simulation",
            len(approaches), system_topic[:60],
        )
        return ["numerical_simulation"]

    logger.info(
        "Approach filter: {}/{} viable for '{}'",
        len(viable), len(approaches), system_topic[:60],
    )
    return viable


async def _generate_one(
    system_topic: str,
    approach: str,
    population: str,
    landscape: Any,
    profile_text: str,
) -> dict[str, Any] | None:
    """Generate one question + score it. Returns result dict or None on failure."""
    async with _PIPELINE_SEMAPHORE:
        return await _generate_one_inner(
            system_topic, approach, population, landscape, profile_text,
        )


async def _generate_one_inner(
    system_topic: str,
    approach: str,
    population: str,
    landscape: Any,
    profile_text: str,
) -> dict[str, Any] | None:
    """Generate one question + score it (under semaphore)."""
    try:
        question = await generate_question(
            system_topic=system_topic,
            approach=approach,
            population=population,
            landscape=landscape,
            student_profile_text=profile_text,
        )
        feasibility = await score_feasibility(
            question=question,
            population=population,
        )
        scores = [
            feasibility.computational_feasibility,
            feasibility.data_availability,
            feasibility.student_accessibility,
            feasibility.scientific_value,
        ]
        return {
            "question": question.question,
            "approach": question.approach,
            "approach_label": MODELING_APPROACHES.get(
                question.approach, {},
            ).get("label", question.approach),
            "key_simplification": question.key_simplification,
            "why_interesting": question.why_interesting,
            "suggested_tools": question.suggested_tools,
            "overall_quality": round(sum(scores) / len(scores), 1),
            "feasibility": {
                "computational": feasibility.computational_feasibility,
                "data_availability": feasibility.data_availability,
                "student_accessibility": feasibility.student_accessibility,
                "scientific_value": feasibility.scientific_value,
                "overall": feasibility.overall_score,
            },
        }
    except Exception:
        logger.exception(
            "Failed to generate question for approach={}", approach,
        )
        return None


async def generate_modeling_questions(
    system_topic: str,
    approach: str = "any",
    project_id: str | None = None,
    student_demographics: dict[str, Any] | None = None,
    candidates_per_approach: int = 3,
) -> dict[str, Any]:
    """Generate modeling research questions — one best per viable approach.

    Args:
        system_topic: The system or phenomenon to model.
        approach: "any", a group key, or a specific approach key.
        project_id: Optional project link.
        student_demographics: Student profile (population derived from this).
        candidates_per_approach: How many candidates to generate per approach.

    Returns:
        Result dict with 1-N questions (one best per viable approach),
        landscape, and metadata.
    """
    population = _derive_population(student_demographics)
    approaches = _resolve_approaches(approach)

    logger.info(
        "Question Workshop [modeling]: system='{}' approach={} "
        "resolved={} population={}",
        system_topic[:60], approach, len(approaches), population,
    )

    from research_mentor.agent.prompts.shared import build_student_profile_context

    profile_text = build_student_profile_context(student_demographics or {})

    # Stage 1: System Exploration (one search, shared across all approaches)
    logger.info("Stage 1: System Exploration")
    landscape, papers = await explore_system(
        system_topic=system_topic,
        approach=approaches[0] if len(approaches) == 1 else "any",
        student_profile_text=profile_text,
    )

    # Stage 2: Filter approaches (skip if only one)
    if len(approaches) > 1:
        logger.info("Stage 2: Filtering {} approaches", len(approaches))
        viable = await _filter_approaches(system_topic, approaches)
    else:
        viable = approaches

    # Stage 3+4: Generate candidates per approach (all in parallel)
    logger.info(
        "Stage 3+4: Generating {} candidates x {} approaches = {} total",
        candidates_per_approach, len(viable),
        candidates_per_approach * len(viable),
    )

    tasks = [
        _generate_one(system_topic, app, population, landscape, profile_text)
        for app in viable
        for _ in range(candidates_per_approach)
    ]
    all_results = await asyncio.gather(*tasks)

    # Group by approach, pick best per approach
    by_approach: dict[str, list[dict[str, Any]]] = {}
    for r in all_results:
        if r is not None:
            app = r["approach"]
            if app not in by_approach:
                by_approach[app] = []
            by_approach[app].append(r)

    best_per_approach: list[dict[str, Any]] = []
    for _app, candidates_list in by_approach.items():
        best = max(candidates_list, key=lambda c: float(c["overall_quality"]))
        best_per_approach.append(best)

    # Final selection: quality filter + similarity dedup + max 3
    scored_candidates = [
        ScoredCandidate(
            question_text=q["question"],
            # Normalize 1-10 → 0-1
            overall_score=q["overall_quality"] / 10.0,
            data=q,
        )
        for q in best_per_approach
    ]
    questions = select_questions(
        scored_candidates,
        quality_threshold=0.6,
        similarity_threshold=0.85,
        max_results=3,
    )

    paper_refs = format_papers_for_prompt(papers, limit=5) if papers else ""

    record: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "project_id": project_id,
        "system_topic": system_topic,
        "approach_input": approach,
        "questions": questions,
        "landscape": {
            "published_models": landscape.published_models,
            "key_variables": landscape.key_variables,
            "simplification_tradeoffs": landscape.simplification_tradeoffs,
            "available_data": landscape.available_data,
            "domain_summary": landscape.domain_summary,
        },
        "papers_found": len(papers),
        "paper_references": paper_refs,
        "metadata": json.dumps({
            "population": population,
            "approaches_evaluated": len(approaches),
            "approaches_viable": len(viable),
            "candidates_generated": len([r for r in all_results if r]),
        }),
    }

    logger.info(
        "Question Workshop [modeling] complete: {} questions from {} "
        "viable approaches (top: '{}' quality={})",
        len(questions), len(viable),
        questions[0]["question"][:60] if questions else "none",
        questions[0]["overall_quality"] if questions else 0,
    )

    return record
