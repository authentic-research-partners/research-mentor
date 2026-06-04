"""Phenomenon Question Workshop orchestrator — 4-stage sequential pipeline.

Chains autonomous stages to generate research questions (IYPT-style format):
1. Domain Explorer — identify interesting phenomena
2. Problem Generator — formulate problem (title, description, investigation)
3. Feasibility Validator — score feasibility on 5 dimensions
4. Creative Refiner — polish for engagement + novelty assessment

~30-40 seconds per problem.  Batch generation uses asyncio.gather.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from loguru import logger

from research_mentor.db import crud
from research_mentor.question_workshop.knowledge_base import FIELDS, PROBLEM_TYPES
from research_mentor.question_workshop.phenomenon.stages.creative_refiner import (
    refine_creatively,
)
from research_mentor.question_workshop.phenomenon.stages.domain_explorer import (
    explore_domain,
)
from research_mentor.question_workshop.phenomenon.stages.feasibility_validator import (
    validate_feasibility,
)
from research_mentor.question_workshop.phenomenon.stages.problem_generator import (
    generate_problem,
)
from research_mentor.question_workshop.question_selector import (
    ScoredCandidate,
    select_questions,
)


async def generate_research_question(
    field: str,
    problem_type: str = "experimental",
    domain: str | None = None,
    user_suggestion: str | None = None,
    project_id: str | None = None,
    student_demographics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a single research question through the 4-stage pipeline.

    Args:
        field: Scientific field (physics, chemistry, biology).
        problem_type: experimental, simulation, data_analysis, or theoretical.
        domain: Optional domain within field (e.g. fluid_dynamics).
        user_suggestion: Optional seed idea from user.
        project_id: Optional project to link generated problem to.
        student_demographics: Optional student profile for adaptation.

    Returns:
        Complete generated problem dict (matches generated_problems table columns).
    """
    if field not in FIELDS:
        raise ValueError(f"Invalid field '{field}'. Must be one of: {', '.join(sorted(FIELDS))}")
    if problem_type not in PROBLEM_TYPES:
        raise ValueError(
            f"Invalid problem_type '{problem_type}'. "
            f"Must be one of: {', '.join(sorted(PROBLEM_TYPES))}"
        )

    # Check seed idea against domain scope if provided
    if user_suggestion and len(user_suggestion.strip()) >= 4:
        from research_mentor.question_workshop.domain import (
            get_domain_classification_prompt,
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            from research_mentor.llm import structured_call
            from research_mentor.question_workshop.gaps.schemas import (
                DomainClassification,
            )

            prompt = get_domain_classification_prompt().format(topic=user_suggestion)
            classification = await structured_call(
                DomainClassification,
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content="Classify this topic."),
                ],
                thinking="high",
                temperature=0.0,
            )
            if classification.domain_type == "refused":
                raise ValueError(
                    "That seed idea goes beyond typical science and involves "
                    "social, political, ethical, or clinical considerations "
                    "that we can\u2019t address with methodology alone. "
                    "Try a different seed idea focused on natural science."
                )
            if classification.domain_type == "social_science":
                from research_mentor.components_registry import get_redirect_phrase

                raise ValueError(
                    get_redirect_phrase("phenomenon", "hypothesis")
                    or "That seed idea sounds like a social science topic."
                )
            logger.debug(
                "Seed idea domain check passed: type={} confidence={}",
                classification.domain_type, classification.confidence,
            )
        except ValueError:
            raise
        except Exception:
            logger.debug("Seed idea domain check failed, proceeding anyway")

    logger.info(
        "Question Workshop [phenomenon]: generating field={} type={} domain={} suggestion={}",
        field, problem_type, domain, user_suggestion[:50] if user_suggestion else None,
    )

    from research_mentor.agent.prompts.shared import build_student_profile_context

    profile_text = build_student_profile_context(student_demographics or {})

    # Stage 1: Domain Exploration
    logger.info("Stage 1/4: Domain Exploration")
    exploration = await explore_domain(
        field=field,
        problem_type=problem_type,
        domain=domain,
        user_suggestion=user_suggestion,
        student_profile_text=profile_text,
    )

    # Stage 2: Problem Generation
    logger.info("Stage 2/4: Problem Generation")
    problem = await generate_problem(
        field=field,
        problem_type=problem_type,
        exploration_data=exploration,
        student_profile_text=profile_text,
    )

    # Stage 3: Feasibility Validation
    logger.info("Stage 3/4: Feasibility Validation")
    validation = await validate_feasibility(
        problem=problem,
        problem_type=problem_type,
    )

    # Stage 4: Creative Refinement (with novelty awareness)
    logger.info("Stage 4/4: Creative Refinement")
    existing_titles = await crud.get_generated_problem_titles(field, problem_type)
    refined = await refine_creatively(
        field=field,
        problem_type=problem_type,
        problem=problem,
        validation=validation,
        existing_titles=existing_titles,
        student_profile_text=profile_text,
    )

    # Assemble final result
    scores = [
        validation.get("material_accessibility", {}).get("score", 5),
        validation.get("experimental_difficulty", {}).get("score", 5),
        validation.get("measurability", {}).get("score", 5),
        validation.get("complexity", {}).get("score", 5),
        validation.get("time_feasibility", {}).get("score", 5),
        refined.get("engagement_score", 5),
    ]
    overall_quality = round(sum(scores) / len(scores), 1)

    record = {
        "id": uuid.uuid4().hex,
        "project_id": project_id,
        "workshop_type": "phenomenon",
        "field": field,
        "problem_type": problem_type,
        "domains": json.dumps(exploration.get("domains", [])),
        "user_suggestion": user_suggestion,
        "title": refined["title"],
        "description": refined["description"],
        "investigation": refined["investigation"],
        "core_concepts": json.dumps(problem.get("core_concepts", [])),
        "materials": json.dumps(exploration.get("materials", [])),
        "feasibility": validation.get("overall_feasibility", "MEDIUM"),
        "recommended": validation.get("recommended", "WITH_MODIFICATIONS"),
        "safety_level": validation.get("safety", {}).get("level", "REQUIRES_SUPERVISION"),
        "complexity_score": validation.get("complexity", {}).get("score"),
        "engagement_score": refined.get("engagement_score"),
        "overall_quality": overall_quality,
        "metadata": json.dumps({
            "novelty_assessment": refined.get("novelty_assessment", ""),
            "changes_made": refined.get("changes_made", []),
            "improvements_suggested": validation.get("improvements", []),
            "surprise_factor": exploration.get("surprise_factor"),
        }),
    }

    # Persist to DB
    stored = await crud.create_generated_problem(**record)

    logger.info(
        "Question Workshop [phenomenon] complete: title='{}' quality={} "
        "feasibility={} recommended={}",
        stored["title"], overall_quality,
        stored["feasibility"], stored["recommended"],
    )
    return stored


async def generate_best_of_n(
    field: str,
    problem_type: str = "experimental",
    domain: str | None = None,
    user_suggestion: str | None = None,
    project_id: str | None = None,
    candidates: int = 3,
    student_demographics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate N questions in parallel, return the best ones.

    All candidates run concurrently (vLLM batches them — same wall time as 1).
    Candidates are scored, deduplicated, and the best are returned.
    All candidates are stored in DB for history.

    Args:
        field: Scientific field.
        problem_type: Problem type.
        domain: Optional domain within field.
        user_suggestion: Optional seed idea.
        project_id: Optional project link.
        candidates: Number of parallel candidates (default 3).
        student_demographics: Optional student profile for adaptation.

    Returns:
        The best generated problem dict.
    """
    logger.info(
        "Question Workshop [phenomenon]: generating best of {} (field={} type={})",
        candidates, field, problem_type,
    )

    tasks = [
        generate_research_question(
            field=field,
            problem_type=problem_type,
            domain=domain,
            user_suggestion=user_suggestion,
            project_id=project_id,
            student_demographics=student_demographics,
        )
        for _ in range(candidates)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out failures
    good: list[dict[str, Any]] = [
        r for r in results if isinstance(r, dict)
    ]

    if not good:
        for r in results:
            if isinstance(r, BaseException):
                raise r
        raise RuntimeError("All candidates failed")

    # Select up to 3: quality filter + similarity dedup
    scored_candidates = [
        ScoredCandidate(
            question_text=p.get("title", "") + " " + p.get("description", ""),
            # Normalize 1-10 → 0-1
            overall_score=p.get("overall_quality", 0) / 10.0,
            data=p,
        )
        for p in good
    ]
    selected = select_questions(
        scored_candidates,
        quality_threshold=0.6,
        similarity_threshold=0.85,
        max_results=3,
    )

    logger.info(
        "Question Workshop [phenomenon] best of {}: {} selected from {} candidates",
        candidates, len(selected), len(good),
    )

    return selected
