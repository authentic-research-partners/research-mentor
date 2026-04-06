"""Forge Workshop orchestrator — 4-stage sequential pipeline.

Chains autonomous stages to generate research tool ideas:
1. Domain Workflow Mapper — map research workflow + pain points
2. Bottleneck Identifier — find 3-5 capability gaps
3. Solution Designer — design tool concepts with MVP specs
4. Feasibility & Impact Scorer — score buildability, novelty, impact, validation

~30-60 seconds per run.  Batch generation uses asyncio.gather.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from loguru import logger

from research_mentor.db import crud
from research_mentor.question_workshop.forge.schemas import TOOL_CATEGORIES
from research_mentor.question_workshop.forge.stages.bottleneck_identifier import (
    identify_bottlenecks,
)
from research_mentor.question_workshop.forge.stages.domain_workflow_mapper import (
    map_domain_workflow,
)
from research_mentor.question_workshop.forge.stages.feasibility_impact_scorer import (
    score_feasibility_impact,
)
from research_mentor.question_workshop.forge.stages.solution_designer import (
    design_solutions,
)
from research_mentor.question_workshop.knowledge_base import FIELDS
from research_mentor.question_workshop.question_selector import (
    ScoredCandidate,
    select_questions,
)


async def generate_tool_ideas(
    field: str,
    tool_category: str = "any",
    seed_idea: str | None = None,
    project_id: str | None = None,
    student_demographics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate research tool ideas through the 4-stage pipeline.

    Args:
        field: Scientific field (physics, chemistry, biology).
        tool_category: software, hardware, methods_protocols, data_tools,
            field_equipment, or any.
        seed_idea: Optional seed idea from user.
        project_id: Optional project to link generated ideas to.
        student_demographics: Optional student profile for adaptation.

    Returns:
        Complete generated problem dict (matches generated_problems table columns).
    """
    if field not in FIELDS:
        raise ValueError(
            f"Invalid field '{field}'. Must be one of: {', '.join(sorted(FIELDS))}"
        )
    if tool_category not in TOOL_CATEGORIES:
        raise ValueError(
            f"Invalid tool_category '{tool_category}'. "
            f"Must be one of: {', '.join(sorted(TOOL_CATEGORIES))}"
        )

    # Check seed idea against domain scope if provided
    if seed_idea and len(seed_idea.strip()) >= 4:
        from research_mentor.question_workshop.domain import (
            get_domain_classification_prompt,
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            from research_mentor.llm import structured_call
            from research_mentor.question_workshop.gaps.schemas import (
                DomainClassification,
            )

            prompt = get_domain_classification_prompt().format(topic=seed_idea)
            classification = await structured_call(
                DomainClassification,
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content="Classify this topic."),
                ],
                thinking="medium",
                temperature=0.0,
            )
            if classification.domain_type == "refused":
                raise ValueError(
                    "That seed idea goes beyond typical science and involves "
                    "social, political, ethical, or clinical considerations "
                    "that we can\u2019t address with methodology alone. "
                    "Try a different seed idea focused on natural science."
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
        "Question Workshop [forge]: generating field={} category={} seed={}",
        field, tool_category, seed_idea[:50] if seed_idea else None,
    )

    from research_mentor.agent.prompts.shared import build_student_profile_context

    profile_text = build_student_profile_context(student_demographics or {})

    # Stage 1: Domain Workflow Mapping
    logger.info("Stage 1/4: Domain Workflow Mapping")
    workflow_data = await map_domain_workflow(
        field=field,
        tool_category=tool_category,
        seed_idea=seed_idea,
        student_profile_text=profile_text,
    )

    # Stage 2: Bottleneck Identification
    logger.info("Stage 2/4: Bottleneck Identification")
    bottleneck_data = await identify_bottlenecks(
        field=field,
        tool_category=tool_category,
        workflow_data=workflow_data,
        student_profile_text=profile_text,
    )

    # Stage 3: Solution Design
    logger.info("Stage 3/4: Solution Design")
    tool_concepts = await design_solutions(
        field=field,
        tool_category=tool_category,
        bottleneck_data=bottleneck_data,
        student_profile_text=profile_text,
        current_tools=workflow_data.get("current_tools", []),
    )

    # Stage 4: Feasibility & Impact Scoring
    logger.info("Stage 4/4: Feasibility & Impact Scoring")
    existing_titles = await crud.get_generated_problem_titles(field, tool_category)
    scores = await score_feasibility_impact(
        field=field,
        tool_concepts=tool_concepts,
        existing_titles=existing_titles,
        student_profile_text=profile_text,
    )

    # Find the best tool (weighted mean — novelty counts 2x to avoid
    # selecting tools that merely duplicate existing free solutions)
    scored_tools = scores.get("scored_tools", [])
    tools = tool_concepts.get("tools", [])

    best_tool_idx = 0
    best_mean = 0.0
    for i, st in enumerate(scored_tools):
        mean = (
            st.get("student_buildable_score", 5)
            + st.get("novelty_score", 5) * 2
            + st.get("impact_score", 5)
            + st.get("validation_path_score", 5)
        ) / 5
        if mean > best_mean:
            best_mean = mean
            best_tool_idx = i

    best_tool = tools[best_tool_idx] if tools else {}
    best_scores = scored_tools[best_tool_idx] if scored_tools else {}

    # Calculate overall quality (mean of 4 dimension scores for best tool)
    quality_scores = [
        best_scores.get("student_buildable_score", 5),
        best_scores.get("novelty_score", 5),
        best_scores.get("impact_score", 5),
        best_scores.get("validation_path_score", 5),
    ]
    overall_quality = round(sum(quality_scores) / len(quality_scores), 1)

    # Assemble all tools with their scores for metadata
    all_tools_with_scores = []
    for i, t in enumerate(tools):
        entry = dict(t)
        if i < len(scored_tools):
            entry["scores"] = scored_tools[i]
        all_tools_with_scores.append(entry)

    # Build description combining problem + solution
    description_parts = []
    if best_tool.get("problem_addressed"):
        description_parts.append(best_tool["problem_addressed"])
    if best_tool.get("solution_design"):
        description_parts.append(best_tool["solution_design"])
    description = " | ".join(description_parts) if description_parts else ""

    record = {
        "id": uuid.uuid4().hex,
        "project_id": project_id,
        "field": field,
        "problem_type": tool_category,
        "workshop_type": "forge",
        "domains": json.dumps(workflow_data.get("research_stages", [])),
        "user_suggestion": seed_idea,
        "title": best_tool.get("name", "Unnamed Tool"),
        "description": description,
        "investigation": best_tool.get("mvp_spec", ""),
        "core_concepts": json.dumps(best_tool.get("required_skills", [])),
        "materials": json.dumps(best_tool.get("required_resources", [])),
        "feasibility": best_scores.get("overall_feasibility", "MEDIUM"),
        "recommended": scores.get("recommended", "WITH_MODIFICATIONS"),
        "safety_level": "SAFE",
        "complexity_score": best_scores.get("student_buildable_score"),
        "engagement_score": best_scores.get("impact_score"),
        "overall_quality": overall_quality,
        "metadata": json.dumps({
            "all_tools": all_tools_with_scores,
            "bottlenecks": bottleneck_data.get("bottlenecks", []),
            "workflow_stages": workflow_data.get("research_stages", []),
            "field_context": workflow_data.get("field_context", ""),
            "recommended_starting_point": scores.get("recommended_starting_point", ""),
            "tool_category": tool_category,
        }),
    }

    # Persist to DB
    stored = await crud.create_generated_problem(**record)

    logger.info(
        "Question Workshop [forge] complete: title='{}' quality={} "
        "feasibility={} recommended={}",
        stored["title"], overall_quality,
        stored["feasibility"], stored["recommended"],
    )
    return stored


async def generate_best_of_n(
    field: str,
    tool_category: str = "any",
    seed_idea: str | None = None,
    project_id: str | None = None,
    candidates: int = 3,
    student_demographics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate N tool idea sets in parallel, return the best ones.

    All candidates run concurrently (vLLM batches them — same wall time as 1).
    The candidate with the highest overall_quality score is returned.
    All candidates are stored in DB for history.

    Args:
        field: Scientific field.
        tool_category: Tool category.
        seed_idea: Optional seed idea.
        project_id: Optional project link.
        candidates: Number of parallel candidates (default 3).
        student_demographics: Optional student profile for adaptation.

    Returns:
        The best generated tool idea dict.
    """
    logger.info(
        "Question Workshop [forge]: generating best of {} (field={} category={})",
        candidates, field, tool_category,
    )

    tasks = [
        generate_tool_ideas(
            field=field,
            tool_category=tool_category,
            seed_idea=seed_idea,
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
        "Question Workshop [forge] best of {}: {} selected from {} candidates",
        candidates, len(selected), len(good),
    )

    return selected
