"""Questioned orchestrator — 4-stage sequential pipeline.

Surfaces papers under scrutiny in the student's field:
1. Field Validator — validate field is in scope
2. Scrutiny Searcher — semantic search for expressions of concern + corrections
3. Context Enricher — LLM-generated pedagogical context for each paper
4. Curator — select and frame the most interesting cases

~30-60 seconds total. No checkpointing (re-run on failure).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from loguru import logger

from research_mentor.db import crud
from research_mentor.question_workshop.questioned.nodes.context_enricher import (
    context_enricher,
)
from research_mentor.question_workshop.questioned.nodes.curator import curator
from research_mentor.question_workshop.questioned.nodes.field_validator import (
    field_validator,
)
from research_mentor.question_workshop.questioned.nodes.scrutiny_searcher import (
    scrutiny_searcher,
)


async def surface_scrutinized_papers(
    field: str,
    project_id: str | None = None,
    student_demographics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full 4-stage Questioned pipeline.

    Args:
        field: Academic field to search (e.g., "psychology", "physics").
        project_id: Optional project to link results to.
        student_demographics: Student profile (age, gradeLevel, country)
            for adapting explanation complexity.

    Returns:
        Complete generated problem dict (stored in generated_problems table),
        or a rejection dict if the field is not valid.
    """
    demographics = student_demographics or {}
    logger.info("Questioned: starting pipeline (field='{}')", field)

    # ===== Stage 1: Field Validation =====
    logger.info("Stage 1/4: Field Validation")
    result_1 = await field_validator(field)

    if not result_1.get("is_valid"):
        validation = result_1["field_validation"]
        logger.warning("Field rejected: {}", validation.rejection_reason)
        return {
            "success": False,
            "rejection_reason": validation.rejection_reason,
            "stage_reached": "field_validation",
        }

    validated_field = result_1["field_validation"].field

    # ===== Stage 2: Scrutiny Search =====
    logger.info("Stage 2/4: Scrutiny Search")
    result_2 = await scrutiny_searcher(validated_field)
    papers = result_2["papers"]
    tool_warnings = result_2.get("tool_warnings", [])

    if not papers:
        logger.info("No scrutinized papers found for '{}'", validated_field)
        return {
            "success": False,
            "rejection_reason": (
                f"No papers under scrutiny were found for '{field}'. "
                "This field may have few current expressions of concern "
                "or corrections. Try a broader field."
            ),
            "stage_reached": "scrutiny_search",
        }

    # ===== Stage 3: Context Enrichment =====
    logger.info("Stage 3/4: Context Enrichment")
    result_3 = await context_enricher(papers, demographics)
    enriched_papers = result_3["enriched_papers"]

    # ===== Stage 4: Curation =====
    logger.info("Stage 4/4: Curation")
    result_4 = await curator(enriched_papers, validated_field, demographics)
    questioned_result = result_4["questioned_result"]

    # ===== Assemble & persist =====
    cases_data = [case.model_dump() for case in questioned_result.cases]
    first_case = questioned_result.cases[0]

    record = {
        "id": uuid.uuid4().hex,
        "project_id": project_id,
        "field": validated_field,
        "problem_type": "scrutiny_analysis",
        "workshop_type": "questioned",
        "domains": json.dumps([]),
        "user_suggestion": field,
        "title": f"Papers Under Scrutiny in {validated_field.replace('_', ' ').title()}",
        "description": questioned_result.meta_lesson,
        "investigation": first_case.what_it_claimed,
        "core_concepts": json.dumps([]),
        "materials": json.dumps([]),
        "feasibility": "HIGH",
        "recommended": "YES",
        "safety_level": "SAFE",
        "complexity_score": None,
        "engagement_score": None,
        "overall_quality": None,
        "metadata": json.dumps({
            "cases": cases_data,
            "field": validated_field,
            "meta_lesson": questioned_result.meta_lesson,
            "papers_found": len(papers),
            "papers_enriched": len(enriched_papers),
            "papers_curated": len(questioned_result.cases),
        }),
    }

    stored = await crud.create_generated_problem(**record)

    logger.info(
        "Questioned complete: field='{}' cases={} papers_found={}",
        validated_field, len(questioned_result.cases), len(papers),
    )

    return {
        "success": True,
        "tool_warnings": tool_warnings,
        **stored,
    }
