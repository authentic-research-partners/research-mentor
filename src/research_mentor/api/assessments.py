"""Assessments API — list/get and generate, nested under projects."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from research_mentor.api.schemas import (
    AssessmentResponse,
    PaginatedAssessments,
    _pagination_info,
)
from research_mentor.db import crud

router = APIRouter(prefix="/api/projects/{project_id}/assessments", tags=["assessments"])


async def _require_project(project_id: str) -> None:
    if await crud.get_project(project_id) is None:
        raise HTTPException(404, "Project not found")


@router.get("", response_model=PaginatedAssessments)
async def list_assessments(
    project_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
) -> PaginatedAssessments:
    await _require_project(project_id)
    result = await crud.list_assessments(project_id, page=page, page_size=page_size)
    return PaginatedAssessments(
        data=[AssessmentResponse(**a) for a in result["data"]],
        pagination=_pagination_info(page, page_size, result["total"]),
    )


@router.get("/{assessment_id}", response_model=AssessmentResponse)
async def get_assessment(project_id: str, assessment_id: str) -> AssessmentResponse:
    await _require_project(project_id)
    assessment = await crud.get_assessment(assessment_id)
    if assessment is None or assessment["project_id"] != project_id:
        raise HTTPException(404, "Assessment not found")
    return AssessmentResponse(**assessment)


@router.post("/generate", response_model=AssessmentResponse, status_code=201)
async def generate_assessment(project_id: str) -> AssessmentResponse:
    """Generate a new AI assessment for a project.

    Uses available project data (milestones, artifacts, conversation memories,
    student profile, previous assessments) to produce a health score, identify
    issues, recommend actions, and write a narrative assessment.

    Two LLM calls: structured analysis + narrative synthesis.
    """
    await _require_project(project_id)

    from research_mentor.assessment import generate_assessment as _generate

    logger.info("Assessment generation requested for project {}", project_id)

    result = await _generate(project_id)

    from research_mentor.config import get_llm_model_info

    assessment = await crud.create_assessment(
        project_id=project_id,
        assessment_date=date.today().isoformat(),
        health_score=result["health_score"],
        issues=result["issues"],
        recommended_actions=result["recommended_actions"],
        general_notes=result["general_notes"],
        metadata={**result["metadata"], **get_llm_model_info()},
    )

    return AssessmentResponse(**assessment)
