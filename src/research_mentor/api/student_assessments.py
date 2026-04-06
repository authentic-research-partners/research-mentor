"""Student Assessments API — cross-project skill tracking."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from research_mentor.api.schemas import (
    PaginatedStudentAssessments,
    StudentAssessmentResponse,
    _pagination_info,
)
from research_mentor.db import crud

router = APIRouter(prefix="/api/student/assessments", tags=["student-assessments"])


@router.get("", response_model=PaginatedStudentAssessments)
async def list_student_assessments(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
) -> PaginatedStudentAssessments:
    """List all cross-project student assessments (paginated)."""
    result = await crud.list_student_assessments(page=page, page_size=page_size)
    return PaginatedStudentAssessments(
        data=[StudentAssessmentResponse(**a) for a in result["data"]],
        pagination=_pagination_info(page, page_size, result["total"]),
    )


@router.get("/latest", response_model=StudentAssessmentResponse | None)
async def get_latest_student_assessment() -> StudentAssessmentResponse | None:
    """Get the most recent cross-project student assessment."""
    assessment = await crud.get_latest_student_assessment()
    if assessment is None:
        return None
    return StudentAssessmentResponse(**assessment)


@router.delete("/{assessment_id}", status_code=204)
async def delete_student_assessment(assessment_id: str) -> None:
    """Delete a student assessment by ID."""
    deleted = await crud.delete_student_assessment(assessment_id)
    if not deleted:
        raise HTTPException(404, "Student assessment not found")


class StudentAssessmentGenerateResponse(BaseModel):
    """Response from assessment generation."""

    assessment: StudentAssessmentResponse
    projects_analyzed: int


@router.post("/generate", response_model=StudentAssessmentGenerateResponse, status_code=201)
async def generate_student_assessment_endpoint() -> StudentAssessmentGenerateResponse:
    """Generate a new cross-project student assessment via LLM.

    Analyzes all projects, their assessments, milestones, and conversation
    memories to produce an overall skill assessment with growth trajectory
    and per-domain skill scores.
    """
    from research_mentor.assessment import generate_student_assessment

    # Get project count for response
    projects_result = await crud.list_projects(page=1, page_size=1)

    try:
        assessment = await generate_student_assessment()
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    return StudentAssessmentGenerateResponse(
        assessment=StudentAssessmentResponse(**assessment),
        projects_analyzed=projects_result["total"],
    )
