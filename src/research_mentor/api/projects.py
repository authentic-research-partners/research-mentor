"""Projects API — full CRUD."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from research_mentor.api.schemas import (
    AssessmentResponse,
    MilestoneResponse,
    PaginatedProjects,
    ProjectCreate,
    ProjectDetailResponse,
    ProjectResponse,
    ProjectUpdate,
    _pagination_info,
)
from research_mentor.db import crud

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(body: ProjectCreate) -> ProjectResponse:
    return ProjectResponse(**await crud.create_project(**body.model_dump()))


@router.get("", response_model=PaginatedProjects)
async def list_projects(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    status: str | None = Query(default=None),
) -> PaginatedProjects:
    result = await crud.list_projects(page=page, page_size=page_size, status=status)
    return PaginatedProjects(
        data=[ProjectResponse(**p) for p in result["data"]],
        pagination=_pagination_info(page, page_size, result["total"]),
    )


@router.get("/{project_id}", response_model=ProjectDetailResponse)
async def get_project(project_id: str) -> ProjectDetailResponse:
    project = await crud.get_project(project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    milestones_result = await crud.list_milestones(project_id)
    assessments_result = await crud.list_assessments(project_id)
    assessments = assessments_result["data"]
    return ProjectDetailResponse(
        **project,
        milestones=[MilestoneResponse(**m) for m in milestones_result["data"]],
        latest_assessment=AssessmentResponse(**assessments[0]) if assessments else None,
    )


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, body: ProjectUpdate) -> ProjectResponse:
    fields = body.model_dump(exclude_unset=True)
    result = await crud.update_project(project_id, **fields)
    if result is None:
        raise HTTPException(404, "Project not found")
    return ProjectResponse(**result)


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str) -> None:
    if not await crud.delete_project(project_id):
        raise HTTPException(404, "Project not found")
