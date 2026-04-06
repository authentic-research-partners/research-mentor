"""Milestones API — nested under projects."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from research_mentor.api.schemas import (
    MilestoneCreate,
    MilestoneResponse,
    MilestoneUpdate,
    PaginatedMilestones,
    _pagination_info,
)
from research_mentor.db import crud

router = APIRouter(prefix="/api/projects/{project_id}/milestones", tags=["milestones"])


async def _require_project(project_id: str) -> None:
    if await crud.get_project(project_id) is None:
        raise HTTPException(404, "Project not found")


@router.post("", response_model=MilestoneResponse, status_code=201)
async def create_milestone(project_id: str, body: MilestoneCreate) -> MilestoneResponse:
    await _require_project(project_id)
    return MilestoneResponse(**await crud.create_milestone(project_id, **body.model_dump()))


@router.get("", response_model=PaginatedMilestones)
async def list_milestones(
    project_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
) -> PaginatedMilestones:
    await _require_project(project_id)
    result = await crud.list_milestones(project_id, page=page, page_size=page_size)
    return PaginatedMilestones(
        data=[MilestoneResponse(**m) for m in result["data"]],
        pagination=_pagination_info(page, page_size, result["total"]),
    )


@router.get("/{milestone_id}", response_model=MilestoneResponse)
async def get_milestone(project_id: str, milestone_id: str) -> MilestoneResponse:
    await _require_project(project_id)
    milestone = await crud.get_milestone(milestone_id)
    if milestone is None or milestone["project_id"] != project_id:
        raise HTTPException(404, "Milestone not found")
    return MilestoneResponse(**milestone)


@router.patch("/{milestone_id}", response_model=MilestoneResponse)
async def update_milestone(
    project_id: str, milestone_id: str, body: MilestoneUpdate,
) -> MilestoneResponse:
    await _require_project(project_id)
    existing = await crud.get_milestone(milestone_id)
    if existing is None or existing["project_id"] != project_id:
        raise HTTPException(404, "Milestone not found")
    fields = body.model_dump(exclude_unset=True)
    result = await crud.update_milestone(milestone_id, **fields)
    if result is None:
        raise HTTPException(404, "Milestone not found")
    return MilestoneResponse(**result)


@router.delete("/{milestone_id}", status_code=204)
async def delete_milestone(project_id: str, milestone_id: str) -> None:
    await _require_project(project_id)
    existing = await crud.get_milestone(milestone_id)
    if existing is None or existing["project_id"] != project_id:
        raise HTTPException(404, "Milestone not found")
    await crud.delete_milestone(milestone_id)
