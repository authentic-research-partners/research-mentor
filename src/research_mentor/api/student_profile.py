"""Student Profile API — singleton GET/PUT."""

from __future__ import annotations

from fastapi import APIRouter

from research_mentor.api.schemas import (
    StudentProfileResponse,
    StudentProfileUpdate,
)
from research_mentor.db import crud

router = APIRouter(prefix="/api/student-profile", tags=["student-profile"])


@router.get("", response_model=StudentProfileResponse)
async def get_student_profile() -> StudentProfileResponse:
    """Get the current student profile settings."""
    profile = await crud.get_student_profile()
    if profile is None:
        return StudentProfileResponse()
    return StudentProfileResponse(**profile)


@router.put("", response_model=StudentProfileResponse)
async def update_student_profile(body: StudentProfileUpdate) -> StudentProfileResponse:
    """Create or update student profile settings.

    Only fields included in the request body are updated;
    omitted fields retain their current values.
    """
    fields = body.model_dump(exclude_unset=True, mode="python")
    result = await crud.upsert_student_profile(**fields)
    return StudentProfileResponse(**result)
