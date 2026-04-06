"""Personas API — read-only endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from research_mentor.api.schemas import (
    PaginatedPersonas,
    PersonaResponse,
    _pagination_info,
)
from research_mentor.db import crud

router = APIRouter(prefix="/api/personas", tags=["personas"])


@router.get("", response_model=PaginatedPersonas)
async def list_personas(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
) -> PaginatedPersonas:
    result = await crud.list_personas(page=page, page_size=page_size)
    return PaginatedPersonas(
        data=[PersonaResponse(**p) for p in result["data"]],
        pagination=_pagination_info(page, page_size, result["total"]),
    )


@router.get("/{persona_id}", response_model=PersonaResponse)
async def get_persona(persona_id: str) -> PersonaResponse:
    persona = await crud.get_persona(persona_id)
    if persona is None:
        raise HTTPException(404, "Persona not found")
    return PersonaResponse(**persona)
