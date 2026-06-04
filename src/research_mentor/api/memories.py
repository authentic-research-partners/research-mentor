"""Memories API — create/list/search under projects, delete by id."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from research_mentor.api.schemas import (
    MemoryCreate,
    MemoryResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    PaginatedMemories,
    _pagination_info,
)
from research_mentor.db import crud

project_router = APIRouter(prefix="/api/projects/{project_id}/memories", tags=["memories"])
memory_router = APIRouter(prefix="/api/memories", tags=["memories"])


async def _require_project(project_id: str) -> None:
    if await crud.get_project(project_id) is None:
        raise HTTPException(404, "Project not found")


@project_router.post("", response_model=MemoryResponse, status_code=201)
async def create_memory(project_id: str, body: MemoryCreate) -> MemoryResponse:
    await _require_project(project_id)
    # Auto-embed for semantic recall. embeddings (sentence-transformers) is a hard
    # dependency, so a failed import is a packaging bug — let it surface rather than
    # silently storing a memory with no vector.
    from research_mentor.embeddings import embed_text

    embedding = await embed_text(body.memory_text)
    memory_id = await crud.store_memory(
        project_id=project_id,
        memory_text=body.memory_text,
        memory_type=body.memory_type,
        session_id=body.session_id,
        importance=body.importance,
        conversation_context=body.conversation_context,
        tags=body.tags,
        embedding=embedding,
    )
    # Find the one we just created
    result = await crud.get_memories_for_project(project_id, page_size=50)
    for m in result["data"]:
        if m["id"] == memory_id:
            return MemoryResponse(**m)
    # Fallback — construct from input
    return MemoryResponse(
        id=memory_id,
        memory_text=body.memory_text,
        memory_type=body.memory_type,
        importance=body.importance,
        tags=body.tags or [],
        created_at="",
    )


@project_router.get("", response_model=PaginatedMemories)
async def list_memories(
    project_id: str,
    type: str | None = None,  # noqa: A002
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
) -> PaginatedMemories:
    await _require_project(project_id)
    result = await crud.get_memories_for_project(
        project_id, memory_type=type, page=page, page_size=page_size,
    )
    return PaginatedMemories(
        data=[MemoryResponse(**m) for m in result["data"]],
        pagination=_pagination_info(page, page_size, result["total"]),
    )


@project_router.post("/search", response_model=list[MemorySearchResponse])
async def search_memories(
    project_id: str, body: MemorySearchRequest,
) -> list[MemorySearchResponse]:
    await _require_project(project_id)
    from research_mentor.embeddings import embed_text

    query_embedding = await embed_text(body.query)
    results = await crud.search_similar_memories(
        project_id, query_embedding, threshold=body.threshold, limit=body.limit,
    )
    return [MemorySearchResponse(**r) for r in results]


@memory_router.delete("/{memory_id}", status_code=204)
async def delete_memory(memory_id: str) -> None:
    if not await crud.delete_memory(memory_id):
        raise HTTPException(404, "Memory not found")
