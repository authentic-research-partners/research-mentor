"""Sessions API — create/list under projects, get/patch/delete by session_id."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from langchain_core.messages import AIMessage, HumanMessage

from research_mentor.api.schemas import (
    PaginatedSessions,
    SessionCreate,
    SessionResponse,
    SessionUpdate,
    _pagination_info,
)
from research_mentor.db import crud

project_router = APIRouter(prefix="/api/projects/{project_id}/sessions", tags=["sessions"])
session_router = APIRouter(prefix="/api/sessions", tags=["sessions"])


async def _require_project(project_id: str) -> None:
    if await crud.get_project(project_id) is None:
        raise HTTPException(404, "Project not found")


@project_router.post("", response_model=SessionResponse, status_code=201)
async def create_session(project_id: str, body: SessionCreate) -> SessionResponse:
    await _require_project(project_id)
    return SessionResponse(**await crud.create_session(project_id, **body.model_dump()))


@project_router.get("", response_model=PaginatedSessions)
async def list_sessions(
    project_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
) -> PaginatedSessions:
    await _require_project(project_id)
    result = await crud.list_sessions(project_id, page=page, page_size=page_size)
    return PaginatedSessions(
        data=[SessionResponse(**s) for s in result["data"]],
        pagination=_pagination_info(page, page_size, result["total"]),
    )


@session_router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    session = await crud.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return SessionResponse(**session)


@session_router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(session_id: str, body: SessionUpdate) -> SessionResponse:
    fields = body.model_dump(exclude_unset=True)
    result = await crud.update_session(session_id, **fields)
    if result is None:
        raise HTTPException(404, "Session not found")
    return SessionResponse(**result)


@session_router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    if not await crud.delete_session(session_id):
        raise HTTPException(404, "Session not found")


@session_router.get("/{session_id}/messages")
async def get_session_messages(session_id: str) -> list[dict[str, Any]]:
    """Load conversation history from checkpointer state."""
    session = await crud.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

    from research_mentor.agent.graph import get_mentor_graph

    graph = await get_mentor_graph()
    state = await graph.aget_state({"configurable": {"thread_id": session_id}})  # type: ignore[attr-defined]

    messages: list[dict[str, Any]] = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    return messages
