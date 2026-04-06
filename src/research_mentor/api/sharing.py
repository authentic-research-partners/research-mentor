"""Sharing & Publication API — REST endpoints for the Sharing module.

Endpoints:
    POST /api/sharing/chat            — Non-streaming chat turn
    POST /api/sharing/chat/stream     — SSE streaming chat turn
    GET  /api/sharing/sessions        — List sessions
    PATCH /api/sharing/sessions/{id}  — Update session (rename)
    DELETE /api/sharing/sessions/{id} — Delete session
    GET  /api/sharing/sessions/{id}/messages — Load conversation history
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from research_mentor.db import crud

router = APIRouter(prefix="/api/sharing", tags=["sharing"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class SharingChatRequest(BaseModel):
    """Request body for Sharing chat."""

    message: str
    session_id: str | None = None  # omit for new, provide to continue
    project_id: str = "default"
    language: str = "en"


class SharingChatResponse(BaseModel):
    """Response from a Sharing chat turn."""

    response: str
    session_id: str
    current_phase: str
    progress: dict[str, Any]


class SharingSessionUpdate(BaseModel):
    """Updateable session fields."""

    title: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _load_student_demographics() -> dict[str, Any]:
    """Auto-load student demographics from profile DB."""
    try:
        from research_mentor.db.crud import get_student_profile

        profile = await get_student_profile()
        if profile:
            return {
                "firstName": profile.get("name") or "",
                "age": profile.get("age"),
                "gradeLevel": (
                    profile.get("grade_level") or profile.get("gradeLevel")
                ),
                "educationLevel": profile.get("education_level"),
                "country": profile.get("country"),
            }
    except Exception:
        logger.debug("Could not load student demographics")
    return {}


def _extract_sharing_progress(state: dict[str, Any]) -> dict[str, Any]:
    """Build the progress dict from Sharing graph state."""
    return {
        "research_summary": state.get("research_summary"),
        "output_type": state.get("output_type"),
        "student_level": state.get("student_level"),
        "context_gathered": state.get("context_gathered", False),
        "phases_visited": state.get("phases_visited", []),
        "matched_venues_count": len(state.get("matched_venues", [])),
        "venue_search_done": state.get("venue_search_done", False),
        "writing_focus": state.get("writing_focus"),
        "communication_topic": state.get("communication_topic"),
        "workflow_complete": state.get("workflow_complete", False),
    }


# ---------------------------------------------------------------------------
# Non-streaming chat
# ---------------------------------------------------------------------------


@router.post("/chat")
async def sharing_chat(body: SharingChatRequest) -> SharingChatResponse:
    """Process one Sharing chat turn (non-streaming)."""
    from langchain_core.messages import AIMessage, HumanMessage

    from research_mentor.llm import set_usage_context
    from research_mentor.sharing.graph import get_sharing_graph
    from research_mentor.sharing.state import initialize_sharing_state

    graph = await get_sharing_graph()

    is_new = body.session_id is None
    session_id = body.session_id or uuid.uuid4().hex

    set_usage_context(
        purpose="sharing", session_id=session_id,
        project_id=body.project_id,
    )

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 30,
    }

    if is_new:
        demographics = await _load_student_demographics()
        input_state = initialize_sharing_state(
            user_id=body.project_id,
            session_id=session_id,
            language=body.language,
            student_demographics=demographics,
        )
        input_state["messages"] = [HumanMessage(content=body.message)]
        initial_title = body.message[:80].strip()
        if len(body.message) > 80:
            initial_title += "…"
        await crud.create_workshop_session(
            project_id=body.project_id,
            workshop_type="sharing",
            session_id=session_id,
            language=body.language,
            current_stage="context_gathering",
            title=initial_title,
        )
    else:
        input_state = {"messages": [HumanMessage(content=body.message)]}

    logger.info(
        "Sharing chat: session={} new={}", session_id, is_new,
    )

    result = await graph.ainvoke(input_state, config)

    response_text = result.get("last_ai_response", "")
    current_phase = result.get("current_phase", "context_gathering")
    progress = _extract_sharing_progress(result)

    if response_text:
        await graph.aupdate_state(
            config,
            {"messages": [AIMessage(content=response_text)]},
        )

    return SharingChatResponse(
        response=response_text,
        session_id=session_id,
        current_phase=current_phase,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# SSE streaming chat
# ---------------------------------------------------------------------------


@router.post("/chat/stream")
async def sharing_chat_stream(body: SharingChatRequest) -> StreamingResponse:
    """Stream a Sharing chat turn via SSE.

    SSE event format:
    - ``data: {"session_id": "..."}``   — first event
    - ``data: {"phase": "..."}``        — current phase
    - ``data: {"token": "..."}``        — streamed token
    - ``data: {"response": "..."}``     — full response (fallback)
    - ``data: {"progress": {...}}``     — extracted state
    - ``: heartbeat``                   — keep-alive
    - ``data: [DONE]``                  — stream complete
    """
    from langchain_core.messages import AIMessage, HumanMessage

    from research_mentor.llm import (
        get_call_stats,
        reset_call_stats,
        set_usage_context,
    )
    from research_mentor.sharing.graph import get_sharing_graph
    from research_mentor.sharing.state import initialize_sharing_state

    graph = await get_sharing_graph()

    is_new = body.session_id is None
    session_id = body.session_id or uuid.uuid4().hex

    set_usage_context(
        purpose="sharing", session_id=session_id,
        project_id=body.project_id,
    )

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 30,
    }

    if is_new:
        demographics = await _load_student_demographics()
        input_state = initialize_sharing_state(
            user_id=body.project_id,
            session_id=session_id,
            language=body.language,
            student_demographics=demographics,
        )
        input_state["messages"] = [HumanMessage(content=body.message)]
        initial_title = body.message[:80].strip()
        if len(body.message) > 80:
            initial_title += "…"
        await crud.create_workshop_session(
            project_id=body.project_id,
            workshop_type="sharing",
            session_id=session_id,
            language=body.language,
            current_stage="context_gathering",
            title=initial_title,
        )
    else:
        input_state = {"messages": [HumanMessage(content=body.message)]}

    logger.info("Sharing stream: session={} new={}", session_id, is_new)

    SHARING_NODES = {
        "entry_validator", "intent_detector",
        "capability_switch",
        "context_gathering", "writing_support",
        "venue_discovery", "communication_guidance",
        "completion_handler",
    }

    HEARTBEAT_INTERVAL = 5
    GLOBAL_TIMEOUT = 180

    async def event_generator() -> Any:
        from research_mentor.tools.status import set_tool_activity_callback

        yield f"data: {json.dumps({'session_id': session_id})}\n\n"

        heartbeat_queue: asyncio.Queue[str] = asyncio.Queue()

        async def on_tool_activity(event: dict[str, Any]) -> None:
            sse = f"data: {json.dumps({'tool_activity': event})}\n\n"
            await heartbeat_queue.put(sse)

        set_tool_activity_callback(on_tool_activity)

        async def send_heartbeats() -> None:
            total = 0
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                total += HEARTBEAT_INTERVAL
                await heartbeat_queue.put(": heartbeat\n\n")
                if total >= GLOBAL_TIMEOUT:
                    err = "Sharing request timeout"
                    await heartbeat_queue.put(
                        f"data: {json.dumps({'error': err})}\n\n",
                    )
                    await heartbeat_queue.put("data: [DONE]\n\n")
                    return

        heartbeat_task = asyncio.create_task(send_heartbeats())

        try:
            reset_call_stats()
            t0 = time.monotonic()
            response_text = ""

            async for event in graph.astream_events(
                input_state, config, version="v2",
            ):
                kind = event.get("event", "")
                name = event.get("name", "")

                if kind == "on_chain_start" and name in SHARING_NODES:
                    yield f"data: {json.dumps({'node': name})}\n\n"

                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk:
                        content = getattr(chunk, "content", "")
                        if content:
                            response_text += content
                            yield (
                                f"data: "
                                f"{json.dumps({'token': content})}\n\n"
                            )

                while not heartbeat_queue.empty():
                    hb = heartbeat_queue.get_nowait()
                    yield hb
                    if "[DONE]" in hb:
                        return

            final_state = await graph.aget_state(config)
            state_values = final_state.values if final_state else {}
            final_response = state_values.get(
                "last_ai_response", response_text,
            )

            if not response_text and final_response:
                yield (
                    f"data: "
                    f"{json.dumps({'response': final_response})}\n\n"
                )
            else:
                yield (
                    f"data: "
                    f"{json.dumps({'response_complete': True})}\n\n"
                )

            progress = _extract_sharing_progress(state_values)
            current_phase = state_values.get(
                "current_phase", "context_gathering",
            )
            yield f"data: {json.dumps({'phase': current_phase})}\n\n"
            yield f"data: {json.dumps({'progress': progress})}\n\n"

            # Update workshop session metadata
            ws_updates: dict[str, Any] = {
                "current_stage": current_phase,
            }
            summary = state_values.get("research_summary")
            if summary:
                ws_updates["title"] = summary[:80]
            ws_meta: dict[str, Any] = {}
            if summary:
                ws_meta["research_summary"] = summary
            output_type = state_values.get("output_type")
            if output_type:
                ws_meta["output_type"] = output_type
            if ws_meta:
                ws_updates["metadata"] = ws_meta
            if state_values.get("workflow_complete"):
                ws_updates["status"] = "completed"
            await crud.update_workshop_session(
                session_id, **ws_updates,
            )

            ai_text = final_response or response_text
            if ai_text:
                await graph.aupdate_state(
                    config,
                    {"messages": [AIMessage(content=ai_text)]},
                )

            yield "data: [DONE]\n\n"

            wall_time = time.monotonic() - t0
            stats = get_call_stats()
            logger.info(
                "Sharing stream complete | session={} phase={} "
                "wall={:.1f}s llm_calls={} "
                "total_tokens={}in/{}out llm_time={:.1f}s",
                session_id, current_phase,
                wall_time, stats["calls"],
                stats["prompt_tokens"],
                stats["completion_tokens"],
                stats["elapsed"],
            )

        except Exception:
            logger.exception(
                "Sharing stream error | session={}", session_id,
            )
            yield (
                f"data: "
                f"{json.dumps({'error': 'Internal server error'})}\n\n"
            )
            yield "data: [DONE]\n\n"

        finally:
            heartbeat_task.cancel()
            set_tool_activity_callback(None)

    return StreamingResponse(
        event_generator(), media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


@router.get("/sessions")
async def list_sharing_sessions(
    project_id: str = Query(...),
    status: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """List Sharing sessions for a project."""
    return await crud.list_workshop_sessions(
        project_id, "sharing", status=status, page=page, page_size=page_size,
    )


@router.patch("/sessions/{session_id}")
async def update_sharing_session(
    session_id: str, body: SharingSessionUpdate,
) -> dict[str, Any]:
    """Update a Sharing session (e.g., rename title)."""
    fields: dict[str, Any] = {}
    if body.title is not None:
        fields["title"] = body.title
    result = await crud.update_workshop_session(session_id, **fields)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_sharing_session(session_id: str) -> None:
    """Delete a Sharing session."""
    if not await crud.delete_workshop_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/sessions/{session_id}/messages")
async def get_sharing_session_messages(session_id: str) -> dict[str, Any]:
    """Load conversation history and state from a Sharing session."""
    from langchain_core.messages import AIMessage, HumanMessage

    from research_mentor.sharing.graph import get_sharing_graph

    session = await crud.get_workshop_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    graph = await get_sharing_graph()
    state = await graph.aget_state(
        {"configurable": {"thread_id": session_id}},
    )

    messages: list[dict[str, str]] = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            messages.append(
                {"role": "assistant", "content": str(msg.content)},
            )

    phase = state.values.get("current_phase", "context_gathering")
    progress = _extract_sharing_progress(state.values)

    return {"messages": messages, "phase": phase, "progress": progress}
