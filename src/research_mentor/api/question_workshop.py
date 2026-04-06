"""Question Workshop API — async job-based generation with cancellation."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from research_mentor.config import load_config
from research_mentor.db import crud
from research_mentor.question_workshop.claims.orchestrator import (
    generate_best_of_n as claims_generate_best_of_n,
)
from research_mentor.question_workshop.claims.schemas import (
    InvestigateClaimRequest,
)
from research_mentor.question_workshop.forge.schemas import (
    TOOL_CATEGORIES,
    ForgeRequest,
)
from research_mentor.question_workshop.knowledge_base import (
    FIELDS,
    PROBLEM_TYPES,
    get_domains_for_field,
)
from research_mentor.question_workshop.orchestrator import generate_best_of_n
from research_mentor.question_workshop.questioned.schemas import (
    SurfaceScrutinyRequest,
)
from research_mentor.question_workshop.retractions.pipeline.schemas import (
    RetractionsGenerateRequest,
)
from research_mentor.question_workshop.schemas import GenerateRequest

router = APIRouter(prefix="/api/question-workshop", tags=["question-workshop"])

# ---------------------------------------------------------------------------
# In-memory job tracker
# ---------------------------------------------------------------------------

JOB_TTL_SECONDS = 300  # remove finished jobs after 5 minutes


@dataclass
class GenerationJob:
    """Tracks a running generation task."""

    id: str
    task: asyncio.Task[Any] | None = None
    status: str = "pending"
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    tools_used: list[dict[str, Any]] = field(default_factory=list)


_jobs: dict[str, GenerationJob] = {}


def _cleanup_old_jobs() -> None:
    """Remove finished jobs older than TTL."""
    now = time.monotonic()
    expired = [
        jid for jid, job in _jobs.items()
        if job.finished_at is not None
        and (now - job.finished_at) > JOB_TTL_SECONDS
    ]
    for jid in expired:
        del _jobs[jid]


def _attach_tool_activity(job: GenerationJob) -> None:
    """Set up the tool activity callback so tool calls are recorded on the job."""
    from research_mentor.tools.status import set_tool_activity_callback

    async def on_tool_activity(event: dict[str, Any]) -> None:
        job.tools_used.append(event)

    set_tool_activity_callback(on_tool_activity)


async def _run_generation(job: GenerationJob, request: GenerateRequest) -> None:
    """Background task that runs generation and updates the job."""
    from research_mentor.llm import set_usage_context

    job.status = "running"
    _attach_tool_activity(job)
    set_usage_context(
        purpose="workshop_phenomenon", project_id=request.project_id,
    )
    try:
        config = load_config()
        n = config.question_workshop.candidates
        demographics = await _load_student_demographics()
        results = await generate_best_of_n(
            field=request.field,
            problem_type=request.problem_type,
            domain=request.domain,
            user_suggestion=request.user_suggestion,
            project_id=request.project_id,
            candidates=n,
            student_demographics=demographics,
        )
        # results is a list of up to 3 selected questions
        job.result = {"questions": results, "success": True}
        job.status = "completed"
        logger.info(
            "Job {} completed: {} questions", job.id, len(results),
        )
    except asyncio.CancelledError:
        job.status = "cancelled"
        logger.info("Job {} cancelled", job.id)
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        logger.exception("Job {} failed: {}", job.id, exc)
        job.finished_at = time.monotonic()


async def _run_investigation(
    job: GenerationJob, request: InvestigateClaimRequest,
) -> None:
    """Background task that runs claim investigation and updates the job."""
    from research_mentor.llm import set_usage_context

    job.status = "running"
    _attach_tool_activity(job)
    set_usage_context(
        purpose="workshop_claims", project_id=request.project_id,
    )
    try:
        config = load_config()
        n = config.question_workshop.candidates
        demographics = await _load_student_demographics()
        results = await claims_generate_best_of_n(
            claim_text=request.claim_text,
            claim_url=request.claim_url,
            article_text=request.article_text,
            field_filter=request.field_filter,
            project_id=request.project_id,
            candidates=n,
            student_demographics=demographics,
        )
        job.result = {"questions": results, "success": True}
        job.status = "completed"
        logger.info(
            "Investigation job {} completed: {} questions",
            job.id, len(results),
        )
    except asyncio.CancelledError:
        job.status = "cancelled"
        logger.info("Investigation job {} cancelled", job.id)
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        logger.exception("Investigation job {} failed: {}", job.id, exc)
        job.finished_at = time.monotonic()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/generate")
async def start_generate(body: GenerateRequest) -> dict[str, Any]:
    """Start async generation, return job_id immediately."""
    _cleanup_old_jobs()

    job_id = uuid.uuid4().hex[:12]
    job = GenerationJob(id=job_id)
    _jobs[job_id] = job

    task = asyncio.create_task(_run_generation(job, body))
    job.task = task

    logger.info(
        "Job {} started: field={} type={}",
        job_id, body.field, body.problem_type,
    )
    return {"job_id": job_id}


@router.post("/investigate")
async def start_investigation(body: InvestigateClaimRequest) -> dict[str, Any]:
    """Start async claim investigation, return job_id immediately."""
    if not body.claim_text and not body.claim_url and not body.article_text:
        raise HTTPException(
            status_code=422,
            detail="Must provide claim_text, claim_url, or article_text",
        )
    _cleanup_old_jobs()

    job_id = uuid.uuid4().hex[:12]
    job = GenerationJob(id=job_id)
    _jobs[job_id] = job

    task = asyncio.create_task(_run_investigation(job, body))
    job.task = task

    logger.info("Investigation job {} started", job_id)
    return {"job_id": job_id}


async def _run_scrutiny_search(
    job: GenerationJob, request: SurfaceScrutinyRequest,
) -> None:
    """Background task that runs Questioned scrutiny search."""
    from research_mentor.llm import set_usage_context
    from research_mentor.question_workshop.questioned.orchestrator import (
        surface_scrutinized_papers,
    )

    job.status = "running"
    _attach_tool_activity(job)
    set_usage_context(
        purpose="workshop_questioned", project_id=request.project_id,
    )
    try:
        demographics = await _load_student_demographics()
        result = await surface_scrutinized_papers(
            field=request.field,
            project_id=request.project_id,
            student_demographics=demographics,
        )
        job.result = result
        job.status = "completed"
        logger.info(
            "Scrutiny job {} completed: field='{}'",
            job.id, request.field,
        )
    except asyncio.CancelledError:
        job.status = "cancelled"
        logger.info("Scrutiny job {} cancelled", job.id)
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        logger.exception("Scrutiny job {} failed: {}", job.id, exc)
        job.finished_at = time.monotonic()


async def _run_forge(
    job: GenerationJob, request: ForgeRequest,
) -> None:
    """Background task that runs tool development generation."""
    from research_mentor.llm import set_usage_context
    from research_mentor.question_workshop.forge.orchestrator import (
        generate_best_of_n as td_generate_best_of_n,
    )

    job.status = "running"
    _attach_tool_activity(job)
    set_usage_context(
        purpose="workshop_forge", project_id=request.project_id,
    )
    try:
        config = load_config()
        n = config.question_workshop.candidates
        demographics = await _load_student_demographics()
        results = await td_generate_best_of_n(
            field=request.field,
            tool_category=request.tool_category,
            seed_idea=request.seed_idea,
            project_id=request.project_id,
            candidates=n,
            student_demographics=demographics,
        )
        # results is a list of up to 3 selected tool ideas
        job.result = {"questions": results, "success": True}
        job.status = "completed"
        logger.info(
            "Forge job {} completed: {} ideas", job.id, len(results),
        )
    except asyncio.CancelledError:
        job.status = "cancelled"
        logger.info("Forge job {} cancelled", job.id)
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        logger.exception("Forge job {} failed: {}", job.id, exc)
        job.finished_at = time.monotonic()


@router.post("/forge")
async def start_forge(body: ForgeRequest) -> dict[str, Any]:
    """Start async tool development generation, return job_id."""
    _cleanup_old_jobs()

    job_id = uuid.uuid4().hex[:12]
    job = GenerationJob(id=job_id)
    _jobs[job_id] = job

    task = asyncio.create_task(_run_forge(job, body))
    job.task = task

    logger.info(
        "Forge job {} started: field={} category={}",
        job_id, body.field, body.tool_category,
    )
    return {"job_id": job_id}


@router.post("/surface-scrutiny")
async def start_scrutiny_search(body: SurfaceScrutinyRequest) -> dict[str, Any]:
    """Start async scrutiny search (Questioned workshop), return job_id."""
    _cleanup_old_jobs()

    job_id = uuid.uuid4().hex[:12]
    job = GenerationJob(id=job_id)
    _jobs[job_id] = job

    task = asyncio.create_task(_run_scrutiny_search(job, body))
    job.task = task

    logger.info("Scrutiny job {} started: field={}", job_id, body.field)
    return {"job_id": job_id}


async def _run_retractions_pipeline(
    job: GenerationJob, request: RetractionsGenerateRequest,
) -> None:
    """Background task that runs the Retractions pipeline."""
    from research_mentor.llm import set_usage_context
    from research_mentor.question_workshop.retractions.pipeline.orchestrator import (
        generate_retraction_questions,
    )

    job.status = "running"
    _attach_tool_activity(job)
    set_usage_context(
        purpose="workshop_retractions_pipeline", project_id=request.project_id,
    )
    try:
        demographics = await _load_student_demographics()
        result = await generate_retraction_questions(
            query=request.query,
            project_id=request.project_id,
            student_demographics=demographics,
        )
        job.result = result
        job.status = "completed"
        logger.info(
            "Retractions pipeline job {} completed: query='{}'",
            job.id, request.query[:60],
        )
    except asyncio.CancelledError:
        job.status = "cancelled"
        logger.info("Retractions pipeline job {} cancelled", job.id)
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        logger.exception("Retractions pipeline job {} failed: {}", job.id, exc)
    finally:
        job.finished_at = time.monotonic()


@router.post("/retractions/generate")
async def start_retractions_pipeline(body: RetractionsGenerateRequest) -> dict[str, Any]:
    """Start async Retractions pipeline generation, return job_id."""
    _cleanup_old_jobs()

    job_id = uuid.uuid4().hex[:12]
    job = GenerationJob(id=job_id)
    _jobs[job_id] = job

    task = asyncio.create_task(_run_retractions_pipeline(job, body))
    job.task = task

    logger.info(
        "Retractions pipeline job {} started: query='{}'",
        job_id, body.query[:60],
    )
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict[str, Any]:
    """Poll job status."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    resp: dict[str, Any] = {
        "job_id": job.id,
        "status": job.status,
    }
    if job.tools_used:
        resp["tools_used"] = job.tools_used
    if job.status == "completed" and job.result is not None:
        resp["result"] = job.result
    if job.status == "failed" and job.error is not None:
        resp["error"] = job.error
    return resp


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel a running generation job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.task is not None and not job.task.done():
        job.task.cancel()
        job.status = "cancelled"
        job.finished_at = time.monotonic()
        logger.info("Job {} cancelled by user", job_id)
    return {"job_id": job_id, "status": job.status}


# ---------------------------------------------------------------------------
# Other endpoints (unchanged)
# ---------------------------------------------------------------------------


@router.get("/fields")
async def get_fields() -> dict[str, Any]:
    """Return available fields, problem types, and domains."""
    result: dict[str, Any] = {
        "fields": sorted(FIELDS),
        "problem_types": sorted(PROBLEM_TYPES),
        "tool_categories": sorted(TOOL_CATEGORIES),
        "domains": {},
    }
    for f in sorted(FIELDS):
        result["domains"][f] = [
            {"id": d["id"], "name": d["name"], "description": d["description"]}
            for d in get_domains_for_field(f)
        ]
    return result


@router.get("/history")
async def list_history(
    field: str | None = Query(default=None),
    problem_type: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    workshop_type: str | None = Query(default=None),
    search: str | None = Query(default=None),
    min_quality: float | None = Query(default=None, ge=1, le=10),
    min_engagement: int | None = Query(default=None, ge=1, le=10),
    min_complexity: int | None = Query(default=None, ge=1, le=10),
    sort_by: str = Query(default="created_at"),
    sort_desc: bool = Query(default=True),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=5, ge=1, le=100),
) -> dict[str, Any]:
    """Paginated list of generated problems."""
    return await crud.list_generated_problems(
        field=field,
        problem_type=problem_type,
        project_id=project_id,
        workshop_type=workshop_type,
        search=search,
        min_quality=min_quality,
        min_engagement=min_engagement,
        min_complexity=min_complexity,
        sort_by=sort_by,
        sort_desc=sort_desc,
        page=page,
        page_size=page_size,
    )


class SaveQuestionRequest(BaseModel):
    session_id: str
    workshop_type: str


_GRAPH_GETTERS: dict[str, str] = {
    "hypothesis": "research_mentor.question_workshop.hypothesis.graph.get_hypothesis_graph",
}


@router.post("/save-question")
async def save_question_from_session(body: SaveQuestionRequest) -> dict[str, Any]:
    """Extract the best research question from an interactive workshop session
    and save it to the generated_problems table."""
    import json
    from importlib import import_module

    from research_mentor.question_workshop.extract_question import (
        extract_question,
        polish_question,
    )

    supported = set(_GRAPH_GETTERS)
    if body.workshop_type not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported workshop type '{body.workshop_type}'. "
            f"Supported: {', '.join(sorted(supported))}",
        )

    session = await crud.get_workshop_session(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check for duplicate saves via metadata
    from research_mentor.db.connection import get_db

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM generated_problems WHERE metadata LIKE ?",
            (f'%{body.session_id}%',),
        )
        if await cursor.fetchone():
            raise HTTPException(
                status_code=409, detail="A question from this session was already saved",
            )

    # Read checkpointed state
    getter_path = _GRAPH_GETTERS[body.workshop_type]
    module_path, func_name = getter_path.rsplit(".", 1)
    module = import_module(module_path)
    get_graph = getattr(module, func_name)
    graph = await get_graph()
    state = await graph.aget_state({"configurable": {"thread_id": body.session_id}})

    if not state or not state.values:
        raise HTTPException(status_code=404, detail="No state found for this session")

    # Extract question
    question = extract_question(body.workshop_type, state.values)
    if question is None:
        raise HTTPException(
            status_code=422,
            detail="This session has not yet produced a saveable research question. "
            "Continue working through the workshop stages.",
        )

    # Polish with LLM — rewrite title/description/investigation into clean prose
    question = await polish_question(question)

    # Attach project_id and source metadata
    question["project_id"] = session.get("project_id")
    question["metadata"] = json.dumps({"source_session_id": body.session_id})

    stored = await crud.create_generated_problem(**question)
    logger.info(
        "Saved question from {} session {} → problem {}",
        body.workshop_type, body.session_id, stored["id"],
    )
    return stored


@router.get("/{problem_id}")
async def get_problem(problem_id: str) -> dict[str, Any]:
    """Get a single generated problem."""
    result = await crud.get_generated_problem(problem_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail="Generated problem not found",
        )
    return result


@router.delete("/{problem_id}", status_code=204)
async def delete_problem(problem_id: str) -> None:
    """Delete a generated problem."""
    deleted = await crud.delete_generated_problem(problem_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail="Generated problem not found",
        )


# ---------------------------------------------------------------------------
# Shared: Domain classification for workshop routing
# ---------------------------------------------------------------------------


async def _classify_domain(message: str) -> str | None:
    """Classify a topic as natural_science, social_science, refused, or off_topic.

    Returns the domain_type string, or None if classification fails.
    Used by Hypothesis and Gaps streaming endpoints to redirect users
    to the appropriate workshop on new sessions.
    """
    if len(message.strip()) < 4:
        return None
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from research_mentor.llm import structured_call
        from research_mentor.question_workshop.domain import (
            get_domain_classification_prompt,
        )
        from research_mentor.question_workshop.gaps.schemas import (
            DomainClassification,
        )

        prompt = get_domain_classification_prompt().format(topic=message)
        result = await structured_call(
            DomainClassification,
            [
                SystemMessage(content=prompt),
                HumanMessage(content="Classify this topic."),
            ],
            thinking="low",
            temperature=0.0,
        )
        return result.domain_type
    except Exception:
        logger.debug("Domain classification failed, skipping guard")
        return None


# ---------------------------------------------------------------------------
# Hypothesis Workshop — interactive chat endpoint
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
                    profile.get("grade") or profile.get("college_year") or ""
                ),
                "country": profile.get("country") or "",
                "educationLevel": profile.get("education_level") or "",
                "domainExpertise": profile.get("domain_expertise", []),
                "professionalExperience": profile.get("professional_experience", []),
                "backgroundNotes": profile.get("background_notes") or "",
            }
    except Exception:
        logger.debug("Could not load student demographics")
    return {}


class HypothesisChatRequest(BaseModel):
    """Request body for Hypothesis workshop chat."""

    message: str
    session_id: str | None = None  # omit for new, provide to continue
    project_id: str = "default"
    language: str = "en"


class HypothesisChatResponse(BaseModel):
    """Response from a Hypothesis workshop chat turn."""

    response: str
    session_id: str
    current_stage: str
    progress: dict[str, Any]


@router.post("/hypothesis/chat")
async def hypothesis_chat(body: HypothesisChatRequest) -> HypothesisChatResponse:
    """Process one Hypothesis workshop chat turn.

    Non-streaming endpoint: sends user message through the Hypothesis graph,
    returns the AI response plus current stage and extracted progress.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    from research_mentor.llm import set_usage_context
    from research_mentor.question_workshop.hypothesis.graph import get_hypothesis_graph
    from research_mentor.question_workshop.hypothesis.state import initialize_hypothesis_state

    graph = await get_hypothesis_graph()

    is_new = body.session_id is None
    session_id = body.session_id or uuid.uuid4().hex

    set_usage_context(
        purpose="workshop_hypothesis", session_id=session_id,
        project_id=body.project_id,
    )

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 30,
    }

    if is_new:
        demographics = await _load_student_demographics()
        input_state = initialize_hypothesis_state(
            user_id=body.project_id,
            session_id=session_id,
            language=body.language,
            student_demographics=demographics,
        )
        input_state["messages"] = [HumanMessage(content=body.message)]
    else:
        input_state = {"messages": [HumanMessage(content=body.message)]}

    logger.info(
        "Hypothesis chat: session={} new={} stage={}",
        session_id, is_new, "start" if is_new else "continue",
    )

    result = await graph.ainvoke(input_state, config)

    response_text = result.get("last_ai_response", "")
    current_stage = result.get("current_stage", "stage_1_discovery")
    progress = _extract_hypothesis_progress(result)

    # Persist AI response so session restore shows both sides
    if response_text:
        await graph.aupdate_state(
            config,
            {"messages": [AIMessage(content=response_text)]},
        )

    return HypothesisChatResponse(
        response=response_text,
        session_id=session_id,
        current_stage=current_stage,
        progress=progress,
    )


def _extract_hypothesis_progress(state: dict[str, Any]) -> dict[str, Any]:
    """Build the progress dict from Hypothesis graph state."""
    return {
        "independent_var": state.get("independent_var"),
        "dependent_var": state.get("dependent_var"),
        "gap_identified": state.get("gap_identified", False),
        "hypothesis": state.get("hypothesis"),
        "hypothesis_direction": state.get("hypothesis_direction"),
        "scope_defined": state.get("scope_defined", False),
        "scope_details": state.get("scope_details"),
        "datasets_searched": state.get("datasets_searched", False),
        "datasets_count": state.get("datasets_count", 0),
        "selected_dataset": state.get("selected_dataset"),
        "confounds_count": state.get("confounds_count", 0),
        "data_alignment_evaluated": state.get(
            "data_alignment_evaluated", False,
        ),
        "statistical_method": state.get("statistical_method"),
        "ethics_discussed": state.get("ethics_discussed", False),
        "software_discussed": state.get("software_discussed", False),
        "timeline_discussed": state.get("timeline_discussed", False),
        "workflow_complete": state.get("workflow_complete", False),
    }


@router.get("/hypothesis/sessions")
async def list_hypothesis_sessions(
    project_id: str = Query(...),
    status: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """List Hypothesis workshop sessions for a project."""
    from research_mentor.db import crud

    return await crud.list_workshop_sessions(
        project_id, "hypothesis", status=status, page=page, page_size=page_size,
    )


class WorkshopSessionUpdate(BaseModel):
    title: str | None = None


@router.patch("/hypothesis/sessions/{session_id}")
async def update_hypothesis_session(
    session_id: str, body: WorkshopSessionUpdate,
) -> dict[str, Any]:
    """Update a Hypothesis session (e.g., rename title)."""
    from research_mentor.db import crud

    fields: dict[str, Any] = {}
    if body.title is not None:
        fields["title"] = body.title
    result = await crud.update_workshop_session(session_id, **fields)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@router.delete("/hypothesis/sessions/{session_id}", status_code=204)
async def delete_hypothesis_session(session_id: str) -> None:
    """Delete a Hypothesis workshop session."""
    if not await crud.delete_workshop_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/hypothesis/sessions/{session_id}/messages")
async def get_hypothesis_session_messages(session_id: str) -> dict[str, Any]:
    """Load conversation history and state from a Hypothesis session."""
    from langchain_core.messages import AIMessage, HumanMessage

    from research_mentor.question_workshop.hypothesis.graph import get_hypothesis_graph

    session = await crud.get_workshop_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    graph = await get_hypothesis_graph()
    state = await graph.aget_state({"configurable": {"thread_id": session_id}})

    messages: list[dict[str, str]] = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": str(msg.content)})

    stage = state.values.get("current_stage", "stage_1_discovery")
    progress = _extract_hypothesis_progress(state.values)

    return {"messages": messages, "stage": stage, "progress": progress}


@router.post("/hypothesis/chat/stream")
async def hypothesis_chat_stream(body: HypothesisChatRequest) -> StreamingResponse:
    """Stream a Hypothesis workshop chat turn via SSE.

    SSE event format:
    - ``data: {"session_id": "..."}``       — first event
    - ``data: {"stage": "..."}``            — current stage
    - ``data: {"token": "..."}``            — streamed token
    - ``data: {"response": "..."}``         — full response (fallback)
    - ``data: {"progress": {...}}``         — extracted state
    - ``: heartbeat``                       — keep-alive
    - ``data: [DONE]``                      — stream complete
    """
    import json

    from langchain_core.messages import AIMessage, HumanMessage

    from research_mentor.db import crud
    from research_mentor.llm import (
        get_call_stats,
        reset_call_stats,
        set_usage_context,
    )
    from research_mentor.question_workshop.hypothesis.graph import get_hypothesis_graph
    from research_mentor.question_workshop.hypothesis.state import initialize_hypothesis_state

    graph = await get_hypothesis_graph()

    is_new = body.session_id is None
    session_id = body.session_id or uuid.uuid4().hex

    set_usage_context(
        purpose="workshop_hypothesis", session_id=session_id,
        project_id=body.project_id,
    )

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 30,
    }

    if is_new:
        demographics = await _load_student_demographics()
        input_state = initialize_hypothesis_state(
            user_id=body.project_id,
            session_id=session_id,
            language=body.language,
            student_demographics=demographics,
        )
        input_state["messages"] = [HumanMessage(content=body.message)]
        # Initial title: truncated first message (upgraded to "X → Y" later)
        initial_title = body.message[:80].strip()
        if len(body.message) > 80:
            initial_title += "…"
        await crud.create_workshop_session(
            project_id=body.project_id,
            workshop_type="hypothesis",
            session_id=session_id,
            language=body.language,
            current_stage="stage_1_discovery",
            title=initial_title,
        )
    else:
        input_state = {"messages": [HumanMessage(content=body.message)]}

    logger.info(
        "Hypothesis stream: session={} new={}", session_id, is_new,
    )

    # Domain guard: redirect or refuse on new sessions
    if is_new:
        domain_type = await _classify_domain(body.message)
        if domain_type == "refused":
            import json as _json

            _refuse_msg = (
                "That topic goes beyond typical science and involves "
                "social, political, ethical, or clinical considerations "
                "that we can\u2019t address with methodology alone. "
                "We recommend working with a faculty advisor for "
                "guidance on that area. "
                "Would you like to explore a different topic?"
            )

            async def _refuse_gen() -> Any:
                sid_evt = _json.dumps({"session_id": session_id})
                resp_evt = _json.dumps({"response": _refuse_msg})
                yield f"data: {sid_evt}\n\n"
                yield f"data: {resp_evt}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                _refuse_gen(), media_type="text/event-stream",
            )

        if domain_type == "natural_science":
            import json as _json

            from research_mentor.components_registry import get_redirect_phrase

            _redirect_msg = (
                get_redirect_phrase("hypothesis", "gaps") or ""
            ) + (
                " Hypothesis is designed for social science research"
                " questions with X\u2192Y causal variables."
                " Would you like to explore a social science"
                " topic here instead?"
            )

            async def _redirect_gen() -> Any:
                sid_evt = _json.dumps({"session_id": session_id})
                resp_evt = _json.dumps({"response": _redirect_msg})
                yield f"data: {sid_evt}\n\n"
                yield f"data: {resp_evt}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                _redirect_gen(), media_type="text/event-stream",
            )

    # Hypothesis nodes to report progress on
    HYPOTHESIS_NODES = {
        "entry_validator",
        "stage_1_discovery", "stage_2_literature",
        "stage_3a_hypothesis", "stage_3b_data",
        "stage_3c_refinement", "stage_3_5_confounds",
        "stage_4_operationalization",
        "stage_5a_analysis", "stage_5b_ethics", "stage_5b_software",
        "stage_5b_timeline", "completion_handler",
    }

    HEARTBEAT_INTERVAL = 5
    GLOBAL_TIMEOUT = 120

    async def event_generator() -> Any:
        from research_mentor.tools.status import set_tool_activity_callback

        yield f"data: {json.dumps({'session_id': session_id})}\n\n"

        heartbeat_queue: asyncio.Queue[str] = asyncio.Queue()
        last_activity = asyncio.Event()

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
                    err = "Hypothesis request timeout"
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
            current_stage = "stage_1_discovery"

            async for event in graph.astream_events(
                input_state, config, version="v2",
            ):
                last_activity.set()
                kind = event.get("event", "")
                name = event.get("name", "")

                # Node progress
                if kind == "on_chain_start" and name in HYPOTHESIS_NODES:
                    yield f"data: {json.dumps({'node': name})}\n\n"

                # Token streaming from any chat model
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk:
                        content = getattr(chunk, "content", "")
                        if content:
                            response_text += content
                            yield (
                                f"data: {json.dumps({'token': content})}\n\n"
                            )

                # Drain heartbeats
                while not heartbeat_queue.empty():
                    hb = heartbeat_queue.get_nowait()
                    yield hb
                    if "[DONE]" in hb:
                        return

            # Get final state for progress
            final_state = await graph.aget_state(config)
            state_values = final_state.values if final_state else {}
            final_response = state_values.get(
                "last_ai_response", response_text,
            )

            # Send full response if no tokens were streamed
            if not response_text and final_response:
                yield (
                    f"data: {json.dumps({'response': final_response})}\n\n"
                )
            else:
                yield f"data: {json.dumps({'response_complete': True})}\n\n"

            # Send progress snapshot
            progress = _extract_hypothesis_progress(state_values)
            progress_stage = state_values.get(
                "current_stage", current_stage,
            )
            yield f"data: {json.dumps({'stage': progress_stage})}\n\n"
            yield f"data: {json.dumps({'progress': progress})}\n\n"

            # Update workshop session metadata BEFORE [DONE]
            # so the frontend sees the updated title when it refreshes
            x_var = state_values.get("independent_var")
            y_var = state_values.get("dependent_var")
            ws_updates: dict[str, Any] = {
                "current_stage": progress_stage,
            }
            if x_var and y_var:
                ws_updates["title"] = f"{x_var} → {y_var}"
            ws_meta: dict[str, Any] = {}
            if x_var:
                ws_meta["independent_var"] = x_var
            if y_var:
                ws_meta["dependent_var"] = y_var
            hypothesis = state_values.get("hypothesis")
            if hypothesis:
                ws_meta["hypothesis"] = hypothesis
            if ws_meta:
                ws_updates["metadata"] = ws_meta
            if state_values.get("workflow_complete"):
                ws_updates["status"] = "completed"
            await crud.update_workshop_session(
                session_id, **ws_updates,
            )

            # Persist AI response as AIMessage so session restore
            # shows both sides of the conversation
            ai_text = final_response or response_text
            if ai_text:
                await graph.aupdate_state(
                    config,
                    {"messages": [AIMessage(content=ai_text)]},
                )

            yield "data: [DONE]\n\n"

            # Log summary (same pattern as main chat stream)
            wall_time = time.monotonic() - t0
            stats = get_call_stats()
            logger.info(
                "Hypothesis stream complete | session={} stage={} "
                "wall={:.1f}s llm_calls={} "
                "total_tokens={}in/{}out llm_time={:.1f}s",
                session_id, progress_stage,
                wall_time, stats["calls"],
                stats["prompt_tokens"], stats["completion_tokens"],
                stats["elapsed"],
            )

        except Exception:
            logger.exception("Hypothesis stream error | session={}", session_id)
            yield (
                f"data: {json.dumps({'error': 'Internal server error'})}\n\n"
            )
            yield "data: [DONE]\n\n"

        finally:
            heartbeat_task.cancel()
            set_tool_activity_callback(None)

    return StreamingResponse(
        event_generator(), media_type="text/event-stream",
    )


# ============================================================================
# Modeling Workshop — Pipeline (question generation)
# ============================================================================


class ModelingGenerateRequest(BaseModel):
    """Request body for Modeling pipeline question generation."""

    system_topic: str
    approach: str = "any"
    project_id: str = "default"


async def _run_modeling_generation(
    job: GenerationJob, request: ModelingGenerateRequest,
) -> None:
    """Background task that runs the modeling pipeline."""
    from research_mentor.llm import set_usage_context
    from research_mentor.question_workshop.modeling.pipeline.orchestrator import (
        generate_modeling_questions,
    )

    job.status = "running"
    _attach_tool_activity(job)
    set_usage_context(
        purpose="workshop_modeling", project_id=request.project_id,
    )
    try:
        demographics = await _load_student_demographics()
        result = await generate_modeling_questions(
            system_topic=request.system_topic,
            approach=request.approach,
            project_id=request.project_id,
            student_demographics=demographics,
        )
        job.result = result
        job.status = "completed"
        logger.info(
            "Modeling job {} completed: {} questions",
            job.id, len(result.get("questions", [])),
        )
    except asyncio.CancelledError:
        job.status = "cancelled"
        logger.info("Modeling job {} cancelled", job.id)
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        logger.exception("Modeling job {} failed: {}", job.id, exc)
    finally:
        job.finished_at = time.monotonic()


@router.post("/modeling/generate")
async def start_modeling_generation(
    body: ModelingGenerateRequest,
) -> dict[str, Any]:
    """Start async modeling question generation, return job_id.

    The pipeline searches for published models, generates 3-5 research
    questions investigable through computational modeling, and scores
    their feasibility. Poll /jobs/{job_id} for results.
    """
    _cleanup_old_jobs()

    job_id = uuid.uuid4().hex[:12]
    job = GenerationJob(id=job_id)
    _jobs[job_id] = job

    task = asyncio.create_task(_run_modeling_generation(job, body))
    job.task = task

    logger.info(
        "Modeling job {} started: system='{}' approach={}",
        job_id, body.system_topic[:60], body.approach,
    )
    return {"job_id": job_id}


@router.get("/modeling/approaches")
async def get_modeling_approaches() -> dict[str, Any]:
    """Return available modeling approaches with descriptions for UI display."""
    from research_mentor.question_workshop.modeling.pipeline.schemas import (
        MODELING_APPROACHES,
    )

    return {"approaches": MODELING_APPROACHES}


# ============================================================================
# Gaps Workshop — Pipeline (question generation)
# ============================================================================


class GapsGenerateRequest(BaseModel):
    """Request body for Gaps pipeline question generation."""

    field_topic: str
    interest_area: str | None = None
    project_id: str = "default"


async def _run_gaps_generation(
    job: GenerationJob, request: GapsGenerateRequest,
) -> None:
    """Background task that runs Gaps pipeline generation."""
    from research_mentor.llm import set_usage_context
    from research_mentor.question_workshop.gaps.pipeline.orchestrator import (
        generate_best_of_n,
    )

    job.status = "running"
    _attach_tool_activity(job)
    set_usage_context(
        purpose="workshop_gaps_pipeline", project_id=request.project_id,
    )
    try:
        config = load_config()
        n = config.question_workshop.candidates
        demographics = await _load_student_demographics()
        result = await generate_best_of_n(
            field_topic=request.field_topic,
            interest_area=request.interest_area,
            candidates=n,
            student_demographics=demographics,
        )
        job.result = result
        job.status = "completed"
        logger.info(
            "Gaps pipeline job {} completed: {} questions",
            job.id, len(result.get("questions", [])),
        )
    except asyncio.CancelledError:
        job.status = "cancelled"
        logger.info("Gaps pipeline job {} cancelled", job.id)
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        logger.exception("Gaps pipeline job {} failed: {}", job.id, exc)
    finally:
        job.finished_at = time.monotonic()


@router.post("/gaps/generate")
async def start_gaps_generation(
    body: GapsGenerateRequest,
) -> dict[str, Any]:
    """Start async gaps pipeline question generation, return job_id.

    The pipeline searches literature, analyzes gaps via multiple strategic
    modes, generates candidate questions, and scores them on FUNDAMENTAL
    criteria. Poll /jobs/{job_id} for results.
    """
    _cleanup_old_jobs()

    job_id = uuid.uuid4().hex[:12]
    job = GenerationJob(id=job_id)
    _jobs[job_id] = job

    task = asyncio.create_task(_run_gaps_generation(job, body))
    job.task = task

    logger.info(
        "Gaps pipeline job {} started: field='{}' interest='{}'",
        job_id, body.field_topic[:60], body.interest_area,
    )
    return {"job_id": job_id}


# ============================================================================
# Theory Workshop — Pipeline (question generation)
# ============================================================================


class TheoryGenerateRequest(BaseModel):
    """Request body for Theory pipeline question generation."""

    observations_text: str
    domain: str
    project_id: str = "default"


async def _run_theory_generation(
    job: GenerationJob, request: TheoryGenerateRequest,
) -> None:
    """Background task that runs Theory pipeline generation."""
    from research_mentor.llm import set_usage_context
    from research_mentor.question_workshop.theory.pipeline.orchestrator import (
        generate_best_of_n,
    )

    job.status = "running"
    _attach_tool_activity(job)
    set_usage_context(
        purpose="workshop_theory_pipeline", project_id=request.project_id,
    )
    try:
        config = load_config()
        n = config.question_workshop.candidates
        demographics = await _load_student_demographics()
        result = await generate_best_of_n(
            observations_text=request.observations_text,
            domain=request.domain,
            project_id=request.project_id,
            candidates=n,
            student_demographics=demographics,
        )
        job.result = result
        job.status = "completed"
        logger.info(
            "Theory pipeline job {} completed: {} questions, quality={}",
            job.id, len(result.get("questions", [])),
            result.get("overall_quality"),
        )
    except asyncio.CancelledError:
        job.status = "cancelled"
        logger.info("Theory pipeline job {} cancelled", job.id)
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        logger.exception("Theory pipeline job {} failed: {}", job.id, exc)
    finally:
        job.finished_at = time.monotonic()


@router.post("/theory/generate")
async def start_theory_generation(
    body: TheoryGenerateRequest,
) -> dict[str, Any]:
    """Start async theory pipeline question generation, return job_id.

    The pipeline maps phenomena, applies abductive reasoning, builds a
    theoretical framework, tests consilience across domains, generates
    theory-grounded questions, and scores feasibility.
    Poll /jobs/{job_id} for results.
    """
    _cleanup_old_jobs()

    job_id = uuid.uuid4().hex[:12]
    job = GenerationJob(id=job_id)
    _jobs[job_id] = job

    task = asyncio.create_task(_run_theory_generation(job, body))
    job.task = task

    logger.info(
        "Theory pipeline job {} started: domain='{}'",
        job_id, body.domain[:60],
    )
    return {"job_id": job_id}

