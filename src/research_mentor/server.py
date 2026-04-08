"""FastAPI application for Personal Research Mentor."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse

from research_mentor import __version__
from research_mentor.api.artifacts import router as artifacts_router
from research_mentor.api.artifacts import types_router as artifact_types_router
from research_mentor.api.assessments import router as assessments_router
from research_mentor.api.data_management import router as data_management_router
from research_mentor.api.memories import (
    memory_router,
)
from research_mentor.api.memories import (
    project_router as memories_project_router,
)
from research_mentor.api.milestones import router as milestones_router
from research_mentor.api.personas import router as personas_router
from research_mentor.api.projects import router as projects_router
from research_mentor.api.question_workshop import router as question_workshop_router
from research_mentor.api.sessions import project_router as sessions_project_router
from research_mentor.api.sessions import session_router as sessions_router
from research_mentor.api.sharing import router as sharing_router
from research_mentor.api.student_assessments import router as student_assessments_router
from research_mentor.api.student_profile import router as student_profile_router
from research_mentor.db import crud
from research_mentor.languages import LANGUAGE_CODES, SUPPORTED_LANGUAGES
from research_mentor.title_generator import maybe_generate_title


async def _check_embedding_integrity() -> None:
    """Detect incomplete embeddings (e.g. interrupted rebuild) and rebuild.

    Checks the ``embeddings_status`` flag in ``app_meta``.  The flag is set to
    ``in_progress`` at the start of ``rebuild_embedding_tables()`` and flipped
    to ``complete`` only after all tables are fully rebuilt and committed.
    If the process crashes mid-rebuild the flag stays ``in_progress`` and this
    check triggers a fresh rebuild on the next startup.
    """
    from research_mentor.db.connection import get_db

    try:
        async with get_db() as db:
            cursor = await db.execute(
                "SELECT value FROM app_meta WHERE key = 'embeddings_status'"
            )
            row = await cursor.fetchone()

        status = row[0] if row else None
        if status is None or status == "complete":
            return

        logger.warning(
            "Embeddings status is '{}' — previous rebuild was interrupted, rebuilding",
            status,
        )
        from research_mentor.embeddings import rebuild_embedding_tables

        await rebuild_embedding_tables()

    except Exception as exc:
        logger.debug("Embedding integrity check skipped: {}", exc)


async def _check_embedding_dimension() -> None:
    """Detect embedding model change and rebuild if needed.

    Compares the model recorded in app_meta (written by rebuild_embedding_tables)
    with the configured model. Catches both dimension changes and same-dimension
    model swaps (different embedding spaces → stored vectors are invalid).
    """
    from research_mentor.config import load_config

    cfg = load_config()
    if not cfg.embeddings.enabled:
        return

    configured_model = cfg.embeddings.model

    try:
        from research_mentor.db.connection import get_db

        async with get_db() as db:
            cursor = await db.execute(
                "SELECT value FROM app_meta WHERE key = 'embedding_model'"
            )
            row = await cursor.fetchone()
            stored_model = row[0] if row else None

        if stored_model == configured_model:
            # Model matches — check for incomplete embeddings (interrupted rebuild)
            await _check_embedding_integrity()
            return

        if stored_model is None:
            logger.info(
                "No embedding model recorded in DB — recording current: {}",
                configured_model,
            )
            async with get_db() as db:
                await db.execute(
                    "INSERT OR REPLACE INTO app_meta (key, value) VALUES (?, ?)",
                    ("embedding_model", configured_model),
                )
                await db.commit()
            return

        logger.warning(
            "Embedding model changed: {} → {} — rebuilding tables",
            stored_model, configured_model,
        )
        from research_mentor.embeddings import rebuild_embedding_tables, reset_model

        reset_model()
        # dimension=None → auto-detected from the newly loaded model
        await rebuild_embedding_tables()

    except Exception as exc:
        logger.debug("Embedding model check skipped: {}", exc)


async def _auto_enable_grobid() -> None:
    """If GROBID is responding but not enabled in config, enable it automatically."""

    from research_mentor.config import load_config, update_user_config
    from research_mentor.grobid import is_grobid_available

    cfg = load_config()
    if cfg.grobid.enabled:
        return

    available = await is_grobid_available(cfg.grobid.url)
    if available:
        update_user_config({"grobid": {"enabled": True}})
        logger.info("GROBID detected at {} — auto-enabled", cfg.grobid.url)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Ensure DB is initialized when app starts.

    If the schema is outdated, the server still starts so the UI can
    present a migration button. Normal API endpoints should check
    health status before operating.
    """
    from research_mentor.db.connection import (
        SchemaMismatchError,
        _resolve_db_path,
        close_all_pools,
        init_db,
        set_db_path,
    )

    try:
        await init_db()
    except SchemaMismatchError as e:
        # Server starts anyway — set the path so health/migrate endpoints work
        logger.warning("Database needs migration: {}", e)
        set_db_path(_resolve_db_path())

    # Check if embedding dimension matches config — rebuild if mismatched
    # (handles manual config.toml edits or env var overrides)
    await _check_embedding_dimension()

    # Pre-load embedding model in background — don't block server startup.
    # If download is needed, user sees progress in console while server is already usable.
    # If a request needs embeddings before this finishes, _get_model() blocks that request only.
    from research_mentor.embeddings import _get_model
    asyncio.get_event_loop().run_in_executor(None, _get_model)

    # Auto-enable GROBID if it's responding but not yet enabled
    await _auto_enable_grobid()

    # Print startup banner AFTER DB + config init (model may still be downloading)
    banner_url = getattr(app.state, "startup_banner_url", None)
    if banner_url:
        from research_mentor.cli import _print_startup_banner
        _print_startup_banner(api_url=banner_url)

    yield

    # Clean up connection pool and shared GROBID HTTP client
    await close_all_pools()
    from research_mentor.grobid import close_client
    await close_client()


app = FastAPI(
    title="Personal Research Mentor",
    version=__version__,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Allow cross-origin requests from the Vite dev server (runs on a different port)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Security headers (defense-in-depth for localhost app)
# ---------------------------------------------------------------------------

class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to every response."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> StarletteResponse:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; font-src 'self' data:",
        )
        return response


app.add_middleware(_SecurityHeadersMiddleware)

app.include_router(personas_router)
app.include_router(projects_router)
app.include_router(milestones_router)
app.include_router(sessions_project_router)
app.include_router(sessions_router)
app.include_router(memories_project_router)
app.include_router(memory_router)
app.include_router(artifacts_router)
app.include_router(artifact_types_router)
app.include_router(assessments_router)
app.include_router(student_profile_router)
app.include_router(student_assessments_router)
app.include_router(data_management_router)
app.include_router(question_workshop_router)
app.include_router(sharing_router)


class GPUInfoResponse(BaseModel):
    """GPU info for the UI."""

    name: str
    vram_gb: int
    compute_capability: float
    fp8_native: bool
    vllm_compatible: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    data_dir: str
    log_file: str
    db_path: str
    config_file: str
    schema_version: int | None = None
    schema_target: int | None = None
    gpu_available: bool = False
    gpu: GPUInfoResponse | None = None


class OfficeRequest(BaseModel):
    message: str
    thread_id: str | None = None  # Omit for new conversation, provide for continuation
    project_id: str
    persona: str = "newton"
    language: str = "en"
    response_length: str = "normal"
    user_requested_guidance_level: str = "adaptive"
    student_progress: dict[str, Any] = Field(default_factory=dict)
    student_skills: dict[str, Any] = Field(default_factory=dict)
    project_context: str = ""
    student_demographics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if v not in LANGUAGE_CODES:
            supported = ", ".join(sorted(LANGUAGE_CODES))
            msg = f"Unsupported language: '{v}'. Supported: {supported}."
            raise ValueError(msg)
        return v

    @field_validator("user_requested_guidance_level")
    @classmethod
    def validate_guidance_level(cls, v: str) -> str:
        valid = ("adaptive", "beginner", "intermediate", "advanced", "expert")
        if v not in valid:
            msg = (
                f"Invalid user_requested_guidance_level: '{v}'. "
                f"Must be one of: {', '.join(valid)}."
            )
            raise ValueError(msg)
        return v


class ToolActivity(BaseModel):
    """Record of a single external tool call."""

    tool: str
    success: bool
    elapsed: float


class OfficeResponse(BaseModel):
    response: str
    thread_id: str
    guidance_type: str | None = None
    persona: str | None = None
    tool_warnings: list[dict[str, Any]] = []
    tools_used: list[ToolActivity] = []


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    from research_mentor.config import get_data_dir, get_user_config_path
    from research_mentor.config import load_config as _lc
    from research_mentor.db.connection import get_schema_version
    from research_mentor.db.schema import SCHEMA_VERSION

    cfg = _lc()
    data_dir = get_data_dir()

    # Read schema version (best-effort — don't crash health check)
    schema_version: int | None = None
    try:
        db_path = Path(cfg.db_path)
        if not db_path.is_absolute():
            db_path = data_dir / db_path
        if db_path.exists():
            current, _ = await get_schema_version(db_path)
            schema_version = current
    except Exception:
        logger.debug("Failed to read schema version from {}", db_path)

    needs_migration = (
        schema_version is not None
        and schema_version < SCHEMA_VERSION
    )

    # GPU detection
    from research_mentor.gpu import detect_gpu

    gpu_info = detect_gpu()
    gpu_response = None
    if gpu_info:
        gpu_response = GPUInfoResponse(
            name=gpu_info.name,
            vram_gb=gpu_info.vram_mb // 1024,
            compute_capability=gpu_info.compute_capability,
            fp8_native=gpu_info.fp8_native,
            vllm_compatible=gpu_info.vllm_compatible,
        )

    return HealthResponse(
        status="migration_required" if needs_migration else "ok",
        version=__version__,
        data_dir=str(data_dir),
        log_file=str(data_dir / cfg.logging.file),
        db_path=cfg.db_path,
        config_file=str(get_user_config_path()),
        schema_version=schema_version,
        schema_target=SCHEMA_VERSION,
        gpu_available=gpu_info is not None and gpu_info.vllm_compatible,
        gpu=gpu_response,
    )


@app.get("/api/languages")
async def list_languages() -> list[dict[str, str]]:
    """List supported languages for AI responses."""
    return SUPPORTED_LANGUAGES


@app.get("/api/logs/tail")
async def logs_tail(lines: int = 100) -> dict[str, Any]:
    """Return the last N lines of the server log file.

    Used by the Settings → System Status panel in power mode.
    """
    from collections import deque
    from pathlib import Path

    from research_mentor.config import get_data_dir
    from research_mentor.config import load_config as _lc

    cfg = _lc()
    log_path = Path(get_data_dir() / cfg.logging.file)

    lines = min(max(lines, 10), 500)  # clamp to 10–500

    if not log_path.exists():
        return {"lines": [], "path": str(log_path), "truncated": False}

    # Read last N lines efficiently using a deque (in thread to avoid blocking)
    import asyncio

    def _read_tail() -> list[str]:
        tail: deque[str] = deque(maxlen=lines)
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                tail.append(line.rstrip("\n"))
        return list(tail)

    result = await asyncio.to_thread(_read_tail)
    return {"lines": result, "path": str(log_path), "truncated": len(result) == lines}


@app.get("/api/tools/status")
async def get_tools_status() -> dict[str, Any]:
    """Return health and usage stats for all external tools.

    Checks configuration (API key presence) and queries tool_usage table
    for request counts. No live API probing — this is a fast, local check.
    """
    from pathlib import Path

    from research_mentor.config import load_config as _lc
    from research_mentor.db.crud import get_tool_usage_monthly, get_tool_usage_summary

    cfg = _lc()

    # Check Brave Search API key (non-blocking)
    import asyncio

    def _check_key(path: Path) -> bool:
        p = path.expanduser()
        return p.exists() and p.read_text().strip() != ""

    brave_key_path = Path(cfg.services.brave_search.api_key_file)
    tavily_key_path = Path(cfg.services.tavily.api_key_file)
    brave_configured = await asyncio.to_thread(_check_key, brave_key_path)
    tavily_configured = await asyncio.to_thread(_check_key, tavily_key_path)

    # Monthly limits from config
    brave_monthly_limit = cfg.rate_limits.brave_search_monthly[0]

    # DB queries
    summary = await get_tool_usage_summary("month")
    monthly = await get_tool_usage_monthly()

    summary_by_tool = {row["tool"]: row for row in summary}

    # Tool definitions with known limits
    rl = cfg.rate_limits
    has_email = bool(cfg.services.polite_email)

    openalex_rl = rl.openalex_polite if has_email else rl.openalex
    pubmed_rl = rl.pubmed_polite if has_email else rl.pubmed

    tool_defs = [
        {
            "tool": "brave_search",
            "display_name": "Brave Search",
            "configured": brave_configured,
            "requires_key": True,
            "monthly_limit": brave_monthly_limit,
            "rate_limit_desc": f"1/sec, {brave_monthly_limit:,}/month",
        },
        {
            "tool": "tavily_search",
            "display_name": "Tavily Search",
            "configured": tavily_configured,
            "requires_key": True,
            "monthly_limit": None,
            "rate_limit_desc": "1/sec",
        },
        {
            "tool": "openalex",
            "display_name": "OpenAlex",
            "configured": True,
            "requires_key": False,
            "monthly_limit": None,
            "rate_limit_desc": f"{openalex_rl[0]}/{openalex_rl[1]}s",
        },
        {
            "tool": "semantic_scholar",
            "display_name": "Semantic Scholar",
            "configured": True,
            "requires_key": False,
            "monthly_limit": None,
            "rate_limit_desc": f"{rl.semantic_scholar[0]}/{rl.semantic_scholar[1]}s",
        },
        {
            "tool": "pubmed",
            "display_name": "PubMed",
            "configured": True,
            "requires_key": False,
            "monthly_limit": None,
            "rate_limit_desc": f"{pubmed_rl[0]}/{pubmed_rl[1]}s",
        },
        {
            "tool": "arxiv",
            "display_name": "arXiv",
            "configured": True,
            "requires_key": False,
            "monthly_limit": None,
            "rate_limit_desc": f"{rl.arxiv[0]}/{rl.arxiv[1]}s",
        },
        {
            "tool": "core_api",
            "display_name": "CORE",
            "configured": True,  # works without key (lower quota)
            "requires_key": False,
            "monthly_limit": None,
            "rate_limit_desc": f"{rl.core_api[0]}/{rl.core_api[1]}s",
        },
        {
            "tool": "unpaywall",
            "display_name": "Unpaywall",
            "configured": has_email,  # requires polite_email
            "requires_key": False,
            "monthly_limit": None,
            "rate_limit_desc": f"{rl.unpaywall[0]}/{rl.unpaywall[1]}s",
        },
        {
            "tool": "pmc",
            "display_name": "PubMed Central",
            "configured": True,  # works without NCBI key (lower rate)
            "requires_key": False,
            "monthly_limit": None,
            "rate_limit_desc": f"{rl.pmc[0]}/{rl.pmc[1]}s",
        },
        {
            "tool": "europepmc",
            "display_name": "Europe PMC",
            "configured": True,
            "requires_key": False,
            "monthly_limit": None,
            "rate_limit_desc": f"{rl.europepmc[0]}/{rl.europepmc[1]}s",
        },
        {
            "tool": "orcid",
            "display_name": "ORCID",
            "configured": True,
            "requires_key": False,
            "monthly_limit": None,
            "rate_limit_desc": f"{rl.orcid[0]}/{rl.orcid[1]}s",
        },
    ]

    tools: list[dict[str, Any]] = []
    for td in tool_defs:
        name = str(td["tool"])
        stats = summary_by_tool.get(name, {})
        month_data = monthly.get(name, {})

        level = "ok"
        if not td["configured"]:
            level = "error"
        elif td["monthly_limit"] and month_data.get("requests", 0) >= td["monthly_limit"]:
            level = "error"

        tools.append({
            "tool": name,
            "display_name": td["display_name"],
            "configured": td["configured"],
            "requires_key": td["requires_key"],
            "level": level,
            "requests_this_month": month_data.get("requests", 0),
            "monthly_limit": td["monthly_limit"],
            "rate_limit_desc": td["rate_limit_desc"],
            "requests_this_period": stats.get("total_requests", 0),
            "successful": stats.get("successful", 0),
            "failed": stats.get("failed", 0),
            "error_rate": stats.get("error_rate", 0.0),
            "avg_elapsed": stats.get("avg_elapsed"),
            "last_used": stats.get("last_used"),
        })

    polite_email = cfg.services.polite_email or ""
    return {"tools": tools, "polite_email": polite_email}


class BackendResponse(BaseModel):
    backend: str


class BackendUpdate(BaseModel):
    backend: str

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        if v not in ("claude-cli", "vllm", "api"):
            msg = f"Invalid backend: '{v}'. Must be 'claude-cli', 'vllm', or 'api'."
            raise ValueError(msg)
        return v


@app.get("/api/config/backend", response_model=BackendResponse)
async def get_backend() -> BackendResponse:
    """Get the current LLM backend."""
    from research_mentor.config import load_config

    config = load_config()
    return BackendResponse(backend=config.backend)


@app.put("/api/config/backend", response_model=BackendResponse)
async def set_backend(request: BackendUpdate) -> BackendResponse:
    """Switch the LLM backend at runtime (takes effect on next chat request)."""
    from research_mentor.config import set_runtime_override

    set_runtime_override("backend", request.backend)
    logger.info("Backend switched to: {}", request.backend)
    return BackendResponse(backend=request.backend)


class ClaudeCLIConfigResponse(BaseModel):
    model: str
    effort: str


_VALID_MODEL_ALIASES = {"sonnet", "opus", "haiku"}
# Full model ID pattern: claude-<family>-<version>, e.g. claude-sonnet-4-6
_FULL_MODEL_PATTERN = re.compile(r"^claude-(sonnet|opus|haiku)-[\d]+(-[\d]+)+$")


class ClaudeCLIConfigUpdate(BaseModel):
    model: str | None = None
    effort: str | None = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # Strip optional [1m] suffix for validation
        base = re.sub(r"\[.*\]$", "", v)
        if base in _VALID_MODEL_ALIASES:
            return v
        if _FULL_MODEL_PATTERN.match(base):
            return v
        msg = (
            f"Invalid model: '{v}'. "
            "Use an alias (sonnet, opus, haiku) or a full model ID (e.g. claude-sonnet-4-6)."
        )
        raise ValueError(msg)

    @field_validator("effort")
    @classmethod
    def validate_effort(cls, v: str | None) -> str | None:
        if v is not None and v not in ("low", "medium", "high"):
            msg = f"Invalid effort: '{v}'. Must be 'low', 'medium', or 'high'."
            raise ValueError(msg)
        return v


@app.get("/api/config/claude-cli", response_model=ClaudeCLIConfigResponse)
async def get_claude_cli_config() -> ClaudeCLIConfigResponse:
    """Get the current Claude CLI model configuration."""
    from research_mentor.config import load_config

    config = load_config()
    return ClaudeCLIConfigResponse(
        model=config.claude_cli.model,
        effort=config.claude_cli.effort,
    )


@app.put("/api/config/claude-cli", response_model=ClaudeCLIConfigResponse)
async def set_claude_cli_config(request: ClaudeCLIConfigUpdate) -> ClaudeCLIConfigResponse:
    """Update Claude CLI model configuration at runtime."""
    from research_mentor.config import ClaudeCLIConfig, load_config, set_runtime_override

    config = load_config()
    current = config.claude_cli
    updated = ClaudeCLIConfig(
        model=request.model or current.model,
        effort=request.effort or current.effort,
        timeout=current.timeout,
        max_parallel_processes=current.max_parallel_processes,
        max_requests_per_minute=current.max_requests_per_minute,
    )
    set_runtime_override("claude_cli", updated)
    logger.info("Claude CLI config updated: model={}, effort={}", updated.model, updated.effort)
    return ClaudeCLIConfigResponse(model=updated.model, effort=updated.effort)


# --- vLLM model management ---


class VLLMConfigResponse(BaseModel):
    active_model: str
    served_model: str | None
    profiles: dict[str, Any]
    switch_status: str
    switch_error: str | None
    vllm_reachable: bool
    # "running", "stopped", "not_created", or None (no container runtime)
    container_status: str | None = None


class VLLMSwitchRequest(BaseModel):
    model: str


class VLLMSwitchStatusResponse(BaseModel):
    status: str
    model: str | None
    error: str | None


@app.get("/api/config/vllm", response_model=VLLMConfigResponse)
async def get_vllm_config() -> VLLMConfigResponse:
    """Get vLLM model configuration, available profiles, and switch status."""
    from research_mentor.config import load_config
    from research_mentor.vllm_manager import get_available_profiles, get_switch_status

    config = load_config()
    status = get_switch_status()

    # Probe vLLM server — get actually served model
    reachable = False
    served_model: str | None = None
    try:
        import httpx

        resp = httpx.get(f"{config.vllm.api_base}/models", timeout=2.0)
        if resp.status_code == 200:
            reachable = True
            models = resp.json().get("data", [])
            if models:
                served_model = models[0].get("id")
    except Exception:
        logger.debug("Failed to reach vLLM at {}/models", config.vllm.api_base)

    # Container status
    container_status: str | None = None
    runtime = config.vllm.container_runtime
    if runtime:
        import shutil
        import subprocess

        container_name = (
            config.vllm.container_name or f"research-mentor-vllm-{config.vllm.model}"
        )
        if shutil.which(runtime):
            # Check running
            run_result = subprocess.run(
                [runtime, "inspect", "--format", "{{.State.Running}}", container_name],
                capture_output=True, text=True,
            )
            if run_result.returncode == 0 and run_result.stdout.strip() == "true":
                container_status = "running"
            elif run_result.returncode == 0:
                container_status = "stopped"
            else:
                container_status = "not_created"

    return VLLMConfigResponse(
        active_model=config.vllm.model,
        served_model=served_model,
        profiles=get_available_profiles(),
        switch_status=status["status"],
        switch_error=status.get("error"),
        vllm_reachable=reachable,
        container_status=container_status,
    )


@app.post("/api/config/vllm/switch", response_model=VLLMSwitchStatusResponse)
async def switch_vllm_model(request: VLLMSwitchRequest) -> VLLMSwitchStatusResponse:
    """Trigger a vLLM model switch. Returns immediately; poll status for progress."""
    from research_mentor.vllm_manager import SwitchStatus, get_switch_status, switch_model

    status = get_switch_status()
    if status["status"] not in (SwitchStatus.IDLE, SwitchStatus.READY, SwitchStatus.FAILED):
        from fastapi import HTTPException

        raise HTTPException(status_code=409, detail="Model switch already in progress")

    asyncio.create_task(switch_model(request.model))
    logger.info("vLLM model switch initiated: {}", request.model)
    return VLLMSwitchStatusResponse(
        status="stopping",
        model=request.model,
        error=None,
    )


class VLLMMetricsResponse(BaseModel):
    available: bool
    kv_cache_percent: float = 0.0
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    generation_tokens_total: float = 0.0
    tokens_per_second: float = 0.0
    sparkline: list[float] = Field(default_factory=list)
    session_tokens: int | None = None


_SPARKLINE_MAX_POINTS = 30


class _VLLMRateTracker:
    """Server-side tok/s rate calculator — avoids client-side timing issues."""

    __slots__ = ("_prev_tokens", "_prev_time", "_sparkline")

    def __init__(self) -> None:
        self._prev_tokens: float | None = None
        self._prev_time: float | None = None
        self._sparkline: list[float] = []

    def update(self, tokens_total: float) -> tuple[float, list[float]]:
        """Record a new sample and return (current_rate, sparkline_points)."""
        import time

        now = time.monotonic()
        prev_tokens = self._prev_tokens
        prev_time = self._prev_time
        self._prev_tokens = tokens_total
        self._prev_time = now

        if prev_tokens is None or prev_time is None:
            return 0.0, list(self._sparkline)

        elapsed = now - prev_time
        if elapsed < 0.5:
            rate = self._sparkline[-1] if self._sparkline else 0.0
            return rate, list(self._sparkline)

        delta = max(tokens_total - prev_tokens, 0.0)
        rate = delta / elapsed

        self._sparkline.append(rate)
        if len(self._sparkline) > _SPARKLINE_MAX_POINTS:
            self._sparkline = self._sparkline[-_SPARKLINE_MAX_POINTS:]

        return rate, list(self._sparkline)

    def reset(self) -> None:
        self._prev_tokens = None
        self._prev_time = None
        self._sparkline.clear()


_vllm_rate_tracker = _VLLMRateTracker()


@app.get("/api/config/vllm/metrics", response_model=VLLMMetricsResponse)
async def get_vllm_metrics(session_id: str | None = None) -> VLLMMetricsResponse:
    """Get live vLLM server metrics (KV cache, queue, tokens, tok/s sparkline).

    Proxies the Prometheus /metrics endpoint on the vLLM server and returns
    a small subset useful for the sidebar status widget. Rate calculation is
    done server-side using monotonic time for accuracy.

    Optionally includes session token usage when ``session_id`` is provided,
    avoiding a separate API call from the UI.
    """
    from research_mentor.config import load_config

    config = load_config()
    # Strip /v1 suffix — /metrics lives on the base server URL
    base_url = re.sub(r"/v1/?$", "", config.vllm.api_base)

    try:
        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{base_url}/metrics")
        if resp.status_code != 200:
            _vllm_rate_tracker.reset()
            return VLLMMetricsResponse(available=False)

        metrics: dict[str, float] = {}
        for line in resp.text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            try:
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0].split("{")[0]
                    val = float(parts[-1])
                    if name in metrics:
                        metrics[name] += val
                    else:
                        metrics[name] = val
            except (ValueError, IndexError):
                continue

        tokens_total = metrics.get("vllm:generation_tokens_total", 0.0)
        tok_s, sparkline = _vllm_rate_tracker.update(tokens_total)

        # Session token usage (optional, avoids extra API call from widget)
        session_tokens: int | None = None
        if session_id:
            from research_mentor.db.crud import get_usage_summary

            summary = await get_usage_summary("all", session_id=session_id)
            session_tokens = summary.get("total_tokens")

        return VLLMMetricsResponse(
            available=True,
            kv_cache_percent=metrics.get("vllm:kv_cache_usage_perc", 0.0) * 100,
            num_requests_running=int(metrics.get("vllm:num_requests_running", 0)),
            num_requests_waiting=int(metrics.get("vllm:num_requests_waiting", 0)),
            generation_tokens_total=tokens_total,
            tokens_per_second=round(tok_s, 1),
            sparkline=sparkline,
            session_tokens=session_tokens,
        )
    except Exception:
        logger.debug("Failed to fetch vLLM metrics")
        _vllm_rate_tracker.reset()
        return VLLMMetricsResponse(available=False)


@app.get("/api/config/vllm/status", response_model=VLLMSwitchStatusResponse)
async def get_vllm_switch_status() -> VLLMSwitchStatusResponse:
    """Get the current vLLM model switch status."""
    from research_mentor.vllm_manager import get_switch_status

    status = get_switch_status()
    return VLLMSwitchStatusResponse(
        status=status["status"],
        model=status.get("model"),
        error=status.get("error"),
    )


# --- Services config (polite_email, Brave API key) ---


class ServicesConfigResponse(BaseModel):
    polite_email: str
    brave_search_configured: bool
    brave_search_key_preview: str  # masked: "BSA...xxxx" or ""
    tavily_configured: bool
    tavily_key_preview: str  # masked: "tvly...xxxx" or ""
    core_api_configured: bool
    core_api_key_preview: str  # masked or ""


class ServicesConfigUpdate(BaseModel):
    polite_email: str | None = None
    brave_search_api_key: str | None = None  # full key to write, or "" to clear
    tavily_api_key: str | None = None  # full key to write, or "" to clear
    core_api_key: str | None = None  # full key to write, or "" to clear

    @field_validator("polite_email")
    @classmethod
    def validate_email(cls, v: str | None) -> str | None:
        if v is not None and v and "@" not in v:
            msg = "Invalid email address."
            raise ValueError(msg)
        return v


def _mask_api_key(key: str) -> str:
    """Return masked version of API key: 'BSA...xxxx' or '' if empty."""
    if not key or len(key) < 8:
        return key
    return f"{key[:3]}...{key[-4:]}"


def _read_brave_key() -> str:
    """Read Brave Search API key from its file. Returns empty string if not configured."""
    from research_mentor.config import load_config

    cfg = load_config()
    key_path = Path(cfg.services.brave_search.api_key_file).expanduser()
    if not key_path.exists():
        return ""
    text = key_path.read_text().strip()
    return text


def _write_brave_key(key: str) -> None:
    """Write (or clear) Brave Search API key file."""
    from research_mentor.config import load_config

    cfg = load_config()
    key_path = Path(cfg.services.brave_search.api_key_file).expanduser()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if key:
        key_path.write_text(key.strip() + "\n")
        key_path.chmod(0o600)
    elif key_path.exists():
        key_path.unlink()


def _read_tavily_key() -> str:
    """Read Tavily API key from its file. Returns empty string if not configured."""
    from research_mentor.config import load_config

    cfg = load_config()
    key_path = Path(cfg.services.tavily.api_key_file).expanduser()
    if not key_path.exists():
        return ""
    text = key_path.read_text().strip()
    return text


def _write_tavily_key(key: str) -> None:
    """Write (or clear) Tavily API key file."""
    from research_mentor.config import load_config

    cfg = load_config()
    key_path = Path(cfg.services.tavily.api_key_file).expanduser()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if key:
        key_path.write_text(key.strip() + "\n")
        key_path.chmod(0o600)
    elif key_path.exists():
        key_path.unlink()


def _read_core_api_key() -> str:
    """Read CORE API key from its file. Returns empty string if not configured."""
    from research_mentor.config import load_config

    cfg = load_config()
    key_path = Path(cfg.services.core_api.api_key_file).expanduser()
    if not key_path.exists():
        return ""
    text = key_path.read_text().strip()
    return text


def _write_core_api_key(key: str) -> None:
    """Write (or clear) CORE API key file."""
    from research_mentor.config import load_config

    cfg = load_config()
    key_path = Path(cfg.services.core_api.api_key_file).expanduser()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if key:
        key_path.write_text(key.strip() + "\n")
        key_path.chmod(0o600)
    elif key_path.exists():
        key_path.unlink()


@app.get("/api/config/services", response_model=ServicesConfigResponse)
async def get_services_config() -> ServicesConfigResponse:
    """Get external services configuration."""
    from research_mentor.config import load_config

    cfg = load_config()
    brave_key = _read_brave_key()
    tavily_key = _read_tavily_key()
    core_key = _read_core_api_key()
    return ServicesConfigResponse(
        polite_email=cfg.services.polite_email,
        brave_search_configured=bool(brave_key),
        brave_search_key_preview=_mask_api_key(brave_key),
        tavily_configured=bool(tavily_key),
        tavily_key_preview=_mask_api_key(tavily_key),
        core_api_configured=bool(core_key),
        core_api_key_preview=_mask_api_key(core_key),
    )


@app.put("/api/config/services", response_model=ServicesConfigResponse)
async def set_services_config(request: ServicesConfigUpdate) -> ServicesConfigResponse:
    """Update external services configuration."""
    from research_mentor.config import load_config, update_user_config

    if request.polite_email is not None:
        update_user_config({"services": {"polite_email": request.polite_email}})
        logger.info("Polite email updated")

    if request.brave_search_api_key is not None:
        _write_brave_key(request.brave_search_api_key)
        action = "updated" if request.brave_search_api_key else "cleared"
        logger.info("Brave Search API key {}", action)

    if request.tavily_api_key is not None:
        _write_tavily_key(request.tavily_api_key)
        action = "updated" if request.tavily_api_key else "cleared"
        logger.info("Tavily API key {}", action)

    if request.core_api_key is not None:
        _write_core_api_key(request.core_api_key)
        action = "updated" if request.core_api_key else "cleared"
        logger.info("CORE API key {}", action)

    cfg = load_config()
    brave_key = _read_brave_key()
    tavily_key = _read_tavily_key()
    core_key = _read_core_api_key()
    return ServicesConfigResponse(
        polite_email=cfg.services.polite_email,
        brave_search_configured=bool(brave_key),
        brave_search_key_preview=_mask_api_key(brave_key),
        tavily_configured=bool(tavily_key),
        tavily_key_preview=_mask_api_key(tavily_key),
        core_api_configured=bool(core_key),
        core_api_key_preview=_mask_api_key(core_key),
    )


# --- Feature toggles ---


class FeaturesConfigResponse(BaseModel):
    embeddings_enabled: bool
    embeddings_model: str
    embeddings_similarity_threshold: float
    embeddings_max_results: int
    max_upload_mb: int
    log_level: str


class FeaturesConfigUpdate(BaseModel):
    embeddings_enabled: bool | None = None
    embeddings_model: str | None = None
    embeddings_similarity_threshold: float | None = None
    embeddings_max_results: int | None = None
    max_upload_mb: int | None = None
    log_level: str | None = None

    @field_validator("embeddings_model")
    @classmethod
    def validate_model(cls, v: str | None) -> str | None:
        if v is not None and "/" not in v:
            msg = "embeddings_model must be a HuggingFace model name (org/model)"
            raise ValueError(msg)
        return v

    @field_validator("embeddings_similarity_threshold")
    @classmethod
    def validate_threshold(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            msg = "similarity_threshold must be between 0.0 and 1.0."
            raise ValueError(msg)
        return v

    @field_validator("embeddings_max_results")
    @classmethod
    def validate_max_results(cls, v: int | None) -> int | None:
        if v is not None and not (1 <= v <= 50):
            msg = "max_results must be between 1 and 50."
            raise ValueError(msg)
        return v

    @field_validator("max_upload_mb")
    @classmethod
    def validate_max_upload(cls, v: int | None) -> int | None:
        if v is not None and not (1 <= v <= 1024):
            msg = "max_upload_mb must be between 1 and 1024."
            raise ValueError(msg)
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str | None) -> str | None:
        if v is not None and v not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            msg = "log_level must be DEBUG, INFO, WARNING, or ERROR."
            raise ValueError(msg)
        return v


@app.get("/api/config/features", response_model=FeaturesConfigResponse)
async def get_features_config() -> FeaturesConfigResponse:
    """Get feature toggle configuration."""
    from research_mentor.config import load_config

    cfg = load_config()
    return FeaturesConfigResponse(
        embeddings_enabled=cfg.embeddings.enabled,
        embeddings_model=cfg.embeddings.model,
        embeddings_similarity_threshold=cfg.embeddings.similarity_threshold,
        embeddings_max_results=cfg.embeddings.max_results,
        max_upload_mb=cfg.storage.max_upload_bytes // (1024 * 1024),
        log_level=cfg.logging.level,
    )


@app.put("/api/config/features", response_model=FeaturesConfigResponse)
async def set_features_config(request: FeaturesConfigUpdate) -> FeaturesConfigResponse:
    """Update feature toggle configuration."""
    from research_mentor.config import load_config, update_user_config

    old_cfg = load_config()
    old_model = old_cfg.embeddings.model

    updates: dict[str, Any] = {}
    if request.embeddings_enabled is not None:
        updates.setdefault("embeddings", {})["enabled"] = request.embeddings_enabled
    if request.embeddings_model is not None:
        updates.setdefault("embeddings", {})["model"] = request.embeddings_model
    if request.embeddings_similarity_threshold is not None:
        updates.setdefault("embeddings", {})["similarity_threshold"] = (
            request.embeddings_similarity_threshold
        )
    if request.embeddings_max_results is not None:
        updates.setdefault("embeddings", {})["max_results"] = request.embeddings_max_results
    if request.max_upload_mb is not None:
        updates.setdefault("storage", {})["max_upload_bytes"] = request.max_upload_mb * 1024 * 1024
    if request.log_level is not None:
        updates.setdefault("logging", {})["level"] = request.log_level

    if updates:
        update_user_config(updates)
        logger.info("Features config updated: {}", list(updates.keys()))

    # If model changed, load it to detect dimension, then rebuild tables
    new_model = request.embeddings_model
    if new_model and new_model != old_model:
        from research_mentor.embeddings import rebuild_embedding_tables, reset_model

        reset_model()  # force reload with new model on next use
        logger.info("Embedding model changed: {} → {} — rebuilding tables", old_model, new_model)
        # dimension=None → auto-detected from the newly loaded model
        await rebuild_embedding_tables()

    cfg = load_config()
    return FeaturesConfigResponse(
        embeddings_enabled=cfg.embeddings.enabled,
        embeddings_model=cfg.embeddings.model,
        embeddings_similarity_threshold=cfg.embeddings.similarity_threshold,
        embeddings_max_results=cfg.embeddings.max_results,
        max_upload_mb=cfg.storage.max_upload_bytes // (1024 * 1024),
        log_level=cfg.logging.level,
    )


# --- Vision config ---


class VisionConfigResponse(BaseModel):
    enabled: bool
    backend: str
    local_model: str
    local_model_cached: bool
    local_model_path: str
    local_model_size_mb: int
    local_model_download_size_mb: int  # approximate download size
    api_provider: str
    api_base_url: str
    api_model: str
    api_key_configured: bool
    api_key_preview: str


class VisionConfigUpdate(BaseModel):
    enabled: bool | None = None
    backend: str | None = None
    api_provider: str | None = None
    api_base_url: str | None = None
    api_model: str | None = None
    api_key: str | None = None  # full key to write, or "" to clear

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str | None) -> str | None:
        if v is not None and v not in ("auto", "local", "claude-cli", "api"):
            msg = f"Invalid vision backend: '{v}'. Must be 'auto', 'local', 'claude-cli', or 'api'."
            raise ValueError(msg)
        return v


def _read_vision_api_key() -> str:
    """Read vision API key from its file. Returns empty string if not configured."""
    from research_mentor.config import load_config as _lc

    cfg = _lc()
    key_path = Path(cfg.vision.api_key_file).expanduser()
    if not key_path.exists():
        return ""
    return key_path.read_text().strip()


def _write_vision_api_key(key: str) -> None:
    """Write (or clear) vision API key file."""
    from research_mentor.config import load_config as _lc

    cfg = _lc()
    key_path = Path(cfg.vision.api_key_file).expanduser()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if key:
        key_path.write_text(key.strip() + "\n")
        key_path.chmod(0o600)
    elif key_path.exists():
        key_path.unlink()


def _build_vision_response() -> VisionConfigResponse:
    """Build VisionConfigResponse from current config + model state."""
    from research_mentor.config import load_config as _lc
    from research_mentor.vision import (
        is_local_model_cached,
        local_model_cache_path,
        local_model_cache_size_bytes,
    )

    cfg = _lc()
    api_key = _read_vision_api_key()
    size_bytes = local_model_cache_size_bytes()

    return VisionConfigResponse(
        enabled=cfg.vision.enabled,
        backend=cfg.vision.backend,
        local_model=cfg.vision.model,
        local_model_cached=is_local_model_cached(),
        local_model_path=local_model_cache_path(),
        local_model_size_mb=size_bytes // (1024 * 1024) if size_bytes else 0,
        local_model_download_size_mb=4000,  # ~4GB approximate
        api_provider=cfg.vision.api_provider,
        api_base_url=cfg.vision.api_base_url,
        api_model=cfg.vision.api_model,
        api_key_configured=bool(api_key),
        api_key_preview=_mask_api_key(api_key),
    )


@app.get("/api/config/vision", response_model=VisionConfigResponse)
async def get_vision_config() -> VisionConfigResponse:
    """Get vision / artifact interpretation configuration."""
    return _build_vision_response()


@app.put("/api/config/vision", response_model=VisionConfigResponse)
async def set_vision_config(request: VisionConfigUpdate) -> VisionConfigResponse:
    """Update vision / artifact interpretation configuration."""
    from research_mentor.config import update_user_config

    updates: dict[str, Any] = {}
    if request.enabled is not None:
        updates.setdefault("vision", {})["enabled"] = request.enabled
    if request.backend is not None:
        updates.setdefault("vision", {})["backend"] = request.backend
    if request.api_provider is not None:
        updates.setdefault("vision", {})["api_provider"] = request.api_provider
    if request.api_base_url is not None:
        updates.setdefault("vision", {})["api_base_url"] = request.api_base_url
    if request.api_model is not None:
        updates.setdefault("vision", {})["api_model"] = request.api_model

    if updates:
        update_user_config(updates)
        logger.info("Vision config updated")

    if request.api_key is not None:
        _write_vision_api_key(request.api_key)
        action = "updated" if request.api_key else "cleared"
        logger.info("Vision API key {}", action)

    return _build_vision_response()


# -- Vision local model download / remove --


_vision_download_in_progress = False


@app.post("/api/config/vision/local-model/download", response_model=VisionConfigResponse)
async def download_vision_local_model() -> VisionConfigResponse:
    """Download the local vision model in the background."""
    global _vision_download_in_progress

    from research_mentor.vision import is_local_model_cached

    if is_local_model_cached():
        return _build_vision_response()

    if _vision_download_in_progress:
        return _build_vision_response()

    _vision_download_in_progress = True

    async def _download() -> None:
        global _vision_download_in_progress
        try:
            from research_mentor.vision import _get_model
            await asyncio.to_thread(_get_model)
            logger.info("Vision local model download complete")
        except Exception:
            logger.exception("Vision local model download failed")
        finally:
            _vision_download_in_progress = False

    asyncio.create_task(_download())
    return _build_vision_response()


@app.delete("/api/config/vision/local-model", response_model=VisionConfigResponse)
async def remove_vision_local_model() -> VisionConfigResponse:
    """Remove the cached local vision model from disk."""
    from research_mentor.vision import remove_local_model_cache

    await asyncio.to_thread(remove_local_model_cache)
    return _build_vision_response()


@app.get("/api/config/vision/local-model/status")
async def vision_local_model_status() -> dict[str, Any]:
    """Check local model download status (for polling during download)."""
    from research_mentor.vision import (
        is_local_model_cached,
        local_model_cache_path,
        local_model_cache_size_bytes,
    )

    return {
        "cached": is_local_model_cached(),
        "downloading": _vision_download_in_progress,
        "path": local_model_cache_path(),
        "size_mb": local_model_cache_size_bytes() // (1024 * 1024),
    }


# --- GROBID config ---


class GrobidConfigResponse(BaseModel):
    enabled: bool
    url: str
    container_name: str
    container_image: str
    available: bool


class GrobidConfigUpdate(BaseModel):
    enabled: bool | None = None
    url: str | None = None


@app.get("/api/config/grobid", response_model=GrobidConfigResponse)
async def get_grobid_config() -> GrobidConfigResponse:
    """Return current GROBID configuration and availability."""

    from research_mentor.config import load_config as _lc
    from research_mentor.grobid import is_grobid_available

    cfg = _lc()
    if cfg.grobid.enabled:
        available = await is_grobid_available(cfg.grobid.url)
    else:
        available = False
    return GrobidConfigResponse(
        enabled=cfg.grobid.enabled,
        url=cfg.grobid.url,
        container_name=cfg.grobid.container_name,
        container_image=cfg.grobid.container_image,
        available=available,
    )


@app.put("/api/config/grobid", response_model=GrobidConfigResponse)
async def set_grobid_config(request: GrobidConfigUpdate) -> GrobidConfigResponse:
    """Update GROBID configuration."""
    from research_mentor.config import update_user_config

    updates: dict[str, Any] = {}
    if request.enabled is not None:
        updates["enabled"] = request.enabled
    if request.url is not None:
        updates["url"] = request.url

    if updates:
        update_user_config({"grobid": updates})
        logger.info("GROBID config updated: {}", updates)

    return await get_grobid_config()


# --- Restore defaults ---


class DefaultsResponse(BaseModel):
    message: str


@app.post("/api/config/defaults", response_model=DefaultsResponse)
async def restore_defaults(clear_api_keys: bool = False) -> DefaultsResponse:
    """Reset all settings to bundled defaults.

    Removes ~/.research-mentor/config.toml and clears runtime overrides.
    Optionally clears API key files too.
    """
    from research_mentor.config import load_config, reset_user_config

    reset_user_config()

    if clear_api_keys:
        cfg = load_config()
        key_path = Path(cfg.services.brave_search.api_key_file).expanduser()
        if key_path.exists():
            key_path.unlink()
        vision_key_path = Path(cfg.vision.api_key_file).expanduser()
        if vision_key_path.exists():
            vision_key_path.unlink()

    logger.info("Settings restored to defaults (clear_api_keys={})", clear_api_keys)
    return DefaultsResponse(message="All settings restored to defaults.")


# --- API backend config ---


class APIConfigResponse(BaseModel):
    provider: str
    base_url: str
    model: str
    api_key_configured: bool
    api_key_preview: str


class APIConfigUpdate(BaseModel):
    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None  # full key to write, or "" to clear


def _read_api_key() -> str:
    """Read API key from its file. Returns empty string if not configured."""
    from research_mentor.config import load_config as _lc

    cfg = _lc()
    key_path = Path(cfg.api.api_key_file).expanduser()
    if not key_path.exists():
        return ""
    return key_path.read_text().strip()


def _write_api_key(key: str) -> None:
    """Write (or clear) API key file."""
    from research_mentor.config import load_config as _lc

    cfg = _lc()
    key_path = Path(cfg.api.api_key_file).expanduser()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if key:
        key_path.write_text(key.strip() + "\n")
        key_path.chmod(0o600)
    elif key_path.exists():
        key_path.unlink()


@app.get("/api/config/api", response_model=APIConfigResponse)
async def get_api_config() -> APIConfigResponse:
    """Get remote API backend configuration."""
    from research_mentor.config import load_config as _lc

    cfg = _lc()
    api_key = _read_api_key()
    return APIConfigResponse(
        provider=cfg.api.provider,
        base_url=cfg.api.base_url,
        model=cfg.api.model,
        api_key_configured=bool(api_key),
        api_key_preview=_mask_api_key(api_key),
    )


@app.put("/api/config/api", response_model=APIConfigResponse)
async def set_api_config(request: APIConfigUpdate) -> APIConfigResponse:
    """Update remote API backend configuration."""
    from research_mentor.config import load_config as _lc
    from research_mentor.config import update_user_config

    updates: dict[str, Any] = {}
    if request.provider is not None:
        updates.setdefault("llm", {}).setdefault("api", {})["provider"] = request.provider
    if request.base_url is not None:
        updates.setdefault("llm", {}).setdefault("api", {})["base_url"] = request.base_url
    if request.model is not None:
        updates.setdefault("llm", {}).setdefault("api", {})["model"] = request.model

    if updates:
        update_user_config(updates)
        logger.info("API config updated")

    if request.api_key is not None:
        _write_api_key(request.api_key)
        action = "updated" if request.api_key else "cleared"
        logger.info("API key {}", action)

    cfg = _lc()
    api_key = _read_api_key()
    return APIConfigResponse(
        provider=cfg.api.provider,
        base_url=cfg.api.base_url,
        model=cfg.api.model,
        api_key_configured=bool(api_key),
        api_key_preview=_mask_api_key(api_key),
    )


# --- Safety config endpoints ---


class SafetyConfigResponse(BaseModel):
    query_heuristic_review: bool
    query_llm_review: bool


class SafetyConfigUpdate(BaseModel):
    query_heuristic_review: bool | None = None
    query_llm_review: bool | None = None


@app.get("/api/config/safety", response_model=SafetyConfigResponse)
async def get_safety_config() -> SafetyConfigResponse:
    """Get safety and privacy configuration."""
    from research_mentor.config import load_config as _lc

    cfg = _lc()
    return SafetyConfigResponse(
        query_heuristic_review=cfg.safety.query_heuristic_review,
        query_llm_review=cfg.safety.query_llm_review,
    )


@app.put("/api/config/safety", response_model=SafetyConfigResponse)
async def set_safety_config(request: SafetyConfigUpdate) -> SafetyConfigResponse:
    """Update safety and privacy configuration."""
    from research_mentor.config import load_config as _lc
    from research_mentor.config import update_user_config

    updates: dict[str, Any] = {}
    if request.query_heuristic_review is not None:
        updates.setdefault("safety", {})["query_heuristic_review"] = (
            request.query_heuristic_review
        )
    if request.query_llm_review is not None:
        updates.setdefault("safety", {})["query_llm_review"] = (
            request.query_llm_review
        )

    if updates:
        update_user_config(updates)
        logger.info("Safety config updated: {}", updates)

    cfg = _lc()
    return SafetyConfigResponse(
        query_heuristic_review=cfg.safety.query_heuristic_review,
        query_llm_review=cfg.safety.query_llm_review,
    )


@app.get("/api/safety/blocked-queries")
async def get_blocked_queries(
    since: float = 0.0,
    limit: int = 50,
) -> dict[str, Any]:
    """Return recently blocked outgoing queries.

    Args:
        since: Unix timestamp — only return blocks after this time (for polling).
        limit: Max results (default 50).
    """
    from research_mentor.db.connection import get_db

    try:
        async with get_db() as db:
            if since > 0:
                # Convert unix timestamp to SQLite datetime for comparison
                from datetime import UTC, datetime

                since_dt = datetime.fromtimestamp(since, tz=UTC).strftime(
                    "%Y-%m-%d %H:%M:%S",
                )
                cursor = await db.execute(
                    "SELECT id, timestamp, tool, query, block_type, reason "
                    "FROM safety_blocked_queries "
                    "WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?",
                    (since_dt, limit),
                )
            else:
                cursor = await db.execute(
                    "SELECT id, timestamp, tool, query, block_type, reason "
                    "FROM safety_blocked_queries "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
            rows = [dict(r) for r in await cursor.fetchall()]
    except Exception:
        # Table may not exist yet (pre-migration) — fall back to in-memory
        from research_mentor.tools.query_safety_review import get_blocked_queries

        rows = get_blocked_queries(since=since)[:limit]

    return {"blocked": rows, "total": len(rows)}


@app.get("/api/safety/blocked-queries/new")
async def get_new_blocked_queries(since: float = 0.0) -> dict[str, Any]:
    """Fast in-memory check for new blocks since a timestamp.

    Used by the chat UI to poll for blocks and show notifications.
    Returns count + latest block (if any) for popup display.
    """
    from research_mentor.tools.query_safety_review import get_blocked_queries

    recent = get_blocked_queries(since=since)
    return {
        "count": len(recent),
        "latest": recent[-1] if recent else None,
    }


# --- Usage endpoints ---


@app.get("/api/usage/summary")
async def get_usage_summary_endpoint(
    period: str = "month",
    backend: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Get aggregated token usage for a period."""
    from research_mentor.db.crud import get_usage_summary

    return await get_usage_summary(
        period,
        backend=backend,
        provider=provider,
        model=model,
        project_id=project_id,
        session_id=session_id,
    )


@app.get("/api/usage/session-tools")
async def get_session_tool_usage_endpoint(session_id: str) -> list[dict[str, Any]]:
    """Get per-tool request counts for a single session."""
    from research_mentor.db.crud import get_session_tool_usage

    return await get_session_tool_usage(session_id)


@app.get("/api/usage/timeseries")
async def get_usage_timeseries_endpoint(
    period: str = "month",
    granularity: str = "day",
) -> list[dict[str, Any]]:
    """Get usage over time for charts."""
    from research_mentor.db.crud import get_usage_timeseries

    return await get_usage_timeseries(period, granularity)


@app.get("/api/usage/by-model")
async def get_usage_by_model_endpoint(
    period: str = "month",
) -> list[dict[str, Any]]:
    """Get token usage broken down by model."""
    from research_mentor.db.crud import get_usage_by_model

    return await get_usage_by_model(period)


@app.get("/api/usage/overview")
async def get_usage_overview_endpoint(
    period: str = "month",
    backend: str | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Get usage overview with conversation counts and avg latency."""
    from research_mentor.db.crud import get_usage_overview

    return await get_usage_overview(period, backend=backend, project_id=project_id)


@app.get("/api/usage/by-project")
async def get_usage_by_project_endpoint(
    period: str = "month",
) -> list[dict[str, Any]]:
    """Get token usage broken down by project."""
    from research_mentor.db.crud import get_usage_by_project

    return await get_usage_by_project(period)


@app.get("/api/usage/page-data")
async def get_usage_page_data(period: str = "month") -> dict[str, Any]:
    """All data for the GenAI Utilization page in one call.

    Returns overview, timeseries, by-project, by-model, and active backend.
    Avoids multiple concurrent HTTP requests from the UI.
    """
    from research_mentor.config import load_config as _lc
    from research_mentor.db.crud import (
        get_tool_usage_summary,
        get_usage_by_model,
        get_usage_by_project,
        get_usage_by_purpose,
        get_usage_overview,
        get_usage_timeseries,
        list_projects,
    )

    granularity = "hour" if period in ("day", "today") else "day"

    overview = await get_usage_overview(period)
    timeseries = await get_usage_timeseries(period, granularity)
    by_project = await get_usage_by_project(period)
    by_model = await get_usage_by_model(period)
    by_purpose = await get_usage_by_purpose(period)
    tool_usage = await get_tool_usage_summary(period)
    backend = _lc().backend

    # Resolve project titles server-side (thin UI layer)
    proj_result = await list_projects(page_size=500)
    project_titles = {p["id"]: p["title"] for p in proj_result["data"]}

    return {
        "overview": overview,
        "timeseries": timeseries,
        "by_project": by_project,
        "by_model": by_model,
        "by_purpose": by_purpose,
        "tool_usage": tool_usage,
        "backend": backend,
        "project_titles": project_titles,
    }


@app.get("/api/usage/budget")
async def get_budget_status_endpoint() -> dict[str, Any]:
    """Get current budget status and utilization."""
    from research_mentor.budget import get_budget_status

    return await get_budget_status()


# --- Budget config endpoints ---


class BudgetConfigResponse(BaseModel):
    enabled: bool
    period: str
    global_limit: int
    providers: dict[str, int]
    models: dict[str, int]


class BudgetConfigUpdate(BaseModel):
    enabled: bool | None = None
    period: str | None = None
    global_limit: int | None = None
    providers: dict[str, int] | None = None
    models: dict[str, int] | None = None

    @field_validator("period")
    @classmethod
    def validate_period(cls, v: str | None) -> str | None:
        if v is not None and v not in ("daily", "weekly", "monthly", "total"):
            msg = f"Invalid period: '{v}'. Must be 'daily', 'weekly', 'monthly', or 'total'."
            raise ValueError(msg)
        return v

    @field_validator("global_limit")
    @classmethod
    def validate_global_limit(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            msg = "global_limit must be >= 0 (0 = unlimited)."
            raise ValueError(msg)
        return v


@app.get("/api/config/budget", response_model=BudgetConfigResponse)
async def get_budget_config() -> BudgetConfigResponse:
    """Get current budget configuration."""
    from research_mentor.config import load_config as _lc

    cfg = _lc()
    return BudgetConfigResponse(
        enabled=cfg.budget.enabled,
        period=cfg.budget.period,
        global_limit=cfg.budget.global_limit,
        providers=cfg.budget.providers,
        models=cfg.budget.models,
    )


@app.put("/api/config/budget", response_model=BudgetConfigResponse)
async def set_budget_config(request: BudgetConfigUpdate) -> BudgetConfigResponse:
    """Update budget configuration."""
    from research_mentor.config import load_config as _lc
    from research_mentor.config import update_user_config

    updates: dict[str, Any] = {}
    if request.enabled is not None:
        updates.setdefault("budget", {})["enabled"] = request.enabled
    if request.period is not None:
        updates.setdefault("budget", {})["period"] = request.period
    if request.global_limit is not None:
        updates.setdefault("budget", {})["global_limit"] = request.global_limit
    if request.providers is not None:
        updates.setdefault("budget", {})["providers"] = request.providers
    if request.models is not None:
        updates.setdefault("budget", {})["models"] = request.models

    if updates:
        update_user_config(updates)
        logger.info("Budget config updated: {}", list(updates.keys()))

    cfg = _lc()
    return BudgetConfigResponse(
        enabled=cfg.budget.enabled,
        period=cfg.budget.period,
        global_limit=cfg.budget.global_limit,
        providers=cfg.budget.providers,
        models=cfg.budget.models,
    )


    # Usage records are now written directly to DB via ContextVar in llm.py.
    # No manual _persist_usage needed.


async def _check_project_not_draft(project_id: str) -> None:
    """Raise HTTP 409 if the project is still in draft status."""
    from fastapi import HTTPException

    project = await crud.get_project(project_id)
    if project and project.get("status") == "draft":
        raise HTTPException(
            status_code=409,
            detail="Cannot chat with a draft project. Set the project status to 'active' first.",
        )


async def _build_input_state(request: OfficeRequest) -> dict[str, Any]:
    """Build the graph input state, auto-loading missing data from the DB.

    The UI only needs to pass what it actively controls (project_id, persona,
    language, guidance_level, response_length). Everything else is loaded
    from the database so the caller doesn't need to assemble it.
    """
    from research_mentor.db.crud import (
        fetch_project_context,
        get_latest_student_assessment,
        get_student_profile,
        get_student_progress,
    )

    project_id = request.project_id

    # Auto-load project context from DB when not provided
    project_context = request.project_context
    if not project_context and project_id:
        project_context = await fetch_project_context(project_id)
        if project_context:
            logger.debug("Auto-loaded project context for project_id={}", project_id)

    # Auto-load student demographics from profile when not provided
    student_demographics = request.student_demographics
    if not student_demographics:
        profile = await get_student_profile()
        if profile:
            student_demographics = {
                "firstName": profile.get("name") or "",
                "age": profile.get("age"),
                "gradeLevel": profile.get("grade") or profile.get("college_year") or "",
                "country": profile.get("country") or "",
                "educationLevel": profile.get("education_level") or "",
                "domainExpertise": profile.get("domain_expertise", []),
                "professionalExperience": profile.get("professional_experience", []),
                "backgroundNotes": profile.get("background_notes") or "",
            }
            logger.debug("Auto-loaded student demographics from profile")

    # Auto-load student progress from DB when not provided
    student_progress = request.student_progress
    if not student_progress and project_id:
        progress = await get_student_progress(project_id)
        if progress and "error" not in progress:
            student_progress = progress
            logger.debug("Auto-loaded student progress for project_id={}", project_id)

    # Auto-load student skills from latest cross-project assessment when not provided
    student_skills = request.student_skills
    if not student_skills:
        assessment = await get_latest_student_assessment()
        if assessment:
            student_skills = {
                "overall_skill_level": assessment.get("overall_skill_level"),
                "research_maturity_score": assessment.get("research_maturity_score"),
                "growth_trajectory": assessment.get("growth_trajectory"),
                "key_strengths": assessment.get("key_strengths", []),
                "growth_areas": assessment.get("growth_areas", []),
                # Per-domain skills for progress_assessor routing
                "avg_critical_reading": assessment.get("avg_critical_reading"),
                "avg_data_analysis_skill": assessment.get("avg_data_analysis_skill"),
                "avg_experimental_design_skill": assessment.get(
                    "avg_experimental_design_skill"
                ),
                "avg_writing_skill": assessment.get("avg_writing_skill"),
                "avg_critical_thinking_skill": assessment.get(
                    "avg_critical_thinking_skill"
                ),
                "avg_time_management_skill": assessment.get(
                    "avg_time_management_skill"
                ),
                "avg_collaboration_skill": assessment.get("avg_collaboration_skill"),
                "avg_research_maturity": assessment.get("avg_research_maturity"),
            }
            logger.debug("Auto-loaded student skills from latest assessment")

    return {
        "messages": [HumanMessage(content=request.message)],
        "project_id": project_id,
        "persona": request.persona,
        "language": request.language,
        "response_length": request.response_length,
        "user_requested_guidance_level": request.user_requested_guidance_level,
        "student_progress": student_progress,
        "student_skills": student_skills,
        "project_context": project_context,
        "student_demographics": student_demographics,
    }


async def _ensure_session(request: OfficeRequest, thread_id: str, *, is_new: bool) -> None:
    """Create or touch the session row so sidebar can list it."""
    if is_new:
        await crud.create_session(
            request.project_id,
            session_id=thread_id,
            persona_id=request.persona,
            language=request.language,
        )
    else:
        await crud.update_session(thread_id)  # touches last_active_at


@app.post("/api/office", response_model=OfficeResponse)
async def office(request: OfficeRequest) -> OfficeResponse:
    """Send a message to the mentor agent and get a response.

    Provide thread_id to continue an existing conversation.
    Omit thread_id to start a new conversation (one will be generated).
    """
    import time

    from fastapi import HTTPException

    from research_mentor.agent.graph import get_mentor_graph
    from research_mentor.budget import BudgetExceededError, check_budget
    from research_mentor.config import load_config as _load_config
    from research_mentor.llm import get_call_stats, reset_call_stats, set_usage_context
    from research_mentor.tools.status import set_tool_activity_callback

    app_cfg = _load_config()

    # Budget check (only enforced for API backend)
    try:
        await check_budget(app_cfg.backend)
    except BudgetExceededError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e

    await _check_project_not_draft(request.project_id)

    graph = await get_mentor_graph()

    is_new = request.thread_id is None
    thread_id = request.thread_id or str(uuid.uuid4())
    input_state = await _build_input_state(request)

    config = {"configurable": {"thread_id": thread_id}}

    await _ensure_session(request, thread_id, is_new=is_new)

    logger.info(
        "Chat request: thread={}, project={}, persona={}",
        thread_id, request.project_id, request.persona,
    )

    # Set usage context — all LLM calls in this request tagged as "office"
    set_usage_context(
        purpose="office", session_id=thread_id, project_id=request.project_id,
    )

    # Collect tool activity for the response
    tools_used: list[ToolActivity] = []

    async def collect_tool_activity(event: dict[str, Any]) -> None:
        tools_used.append(ToolActivity(**event))

    set_tool_activity_callback(collect_tool_activity)

    reset_call_stats()
    t0 = time.monotonic()

    try:
        result: dict[str, Any] = await graph.ainvoke(input_state, config)  # type: ignore[attr-defined]
    finally:
        set_tool_activity_callback(None)

    wall_time = time.monotonic() - t0
    stats = get_call_stats()
    logger.info(
        "Graph complete | thread={} wall={:.1f}s llm_calls={} "
        "total_tokens={}in/{}out llm_time={:.1f}s",
        thread_id,
        wall_time,
        stats["calls"],
        stats["prompt_tokens"],
        stats["completion_tokens"],
        stats["elapsed"],
    )

    final_response = result.get("final_response", "")
    if not final_response:
        final_response = result.get("response", "No response generated.")

    # Increment message count and auto-generate/review session title (non-blocking)
    msg_count = await crud.increment_session_message_count(thread_id)
    asyncio.create_task(
        maybe_generate_title(thread_id, request.message, final_response, msg_count),
    )

    # Auto-trigger daily assessments (non-blocking background tasks)
    if request.project_id != "default":
        from research_mentor.assessment import (
            maybe_trigger_assessment,
            maybe_trigger_student_assessment,
        )

        asyncio.create_task(maybe_trigger_assessment(request.project_id))
        asyncio.create_task(maybe_trigger_student_assessment())

    return OfficeResponse(
        response=final_response,
        thread_id=thread_id,
        guidance_type=result.get("guidance_type"),
        persona=request.persona,
        tool_warnings=result.get("tool_warnings", []),
        tools_used=tools_used,
    )


@app.post("/api/office/stream")
async def office_stream(request: OfficeRequest) -> StreamingResponse:
    """Stream the mentor agent's response via SSE.

    SSE event format:
    - ``data: {"thread_id": "..."}``        — first event, connection confirmation
    - ``data: {"node": "planner"}``         — node progress (which node is running)
    - ``data: {"guidance_type": "..."}``    — guidance type from progress_assessor
    - ``data: {"tool_activity": {...}}``    — real-time tool call completion
    - ``data: {"tool_warnings": [...]}``    — tool errors/limits from info_gathering
    - ``data: {"token": "..."}``            — streamed token from presenter LLM
    - ``data: {"response": "..."}``         — full final response (after presenter completes)
    - ``: heartbeat``                       — SSE comment keep-alive (every 5s)
    - ``data: [DONE]``                      — stream complete

    Heartbeat mechanism prevents proxy timeouts during long graph processing
    (60-80s). Two-tier timeout: sliding window (inactivity) + global safety limit.
    """
    import asyncio
    import time

    from fastapi import HTTPException

    from research_mentor.agent.graph import get_mentor_graph
    from research_mentor.budget import BudgetExceededError, check_budget
    from research_mentor.config import load_config
    from research_mentor.llm import get_call_stats, reset_call_stats, set_usage_context
    from research_mentor.tools.status import set_tool_activity_callback

    app_config = load_config()

    # Budget check (only enforced for API backend)
    try:
        await check_budget(app_config.backend)
    except BudgetExceededError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e

    await _check_project_not_draft(request.project_id)

    graph = await get_mentor_graph()

    is_new = request.thread_id is None
    thread_id = request.thread_id or str(uuid.uuid4())
    input_state = await _build_input_state(request)

    config = {"configurable": {"thread_id": thread_id}}

    await _ensure_session(request, thread_id, is_new=is_new)

    # Heartbeat configuration
    HEARTBEAT_INTERVAL = 5          # seconds between heartbeats
    INACTIVITY_TIMEOUT = app_config.stream_inactivity_timeout
    GLOBAL_TIMEOUT = app_config.stream_global_timeout

    # Nodes to report progress on (skip internal/trivial ones)
    PROGRESS_NODES = {
        "progress_assessor", "input_safety_review", "planner", "info_gathering",
        "expert_router", "understand_starting_point", "scaffolding_guide",
        "guided_discovery", "pure_socratic", "reflection", "safety_expert",
        "ethics_expert", "communication_expert", "output_safety_validator",
        "presenter", "memory_writer",
    }

    # Set usage context — all LLM calls in this stream tagged as "office"
    set_usage_context(
        purpose="office", session_id=thread_id, project_id=request.project_id,
    )

    logger.info(
        "Stream request: thread={}, project={}, persona={}",
        thread_id, request.project_id, request.persona,
    )

    async def event_generator() -> Any:
        # Send thread_id as first event
        yield f"data: {json.dumps({'thread_id': thread_id})}\n\n"

        # Shared queue for heartbeats + tool activity events
        heartbeat_queue: asyncio.Queue[str] = asyncio.Queue()

        # Tool activity callback — pushes SSE events into the queue
        async def on_tool_activity(event: dict[str, Any]) -> None:
            sse = f"data: {json.dumps({'tool_activity': event})}\n\n"
            await heartbeat_queue.put(sse)

        set_tool_activity_callback(on_tool_activity)
        last_activity = asyncio.Event()

        async def send_heartbeats() -> None:
            silence_duration = 0
            total_duration = 0
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                total_duration += HEARTBEAT_INTERVAL

                if last_activity.is_set():
                    silence_duration = 0
                    last_activity.clear()
                else:
                    silence_duration += HEARTBEAT_INTERVAL

                await heartbeat_queue.put(": heartbeat\n\n")

                # Global timeout
                if total_duration >= GLOBAL_TIMEOUT:
                    logger.warning(
                        "Stream global timeout | thread={} total={}s", thread_id, total_duration,
                    )
                    err = "Request timeout — processing took too long"
                    await heartbeat_queue.put(
                        f"data: {json.dumps({'error': err})}\n\n"
                    )
                    await heartbeat_queue.put("data: [DONE]\n\n")
                    return

                # Sliding window inactivity timeout
                if silence_duration >= INACTIVITY_TIMEOUT:
                    logger.warning(
                        "Stream inactivity timeout | thread={} silence={}s",
                        thread_id, silence_duration,
                    )
                    err = "Request timeout — no response from system"
                    await heartbeat_queue.put(
                        f"data: {json.dumps({'error': err})}\n\n"
                    )
                    await heartbeat_queue.put("data: [DONE]\n\n")
                    return

        heartbeat_task = asyncio.create_task(send_heartbeats())

        try:
            reset_call_stats()
            t0 = time.monotonic()
            response_text = ""

            async for event in graph.astream_events(input_state, config, version="v2"):  # type: ignore[attr-defined]
                last_activity.set()

                kind = event.get("event", "")
                name = event.get("name", "")
                metadata = event.get("metadata", {})
                langgraph_node = metadata.get("langgraph_node", "")

                # --- Node progress ---
                if kind == "on_chain_start" and name in PROGRESS_NODES:
                    yield f"data: {json.dumps({'node': name})}\n\n"

                # --- Guidance type from progress_assessor ---
                elif kind == "on_chain_end" and name == "progress_assessor":
                    output = event.get("data", {}).get("output", {})
                    guidance = output.get("guidance_type", "")
                    if guidance:
                        yield f"data: {json.dumps({'guidance_type': guidance})}\n\n"

                # --- Tool warnings from info_gathering ---
                elif kind == "on_chain_end" and name == "info_gathering":
                    output = event.get("data", {}).get("output", {})
                    warnings = output.get("tool_warnings", [])
                    if warnings:
                        yield f"data: {json.dumps({'tool_warnings': warnings})}\n\n"

                # --- Token streaming from presenter LLM ---
                elif kind == "on_chat_model_stream" and langgraph_node == "presenter":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk:
                        content = getattr(chunk, "content", "")
                        if content:
                            response_text += content
                            yield f"data: {json.dumps({'token': content})}\n\n"

                # --- Full response when presenter completes ---
                elif kind == "on_chain_end" and name == "presenter":
                    output = event.get("data", {}).get("output", {})
                    final = output.get("final_response", "")
                    if final:
                        # Only send full response if we didn't stream tokens
                        if not response_text:
                            response_text = final
                            yield f"data: {json.dumps({'response': final})}\n\n"
                        else:
                            # Send as metadata — client already has tokens
                            yield f"data: {json.dumps({'response_complete': True})}\n\n"

                # --- Drain heartbeat queue between events ---
                while not heartbeat_queue.empty():
                    hb = heartbeat_queue.get_nowait()
                    yield hb
                    if "[DONE]" in hb:
                        return

            wall_time = time.monotonic() - t0
            stats = get_call_stats()
            logger.info(
                "Stream complete | thread={} wall={:.1f}s llm_calls={} "
                "total_tokens={}in/{}out llm_time={:.1f}s",
                thread_id, wall_time, stats["calls"],
                stats["prompt_tokens"], stats["completion_tokens"], stats["elapsed"],
            )

            # Usage records are written automatically via ContextVar

            # Increment message count and auto-generate/review session title (non-blocking)
            msg_count = await crud.increment_session_message_count(thread_id)
            if response_text:
                asyncio.create_task(
                    maybe_generate_title(
                        thread_id, request.message, response_text, msg_count,
                    ),
                )

            # Auto-trigger daily assessments (non-blocking)
            if request.project_id != "default":
                from research_mentor.assessment import (
                    maybe_trigger_assessment,
                    maybe_trigger_student_assessment,
                )

                asyncio.create_task(maybe_trigger_assessment(request.project_id))
                asyncio.create_task(maybe_trigger_student_assessment())

            yield "data: [DONE]\n\n"

        except Exception:
            logger.exception("Stream error | thread={}", thread_id)
            yield f"data: {json.dumps({'error': 'Internal server error'})}\n\n"
            yield "data: [DONE]\n\n"

        finally:
            heartbeat_task.cancel()
            set_tool_activity_callback(None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Static file mount for React UI ──────────────────────────────────────────
# MUST come after all API route registrations so /api/* takes priority.
_ui_static = Path(__file__).parent / "ui_static"
if _ui_static.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_ui_static, html=True), name="ui")
