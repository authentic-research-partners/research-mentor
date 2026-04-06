"""Standardized tool status reporting.

Every external tool (web search, academic search, etc.) returns a ToolStatus
alongside its results so that callers can distinguish "searched and found nothing"
from "search failed" and surface actionable messages to the user.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

ToolStatusLevel = Literal["ok", "degraded", "error"]
ToolErrorCategory = Literal[
    "none",          # no error
    "rate_limit",    # 429 / quota exhausted
    "auth",          # 401/403 / invalid key
    "timeout",       # request timed out
    "unavailable",   # not configured (missing key, missing package)
    "network",       # connection / HTTP error
    "safety",        # query blocked by safety review
]


class ToolStatus(BaseModel):
    """Standardized status from an external tool invocation."""

    source: str = Field(description="Tool name (e.g. 'Brave Search', 'OpenAlex')")
    level: ToolStatusLevel = Field(description="ok / degraded / error")
    category: ToolErrorCategory = Field(default="none", description="Error category")
    message: str = Field(
        default="",
        description="User-facing message (empty when level=ok)",
    )

    @staticmethod
    def ok(source: str) -> ToolStatus:
        return ToolStatus(source=source, level="ok")

    @staticmethod
    def error(
        source: str, category: ToolErrorCategory, message: str,
    ) -> ToolStatus:
        return ToolStatus(
            source=source, level="error", category=category, message=message,
        )

    @staticmethod
    def degraded(
        source: str, category: ToolErrorCategory, message: str,
    ) -> ToolStatus:
        return ToolStatus(
            source=source, level="degraded", category=category, message=message,
        )


# ---------------------------------------------------------------------------
# Real-time tool activity callback (ContextVar — set per-request)
# ---------------------------------------------------------------------------

ToolActivityCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]

_tool_activity_callback: ContextVar[ToolActivityCallback | None] = ContextVar(
    "_tool_activity_callback", default=None,
)


def set_tool_activity_callback(cb: ToolActivityCallback | None) -> None:
    """Set a callback for the current async context to receive tool activity events.

    Called by streaming endpoints before invoking the graph.  The callback
    receives dicts like ``{"tool": "openalex", "success": True, "elapsed": 0.8}``.
    """
    _tool_activity_callback.set(cb)


# ---------------------------------------------------------------------------
# Shared helpers — DRY tool usage recording & error result construction
# ---------------------------------------------------------------------------


async def record_tool_usage(
    tool: str,
    *,
    query: str | None = None,
    results: list[dict[str, Any]] | None = None,
    success: bool | None = None,
    error_category: str | None = None,
    elapsed: float,
) -> None:
    """Fire-and-forget recording of an external tool API call.

    When *results* is provided and *success* is None, automatically detects
    errors by scanning for ``"error"`` keys and extracting the ToolStatus
    category.  Callers that already know success/error_category can pass them
    directly.

    If a tool activity callback is set (via :func:`set_tool_activity_callback`),
    it is invoked with the event data for real-time UI streaming.
    """
    if success is None and results is not None:
        has_error = any("error" in r for r in results)
        if has_error:
            for r in results:
                ts = r.get("tool_status")
                if ts and ts.level != "ok":
                    error_category = ts.category
                    break
        success = not has_error
    elif success is None:
        success = True

    # DB write (fire-and-forget, must not block callback)
    try:
        from research_mentor.db.crud import store_tool_usage

        await store_tool_usage(
            tool,
            query=query[:200] if query else None,
            success=success,
            error_category=error_category,
            elapsed_seconds=round(elapsed, 3),
        )
    except Exception:
        logger.debug("Failed to record tool usage for {}", tool)

    # Notify real-time listener (SSE stream) — always runs
    try:
        cb = _tool_activity_callback.get(None)
        if cb is not None:
            await cb({
                "tool": tool,
                "success": success,
                "elapsed": round(elapsed, 3),
            })
    except Exception:
        logger.debug("Failed to notify tool activity callback for {}", tool)


def make_error_result(
    source: str,
    category: ToolErrorCategory,
    message: str,
) -> list[dict[str, Any]]:
    """Build the standard single-element error result list for a failed tool call."""
    return [{
        "error": message,
        "source": source,
        "tool_status": ToolStatus.error(source, category, message),
    }]
