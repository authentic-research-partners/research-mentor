"""Outgoing query safety reviewer.

Reviews LLM-generated search queries before they are sent to external APIs.
Catches accidental leaks of personal information, institutional names,
unpublished research specifics, or other sensitive data that the user
wouldn't want shared with third-party search providers.

This is NOT adversarial defense (a compromised model could bypass its own
reviewer). It catches **unintentional** leaks — the LLM carelessly including
context details in a search query.

Called by search_web() and search_and_enrich() before external API calls.

Fail-closed: if the LLM reviewer crashes, the query is blocked.
Two independent toggles in ``[safety]``:
- ``query_heuristic_review`` — regex patterns (default: on)
- ``query_llm_review`` — LLM structured call (default: on)
Power-mode users can disable each layer via the Settings UI.
"""

from __future__ import annotations

import re
import time
from collections import deque
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from research_mentor.config import load_config

# ---------------------------------------------------------------------------
# Blocked query log (in-memory ring buffer, exposed via API)
# ---------------------------------------------------------------------------

_MAX_BLOCKED = 100


class BlockedQuery(BaseModel):
    """Record of a blocked outgoing query."""

    timestamp: float = Field(description="Unix timestamp")
    tool: str = Field(description="Tool that tried to send the query")
    query: str = Field(description="The blocked query (truncated to 200 chars)")
    block_type: str = Field(description="'heuristic' or 'llm' or 'llm_crash'")
    reason: str = Field(description="Why it was blocked")


_blocked_log: deque[BlockedQuery] = deque(maxlen=_MAX_BLOCKED)


def _record_block(
    tool: str, query: str, block_type: str, reason: str,
) -> None:
    """Append a blocked query to the in-memory log and persist to DB."""
    entry = BlockedQuery(
        timestamp=time.time(),
        tool=tool,
        query=query[:200],
        block_type=block_type,
        reason=reason,
    )
    _blocked_log.append(entry)

    # Persist to DB (fire-and-forget, don't block the caller)
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_persist_block(entry))
    except RuntimeError:
        logger.debug("No running event loop — skipping safety-block DB persist")


async def _persist_block(entry: BlockedQuery) -> None:
    """Write a blocked query record to the database."""
    try:
        from research_mentor.db.connection import get_db

        async with get_db() as db:
            await db.execute(
                "INSERT INTO safety_blocked_queries "
                "(tool, query, block_type, reason) VALUES (?, ?, ?, ?)",
                (entry.tool, entry.query, entry.block_type, entry.reason),
            )
            await db.commit()
    except Exception:
        logger.debug("Failed to persist blocked query to DB")


def get_blocked_queries(*, since: float = 0.0) -> list[dict[str, Any]]:
    """Return blocked queries, optionally filtered by timestamp.

    Used by the API endpoint and the Settings UI.
    """
    items = [b for b in _blocked_log if b.timestamp > since]
    return [b.model_dump() for b in items]


def get_blocked_count() -> int:
    """Total number of blocked queries in the ring buffer."""
    return len(_blocked_log)

# ---------------------------------------------------------------------------
# Heuristic pre-filter (fast, no LLM call needed)
# ---------------------------------------------------------------------------

# Patterns that should never appear in a search query
_SUSPICIOUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}"), "email address"),
    (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "phone number"),
    (re.compile(r"/(?:home|Users|tmp|var|etc)/\S+"), "file path"),
    (re.compile(r"(?:api[_-]?key|password|secret|token)\s*[:=]\s*\S+", re.I), "credential"),
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "API key (sk-...)"),
    (re.compile(r"ssh-(?:rsa|ed25519)\s"), "SSH key"),
]


def _heuristic_check(query: str) -> str | None:
    """Return a reason string if the query contains obviously sensitive data, else None."""
    for pattern, description in _SUSPICIOUS_PATTERNS:
        if pattern.search(query):
            return description
    return None


# ---------------------------------------------------------------------------
# LLM-based review (catches subtle leaks heuristics miss)
# ---------------------------------------------------------------------------


class QuerySafetyResult(BaseModel):
    """Structured output from the query safety reviewer."""

    safe: bool = Field(description="True if the query is safe to send to external APIs")
    reason: str = Field(
        max_length=500,
        description="Brief explanation of why the query is safe or unsafe",
    )


_REVIEW_PROMPT = """\
You are a privacy reviewer. You will receive a search query that is about to be \
sent to an external academic search API or web search engine.

Decide whether this query is SAFE to send externally. A safe query contains only \
general research terms, scientific concepts, or topic keywords.

Flag as UNSAFE if the query contains:
- A person's full name (first + last) combined with identifying context
- A specific school, lab, or institution name tied to a person
- Unpublished results, specific data values, or proprietary methodology details
- Personal details (age, grade, location, medical conditions)
- Anything that reads like conversation content pasted into a query rather than \
a genuine search

A query like "water filtration efficiency activated carbon" is SAFE — it's a \
normal research topic search.
A query like "Sarah Chen 10th grade Lincoln High water filtration project" is \
UNSAFE — it identifies a specific student.

Single institution or researcher names alone (e.g., "MIT carbon nanotube research") \
are SAFE — these are normal academic searches.

Be concise. Most queries are safe.
"""


async def review_outgoing_query(query: str, tool_name: str) -> bool:
    """Review a query before it is sent to an external API.

    Returns True if safe to proceed, False if the query should be blocked.
    Blocked queries are logged as warnings with the reason.

    Fail-closed: if the LLM reviewer crashes, the query is blocked.
    This is intentional — safety failures should not silently bypass review.
    Users can disable each layer independently via config/UI.
    """
    cfg = load_config()

    # Layer 1: fast heuristic check (regex, no LLM call)
    if cfg.safety.query_heuristic_review:
        heuristic_reason = _heuristic_check(query)
        if heuristic_reason is not None:
            _record_block(tool_name, query, "heuristic", heuristic_reason)
            logger.warning(
                "Query blocked by heuristic ({}): tool={}, query={!r}",
                heuristic_reason, tool_name, query[:200],
            )
            return False

    # Layer 2: LLM review for subtler leaks
    if not cfg.safety.query_llm_review:
        return True

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from research_mentor.llm import structured_call

        result = await structured_call(
            QuerySafetyResult,
            [
                SystemMessage(content=_REVIEW_PROMPT),
                HumanMessage(content=f"Query: {query}"),
            ],
            thinking="off",
        )

        if not result.safe:
            _record_block(tool_name, query, "llm", result.reason)
            logger.warning(
                "Query blocked by LLM reviewer: tool={}, reason={}, query={!r}",
                tool_name, result.reason, query[:200],
            )
            return False

        return True

    except Exception:
        # Fail-closed: reviewer crash → block the query
        _record_block(tool_name, query, "llm_crash", "LLM reviewer crashed")
        logger.warning(
            "Query safety review crashed, blocking query (fail-closed): "
            "tool={}, query={!r}",
            tool_name, query[:200],
        )
        return False
