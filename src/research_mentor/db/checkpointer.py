"""LangGraph checkpointer backed by SQLite.

Uses langgraph's built-in AsyncSqliteSaver for persisting
agent graph state across conversations.
"""

from __future__ import annotations

from typing import Any

import aiosqlite
from loguru import logger

from research_mentor.config import get_data_dir

_checkpointer: Any = None
_conn: aiosqlite.Connection | None = None


async def get_checkpointer() -> Any:
    """Get or create the LangGraph SQLite checkpointer.

    Returns an AsyncSqliteSaver instance. Connection stays open for the
    lifetime of the process (singleton).
    """
    global _checkpointer, _conn
    if _checkpointer is not None:
        return _checkpointer

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = get_data_dir() / "checkpoints.db"
    logger.info("Initializing LangGraph checkpointer at {}", db_path)

    _conn = await aiosqlite.connect(str(db_path))
    _checkpointer = AsyncSqliteSaver(_conn)
    await _checkpointer.setup()

    return _checkpointer


async def reset_checkpointer() -> None:
    """Reset the checkpointer singleton, closing any open connection.

    Attempts a clean async close first. If the event loop has changed
    (common in pytest-xdist), falls back to closing the underlying
    sqlite3 connection directly and sending the stop sentinel to
    aiosqlite's worker thread queue, then joining the thread.
    """
    global _checkpointer, _conn
    conn = _conn
    _checkpointer = None
    _conn = None
    if conn is None:
        return
    try:
        await conn.close()
    except Exception:
        _force_close_aiosqlite(conn)
    # Ensure the worker thread has fully exited before returning,
    # even after a successful async close. Without this, the thread
    # may still be running when pytest tears down the event loop.
    if conn._thread.is_alive():
        conn._thread.join(timeout=2.0)


def _force_close_aiosqlite(conn: aiosqlite.Connection) -> None:
    """Forcefully close an aiosqlite connection when async close fails.

    Closes the raw sqlite3 connection, then sends the internal stop
    sentinel via the command queue so the worker thread exits cleanly.
    """
    from aiosqlite.core import _STOP_RUNNING_SENTINEL

    try:
        if conn._connection is not None:
            conn._connection.close()
            conn._connection = None
    except Exception:
        logger.debug("Failed to close raw sqlite3 connection during force-close")
    # Send stop sentinel with no future (None) — the worker thread
    # checks ``result is _STOP_RUNNING_SENTINEL`` and breaks.
    try:
        conn._tx.put_nowait((None, lambda: _STOP_RUNNING_SENTINEL))
    except Exception:
        logger.debug("Failed to send stop sentinel to aiosqlite worker thread")
    if conn._thread.is_alive():
        conn._thread.join(timeout=2.0)
