"""SQLite connection pool via aiosqlitepool.

Provides a pool of pre-configured connections (WAL mode, busy_timeout,
foreign keys, sqlite_vec extension) to avoid per-request connection overhead
and reduce "database is locked" contention.

Uses ``aiosqlitepool.SQLiteConnectionPool`` for proper lifecycle management,
health checks, acquisition timeouts, and clean shutdown.

Usage:
    from research_mentor.db.connection import get_db, init_db

    await init_db()  # run once at startup (creates tables, seeds data, opens pool)

    async with get_db() as db:
        row = await db.execute_fetchone("SELECT * FROM projects WHERE id = ?", (pid,))
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path

import aiosqlite
import sqlite_vec
from aiosqlitepool import SQLiteConnectionPool
from loguru import logger

from research_mentor.config import get_data_dir, load_config
from research_mentor.db.schema import MIGRATIONS, SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Pool configuration
# ---------------------------------------------------------------------------

# How long SQLite waits for a write lock before raising "database is locked"
# (milliseconds).  5 seconds is generous enough for typical write contention.
_BUSY_TIMEOUT_MS = 5000

# How long get_db() waits for a free connection before raising an error.
_ACQUISITION_TIMEOUT_S = 10

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

# Module-level DB path (set during init_db)
_db_path: Path | None = None

# Per-task DB path override (set via use_db context manager).
# When set, get_db() uses this instead of the module-level _db_path.
# Python copies contextvars on asyncio.Task creation, so each task
# sees its own value — no races between concurrent eval scenarios.
_db_path_override: ContextVar[Path | None] = ContextVar("_db_path_override", default=None)

# Connection pools keyed by resolved path (supports both main DB and use_db overrides).
_pools: dict[Path, SQLiteConnectionPool] = {}
_pool_lock = asyncio.Lock()


class SchemaMismatchError(Exception):
    """Raised when the database schema version doesn't match the code."""

    def __init__(self, current: int, target: int) -> None:
        self.current = current
        self.target = target
        super().__init__(
            f"Database schema is v{current}, but v{target} is required. "
            f"Run 'research-mentor migrate' to update your database."
        )


def _resolve_db_path() -> Path:
    """Resolve the database file path from config."""
    config = load_config()
    db_path = Path(config.db_path)
    if not db_path.is_absolute():
        db_path = get_data_dir() / db_path
    return db_path


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------


def _make_connection_factory(
    db_path: Path,
) -> Callable[[], Awaitable[aiosqlite.Connection]]:
    """Return an async callable that creates a configured aiosqlite connection."""

    async def _factory() -> aiosqlite.Connection:
        conn = await aiosqlite.connect(db_path)
        await conn.enable_load_extension(True)
        await conn.load_extension(sqlite_vec.loadable_path())
        await conn.enable_load_extension(False)
        await conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = aiosqlite.Row
        return conn

    return _factory


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------


def _pool_size() -> int:
    """Read pool size from config (default 4)."""
    return load_config().db_pool_size


async def _create_pool(db_path: Path) -> SQLiteConnectionPool:
    """Create a connection pool for *db_path*."""
    size = _pool_size()
    pool = SQLiteConnectionPool(
        connection_factory=_make_connection_factory(db_path),
        pool_size=size,
        acquisition_timeout=_ACQUISITION_TIMEOUT_S,
    )
    logger.debug("Connection pool opened (max {} connections) for {}", size, db_path)
    return pool


async def _get_pool(db_path: Path) -> SQLiteConnectionPool:
    """Get or create the pool for *db_path* (thread-safe via asyncio.Lock)."""
    pool = _pools.get(db_path)
    if pool is not None:
        return pool
    async with _pool_lock:
        # Double-check after acquiring lock
        pool = _pools.get(db_path)
        if pool is not None:
            return pool
        pool = await _create_pool(db_path)
        _pools[db_path] = pool
        return pool


async def close_pool(db_path: Path | None = None) -> None:
    """Close all connections in the pool for *db_path* (or the main DB).

    Safe to call multiple times.  Called automatically during server shutdown.
    """
    path = db_path or _db_path
    if path is None:
        return
    pool = _pools.pop(path, None)
    if pool is None:
        return
    await pool.close()
    logger.debug("Connection pool closed for {}", path)


async def close_all_pools() -> None:
    """Close every open pool.  Called during server shutdown."""
    paths = list(_pools.keys())
    for path in paths:
        await close_pool(path)


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


async def get_schema_version(db_path: Path | None = None) -> tuple[int, int]:
    """Read the current and target schema versions.

    Returns:
        (current_version, target_version) tuple.
        current_version is 0 if the DB has no schema_version table.
    """
    path = db_path or _resolve_db_path()
    current_version = 0
    async with aiosqlite.connect(path) as db:
        try:
            cursor = await db.execute("SELECT MAX(version) FROM schema_version")
            row = await cursor.fetchone()
            if row and row[0] is not None:
                current_version = row[0]
        except aiosqlite.OperationalError:
            pass  # Table doesn't exist yet — version 0
    return current_version, SCHEMA_VERSION


async def init_db(db_path: Path | None = None) -> Path:
    """Initialize the database and open the connection pool.

    - If no DB file exists: creates fresh (all migrations + seed data).
    - If DB exists and schema is current: ready to use.
    - If DB exists but schema is outdated: raises SchemaMismatchError.

    Always sets the module-level ``_db_path`` so ``get_db()`` can find it.
    The connection pool is created lazily on first ``get_db()`` call, or
    eagerly here when no explicit ``db_path`` is passed (server startup).

    For callers that need per-task isolation, call ``init_db(path)``
    then wrap work in ``use_db(path)`` — the ``ContextVar`` scopes all
    ``get_db()`` calls to the override path and cleans up on exit.

    Args:
        db_path: Override database path (tests/evals).

    Returns:
        The resolved database path.

    Raises:
        SchemaMismatchError: If the DB exists but needs migration.
    """
    global _db_path
    resolved = db_path or _resolve_db_path()
    _db_path = resolved

    if not resolved.exists():
        logger.trace("Creating fresh database at {}", resolved)
        await _create_fresh_db(resolved)
        if db_path is None:
            # Server startup — open pool eagerly
            await _get_pool(resolved)
        return resolved

    # DB exists — check version
    current, target = await get_schema_version(resolved)
    if current < target:
        raise SchemaMismatchError(current, target)

    if db_path is None:
        # Server startup — open pool eagerly
        await _get_pool(resolved)
    logger.trace("Database ready (schema v{})", current)
    return resolved


async def _create_fresh_db(db_path: Path) -> None:
    """Create a new database with all migrations applied and seed data loaded."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(db_path) as db:
        await db.enable_load_extension(True)
        await db.load_extension(sqlite_vec.loadable_path())
        await db.enable_load_extension(False)
        await db.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")

        for version in range(1, SCHEMA_VERSION + 1):
            if version in MIGRATIONS:
                logger.trace("Applying migration v{}", version)
                await db.executescript(MIGRATIONS[version])
                await db.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (version,),
                )
                await db.commit()

        await _seed_personas(db)

    logger.trace("Database created (schema v{})", SCHEMA_VERSION)


def set_db_path(db_path: Path) -> None:
    """Set the module-level DB path without running migrations.

    Used after ``run_migration()`` completes to point the server at the
    migrated database.
    """
    global _db_path
    _db_path = db_path


async def _seed_personas(db: aiosqlite.Connection) -> None:
    """Seed teaching personas from bundled JSON."""
    personas_path = Path(__file__).parent.parent / "seed_data" / "personas.json"
    if not personas_path.exists():
        logger.warning("Personas seed file not found at {}", personas_path)
        return

    with open(personas_path) as f:
        data = json.load(f)

    personas = data.get("personas", [])
    count = 0
    for p in personas:
        await db.execute(
            """
            INSERT OR IGNORE INTO teaching_personas
                (persona_id, full_name, category, birth_year, death_year,
                 brief_description, biography, notable_works, key_achievements,
                 teaching_style_notes, persona_type, is_active, display_order)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                p["persona_id"],
                p["full_name"],
                p["category"],
                p.get("birth_year"),
                p.get("death_year"),
                p.get("brief_description"),
                p.get("biography"),
                json.dumps(p.get("notable_works", [])),
                json.dumps(p.get("key_achievements", [])),
                p.get("teaching_style_notes"),
                p.get("persona_type", "historical"),
                1 if p.get("is_active", True) else 0,
                p.get("display_order", 0),
            ),
        )
        count += 1

    await db.commit()
    logger.trace("Seeded {} teaching personas", count)


@asynccontextmanager
async def use_db(db_path: Path) -> AsyncIterator[None]:
    """Set the DB path for the current async task scope.

    All ``get_db()`` calls within this scope (including transitive calls
    through crud functions) use *db_path* instead of the module-level global.

    Python copies ``ContextVar`` values when creating ``asyncio.Task``
    objects, so concurrent tasks each see their own path — no races.

    Usage (per-task DB isolation)::

        async with use_db(tmp_dir / "eval.db"):
            await generate_assessment(project_id)
    """
    token = _db_path_override.set(db_path)
    try:
        yield
    finally:
        _db_path_override.reset(token)
        # Clean up pool for override paths when scope ends
        await close_pool(db_path)


@asynccontextmanager
async def get_db() -> AsyncIterator[aiosqlite.Connection]:
    """Get an async database connection from the pool.

    Checks the per-task ``_db_path_override`` first (set via ``use_db``),
    then falls back to the module-level ``_db_path`` (set via ``init_db``).

    The connection is returned to the pool when the context manager exits.

    Usage:
        async with get_db() as db:
            cursor = await db.execute("SELECT ...")
    """
    path = _db_path_override.get() or _db_path
    if path is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    pool = await _get_pool(path)
    async with pool.connection() as conn:
        yield conn
