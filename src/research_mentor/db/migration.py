"""Safe database migration engine (Sqitch-inspired, SQLite-adapted).

Copy-on-write migration: never mutates the original database file.
Instead, copies it, applies migrations to the copy, verifies, and swaps.
The original becomes a timestamped backup.

Usage (CLI):
    from research_mentor.db.migration import get_migration_status, run_migration

    status = await get_migration_status(db_path)
    result = await run_migration(db_path)
"""

from __future__ import annotations

import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import sqlite_vec
from loguru import logger

from research_mentor.db.schema import MIGRATIONS, SCHEMA_VERSION, VERIFY

BACKUP_DIR_NAME = "backups"
BACKUP_MAX_AGE_DAYS = 30
_BACKUP_FILENAME_RE = re.compile(
    r"^(?P<stem>.+)\.pre-v(?P<version>\d+)\.(?P<ts>\d{8}-\d{6})$"
)


@dataclass
class MigrationStatus:
    """Current migration state of the database."""

    current_version: int
    target_version: int
    pending_count: int
    db_path: Path


@dataclass
class MigrationResult:
    """Outcome of a migration run."""

    success: bool
    from_version: int
    to_version: int
    backup_path: Path | None
    error: str | None = None


@dataclass
class BackupInfo:
    """Metadata for a database backup file."""

    filename: str
    size_bytes: int
    created_at: str  # ISO 8601
    pre_version: int | None  # parsed from filename


class MigrationError(Exception):
    """Raised when migration fails."""


async def get_migration_status(db_path: Path) -> MigrationStatus:
    """Read the current migration status without modifying anything."""
    current_version = 0
    async with aiosqlite.connect(db_path) as db:
        try:
            cursor = await db.execute("SELECT MAX(version) FROM schema_version")
            row = await cursor.fetchone()
            if row and row[0] is not None:
                current_version = row[0]
        except aiosqlite.OperationalError:
            pass
    return MigrationStatus(
        current_version=current_version,
        target_version=SCHEMA_VERSION,
        pending_count=SCHEMA_VERSION - current_version,
        db_path=db_path,
    )


async def run_migration(db_path: Path) -> MigrationResult:
    """Run pending migrations using copy-on-write strategy.

    1. Check DB is not locked by another process
    2. Checkpoint WAL on original, then copy the .db file
    3. Apply each pending migration to the copy (with verify)
    4. Checkpoint WAL on copy, then swap: original → backup, copy → original
    """
    status = await get_migration_status(db_path)
    if status.pending_count == 0:
        return MigrationResult(
            success=True,
            from_version=status.current_version,
            to_version=status.current_version,
            backup_path=None,
        )

    from_version = status.current_version

    # 1. Check not locked
    _check_db_not_locked(db_path)

    # 2. Clean up stale working copy from interrupted previous run
    working_copy = db_path.parent / f"{db_path.name}.migrating"
    if working_copy.exists():
        working_copy.unlink()
        logger.warning("Removed stale working copy from interrupted migration")

    # 3. Checkpoint WAL on original, then copy
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shutil.copy2(db_path, working_copy)

    # 4. Apply migrations to the copy
    try:
        async with aiosqlite.connect(working_copy) as db:
            await db.enable_load_extension(True)
            await db.load_extension(sqlite_vec.loadable_path())
            await db.enable_load_extension(False)
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA foreign_keys=ON")

            for version in range(from_version + 1, SCHEMA_VERSION + 1):
                if version not in MIGRATIONS:
                    continue

                logger.info("Applying migration v{}", version)
                await _apply_migration_sql(db, MIGRATIONS[version], version)

                # Verify this migration
                verify_queries = VERIFY.get(version, [])
                await _run_verify(db, version, verify_queries)
                logger.info("Verified migration v{}", version)

        # 5. Checkpoint WAL on copy before swap
        async with aiosqlite.connect(working_copy) as db:
            await db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    except Exception as e:
        # Clean up working copy — original is untouched
        if working_copy.exists():
            working_copy.unlink()
        # Also clean up any WAL/SHM from the working copy
        for suffix in ("-wal", "-shm"):
            wal_file = working_copy.parent / f"{working_copy.name}{suffix}"
            if wal_file.exists():
                wal_file.unlink()

        error_msg = f"Migration failed at: {e}"
        logger.error(error_msg)
        return MigrationResult(
            success=False,
            from_version=from_version,
            to_version=SCHEMA_VERSION,
            backup_path=None,
            error=error_msg,
        )

    # 6. Swap: original → backup, copy → original
    backups_dir = db_path.parent / BACKUP_DIR_NAME
    backups_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup_name = f"{db_path.name}.pre-v{SCHEMA_VERSION}.{timestamp}"
    backup_path = backups_dir / backup_name

    shutil.move(str(db_path), str(backup_path))
    shutil.move(str(working_copy), str(db_path))

    # Clean up any leftover WAL/SHM files from the working copy
    for suffix in ("-wal", "-shm"):
        wal_file = db_path.parent / f"{db_path.name}{suffix}"
        if wal_file.exists():
            wal_file.unlink()

    # 7. Clean up old backups
    deleted = cleanup_old_backups(backups_dir)
    if deleted:
        logger.info("Cleaned up {} old backup(s)", len(deleted))

    return MigrationResult(
        success=True,
        from_version=from_version,
        to_version=SCHEMA_VERSION,
        backup_path=backup_path,
    )


async def _apply_migration_sql(
    db: aiosqlite.Connection, sql: str, version: int
) -> None:
    """Apply a single migration's SQL within an explicit transaction.

    Splits multi-statement SQL on ``';\\n'`` and executes each statement
    individually. This avoids ``executescript()`` which auto-commits and
    prevents proper rollback on failure.
    """
    statements = _split_sql(sql)
    if not statements:
        return

    await db.execute("BEGIN")
    try:
        for stmt in statements:
            await db.execute(stmt)
        await db.execute(
            "INSERT INTO schema_version (version) VALUES (?)", (version,)
        )
        await db.commit()
    except Exception:
        await db.rollback()
        raise


async def _run_verify(
    db: aiosqlite.Connection, version: int, queries: list[str]
) -> None:
    """Run verify queries for a migration version. Raise on failure."""
    for i, query in enumerate(queries):
        try:
            await db.execute(query)
        except Exception as e:
            raise MigrationError(
                f"Verify failed for v{version}, query {i + 1}: {e}\n"
                f"  SQL: {query}"
            ) from e


def _split_sql(sql: str) -> list[str]:
    """Split multi-statement SQL into individual statements.

    Splits on semicolons followed by a newline. This handles all migration
    SQL in this codebase correctly (no semicolons inside string literals).
    Strips comments and empty statements.
    """
    parts = sql.split(";\n")
    statements = []
    for part in parts:
        # Strip leading/trailing whitespace and comments-only fragments
        cleaned = part.strip()
        if not cleaned:
            continue
        # Skip comment-only blocks
        lines = [
            line for line in cleaned.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if not lines:
            continue
        # Re-add trailing semicolon for execution
        statements.append(cleaned.rstrip(";") + ";")
    return statements


def _check_db_not_locked(db_path: Path) -> None:
    """Probe the database for exclusive access.

    Raises MigrationError if another process (e.g. the server) holds a lock.
    Uses synchronous sqlite3 since this is a quick probe before async work.
    """
    conn = sqlite3.connect(str(db_path), timeout=1)
    try:
        conn.execute("BEGIN EXCLUSIVE")
        conn.execute("ROLLBACK")
    except sqlite3.OperationalError as e:
        raise MigrationError(
            "Database is locked — the server may be running. "
            "Stop the server before migrating."
        ) from e
    finally:
        conn.close()


def list_backups(backups_dir: Path) -> list[BackupInfo]:
    """List all backup files with metadata, sorted newest first."""
    if not backups_dir.exists():
        return []

    backups = []
    for f in backups_dir.iterdir():
        if not f.is_file():
            continue
        match = _BACKUP_FILENAME_RE.match(f.name)
        pre_version = int(match.group("version")) if match else None
        stat = f.stat()
        backups.append(
            BackupInfo(
                filename=f.name,
                size_bytes=stat.st_size,
                created_at=datetime.fromtimestamp(
                    stat.st_mtime, tz=UTC
                ).isoformat(),
                pre_version=pre_version,
            )
        )

    backups.sort(key=lambda b: b.created_at, reverse=True)
    return backups


async def restore_backup(
    db_path: Path, backup_filename: str
) -> tuple[Path, int]:
    """Restore a database from a backup file.

    1. Verify backup exists and is a valid SQLite file
    2. Back up the current DB (so restore is reversible)
    3. Copy backup → db_path
    4. If schema is behind current, run migrations on the restored DB

    Returns (backup_of_current, restored_schema_version).
    Raises MigrationError on failure.
    """
    backups_dir = db_path.parent / BACKUP_DIR_NAME
    backup_path = backups_dir / backup_filename

    if not backup_path.exists():
        raise MigrationError(f"Backup file not found: {backup_filename}")

    # Verify it's a valid SQLite file with a schema_version table
    try:
        async with aiosqlite.connect(backup_path) as db:
            cursor = await db.execute("SELECT MAX(version) FROM schema_version")
            row = await cursor.fetchone()
            backup_version = row[0] if row and row[0] is not None else 0
    except Exception as e:
        raise MigrationError(f"Backup is not a valid database: {e}") from e

    if backup_version == 0:
        raise MigrationError("Backup has no schema version — cannot restore")

    # Check not locked
    if db_path.exists():
        _check_db_not_locked(db_path)

    # Back up current DB (so restore is reversible)
    current_backup_path: Path | None = None
    if db_path.exists():
        backups_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        current_backup_name = f"{db_path.name}.pre-restore.{timestamp}"
        current_backup_path = backups_dir / current_backup_name

        # Checkpoint WAL before copying
        async with aiosqlite.connect(db_path) as db:
            await db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        shutil.copy2(db_path, current_backup_path)
        logger.info("Backed up current DB to {}", current_backup_path.name)

    # Replace current DB with backup
    if db_path.exists():
        db_path.unlink()
    # Clean up WAL/SHM from old DB
    for suffix in ("-wal", "-shm"):
        wal_file = db_path.parent / f"{db_path.name}{suffix}"
        if wal_file.exists():
            wal_file.unlink()

    shutil.copy2(backup_path, db_path)
    logger.info("Restored DB from {}", backup_filename)

    # If backup is behind current schema, run migrations
    if backup_version < SCHEMA_VERSION:
        logger.info(
            "Backup is v{}, running migrations to v{}",
            backup_version, SCHEMA_VERSION,
        )
        result = await run_migration(db_path)
        if not result.success:
            raise MigrationError(
                f"Restored DB but migration failed: {result.error}"
            )
        return current_backup_path or backup_path, SCHEMA_VERSION

    return current_backup_path or backup_path, backup_version


def cleanup_old_backups(
    backups_dir: Path, max_age_days: int = BACKUP_MAX_AGE_DAYS
) -> list[Path]:
    """Delete backup files older than max_age_days. Returns deleted paths."""
    if not backups_dir.exists():
        return []

    cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
    deleted: list[Path] = []

    for f in backups_dir.iterdir():
        if not f.is_file():
            continue
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=UTC)
        if mtime < cutoff:
            f.unlink()
            deleted.append(f)
            logger.info("Deleted old backup: {}", f.name)

    return deleted
