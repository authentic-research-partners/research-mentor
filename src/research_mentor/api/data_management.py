"""Data management API: export, import, reset, backup/restore."""

from __future__ import annotations

import json as json_mod
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, UploadFile
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel

from research_mentor.db import crud

router = APIRouter(prefix="/api", tags=["data-management"])

# Max decompressed size for ZIP imports: 500 MB.  Our export ZIPs contain
# JSON + artifact files; anything larger than this is suspicious.
_MAX_IMPORT_DECOMPRESSED_BYTES = 500 * 1024 * 1024


async def _build_export_zip(project_id: str | None = None) -> bytes:
    """Build an export ZIP in memory. Shared by the export endpoint and pre-import backup."""
    import io
    import zipfile

    from research_mentor.config import load_config

    projects_result = await crud.list_projects(page_size=1000)
    projects: list[dict[str, Any]] = projects_result["data"]
    if project_id:
        projects = [p for p in projects if p["id"] == project_id]

    exported: list[dict[str, Any]] = []
    for proj in projects:
        pid = proj["id"]
        proj_data: dict[str, Any] = {
            "project": proj,
            "milestones": (await crud.list_milestones(pid, page_size=1000))["data"],
            "sessions": (await crud.list_sessions(pid, page_size=1000))["data"],
            "assessments": (await crud.list_assessments(pid, page_size=1000))["data"],
            "artifacts": (await crud.list_artifacts(pid, page_size=1000))["data"],
            "memories": (await crud.get_memories_for_project(pid, page_size=1000))["data"],
        }
        await _attach_session_messages(proj_data)
        exported.append(proj_data)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.json", json_mod.dumps(exported, indent=2, default=str))
        config = load_config()
        artifacts_root = config.storage.get_artifacts_path()
        if artifacts_root.exists():
            for file_path in artifacts_root.rglob("*"):
                if file_path.is_file():
                    arc_name = "artifacts/" + str(file_path.relative_to(artifacts_root))
                    zf.write(file_path, arc_name)

    return buf.getvalue()


@router.get("/export")
async def export_data(
    project_id: str | None = Query(default=None, description="Export only a specific project"),
) -> Response:
    """Export all data as a ZIP: database JSON + uploaded artifact files."""
    from datetime import datetime

    zip_bytes = await _build_export_zip(project_id)
    stamp = datetime.now().strftime("%Y-%m-%d")
    logger.info("Exported data as ZIP")

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=research-mentor-export-{stamp}.zip",
        },
    )


async def _attach_session_messages(proj_data: dict[str, Any]) -> None:
    """Attach LangGraph checkpoint messages to each session."""
    from research_mentor.config import get_data_dir

    checkpoints_file = get_data_dir() / "checkpoints.db"
    if not checkpoints_file.exists():
        return

    import aiosqlite

    async with aiosqlite.connect(str(checkpoints_file)) as db:
        db.row_factory = aiosqlite.Row
        for session in proj_data["sessions"]:
            thread_id = session["session_id"]
            try:
                cursor = await db.execute(
                    """
                    SELECT checkpoint
                    FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                    """,
                    (thread_id,),
                )
                row = await cursor.fetchone()
                if row and row["checkpoint"]:
                    checkpoint = json_mod.loads(row["checkpoint"])
                    channel_values = checkpoint.get("channel_values", {})
                    messages = channel_values.get("messages", [])
                    session["messages"] = messages
            except Exception:
                logger.debug("Could not read checkpoints for session {}", thread_id)


async def _reset_all() -> Path:
    """Wipe DB, checkpoints, and artifacts, then re-init. Returns new DB path."""
    import shutil

    from research_mentor.config import load_config
    from research_mentor.db.checkpointer import reset_checkpointer
    from research_mentor.db.connection import _db_path as _current_db_path
    from research_mentor.db.connection import _resolve_db_path, close_pool, init_db

    await reset_checkpointer()

    # Close the connection pool before deleting the DB file —
    # otherwise pooled connections hold references to the deleted file.
    db_file = _current_db_path or _resolve_db_path()
    await close_pool(db_file)

    config = load_config()
    # Use the active DB path (may be a test temp dir) rather than config default
    db_file = _current_db_path or _resolve_db_path()
    checkpoints_file = db_file.parent / "checkpoints.db"

    for f in (db_file, checkpoints_file):
        if not f.exists():
            continue
        for suffix in ("", "-wal", "-shm"):
            p = f.parent / (f.name + suffix)
            if p.exists():
                p.unlink()
                logger.debug("Deleted {}", p)

    artifacts_dir = config.storage.get_artifacts_path()
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Deleted artifact files at {}", artifacts_dir)

    # Re-init at the same path (preserves test isolation)
    result = await init_db(db_path=db_file)
    logger.info("Database reset. Re-initialized at {}", result)
    return result


@router.post("/reset")
async def reset_database() -> dict[str, str]:
    """Wipe the database, checkpoints, and uploaded artifacts. Start fresh."""
    db_path = await _reset_all()
    return {"status": "ok", "message": f"Database reset. Re-initialized at {db_path}"}


class ImportResponse(BaseModel):
    status: str
    projects: int
    sessions: int
    milestones: int
    assessments: int
    artifacts: int
    memories: int
    messages_restored: int
    backup_path: str | None


@router.post("/import", response_model=ImportResponse)
async def import_data(file: UploadFile) -> ImportResponse:
    """Import data from an export ZIP (produced by GET /api/export).

    Saves a backup ZIP of current data first, then resets the database,
    checkpoints, and artifacts, and imports all data from the uploaded ZIP.
    """
    import asyncio
    import io
    import uuid
    import zipfile
    from datetime import datetime

    import aiosqlite

    from research_mentor.config import get_data_dir, load_config
    from research_mentor.db.migration import BACKUP_DIR_NAME

    # Read and validate ZIP
    contents = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(contents))
    except zipfile.BadZipFile:
        raise_http_error(400, "Invalid ZIP file.")

    if "data.json" not in zf.namelist():
        raise_http_error(400, "ZIP missing data.json — not a valid export archive.")

    # ZIP bomb protection: reject if total decompressed size exceeds cap
    total_uncompressed = sum(info.file_size for info in zf.infolist())
    if total_uncompressed > _MAX_IMPORT_DECOMPRESSED_BYTES:
        raise_http_error(
            400,
            f"ZIP decompressed size too large ({total_uncompressed} bytes, "
            f"max {_MAX_IMPORT_DECOMPRESSED_BYTES}). Refusing import.",
        )

    try:
        data: list[dict[str, Any]] = json_mod.loads(zf.read("data.json"))
    except (json_mod.JSONDecodeError, UnicodeDecodeError) as e:
        raise_http_error(400, f"Cannot parse data.json: {e}")

    if not isinstance(data, list):
        raise_http_error(400, "data.json must be a JSON array of project bundles.")

    # Save a backup ZIP of current data before replacing
    backup_zip_path: str | None = None
    try:
        existing_projects = await crud.list_projects(page_size=1)
        has_data = existing_projects["total"] > 0
    except Exception:
        logger.debug("Failed to check existing projects before import")
        has_data = False

    if has_data:
        from research_mentor.db.connection import _db_path as _current_db_path

        backup_zip_bytes = await _build_export_zip()
        backup_dir = (
            (_current_db_path.parent if _current_db_path else get_data_dir())
            / BACKUP_DIR_NAME
        )
        backup_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_file = backup_dir / f"pre-import-{stamp}.zip"
        await asyncio.to_thread(backup_file.write_bytes, backup_zip_bytes)
        backup_zip_path = str(backup_file)
        logger.info("Saved pre-import backup to {}", backup_file.name)

    # Reset everything and re-init fresh DB
    await _reset_all()

    from research_mentor.db.connection import _db_path as _current_db_path

    config = load_config()
    artifacts_root = config.storage.get_artifacts_path()
    # Use the active DB path's parent (may be test temp dir)
    db_parent = _current_db_path.parent if _current_db_path else get_data_dir()
    checkpoints_file = db_parent / "checkpoints.db"

    counts = {
        "projects": 0, "sessions": 0, "milestones": 0,
        "assessments": 0, "artifacts": 0, "memories": 0,
        "messages_restored": 0,
    }

    from research_mentor.db.connection import get_db

    async with get_db() as db:
        for bundle in data:
            proj = bundle.get("project", {})
            pid = proj.get("id")
            if not pid:
                continue

            # Project
            await db.execute(
                """INSERT INTO projects
                   (id, title, research_question, status, start_date, end_date,
                    metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, proj.get("title", ""), proj.get("research_question", ""),
                 proj.get("status", "draft"), proj.get("start_date"),
                 proj.get("end_date"), _json_str(proj.get("metadata", {})),
                 proj.get("created_at"), proj.get("updated_at")),
            )
            counts["projects"] += 1

            # Milestones
            for m in bundle.get("milestones", []):
                await db.execute(
                    """INSERT INTO milestones
                       (id, project_id, title, description, status, start_date,
                        due_date, completed_at, display_order, metadata,
                        created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (m.get("id"), pid, m.get("title", ""), m.get("description"),
                     m.get("status", "pending"), m.get("start_date"),
                     m.get("due_date"), m.get("completed_at"),
                     m.get("display_order", 0), _json_str(m.get("metadata", {})),
                     m.get("created_at"), m.get("updated_at")),
                )
                counts["milestones"] += 1

            # Sessions
            for s in bundle.get("sessions", []):
                await db.execute(
                    """INSERT INTO sessions
                       (session_id, project_id, title, persona_id, is_active,
                        language, metadata, created_at, last_active_at,
                        title_source, message_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (s.get("session_id"), pid, s.get("title"),
                     s.get("persona_id"), s.get("is_active", 1),
                     s.get("language", "en"), _json_str(s.get("metadata", {})),
                     s.get("created_at"), s.get("last_active_at"),
                     s.get("title_source", "auto"), s.get("message_count", 0)),
                )
                counts["sessions"] += 1

            # Assessments
            for a in bundle.get("assessments", []):
                await db.execute(
                    """INSERT INTO project_assessments
                       (id, project_id, assessment_date, health_score, issues,
                        recommended_actions, general_notes, metadata,
                        created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (a.get("id"), pid, a.get("assessment_date"),
                     a.get("health_score"), _json_str(a.get("issues", [])),
                     _json_str(a.get("recommended_actions", [])),
                     a.get("general_notes"), _json_str(a.get("metadata", {})),
                     a.get("created_at"), a.get("updated_at")),
                )
                counts["assessments"] += 1

            # Artifacts (metadata — files restored from ZIP below)
            for art in bundle.get("artifacts", []):
                aid = art.get("id")
                fname = art.get("file_name", "")
                # Rewrite file_path to local artifacts root
                file_path = str(artifacts_root / pid / aid / fname) if aid else ""
                await db.execute(
                    """INSERT INTO artifacts
                       (id, project_id, artifact_type, file_name, file_path,
                        mime_type, file_size_bytes, description, extracted_text,
                        metadata, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (aid, pid, art.get("artifact_type", ""),
                     fname, file_path, art.get("mime_type", ""),
                     art.get("file_size_bytes", 0), art.get("description"),
                     art.get("extracted_text"), _json_str(art.get("metadata", {})),
                     art.get("created_at"), art.get("updated_at")),
                )
                counts["artifacts"] += 1

            # Memories
            for mem in bundle.get("memories", []):
                await db.execute(
                    """INSERT INTO conversation_memories
                       (id, project_id, memory_text, memory_type, importance,
                        tags, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (mem.get("id"), pid, mem.get("memory_text", ""),
                     mem.get("memory_type", "general"),
                     mem.get("importance", 5), _json_str(mem.get("tags", [])),
                     mem.get("created_at")),
                )
                counts["memories"] += 1

        await db.commit()

    # Restore artifact files from ZIP
    for name in zf.namelist():
        if not name.startswith("artifacts/") or name.endswith("/"):
            continue
        relative = name[len("artifacts/"):]
        dest = artifacts_root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(name))

    # Restore checkpoint messages
    sessions_with_messages = [
        s for bundle in data
        for s in bundle.get("sessions", [])
        if s.get("messages")
    ]
    if sessions_with_messages:
        async with aiosqlite.connect(str(checkpoints_file)) as cp_db:
            await cp_db.execute(
                """CREATE TABLE IF NOT EXISTS checkpoints (
                       thread_id TEXT NOT NULL,
                       checkpoint_ns TEXT NOT NULL DEFAULT '',
                       checkpoint_id TEXT NOT NULL,
                       parent_checkpoint_id TEXT,
                       type TEXT,
                       checkpoint BLOB,
                       metadata BLOB,
                       PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                   )"""
            )
            await cp_db.execute(
                """CREATE TABLE IF NOT EXISTS writes (
                       thread_id TEXT NOT NULL,
                       checkpoint_ns TEXT NOT NULL DEFAULT '',
                       checkpoint_id TEXT NOT NULL,
                       task_id TEXT NOT NULL,
                       idx INTEGER NOT NULL,
                       channel TEXT NOT NULL,
                       type TEXT,
                       value BLOB,
                       PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id,
                                    task_id, idx)
                   )"""
            )
            for s in sessions_with_messages:
                checkpoint_data = json_mod.dumps({
                    "channel_values": {"messages": s["messages"]},
                })
                await cp_db.execute(
                    """INSERT OR REPLACE INTO checkpoints
                       (thread_id, checkpoint_ns, checkpoint_id, checkpoint)
                       VALUES (?, '', ?, ?)""",
                    (s["session_id"], uuid.uuid4().hex, checkpoint_data),
                )
                counts["messages_restored"] += len(s["messages"])
            await cp_db.commit()

    zf.close()
    logger.info("Imported {} projects from ZIP", counts["projects"])

    return ImportResponse(status="ok", backup_path=backup_zip_path, **counts)


def _json_str(value: Any) -> str:
    """Ensure a value is stored as a JSON string (not a parsed object)."""
    if isinstance(value, str):
        return value
    return json_mod.dumps(value, default=str)


class BackupResponse(BaseModel):
    filename: str
    size_bytes: int
    size_human: str
    created_at: str
    pre_version: int | None


def _human_size(size_bytes: int) -> str:
    """Format bytes as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.0f} {unit}" if unit == "B" else f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} TB"


@router.get("/backups", response_model=list[BackupResponse])
async def get_backups() -> list[BackupResponse]:
    """List database backup files created by the migration system."""
    from research_mentor.config import get_data_dir
    from research_mentor.db.migration import BACKUP_DIR_NAME, list_backups

    backups_dir = get_data_dir() / BACKUP_DIR_NAME
    backups = list_backups(backups_dir)
    return [
        BackupResponse(
            filename=b.filename,
            size_bytes=b.size_bytes,
            size_human=_human_size(b.size_bytes),
            created_at=b.created_at,
            pre_version=b.pre_version,
        )
        for b in backups
    ]


class MigrateResponse(BaseModel):
    status: str
    from_version: int
    to_version: int
    backup_path: str | None
    error: str | None = None


@router.post("/migrate", response_model=MigrateResponse)
async def migrate_database() -> MigrateResponse:
    """Run pending database migrations.

    Creates a backup, applies migrations to a copy, verifies each one,
    and swaps the copy into place. The original database is never modified.
    """
    from pathlib import Path

    from research_mentor.config import get_data_dir, load_config
    from research_mentor.db.connection import init_db, set_db_path
    from research_mentor.db.migration import MigrationError, run_migration

    config = load_config()
    data_dir = get_data_dir()
    db_path = Path(config.db_path)
    if not db_path.is_absolute():
        db_path = data_dir / db_path

    if not db_path.exists():
        raise_http_error(404, "No database found. Start fresh with 'research-mentor init'.")

    try:
        result = await run_migration(db_path)
    except MigrationError as e:
        raise_http_error(409, str(e))

    if not result.success:
        return MigrateResponse(
            status="error",
            from_version=result.from_version,
            to_version=result.to_version,
            backup_path=None,
            error=result.error,
        )

    # Re-initialize connection to the migrated DB
    set_db_path(db_path)
    # Run init_db to confirm the migrated DB is valid and set up properly
    await init_db(db_path=db_path)

    logger.info(
        "Database migrated v{} -> v{}, backup at {}",
        result.from_version, result.to_version, result.backup_path,
    )

    return MigrateResponse(
        status="ok",
        from_version=result.from_version,
        to_version=result.to_version,
        backup_path=str(result.backup_path) if result.backup_path else None,
    )


class RestoreRequest(BaseModel):
    filename: str


class RestoreResponse(BaseModel):
    status: str
    restored_from: str
    schema_version: int
    previous_backup: str | None


@router.post("/restore-backup", response_model=RestoreResponse)
async def restore_backup(body: RestoreRequest) -> RestoreResponse:
    """Restore the database from a backup file.

    Backs up the current database first (so restore is reversible),
    then replaces it with the selected backup. Runs any pending
    migrations if the backup is from an older schema version.
    """
    from pathlib import Path

    from research_mentor.config import get_data_dir, load_config
    from research_mentor.db.connection import init_db, set_db_path
    from research_mentor.db.migration import MigrationError
    from research_mentor.db.migration import restore_backup as do_restore

    config = load_config()
    data_dir = get_data_dir()
    db_path = Path(config.db_path)
    if not db_path.is_absolute():
        db_path = data_dir / db_path

    try:
        prev_backup, schema_version = await do_restore(db_path, body.filename)
    except MigrationError as e:
        raise_http_error(409, str(e))

    # Re-initialize connection to the restored DB
    set_db_path(db_path)
    await init_db(db_path=db_path)

    logger.info("Restored database from {}", body.filename)

    return RestoreResponse(
        status="ok",
        restored_from=body.filename,
        schema_version=schema_version,
        previous_backup=prev_backup.name if prev_backup else None,
    )


def raise_http_error(status_code: int, detail: str) -> None:
    """Raise an HTTP error — extracted to keep endpoint functions clean."""
    from fastapi import HTTPException

    raise HTTPException(status_code=status_code, detail=detail)
