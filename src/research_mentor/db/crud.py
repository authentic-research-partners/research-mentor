"""CRUD operations for agent nodes and REST API.

Consolidates all DB operations needed by the agent graph and API endpoints.
All functions use the async context manager from connection.py.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import aiosqlite
from loguru import logger

from research_mentor.db.connection import get_db

# Column allowlists for dynamic UPDATE queries (defense-in-depth against SQL injection)
_PROJECT_UPDATABLE = frozenset({
    "title", "research_question", "status",
    "start_date", "end_date", "metadata",
})
_MILESTONE_UPDATABLE = frozenset({
    "title", "description", "status", "order_index", "metadata",
})
_SESSION_UPDATABLE = frozenset({
    "title", "title_source", "persona", "language", "is_active", "metadata",
})
_ARTIFACT_UPDATABLE = frozenset({
    "title", "description", "file_name", "file_path", "file_type", "file_size",
    "extracted_text", "metadata",
})


def _validate_fields(fields: dict[str, Any], allowed: frozenset[str], table: str) -> None:
    """Raise ValueError if any field name is not in the allowlist."""
    bad = set(fields) - allowed
    if bad:
        raise ValueError(f"Invalid column(s) for {table}: {bad}")


def _row_to_dict(row: aiosqlite.Row, json_fields: tuple[str, ...] = ()) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a dict, parsing JSON fields."""
    d = dict(row)
    for field in json_fields:
        if d.get(field) is not None:
            d[field] = json.loads(d[field])
        elif field in d:
            d[field] = {} if field == "metadata" else []
    if "is_active" in d:
        d["is_active"] = bool(d["is_active"])
    return d


async def get_persona_by_id(persona_id: str) -> dict[str, Any] | None:
    """Fetch a teaching persona by ID.

    Used by: planner, presenter, persona_instructions.
    """
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM teaching_personas WHERE persona_id = ?",
            (persona_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "personaId": row["persona_id"],
            "fullName": row["full_name"],
            "category": row["category"],
            "birthYear": row["birth_year"],
            "deathYear": row["death_year"],
            "briefDescription": row["brief_description"],
            "biography": row["biography"],
            "notableWorks": json.loads(row["notable_works"] or "[]"),
            "keyAchievements": json.loads(row["key_achievements"] or "[]"),
            "teachingStyleNotes": row["teaching_style_notes"],
            "personaType": row["persona_type"],
            "isActive": bool(row["is_active"]),
            "displayOrder": row["display_order"],
        }


async def update_student_attributes(
    project_id: str,
    courses_taken: list[str] | None = None,
    technical_skills: list[str] | None = None,
    research_interests: list[str] | None = None,
    llm_info: dict[str, str] | None = None,
) -> None:
    """Update discovered student attributes.

    Used by: student_discovery node.
    Merges with existing attributes (doesn't overwrite).
    """
    async with get_db() as db:
        # Get current metadata
        cursor = await db.execute(
            "SELECT metadata FROM projects WHERE id = ?", (project_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            logger.warning("Project {} not found for attribute update", project_id)
            return

        metadata = json.loads(row["metadata"] or "{}")
        discovered = metadata.get("discovered_attributes", {})

        if courses_taken:
            existing = set(discovered.get("courses_taken", []))
            existing.update(courses_taken)
            discovered["courses_taken"] = sorted(existing)

        if technical_skills:
            existing = set(discovered.get("technical_skills", []))
            existing.update(technical_skills)
            discovered["technical_skills"] = sorted(existing)

        if research_interests:
            existing = set(discovered.get("research_interests", []))
            existing.update(research_interests)
            discovered["research_interests"] = sorted(existing)

        if llm_info:
            discovered["last_updated_by"] = llm_info

        metadata["discovered_attributes"] = discovered
        await db.execute(
            "UPDATE projects SET metadata = ?, updated_at = datetime('now') WHERE id = ?",
            (json.dumps(metadata), project_id),
        )
        await db.commit()
    logger.info("Updated student attributes for project {}", project_id)


async def get_student_progress(project_id: str) -> dict[str, Any]:
    """Get project progress summary.

    Used by: progress_lookup tool.
    """
    async with get_db() as db:
        # Get project info
        cursor = await db.execute(
            "SELECT title, research_question, status, metadata FROM projects WHERE id = ?",
            (project_id,),
        )
        project = await cursor.fetchone()
        if project is None:
            return {"error": "Project not found"}

        # Get milestones
        cursor = await db.execute(
            "SELECT title, status, due_date FROM milestones WHERE project_id = ? ORDER BY display_order",
            (project_id,),
        )
        milestones = [
            {"title": r["title"], "status": r["status"], "dueDate": r["due_date"]}
            for r in await cursor.fetchall()
        ]

        # Get latest assessment
        cursor = await db.execute(
            "SELECT health_score, general_notes, assessment_date FROM project_assessments "
            "WHERE project_id = ? ORDER BY assessment_date DESC LIMIT 1",
            (project_id,),
        )
        assessment = await cursor.fetchone()

        return {
            "title": project["title"],
            "researchQuestion": project["research_question"],
            "status": project["status"],
            "milestones": milestones,
            "latestAssessment": {
                "healthScore": assessment["health_score"],
                "notes": assessment["general_notes"],
                "date": assessment["assessment_date"],
            }
            if assessment
            else None,
        }


async def fetch_project_context(project_id: str) -> str:
    """Build a text summary of the project for LLM context.

    Loads project details, milestones, latest assessment, and student attributes
    from the database and formats them into a single string for use in prompts.
    Returns empty string if project not found.
    """
    async with get_db() as db:
        # Project info
        cursor = await db.execute(
            "SELECT title, research_question, status, "
            "start_date, end_date, metadata FROM projects WHERE id = ?",
            (project_id,),
        )
        project = await cursor.fetchone()
        if project is None:
            return ""

        parts: list[str] = []

        # Project overview
        parts.append(f"**Project:** {project['title']}")
        parts.append(f"**Research Question:** {project['research_question']}")
        parts.append(f"**Status:** {project['status']}")
        if project["start_date"] or project["end_date"]:
            timeline = f"**Timeline:** {project['start_date'] or '?'} → {project['end_date'] or '?'}"
            parts.append(timeline)

        # Student attributes from metadata
        meta = json.loads(project["metadata"]) if project["metadata"] else {}
        discovered = meta.get("discovered_attributes", {})
        if discovered:
            attr_parts: list[str] = []
            if discovered.get("coursesTaken"):
                attr_parts.append(f"Courses: {', '.join(discovered['coursesTaken'])}")
            if discovered.get("technicalSkills"):
                attr_parts.append(f"Skills: {', '.join(discovered['technicalSkills'])}")
            if discovered.get("researchInterests"):
                attr_parts.append(f"Interests: {', '.join(discovered['researchInterests'])}")
            if attr_parts:
                parts.append("**Student Background:** " + " | ".join(attr_parts))

        # Milestones
        cursor = await db.execute(
            "SELECT title, status FROM milestones WHERE project_id = ? ORDER BY display_order",
            (project_id,),
        )
        milestones = await cursor.fetchall()
        if milestones:
            ms_lines = [f"  - {m['title']} ({m['status']})" for m in milestones]
            parts.append("**Milestones:**\n" + "\n".join(ms_lines))

        # Latest assessment
        cursor = await db.execute(
            "SELECT health_score, general_notes FROM project_assessments "
            "WHERE project_id = ? ORDER BY assessment_date DESC LIMIT 1",
            (project_id,),
        )
        assessment = await cursor.fetchone()
        if assessment:
            score = assessment["health_score"]
            notes = assessment["general_notes"] or ""
            parts.append(f"**Health Score:** {score}/100" + (f" — {notes}" if notes else ""))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Conversation Memories (semantic memory with sqlite-vec)
# ---------------------------------------------------------------------------


async def store_memory(
    project_id: str,
    memory_text: str,
    memory_type: str,
    *,
    session_id: str | None = None,
    importance: int = 5,
    conversation_context: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    embedding: bytes | None = None,
) -> str:
    """Store a conversation memory with optional embedding in sqlite-vec.

    Inserts metadata into conversation_memories and the vector into
    memory_embeddings (vec0 virtual table), linked by rowid.
    Returns the memory ID.
    """
    memory_id = str(uuid.uuid4())
    async with get_db() as db:
        cursor = await db.execute(
            """
            INSERT INTO conversation_memories
                (id, project_id, session_id, memory_text, memory_type,
                 importance, conversation_context, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                project_id,
                session_id,
                memory_text,
                memory_type,
                importance,
                json.dumps(conversation_context) if conversation_context else None,
                json.dumps(tags) if tags else None,
            ),
        )
        rowid = cursor.lastrowid

        if embedding is not None and rowid is not None:
            await db.execute(
                "INSERT INTO memory_embeddings(rowid, embedding) VALUES (?, ?)",
                (rowid, embedding),
            )

        await db.commit()
    logger.debug("Stored memory {} (type={}, importance={})", memory_id, memory_type, importance)
    return memory_id


async def get_memories_for_project(
    project_id: str,
    *,
    page: int = 1,
    page_size: int = 10,
    memory_type: str | None = None,
) -> dict[str, Any]:
    """Get memories for a project, ordered by importance then recency."""
    offset = (page - 1) * page_size
    async with get_db() as db:
        where = "WHERE project_id = ?"
        params: list[Any] = [project_id]
        if memory_type:
            where += " AND memory_type = ?"
            params.append(memory_type)

        cursor = await db.execute(
            f"SELECT COUNT(*) FROM conversation_memories {where}", params,  # nosec B608
        )
        row = await cursor.fetchone()
        assert row is not None
        total = row[0]

        sql = f"""
            SELECT id, memory_text, memory_type, importance, tags, created_at
            FROM conversation_memories
            {where}
            ORDER BY importance DESC, created_at DESC
            LIMIT ? OFFSET ?
            """  # nosec B608 — where built from hardcoded clauses
        cursor = await db.execute(sql,
            [*params, page_size, offset],
        )
        rows = await cursor.fetchall()

    data = [
        {
            "id": r["id"],
            "memory_text": r["memory_text"],
            "memory_type": r["memory_type"],
            "importance": r["importance"],
            "tags": json.loads(r["tags"]) if r["tags"] else [],
            "created_at": r["created_at"],
        }
        for r in rows
    ]
    return {"data": data, "total": total}


async def search_similar_memories(
    project_id: str,
    query_embedding: bytes,
    *,
    threshold: float = 0.35,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Find memories similar to query via sqlite-vec KNN search.

    Uses vec0 cosine distance: distance = 1 - cosine_similarity.
    Threshold is cosine similarity (0.35), converted to max distance (0.65).
    """
    max_distance = 1.0 - threshold

    async with get_db() as db:
        # vec0 KNN requires `k = ?` in WHERE (not SQL LIMIT).
        # Filter by project_id and max distance in outer query.
        cursor = await db.execute(
            """
            SELECT id, memory_text, memory_type, importance, tags,
                   created_at, distance
            FROM (
                SELECT
                    cm.id,
                    cm.memory_text,
                    cm.memory_type,
                    cm.importance,
                    cm.tags,
                    cm.created_at,
                    cm.project_id,
                    me.distance
                FROM memory_embeddings me
                INNER JOIN conversation_memories cm ON cm.rowid = me.rowid
                WHERE me.embedding MATCH ?
                    AND k = ?
            )
            WHERE project_id = ?
                AND distance <= ?
            ORDER BY distance
            LIMIT ?
            """,
            (query_embedding, limit * 4, project_id, max_distance, limit),
        )
        rows = await cursor.fetchall()

    return [
        {
            "id": r["id"],
            "memory_text": r["memory_text"],
            "memory_type": r["memory_type"],
            "importance": r["importance"],
            "tags": json.loads(r["tags"]) if r["tags"] else [],
            "created_at": r["created_at"],
            "similarity": round(1.0 - r["distance"], 4),
        }
        for r in rows
    ]


async def touch_memory(memory_id: str) -> None:
    """Update accessed_at and increment access_count for a memory."""
    async with get_db() as db:
        await db.execute(
            """
            UPDATE conversation_memories
            SET accessed_at = datetime('now'), access_count = access_count + 1
            WHERE id = ?
            """,
            (memory_id,),
        )
        await db.commit()


# ---------------------------------------------------------------------------
# REST API CRUD operations
# ---------------------------------------------------------------------------

_PERSONA_JSON_FIELDS = ("notable_works", "key_achievements")


async def list_personas(
    *, page: int = 1, page_size: int = 10,
) -> dict[str, Any]:
    """List active teaching personas ordered by display_order."""
    offset = (page - 1) * page_size
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM teaching_personas WHERE is_active = 1"
        )
        row = await cursor.fetchone()
        assert row is not None
        total = row[0]
        cursor = await db.execute(
            "SELECT * FROM teaching_personas WHERE is_active = 1"
            " ORDER BY display_order LIMIT ? OFFSET ?",
            (page_size, offset),
        )
        rows = await cursor.fetchall()
    return {"data": [_row_to_dict(r, _PERSONA_JSON_FIELDS) for r in rows], "total": total}


async def get_persona(persona_id: str) -> dict[str, Any] | None:
    """Get a single persona by ID (snake_case keys)."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM teaching_personas WHERE persona_id = ?", (persona_id,)
        )
        row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_dict(row, _PERSONA_JSON_FIELDS)


# --- Projects ---

_PROJECT_JSON_FIELDS = ("metadata",)


async def create_project(
    title: str,
    research_question: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a project and return it."""
    project_id = str(uuid.uuid4())
    metadata = kwargs.pop("metadata", {})
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO projects (id, title, research_question, status,
                                  start_date, end_date, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                title,
                research_question,
                kwargs.get("status", "draft"),
                kwargs.get("start_date"),
                kwargs.get("end_date"),
                json.dumps(metadata),
            ),
        )
        await db.commit()
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
    assert row is not None
    return _row_to_dict(row, _PROJECT_JSON_FIELDS)


async def get_project(project_id: str) -> dict[str, Any] | None:
    """Get a single project by ID."""
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_dict(row, _PROJECT_JSON_FIELDS)


async def list_projects(
    *, page: int = 1, page_size: int = 10, status: str | None = None,
) -> dict[str, Any]:
    """List projects ordered by created_at DESC, optionally filtered by status."""
    offset = (page - 1) * page_size
    where = ""
    params: list[Any] = []
    if status:
        where = "WHERE status = ?"
        params.append(status)
    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT COUNT(*) FROM projects {where}", params,  # nosec B608
        )
        row = await cursor.fetchone()
        assert row is not None
        total = row[0]
        cursor = await db.execute(
            f"SELECT * FROM projects {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",  # nosec B608
            (*params, page_size, offset),
        )
        rows = await cursor.fetchall()
    return {"data": [_row_to_dict(r, _PROJECT_JSON_FIELDS) for r in rows], "total": total}


async def update_project(project_id: str, **fields: Any) -> dict[str, Any] | None:
    """Update project fields. Returns updated project or None if not found."""
    if not fields:
        return await get_project(project_id)
    _validate_fields(fields, _PROJECT_UPDATABLE, "projects")
    if "metadata" in fields:
        fields["metadata"] = json.dumps(fields["metadata"])
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = [*fields.values(), project_id]
    async with get_db() as db:
        cursor = await db.execute(
            f"UPDATE projects SET {set_clause}, updated_at = datetime('now') WHERE id = ?",  # nosec B608
            values,
        )
        if cursor.rowcount == 0:
            return None
        await db.commit()
    return await get_project(project_id)


async def delete_project(project_id: str) -> bool:
    """Delete a project (cascades to milestones, sessions, etc). Returns True if deleted."""
    async with get_db() as db:
        cursor = await db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        await db.commit()
    return cursor.rowcount > 0


# --- Milestones ---

_MILESTONE_JSON_FIELDS = ("metadata",)


async def create_milestone(project_id: str, title: str, **kwargs: Any) -> dict[str, Any]:
    """Create a milestone. Caller must verify project exists."""
    milestone_id = str(uuid.uuid4())
    metadata = kwargs.pop("metadata", {})
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO milestones (id, project_id, title, description, status,
                                    start_date, due_date, display_order, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                milestone_id,
                project_id,
                title,
                kwargs.get("description"),
                kwargs.get("status", "pending"),
                kwargs.get("start_date"),
                kwargs.get("due_date"),
                kwargs.get("display_order", 0),
                json.dumps(metadata),
            ),
        )
        await db.commit()
        cursor = await db.execute("SELECT * FROM milestones WHERE id = ?", (milestone_id,))
        row = await cursor.fetchone()
    assert row is not None
    return _row_to_dict(row, _MILESTONE_JSON_FIELDS)


async def get_milestone(milestone_id: str) -> dict[str, Any] | None:
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM milestones WHERE id = ?", (milestone_id,))
        row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_dict(row, _MILESTONE_JSON_FIELDS)


async def list_milestones(
    project_id: str, *, page: int = 1, page_size: int = 10,
) -> dict[str, Any]:
    offset = (page - 1) * page_size
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM milestones WHERE project_id = ?", (project_id,),
        )
        row = await cursor.fetchone()
        assert row is not None
        total = row[0]
        cursor = await db.execute(
            "SELECT * FROM milestones WHERE project_id = ?"
            " ORDER BY display_order LIMIT ? OFFSET ?",
            (project_id, page_size, offset),
        )
        rows = await cursor.fetchall()
    return {"data": [_row_to_dict(r, _MILESTONE_JSON_FIELDS) for r in rows], "total": total}


async def update_milestone(milestone_id: str, **fields: Any) -> dict[str, Any] | None:
    if not fields:
        return await get_milestone(milestone_id)
    _validate_fields(fields, _MILESTONE_UPDATABLE, "milestones")
    if "metadata" in fields:
        fields["metadata"] = json.dumps(fields["metadata"])
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = [*fields.values(), milestone_id]
    async with get_db() as db:
        cursor = await db.execute(
            f"UPDATE milestones SET {set_clause}, updated_at = datetime('now') WHERE id = ?",  # nosec B608
            values,
        )
        if cursor.rowcount == 0:
            return None
        await db.commit()
    return await get_milestone(milestone_id)


async def delete_milestone(milestone_id: str) -> bool:
    async with get_db() as db:
        cursor = await db.execute("DELETE FROM milestones WHERE id = ?", (milestone_id,))
        await db.commit()
    return cursor.rowcount > 0


# --- Sessions ---

_SESSION_JSON_FIELDS = ("metadata",)


async def create_session(project_id: str, **kwargs: Any) -> dict[str, Any]:
    session_id = kwargs.pop("session_id", None) or str(uuid.uuid4())
    metadata = kwargs.pop("metadata", {})
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO sessions (session_id, project_id, title, persona_id,
                                  language, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                project_id,
                kwargs.get("title"),
                kwargs.get("persona_id"),
                kwargs.get("language", "en"),
                json.dumps(metadata),
            ),
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
    assert row is not None
    return _row_to_dict(row, _SESSION_JSON_FIELDS)


async def get_session(session_id: str) -> dict[str, Any] | None:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_dict(row, _SESSION_JSON_FIELDS)


async def list_sessions(
    project_id: str, *, page: int = 1, page_size: int = 10,
) -> dict[str, Any]:
    offset = (page - 1) * page_size
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM sessions WHERE project_id = ? AND is_active = 1",
            (project_id,),
        )
        row = await cursor.fetchone()
        assert row is not None
        total = row[0]
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE project_id = ? AND is_active = 1"
            " ORDER BY last_active_at DESC LIMIT ? OFFSET ?",
            (project_id, page_size, offset),
        )
        rows = await cursor.fetchall()
    return {"data": [_row_to_dict(r, _SESSION_JSON_FIELDS) for r in rows], "total": total}


async def update_session(session_id: str, **fields: Any) -> dict[str, Any] | None:
    if not fields:
        # Touch last_active_at even with no field changes
        async with get_db() as db:
            cursor = await db.execute(
                "UPDATE sessions SET last_active_at = datetime('now') WHERE session_id = ?",
                (session_id,),
            )
            if cursor.rowcount == 0:
                return None
            await db.commit()
        return await get_session(session_id)
    _validate_fields(fields, _SESSION_UPDATABLE, "sessions")
    if "metadata" in fields:
        fields["metadata"] = json.dumps(fields["metadata"])
    if "is_active" in fields:
        fields["is_active"] = 1 if fields["is_active"] else 0
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = [*fields.values(), session_id]
    async with get_db() as db:
        cursor = await db.execute(
            f"UPDATE sessions SET {set_clause}, last_active_at = datetime('now') WHERE session_id = ?",  # nosec B608
            values,
        )
        if cursor.rowcount == 0:
            return None
        await db.commit()
    return await get_session(session_id)


async def increment_session_message_count(session_id: str) -> int:
    """Increment message_count by 1 and return the new value."""
    async with get_db() as db:
        cursor = await db.execute(
            "UPDATE sessions SET message_count = message_count + 1, "
            "last_active_at = datetime('now') WHERE session_id = ?",
            (session_id,),
        )
        if cursor.rowcount == 0:
            return 0
        await db.commit()
        cursor = await db.execute(
            "SELECT message_count FROM sessions WHERE session_id = ?", (session_id,),
        )
        row = await cursor.fetchone()
    return row[0] if row else 0


async def delete_session(session_id: str) -> bool:
    async with get_db() as db:
        cursor = await db.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
        await db.commit()
    return cursor.rowcount > 0


# --- Workshop Sessions ---

_WORKSHOP_SESSION_JSON_FIELDS = ("metadata",)
_WORKSHOP_SESSION_UPDATABLE = frozenset({
    "status", "current_stage", "title", "metadata", "language",
})


async def create_workshop_session(
    project_id: str,
    workshop_type: str,
    *,
    session_id: str | None = None,
    language: str = "en",
    current_stage: str | None = None,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a new workshop session record."""
    sid = session_id or str(uuid.uuid4())
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO workshop_sessions
                (session_id, project_id, workshop_type, language,
                 current_stage, title, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sid, project_id, workshop_type, language,
                current_stage, title, json.dumps(metadata or {}),
            ),
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT * FROM workshop_sessions WHERE session_id = ?", (sid,),
        )
        row = await cursor.fetchone()
    assert row is not None
    return _row_to_dict(row, _WORKSHOP_SESSION_JSON_FIELDS)


async def get_workshop_session(session_id: str) -> dict[str, Any] | None:
    """Get a workshop session by ID."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM workshop_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_dict(row, _WORKSHOP_SESSION_JSON_FIELDS)


async def list_workshop_sessions(
    project_id: str,
    workshop_type: str | None = None,
    *,
    status: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """List workshop sessions for a project, optionally filtered."""
    conditions = ["project_id = ?"]
    params: list[Any] = [project_id]

    if workshop_type:
        conditions.append("workshop_type = ?")
        params.append(workshop_type)
    if status:
        conditions.append("status = ?")
        params.append(status)

    where = " AND ".join(conditions)
    offset = (page - 1) * page_size

    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT COUNT(*) FROM workshop_sessions WHERE {where}",  # nosec B608
            params,
        )
        row = await cursor.fetchone()
        assert row is not None
        total = row[0]

        cursor = await db.execute(
            f"SELECT * FROM workshop_sessions WHERE {where}"  # nosec B608
            " ORDER BY last_active_at DESC LIMIT ? OFFSET ?",
            [*params, page_size, offset],
        )
        rows = await cursor.fetchall()

    return {
        "data": [_row_to_dict(r, _WORKSHOP_SESSION_JSON_FIELDS) for r in rows],
        "total": total,
    }


async def update_workshop_session(
    session_id: str, **fields: Any,
) -> dict[str, Any] | None:
    """Update a workshop session. Always touches last_active_at."""
    if not fields:
        async with get_db() as db:
            cursor = await db.execute(
                "UPDATE workshop_sessions SET last_active_at = datetime('now') "
                "WHERE session_id = ?",
                (session_id,),
            )
            if cursor.rowcount == 0:
                return None
            await db.commit()
        return await get_workshop_session(session_id)

    _validate_fields(fields, _WORKSHOP_SESSION_UPDATABLE, "workshop_sessions")
    if "metadata" in fields:
        fields["metadata"] = json.dumps(fields["metadata"])
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = [*fields.values(), session_id]
    async with get_db() as db:
        cursor = await db.execute(
            f"UPDATE workshop_sessions SET {set_clause}, "  # nosec B608
            "last_active_at = datetime('now') WHERE session_id = ?",
            values,
        )
        if cursor.rowcount == 0:
            return None
        await db.commit()
    return await get_workshop_session(session_id)


async def delete_workshop_session(session_id: str) -> bool:
    """Delete a workshop session by ID. Returns True if deleted."""
    async with get_db() as db:
        cursor = await db.execute(
            "DELETE FROM workshop_sessions WHERE session_id = ?",
            (session_id,),
        )
        await db.commit()
    return cursor.rowcount > 0


# --- Memories (additional) ---


async def delete_memory(memory_id: str) -> bool:
    """Delete a memory by ID. Returns True if deleted."""
    async with get_db() as db:
        # Get rowid first to delete from vec0 table
        cursor = await db.execute(
            "SELECT rowid FROM conversation_memories WHERE id = ?", (memory_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return False
        rowid = row[0]
        # Delete from vec0 table (may not have an embedding)
        await db.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (rowid,))
        await db.execute("DELETE FROM conversation_memories WHERE id = ?", (memory_id,))
        await db.commit()
    return True


# --- Artifacts ---

_ARTIFACT_JSON_FIELDS = ("metadata",)


async def create_artifact(project_id: str, **kwargs: Any) -> dict[str, Any]:
    artifact_id = str(uuid.uuid4())
    metadata = kwargs.pop("metadata", {})
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO artifacts (id, project_id, artifact_type, file_name, file_path,
                                   mime_type, file_size_bytes, description, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                project_id,
                kwargs["artifact_type"],
                kwargs["file_name"],
                kwargs["file_path"],
                kwargs["mime_type"],
                kwargs["file_size_bytes"],
                kwargs.get("description"),
                json.dumps(metadata),
            ),
        )
        await db.commit()
        cursor = await db.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
        row = await cursor.fetchone()
    assert row is not None
    return _row_to_dict(row, _ARTIFACT_JSON_FIELDS)


async def get_artifact(artifact_id: str) -> dict[str, Any] | None:
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
        row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_dict(row, _ARTIFACT_JSON_FIELDS)


async def list_artifacts(
    project_id: str, *, page: int = 1, page_size: int = 10,
) -> dict[str, Any]:
    offset = (page - 1) * page_size
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM artifacts WHERE project_id = ?", (project_id,),
        )
        row = await cursor.fetchone()
        assert row is not None
        total = row[0]
        cursor = await db.execute(
            "SELECT * FROM artifacts WHERE project_id = ?"
            " ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (project_id, page_size, offset),
        )
        rows = await cursor.fetchall()
    return {"data": [_row_to_dict(r, _ARTIFACT_JSON_FIELDS) for r in rows], "total": total}


async def update_artifact(artifact_id: str, **fields: Any) -> dict[str, Any] | None:
    if not fields:
        return await get_artifact(artifact_id)
    _validate_fields(fields, _ARTIFACT_UPDATABLE, "artifacts")
    if "metadata" in fields:
        fields["metadata"] = json.dumps(fields["metadata"])
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = [*fields.values(), artifact_id]
    async with get_db() as db:
        cursor = await db.execute(
            f"UPDATE artifacts SET {set_clause}, updated_at = datetime('now') WHERE id = ?",  # nosec B608
            values,
        )
        if cursor.rowcount == 0:
            return None
        await db.commit()
    return await get_artifact(artifact_id)


async def delete_artifact(artifact_id: str) -> bool:
    async with get_db() as db:
        cursor = await db.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
        await db.commit()
    return cursor.rowcount > 0


async def store_artifact_chunks(
    artifact_id: str,
    chunks: list[str],
    embeddings: list[bytes],
) -> int:
    """Insert text chunks and their embeddings for an artifact.

    Inserts into artifact_chunks and artifact_embeddings atomically.
    Returns the number of chunks stored.
    """
    if not chunks:
        return 0
    async with get_db() as db:
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            cursor = await db.execute(
                """
                INSERT INTO artifact_chunks (artifact_id, chunk_index, chunk_text)
                VALUES (?, ?, ?)
                """,
                (artifact_id, i, chunk),
            )
            rowid = cursor.lastrowid
            if rowid is not None:
                await db.execute(
                    "INSERT INTO artifact_embeddings(rowid, embedding) VALUES (?, ?)",
                    (rowid, embedding),
                )
        await db.commit()
    logger.debug("Stored {} artifact chunks for artifact {}", len(chunks), artifact_id)
    return len(chunks)


async def search_similar_artifacts(
    project_id: str,
    query_embedding: bytes,
    *,
    threshold: float = 0.35,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Find artifact chunks similar to query via sqlite-vec KNN search.

    Uses vec0 cosine distance: distance = 1 - cosine_similarity.
    Filters by project_id via join on artifacts table.
    """
    max_distance = 1.0 - threshold

    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT ac.artifact_id, a.file_name, a.description,
                   ac.chunk_text, ae.distance
            FROM (
                SELECT rowid, distance
                FROM artifact_embeddings
                WHERE embedding MATCH ?
                    AND k = ?
            ) ae
            INNER JOIN artifact_chunks ac ON ac.id = ae.rowid
            INNER JOIN artifacts a ON a.id = ac.artifact_id
            WHERE a.project_id = ?
                AND ae.distance <= ?
            ORDER BY ae.distance
            LIMIT ?
            """,
            (query_embedding, limit * 4, project_id, max_distance, limit),
        )
        rows = await cursor.fetchall()

    return [
        {
            "artifact_id": r["artifact_id"],
            "file_name": r["file_name"],
            "description": r["description"],
            "chunk_text": r["chunk_text"],
            "similarity": round(1.0 - r["distance"], 4),
        }
        for r in rows
    ]


async def delete_artifact_chunks(artifact_id: str) -> None:
    """Delete all chunks and embeddings for an artifact.

    Must delete vec0 entries first (CASCADE won't clean virtual tables),
    then delete chunks.
    """
    async with get_db() as db:
        # Get chunk rowids to delete from vec0
        cursor = await db.execute(
            "SELECT id FROM artifact_chunks WHERE artifact_id = ?",
            (artifact_id,),
        )
        rows = await cursor.fetchall()
        for row in rows:
            await db.execute(
                "DELETE FROM artifact_embeddings WHERE rowid = ?", (row["id"],),
            )
        await db.execute(
            "DELETE FROM artifact_chunks WHERE artifact_id = ?", (artifact_id,),
        )
        await db.commit()
    logger.debug("Deleted artifact chunks for artifact {}", artifact_id)


# --- Assessments ---

_ASSESSMENT_JSON_FIELDS = ("issues", "recommended_actions", "metadata")


async def create_assessment(
    project_id: str,
    assessment_date: str,
    health_score: int | None = None,
    issues: list[dict[str, Any]] | None = None,
    recommended_actions: list[dict[str, Any]] | None = None,
    general_notes: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a project assessment. Returns the created assessment."""
    assessment_id = str(uuid.uuid4())
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO project_assessments
                (id, project_id, assessment_date, health_score,
                 issues, recommended_actions, general_notes, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                assessment_id,
                project_id,
                assessment_date,
                health_score,
                json.dumps(issues or []),
                json.dumps(recommended_actions or []),
                general_notes,
                json.dumps(metadata or {}),
            ),
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT * FROM project_assessments WHERE id = ?", (assessment_id,),
        )
        row = await cursor.fetchone()
    assert row is not None
    return _row_to_dict(row, _ASSESSMENT_JSON_FIELDS)


async def list_assessments(
    project_id: str, *, page: int = 1, page_size: int = 10,
) -> dict[str, Any]:
    offset = (page - 1) * page_size
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM project_assessments WHERE project_id = ?",
            (project_id,),
        )
        row = await cursor.fetchone()
        assert row is not None
        total = row[0]
        cursor = await db.execute(
            "SELECT * FROM project_assessments WHERE project_id = ?"
            " ORDER BY assessment_date DESC LIMIT ? OFFSET ?",
            (project_id, page_size, offset),
        )
        rows = await cursor.fetchall()
    return {"data": [_row_to_dict(r, _ASSESSMENT_JSON_FIELDS) for r in rows], "total": total}


# --- Student Profile (singleton) ---


def _parse_json_list(
    raw: str | None, adapter: Any, field_name: str,
) -> list[dict[str, Any]]:
    """Parse a JSON column into a list of validated Pydantic dicts.

    Returns [] and logs a warning if the data is missing or malformed.
    ``adapter`` must be a ``pydantic.TypeAdapter[list[SomeModel]]``.
    """
    if not raw:
        return []
    try:
        return [item.model_dump() for item in adapter.validate_json(raw)]
    except Exception:
        logger.warning("Ignoring malformed {} JSON in student_profile: {!r}", field_name, raw)
        return []


async def get_student_profile() -> dict[str, Any] | None:
    """Get the student profile (singleton row)."""
    from pydantic import TypeAdapter

    from research_mentor.api.schemas import DomainExpertise, ProfessionalExperience

    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM student_profile WHERE id = 1")
        row = await cursor.fetchone()
    if row is None:
        return None

    _expertise_adapter = TypeAdapter(list[DomainExpertise])
    _experience_adapter = TypeAdapter(list[ProfessionalExperience])

    return {
        "name": row["name"],
        "age": row["age"],
        "country": row["country"],
        "grade": row["grade"],
        "college_year": row["college_year"],
        "language": row["language"],
        "persona": row["persona"],
        "education_level": row["education_level"],
        "domain_expertise": _parse_json_list(
            row["domain_expertise"], _expertise_adapter, "domain_expertise",
        ),
        "professional_experience": _parse_json_list(
            row["professional_experience"], _experience_adapter, "professional_experience",
        ),
        "background_notes": row["background_notes"],
        "institution": row["institution"],
        "institution_ror": row["institution_ror"],
        "city": row["city"],
        "state_region": row["state_region"],
        "zip_code": row["zip_code"],
        "license_accepted": bool(row["license_accepted"]),
        "onboarding_completed": bool(row["onboarding_completed"]),
        "dark_mode": row["dark_mode"],
        "view_mode": row["view_mode"],
        "active_project_id": row["active_project_id"],
        "updated_at": row["updated_at"],
    }


async def upsert_student_profile(
    *,
    name: str | None = None,
    age: int | None = None,
    country: str | None = None,
    grade: str | None = None,
    college_year: str | None = None,
    language: str | None = None,
    persona: str | None = None,
    education_level: str | None = None,
    domain_expertise: list[dict[str, Any]] | None = None,
    professional_experience: list[dict[str, Any]] | None = None,
    background_notes: str | None = None,
    institution: str | None = None,
    institution_ror: str | None = None,
    city: str | None = None,
    state_region: str | None = None,
    zip_code: str | None = None,
    license_accepted: bool | None = None,
    onboarding_completed: bool | None = None,
    dark_mode: str | None = None,
    view_mode: str | None = None,
    active_project_id: str | None = None,
) -> dict[str, Any]:
    """Create or update the student profile. Returns the updated profile.

    Uses SQLite UPSERT (INSERT ... ON CONFLICT DO UPDATE) for atomicity.
    Only updates fields that are explicitly provided (not None).
    """
    # Serialize list fields to JSON for storage
    expertise_json = json.dumps(domain_expertise) if domain_expertise is not None else None
    experience_json = (
        json.dumps(professional_experience) if professional_experience is not None else None
    )

    async with get_db() as db:
        # Check if profile exists
        cursor = await db.execute("SELECT * FROM student_profile WHERE id = 1")
        existing = await cursor.fetchone()

        if existing is None:
            # Insert new row
            await db.execute(
                """
                INSERT INTO student_profile
                    (id, name, age, country, grade, college_year, language, persona,
                     education_level, domain_expertise, professional_experience,
                     background_notes, institution, institution_ror, city, state_region,
                     zip_code, license_accepted, onboarding_completed, dark_mode,
                     view_mode, active_project_id)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?)
                """,
                (name, age, country, grade, college_year, language or "en",
                 persona or "newton", education_level,
                 expertise_json or "[]", experience_json or "[]",
                 background_notes, institution, institution_ror, city, state_region,
                 zip_code,
                 int(license_accepted) if license_accepted is not None else 0,
                 int(onboarding_completed) if onboarding_completed is not None else 0,
                 dark_mode or "system", view_mode or "simple", active_project_id),
            )
        else:
            # Update only provided fields (keep existing values for None)
            new_name = name if name is not None else existing["name"]
            new_age = age if age is not None else existing["age"]
            new_country = country if country is not None else existing["country"]
            new_grade = grade if grade is not None else existing["grade"]
            new_college_year = (
                college_year if college_year is not None else existing["college_year"]
            )
            new_language = language if language is not None else existing["language"]
            new_persona = persona if persona is not None else existing["persona"]
            new_education = (
                education_level if education_level is not None
                else existing["education_level"]
            )
            new_expertise = (
                expertise_json if expertise_json is not None
                else existing["domain_expertise"]
            )
            new_experience = (
                experience_json if experience_json is not None
                else existing["professional_experience"]
            )
            new_notes = (
                background_notes if background_notes is not None
                else existing["background_notes"]
            )
            new_institution = (
                institution if institution is not None else existing["institution"]
            )
            new_institution_ror = (
                institution_ror if institution_ror is not None
                else existing["institution_ror"]
            )
            new_city = city if city is not None else existing["city"]
            new_state_region = (
                state_region if state_region is not None else existing["state_region"]
            )
            new_zip_code = (
                zip_code if zip_code is not None else existing["zip_code"]
            )
            new_license = (
                int(license_accepted) if license_accepted is not None
                else existing["license_accepted"]
            )
            new_onboarding = (
                int(onboarding_completed) if onboarding_completed is not None
                else existing["onboarding_completed"]
            )
            new_dark_mode = (
                dark_mode if dark_mode is not None else existing["dark_mode"]
            )
            new_view_mode = (
                view_mode if view_mode is not None else existing["view_mode"]
            )
            new_active_project = (
                active_project_id if active_project_id is not None
                else existing["active_project_id"]
            )
            await db.execute(
                """
                UPDATE student_profile
                SET name = ?, age = ?, country = ?, grade = ?, college_year = ?,
                    language = ?, persona = ?,
                    education_level = ?, domain_expertise = ?,
                    professional_experience = ?, background_notes = ?,
                    institution = ?, institution_ror = ?, city = ?, state_region = ?,
                    zip_code = ?, license_accepted = ?, onboarding_completed = ?,
                    dark_mode = ?, view_mode = ?, active_project_id = ?,
                    updated_at = datetime('now')
                WHERE id = 1
                """,
                (new_name, new_age, new_country, new_grade, new_college_year,
                 new_language, new_persona,
                 new_education, new_expertise, new_experience, new_notes,
                 new_institution, new_institution_ror, new_city, new_state_region,
                 new_zip_code, new_license, new_onboarding, new_dark_mode,
                 new_view_mode, new_active_project),
            )
        await db.commit()

    result = await get_student_profile()
    assert result is not None
    return result


async def get_assessment(assessment_id: str) -> dict[str, Any] | None:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM project_assessments WHERE id = ?", (assessment_id,)
        )
        row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_dict(row, _ASSESSMENT_JSON_FIELDS)


# --- Engagement Metrics (computed, not stored) ---


async def compute_engagement_metrics(project_id: str) -> dict[str, Any]:
    """Compute engagement metrics from existing tables. No new table needed.

    Returns session count, chat interactions, artifact count, milestone
    progress, memory count, and days since last activity.
    """
    async with get_db() as db:
        # Session count
        cursor = await db.execute(
            "SELECT COUNT(*) FROM sessions WHERE project_id = ?", (project_id,),
        )
        row = await cursor.fetchone()
        assert row is not None
        session_count = row[0]

        # Chat interaction count (LLM calls for this project)
        cursor = await db.execute(
            "SELECT COUNT(*) FROM llm_usage WHERE project_id = ?", (project_id,),
        )
        row = await cursor.fetchone()
        assert row is not None
        chat_interaction_count = row[0]

        # Artifact count
        cursor = await db.execute(
            "SELECT COUNT(*) FROM artifacts WHERE project_id = ?", (project_id,),
        )
        row = await cursor.fetchone()
        assert row is not None
        artifact_count = row[0]

        # Milestone progress
        cursor = await db.execute(
            "SELECT COUNT(*) as total,"
            " SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed"
            " FROM milestones WHERE project_id = ?",
            (project_id,),
        )
        row = await cursor.fetchone()
        assert row is not None
        milestone_total = row["total"]
        milestone_completed = row["completed"] or 0

        # Overdue milestones
        cursor = await db.execute(
            "SELECT COUNT(*) FROM milestones"
            " WHERE project_id = ? AND status = 'pending'"
            " AND due_date IS NOT NULL AND due_date < date('now')",
            (project_id,),
        )
        row = await cursor.fetchone()
        assert row is not None
        overdue_count = row[0]

        # Memory count
        cursor = await db.execute(
            "SELECT COUNT(*) FROM conversation_memories WHERE project_id = ?",
            (project_id,),
        )
        row = await cursor.fetchone()
        assert row is not None
        memory_count = row[0]

        # Days since last activity
        cursor = await db.execute(
            "SELECT MAX(last_active_at) as last_active"
            " FROM sessions WHERE project_id = ?",
            (project_id,),
        )
        row = await cursor.fetchone()
        last_active = row["last_active"] if row else None
        days_since_last_activity: int | None = None
        if last_active:
            from datetime import datetime

            try:
                last_dt = datetime.fromisoformat(last_active)
                days_since_last_activity = (datetime.now() - last_dt).days
            except ValueError:
                pass

    return {
        "session_count": session_count,
        "chat_interaction_count": chat_interaction_count,
        "artifact_count": artifact_count,
        "milestone_total": milestone_total,
        "milestone_completed": milestone_completed,
        "overdue_count": overdue_count,
        "memory_count": memory_count,
        "days_since_last_activity": days_since_last_activity,
    }


# --- Student Assessments (cross-project) ---

_STUDENT_ASSESSMENT_JSON_FIELDS = (
    "key_strengths", "growth_areas", "recommendations", "metadata",
)


async def create_student_assessment(
    assessment_date: str,
    *,
    overall_skill_level: str | None = None,
    research_maturity_score: int | None = None,
    growth_trajectory: str | None = None,
    key_strengths: list[str] | None = None,
    growth_areas: list[str] | None = None,
    cross_project_patterns: str | None = None,
    recommendations: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    avg_critical_reading: int | None = None,
    avg_data_analysis_skill: int | None = None,
    avg_experimental_design_skill: int | None = None,
    avg_writing_skill: int | None = None,
    avg_critical_thinking_skill: int | None = None,
    avg_time_management_skill: int | None = None,
    avg_collaboration_skill: int | None = None,
    avg_research_maturity: int | None = None,
) -> dict[str, Any]:
    """Create a cross-project student assessment. Returns the created assessment."""
    assessment_id = str(uuid.uuid4())
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO student_assessments
                (id, assessment_date, overall_skill_level, research_maturity_score,
                 growth_trajectory, key_strengths, growth_areas,
                 cross_project_patterns, recommendations, metadata,
                 avg_critical_reading, avg_data_analysis_skill,
                 avg_experimental_design_skill, avg_writing_skill,
                 avg_critical_thinking_skill, avg_time_management_skill,
                 avg_collaboration_skill, avg_research_maturity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                assessment_id,
                assessment_date,
                overall_skill_level,
                research_maturity_score,
                growth_trajectory,
                json.dumps(key_strengths or []),
                json.dumps(growth_areas or []),
                cross_project_patterns,
                json.dumps(recommendations or []),
                json.dumps(metadata or {}),
                avg_critical_reading,
                avg_data_analysis_skill,
                avg_experimental_design_skill,
                avg_writing_skill,
                avg_critical_thinking_skill,
                avg_time_management_skill,
                avg_collaboration_skill,
                avg_research_maturity,
            ),
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT * FROM student_assessments WHERE id = ?", (assessment_id,),
        )
        row = await cursor.fetchone()
    assert row is not None
    return _row_to_dict(row, _STUDENT_ASSESSMENT_JSON_FIELDS)


async def delete_student_assessment(assessment_id: str) -> bool:
    """Delete a student assessment by ID.  Returns True if a row was deleted."""
    async with get_db() as db:
        cursor = await db.execute(
            "DELETE FROM student_assessments WHERE id = ?", (assessment_id,),
        )
        await db.commit()
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# LLM Usage Tracking
# ---------------------------------------------------------------------------


async def store_llm_usage(record: dict[str, Any]) -> None:
    """Insert a single LLM usage record."""
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO llm_usage
                (backend, provider, model, prompt_tokens, completion_tokens,
                 session_id, project_id, call_type, elapsed_seconds, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["backend"],
                record["provider"],
                record["model"],
                record.get("prompt_tokens", 0),
                record.get("completion_tokens", 0),
                record.get("session_id"),
                record.get("project_id"),
                record.get("call_type", "chat"),
                record.get("elapsed_seconds"),
                record.get("error", 0),
            ),
        )
        await db.commit()


async def store_llm_usage_batch(
    records: list[dict[str, Any]],
    *,
    session_id: str | None = None,
    project_id: str | None = None,
    purpose: str = "office",
) -> int:
    """Insert multiple LLM usage records. Returns count inserted."""
    if not records:
        return 0
    async with get_db() as db:
        for r in records:
            await db.execute(
                """
                INSERT INTO llm_usage
                    (backend, provider, model, prompt_tokens,
                     completion_tokens, session_id, project_id,
                     call_type, elapsed_seconds, error, purpose)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["backend"],
                    r["provider"],
                    r["model"],
                    r.get("prompt_tokens", 0),
                    r.get("completion_tokens", 0),
                    session_id or r.get("session_id"),
                    project_id or r.get("project_id"),
                    r.get("call_type", "chat"),
                    r.get("elapsed_seconds"),
                    r.get("error", 0),
                    purpose,
                ),
            )
        await db.commit()
    logger.debug("Stored {} LLM usage records (purpose={})", len(records), purpose)
    return len(records)


async def get_usage_summary(
    period: str = "month",
    *,
    backend: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Get aggregated token usage for a period.

    period: "day", "week", "month", "all"
    """
    where_clauses = []
    params: list[Any] = []

    ts_filter = _period_to_sql(period)
    if ts_filter:
        where_clauses.append(ts_filter)

    if backend:
        where_clauses.append("backend = ?")
        params.append(backend)
    if provider:
        where_clauses.append("provider = ?")
        params.append(provider)
    if model:
        where_clauses.append("model = ?")
        params.append(model)
    if project_id:
        where_clauses.append("project_id = ?")
        params.append(project_id)
    if session_id:
        where_clauses.append("session_id = ?")
        params.append(session_id)

    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    async with get_db() as db:
        sql = f"""
            SELECT
                COUNT(*) as total_calls,
                COALESCE(SUM(prompt_tokens), 0) as total_prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) as total_completion_tokens,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(elapsed_seconds), 0) as total_elapsed,
                COALESCE(SUM(error), 0) as total_errors
            FROM llm_usage
            {where}
            """  # nosec B608 — where built from hardcoded clauses
        cursor = await db.execute(sql, params)
        row = await cursor.fetchone()

    assert row is not None
    return {
        "total_calls": row["total_calls"],
        "total_prompt_tokens": row["total_prompt_tokens"],
        "total_completion_tokens": row["total_completion_tokens"],
        "total_tokens": row["total_tokens"],
        "total_elapsed": round(row["total_elapsed"], 1),
        "total_errors": row["total_errors"],
    }


async def get_usage_overview(
    period: str = "month",
    *,
    backend: str | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Get usage overview with conversation counts and avg latency.

    Returns conversations (distinct sessions), total calls, tokens,
    errors, and average elapsed time per conversation.
    """
    where_clauses: list[str] = []
    params: list[Any] = []

    ts_filter = _period_to_sql(period)
    if ts_filter:
        where_clauses.append(ts_filter)
    if backend:
        where_clauses.append("backend = ?")
        params.append(backend)
    if project_id:
        where_clauses.append("project_id = ?")
        params.append(project_id)

    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    async with get_db() as db:
        sql = f"""
            SELECT
                COUNT(DISTINCT session_id) as conversations,
                COUNT(*) as total_calls,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(prompt_tokens), 0) as total_prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) as total_completion_tokens,
                COALESCE(SUM(error), 0) as total_errors,
                CASE WHEN COUNT(DISTINCT session_id) > 0
                     THEN COALESCE(SUM(elapsed_seconds), 0)
                          / COUNT(DISTINCT session_id)
                     ELSE 0
                END as avg_session_elapsed
            FROM llm_usage
            {where}
            """  # nosec B608 — where built from hardcoded clauses
        cursor = await db.execute(sql, params)
        row = await cursor.fetchone()

    assert row is not None
    return {
        "conversations": row["conversations"],
        "total_calls": row["total_calls"],
        "total_tokens": row["total_tokens"],
        "total_prompt_tokens": row["total_prompt_tokens"],
        "total_completion_tokens": row["total_completion_tokens"],
        "total_errors": row["total_errors"],
        "avg_session_elapsed": round(row["avg_session_elapsed"], 1),
    }


async def get_usage_by_project(period: str = "month") -> list[dict[str, Any]]:
    """Get token usage broken down by project."""
    ts_filter = _period_to_sql(period)
    where = f"WHERE {ts_filter}" if ts_filter else ""

    async with get_db() as db:
        sql = f"""
            SELECT
                project_id,
                COUNT(DISTINCT session_id) as conversations,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                MAX(timestamp) as last_active
            FROM llm_usage
            {where}
            GROUP BY project_id
            ORDER BY total_tokens DESC
            """  # nosec B608 — where built from hardcoded clauses
        cursor = await db.execute(sql)
        rows = await cursor.fetchall()

    return [
        {
            "project_id": r["project_id"],
            "conversations": r["conversations"],
            "total_tokens": r["total_tokens"],
            "last_active": r["last_active"],
        }
        for r in rows
    ]


async def get_usage_timeseries(
    period: str = "month",
    granularity: str = "day",
) -> list[dict[str, Any]]:
    """Get usage over time for charting.

    granularity: "hour", "day"
    """
    ts_filter = _period_to_sql(period)
    where = f"WHERE {ts_filter}" if ts_filter else ""

    if granularity == "hour":
        group_expr = "strftime('%Y-%m-%d %H:00', timestamp)"
    else:
        group_expr = "strftime('%Y-%m-%d', timestamp)"

    async with get_db() as db:
        sql = f"""
            SELECT
                {group_expr} as period,
                COUNT(*) as calls,
                COUNT(DISTINCT session_id) as conversations,
                COALESCE(SUM(prompt_tokens), 0) as prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) as completion_tokens,
                COALESCE(SUM(total_tokens), 0) as total_tokens
            FROM llm_usage
            {where}
            GROUP BY period
            ORDER BY period
            """  # nosec B608 — where/group_expr built from hardcoded clauses
        cursor = await db.execute(sql)
        rows = await cursor.fetchall()

    return [
        {
            "period": r["period"],
            "calls": r["calls"],
            "conversations": r["conversations"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "total_tokens": r["total_tokens"],
        }
        for r in rows
    ]


async def get_usage_by_model(period: str = "month") -> list[dict[str, Any]]:
    """Get token usage broken down by backend/provider/model."""
    ts_filter = _period_to_sql(period)
    where = f"WHERE {ts_filter}" if ts_filter else ""

    async with get_db() as db:
        sql = f"""
            SELECT
                backend,
                provider,
                model,
                COUNT(*) as calls,
                COUNT(DISTINCT session_id) as conversations,
                COALESCE(SUM(prompt_tokens), 0) as prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) as completion_tokens,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                AVG(CASE WHEN elapsed_seconds > 0 THEN elapsed_seconds END)
                    as avg_elapsed
            FROM llm_usage
            {where}
            GROUP BY backend, provider, model
            ORDER BY total_tokens DESC
            """  # nosec B608 — where built from hardcoded clauses
        cursor = await db.execute(sql)
        rows = await cursor.fetchall()

    return [
        {
            "backend": r["backend"],
            "provider": r["provider"],
            "model": r["model"],
            "calls": r["calls"],
            "conversations": r["conversations"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "total_tokens": r["total_tokens"],
            "avg_elapsed": round(r["avg_elapsed"], 1) if r["avg_elapsed"] else None,
        }
        for r in rows
    ]


async def get_usage_by_purpose(
    period: str = "month",
) -> list[dict[str, Any]]:
    """Get token usage broken down by purpose.

    Purposes: office, workshop_phenomenon, workshop_claims,
    workshop_hypothesis, assessment, title, vision.
    Legacy records may still use 'question_workshop', 'sage',
    'workshop_sage', 'workshop_oneshot', or 'workshop_investigator'.
    """
    ts_filter = _period_to_sql(period)
    where = f"WHERE {ts_filter}" if ts_filter else ""

    async with get_db() as db:
        sql = f"""
            SELECT
                purpose,
                COUNT(*) as calls,
                COALESCE(SUM(prompt_tokens), 0) as prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) as completion_tokens,
                COALESCE(SUM(total_tokens), 0) as total_tokens
            FROM llm_usage
            {where}
            GROUP BY purpose
            ORDER BY total_tokens DESC
            """  # nosec B608
        cursor = await db.execute(sql)
        rows = await cursor.fetchall()

    return [
        {
            "purpose": r["purpose"],
            "calls": r["calls"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "total_tokens": r["total_tokens"],
        }
        for r in rows
    ]


async def get_period_usage_for_budget(
    period: str,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> int:
    """Get total tokens used in a period, filtered by provider/model. For budget checks."""
    where_clauses = ["backend = 'api'"]
    params: list[Any] = []

    ts_filter = _period_to_sql(period)
    if ts_filter:
        where_clauses.append(ts_filter)

    if provider:
        where_clauses.append("provider = ?")
        params.append(provider)
    if model:
        where_clauses.append("model = ?")
        params.append(model)

    where = f"WHERE {' AND '.join(where_clauses)}"

    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT COALESCE(SUM(total_tokens), 0) as total FROM llm_usage {where}",  # nosec B608
            params,
        )
        row = await cursor.fetchone()
    assert row is not None
    return row["total"]  # type: ignore[no-any-return]


def _period_to_sql(period: str) -> str:
    """Convert a period name to a SQL WHERE clause on timestamp."""
    if period in ("day", "today", "daily"):
        return "timestamp >= datetime('now', '-1 day')"
    if period in ("week", "weekly"):
        return "timestamp >= datetime('now', '-7 days')"
    if period in ("month", "monthly"):
        return "timestamp >= datetime('now', '-30 days')"
    if period in ("all", "total"):
        return ""
    return "timestamp >= datetime('now', '-30 days')"


async def get_latest_student_assessment() -> dict[str, Any] | None:
    """Get the most recent cross-project student assessment."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM student_assessments ORDER BY assessment_date DESC LIMIT 1"
        )
        row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_dict(row, _STUDENT_ASSESSMENT_JSON_FIELDS)


async def list_student_assessments(
    *, page: int = 1, page_size: int = 10,
) -> dict[str, Any]:
    """List cross-project student assessments ordered by date DESC."""
    offset = (page - 1) * page_size
    async with get_db() as db:
        cursor = await db.execute("SELECT COUNT(*) FROM student_assessments")
        row = await cursor.fetchone()
        assert row is not None
        total = row[0]
        cursor = await db.execute(
            "SELECT * FROM student_assessments ORDER BY assessment_date DESC"
            " LIMIT ? OFFSET ?",
            (page_size, offset),
        )
        rows = await cursor.fetchall()
    return {
        "data": [_row_to_dict(r, _STUDENT_ASSESSMENT_JSON_FIELDS) for r in rows],
        "total": total,
    }


# ---------------------------------------------------------------------------
# Generated Problems (Question Workshop)
# ---------------------------------------------------------------------------

_GENERATED_PROBLEM_JSON_FIELDS = (
    "domains", "core_concepts", "materials", "metadata",
)

_GENERATED_PROBLEM_COLS = frozenset({
    "id", "project_id", "field", "problem_type", "domains", "user_suggestion",
    "title", "description", "investigation", "core_concepts", "materials",
    "feasibility", "recommended", "safety_level", "complexity_score",
    "engagement_score", "overall_quality", "metadata", "created_at",
    "workshop_type",
})


async def create_generated_problem(**kwargs: Any) -> dict[str, Any]:
    """Insert a generated problem and return the full row."""
    problem_id = kwargs.get("id") or uuid.uuid4().hex
    kwargs["id"] = problem_id
    _validate_fields(kwargs, _GENERATED_PROBLEM_COLS, "generated_problems")
    cols = list(kwargs.keys())
    placeholders = ", ".join("?" for _ in cols)
    col_names = ", ".join(cols)  # nosec B608 — validated above
    values = [kwargs[c] for c in cols]
    async with get_db() as db:
        await db.execute(
            f"INSERT INTO generated_problems ({col_names}) VALUES ({placeholders})",  # nosec B608 — cols validated by _GENERATED_PROBLEM_COLS
            values,
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT * FROM generated_problems WHERE id = ?", (problem_id,),
        )
        row = await cursor.fetchone()
    assert row is not None
    return _row_to_dict(row, _GENERATED_PROBLEM_JSON_FIELDS)


async def get_generated_problem(problem_id: str) -> dict[str, Any] | None:
    """Get a single generated problem by ID."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM generated_problems WHERE id = ?", (problem_id,),
        )
        row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_dict(row, _GENERATED_PROBLEM_JSON_FIELDS)


_ALLOWED_SORT_COLUMNS = {
    "created_at", "overall_quality", "engagement_score", "complexity_score",
}


async def list_generated_problems(
    *,
    field: str | None = None,
    problem_type: str | None = None,
    project_id: str | None = None,
    workshop_type: str | None = None,
    search: str | None = None,
    min_quality: float | None = None,
    min_engagement: int | None = None,
    min_complexity: int | None = None,
    sort_by: str = "created_at",
    sort_desc: bool = True,
    page: int = 1,
    page_size: int = 10,
) -> dict[str, Any]:
    """List generated problems with optional filters, search, and sorting."""
    conditions: list[str] = []
    params: list[Any] = []
    if workshop_type:
        conditions.append("workshop_type = ?")
        params.append(workshop_type)
    if field:
        conditions.append("field = ?")
        params.append(field)
    if problem_type:
        conditions.append("problem_type = ?")
        params.append(problem_type)
    if project_id:
        conditions.append("project_id = ?")
        params.append(project_id)
    if search:
        like = f"%{search}%"
        conditions.append(
            "(title LIKE ? OR description LIKE ? OR investigation LIKE ?"
            " OR user_suggestion LIKE ?)"
        )
        params.extend([like, like, like, like])
    if min_quality is not None:
        conditions.append("overall_quality >= ?")
        params.append(min_quality)
    if min_engagement is not None:
        conditions.append("engagement_score >= ?")
        params.append(min_engagement)
    if min_complexity is not None:
        conditions.append("complexity_score >= ?")
        params.append(min_complexity)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    col = sort_by if sort_by in _ALLOWED_SORT_COLUMNS else "created_at"
    direction = "DESC" if sort_desc else "ASC"
    offset = (page - 1) * page_size
    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT COUNT(*) FROM generated_problems{where}", params,  # nosec B608 — where built from hardcoded conditions
        )
        row = await cursor.fetchone()
        assert row is not None
        total = row[0]
        cursor = await db.execute(
            f"SELECT * FROM generated_problems{where}"  # nosec B608 — where/col/direction from allowlist
            f" ORDER BY {col} {direction} LIMIT ? OFFSET ?",
            [*params, page_size, offset],
        )
        rows = await cursor.fetchall()
    return {
        "data": [_row_to_dict(r, _GENERATED_PROBLEM_JSON_FIELDS) for r in rows],
        "total": total,
    }


async def delete_generated_problem(problem_id: str) -> bool:
    """Delete a generated problem. Returns True if deleted."""
    async with get_db() as db:
        cursor = await db.execute(
            "DELETE FROM generated_problems WHERE id = ?", (problem_id,),
        )
        await db.commit()
    return cursor.rowcount > 0


async def get_generated_problem_titles(
    field: str,
    problem_type: str,
    limit: int = 20,
    workshop_type: str | None = None,
) -> list[str]:
    """Get titles of recently generated problems for novelty awareness."""
    conditions = ["field = ?", "problem_type = ?"]
    params: list[Any] = [field, problem_type]
    if workshop_type:
        conditions.append("workshop_type = ?")
        params.append(workshop_type)
    where = " AND ".join(conditions)
    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT title FROM generated_problems WHERE {where}"  # nosec B608 — where from hardcoded conditions
            " ORDER BY created_at DESC LIMIT ?",
            [*params, limit],
        )
        rows = await cursor.fetchall()
    return [row[0] for row in rows]


# ---------------------------------------------------------------------------
# External Tool Usage Tracking
# ---------------------------------------------------------------------------


async def store_tool_usage(
    tool: str,
    *,
    query: str | None = None,
    success: bool = True,
    error_category: str | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    """Record an external tool API call.

    ``session_id`` is read automatically from the LLM usage ContextVar
    (set by ``set_usage_context()`` before graph execution).
    """
    from research_mentor.llm import _usage_session_id

    session_id = _usage_session_id.get(None)

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO tool_usage
                (tool, query, success, error_category, elapsed_seconds, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (tool, query, int(success), error_category, elapsed_seconds, session_id),
        )
        await db.commit()


async def get_session_tool_usage(session_id: str) -> list[dict[str, Any]]:
    """Get per-tool request counts for a single session.

    Returns list of dicts: tool, count.
    """
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT tool, COUNT(*) as count
            FROM tool_usage
            WHERE session_id = ?
            GROUP BY tool
            ORDER BY count DESC
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def get_tool_usage_summary(period: str = "month") -> list[dict[str, Any]]:
    """Get per-tool aggregated usage for a period.

    Returns list of dicts: tool, total_requests, successful, failed,
    error_rate, errors_by_category, avg_elapsed, last_used.

    ``errors_by_category`` is a dict mapping error_category → count
    (e.g. ``{"timeout": 3, "rate_limit": 1}``).
    """
    ts_filter = _period_to_sql(period)
    where = f"WHERE {ts_filter}" if ts_filter else ""

    async with get_db() as db:
        cursor = await db.execute(
            f"""
            SELECT
                tool,
                COUNT(*) as total_requests,
                SUM(success) as successful,
                COUNT(*) - SUM(success) as failed,
                ROUND(
                    100.0 * (COUNT(*) - SUM(success)) / COUNT(*), 1
                ) as error_rate,
                ROUND(AVG(CASE WHEN elapsed_seconds > 0 THEN elapsed_seconds END), 3)
                    as avg_elapsed,
                MAX(timestamp) as last_used,
                SUM(CASE WHEN error_category = 'timeout' THEN 1 ELSE 0 END)
                    as errors_timeout,
                SUM(CASE WHEN error_category = 'rate_limit' THEN 1 ELSE 0 END)
                    as errors_rate_limit,
                SUM(CASE WHEN error_category = 'network' THEN 1 ELSE 0 END)
                    as errors_network,
                SUM(CASE WHEN error_category = 'auth' THEN 1 ELSE 0 END)
                    as errors_auth,
                SUM(CASE WHEN error_category = 'unavailable' THEN 1 ELSE 0 END)
                    as errors_unavailable
            FROM tool_usage
            {where}
            GROUP BY tool
            ORDER BY total_requests DESC
            """,  # nosec B608 — where built from _period_to_sql
        )
        rows = await cursor.fetchall()

    results = []
    for row in rows:
        d = dict(row)
        d["errors_by_category"] = {
            cat: d.pop(f"errors_{cat}")
            for cat in ("timeout", "rate_limit", "network", "auth", "unavailable")
            if d.get(f"errors_{cat}", 0) > 0
        }
        results.append(d)
    return results


async def get_tool_usage_monthly(tool: str | None = None) -> dict[str, Any]:
    """Get current-month request counts, useful for quota monitoring.

    Returns: total_this_month, per-tool breakdown.
    """
    where_clauses = ["timestamp >= datetime('now', 'start of month')"]
    params: list[Any] = []
    if tool:
        where_clauses.append("tool = ?")
        params.append(tool)

    where = f"WHERE {' AND '.join(where_clauses)}"

    async with get_db() as db:
        cursor = await db.execute(
            f"""
            SELECT
                tool,
                COUNT(*) as requests,
                SUM(success) as successful,
                COUNT(*) - SUM(success) as failed
            FROM tool_usage
            {where}
            GROUP BY tool
            """,  # nosec B608 — where built from hardcoded clauses
            params,
        )
        rows = await cursor.fetchall()

    return {row["tool"]: dict(row) for row in rows}


# ---------------------------------------------------------------------------
# Retraction Watch data cache
# ---------------------------------------------------------------------------


async def get_retraction_watch_last_refresh() -> str | None:
    """Return the last refresh timestamp, or None if never refreshed."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT value FROM retraction_watch_meta WHERE key = 'last_refresh'",
        )
        row = await cursor.fetchone()
    return row["value"] if row else None


async def set_retraction_watch_last_refresh(iso_timestamp: str) -> None:
    """Set the last refresh timestamp."""
    async with get_db() as db:
        await db.execute(
            """INSERT INTO retraction_watch_meta (key, value)
               VALUES ('last_refresh', ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (iso_timestamp,),
        )
        await db.commit()


async def get_retraction_watch_count() -> int:
    """Return the number of rows in retraction_watch."""
    async with get_db() as db:
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM retraction_watch")
        row = await cursor.fetchone()
    return row["cnt"] if row else 0


async def clear_retraction_watch() -> None:
    """Clear all retraction_watch data and embeddings (for full refresh)."""
    async with get_db() as db:
        await db.execute("DELETE FROM retraction_watch")
        await db.execute("DELETE FROM retraction_watch_embeddings")
        await db.execute(
            "DELETE FROM retraction_watch_meta WHERE key = 'last_refresh'",
        )
        await db.commit()


async def get_retraction_watch_content_hashes() -> dict[str, tuple[int, str]]:
    """Return ``{match_key: (row_id, content_hash)}`` for all rows.

    Used by the incremental refresh to diff against incoming CSV data.
    """
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, match_key, content_hash FROM retraction_watch "
            "WHERE match_key IS NOT NULL",
        )
        rows = await cursor.fetchall()
    return {row["match_key"]: (row["id"], row["content_hash"]) for row in rows}


async def get_retraction_watch_match_key_count() -> int:
    """Count rows that have a ``match_key`` set.

    Returns 0 for pre-V23 databases where the column exists but is all NULL.
    Used to detect first-run migration.
    """
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM retraction_watch "
            "WHERE match_key IS NOT NULL",
        )
        row = await cursor.fetchone()
    return row["cnt"] if row else 0


_RW_INSERT_COLS = (
    "original_doi, retraction_doi, title, authors, journal, "
    "publisher, country, subject, retraction_date, "
    "retraction_nature, reason, article_type, paywalled, "
    "urls, updated_at, abstract, match_key, content_hash"
)

_RW_UPDATE_COLS = (
    "original_doi=?, retraction_doi=?, title=?, authors=?, journal=?, "
    "publisher=?, country=?, subject=?, retraction_date=?, "
    "retraction_nature=?, reason=?, article_type=?, paywalled=?, "
    "urls=?, updated_at=?, abstract=?, content_hash=?"
)


def _row_values(row: dict[str, Any]) -> tuple[Any, ...]:
    """Extract column values from a row dict (insert order)."""
    return (
        row.get("original_doi"),
        row.get("retraction_doi"),
        row["title"],
        row.get("authors"),
        row.get("journal"),
        row.get("publisher"),
        row.get("country"),
        row.get("subject"),
        row.get("retraction_date"),
        row["retraction_nature"],
        row.get("reason"),
        row.get("article_type"),
        row.get("paywalled"),
        row.get("urls"),
        row["updated_at"],
        row.get("abstract"),
        row["match_key"],
        row["content_hash"],
    )


def _row_update_values(row: dict[str, Any]) -> tuple[Any, ...]:
    """Extract column values for UPDATE SET clause (no match_key)."""
    return (
        row.get("original_doi"),
        row.get("retraction_doi"),
        row["title"],
        row.get("authors"),
        row.get("journal"),
        row.get("publisher"),
        row.get("country"),
        row.get("subject"),
        row.get("retraction_date"),
        row["retraction_nature"],
        row.get("reason"),
        row.get("article_type"),
        row.get("paywalled"),
        row.get("urls"),
        row["updated_at"],
        row.get("abstract"),
        row["content_hash"],
    )


async def sync_retraction_watch_incremental(
    new_rows: list[dict[str, Any]],
    new_embeddings: list[bytes],
    changed_rows: list[dict[str, Any]],
    changed_embeddings: list[bytes],
    changed_row_ids: list[int],
    deleted_row_ids: list[int],
) -> tuple[int, int, int]:
    """Apply incremental changes to retraction_watch.

    Returns ``(inserted, updated, deleted)`` counts.
    """
    async with get_db() as db:
        # Inserts
        placeholders = ", ".join("?" * 18)
        for row, emb in zip(new_rows, new_embeddings, strict=True):
            cursor = await db.execute(
                f"INSERT INTO retraction_watch ({_RW_INSERT_COLS}) "  # nosec B608 — _RW_INSERT_COLS is a constant
                f"VALUES ({placeholders})",
                _row_values(row),
            )
            rowid = cursor.lastrowid
            if rowid is not None:
                await db.execute(
                    "INSERT INTO retraction_watch_embeddings(rowid, embedding) "
                    "VALUES (?, ?)",
                    (rowid, emb),
                )

        # Updates
        for row, emb, old_id in zip(
            changed_rows, changed_embeddings, changed_row_ids, strict=True,
        ):
            await db.execute(
                f"UPDATE retraction_watch SET {_RW_UPDATE_COLS} WHERE id = ?",  # nosec B608 — _RW_UPDATE_COLS is a constant
                (*_row_update_values(row), old_id),
            )
            await db.execute(
                "DELETE FROM retraction_watch_embeddings WHERE rowid = ?",
                (old_id,),
            )
            await db.execute(
                "INSERT INTO retraction_watch_embeddings(rowid, embedding) "
                "VALUES (?, ?)",
                (old_id, emb),
            )

        # Deletes
        for row_id in deleted_row_ids:
            await db.execute(
                "DELETE FROM retraction_watch_embeddings WHERE rowid = ?",
                (row_id,),
            )
            await db.execute(
                "DELETE FROM retraction_watch WHERE id = ?",
                (row_id,),
            )

        await db.commit()

    inserted = len(new_rows)
    updated = len(changed_rows)
    deleted = len(deleted_row_ids)
    logger.info(
        "Retraction Watch sync: {} inserted, {} updated, {} deleted",
        inserted, updated, deleted,
    )
    return inserted, updated, deleted


async def get_retraction_watch_by_doi(doi: str) -> dict[str, Any] | None:
    """Look up a retraction_watch entry by original or retraction DOI."""
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT * FROM retraction_watch
               WHERE original_doi = ? OR retraction_doi = ?
               LIMIT 1""",
            (doi, doi),
        )
        row = await cursor.fetchone()
    return dict(row) if row else None


async def search_retraction_watch_semantic(
    query_embedding: bytes,
    *,
    nature: str | list[str] | None = None,
    limit: int = 10,
    threshold: float = 0.35,
) -> list[dict[str, Any]]:
    """Semantic search over retraction_watch via sqlite-vec KNN.

    Same pattern as search_similar_memories: join vec0 virtual table
    with retraction_watch on rowid, filter by cosine distance.
    """
    max_distance = 1.0 - threshold

    # Build nature filter clause
    nature_clause = ""
    nature_params: list[Any] = []
    if nature is not None:
        if isinstance(nature, str):
            nature_clause = "AND rw.retraction_nature = ?"
            nature_params = [nature]
        else:
            placeholders = ", ".join("?" * len(nature))
            nature_clause = f"AND rw.retraction_nature IN ({placeholders})"
            nature_params = list(nature)

    async with get_db() as db:
        cursor = await db.execute(
            f"""SELECT rw.*, vec.distance
                FROM retraction_watch_embeddings vec
                JOIN retraction_watch rw ON rw.id = vec.rowid
                WHERE vec.embedding MATCH ?
                  AND vec.k = ?
                  {nature_clause}
                ORDER BY vec.distance ASC""",  # nosec B608 — nature_clause from hardcoded filter
            [query_embedding, limit * 3, *nature_params],
        )
        rows = await cursor.fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        if row["distance"] <= max_distance and len(results) < limit:
            d = dict(row)
            d["similarity"] = 1.0 - d.pop("distance")
            results.append(d)

    return results


# ---------------------------------------------------------------------------
# Collaboration opportunities
# ---------------------------------------------------------------------------


async def store_collaboration_opportunity(
    project_id: str,
    *,
    gap_type: str,
    need_description: str,
    resource_type: str,
    resource_name: str,
    resource_affiliation: str | None = None,
    resource_url: str | None = None,
    resource_description: str | None = None,
    openalex_author_id: str | None = None,
    openalex_institution_id: str | None = None,
    proximity: str | None = None,
    expires_at: str | None = None,
    session_id: str | None = None,
) -> int:
    """Store a collaboration opportunity found by the scout.

    Returns the row ID of the inserted opportunity.
    """
    if expires_at is None:
        from research_mentor.config import load_config

        cfg = load_config()
        expiry_days = int(cfg.collaboration.result_expiry_days)
        # Compute expiry in Python to avoid SQL interpolation
        from datetime import datetime, timedelta, timezone

        expires_at = (
            datetime.now(timezone.utc) + timedelta(days=expiry_days)
        ).isoformat()

    async with get_db() as db:
        cursor = await db.execute(
            """
            INSERT INTO collaboration_opportunities
                (project_id, gap_type, need_description,
                 resource_type, resource_name, resource_affiliation,
                 resource_url, resource_description,
                 openalex_author_id, openalex_institution_id,
                 proximity, expires_at, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (project_id, gap_type, need_description,
             resource_type, resource_name, resource_affiliation,
             resource_url, resource_description,
             openalex_author_id, openalex_institution_id,
             proximity, expires_at, session_id),
        )
        await db.commit()
        return cursor.lastrowid  # type: ignore[return-value]


async def get_collaboration_opportunities(
    project_id: str,
    *,
    gap_type: str | None = None,
    unsurfaced_only: bool = False,
    exclude_expired: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve collaboration opportunities for a project."""
    clauses = ["project_id = ?"]
    params: list[str] = [project_id]

    if gap_type:
        clauses.append("gap_type = ?")
        params.append(gap_type)
    if unsurfaced_only:
        clauses.append("surfaced_to_student = 0")
    if exclude_expired:
        clauses.append("expires_at > datetime('now')")

    where = " AND ".join(clauses)

    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT * FROM collaboration_opportunities WHERE {where} ORDER BY created_at DESC",  # nosec B608 — where from hardcoded clauses
            params,
        )
        rows = await cursor.fetchall()

    return [dict(row) for row in rows]


async def mark_opportunity_surfaced(opportunity_id: int) -> None:
    """Mark a collaboration opportunity as surfaced to the student."""
    async with get_db() as db:
        await db.execute(
            "UPDATE collaboration_opportunities SET surfaced_to_student = 1 WHERE id = ?",
            (opportunity_id,),
        )
        await db.commit()


async def update_opportunity_response(opportunity_id: int, response: str) -> None:
    """Update the student's response to a collaboration opportunity."""
    async with get_db() as db:
        await db.execute(
            "UPDATE collaboration_opportunities SET student_response = ? WHERE id = ?",
            (response, opportunity_id),
        )
        await db.commit()
