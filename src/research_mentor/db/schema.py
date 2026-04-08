"""SQLite schema for Personal Research Mentor.

Single-user adaptation of the hosted PostgreSQL schema. Key differences:
- UUID → TEXT (SQLite has no native UUID)
- JSONB → TEXT (stored as JSON strings)
- Embeddings stored as BLOB (struct-packed float32, configurable dim)
- No multi-tenant fields (organization_id, supervisor_id removed)
- Timestamps stored as ISO 8601 TEXT
"""

from __future__ import annotations

# Schema version — increment when adding migrations
SCHEMA_VERSION = 29

# All tables created in a single migration for v1
SCHEMA_V1 = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Teaching personas (pre-1900 scientists + modern approaches)
CREATE TABLE IF NOT EXISTS teaching_personas (
    persona_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    category TEXT NOT NULL,
    birth_year INTEGER,
    death_year INTEGER,
    brief_description TEXT,
    biography TEXT,
    notable_works TEXT,  -- JSON array
    key_achievements TEXT,  -- JSON array
    teaching_style_notes TEXT,
    persona_type TEXT NOT NULL DEFAULT 'historical',
    is_active INTEGER NOT NULL DEFAULT 1,
    display_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Research projects
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    research_question TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    work_level TEXT NOT NULL DEFAULT 'intermediate',  -- dropped in V12
    start_date TEXT,
    end_date TEXT,
    metadata TEXT DEFAULT '{}',  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Project milestones
CREATE TABLE IF NOT EXISTS milestones (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    start_date TEXT,
    due_date TEXT,
    completed_at TEXT,
    display_order INTEGER NOT NULL DEFAULT 0,
    metadata TEXT DEFAULT '{}',  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Project artifacts (files, documents, code)
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    artifact_type TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    description TEXT,
    llm_summary TEXT,
    llm_summary_generated_at TEXT,
    metadata TEXT DEFAULT '{}',  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Project assessments (AI-generated health checks)
CREATE TABLE IF NOT EXISTS project_assessments (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    assessment_date TEXT NOT NULL,
    health_score INTEGER CHECK (health_score >= 0 AND health_score <= 100),
    issues TEXT DEFAULT '[]',  -- JSON array
    recommended_actions TEXT DEFAULT '[]',  -- JSON array
    general_notes TEXT,
    metadata TEXT DEFAULT '{}',  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Chat sessions
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT,
    persona_id TEXT REFERENCES teaching_personas(persona_id),
    is_active INTEGER NOT NULL DEFAULT 1,
    language TEXT NOT NULL DEFAULT 'en',
    metadata TEXT DEFAULT '{}',  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_active_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Conversation memories (cross-session semantic memory)
CREATE TABLE IF NOT EXISTS conversation_memories (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
    session_id TEXT REFERENCES sessions(session_id),
    memory_text TEXT NOT NULL CHECK (length(memory_text) > 0),
    memory_type TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 5 CHECK (importance >= 1 AND importance <= 10),
    conversation_context TEXT,  -- JSON
    tags TEXT,  -- JSON array
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    accessed_at TEXT,
    access_count INTEGER NOT NULL DEFAULT 0
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_milestones_project ON milestones(project_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project_id);
CREATE INDEX IF NOT EXISTS idx_assessments_project ON project_assessments(project_id);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_memories_project ON conversation_memories(project_id);
CREATE INDEX IF NOT EXISTS idx_memories_session ON conversation_memories(session_id);
CREATE INDEX IF NOT EXISTS idx_personas_active ON teaching_personas(is_active);
"""

# V2: Add sqlite-vec virtual table for semantic memory search.
# Uses cosine distance (1 - cosine_similarity) — lower = more similar.
# The vec0 table stores embeddings keyed by memory_id (rowid is an integer,
# so we use a separate mapping via conversation_memories.rowid).
#
# Architecture:
# - conversation_memories: metadata (text, type, importance, tags)
# - memory_embeddings: vec0 virtual table (rowid → float[384] embedding)
# - Join on rowid to get metadata + similarity in one query
SCHEMA_V2 = """
-- Vector index for semantic search (sqlite-vec)
CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
    embedding float[384] distance_metric=cosine
);

-- Index for efficient memory lookup by type + importance
CREATE INDEX IF NOT EXISTS idx_memories_type_importance
    ON conversation_memories(project_id, memory_type, importance DESC);
"""

# V3: Add artifact text extraction and embedding tables.
# Extracts text from uploaded files (pdfplumber/python-docx/csv), chunks it, and embeds
# for semantic search. Same pattern as conversation_memories → memory_embeddings.
#
# Architecture:
# - artifacts.extracted_text: full extracted text stored on the artifact row
# - artifact_chunks: individual text chunks with ordering
# - artifact_embeddings: vec0 virtual table (rowid → float[384] embedding)
# - Join artifact_chunks on rowid to get chunk_text + similarity
SCHEMA_V3 = """
-- Add extracted text column to artifacts
ALTER TABLE artifacts ADD COLUMN extracted_text TEXT;

-- Artifact text chunks (for embedding search)
CREATE TABLE IF NOT EXISTS artifact_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_artifact_chunks_artifact ON artifact_chunks(artifact_id);

-- Vector index for artifact semantic search (sqlite-vec)
CREATE VIRTUAL TABLE IF NOT EXISTS artifact_embeddings USING vec0(
    embedding float[384] distance_metric=cosine
);
"""

# V4: Add student profile table (singleton — single-user app).
# Stores user-provided demographic settings that agent nodes can use
# to adapt guidance (age-appropriate language, country-specific curricula, etc.).
SCHEMA_V4 = """
CREATE TABLE IF NOT EXISTS student_profile (
    id INTEGER PRIMARY KEY,  -- singleton row (id=1)
    age INTEGER,
    country TEXT,
    grade TEXT,           -- e.g. "10th", "AP", "IB Year 2"
    college_year TEXT,    -- e.g. "freshman", "sophomore", "PhD year 2"
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# V5: Add name column to student_profile.
SCHEMA_V5 = """
ALTER TABLE student_profile ADD COLUMN name TEXT;
"""

# V6: Add preferred language to student_profile.
SCHEMA_V6 = """
ALTER TABLE student_profile ADD COLUMN language TEXT DEFAULT 'en';
"""

# V7: Add default persona preference to student_profile.
SCHEMA_V7 = """
ALTER TABLE student_profile ADD COLUMN persona TEXT DEFAULT 'newton';
"""

# V8: Add cross-project student assessments table.
# Tracks overall student skill development across all projects,
# unlike project_assessments which are per-project health checks.
SCHEMA_V8 = """
CREATE TABLE IF NOT EXISTS student_assessments (
    id TEXT PRIMARY KEY,
    assessment_date TEXT NOT NULL,
    overall_skill_level TEXT,
    research_maturity_score INTEGER CHECK (
        research_maturity_score >= 0 AND research_maturity_score <= 100
    ),
    growth_trajectory TEXT,
    key_strengths TEXT DEFAULT '[]',
    growth_areas TEXT DEFAULT '[]',
    cross_project_patterns TEXT,
    recommendations TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# V9: Add LLM usage tracking table.
# Tracks token usage across all backends (claude-cli, vllm, api)
# for observability and budget enforcement (api backend only).
SCHEMA_V9 = """
CREATE TABLE IF NOT EXISTS llm_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    backend TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER GENERATED ALWAYS AS (prompt_tokens + completion_tokens) STORED,
    session_id TEXT,
    project_id TEXT,
    call_type TEXT NOT NULL DEFAULT 'chat',
    elapsed_seconds REAL,
    error INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_llm_usage_timestamp ON llm_usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_usage_backend_provider_model
    ON llm_usage(backend, provider, model);
CREATE INDEX IF NOT EXISTS idx_llm_usage_project ON llm_usage(project_id);
CREATE INDEX IF NOT EXISTS idx_llm_usage_session ON llm_usage(session_id);
"""

# V10: Add per-domain skill dimension columns to student_assessments.
# These 8 columns match what progress_assessor expects for domain-specific
# routing (avg_critical_reading, avg_data_analysis_skill, etc.).
# Without these, the assessor's domain routing is dead code — it always
# falls back to overall_progress because the fields are never populated.
SCHEMA_V10 = """
ALTER TABLE student_assessments ADD COLUMN avg_critical_reading INTEGER;
ALTER TABLE student_assessments ADD COLUMN avg_data_analysis_skill INTEGER;
ALTER TABLE student_assessments ADD COLUMN avg_experimental_design_skill INTEGER;
ALTER TABLE student_assessments ADD COLUMN avg_writing_skill INTEGER;
ALTER TABLE student_assessments ADD COLUMN avg_critical_thinking_skill INTEGER;
ALTER TABLE student_assessments ADD COLUMN avg_time_management_skill INTEGER;
ALTER TABLE student_assessments ADD COLUMN avg_collaboration_skill INTEGER;
ALTER TABLE student_assessments ADD COLUMN avg_research_maturity INTEGER;
"""

# V11: Add title management columns to sessions.
# - title_source: 'auto' (LLM-generated) or 'user' (manually renamed) — user titles
#   are never overwritten by auto-review.
# - message_count: incremented on each chat turn, used to decide when to re-evaluate
#   the auto-generated title (at turns 5 and 15).
SCHEMA_V11 = """
ALTER TABLE sessions ADD COLUMN title_source TEXT NOT NULL DEFAULT 'auto';
ALTER TABLE sessions ADD COLUMN message_count INTEGER NOT NULL DEFAULT 0;
"""

# V12: Remove work_level column from projects.
# Redundant with the per-request guidance_level sent on each chat turn.
SCHEMA_V12 = """
ALTER TABLE projects DROP COLUMN work_level;
"""

# V13: Question Workshop — generated research problems.
SCHEMA_V13 = """
CREATE TABLE IF NOT EXISTS generated_problems (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
    field TEXT NOT NULL,
    problem_type TEXT NOT NULL,
    domains TEXT,
    user_suggestion TEXT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    investigation TEXT NOT NULL,
    core_concepts TEXT,
    materials TEXT,
    feasibility TEXT NOT NULL,
    recommended TEXT NOT NULL,
    safety_level TEXT,
    complexity_score INTEGER,
    engagement_score INTEGER,
    overall_quality REAL,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_generated_problems_field_type
    ON generated_problems(field, problem_type);
CREATE INDEX IF NOT EXISTS idx_generated_problems_project
    ON generated_problems(project_id);
"""

# V14: Add purpose column to llm_usage.
# Tracks which feature generated the usage (chat, question_workshop,
# assessment, title, artifact). Defaults to 'chat' for existing rows.
SCHEMA_V14 = """
ALTER TABLE llm_usage ADD COLUMN purpose TEXT NOT NULL DEFAULT 'chat';
"""

# V15: Question Workshop — add workshop_type to distinguish pipeline types.
SCHEMA_V15 = """
ALTER TABLE generated_problems ADD COLUMN workshop_type TEXT NOT NULL DEFAULT 'one_shot';
CREATE INDEX IF NOT EXISTS idx_generated_problems_workshop_type
    ON generated_problems(workshop_type);
"""

# V16: External tool usage tracking (web search, academic search APIs).
# Records every external API call for observability, quota monitoring
# (especially Brave Search monthly limit), and the GenAI Utilization page.
SCHEMA_V16 = """
CREATE TABLE IF NOT EXISTS tool_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    tool TEXT NOT NULL,
    query TEXT,
    success INTEGER NOT NULL DEFAULT 1,
    error_category TEXT,
    elapsed_seconds REAL
);

CREATE INDEX IF NOT EXISTS idx_tool_usage_timestamp ON tool_usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_tool_usage_tool ON tool_usage(tool);
"""

# V17: Workshop sessions — lightweight metadata index for Hypothesis (and future
# workshops). State lives in the LangGraph checkpointer; this table provides
# listing, resuming, and status tracking.
SCHEMA_V17 = """
CREATE TABLE IF NOT EXISTS workshop_sessions (
    session_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    workshop_type TEXT NOT NULL,  -- 'hypothesis', 'phenomenon', 'claims'
    status TEXT NOT NULL DEFAULT 'active',  -- 'active', 'completed', 'abandoned'
    current_stage TEXT,  -- e.g. 'stage_1_discovery', 'stage_2_literature'
    title TEXT,  -- auto-generated from variables once identified
    language TEXT NOT NULL DEFAULT 'en',
    metadata TEXT DEFAULT '{}',  -- JSON (variables, hypothesis, etc.)
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_active_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_workshop_sessions_project
    ON workshop_sessions(project_id, workshop_type);
CREATE INDEX IF NOT EXISTS idx_workshop_sessions_status
    ON workshop_sessions(project_id, status);
"""

# V18: Add session_id to tool_usage for per-session tool tracking.
SCHEMA_V18 = """
ALTER TABLE tool_usage ADD COLUMN session_id TEXT;
CREATE INDEX IF NOT EXISTS idx_tool_usage_session ON tool_usage(session_id);
"""

# V19: Rename workshop types — one_shot→phenomenon, claim_investigator→claims,
# sage→hypothesis, nova→gaps. Updates stored values in all affected tables.
SCHEMA_V19 = """
UPDATE generated_problems SET workshop_type = 'phenomenon' WHERE workshop_type = 'one_shot';
UPDATE generated_problems SET workshop_type = 'claims' WHERE workshop_type = 'claim_investigator';
UPDATE generated_problems SET workshop_type = 'hypothesis' WHERE workshop_type = 'sage';
UPDATE generated_problems SET workshop_type = 'gaps' WHERE workshop_type = 'nova';

UPDATE workshop_sessions SET workshop_type = 'phenomenon' WHERE workshop_type = 'one_shot';
UPDATE workshop_sessions SET workshop_type = 'claims' WHERE workshop_type = 'claim_investigator';
UPDATE workshop_sessions SET workshop_type = 'hypothesis' WHERE workshop_type = 'sage';
UPDATE workshop_sessions SET workshop_type = 'gaps' WHERE workshop_type = 'nova';

UPDATE llm_usage SET purpose = 'workshop_phenomenon' WHERE purpose = 'workshop_oneshot';
UPDATE llm_usage SET purpose = 'workshop_claims' WHERE purpose = 'workshop_investigator';
UPDATE llm_usage SET purpose = 'workshop_hypothesis' WHERE purpose = 'workshop_sage';
UPDATE llm_usage SET purpose = 'workshop_gaps' WHERE purpose = 'workshop_nova';
"""

# V20: Rename purpose='chat' → 'office' in llm_usage (Chat page renamed to Office).
SCHEMA_V20 = """
UPDATE llm_usage SET purpose = 'office' WHERE purpose = 'chat';
"""

# V21: Retraction Watch data cache — shared by Questioned and Retractions workshops.
# Populated from the daily-updated CSV at gitlab.com/crossref/retraction-watch-data.
# Covers retractions, expressions of concern, and corrections with reasons.
# Embeddings enable semantic search (same pattern as memory_embeddings, artifact_embeddings).
SCHEMA_V21 = """
CREATE TABLE IF NOT EXISTS retraction_watch (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_doi TEXT,
    retraction_doi TEXT,
    title TEXT NOT NULL,
    authors TEXT,
    journal TEXT,
    publisher TEXT,
    country TEXT,
    subject TEXT,
    retraction_date TEXT,
    retraction_nature TEXT NOT NULL,
    reason TEXT,
    article_type TEXT,
    paywalled TEXT,
    urls TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rw_nature ON retraction_watch(retraction_nature);
CREATE INDEX IF NOT EXISTS idx_rw_subject ON retraction_watch(subject);
CREATE INDEX IF NOT EXISTS idx_rw_original_doi ON retraction_watch(original_doi);

CREATE VIRTUAL TABLE IF NOT EXISTS retraction_watch_embeddings USING vec0(
    embedding float[384] distance_metric=cosine
);

CREATE TABLE IF NOT EXISTS retraction_watch_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# V22: Add education level, domain expertise, professional experience, and
# free-form background notes to student profile.  Expertise and experience
# are stored as JSON arrays validated by Pydantic on read.
SCHEMA_V22 = """
ALTER TABLE student_profile ADD COLUMN education_level TEXT;
ALTER TABLE student_profile ADD COLUMN domain_expertise TEXT DEFAULT '[]';
ALTER TABLE student_profile ADD COLUMN professional_experience TEXT DEFAULT '[]';
ALTER TABLE student_profile ADD COLUMN background_notes TEXT;
"""

# V23: Incremental Retraction Watch refresh — add abstract (from OpenAlex),
# match_key (stable identity across CSV refreshes), and content_hash
# (SHA-256 of CSV fields to detect changes).
SCHEMA_V23 = """
ALTER TABLE retraction_watch ADD COLUMN abstract TEXT;
ALTER TABLE retraction_watch ADD COLUMN match_key TEXT;
ALTER TABLE retraction_watch ADD COLUMN content_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_rw_match_key ON retraction_watch(match_key);
"""

# V24: Collaboration scout — student profile location + collaboration_opportunities table
SCHEMA_V24 = """
ALTER TABLE student_profile ADD COLUMN institution TEXT;
ALTER TABLE student_profile ADD COLUMN institution_ror TEXT;
ALTER TABLE student_profile ADD COLUMN city TEXT;
ALTER TABLE student_profile ADD COLUMN state_region TEXT;
ALTER TABLE student_profile ADD COLUMN zip_code TEXT;

CREATE TABLE IF NOT EXISTS collaboration_opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    gap_type TEXT NOT NULL,
    need_description TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_name TEXT NOT NULL,
    resource_affiliation TEXT,
    resource_url TEXT,
    resource_description TEXT,
    openalex_author_id TEXT,
    openalex_institution_id TEXT,
    proximity TEXT,
    surfaced_to_student INTEGER DEFAULT 0,
    student_response TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL,
    session_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_collab_opp_project
    ON collaboration_opportunities(project_id);
CREATE INDEX IF NOT EXISTS idx_collab_opp_gap_type
    ON collaboration_opportunities(project_id, gap_type);
"""

# V25: Move UI preferences from localStorage / config.toml into the database.
# These were previously stored client-side (lost on port change) or in config
# (never actually read by backend code). Now persisted in student_profile.
SCHEMA_V25 = """
ALTER TABLE student_profile ADD COLUMN license_accepted INTEGER NOT NULL DEFAULT 0;
ALTER TABLE student_profile ADD COLUMN onboarding_completed INTEGER NOT NULL DEFAULT 0;
ALTER TABLE student_profile ADD COLUMN dark_mode TEXT DEFAULT 'system';
ALTER TABLE student_profile ADD COLUMN view_mode TEXT DEFAULT 'simple';
ALTER TABLE student_profile ADD COLUMN active_project_id TEXT;
"""

# V26: Upgrade embedding tables from 384-dim to 768-dim (snowflake-arctic-embed-m).
# vec0 virtual tables cannot be ALTERed — must drop and recreate.
# Existing embeddings are lost; source data (conversation_memories, artifact_chunks,
# retraction_watch) is preserved. Re-embedding happens via rebuild_embedding_tables()
# on first server startup or when triggered from settings.
SCHEMA_V26 = """
DROP TABLE IF EXISTS memory_embeddings;
CREATE VIRTUAL TABLE memory_embeddings USING vec0(
    embedding float[768] distance_metric=cosine
);

DROP TABLE IF EXISTS artifact_embeddings;
CREATE VIRTUAL TABLE artifact_embeddings USING vec0(
    embedding float[768] distance_metric=cosine
);

DROP TABLE IF EXISTS retraction_watch_embeddings;
CREATE VIRTUAL TABLE retraction_watch_embeddings USING vec0(
    embedding float[768] distance_metric=cosine
);
"""

# V27: Add app_meta key-value table for tracking application-level state.
# Used to store which embedding model produced the current vectors so we can
# detect model changes (even between models with the same dimension).
SCHEMA_V27 = """
CREATE TABLE IF NOT EXISTS app_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR IGNORE INTO app_meta (key, value)
    VALUES ('embedding_model', 'Snowflake/snowflake-arctic-embed-m');
"""

# V28: Safety blocked queries log — persists across restarts, visible in Settings UI.
SCHEMA_V28 = """
CREATE TABLE IF NOT EXISTS safety_blocked_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    tool TEXT NOT NULL,
    query TEXT NOT NULL,
    block_type TEXT NOT NULL,
    reason TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_safety_blocked_ts
    ON safety_blocked_queries(timestamp);
"""

# V29: Add analysis column to artifacts.
# Stores AI-generated interpretations (image descriptions, pandas summaries, scipy results).
# Distinct from extracted_text which holds raw text extraction (PDF pages, DOCX paragraphs).
# Images: analysis only. Data: both (extracted_text = raw table, analysis = stats).
# Papers/documents: extracted_text only.
SCHEMA_V29 = """
ALTER TABLE artifacts ADD COLUMN analysis TEXT;
"""

# Verify queries — each must succeed (no error) on a correctly-migrated database.
# Used by the migration engine to confirm each migration landed correctly.
# Patterns:
#   SELECT col FROM table LIMIT 0           — verifies table + column exist (zero-cost)
#   SELECT rowid FROM vtable LIMIT 0        — verifies vec0 virtual table exists
#   SELECT 1 FROM sqlite_master WHERE ...   — verifies index/table metadata
VERIFY: dict[int, list[str]] = {
    1: [
        "SELECT version, applied_at FROM schema_version LIMIT 0",
        "SELECT persona_id, full_name, category FROM teaching_personas LIMIT 0",
        "SELECT id, title, research_question, status FROM projects LIMIT 0",
        "SELECT id, project_id, title, status FROM milestones LIMIT 0",
        "SELECT id, project_id, artifact_type, file_name FROM artifacts LIMIT 0",
        "SELECT id, project_id, health_score FROM project_assessments LIMIT 0",
        "SELECT session_id, project_id, is_active FROM sessions LIMIT 0",
        "SELECT id, memory_text, memory_type, importance FROM conversation_memories LIMIT 0",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_milestones_project'",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_artifacts_project'",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_assessments_project'",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_sessions_project'",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_memories_project'",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_memories_session'",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_personas_active'",
    ],
    2: [
        "SELECT rowid FROM memory_embeddings LIMIT 0",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_memories_type_importance'",
    ],
    3: [
        "SELECT extracted_text FROM artifacts LIMIT 0",
        "SELECT id, artifact_id, chunk_index, chunk_text FROM artifact_chunks LIMIT 0",
        "SELECT rowid FROM artifact_embeddings LIMIT 0",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_artifact_chunks_artifact'",
    ],
    4: [
        "SELECT id, age, country, grade, college_year FROM student_profile LIMIT 0",
    ],
    5: [
        "SELECT name FROM student_profile LIMIT 0",
    ],
    6: [
        "SELECT language FROM student_profile LIMIT 0",
    ],
    7: [
        "SELECT persona FROM student_profile LIMIT 0",
    ],
    8: [
        "SELECT id, assessment_date, overall_skill_level, research_maturity_score"
        " FROM student_assessments LIMIT 0",
    ],
    9: [
        "SELECT id, timestamp, backend, provider, model, prompt_tokens,"
        " completion_tokens, total_tokens FROM llm_usage LIMIT 0",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_llm_usage_timestamp'",
        "SELECT 1 FROM sqlite_master WHERE type='index'"
        " AND name='idx_llm_usage_backend_provider_model'",
    ],
    10: [
        "SELECT avg_critical_reading, avg_data_analysis_skill,"
        " avg_experimental_design_skill, avg_writing_skill,"
        " avg_critical_thinking_skill, avg_time_management_skill,"
        " avg_collaboration_skill, avg_research_maturity"
        " FROM student_assessments LIMIT 0",
    ],
    11: [
        "SELECT title_source, message_count FROM sessions LIMIT 0",
    ],
    12: [
        # work_level column should be gone after DROP COLUMN
        "SELECT 1 WHERE (SELECT count(*) FROM pragma_table_info('projects')"
        " WHERE name='work_level') = 0",
    ],
    13: [
        "SELECT id, project_id, field, problem_type, title, description,"
        " investigation, feasibility, recommended FROM generated_problems LIMIT 0",
        "SELECT 1 FROM sqlite_master WHERE type='index'"
        " AND name='idx_generated_problems_field_type'",
        "SELECT 1 FROM sqlite_master WHERE type='index'"
        " AND name='idx_generated_problems_project'",
    ],
    14: [
        "SELECT purpose FROM llm_usage LIMIT 0",
    ],
    15: [
        "SELECT workshop_type FROM generated_problems LIMIT 0",
        "SELECT 1 FROM sqlite_master WHERE type='index'"
        " AND name='idx_generated_problems_workshop_type'",
    ],
    16: [
        "SELECT id, timestamp, tool, query, success FROM tool_usage LIMIT 0",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_tool_usage_timestamp'",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_tool_usage_tool'",
    ],
    17: [
        "SELECT session_id, project_id, workshop_type, status, current_stage"
        " FROM workshop_sessions LIMIT 0",
        "SELECT 1 FROM sqlite_master WHERE type='index'"
        " AND name='idx_workshop_sessions_project'",
        "SELECT 1 FROM sqlite_master WHERE type='index'"
        " AND name='idx_workshop_sessions_status'",
    ],
    18: [
        "SELECT session_id FROM tool_usage LIMIT 0",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_tool_usage_session'",
    ],
    19: [
        # Data-only migration (renames). Tables verified by earlier versions.
        # No structural verify needed — the UPDATE statements are idempotent.
    ],
    20: [
        # Data-only migration (rename purpose). No structural verify needed.
    ],
    21: [
        "SELECT id, original_doi, title, retraction_nature FROM retraction_watch LIMIT 0",
        "SELECT rowid FROM retraction_watch_embeddings LIMIT 0",
        "SELECT key, value FROM retraction_watch_meta LIMIT 0",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_rw_nature'",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_rw_subject'",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_rw_original_doi'",
    ],
    22: [
        "SELECT education_level, domain_expertise, professional_experience,"
        " background_notes FROM student_profile LIMIT 0",
    ],
    23: [
        "SELECT abstract, match_key, content_hash FROM retraction_watch LIMIT 0",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_rw_match_key'",
    ],
    24: [
        "SELECT institution, institution_ror, city, state_region, zip_code"
        " FROM student_profile LIMIT 0",
        "SELECT id, project_id, gap_type, resource_type, resource_name"
        " FROM collaboration_opportunities LIMIT 0",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_collab_opp_project'",
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_collab_opp_gap_type'",
    ],
    25: [
        "SELECT license_accepted, onboarding_completed, dark_mode, view_mode,"
        " active_project_id FROM student_profile LIMIT 0",
    ],
    26: [
        "SELECT rowid FROM memory_embeddings LIMIT 0",
        "SELECT rowid FROM artifact_embeddings LIMIT 0",
        "SELECT rowid FROM retraction_watch_embeddings LIMIT 0",
    ],
    27: [
        "SELECT key, value FROM app_meta LIMIT 0",
        "SELECT value FROM app_meta WHERE key = 'embedding_model'",
    ],
    28: [
        "SELECT id, timestamp, tool, query, block_type, reason "
        "FROM safety_blocked_queries LIMIT 0",
        "SELECT 1 FROM sqlite_master WHERE type='index' "
        "AND name='idx_safety_blocked_ts'",
    ],
    29: [
        "SELECT analysis FROM artifacts LIMIT 0",
    ],
}

# Map of version → SQL to apply
MIGRATIONS: dict[int, str] = {
    1: SCHEMA_V1,
    2: SCHEMA_V2,
    3: SCHEMA_V3,
    4: SCHEMA_V4,
    5: SCHEMA_V5,
    6: SCHEMA_V6,
    7: SCHEMA_V7,
    8: SCHEMA_V8,
    9: SCHEMA_V9,
    10: SCHEMA_V10,
    11: SCHEMA_V11,
    12: SCHEMA_V12,
    13: SCHEMA_V13,
    14: SCHEMA_V14,
    15: SCHEMA_V15,
    16: SCHEMA_V16,
    17: SCHEMA_V17,
    18: SCHEMA_V18,
    19: SCHEMA_V19,
    20: SCHEMA_V20,
    21: SCHEMA_V21,
    22: SCHEMA_V22,
    23: SCHEMA_V23,
    24: SCHEMA_V24,
    25: SCHEMA_V25,
    26: SCHEMA_V26,
    27: SCHEMA_V27,
    28: SCHEMA_V28,
    29: SCHEMA_V29,
}
