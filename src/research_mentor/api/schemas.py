"""Pydantic request/response models for REST API endpoints."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class PaginationInfo(BaseModel):
    page: int
    page_size: int
    total: int
    total_pages: int


def _pagination_info(page: int, page_size: int, total: int) -> PaginationInfo:
    return PaginationInfo(
        page=page,
        page_size=page_size,
        total=total,
        total_pages=max(1, math.ceil(total / page_size)),
    )

# ---------------------------------------------------------------------------
# Personas (read-only)
# ---------------------------------------------------------------------------


class PersonaResponse(BaseModel):
    persona_id: str
    full_name: str
    category: str
    birth_year: int | None = None
    death_year: int | None = None
    brief_description: str | None = None
    biography: str | None = None
    notable_works: list[str] = Field(default_factory=list)
    key_achievements: list[str] = Field(default_factory=list)
    teaching_style_notes: str | None = None
    persona_type: str
    is_active: bool
    display_order: int


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


class ProjectCreate(BaseModel):
    title: str
    research_question: str
    status: str = "draft"
    start_date: str | None = None
    end_date: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class ProjectUpdate(BaseModel):
    title: str | None = None
    research_question: str | None = None
    status: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    metadata: dict[str, object] | None = None


class ProjectResponse(BaseModel):
    id: str
    title: str
    research_question: str
    status: str
    start_date: str | None = None
    end_date: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class ProjectDetailResponse(ProjectResponse):
    milestones: list[MilestoneResponse] = Field(default_factory=list)
    latest_assessment: AssessmentResponse | None = None


# ---------------------------------------------------------------------------
# Milestones
# ---------------------------------------------------------------------------


class MilestoneCreate(BaseModel):
    title: str
    description: str | None = None
    status: str = "pending"
    start_date: str | None = None
    due_date: str | None = None
    display_order: int = 0
    metadata: dict[str, object] = Field(default_factory=dict)


class MilestoneUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    status: str | None = None
    start_date: str | None = None
    due_date: str | None = None
    completed_at: str | None = None
    display_order: int | None = None
    metadata: dict[str, object] | None = None


class MilestoneResponse(BaseModel):
    id: str
    project_id: str
    title: str
    description: str | None = None
    status: str
    start_date: str | None = None
    due_date: str | None = None
    completed_at: str | None = None
    display_order: int
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class SessionCreate(BaseModel):
    title: str | None = None
    persona_id: str | None = None
    language: str = "en"
    metadata: dict[str, object] = Field(default_factory=dict)


class SessionUpdate(BaseModel):
    title: str | None = None
    persona_id: str | None = None
    is_active: bool | None = None
    language: str | None = None
    metadata: dict[str, object] | None = None


class SessionResponse(BaseModel):
    session_id: str
    project_id: str
    title: str | None = None
    persona_id: str | None = None
    is_active: bool
    language: str
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: str
    last_active_at: str


# ---------------------------------------------------------------------------
# Memories
# ---------------------------------------------------------------------------


class MemoryCreate(BaseModel):
    memory_text: str
    memory_type: str
    session_id: str | None = None
    importance: int = Field(default=5, ge=1, le=10)
    conversation_context: dict[str, object] | None = None
    tags: list[str] | None = None


class MemorySearchRequest(BaseModel):
    query: str
    threshold: float = 0.35
    limit: int = 5


class MemoryResponse(BaseModel):
    id: str
    memory_text: str
    memory_type: str
    importance: int
    tags: list[str] = Field(default_factory=list)
    created_at: str


class MemorySearchResponse(MemoryResponse):
    similarity: float


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


class ArtifactCreate(BaseModel):
    artifact_type: str
    file_name: str
    file_path: str
    mime_type: str
    file_size_bytes: int
    description: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class ArtifactUpdate(BaseModel):
    artifact_type: str | None = None
    file_name: str | None = None
    file_path: str | None = None
    mime_type: str | None = None
    file_size_bytes: int | None = None
    description: str | None = None
    llm_summary: str | None = None
    metadata: dict[str, object] | None = None


class ArtifactResponse(BaseModel):
    id: str
    project_id: str
    artifact_type: str
    file_name: str
    file_path: str
    mime_type: str
    file_size_bytes: int
    description: str | None = None
    extracted_text: str | None = None
    llm_summary: str | None = None
    llm_summary_generated_at: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Assessments (read-only)
# ---------------------------------------------------------------------------


class AssessmentResponse(BaseModel):
    id: str
    project_id: str
    assessment_date: str
    health_score: int | None = None
    issues: list[object] = Field(default_factory=list)
    recommended_actions: list[object] = Field(default_factory=list)
    general_notes: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Student Assessments (cross-project)
# ---------------------------------------------------------------------------


class StudentAssessmentResponse(BaseModel):
    id: str
    assessment_date: str
    overall_skill_level: str | None = None
    research_maturity_score: int | None = None
    growth_trajectory: str | None = None
    key_strengths: list[str] = Field(default_factory=list)
    growth_areas: list[str] = Field(default_factory=list)
    cross_project_patterns: str | None = None
    recommendations: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    # Per-domain skill dimensions (V10 schema, used by progress_assessor)
    avg_critical_reading: int | None = None
    avg_data_analysis_skill: int | None = None
    avg_experimental_design_skill: int | None = None
    avg_writing_skill: int | None = None
    avg_critical_thinking_skill: int | None = None
    avg_time_management_skill: int | None = None
    avg_collaboration_skill: int | None = None
    avg_research_maturity: int | None = None
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Student Profile
# ---------------------------------------------------------------------------


class DomainExpertise(BaseModel):
    """A single domain expertise entry — e.g. Physics at AP level."""

    domain: str = Field(..., min_length=1, max_length=200)
    level: str = Field(..., min_length=1, max_length=100)  # e.g. "AP", "Bachelor's", "PhD"


class ProfessionalExperience(BaseModel):
    """A single professional experience entry."""

    field: str = Field(..., min_length=1, max_length=200)
    years: int = Field(..., ge=0, le=80)


class StudentProfileUpdate(BaseModel):
    name: str | None = None
    age: int | None = Field(default=None, ge=10, le=120)
    country: str | None = None
    grade: str | None = None  # e.g. "10th", "AP", "IB Year 2"
    college_year: str | None = None  # e.g. "freshman", "sophomore", "PhD year 2"
    language: str | None = None  # e.g. "en", "es", "ru"
    persona: str | None = None  # e.g. "newton", "curie"
    # middle_school | high_school | associate | bachelors | masters | phd | postdoc | professional
    education_level: str | None = None
    domain_expertise: list[DomainExpertise] | None = None
    professional_experience: list[ProfessionalExperience] | None = None
    background_notes: str | None = None  # free-form additional context
    institution: str | None = None  # e.g. "University of Michigan"
    city: str | None = None  # e.g. "Ann Arbor"
    state_region: str | None = None  # e.g. "Michigan"
    zip_code: str | None = None  # e.g. "48109" (US) or postal code (international)
    license_accepted: bool | None = None
    onboarding_completed: bool | None = None
    dark_mode: str | None = None  # "dark", "light", "system"
    view_mode: str | None = None  # "power", "simple"
    active_project_id: str | None = None


class StudentProfileResponse(BaseModel):
    name: str | None = None
    age: int | None = None
    country: str | None = None
    grade: str | None = None
    college_year: str | None = None
    language: str | None = None
    persona: str | None = None
    education_level: str | None = None
    domain_expertise: list[DomainExpertise] = Field(default_factory=list)
    professional_experience: list[ProfessionalExperience] = Field(default_factory=list)
    background_notes: str | None = None
    institution: str | None = None
    institution_ror: str | None = None  # auto-resolved from institution name
    city: str | None = None
    state_region: str | None = None
    zip_code: str | None = None
    license_accepted: bool = False
    onboarding_completed: bool = False
    dark_mode: str = "system"
    view_mode: str = "simple"
    active_project_id: str | None = None
    updated_at: str | None = None


# ---------------------------------------------------------------------------
# Paginated response wrappers
# ---------------------------------------------------------------------------


class PaginatedPersonas(BaseModel):
    data: list[PersonaResponse]
    pagination: PaginationInfo


class PaginatedProjects(BaseModel):
    data: list[ProjectResponse]
    pagination: PaginationInfo


class PaginatedMilestones(BaseModel):
    data: list[MilestoneResponse]
    pagination: PaginationInfo


class PaginatedSessions(BaseModel):
    data: list[SessionResponse]
    pagination: PaginationInfo


class PaginatedMemories(BaseModel):
    data: list[MemoryResponse]
    pagination: PaginationInfo


class PaginatedArtifacts(BaseModel):
    data: list[ArtifactResponse]
    pagination: PaginationInfo


class PaginatedAssessments(BaseModel):
    data: list[AssessmentResponse]
    pagination: PaginationInfo


class PaginatedStudentAssessments(BaseModel):
    data: list[StudentAssessmentResponse]
    pagination: PaginationInfo


# Forward reference updates
ProjectDetailResponse.model_rebuild()
