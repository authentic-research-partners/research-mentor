"""Pydantic schemas for the Questioned workshop pipeline."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ScrutinyType(StrEnum):
    """Type of scrutiny a paper is under."""

    EXPRESSION_OF_CONCERN = "expression of concern"
    CORRECTION = "correction"
    CRITIQUE = "critique"


class FieldValidationResult(BaseModel):
    """Stage 1 output: field validation."""

    is_valid: bool
    field: str
    field_category: str = ""  # natural_science or social_science
    rejection_reason: str | None = None


class IsValidField(BaseModel):
    """Structured extraction: is this a real academic field?"""

    is_real_field: bool = Field(description="Whether this is a real academic field")
    normalized_name: str | None = Field(
        max_length=150,
        description="The canonical name of the field, or null if not a real field",
    )
    broad_category: str | None = Field(
        max_length=50,
        description="'natural_science' or 'social_science' or 'other', or null if not a real field",
    )


class ScrutinizedPaper(BaseModel):
    """A paper currently under scrutiny (from Retraction Watch DB)."""

    doi: str | None = None
    title: str
    authors: str | None = None
    year: str | None = None
    journal: str | None = None
    scrutiny_type: str  # expression_of_concern, correction
    reason: str | None = None
    abstract: str | None = None  # from Retraction Watch DB or OpenAlex
    citation_count: int | None = None
    similarity: float | None = None  # from semantic search


class EnrichedPaper(BaseModel):
    """A scrutinized paper enriched with context from LLM analysis."""

    doi: str | None = None
    title: str
    authors: str | None = None
    year: str | None = None
    journal: str | None = None
    scrutiny_type: str
    reason: str | None = None
    citation_count: int | None = None
    what_it_claimed: str = Field(
        max_length=500,
        description="What the original paper claimed (1-2 sentences)",
    )
    why_flagged: str = Field(
        max_length=500,
        description="Why this paper is under scrutiny (1-2 sentences)",
    )
    resolution_status: str = Field(
        max_length=50,
        description="Current status: unresolved, corrected, retracted, or under review",
    )


class PaperEnrichment(BaseModel):
    """LLM extraction schema for context enrichment (Stage 3)."""

    what_it_claimed: str = Field(
        max_length=500,
        description="What the original paper claimed — 1-2 sentences",
    )
    why_flagged: str = Field(
        max_length=500,
        description="Why this paper is under scrutiny — 1-2 sentences",
    )
    resolution_status: str = Field(
        max_length=50,
        description="Current status: 'unresolved', 'corrected', 'retracted', or 'under review'",
    )


class CuratedCase(BaseModel):
    """A curated case for the student with pedagogical framing."""

    title: str
    authors: str | None = None
    journal: str | None = None
    year: str | None = None
    doi: str | None = None
    scrutiny_type: str
    reason: str | None = None
    citation_count: int | None = None
    what_it_claimed: str
    why_flagged: str
    resolution_status: str
    pedagogical_lesson: str = Field(
        max_length=500,
        description="What this case teaches about scientific skepticism (1-2 sentences)",
    )
    discussion_questions: list[str] = Field(
        description="2-3 critical thinking questions for the student",
        min_length=1,
    )


class QuestionedResult(BaseModel):
    """Final output of the Questioned pipeline (Stage 4)."""

    cases: list[CuratedCase] = Field(min_length=1)
    field: str
    meta_lesson: str = Field(
        max_length=800,
        description="Overarching lesson about scientific self-correction (2-3 sentences)",
    )


class CurationOutput(BaseModel):
    """LLM extraction schema for curation (Stage 4)."""

    selected_indices: list[int] = Field(
        description="0-based indices of the most pedagogically interesting papers",
    )
    pedagogical_lessons: list[str] = Field(
        description="One lesson per selected paper",
    )
    discussion_questions: list[list[str]] = Field(
        description="2-3 questions per selected paper",
    )
    meta_lesson: str = Field(
        max_length=800,
        description="Overarching lesson about scientific self-correction",
    )


class SurfaceScrutinyRequest(BaseModel):
    """API request body for the Questioned workshop."""

    field: str = Field(description="Academic field to search (e.g., 'psychology')")
    project_id: str | None = None
