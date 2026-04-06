"""Pipeline-specific schemas for Gaps question generation.

Reuses DomainClassification, LandscapePatterns, StrategyOutput, and
FUNDAMENTALScore from the interactive gaps/schemas.py. Adds pipeline-only
schemas for stage composition.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LandscapeResult(BaseModel):
    """Stage 1 output — landscape exploration results."""

    papers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Papers returned from search_and_enrich",
    )
    consensus: list[str] = Field(
        default_factory=list,
        description="Key findings the field agrees on",
    )
    debates: list[str] = Field(
        default_factory=list,
        description="Active controversies or conflicting findings",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Explicitly stated unknowns or 'future research' items",
    )
    frontiers: list[str] = Field(
        default_factory=list,
        description="Methodological limits — what can't be measured yet",
    )
    field_maturity: str = Field(
        default="established",
        max_length=50,
        description="'emerging', 'established', or 'mature'",
    )
    domain_summary: str = Field(
        default="",
        max_length=1500,
        description="One-paragraph overview of the field's current state",
    )


class StrategyFindings(BaseModel):
    """Findings from one strategic analysis mode."""

    strategy_name: str = Field(max_length=100, description="Which strategy was applied")
    findings: list[str] = Field(
        default_factory=list,
        description="Key findings from the strategy (2-5 items)",
    )
    potential_questions: list[str] = Field(
        default_factory=list,
        description="Candidate research questions suggested by findings (1-3 items)",
    )
    papers_referenced: list[str] = Field(
        default_factory=list,
        description="Titles of papers that support these findings",
    )


class GapAnalysisResult(BaseModel):
    """Stage 2 output — gap analysis across multiple strategic modes."""

    strategies_applied: list[StrategyFindings] = Field(
        default_factory=list,
        description="Findings from each applied strategy",
    )
    all_candidate_questions: list[str] = Field(
        default_factory=list,
        description="Deduplicated candidate questions across all modes",
    )


class CandidateQuestion(BaseModel):
    """A single candidate research question with provenance."""

    question: str = Field(max_length=500, description="The research question")
    gap_addressed: str = Field(max_length=500, description="The gap this question addresses")
    mode_origin: str = Field(
        max_length=100,
        description="Strategic mode that surfaced this question",
    )
    literature_support: list[str] = Field(
        default_factory=list,
        description="Paper references supporting this question",
    )
    why_fundamental: str = Field(
        default="",
        max_length=500,
        description="1-2 sentences on why this is a fundamental question",
    )


class CandidateQuestions(BaseModel):
    """Stage 3 output — generated candidate questions."""

    questions: list[CandidateQuestion] = Field(
        default_factory=list,
        description="5-8 candidate research questions",
    )


class ScoredQuestion(BaseModel):
    """A candidate question with FUNDAMENTAL scores."""

    question: str = Field(max_length=500, description="The research question")
    gap_addressed: str = Field(max_length=500, description="The gap this question addresses")
    mode_origin: str = Field(
        max_length=100,
        description="Strategic mode that surfaced this question",
    )
    literature_support: list[str] = Field(
        default_factory=list,
        description="Paper references",
    )
    why_fundamental: str = Field(default="", max_length=500, description="Why this is fundamental")
    scores: dict[str, Any] = Field(
        default_factory=dict,
        description="FUNDAMENTAL criterion scores (11 criteria + overall + reasoning)",
    )


class ScoredQuestions(BaseModel):
    """Stage 4 output — final scored and ranked questions."""

    questions: list[ScoredQuestion] = Field(
        default_factory=list,
        description="Top 3-5 questions ranked by weighted FUNDAMENTAL score",
    )


# Threshold for FUNDAMENTAL score validation
FUNDAMENTAL_PASS_THRESHOLD = 0.8
