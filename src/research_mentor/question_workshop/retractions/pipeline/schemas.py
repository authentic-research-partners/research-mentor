"""Pydantic schemas for the Retractions pipeline.

Each stage produces structured output validated by these models.
The final output is a ranked list of research questions grounded
in gaps re-opened by retracted papers.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# DOI detection
# ---------------------------------------------------------------------------

DOI_REGEX = re.compile(r"10\.\d{4,}/\S+")


def looks_like_doi(text: str) -> bool:
    """Return True if *text* matches a DOI pattern."""
    return bool(DOI_REGEX.search(text.strip()))


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------

FAILURE_TYPES = frozenset({
    "fabrication",
    "falsification",
    "honest_error",
    "statistical_manipulation",
    "image_manipulation",
    "plagiarism",
    "irreproducible",
    "methodology_flawed",
    "peer_review_failure",
})


# ---------------------------------------------------------------------------
# Stage 1: Case Retriever
# ---------------------------------------------------------------------------


class RetractionCase(BaseModel):
    """A single retracted paper with metadata."""

    doi: str
    title: str
    journal: str
    year: int | None = None
    retraction_reason_raw: str = Field(
        max_length=500,
        description="Raw reason string from Retraction Watch DB",
    )
    original_claim: str = Field(
        max_length=500,
        description="What the paper claimed (1-2 sentences)",
    )
    field: str

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "doi": "10.1234/example.2020.001",
            "title": "Effects of X on Y: a randomized trial",
            "journal": "Journal of Example Science",
            "year": 2020,
            "retraction_reason_raw": "Results Not Reproducible;Error in Data",
            "original_claim": (
                "The paper claimed that treatment X significantly reduced "
                "symptom Y in a sample of 200 participants."
            ),
            "field": "psychology",
        },
    })


class RetractionCases(BaseModel):
    """Stage 1 output: selected retraction cases."""

    cases: list[RetractionCase] = Field(min_length=1)
    mode: str = Field(description="'browse' or 'direct'")


class CaseSelection(BaseModel):
    """LLM extraction schema for case selection (browse mode)."""

    selected_indices: list[int] = Field(
        description="0-based indices of the 3-5 most diverse and interesting cases",
    )
    original_claims: list[str] = Field(
        description="One plain-language claim per selected case (1-2 sentences each)",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "selected_indices": [0, 2, 5],
            "original_claims": [
                "The paper claimed that compound X inhibits enzyme Y with IC50 of 5nM.",
                "The study reported a strong correlation between A and B in 500 samples.",
                "The authors proposed a new mechanism for C based on simulation data.",
            ],
        },
    })


class SingleCaseClaim(BaseModel):
    """LLM extraction schema for a single case's original claim (direct mode)."""

    original_claim: str = Field(
        description="What the paper claimed, in plain language (1-2 sentences)",
    )


# ---------------------------------------------------------------------------
# Stage 2: Failure Analyzer
# ---------------------------------------------------------------------------


class FailureAnalysis(BaseModel):
    """Failure analysis for a single retracted paper."""

    failure_type: str = Field(
        max_length=50,
        description=(
            "One of: fabrication, falsification, honest_error, "
            "statistical_manipulation, image_manipulation, plagiarism, "
            "irreproducible, methodology_flawed, peer_review_failure"
        ),
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the failure type classification (0.0-1.0)",
    )
    methodology_issues: list[str] = Field(
        description="Specific methodological issues identified (2-4 items)",
    )
    what_went_wrong: str = Field(
        max_length=800,
        description="Population-appropriate explanation of what went wrong (2-3 sentences)",
    )
    process_lesson: str = Field(
        max_length=500,
        description="What this reveals about the research process (1-2 sentences)",
    )
    systemic_factors: list[str] = Field(
        description=(
            "Systemic factors that contributed: publish-or-perish, "
            "missing pre-registration, inadequate peer review, etc. (1-3 items)"
        ),
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "failure_type": "statistical_manipulation",
            "confidence": 0.85,
            "methodology_issues": [
                "P-hacking: multiple comparisons without correction",
                "Selective reporting of favorable outcomes only",
            ],
            "what_went_wrong": (
                "The researchers ran many statistical tests and only reported "
                "the ones that showed significant results. This inflated the "
                "apparent effect size and made findings seem more reliable."
            ),
            "process_lesson": (
                "Pre-registration of analysis plans prevents selective "
                "reporting by locking in the statistical approach before "
                "seeing the data."
            ),
            "systemic_factors": [
                "Pressure to publish statistically significant results",
                "No pre-registration requirement from the journal",
            ],
        },
    })


class AnalyzedCase(BaseModel):
    """A retraction case with failure analysis attached."""

    case: RetractionCase
    analysis: FailureAnalysis


class AnalyzedCases(BaseModel):
    """Stage 2 output: cases with failure analysis."""

    cases: list[AnalyzedCase] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Stage 3: Impact Assessor
# ---------------------------------------------------------------------------


class ImpactAnalysis(BaseModel):
    """Citation impact and reopened gap analysis for a retracted paper."""

    citation_count: int = Field(
        ge=0,
        description="Number of citations found",
    )
    field_impact: str = Field(
        max_length=500,
        description="1-2 sentence summary of impact on the field",
    )
    affected_conclusions: list[str] = Field(
        description="Field conclusions that were affected by this retraction (1-3 items)",
    )
    reopened_gap: str = Field(
        max_length=500,
        description=(
            "What question is now unanswered again because of this "
            "retraction (1-2 sentences)"
        ),
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "citation_count": 42,
            "field_impact": (
                "This paper was widely cited as evidence for mechanism X. "
                "Its retraction leaves the field without strong evidence "
                "for this mechanism."
            ),
            "affected_conclusions": [
                "The widely accepted model of X-mediated Y regulation",
                "Treatment protocols based on the reported dose-response curve",
            ],
            "reopened_gap": (
                "Whether compound X actually inhibits enzyme Y remains "
                "unresolved — the original evidence was fabricated."
            ),
        },
    })


class ImpactedCase(BaseModel):
    """A retraction case with failure analysis and impact assessment."""

    case: RetractionCase
    analysis: FailureAnalysis
    impact: ImpactAnalysis


class ImpactedCases(BaseModel):
    """Stage 3 output: cases with impact assessment."""

    cases: list[ImpactedCase] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Stage 4: Question Generator
# ---------------------------------------------------------------------------


class RetractionQuestion(BaseModel):
    """A research question generated from a retraction's reopened gap."""

    question: str = Field(
        max_length=500,
        description="The research question addressing the gap left by the retraction",
    )
    source_case_doi: str = Field(
        max_length=200,
        description="DOI of the retraction case this question came from",
    )
    reopened_gap: str = Field(
        max_length=500,
        description="What the retraction left unanswered (1-2 sentences)",
    )
    avoids_original_mistakes: str = Field(
        max_length=500,
        description="How this research design prevents the original failure (1-2 sentences)",
    )
    methodology_improvements: list[str] = Field(
        description="Specific methodology improvements over the original study (2-4 items)",
    )
    why_interesting: str = Field(
        max_length=500,
        description="Why this question is scientifically interesting (1-2 sentences)",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "question": (
                "Does pre-registered replication of the X-Y inhibition assay, "
                "using blinded analysis and independent synthesis of compound X, "
                "reproduce the original dose-response relationship?"
            ),
            "source_case_doi": "10.1234/example.2020.001",
            "reopened_gap": (
                "Whether compound X actually inhibits enzyme Y remains unresolved."
            ),
            "avoids_original_mistakes": (
                "Pre-registration locks the analysis plan before data collection. "
                "Independent synthesis eliminates the possibility of contaminated reagents."
            ),
            "methodology_improvements": [
                "Pre-registered analysis plan",
                "Blinded data analysis",
                "Independent synthesis of compound X",
                "Larger sample size with power analysis",
            ],
            "why_interesting": (
                "If confirmed, the X-Y mechanism would reshape understanding "
                "of the regulatory pathway. If refuted, alternative mechanisms "
                "must be explored."
            ),
        },
    })


class CaseQuestions(BaseModel):
    """LLM extraction schema: questions generated from one retraction case."""

    questions: list[RetractionQuestion] = Field(
        min_length=1, max_length=2,
        description="1-2 research questions from this case's reopened gap",
    )


class RetractionQuestions(BaseModel):
    """Stage 4 output: all generated questions across cases."""

    questions: list[RetractionQuestion] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Stage 5: Feasibility Scorer
# ---------------------------------------------------------------------------


class RetractionFeasibility(BaseModel):
    """Feasibility assessment for a retraction-derived research question."""

    data_availability: float = Field(
        ge=0, le=10,
        description="Can this be investigated with available data/tools? (0-10)",
    )
    student_accessibility: float = Field(
        ge=0, le=10,
        description="Appropriate for this student level? (0-10)",
    )
    scientific_value: float = Field(
        ge=0, le=10,
        description="How important is re-answering this question? (0-10)",
    )
    methodology_soundness: float = Field(
        ge=0, le=10,
        description="Do the proposed improvements address the original failure? (0-10)",
    )
    integrity_value: float = Field(
        ge=0, le=10,
        description="How much does this teach about research integrity? (0-10)",
    )
    overall_score: float = Field(
        ge=0, le=10,
        description="Overall recommendation score (0-10)",
    )
    reasoning: str = Field(
        max_length=800,
        description="Brief reasoning for the scores (2-3 sentences)",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "data_availability": 7.0,
            "student_accessibility": 8.0,
            "scientific_value": 9.0,
            "methodology_soundness": 8.5,
            "integrity_value": 7.5,
            "overall_score": 8.0,
            "reasoning": (
                "This question addresses a significant gap with clear methodology "
                "improvements. The required reagents and equipment are widely "
                "available. A motivated undergraduate could execute this study."
            ),
        },
    })


class ScoredRetractionQuestion(BaseModel):
    """A research question with its source case, failure type, and feasibility score."""

    question: str
    source_case: RetractionCase
    failure_type: str
    reopened_gap: str
    avoids_original_mistakes: str
    methodology_improvements: list[str]
    why_interesting: str
    feasibility: RetractionFeasibility


class ScoredRetractionQuestions(BaseModel):
    """Stage 5 output: ranked questions with feasibility scores."""

    questions: list[ScoredRetractionQuestion] = Field(min_length=1)


# ---------------------------------------------------------------------------
# API request
# ---------------------------------------------------------------------------


class RetractionsGenerateRequest(BaseModel):
    """API request body for the Retractions pipeline."""

    query: str = Field(
        description="Field name (e.g., 'psychology') OR DOI/title of a specific paper",
    )
    project_id: str | None = None
