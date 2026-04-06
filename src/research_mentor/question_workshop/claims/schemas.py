"""Pydantic schemas for the Claim Investigator pipeline.

Consolidated from progress_mentor's claim.py, evidence.py, and project.py.
All schemas use Pydantic v2 conventions (model_config instead of class Config).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Stage 1: Claim Extractor
# ---------------------------------------------------------------------------


class SourceType(StrEnum):
    """Type of source where the claim originated."""

    MAINSTREAM_MEDIA = "mainstream_media"
    WELLNESS_BLOG = "wellness_blog"
    SOCIAL_MEDIA = "social_media"
    INFLUENCER = "influencer"
    PRESS_RELEASE = "press_release"
    UNKNOWN = "unknown"


class ClaimData(BaseModel):
    """Structured representation of a scientific claim extracted from media."""

    original_text: str = Field(
        max_length=1000,
        description="Exact quote or paraphrase of the claim from the source",
    )
    normalized_form: str = Field(
        max_length=300,
        description=(
            "Standardized form: 'X affects/causes/is linked to Y'. "
            "Example: 'Lemon water consumption -> liver detoxification'"
        ),
    )
    independent_var: str = Field(
        max_length=150,
        description="The independent variable (X) — the cause or influencing factor",
    )
    dependent_var: str = Field(
        max_length=150,
        description="The dependent variable (Y) — the effect or outcome",
    )
    field: str = Field(
        max_length=50,
        description="Scientific field: 'physics', 'chemistry', or 'biology'",
    )
    claim_type: str = Field(
        max_length=50,
        description=(
            "Type of claim: 'causal' (X causes Y), 'correlational' (X linked to Y), "
            "'property' (X has property Y), or 'process' (X works by Y)"
        ),
    )
    source_type: SourceType = Field(
        default=SourceType.UNKNOWN,
        description="Type of source where the claim originated",
    )
    source_name: str = Field(
        default="Unknown",
        max_length=200,
        description="Name of the source (e.g., 'New York Times', 'wellness blog')",
    )
    source_url: str | None = Field(
        default=None,
        max_length=500,
        description="URL of the source article if available",
    )

    @field_validator("field")
    @classmethod
    def validate_field(cls, v: str) -> str:
        allowed = ["physics", "chemistry", "biology"]
        v_lower = v.lower().strip()
        if v_lower not in allowed:
            raise ValueError(f"Field must be one of {allowed}, got '{v}'")
        return v_lower

    @field_validator("claim_type")
    @classmethod
    def validate_claim_type(cls, v: str) -> str:
        allowed = ["causal", "correlational", "property", "process"]
        v_lower = v.lower().strip()
        if v_lower not in allowed:
            raise ValueError(f"Claim type must be one of {allowed}, got '{v}'")
        return v_lower

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "original_text": (
                "Drinking warm lemon water every morning detoxifies "
                "the liver and boosts metabolism"
            ),
            "normalized_form": (
                "Lemon water consumption -> liver detoxification + metabolism increase"
            ),
            "independent_var": "lemon water consumption",
            "dependent_var": "liver detoxification and metabolism",
            "field": "biology",
            "claim_type": "causal",
            "source_type": "wellness_blog",
            "source_name": "HealthyLiving Blog",
            "source_url": "https://example.com/lemon-water-benefits",
        },
    })


class IsScienceClaim(BaseModel):
    """Quick pre-check: is the input a testable science claim?"""

    is_valid: bool = Field(
        description="True if the text contains a testable natural or social science claim",
    )
    rejection_reason: str | None = Field(
        default=None,
        max_length=500,
        description="Why the claim was rejected (medical, political, not science, etc.)",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "is_valid": True,
            "rejection_reason": None,
        },
    })


# Backward-compatible alias
IsNaturalScienceClaim = IsScienceClaim


# ---------------------------------------------------------------------------
# Stage 2: Evidence Searcher
# ---------------------------------------------------------------------------


class EvidenceLevel(StrEnum):
    """Level of scientific evidence supporting a claim."""

    WELL_SUPPORTED = "WELL_SUPPORTED"
    WEAKLY_SUPPORTED = "WEAKLY_SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class PaperSummary(BaseModel):
    """Summary of a relevant academic paper."""

    title: str = Field(max_length=500, description="Paper title")
    authors: str = Field(max_length=300, description="Author names (first author et al.)")
    year: int | None = Field(default=None, description="Publication year")
    journal: str | None = Field(default=None, max_length=200, description="Journal name")

    @field_validator("year", mode="before")
    @classmethod
    def coerce_year(cls, v: object) -> int | None:
        if v is None or v == "":
            return None
        if isinstance(v, int):
            return v
        try:
            return int(str(v))
        except (ValueError, TypeError):
            return None
    relevance_summary: str = Field(
        max_length=500,
        description="Brief summary of how this paper relates to the claim",
    )
    supports_claim: str | None = Field(
        default=None,
        max_length=50,
        description="'yes' if supports, 'no' if contradicts, 'partial' or 'unclear' otherwise",
    )
    doi: str | None = Field(default=None, max_length=200, description="DOI if available")
    url: str | None = Field(default=None, max_length=500, description="URL to paper")


class EvidenceResult(BaseModel):
    """Result of evidence search for a claim (Stage 2 output)."""

    level: EvidenceLevel = Field(
        description="Overall evidence level for the claim",
    )
    summary: str = Field(
        max_length=1500,
        description="Human-readable summary of what science says about this claim",
    )
    papers_found: int = Field(
        default=0,
        description="Total number of potentially relevant papers found",
    )
    relevant_papers: list[PaperSummary] = Field(
        default_factory=list,
        description="Most relevant papers (up to 5)",
    )
    search_queries_used: list[str] = Field(
        default_factory=list,
        description="Search queries used to find evidence",
    )
    what_science_says: str = Field(
        max_length=2000,
        description=(
            "Detailed explanation of what peer-reviewed science actually says "
            "(or doesn't say) about this claim"
        ),
    )
    related_science: str | None = Field(
        default=None,
        max_length=2000,
        description=(
            "Related legitimate science that DOES exist "
            "(even if claim is unsupported)"
        ),
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "level": "NOT_SUPPORTED",
            "summary": (
                "No peer-reviewed studies support lemon water "
                "'detoxification' claims"
            ),
            "papers_found": 0,
            "relevant_papers": [],
            "search_queries_used": [
                "lemon water detoxification liver",
                "citrus metabolism clinical trial",
            ],
            "what_science_says": (
                "We searched PubMed and Google Scholar and found 0 clinical "
                "trials on 'lemon water detoxification'. The concept of 'detox' "
                "through food is not recognized in medical science."
            ),
            "related_science": (
                "Vitamin C has known biological functions, but lemon water "
                "contains relatively little. Hydration is important but plain "
                "water works identically."
            ),
        },
    })


# ---------------------------------------------------------------------------
# Stage 3: Gap Analyzer
# ---------------------------------------------------------------------------


class GapAnalysis(BaseModel):
    """Analysis of the gap between journalist 'research' and real research."""

    journalist_did: list[str] = Field(
        min_length=2,
        description=(
            "What the journalist likely did "
            "(e.g., 'Read other blogs', 'Quoted non-experts')"
        ),
    )
    time_spent: str = Field(
        max_length=300,
        description="Estimated time journalist spent (e.g., '2-3 days of Googling')",
    )
    real_research_requires: list[str] = Field(
        min_length=3,
        description=(
            "What proper scientific research would require "
            "(e.g., 'Controlled trial with 500+ participants')"
        ),
    )
    why_hard_to_test: list[str] = Field(
        min_length=2,
        description=(
            "Barriers to directly testing this claim "
            "(e.g., 'Requires $10,000+ equipment')"
        ),
    )
    key_insight: str = Field(
        max_length=1000,
        description=(
            "The educational punchline — why this matters "
            "(e.g., 'This is why confident headlines don't equal confident science')"
        ),
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "journalist_did": [
                "Read other wellness blogs making similar claims",
                "Found quotes from 'health coaches' (not researchers)",
                "Cited no peer-reviewed studies",
                "Published confident headline",
            ],
            "time_spent": "~2-3 days of internet research",
            "real_research_requires": [
                "Measurable definition of 'detoxification' (none exists)",
                "Blood tests for liver function (medical supervision)",
                "Indirect calorimetry for metabolism ($10,000+ equipment)",
                "Controlled trial with hundreds of participants",
                "Months to years of data collection",
                "Ethics board approval for human subjects",
            ],
            "why_hard_to_test": [
                "'Detoxification' has no measurable scientific definition",
                "Liver function requires medical-grade testing",
                "Long-term effects require years of controlled study",
            ],
            "key_insight": (
                "This is why the journalist didn't do it. This is what "
                "separates real research from Googling for a few days."
            ),
        },
    })


# ---------------------------------------------------------------------------
# Stage 4: Project Framer
# ---------------------------------------------------------------------------


class ResearchDirection(BaseModel):
    """Open-ended research direction for the student (Stage 4 output)."""

    title: str = Field(
        min_length=3,
        max_length=150,
        description="Catchy 2-4 word title hinting at the investigation",
    )
    description: str = Field(
        min_length=50,
        max_length=2000,
        description=(
            "2-4 sentences: What media claims, the scientific gap, "
            "and what's actually investigable"
        ),
    )
    investigation: str = Field(
        min_length=30,
        max_length=1000,
        description=(
            "Open-ended directive starting with 'Investigate...', "
            "'Explore...', or 'Study...'. "
            "Frames WHAT to investigate, not HOW."
        ),
    )
    project_duration: str = Field(
        max_length=150,
        description="Estimated project duration (e.g., '2-4 months')",
    )
    project_type: str = Field(
        max_length=50,
        description="Type: 'experimental', 'data_analysis', or 'survey'",
    )
    underlying_phenomenon: str = Field(
        max_length=500,
        description="The real scientific phenomenon that can be investigated",
    )

    @field_validator("investigation")
    @classmethod
    def validate_investigation_starts_with_action(cls, v: str) -> str:
        action_verbs = ["investigate", "study", "explore", "examine"]
        v_lower = v.lower().strip()
        if not any(v_lower.startswith(verb) for verb in action_verbs):
            raise ValueError(
                f"Investigation must start with one of: {', '.join(action_verbs)}"
            )
        return v

    @field_validator("project_type")
    @classmethod
    def validate_project_type(cls, v: str) -> str:
        allowed = ["experimental", "data_analysis", "survey"]
        v_lower = v.lower().strip()
        if v_lower not in allowed:
            raise ValueError(f"Project type must be one of {allowed}, got '{v}'")
        return v_lower

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "title": "Lemon Water Chemistry",
            "description": (
                "Wellness influencers claim lemon water 'alkalizes the body' "
                "despite being acidic, and provides health benefits no other "
                "water can match. What actually happens when acidic lemon juice "
                "meets the chemistry of digestion? Does lemon water have any "
                "measurable properties that distinguish it from plain water?"
            ),
            "investigation": (
                "Investigate the chemistry of lemon water — its pH, buffering "
                "capacity, and behavior when mixed with acids of varying "
                "strengths. Explore whether any measurable chemical property "
                "could plausibly support the health claims made in popular media."
            ),
            "project_duration": "2-4 months",
            "project_type": "experimental",
            "underlying_phenomenon": (
                "Acid-base chemistry and buffering in biological systems"
            ),
        },
    })


# ---------------------------------------------------------------------------
# Stage 5: Question Generator
# ---------------------------------------------------------------------------


class CriticalQuestions(BaseModel):
    """Critical thinking questions for the student's journey (Stage 5 output)."""

    about_journalist: list[str] = Field(
        min_length=2,
        description="Questions about the journalist's 'research' process",
    )
    about_evidence: list[str] = Field(
        min_length=2,
        description="Questions about scientific evidence and proof",
    )
    about_project: list[str] = Field(
        min_length=2,
        description="Questions about the student's own project and its limitations",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "about_journalist": [
                "Why did the article cite no scientific studies?",
                "Who profits from people believing this claim?",
                "What questions should the journalist have asked?",
            ],
            "about_evidence": [
                "Why isn't 'I feel better' the same as 'measurably healthier'?",
                "What would a real clinical trial need to include?",
                "Why do scientists reject the concept of 'detox diets'?",
            ],
            "about_project": [
                "What CAN your investigation tell you?",
                "What conclusions are beyond what your project can prove?",
                "How is 3 months of your work different from 3 days of Googling?",
            ],
        },
    })


# ---------------------------------------------------------------------------
# API request
# ---------------------------------------------------------------------------


class InvestigateClaimRequest(BaseModel):
    """API request body for the /investigate endpoint."""

    claim_text: str | None = Field(
        default=None,
        description="Direct claim text to investigate",
    )
    claim_url: str | None = Field(
        default=None,
        description="URL to article containing the claim",
    )
    article_text: str | None = Field(
        default=None,
        description="Full article text to extract claims from",
    )
    field_filter: str | None = Field(
        default=None,
        description="Optional field filter (physics, chemistry, biology)",
    )
    project_id: str | None = Field(
        default=None,
        description="Optional project to link the investigation to",
    )

    @field_validator("field_filter")
    @classmethod
    def validate_field_filter(cls, v: str | None) -> str | None:
        if v is None:
            return None
        allowed = ["physics", "chemistry", "biology"]
        v_lower = v.lower().strip()
        if v_lower not in allowed:
            raise ValueError(f"field_filter must be one of {allowed}, got '{v}'")
        return v_lower
