"""Gaps Extraction Schemas — Pydantic models for structured LLM output.

All schemas used by Gaps nodes for deterministic extraction via structured_call().
"""

from pydantic import BaseModel, Field, field_validator

_TRUTHY = frozenset({"true", "yes", "1", "t", "y"})
_FALSY = frozenset({"false", "no", "0", "f", "n"})


def _coerce_bool(v: object) -> object:
    """Coerce string bools from LLM output (e.g. "True" -> True)."""
    if isinstance(v, str):
        low = v.strip().lower()
        if low in _TRUTHY:
            return True
        if low in _FALSY:
            return False
    return v


# ==================== Entry Validator ====================


class InputValidationResult(BaseModel):
    """Result of input validation."""

    is_security_threat: bool = Field(
        description="True if message contains security threat (attack attempt)",
    )
    threat_type: str = Field(
        max_length=50,
        description=(
            "Type of threat if detected: 'instruction_override', 'system_probe', "
            "'jailbreak', 'data_access', 'social_engineering', 'none'"
        ),
    )
    threat_explanation: str = Field(
        max_length=500,
        description="Explanation of the threat, if any",
    )
    is_legitimate_question: bool = Field(
        description="True if message is a legitimate research-related question",
    )

    @field_validator("is_security_threat", "is_legitimate_question", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Domain Classification ====================


class DomainClassification(BaseModel):
    """Classify whether a topic is natural science, social science, or off-topic."""

    domain_type: str = Field(
        max_length=50,
        description=(
            "One of: 'natural_science' (physics, chemistry, biology, "
            "earth science, astronomy, materials science), "
            "'social_science' (psychology, sociology, education, economics, "
            "political science, linguistics, anthropology), "
            "'refused' (medicine, clinical research, pharmacology, nutrition, "
            "public health policy), "
            "'off_topic' (not a research topic at all)"
        ),
    )
    confidence: str = Field(
        max_length=50,
        description="'high' or 'low' — how confident is the classification",
    )


# ==================== User Intent Detection ====================


class UserIntent(BaseModel):
    """Detect user's intent/emotional state before generating a response.

    This runs BEFORE the chat LLM so the response can adapt to the user's state.
    """

    intent: str = Field(
        max_length=50,
        description=(
            "One of: 'engaged' (actively exploring, asking substantive questions), "
            "'frustrated' (stuck, nothing clicking, expressing difficulty — "
            "but NOT asking to change topic or go back), "
            "'broadening' (explicitly asking to go back, try different area, "
            "broaden scope, start over — the user wants to LEAVE the current focus), "
            "'refining' (proposing a new/modified question), "
            "'asking_how' (asking for specific guidance on how to improve), "
            "'general' (other conversation)"
        ),
    )
    proposed_question: str | None = Field(
        default=None,
        max_length=500,
        description="If user proposed a new or refined research question, extract it here",
    )


# ==================== Phase 1: Domain Exploration ====================


class LandscapePatterns(BaseModel):
    """Patterns extracted from a corpus of papers in a research domain."""

    consensus: list[str] = Field(
        default_factory=list,
        description="Key findings the field agrees on (2-5 items)",
    )
    debates: list[str] = Field(
        default_factory=list,
        description="Active controversies or conflicting findings (1-4 items)",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Explicitly stated unknowns or 'future research' items (2-5 items)",
    )
    frontiers: list[str] = Field(
        default_factory=list,
        description="Methodological limits — what can't be measured yet (0-3 items)",
    )
    domain_summary: str = Field(
        max_length=1500,
        description="One-paragraph overview of the field's current state",
    )


class ModeRecommendation(BaseModel):
    """Recommended strategic mode based on field and researcher characteristics."""

    recommended_mode: str = Field(
        max_length=50,
        description=(
            "One of: 'explicit_mining', 'contradiction_detection', "
            "'theoretical_probing', 'mechanistic_dissection', 'integration_synthesis'"
        ),
    )
    reasoning: str = Field(
        max_length=800,
        description="Why this mode fits the field maturity and researcher interests",
    )
    field_maturity: str = Field(
        max_length=50,
        description="Assessed maturity: 'emerging', 'established', or 'mature'",
    )


class FocusAreaDetection(BaseModel):
    """Detect whether user has selected a specific focus area from the landscape."""

    has_selected_focus: bool = Field(
        description="True if user expressed a specific area they want to explore deeply",
    )
    focus_area: str | None = Field(
        default=None,
        max_length=300,
        description="The specific focus area, if selected",
    )
    researcher_style: str | None = Field(
        default=None,
        max_length=50,
        description=(
            "Detected researcher style if expressed: "
            "'experimental', 'theoretical', 'computational', or null"
        ),
    )

    @field_validator("has_selected_focus", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Phase 2: Deep Diving ====================


class StrategyOutput(BaseModel):
    """Findings from applying one analytical strategy to a set of papers."""

    strategy_name: str = Field(
        max_length=100,
        description="Which strategy was applied",
    )
    findings: list[str] = Field(
        default_factory=list,
        description="Key findings from the strategy (2-5 items)",
    )
    potential_questions: list[str] = Field(
        default_factory=list,
        description="Candidate research questions suggested by the findings (1-3 items)",
    )
    papers_referenced: list[str] = Field(
        default_factory=list,
        description="Titles of papers that support these findings",
    )


class CitedResponse(BaseModel):
    """Conversational response with paper citations.

    The LLM returns text + which papers it referenced (by number from
    the prompt list). Code expands paper numbers into real citations.
    """

    text: str = Field(
        max_length=1000,
        description=(
            "Your 2-3 sentence response to the student. "
            "Do NOT include paper titles or author names in the text."
        ),
    )
    papers_cited: list[int] = Field(
        default_factory=list,
        description=(
            "Which paper numbers from the list you referenced in your response. "
            "Use the numbers from the papers list (1, 2, 3, etc.). "
            "Empty list if no papers were relevant to this response."
        ),
    )


class QuestionExtraction(BaseModel):
    """Extract candidate research questions from conversation."""

    questions_articulated: list[str] = Field(
        default_factory=list,
        description="Research questions the user has formulated in the conversation",
    )
    questions_count: int = Field(
        default=0,
        description="Number of distinct questions identified",
    )
    needs_scope_broadening: bool = Field(
        default=False,
        description=(
            "True if user explicitly asks to go back, broaden, or try a different area"
        ),
    )
    ready_for_validation: bool = Field(
        default=False,
        description="True if 2+ meaningful candidate questions have been formulated",
    )

    @field_validator("needs_scope_broadening", "ready_for_validation", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Phase 3: Validation ====================


class FUNDAMENTALScore(BaseModel):
    """FUNDAMENTAL criteria evaluation for a research question.

    11 criteria scored 0.0-1.0:
    F-alsifiable, U-naddressed, N-ecessary, D-eep, A-pproachable,
    M-eaningful, E-xciting, N-arrowed, T-heoretically-grounded,
    A-rticulated, L-iterature-based.
    """

    question: str = Field(max_length=500, description="The research question being evaluated")
    falsifiable: float = Field(ge=0, le=1, description="Can it be proven wrong?")
    unaddressed: float = Field(ge=0, le=1, description="Genuinely unstudied in literature?")
    necessary: float = Field(ge=0, le=1, description="Must be answered for field progress?")
    deep: float = Field(
        ge=0, le=1, description="Reveals principles/mechanisms, not just phenomena?"
    )
    approachable: float = Field(
        ge=0, le=1, description="Addressable with current/near-future methods?"
    )
    meaningful: float = Field(ge=0, le=1, description="Answer changes understanding significantly?")
    exciting: float = Field(ge=0, le=1, description="Inspires the researcher?")
    narrowed: float = Field(ge=0, le=1, description="Specific enough to guide experiments/theory?")
    theoretically_grounded: float = Field(
        ge=0, le=1, description="Connects to existing frameworks?"
    )
    articulated: float = Field(ge=0, le=1, description="Stated clearly and unambiguously?")
    literature_based: float = Field(
        ge=0, le=1, description="Emerges from synthesis, not speculation?"
    )
    overall: float = Field(ge=0, le=1, description="Weighted overall score")
    reasoning: str = Field(max_length=800, description="Brief justification for the scores")
    refinement_suggestions: list[str] = Field(
        default_factory=list,
        description="Specific suggestions to improve low-scoring criteria (if any)",
    )


class ValidationProgress(BaseModel):
    """Track refinement progress during Phase 3."""

    questions_refined: bool = Field(
        default=False,
        description="True if user has refined any question based on feedback",
    )
    all_scores_low: bool = Field(
        default=False,
        description="True if ALL questions score below threshold — backward jump to Phase 2",
    )
    has_validated_question: bool = Field(
        default=False,
        description="True if at least one question scores >= 0.8 overall",
    )
    ready_for_completion: bool = Field(
        default=False,
        description="True if user is satisfied with validated question(s)",
    )

    @field_validator(
        "questions_refined", "all_scores_low", "has_validated_question",
        "ready_for_completion", mode="before",
    )
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Completion ====================


class CompletionIntent(BaseModel):
    """Classify user intent during completion phase."""

    wants_export: bool = Field(
        default=False,
        description="User wants to export/save the final question formulation",
    )
    has_followup: bool = Field(
        default=False,
        description="User has a follow-up question about their research question",
    )
    wants_to_end: bool = Field(
        default=False,
        description="User wants to end the session",
    )

    @field_validator("wants_export", "has_followup", "wants_to_end", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)
