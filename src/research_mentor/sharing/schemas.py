"""Sharing Extraction Schemas — Pydantic models for structured LLM output.

All schemas used by Sharing nodes for deterministic extraction via structured_call().
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
        description="True if message is a legitimate sharing/communication question",
    )

    @field_validator("is_security_threat", "is_legitimate_question", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Intent Detection ====================


class SharingIntentDetection(BaseModel):
    """Detect user's intent and whether they want to switch capabilities."""

    intent: str = Field(
        max_length=50,
        description=(
            "One of: 'engaged' (actively working on current phase), "
            "'frustrated' (stuck, expressing difficulty), "
            "'asking_how' (asking for specific guidance), "
            "'capability_switch' (wants to switch to a different capability — "
            "e.g. 'can we look at venues?', 'help me with my poster'), "
            "'ready_to_complete' (wants to wrap up — 'I think I have what I need'), "
            "'general' (other conversation)"
        ),
    )
    requested_capability: str | None = Field(
        default=None,
        max_length=50,
        description=(
            "Target phase if intent='capability_switch': "
            "'writing_support', 'venue_discovery', 'communication_guidance', "
            "'completion'. Null otherwise."
        ),
    )


# ==================== Context Gathering ====================


class ContextGatheringExtraction(BaseModel):
    """Extract research context from the conversation so far."""

    has_research_summary: bool = Field(
        description="True if the student has described what their research is about",
    )
    research_summary: str | None = Field(
        default=None,
        max_length=500,
        description="Brief summary of the student's research (1-2 sentences)",
    )
    output_type: str | None = Field(
        default=None,
        max_length=50,
        description=(
            "What the student wants to create: 'paper', 'poster', 'presentation', "
            "'video', 'blog', 'infographic', 'undecided', or null if not yet mentioned"
        ),
    )
    target_audience: str | None = Field(
        default=None,
        max_length=50,
        description=(
            "Who the student wants to reach: 'peers', 'general_public', "
            "'experts', 'teachers', 'community', or null if not yet mentioned"
        ),
    )
    student_level: str | None = Field(
        default=None,
        max_length=50,
        description=(
            "Detected education level: 'middle_school', 'high_school', "
            "'university', 'adult', or null if unclear"
        ),
    )
    research_field: str | None = Field(
        default=None,
        max_length=50,
        description=(
            "Detected research field category: 'natural_science', "
            "'social_science', or null if unclear"
        ),
    )
    context_complete: bool = Field(
        description=(
            "True if we have enough context to help: at minimum "
            "research_summary AND (output_type OR a clear communication goal)"
        ),
    )

    @field_validator(
        "has_research_summary", "context_complete", mode="before",
    )
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Writing Support ====================


class WritingFocusDetection(BaseModel):
    """Classify what kind of writing help the student needs."""

    writing_focus: str = Field(
        max_length=50,
        description=(
            "One of: 'structure' (organizing their document), "
            "'feedback' (critique of what they've written), "
            "'revision' (help interpreting and responding to feedback they received), "
            "'simulated_review' (role-play as a peer reviewer)"
        ),
    )
    draft_section: str | None = Field(
        default=None,
        max_length=50,
        description=(
            "Which section they're asking about: 'abstract', 'introduction', "
            "'methods', 'results', 'discussion', 'conclusion', 'poster_layout', "
            "'general', or null"
        ),
    )


# ==================== Venue Discovery ====================


class VenueSearchParameters(BaseModel):
    """Extract venue search parameters from the conversation."""

    research_topic: str = Field(
        max_length=300,
        description="The student's research topic for venue matching",
    )
    student_age: int | None = Field(
        default=None,
        description="Student's age if known",
    )
    education_level: str | None = Field(
        default=None,
        description="'middle_school', 'high_school', 'university', or null",
    )
    field: str | None = Field(
        default=None,
        description="'natural_science', 'social_science', or null",
    )
    output_format: str | None = Field(
        default=None,
        description=(
            "Preferred format: 'paper', 'poster', 'oral', 'video', 'blog', "
            "'visual_art', 'photography', 'data_contribution', or null"
        ),
    )
    cost_preference: str | None = Field(
        default=None,
        description="'free', 'low', 'any', or null",
    )


# ==================== Communication Guidance ====================


class CommunicationSubTopic(BaseModel):
    """Route within the communication guidance phase."""

    sub_topic: str = Field(
        max_length=50,
        description=(
            "One of: 'poster' (poster design and content), "
            "'presentation' (oral presentation skills), "
            "'digital_media' (YouTube, TikTok, blog, podcast), "
            "'science_art' (scientific illustration, data viz, research photography), "
            "'authorship' (collaborative authorship, credit, CRediT taxonomy), "
            "'open_science' (data sharing, code sharing, preregistration)"
        ),
    )
    specific_question: str | None = Field(
        default=None,
        max_length=500,
        description="The student's specific question within this sub-topic",
    )


# ==================== Completion ====================


class CompletionIntent(BaseModel):
    """Classify user intent during completion phase."""

    has_followup: bool = Field(
        default=False,
        description="User has a follow-up question about communication",
    )
    wants_to_end: bool = Field(
        default=False,
        description="User wants to end the session",
    )
    new_questions_identified: list[str] = Field(
        default_factory=list,
        description=(
            "New research questions surfaced from the communication experience "
            "(e.g. 'audience asked about winter data', 'reviewer questioned sample size')"
        ),
    )

    @field_validator("has_followup", "wants_to_end", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)
