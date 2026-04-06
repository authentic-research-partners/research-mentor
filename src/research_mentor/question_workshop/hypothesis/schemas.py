"""Hypothesis Extraction Schemas — Pydantic models for structured LLM output.

All schemas preserved verbatim from production Hypothesis to maintain eval compatibility.
"""

from pydantic import BaseModel, Field, field_validator

_TRUTHY = frozenset({"true", "yes", "1", "t", "y"})
_FALSY = frozenset({"false", "no", "0", "f", "n"})


def _coerce_bool(v: object) -> object:
    """Coerce string bools from LLM output (e.g. "True" → True)."""
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
        description="True if message is a legitimate research question",
    )

    @field_validator("is_security_threat", "is_legitimate_question", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Search Decision ====================


class SearchDecision(BaseModel):
    """Structured output for autonomous search decisions (paper/dataset)."""

    should_search: bool = Field(
        description="Whether the system should search now",
    )

    @field_validator("should_search", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Stage 1: Discovery ====================


class VariableExtraction(BaseModel):
    """Structured output for variable extraction."""

    independent_var: str | None = Field(
        default=None,
        max_length=300,
        description=(
            "Independent variable (X) - what might CAUSE Y. "
            "Complete conceptual phrase, not just noun."
        ),
    )
    dependent_var: str | None = Field(
        default=None,
        max_length=300,
        description=(
            "Dependent variable (Y) - what they want to EXPLAIN. "
            "Complete conceptual phrase, not just noun."
        ),
    )
    student_confirmed: bool = Field(
        default=False,
        description=(
            "True ONLY if the student has explicitly named or clearly confirmed "
            "BOTH variables in their own words. Vague agreement ('yeah', 'sure', "
            "'that makes sense', 'ok') does NOT count as confirmation — the student "
            "must state or clearly reference specific variables."
        ),
    )

    @field_validator("student_confirmed", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Stage 2: Literature ====================


class Stage2Progress(BaseModel):
    """Structured output for Stage 2 progress extraction."""

    gap_identified: bool = Field(
        description="Whether student has identified a research gap",
    )
    gap_description: str | None = Field(
        default=None,
        max_length=500,
        description="Brief description of the research gap identified",
    )
    ideal_measurements_x: str | None = Field(
        default=None,
        max_length=300,
        description="How X (independent variable) should ideally be measured",
    )
    ideal_measurements_y: str | None = Field(
        default=None,
        max_length=300,
        description="How Y (dependent variable) should ideally be measured",
    )
    review_mode: str | None = Field(
        default=None,
        max_length=50,
        description="Student's choice: 'guided' or 'independent' (null if not yet chosen)",
    )

    @field_validator("gap_identified", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Stage 3A: Hypothesis ====================


class Stage3AProgress(BaseModel):
    """Structured output for Stage 3A hypothesis progress extraction."""

    scope_defined: bool = Field(
        default=False,
        description="Whether student has defined research scope (when, where, population)",
    )
    scope_when: str | None = Field(
        default=None,
        max_length=150,
        description="Time period for research (e.g., '2020-2023', 'last 5 years')",
    )
    scope_where: str | None = Field(
        default=None,
        max_length=150,
        description="Geographic location (e.g., 'United States', 'urban areas', 'online')",
    )
    scope_population: str | None = Field(
        default=None,
        max_length=200,
        description="Target population (e.g., 'teenagers 13-18', 'college students', 'adults')",
    )
    hypothesis: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "Full hypothesis statement "
            "(e.g., 'Increased social media usage leads to higher anxiety')"
        ),
    )
    hypothesis_direction: str | None = Field(
        default=None,
        max_length=50,
        description="Direction of relationship: 'increase', 'decrease', 'u-shaped', or 'no effect'",
    )
    theory: str | None = Field(
        default=None,
        max_length=500,
        description="Theoretical mechanism explaining WHY X causes Y",
    )
    ready_for_datasets: bool = Field(
        default=False,
        description=(
            "Whether student has completed all components and is ready for dataset search"
        ),
    )

    @field_validator("scope_defined", "ready_for_datasets", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Stage 3B: Datasets ====================


class Stage3BProgress(BaseModel):
    """Structured output for Stage 3B dataset evaluation progress."""

    dataset_selected: bool = Field(
        default=False,
        description="Whether student has selected a specific dataset",
    )
    selected_dataset_name: str | None = Field(
        default=None,
        max_length=300,
        description="Name or identifier of the selected dataset",
    )
    needs_scope_adjustment: bool = Field(
        default=False,
        description="Whether scope needs adjustment to match available data",
    )
    feasibility_concerns: list[str] = Field(
        default_factory=list,
        description=(
            "List of feasibility concerns identified "
            "(measurement quality, missing variables, etc.)"
        ),
    )
    ready_for_confounds: bool = Field(
        default=False,
        description=(
            "Whether dataset is selected and student is ready to identify confounds"
        ),
    )
    needs_variable_rethink: bool = Field(
        default=False,
        description=(
            "Whether no datasets exist for ANY reasonable scope variation — "
            "variables themselves need rethinking (backward jump to Stage 1)"
        ),
    )

    @field_validator(
        "dataset_selected", "needs_scope_adjustment", "ready_for_confounds",
        "needs_variable_rethink",
        mode="before",
    )
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Stage 3C: Refinement ====================


class Stage3CProgress(BaseModel):
    """Structured output for Stage 3C scope adjustment progress."""

    scope_adjusted: bool = Field(
        default=False,
        description="Whether student has adjusted scope to match data availability",
    )
    adjusted_when: str | None = Field(
        default=None,
        max_length=150,
        description="Adjusted time period (e.g., '2015-2020' instead of '2020-2023')",
    )
    adjusted_where: str | None = Field(
        default=None,
        max_length=150,
        description="Adjusted location (e.g., 'California' instead of 'United States')",
    )
    adjusted_population: str | None = Field(
        default=None,
        max_length=200,
        description="Adjusted population (e.g., 'college students' instead of 'all adults')",
    )
    hypothesis_changed: bool = Field(
        default=False,
        description="RED FLAG: Whether hypothesis direction was changed (should ALWAYS be False)",
    )
    ready_for_confounds: bool = Field(
        default=False,
        description=(
            "Whether scope adjustment is complete and ready for confound identification"
        ),
    )

    @field_validator(
        "scope_adjusted", "hypothesis_changed", "ready_for_confounds",
        mode="before",
    )
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Stage 3.5: Confounds ====================


class Confound(BaseModel):
    """Single confounding variable."""

    name: str = Field(max_length=200, description="Name of the confounding variable")
    affects_x: bool = Field(
        description="Whether this confound affects X (independent variable)",
    )
    affects_y: bool = Field(
        description="Whether this confound affects Y (dependent variable)",
    )

    @field_validator("affects_x", "affects_y", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


class ConfoundExtraction(BaseModel):
    """Structured output for confound extraction."""

    confounds: list[Confound] = Field(
        default_factory=list,
        description="List of confounding variables mentioned by the student",
    )
    hypothesis_undermined: bool = Field(
        default=False,
        description=(
            "Whether a confound so thoroughly explains both X and Y that "
            "the causal hypothesis is likely spurious (backward jump to Stage 3A)"
        ),
    )

    @field_validator("hypothesis_undermined", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Stage 4: Operationalization ====================


class MeasurementExtraction(BaseModel):
    """Extract measurement specifications from conversation."""

    x_measurement: str | None = Field(
        default=None,
        max_length=300,
        description="How the independent variable (X) is measured",
    )
    y_measurement: str | None = Field(
        default=None,
        max_length=300,
        description="How the dependent variable (Y) is measured",
    )


class OperationalizationProgress(BaseModel):
    """Structured output for operationalization progress."""

    data_alignment_evaluated: bool = Field(
        description=(
            "Whether student has discussed measurement quality and data alignment"
        ),
    )
    measurement_quality_notes: str | None = Field(
        default=None,
        max_length=500,
        description="Brief notes on what was discussed about measurement quality",
    )
    alignment_too_poor: bool = Field(
        default=False,
        description=(
            "Whether dataset measures are too different from ideal measurements — "
            "need different dataset or scope revision (backward jump to Stage 3C)"
        ),
    )

    @field_validator("data_alignment_evaluated", "alignment_too_poor", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Stage 5A: Analysis ====================


class AnalysisProgress(BaseModel):
    """Structured output for analysis progress."""

    statistical_method: str | None = Field(
        default=None,
        max_length=150,
        description=(
            "Statistical method chosen "
            "(e.g., 'regression', 't-test', 'ANOVA', 'chi-square', 'logistic regression')"
        ),
    )
    h0_defined: bool = Field(
        default=False,
        description="Whether null hypothesis (H0) has been defined",
    )
    h1_defined: bool = Field(
        default=False,
        description="Whether alternative hypothesis (H1) has been defined",
    )

    @field_validator("h0_defined", "h1_defined", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Stage 5B: Resources ====================


class Stage5BEthicsProgress(BaseModel):
    """Structured output for Stage 5B ethics discussion progress."""

    ethics_discussed: bool = Field(
        description="Whether ethical considerations have been discussed",
    )
    ethics_notes: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "Brief notes on ethical considerations discussed "
            "(e.g., 'secondary data — no direct participants', "
            "'involves surveys — institutional review recommended')"
        ),
    )

    @field_validator("ethics_discussed", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


class Stage5BSoftwareProgress(BaseModel):
    """Structured output for Stage 5B software discussion progress."""

    software_discussed: bool = Field(
        description="Whether software selection has been discussed",
    )
    selected_software: str | None = Field(
        default=None,
        max_length=150,
        description="Software choice (R, Python, SPSS, Excel, etc.)",
    )
    rationale: str | None = Field(
        default=None,
        max_length=500,
        description="Why this software was chosen",
    )

    @field_validator("software_discussed", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


class Stage5BTimelineProgress(BaseModel):
    """Structured output for Stage 5B timeline discussion progress."""

    timeline_discussed: bool = Field(
        description="Whether research timeline has been discussed",
    )
    data_collection_weeks: int | None = Field(
        default=None,
        description="Estimated weeks for data collection",
    )
    analysis_weeks: int | None = Field(
        default=None,
        description="Estimated weeks for analysis",
    )
    writeup_weeks: int | None = Field(
        default=None,
        description="Estimated weeks for write-up",
    )
    total_weeks: int | None = Field(
        default=None,
        description="Total estimated timeline in weeks",
    )

    @field_validator("timeline_discussed", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Completion ====================


class CompletionIntent(BaseModel):
    """Structured output for completion handler intent classification."""

    wants_pdf: bool = Field(
        description="User explicitly wants PDF summary generated",
    )
    has_question: bool = Field(
        description="User has a follow-up question about their research plan",
    )
    wants_to_end: bool = Field(
        description="User is done and wants to end conversation",
    )
    question_topic: str | None = Field(
        default=None,
        max_length=300,
        description="If has_question=True, what stage/topic is the question about?",
    )

    @field_validator("wants_pdf", "has_question", "wants_to_end", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


# ==================== Semantic Validation ====================


class VariableSubstantiveness(BaseModel):
    """Classify whether a proposed research variable is substantive."""

    is_substantive: bool = Field(
        description=(
            "True if the text is a meaningful research variable or concept "
            "(even a single word like 'age', 'income', 'pollution'). "
            "False if it's a trivial response (yes/no/ok/idk), random characters, "
            "or not a research concept at all."
        ),
    )

    @field_validator("is_substantive", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)


class ReviewModeClassification(BaseModel):
    """Classify whether a student wants guided or independent paper review."""

    mode: str | None = Field(
        default=None,
        max_length=50,
        description=(
            "The student's preferred review mode: "
            "'guided' if they want the mentor to walk through papers together, "
            "'independent' if they want to review papers on their own, "
            "null if the message doesn't express a clear preference."
        ),
    )
