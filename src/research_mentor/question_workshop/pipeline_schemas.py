"""Shared schemas for the phase pipeline — split cognitive load across focused calls.

These schemas support the 4-step pipeline that replaces the monolithic CitedResponse
pattern in interactive workshops (Modeling, Theory, Gaps):

Step 1 (parallel): ContextSummary + ProgressExtraction (existing per-workshop schemas)
Step 2 (sequential): PedagogicalDecision
Step 3 (sequential): CitedResponse (existing, but with a much simpler prompt)
Step 4 (optional): PreloadingCheck
"""

from pydantic import BaseModel, Field, field_validator

_TRUTHY = frozenset({"true", "yes", "1", "t", "y"})
_FALSY = frozenset({"false", "no", "0", "f", "n"})


# ==================== Step 3: Cited Response ====================


class CitedResponse(BaseModel):
    """Conversational response with paper citations.

    Shared across all workshops. The LLM returns text + which papers it
    referenced (by number from the prompt list). Code expands paper numbers
    into real citations via format_cited_response().
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


def _coerce_bool(v: object) -> object:
    """Coerce string bools from LLM output (e.g. "True" -> True)."""
    if isinstance(v, str):
        low = v.strip().lower()
        if low in _TRUTHY:
            return True
        if low in _FALSY:
            return False
    return v


# ==================== Step 1: Context Summary ====================


class ContextSummary(BaseModel):
    """What has the student established so far?

    Extracted from conversation history. Used by PedagogicalDecision (Step 2)
    to decide what to probe next. NEVER shown to the student — only feeds
    into the decision step.
    """

    student_contributions: list[str] = Field(
        description=(
            "Key points the STUDENT has contributed (not the mentor). "
            "Use the student's own words. 3-5 items max."
        ),
    )
    questions_already_asked: list[str] = Field(
        description=(
            "Questions the MENTOR has already asked in this conversation. "
            "Used to avoid repetition. 2-4 items, most recent first."
        ),
    )
    latest_student_point: str = Field(
        max_length=300,
        description=(
            "What the student's most recent message is about, in one sentence."
        ),
    )
    conversation_stage: str = Field(
        max_length=50,
        description=(
            "How far along the conversation is: "
            "'early' (1-2 exchanges), 'developing' (3-5), 'mature' (6+)"
        ),
    )


# ==================== Step 2: Pedagogical Decision ====================


class PedagogicalDecision(BaseModel):
    """What ONE thing to probe next and how.

    Takes context summary + progress gaps + phase-specific rules as input.
    Outputs a focused decision that the response generator executes.
    """

    focus_point: str = Field(
        max_length=300,
        description=(
            "The single most important thing to probe next, in one sentence. "
            "Must be something the student has NOT yet addressed."
        ),
    )
    pedagogical_move: str = Field(
        max_length=50,
        description=(
            "One of: "
            "'open_question' (ask what they think — no hints), "
            "'challenge' (ask WHY or pose a counterfactual), "
            "'probe_deeper' (ask for specifics about something they said), "
            "'redirect' (they went off track, bring back to phase topic), "
            "'transition' (ready for next phase, acknowledge and bridge), "
            "'empathize' (they are stuck, lower the bar and encourage)"
        ),
    )
    papers_to_reference: list[int] = Field(
        default_factory=list,
        description=(
            "Paper numbers relevant to this focus point. "
            "0-2 papers max. Empty list if no papers are relevant."
        ),
    )
    avoid: str = Field(
        default="",
        max_length=300,
        description=(
            "What NOT to do in the response. Examples: "
            "'do not suggest components', 'do not repeat the question about X', "
            "'do not list examples'. Empty string if no specific avoidance needed."
        ),
    )


# ==================== Step 4: Preloading Check ====================


class PreloadingCheck(BaseModel):
    """Does the response pre-load answers the student hasn't articulated?

    Pre-loading = providing information, listing examples, naming specific
    concepts, or narrating the answer before the student has said it.

    NOT pre-loading: asking questions, acknowledging what the student said,
    using generic references to the student's own words.
    """

    contains_preloading: bool = Field(
        description=(
            "TRUE if the response contains claims, answers, explanations, "
            "examples, or specific content that the STUDENT has not articulated. "
            "Asking questions is NOT preloading. Acknowledging what the student "
            "said is NOT preloading."
        ),
    )
    suggestion: str = Field(
        default="",
        max_length=500,
        description=(
            "If preloading detected: how to rephrase as a question instead of "
            "a statement. If no preloading: empty string."
        ),
    )

    @field_validator("contains_preloading", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> object:
        return _coerce_bool(v)
