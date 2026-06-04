"""Pydantic schemas for the Question Workshop pipeline.

Each stage produces a structured output validated by these models.
API request/response models are also defined here.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from research_mentor.prompt_limits import Target

# ---------------------------------------------------------------------------
# Stage 1: Domain Exploration
# ---------------------------------------------------------------------------


class ExplorationData(BaseModel):
    """Structured output for domain exploration stage."""

    domains: list[str] = Field(
        description=(
            "Scientific domains involved (1-3 domain IDs). "
            "Examples: fluid_dynamics, mechanics, microbiology, reaction_kinetics"
        ),
    )

    phenomena: list[str] = Field(
        description=(
            "Interesting phenomena identified (1-5). "
            "Must be observable, surprising, and accessible to high school students."
        ),
    )

    materials: list[str] = Field(
        description=(
            "Required materials for experiments (3+ items). "
            "Must be accessible to a high school lab."
        ),
    )

    feasibility_assessment: str = Field(
        max_length=50,
        description="Initial feasibility: HIGH, MEDIUM, or LOW",
    )

    surprise_factor: str | None = Field(
        None,
        max_length=500,
        description="Brief description of what makes this surprising.",
    )

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "domains": ["fluid_dynamics", "thermal_physics"],
                "phenomena": [
                    "Leidenfrost effect",
                    "droplet levitation",
                    "vapor cushion dynamics",
                ],
                "materials": ["water", "metal hot plate", "thermometer", "pipette", "timer"],
                "feasibility_assessment": "MEDIUM",
                "surprise_factor": "Droplet floats without touching the scorching surface",
            },    })


# ---------------------------------------------------------------------------
# Stage 2: Problem Generation
# ---------------------------------------------------------------------------


class ProblemComponents(BaseModel):
    """Structured output for problem generation stage (IYPT-style format)."""

    title: Annotated[str, Target(words=4)] = Field(
        max_length=100,
        description="Problem title (2-4 words, catchy and memorable).",
    )

    description: Annotated[str, Target(words=80)] = Field(
        max_length=800,
        description="Problem description (2-4 sentences). Setup + phenomenon + surprise.",
    )

    investigation: Annotated[str, Target(words=45)] = Field(
        max_length=500,
        description=(
            "Investigation directive (1-2 sentences). "
            "Must start with Investigate/Study/Explore/Examine "
            "(or Derive/Prove/Show for theoretical problems)."
        ),
    )

    core_concepts: list[str] = Field(
        description="Core scientific concepts involved (2-6).",
    )

    @field_validator("investigation")
    @classmethod
    def validate_investigation_starts_with_action(cls, v: str) -> str:
        """Ensure investigation starts with proper action verb."""
        action_verbs = [
            "investigate", "study", "explore", "examine",
            "derive", "prove", "show", "establish",
        ]
        if not any(v.lower().startswith(verb) for verb in action_verbs):
            raise ValueError(
                f"Investigation must start with one of: {', '.join(action_verbs)}"
            )
        return v

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "title": "Vapor Maze",
                "description": (
                    "Drop a tiny water droplet onto a scorching metal maze. "
                    "Watch as it magically glides, levitated by its own vapor cushion. "
                    "Can you guide the droplet through the intricate pathways "
                    "without it touching the metal?"
                ),
                "investigation": (
                    "Investigate how the maze geometry, surface temperature, and "
                    "droplet size affect the droplet's velocity, stability, and "
                    "success rate in navigating the vapor-supported path."
                ),
                "core_concepts": [
                    "Leidenfrost effect",
                    "fluid dynamics",
                    "heat transfer",
                    "surface tension",
                ],
            },    })


# ---------------------------------------------------------------------------
# Stage 3: Feasibility Validation
# ---------------------------------------------------------------------------


class FeasibilityScores(BaseModel):
    """Structured output for feasibility validation (all scores 1-10)."""

    material_accessibility_score: int = Field(
        ge=1, le=10,
        description="Material accessibility (10=household items, 1=rare equipment).",
    )
    experimental_difficulty_score: int = Field(
        ge=1, le=10,
        description="Experimental difficulty (10=simple, 1=expert-level).",
    )
    measurability_score: int = Field(
        ge=1, le=10,
        description="Measurability (10=ruler/stopwatch, 1=qualitative only).",
    )
    complexity_score: int = Field(
        ge=1, le=10,
        description="Complexity (target 5-7 for high school research).",
    )
    time_feasibility_score: int = Field(
        ge=1, le=10,
        description="Time feasibility (10=1-2 months, 1=>6 months).",
    )
    safety_level: str = Field(
        max_length=50,
        description="SAFE, REQUIRES_SUPERVISION, or UNSAFE.",
    )
    overall_feasibility: str = Field(
        max_length=50,
        description="HIGH, MEDIUM, or LOW.",
    )
    recommended: str = Field(
        max_length=50,
        description="YES, WITH_MODIFICATIONS, or NO.",
    )
    improvements_suggested: list[str] = Field(
        default_factory=list,
        description="Suggested improvements (empty if recommended=YES).",
    )

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "material_accessibility_score": 8,
                "experimental_difficulty_score": 6,
                "measurability_score": 7,
                "complexity_score": 6,
                "time_feasibility_score": 8,
                "safety_level": "REQUIRES_SUPERVISION",
                "overall_feasibility": "MEDIUM",
                "recommended": "YES",
                "improvements_suggested": [],
            },    })


# ---------------------------------------------------------------------------
# Stage 4: Creative Refinement
# ---------------------------------------------------------------------------


class RefinedProblem(BaseModel):
    """Structured output for creative refinement stage."""

    title: Annotated[str, Target(words=4)] = Field(
        max_length=100,
        description="Refined title (2-4 words, enhanced for catchiness).",
    )
    description: Annotated[str, Target(words=80)] = Field(
        max_length=800,
        description="Refined description (2-4 sentences, enhanced for vividness).",
    )
    investigation: Annotated[str, Target(words=45)] = Field(
        max_length=500,
        description="Refined investigation (1-2 sentences, enhanced for clarity).",
    )
    engagement_score: int = Field(
        ge=1, le=10,
        description="Engagement score (1-10, how exciting for students).",
    )
    novelty_assessment: str = Field(
        default="",
        max_length=800,
        description=(
            "Assessment of whether this problem is novel or resembles "
            "well-known competition problems. Suggestions to differentiate if needed."
        ),
    )
    changes_made: list[str] = Field(
        default_factory=list,
        description="List of specific refinements made.",
    )

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "title": "Vapor Maze",
                "description": (
                    "Drop a tiny water droplet onto a scorching metal maze. "
                    "Watch as it magically glides, levitated by its own vapor cushion."
                ),
                "investigation": (
                    "Investigate how the maze geometry, surface temperature, and "
                    "droplet size affect the droplet's velocity and stability."
                ),
                "engagement_score": 8,
                "novelty_assessment": (
                    "Novel — no well-known competition problem "
                    "uses maze geometry with Leidenfrost effect."
                ),
                "changes_made": [
                    "Enhanced title to 'Vapor Maze' for evocative imagery",
                    "Added sensory verbs ('watch', 'glides', 'levitated')",
                ],
            },    })


# ---------------------------------------------------------------------------
# API Request / Response
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Request body for single problem generation."""

    field: str = Field(description="Scientific field: physics, chemistry, or biology")
    problem_type: str = Field(
        default="experimental",
        description="experimental, simulation, data_analysis, or theoretical",
    )
    domain: str | None = Field(
        default=None,
        description="Optional domain within field (e.g. fluid_dynamics)",
    )
    user_suggestion: str | None = Field(
        default=None,
        description="Optional seed idea from the user",
    )
    project_id: str | None = Field(
        default=None,
        description="Optional project to link the generated problem to",
    )
