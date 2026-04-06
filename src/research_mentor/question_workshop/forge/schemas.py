"""Pydantic schemas for the Forge Workshop pipeline.

Each stage produces a structured output validated by these models.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOL_CATEGORIES = frozenset({
    "software", "hardware", "methods_protocols",
    "data_tools", "field_equipment", "any",
})

TOOL_CATEGORY_LABELS: dict[str, str] = {
    "software": "software tools (analysis scripts, packages, pipelines, visualization, databases)",
    "hardware": "hardware instruments (sensors, measurement devices, lab equipment, electronics)",
    "methods_protocols": "methods and protocols (lab procedures, sample preparation, workflows)",
    "data_tools": "data infrastructure (collection systems, annotation tools, standards, formats)",
    "field_equipment": "field equipment (portable devices, sampling gear, monitoring systems)",
    "any": "any type of research tool (software, hardware, methods, data, or field equipment)",
}


# ---------------------------------------------------------------------------
# API Request
# ---------------------------------------------------------------------------


class ForgeRequest(BaseModel):
    """Request body for tool development generation."""

    field: str = Field(description="Scientific field: physics, chemistry, or biology")
    tool_category: str = Field(
        default="any",
        description="software, hardware, methods_protocols, data_tools, field_equipment, or any",
    )
    seed_idea: str | None = Field(
        default=None,
        description="Optional seed idea from the user",
    )
    project_id: str | None = Field(
        default=None,
        description="Optional project to link the generated tool ideas to",
    )


# ---------------------------------------------------------------------------
# Stage 1: Domain Workflow Mapping
# ---------------------------------------------------------------------------


class WorkflowMap(BaseModel):
    """Structured output for domain workflow mapping stage."""

    research_stages: list[str] = Field(
        description=(
            "Key stages in this field's research workflow (3-6). "
            "Examples: sample_collection, data_acquisition, analysis, visualization"
        ),
    )

    pain_points: list[str] = Field(
        description=(
            "Workflow bottlenecks where researchers spend disproportionate "
            "time, effort, or money (3-5 items)."
        ),
    )

    current_tools: list[str] = Field(
        description="Tools and methods currently used in this workflow (3-8 items).",
    )

    field_context: str = Field(
        max_length=1000,
        description="Brief context about the field's research workflow and its challenges.",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "research_stages": [
                "sample_collection", "DNA_extraction",
                "sequencing", "bioinformatics_analysis", "visualization",
            ],
            "pain_points": [
                "Manual colony counting takes hours per experiment",
                "No standardized protocol for field sample preservation",
                "Data format incompatibility between sequencing platforms",
            ],
            "current_tools": [
                "manual counting", "ImageJ", "BLAST", "R/Bioconductor",
            ],
            "field_context": (
                "Soil microbiology research involves collecting field samples, "
                "extracting DNA, sequencing via 16S rRNA amplicon methods, and "
                "analyzing community composition. The bioinformatics pipeline "
                "is the primary bottleneck for non-computational researchers."
            ),
        },
    })


# ---------------------------------------------------------------------------
# Stage 2: Bottleneck Identification
# ---------------------------------------------------------------------------


class Bottleneck(BaseModel):
    """A single capability gap in the research workflow."""

    name: str = Field(
        max_length=100,
        description="Short name for this bottleneck.",
    )
    description: str = Field(
        max_length=500,
        description="What capability is missing and why it matters.",
    )
    affected_stage: str = Field(
        max_length=150,
        description="Which workflow stage this bottleneck affects.",
    )
    severity: str = Field(
        max_length=50,
        description="HIGH, MEDIUM, or LOW — how much this slows research.",
    )


class BottleneckAnalysis(BaseModel):
    """Structured output for bottleneck identification stage."""

    bottlenecks: list[Bottleneck] = Field(
        description="3-5 concrete capability gaps where a new tool would have high impact.",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "bottlenecks": [
                {
                    "name": "Manual colony counting",
                    "description": (
                        "Researchers manually count bacterial colonies on agar plates, "
                        "taking 2-4 hours per experiment with high inter-observer variability."
                    ),
                    "affected_stage": "data_acquisition",
                    "severity": "HIGH",
                },
                {
                    "name": "No field-to-lab sample tracker",
                    "description": (
                        "Sample metadata is recorded on paper in the field and manually "
                        "entered into spreadsheets, causing transcription errors."
                    ),
                    "affected_stage": "sample_collection",
                    "severity": "MEDIUM",
                },
            ],
        },
    })


# ---------------------------------------------------------------------------
# Stage 3: Solution Design
# ---------------------------------------------------------------------------


class ToolConcept(BaseModel):
    """A single tool concept designed to address a research bottleneck."""

    name: str = Field(
        max_length=100,
        description="Tool name (2-5 words, descriptive and memorable).",
    )
    problem_addressed: str = Field(
        max_length=800,
        description="What bottleneck this tool solves and why it matters.",
    )
    solution_design: str = Field(
        max_length=800,
        description="How the tool works at a high level.",
    )
    mvp_spec: str = Field(
        max_length=800,
        description="Minimum viable version a student could actually build.",
    )
    required_skills: list[str] = Field(
        description="Technologies or skills needed to build the MVP (2-6 items).",
    )
    required_resources: list[str] = Field(
        description="Materials, equipment, or data needed (2-6 items).",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "ColonySnap Counter",
            "problem_addressed": (
                "Manual colony counting on agar plates takes 2-4 hours per experiment "
                "with high inter-observer variability. This tool automates counting "
                "from smartphone photos, saving researchers hours of tedious work."
            ),
            "solution_design": (
                "A Python script that takes a photo of an agar plate, applies "
                "image segmentation to identify individual colonies, and returns "
                "a count with a confidence score and annotated image."
            ),
            "mvp_spec": (
                "Python script using OpenCV for circle detection on high-contrast "
                "agar plate photos. Input: JPEG image. Output: colony count + "
                "annotated image with detected colonies circled. "
                "Validate against manual counts on 20 plates."
            ),
            "required_skills": [
                "Python programming", "OpenCV basics",
                "image processing fundamentals",
            ],
            "required_resources": [
                "computer with Python", "smartphone camera",
                "agar plates with colonies for testing",
            ],
        },
    })


class ToolConcepts(BaseModel):
    """Structured output for solution design stage."""

    tools: list[ToolConcept] = Field(
        description="3-5 tool concepts, each addressing a different bottleneck.",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "tools": [{
                "name": "ColonySnap Counter",
                "problem_addressed": "Manual colony counting takes 2-4 hours.",
                "solution_design": "OpenCV-based colony detection from photos.",
                "mvp_spec": "Python script using OpenCV for circle detection.",
                "required_skills": ["Python", "OpenCV basics"],
                "required_resources": ["computer", "smartphone camera"],
            }],
        },
    })


# ---------------------------------------------------------------------------
# Stage 4: Feasibility & Impact Scoring
# ---------------------------------------------------------------------------


class ScoredTool(BaseModel):
    """Feasibility and impact scores for a single tool concept."""

    name: str = Field(
        max_length=100,
        description="Tool name (must match a tool from Stage 3).",
    )
    student_buildable_score: int = Field(
        ge=1, le=10,
        description="Can a student at this level build this? (10=easy, 1=expert-only).",
    )
    novelty_score: int = Field(
        ge=1, le=10,
        description="Does this tool already exist? (10=nothing like it, 1=widely available).",
    )
    impact_score: int = Field(
        ge=1, le=10,
        description="How many researchers would benefit? (10=entire field, 1=very niche).",
    )
    validation_path_score: int = Field(
        ge=1, le=10,
        description="How easy to prove it works? (10=simple comparison, 1=no clear test).",
    )
    overall_feasibility: str = Field(
        max_length=50,
        description="HIGH, MEDIUM, or LOW.",
    )
    validation_approach: str = Field(
        max_length=500,
        description="How the student would validate this tool works correctly.",
    )


class ScoredTools(BaseModel):
    """Structured output for feasibility & impact scoring stage."""

    scored_tools: list[ScoredTool] = Field(
        description="Scores for each tool concept from Stage 3.",
    )
    recommended_starting_point: str = Field(
        max_length=800,
        description="Which tool to build first and why (best buildability + impact combo).",
    )
    recommended: str = Field(
        max_length=50,
        description="YES, WITH_MODIFICATIONS, or NO — overall recommendation.",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "scored_tools": [
                {
                    "name": "ColonySnap Counter",
                    "student_buildable_score": 8,
                    "novelty_score": 6,
                    "impact_score": 7,
                    "validation_path_score": 9,
                    "overall_feasibility": "HIGH",
                    "validation_approach": (
                        "Count colonies manually on 20 plates, then run the tool "
                        "on photos of the same plates. Compare automated vs manual "
                        "counts and calculate agreement percentage."
                    ),
                },
            ],
            "recommended_starting_point": (
                "Start with ColonySnap Counter: highest buildability (8/10), "
                "clear validation path (compare to manual counts), and immediate "
                "practical value for any microbiology lab."
            ),
            "recommended": "YES",
        },
    })
