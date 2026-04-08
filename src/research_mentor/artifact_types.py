"""Artifact type definitions — single source of truth.

Used by the API (format validation, upload endpoint), vision prompts
(type labels), and served to the UI via /api/artifact-types.

Adding or changing a type here automatically updates all three.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactTypeInfo:
    """Complete definition of an artifact type."""

    value: str
    label: str
    description: str
    accept: str  # comma-separated file extensions for the file input
    placeholder: str  # placeholder text for the description field
    vision_label: str  # how the type is described in the vision prompt


ARTIFACT_TYPES: tuple[ArtifactTypeInfo, ...] = (
    ArtifactTypeInfo(
        value="paper",
        label="Paper",
        description="Academic paper or journal article (PDF only)",
        accept=".pdf",
        placeholder=(
            "e.g., Key paper on CRISPR efficiency in zebrafish, focus on Methods section"
        ),
        vision_label="a figure extracted from an academic paper",
    ),
    ArtifactTypeInfo(
        value="document",
        label="Document",
        description="Your own writing — lab reports, drafts, notes, essays",
        accept=".pdf,.docx,.txt,.md,.rst",
        placeholder=(
            "e.g., My first draft of the literature review for the pH experiment"
        ),
        vision_label="a document page",
    ),
    ArtifactTypeInfo(
        value="data",
        label="Data",
        description=(
            "Upload measurements, observations, or calculations from your experiments. "
            "Your data will be automatically analyzed and presented to your mentor. "
            "CSV or TSV format with column headers in the first row. "
            "Keep numeric columns as plain numbers (no units or text mixed in)"
        ),
        accept=".csv,.tsv",
        placeholder=(
            "e.g., Growth height (cm) for 30 plants across 3 fertilizer types "
            "(A, B, control), 10 per group"
        ),
        vision_label="a data file",
    ),
    ArtifactTypeInfo(
        value="photograph",
        label="Photograph",
        description="Photos of lab setups, field sites, equipment, specimens",
        accept=".png,.jpg,.jpeg,.gif,.webp,.bmp,.tiff,.tif",
        placeholder=(
            "e.g., Photo of my experimental setup showing the beam deflection apparatus"
        ),
        vision_label=(
            "a photograph (lab setup, field site, equipment, or specimen)"
        ),
    ),
    ArtifactTypeInfo(
        value="scientific_image",
        label="Scientific Image",
        description=(
            "Instrument output — microscopy, gels, spectra, X-rays, blots, chromatography"
        ),
        accept=".png,.jpg,.jpeg,.tiff,.tif,.bmp",
        placeholder=(
            "e.g., 400x magnification of onion cells after staining, or NMR spectrum "
            "with unusual peaks around 7 ppm"
        ),
        vision_label=(
            "instrument output (microscopy, gels, spectra, X-rays, blots, chromatography)"
        ),
    ),
    ArtifactTypeInfo(
        value="chart",
        label="Chart",
        description="Charts, graphs, plots, flowcharts, concept maps, diagrams",
        accept=".png,.jpg,.jpeg,.gif,.webp",
        placeholder=(
            "e.g., Graph of growth rate vs. concentration from my results, "
            "unsure about the outlier at 50mM"
        ),
        vision_label="a chart, graph, plot, flowchart, concept map, or diagram",
    ),
    ArtifactTypeInfo(
        value="handwriting",
        label="Handwriting",
        description="Handwritten notes, equations, lab notebook pages",
        accept=".png,.jpg,.jpeg,.tiff,.tif,.bmp",
        placeholder=(
            "e.g., My notes from today's lab session with the equations I derived"
        ),
        vision_label="handwritten notes, equations, or lab notebook page",
    ),
)

# Lookup helpers
ARTIFACT_TYPE_MAP: dict[str, ArtifactTypeInfo] = {t.value: t for t in ARTIFACT_TYPES}

ALLOWED_FORMATS: dict[str, frozenset[str]] = {
    t.value: frozenset(ext.strip() for ext in t.accept.split(","))
    for t in ARTIFACT_TYPES
}

VISION_LABELS: dict[str, str] = {t.value: t.vision_label for t in ARTIFACT_TYPES}
# Internal type for images extracted from PDFs (not user-facing)
VISION_LABELS["paper_figure"] = "a figure extracted from an academic paper"
