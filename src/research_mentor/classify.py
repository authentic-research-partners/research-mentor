"""PDF structural classification for smart vision routing.

Inspects PDF structure using pdfplumber to classify documents as text-only,
visual (contains diagrams/charts), or scanned (no text layer). This enables
routing text-only PDFs to pdfplumber (free/fast) while sending visual or
scanned documents to SOTA vision backends.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from loguru import logger

# Internal heuristics — not user-configurable
_IMAGE_COVERAGE_THRESHOLD = 0.05  # 5% of page area
_MIN_TEXT_LENGTH = 50  # chars to consider page as having text


class DocumentClass(enum.Enum):
    """Classification of a PDF document based on its structural content."""

    TEXT_ONLY = "text_only"  # All pages: text present, no significant images → pdfplumber
    VISUAL = "visual"  # At least one page has significant image coverage → SOTA
    SCANNED = "scanned"  # At least one page has no text layer → SOTA


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Result of PDF structural classification."""

    classification: DocumentClass
    total_pages: int
    pages_with_text: int
    pages_with_images: int
    max_image_coverage: float  # 0.0-1.0


def classify_pdf(file_path: str | Path) -> ClassificationResult:
    """Classify a PDF document based on its structural content.

    Inspects each page for text content and image coverage to determine
    whether the document can be handled by pdfplumber (text-only) or
    needs a SOTA vision backend (visual/scanned).

    Args:
        file_path: Path to the PDF file.

    Returns:
        ClassificationResult with the document class and page statistics.
    """
    path = Path(file_path)
    has_scanned = False
    has_visual = False
    pages_with_text = 0
    pages_with_images = 0
    max_image_coverage = 0.0

    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        if total_pages == 0:
            return ClassificationResult(
                classification=DocumentClass.SCANNED,
                total_pages=0,
                pages_with_text=0,
                pages_with_images=0,
                max_image_coverage=0.0,
            )

        for page in pdf.pages:
            # Check text content
            text = page.extract_text() or ""
            has_text = len(text.strip()) >= _MIN_TEXT_LENGTH

            # Calculate image coverage
            page_area = page.width * page.height
            image_area = 0.0
            if page.images and page_area > 0:
                for img in page.images:
                    img_width = img["x1"] - img["x0"]
                    img_height = img["bottom"] - img["top"]
                    image_area += img_width * img_height
            coverage = image_area / page_area if page_area > 0 else 0.0
            has_significant_images = coverage > _IMAGE_COVERAGE_THRESHOLD

            if has_text:
                pages_with_text += 1
            if has_significant_images:
                pages_with_images += 1
            max_image_coverage = max(max_image_coverage, coverage)

            # Classify this page
            if not has_text and page.images:
                has_scanned = True
            elif has_text and has_significant_images:
                has_visual = True

    # Document-level classification: worst case wins
    if has_scanned:
        classification = DocumentClass.SCANNED
    elif has_visual:
        classification = DocumentClass.VISUAL
    else:
        classification = DocumentClass.TEXT_ONLY

    logger.debug(
        "classify_pdf {}: {} (pages={}, text={}, images={}, max_coverage={:.1%})",
        path.name,
        classification.value,
        total_pages,
        pages_with_text,
        pages_with_images,
        max_image_coverage,
    )

    return ClassificationResult(
        classification=classification,
        total_pages=total_pages,
        pages_with_text=pages_with_text,
        pages_with_images=pages_with_images,
        max_image_coverage=max_image_coverage,
    )
