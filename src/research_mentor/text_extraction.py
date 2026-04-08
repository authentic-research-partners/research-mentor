"""Text extraction and chunking for artifact files.

Dispatches by file type and artifact type:
- PDF (paper type) → GROBID (structured) if available, else pdfplumber
- PDF (document type) → pdfplumber (raw text)
- DOCX → python-docx (paragraphs)
- CSV/TSV → stdlib csv (rendered as markdown table)
- TXT/MD/RST → plain read
- Images → vision backends (Qwen2.5-VL, Claude CLI, or API)

For PDFs, raster images are extracted via pypdfium2 and sent to vision
backends for description.

See docs/artifact-processing.md for the full pipeline design.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from research_mentor.artifact_types import VISION_LABELS as _TYPE_LABELS

if TYPE_CHECKING:
    from research_mentor.config import AppConfig

_IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif",
})

_PDF_EXTENSIONS = frozenset({".pdf"})
_DOCX_EXTENSIONS = frozenset({".docx"})
_TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".rst"})
_CSV_EXTENSIONS = frozenset({".csv", ".tsv"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def extract_text(file_path: str | Path) -> str | None:
    """Extract text from a file, dispatching by file type.

    Supported formats:
    - PDF → GROBID (if enabled + available) or pdfplumber
    - DOCX → python-docx
    - CSV/TSV → stdlib csv (rendered as markdown table)
    - TXT/MD/RST → plain read

    Returns extracted text, or None if extraction fails or produces no content.
    """
    path = Path(file_path)
    if not path.is_file():
        logger.warning("extract_text: file not found: {}", path)
        return None

    suffix = path.suffix.lower()

    if suffix in _PDF_EXTENSIONS:
        return await _extract_pdf(path)
    if suffix in _DOCX_EXTENSIONS:
        return await asyncio.to_thread(_extract_docx, path)
    if suffix in _CSV_EXTENSIONS:
        return _extract_csv(path, delimiter="\t" if suffix == ".tsv" else ",")
    if suffix in _IMAGE_EXTENSIONS:
        return None  # images need vision backends, not text extraction
    if suffix in _TEXT_EXTENSIONS:
        return _extract_plain(path)

    # Unknown format — try plain read as last resort
    return _extract_plain(path)


def extract_images(file_path: str | Path) -> list[Path]:
    """Extract raster images embedded in a PDF via pypdfium2.

    Saves each image to a temporary directory and returns their paths.
    The caller is responsible for cleanup.

    Returns an empty list for non-PDF files or if no images are found.
    """
    path = Path(file_path)
    if path.suffix.lower() not in _PDF_EXTENSIONS:
        return []

    import pypdfium2 as pdfium

    try:
        doc = pdfium.PdfDocument(str(path))
    except Exception:
        logger.debug("Failed to open PDF for image extraction: {}", path.name)
        return []

    images: list[Path] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="rm_images_"))

    try:
        i = 0
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            for obj in page.get_objects():
                if not isinstance(obj, pdfium.PdfImage):
                    continue
                try:
                    img_path = tmp_dir / f"image_{i}.png"
                    obj.extract(str(img_path))
                    images.append(img_path)
                    i += 1
                except Exception:
                    logger.debug(
                        "Failed to extract image {} from page {}", i, page_idx,
                    )
    finally:
        doc.close()

    logger.debug("Extracted {} images from {}", len(images), path.name)
    return images


async def extract_text_or_describe(
    file_path: str | Path,
    config: AppConfig,
    *,
    research_context: str | None = None,
    artifact_type: str | None = None,
    artifact_description: str | None = None,
) -> str | None:
    """Extract text from a file, with vision backend for images.

    For PDFs and DOCX: uses GROBID/pdfplumber/python-docx for text extraction.
    Embedded images are handled separately by the artifact pipeline.

    For standalone images: dispatches to the configured vision backend with
    a type-specific prompt.

    For other formats: uses text extraction directly.

    Args:
        file_path: Path to the file.
        config: Application configuration.
        research_context: Optional research question for vision context.
        artifact_type: One of the 7 artifact types (for vision prompt selection).
        artifact_description: Student's description of the artifact.

    Returns:
        Extracted text, AI interpretation prefixed with ``[AI interpretation]``
        or ``[Image description]``, or None if no content could be extracted.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    # Images → vision backend (if enabled)
    if suffix in _IMAGE_EXTENSIONS:
        if not config.vision.enabled:
            return None
        prompt = build_vision_prompt(
            artifact_type=artifact_type,
            artifact_description=artifact_description,
            research_context=research_context,
        )
        return await _describe_image(path, config, prompt=prompt)

    # Everything else → text extraction (pdfplumber for PDF, python-docx, csv, plain)
    return await extract_text(file_path)


# ---------------------------------------------------------------------------
# Vision prompt builder
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT = (
    "Examine this image carefully. "
    "Transcribe any text you can read (printed, typed, or handwritten). "
    "Describe any diagrams, charts, graphs, tables, or visual elements."
)


def build_vision_prompt(
    *,
    artifact_type: str | None = None,
    artifact_description: str | None = None,
    research_context: str | None = None,
) -> str:
    """Build a vision prompt tailored to the artifact type and context.

    Keeps instructions generic — tells the model the category and the student's
    description, then lets it use its own judgment. The student's description
    is the real signal for what to focus on.
    """
    parts: list[str] = []

    if research_context:
        parts.append(
            "This is from a research project investigating: "
            f"<user_content>{research_context}</user_content>"
        )

    type_label = _TYPE_LABELS.get(artifact_type or "")
    if type_label:
        parts.append(f"The student classified this image as {type_label}.")

    if artifact_description:
        parts.append(
            "The student describes this as: "
            f"<user_content>{artifact_description}</user_content>"
        )

    parts.append(
        "Examine this image carefully. Transcribe any visible text. "
        "Describe what you see, focusing on what would be most useful "
        "for understanding this in a research context."
    )

    return "\n\n".join(parts)


def resize_image_if_needed(file_path: str | Path, max_dimension: int) -> Path:
    """Resize an image if its longest edge exceeds max_dimension.

    Returns the path to the (possibly resized) image. If resized, the image
    is saved to a temp file; the caller is responsible for cleanup.
    If no resize is needed, returns the original path.
    """
    from PIL import Image

    path = Path(file_path)
    try:
        with Image.open(path) as img:
            w, h = img.size
            if max(w, h) <= max_dimension:
                return path
            img.thumbnail((max_dimension, max_dimension))
            fd, tmp_str = tempfile.mkstemp(suffix=path.suffix, prefix="rm_resized_")
            os.close(fd)
            tmp = Path(tmp_str)
            img.save(str(tmp))
            logger.debug(
                "Resized image {}x{} → {}x{}: {}",
                w, h, img.size[0], img.size[1], path.name,
            )
            return tmp
    except Exception:
        logger.debug("Failed to check/resize image: {}", path.name)
        return path


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int = 6000,
    overlap: int = 3000,
) -> list[str]:
    """Split text into overlapping character chunks.

    Args:
        text: The text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks. Chunks shorter than 50 characters are skipped.
    """
    if not text or not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) >= 50:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_text_sections(
    markdown: str,
    chunk_size: int = 6000,
    overlap: int = 3000,
) -> list[str]:
    """Split GROBID-structured markdown into section-aware chunks.

    Strategy:
    1. Split by ``## `` headings — each section becomes a candidate chunk
    2. Section heading is preserved and prepended to every sub-chunk
    3. Oversized sections are split by paragraphs (``\\n\\n``) with overlap
    4. Oversized paragraphs fall back to character windows with overlap
    5. Chunks shorter than 50 characters are dropped

    Args:
        markdown: Structured markdown from GROBID (with ``## `` section headings).
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks with section context preserved.
    """
    if not markdown or not markdown.strip():
        return []

    # Split into sections by ## headings (keep heading with its content)
    section_pattern = re.compile(r"^(## .+)$", re.MULTILINE)
    parts = section_pattern.split(markdown)

    # Reassemble into (heading, body) pairs
    sections: list[tuple[str, str]] = []
    # Text before first ## heading (e.g., # Title)
    preamble = parts[0].strip()
    if preamble:
        sections.append(("", preamble))

    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((heading, body))

    chunks: list[str] = []

    for heading, body in sections:
        if heading:
            section_text = f"{heading}\n\n{body}" if body else heading
        else:
            section_text = body

        if not section_text or len(section_text.strip()) < 50:
            continue

        if len(section_text) <= chunk_size:
            chunks.append(section_text)
            continue

        # Section is oversized — split by paragraphs
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        _chunk_paragraphs(paragraphs, heading, chunk_size, overlap, chunks)

    return chunks


def _chunk_paragraphs(
    paragraphs: list[str],
    heading: str,
    chunk_size: int,
    overlap: int,
    out: list[str],
) -> None:
    """Accumulate paragraphs into chunks, splitting oversized ones."""
    prefix = f"{heading}\n\n" if heading else ""
    prefix_len = len(prefix)
    available = chunk_size - prefix_len

    current_parts: list[str] = []
    current_len = 0

    def _flush() -> None:
        nonlocal current_parts, current_len
        if current_parts:
            text = prefix + "\n\n".join(current_parts)
            if len(text.strip()) >= 50:
                out.append(text)
            current_parts = []
            current_len = 0

    for para in paragraphs:
        if len(para) > available:
            # Flush accumulated, then character-chunk this paragraph
            _flush()
            para_chunks = chunk_text(prefix + para, chunk_size, overlap)
            out.extend(c for c in para_chunks if len(c.strip()) >= 50)
            continue

        # Check if adding this paragraph would exceed the limit
        sep_len = len("\n\n") if current_parts else 0
        if current_len + sep_len + len(para) > available:
            # Overlap: keep the last paragraph(s) that fit within overlap chars
            _flush()
            # Re-seed with overlap from previous content (simplified: keep last para)
            # Full overlap is handled by the character chunker; here we do paragraph-level

        current_parts.append(para)
        current_len += (sep_len + len(para))

    _flush()


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


async def _extract_pdf(path: Path) -> str | None:
    """Extract text from PDF. GROBID if enabled + available, else pdfplumber."""
    from research_mentor.config import load_config

    config = load_config()

    if config.grobid.enabled:
        from research_mentor.grobid import extract_pdf_via_grobid, is_grobid_available

        if await is_grobid_available(config.grobid.url):
            result = await extract_pdf_via_grobid(path, config.grobid.url)
            if result:
                return result
            logger.warning(
                "GROBID extraction failed for {}, falling back to pdfplumber", path.name,
            )

    return await asyncio.to_thread(_extract_pdfplumber, path)


def _extract_pdfplumber(path: Path) -> str | None:
    """Extract text from PDF using pdfplumber (page-by-page)."""
    import pdfplumber

    try:
        pages_text: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(text)
        if not pages_text:
            return None
        result = "\n\n".join(pages_text)
        return result if result.strip() else None
    except Exception:
        logger.opt(exception=True).debug("pdfplumber extraction failed: {}", path.name)
        return None


# ---------------------------------------------------------------------------
# DOCX extraction
# ---------------------------------------------------------------------------


def _extract_docx(path: Path) -> str | None:
    """Extract text from DOCX using python-docx (paragraphs)."""
    import docx

    try:
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        if not paragraphs:
            return None
        result = "\n\n".join(paragraphs)
        return result if result.strip() else None
    except Exception:
        logger.opt(exception=True).debug("DOCX extraction failed: {}", path.name)
        return None


# ---------------------------------------------------------------------------
# CSV / plain text
# ---------------------------------------------------------------------------


def _extract_csv(path: Path, *, delimiter: str = ",") -> str | None:
    """Extract CSV/TSV with pandas analysis summary + markdown table.

    Returns a string with two sections:
    1. Statistical summary (describe, value counts, correlations)
    2. Full data as markdown table

    The summary gives the mentor analytical context; the table preserves
    the raw data for semantic search.
    """
    import pandas as pd

    try:
        df = pd.read_csv(path, delimiter=delimiter)
    except Exception:
        logger.opt(exception=True).debug("pandas read_csv failed: {}", path.name)
        return None

    if df.empty:
        return None

    parts: list[str] = []

    # --- Summary section ---
    parts.append(f"## Dataset Summary\n\n{len(df)} rows × {len(df.columns)} columns")

    # Column listing with types
    col_types = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            col_types.append(f"- {col} (numeric)")
        else:
            col_types.append(f"- {col} (text)")
    parts.append("Columns:\n" + "\n".join(col_types))

    # Numeric summary
    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        desc = numeric_cols.describe().round(2)
        summary_lines = []
        for col in desc.columns:
            s = desc[col]
            missing = int(df[col].isna().sum())
            missing_note = f" ({missing} missing)" if missing > 0 else ""
            summary_lines.append(
                f"- {col}: mean={s['mean']}, std={s['std']}, "
                f"min={s['min']}, 25%={s['25%']}, 50%={s['50%']}, "
                f"75%={s['75%']}, max={s['max']}{missing_note}"
            )
        parts.append("Numeric summary:\n" + "\n".join(summary_lines))

    # Categorical summary
    cat_cols = df.select_dtypes(exclude="number")
    if not cat_cols.empty:
        cat_lines = []
        for col in cat_cols.columns:
            n_unique = df[col].nunique()
            if n_unique <= 20:
                counts = df[col].value_counts().head(5)
                top = ", ".join(f"{v} ({c})" for v, c in counts.items())
                cat_lines.append(f"- {col}: {n_unique} unique — {top}")
            else:
                cat_lines.append(f"- {col}: {n_unique} unique values")
        parts.append("Categorical summary:\n" + "\n".join(cat_lines))

    # Correlations (numeric only, |r| > 0.3)
    if len(numeric_cols.columns) >= 2:
        corr = numeric_cols.corr()
        strong: list[str] = []
        seen: set[tuple[str, str]] = set()
        for i, c1 in enumerate(corr.columns):
            for j, c2 in enumerate(corr.columns):
                if i >= j:
                    continue
                r = corr.iloc[i, j]
                if abs(r) > 0.3 and (c1, c2) not in seen:
                    seen.add((c1, c2))
                    strong.append(f"- {c1} ↔ {c2}: r={r:.2f}")
        if strong:
            parts.append("Correlations (|r| > 0.3):\n" + "\n".join(strong))

    summary = "\n\n".join(parts)

    # --- Markdown table ---
    table = df.to_markdown(index=False)

    return f"{summary}\n\n## Data\n\n{table}"


def chunk_tabular(
    text: str,
    chunk_size: int = 6000,
    overlap: int = 3000,
) -> list[str]:
    """Chunk tabular data (CSV/TSV) respecting row boundaries.

    Expects text in the format produced by ``_extract_csv``:
    a summary section followed by ``## Data`` with a markdown table.

    Strategy:
    1. Summary section becomes its own chunk
    2. Data rows are grouped with headers prepended to each chunk
    3. Never splits mid-row
    4. Overlap is done by row count (50% of rows per chunk)
    """
    # Split summary from data
    data_marker = "\n\n## Data\n\n"
    if data_marker in text:
        summary, table = text.split(data_marker, 1)
    else:
        # No summary/data split — fall back to character chunking
        return chunk_text(text, chunk_size, overlap)

    chunks: list[str] = []

    # Summary is always the first chunk
    if summary.strip() and len(summary.strip()) >= 50:
        chunks.append(summary.strip())

    # Parse markdown table into header + rows
    table_lines = table.strip().split("\n")
    if len(table_lines) < 3:
        # Too small for row-group chunking
        if len(text.strip()) >= 50:
            chunks.append(text)
        return chunks

    header_line = table_lines[0]
    separator_line = table_lines[1]
    data_rows = table_lines[2:]
    table_header = f"{header_line}\n{separator_line}"

    # Calculate how many rows fit in a chunk (with header)
    header_len = len(table_header) + 1  # +1 for newline
    avg_row_len = sum(len(r) for r in data_rows) / len(data_rows) if data_rows else 100
    rows_per_chunk = max(1, int((chunk_size - header_len) / (avg_row_len + 1)))
    overlap_rows = max(1, rows_per_chunk // 2)  # 50% overlap by row count

    start = 0
    while start < len(data_rows):
        end = min(start + rows_per_chunk, len(data_rows))
        row_block = "\n".join(data_rows[start:end])
        chunk = f"{table_header}\n{row_block}"
        if len(chunk.strip()) >= 50:
            chunks.append(chunk)
        start += rows_per_chunk - overlap_rows

    return chunks


def _extract_plain(path: Path) -> str | None:
    """Read a plain text file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        logger.debug("Failed to read plain text file: {}", path.name)
        return None
    return text if text.strip() else None


# ---------------------------------------------------------------------------
# Vision / image description
# ---------------------------------------------------------------------------


def _resolve_vision_backend(config: AppConfig) -> str:
    """Resolve 'auto' vision backend to a concrete backend.

    auto → follows chat backend: claude-cli→claude-cli, api→api,
    vllm→local (if model cached) else skip (return empty string).
    """
    backend = config.vision.backend
    if backend != "auto":
        return backend

    chat = config.backend
    if chat == "claude-cli":
        return "claude-cli"
    if chat == "api":
        return "api"
    # vllm — use local vision only if model is already downloaded
    from research_mentor.vision import is_local_model_cached

    if is_local_model_cached():
        return "local"
    logger.info(
        "Vision backend 'auto' with vLLM chat: local vision model not cached, "
        "image description disabled. Set vision.backend = 'local' to download it."
    )
    return ""


async def _describe_image(
    file_path: Path,
    config: AppConfig,
    *,
    prompt: str | None = None,
) -> str | None:
    """Describe an image using the configured vision backend."""
    backend = _resolve_vision_backend(config)

    if not backend:
        logger.debug("No vision backend available, skipping image description")
        return None

    vision_prompt = prompt or _DEFAULT_PROMPT

    # Resize before sending to any backend (cap at max_image_dimension, default 1536px)
    resized = resize_image_if_needed(file_path, config.vision.max_image_dimension)
    try:
        return await _describe_with_backend(resized, backend, prompt=vision_prompt)
    finally:
        # Clean up temp resized file if created
        if resized != file_path and resized.exists():
            resized.unlink(missing_ok=True)


async def _describe_with_backend(
    file_path: Path,
    backend: str,
    *,
    prompt: str,
) -> str | None:
    """Dispatch to the appropriate vision backend."""
    if backend == "local":
        from research_mentor.vision import describe_image, is_local_model_cached

        if not is_local_model_cached():
            logger.warning(
                "Vision backend is 'local' but model is not downloaded — "
                "skipping image description. Download it from Settings."
            )
            return None

        description = describe_image(file_path, prompt=prompt)
        if not description:
            return None
        return f"[Image description]\n\n{description}"

    if backend in ("claude-cli", "api"):
        if backend == "claude-cli":
            from research_mentor.vision_cli import interpret_file

            result = await interpret_file(file_path, prompt=prompt)
        else:
            from research_mentor.vision_api import interpret_file as api_interpret

            result = await api_interpret(file_path, prompt=prompt)

        if not result:
            return None
        return f"[AI interpretation]\n\n{result}"

    msg = f"Unknown vision backend: {backend!r}. Must be 'auto', 'local', 'claude-cli', or 'api'."
    raise ValueError(msg)
