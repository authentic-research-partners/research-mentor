"""Text extraction and chunking for artifact files.

Dispatches by file type:
- PDF → Docling (layout-aware), with optional GROBID upgrade for academic papers
- DOCX → Docling (layout-aware)
- CSV/TSV → stdlib csv (rendered as markdown table)
- TXT/MD/RST → plain read
- Images → vision backends (Qwen3-VL, Claude CLI, or API)

Docling also extracts embedded images from PDFs/DOCX, which are sent to
vision backends for description and embedded separately.
"""

from __future__ import annotations

import asyncio
import csv
import io
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from research_mentor.config import AppConfig

_IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg",
})

_PDF_EXTENSIONS = frozenset({".pdf"})
_DOCX_EXTENSIONS = frozenset({".docx"})
_DOCLING_EXTENSIONS = _PDF_EXTENSIONS | _DOCX_EXTENSIONS
_TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".rst"})
_CSV_EXTENSIONS = frozenset({".csv", ".tsv"})


async def extract_text(file_path: str | Path) -> str | None:
    """Extract text from a file, dispatching by file type.

    Supported formats:
    - PDF → GROBID (if enabled + available) or Docling
    - DOCX → Docling
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
        return await asyncio.to_thread(_extract_docling, path)
    if suffix in _CSV_EXTENSIONS:
        return _extract_csv(path, delimiter="\t" if suffix == ".tsv" else ",")
    if suffix in _IMAGE_EXTENSIONS:
        return None  # images need vision backends, not text extraction
    if suffix in _TEXT_EXTENSIONS:
        return _extract_plain(path)

    # Unknown format — try plain read as last resort
    return _extract_plain(path)


async def _extract_pdf(path: Path) -> str | None:
    """Extract text from PDF. GROBID if enabled + available, else Docling."""
    from research_mentor.config import load_config

    config = load_config()

    if config.grobid.enabled:
        from research_mentor.grobid import extract_pdf_via_grobid, is_grobid_available

        if await is_grobid_available(config.grobid.url):
            result = await extract_pdf_via_grobid(path, config.grobid.url)
            if result:
                return result
            logger.warning(
                "GROBID extraction failed for {}, falling back to Docling", path.name,
            )

    return await asyncio.to_thread(_extract_docling, path)


def _extract_docling(path: Path) -> str | None:
    """Extract text from PDF or DOCX using Docling (layout-aware, CPU-only)."""
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc.base import ImageRefMode

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)},
    )
    result = converter.convert(str(path))
    md = result.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)
    return md if md and md.strip() else None


def extract_images(file_path: str | Path) -> list[Path]:
    """Extract embedded images from a PDF or DOCX via Docling.

    Saves each image to a temporary directory and returns their paths.
    The caller is responsible for cleanup.

    Returns an empty list for non-PDF/DOCX files or if no images are found.
    """
    path = Path(file_path)
    if path.suffix.lower() not in _DOCLING_EXTENSIONS:
        return []

    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc.document import PictureItem

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)},
    )
    result = converter.convert(str(path))
    doc = result.document

    images: list[Path] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="rm_images_"))

    i = 0
    for item, _level in doc.iterate_items(traverse_pictures=True):
        if not isinstance(item, PictureItem):
            continue
        img = item.get_image(doc)
        if img is None:
            continue
        img_path = tmp_dir / f"image_{i}.png"
        img.save(str(img_path))
        images.append(img_path)
        i += 1

    logger.debug("Extracted {} images from {}", len(images), path.name)
    return images


def _extract_csv(path: Path, *, delimiter: str = ",") -> str | None:
    """Extract CSV/TSV as a markdown table."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return None

    reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return None

    # Build markdown table
    header = rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        # Pad or trim row to match header length
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[:len(header)]) + " |")

    return "\n".join(lines)


def _extract_plain(path: Path) -> str | None:
    """Read a plain text file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        logger.debug("Failed to read plain text file: {}", path.name)
        return None
    return text if text.strip() else None


def chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200,
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


async def extract_text_or_describe(
    file_path: str | Path,
    config: AppConfig,
    *,
    research_context: str | None = None,
) -> str | None:
    """Extract text from a file, with vision backend for images.

    For PDFs and DOCX: uses Docling/GROBID for text extraction (via
    ``extract_text``). Embedded images are handled separately by the
    artifact pipeline.

    For standalone images: dispatches to the configured vision backend.

    For other formats: uses text extraction directly.

    Args:
        file_path: Path to the file.
        config: Application configuration.
        research_context: Optional research question for vision context.

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
        return await _describe_image(path, config, research_context=research_context)

    # Everything else → text extraction (Docling for PDF/DOCX, csv, plain)
    return await extract_text(file_path)


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
    research_context: str | None = None,
) -> str | None:
    """Describe an image using the configured vision backend."""
    backend = _resolve_vision_backend(config)

    if not backend:
        logger.debug("No vision backend available, skipping image description")
        return None

    if backend == "local":
        from research_mentor.vision import describe_image, is_local_model_cached

        if not is_local_model_cached():
            logger.warning(
                "Vision backend is 'local' but model is not downloaded — "
                "skipping image description. Download it from Settings."
            )
            return None

        description = describe_image(file_path)
        if not description:
            return None
        return f"[Image description]\n\n{description}"

    if backend in ("claude-cli", "api"):
        return await _interpret_with_backend(
            file_path, backend, research_context=research_context,
        )

    msg = f"Unknown vision backend: {backend!r}. Must be 'auto', 'local', 'claude-cli', or 'api'."
    raise ValueError(msg)


async def _interpret_with_backend(
    file_path: Path,
    backend: str,
    *,
    research_context: str | None = None,
) -> str | None:
    """Dispatch file interpretation to the specified SOTA backend."""
    if backend == "claude-cli":
        from research_mentor.vision_cli import interpret_file

        result = await interpret_file(file_path, research_context=research_context)
    else:
        from research_mentor.vision_api import interpret_file as api_interpret

        result = await api_interpret(file_path, research_context=research_context)

    if not result:
        return None
    return f"[AI interpretation]\n\n{result}"
