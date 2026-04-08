"""Artifacts API — nested under projects.

Provides metadata CRUD plus file upload/download via local filesystem storage.
Uploaded files are automatically text-extracted, chunked, and embedded for
semantic search (if embeddings are enabled).

See docs/artifact-processing.md for the full pipeline design.
"""

from __future__ import annotations

import asyncio
import mimetypes
import shutil
from pathlib import Path, PurePath
from typing import Any

import pandas as pd
from fastapi import APIRouter, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from loguru import logger

from research_mentor.api.schemas import (
    ArtifactCreate,
    ArtifactResponse,
    ArtifactType,
    ArtifactUpdate,
    PaginatedArtifacts,
    _pagination_info,
)
from research_mentor.artifact_types import ALLOWED_FORMATS, ARTIFACT_TYPES
from research_mentor.config import load_config
from research_mentor.db import crud
from research_mentor.storage import (
    delete_artifact_files,
    get_artifact_path,
    save_artifact_file,
)

router = APIRouter(prefix="/api/projects/{project_id}/artifacts", tags=["artifacts"])

# Separate router for non-project-scoped artifact endpoints
types_router = APIRouter(tags=["artifacts"])


@types_router.get("/api/artifact-types")
async def get_artifact_types() -> list[dict[str, str]]:
    """Return artifact type definitions for the UI (single source of truth)."""
    return [
        {
            "value": t.value,
            "label": t.label,
            "description": t.description,
            "accept": t.accept,
            "placeholder": t.placeholder,
        }
        for t in ARTIFACT_TYPES
    ]


def _read_dataframe(file_path: str) -> pd.DataFrame | None:
    """Read a CSV/TSV file into a pandas DataFrame, or None on failure."""
    path = Path(file_path)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        return pd.read_csv(path, delimiter=delimiter)
    except Exception:
        logger.debug("Failed to read dataframe from {}", path.name)
        return None


async def _extract_and_embed(
    artifact_id: str,
    file_path: str,
    research_context: str | None,
    artifact_type: str | None = None,
    artifact_description: str | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Extract text and/or analysis from an artifact, chunk, and embed.

    Behavior depends on artifact type:
    - Vision types (photograph, scientific_image, chart, handwriting):
      AI description → stored in ``analysis``, embedded as one chunk
    - Data (CSV/TSV): pandas summary → ``analysis``, raw table → ``extracted_text``,
      summary + row-group chunks embedded
    - Text types (paper, document): raw text → ``extracted_text``,
      section-aware or character chunks embedded

    Returns (artifact_row_or_None, warnings_list).
    """
    warnings: list[str] = []
    config = load_config()
    if not config.embeddings.enabled:
        return None, warnings

    from research_mentor.embeddings import embed_texts
    from research_mentor.text_extraction import (
        chunk_text,
        chunk_text_sections,
        extract_images,
        extract_text_or_describe,
    )

    _VISION_TYPES = {"photograph", "scientific_image", "chart", "handwriting"}

    text = await extract_text_or_describe(
        file_path, config,
        research_context=research_context,
        artifact_type=artifact_type,
        artifact_description=artifact_description,
    )

    total_chunks = 0
    chunk_size = config.embeddings.artifact_chunk_size
    chunk_overlap = config.embeddings.artifact_chunk_overlap

    if text and artifact_type in _VISION_TYPES:
        # Vision types: AI description → analysis column
        await crud.update_artifact(artifact_id, analysis=text)

        # Embed description + analysis as one chunk for retrieval
        combined = text
        if artifact_description:
            combined = f"Student description: {artifact_description}\n\n{text}"
        if len(combined.strip()) >= 50:
            embeddings = await embed_texts([combined])
            total_chunks += await crud.store_artifact_chunks(
                artifact_id, [combined], embeddings,
            )

    elif text and artifact_type == "data":
        # Data: split summary from raw table
        data_marker = "\n\n## Data\n\n"
        if data_marker in text:
            summary, raw_table = text.split(data_marker, 1)
        else:
            summary = None
            raw_table = None

        # Run statistical analysis (best-effort — don't fail extraction)
        stats_text: str | None = None
        try:
            from research_mentor.data_analysis import analyze_dataframe

            profile = await crud.get_student_profile()
            df = _read_dataframe(file_path)
            if df is not None and not df.empty:
                stats_text, _results = await analyze_dataframe(
                    df,
                    description=artifact_description,
                    research_question=research_context,
                    student_profile=profile,
                )
        except Exception:
            logger.opt(exception=True).warning(
                "Statistical analysis failed for artifact {}", artifact_id,
            )

        # Build combined analysis: description + pandas summary + stats
        analysis_parts: list[str] = []
        if summary:
            analysis_parts.append(summary)
        if stats_text:
            analysis_parts.append(stats_text)
        combined_analysis = "\n\n".join(analysis_parts) if analysis_parts else None

        # Store: analysis column = combined, extracted_text = raw table
        await crud.update_artifact(
            artifact_id,
            analysis=combined_analysis,
            extracted_text=raw_table,
        )

        # Embed combined analysis chunk (description + summary + stats)
        # for rich semantic search by the mentor
        if combined_analysis:
            analysis_chunk = combined_analysis
            if artifact_description:
                analysis_chunk = (
                    f"Student description: {artifact_description}\n\n"
                    + analysis_chunk
                )
            analysis_embeddings = await embed_texts([analysis_chunk])
            total_chunks += await crud.store_artifact_chunks(
                artifact_id, [analysis_chunk], analysis_embeddings,
            )

    elif text:
        # Text types (paper, document): raw text → extracted_text
        await crud.update_artifact(artifact_id, extracted_text=text)

        if text.startswith("# ") and "\n## " in text:
            chunks = chunk_text_sections(text, chunk_size, chunk_overlap)
        else:
            chunks = chunk_text(text, chunk_size, chunk_overlap)

        # Prepend student description to first chunk so semantic search
        # matches the student's intent (e.g., "focus on Methods section")
        if chunks and artifact_description:
            chunks[0] = (
                f"Student description: {artifact_description}\n\n"
                + chunks[0]
            )

        if chunks:
            embeddings = await embed_texts(chunks)
            total_chunks += await crud.store_artifact_chunks(
                artifact_id, chunks, embeddings,
            )

    # Extract and describe raster images from PDFs
    if config.vision.enabled:
        image_paths = await asyncio.to_thread(extract_images, file_path)
        max_images = config.vision.max_images_per_artifact
        if len(image_paths) > max_images:
            skipped = len(image_paths) - max_images
            image_paths = image_paths[:max_images]
            warnings.append(
                f"This file contains {max_images + skipped} images. "
                f"Only the first {max_images} were processed — "
                f"{skipped} were skipped to keep processing time reasonable."
            )
        try:
            for img_path in image_paths:
                description = await extract_text_or_describe(
                    img_path, config,
                    research_context=research_context,
                    artifact_type="paper_figure",
                    artifact_description=artifact_description,
                )
                if description:
                    img_chunks = chunk_text(
                        description,
                        chunk_size=config.embeddings.artifact_chunk_size,
                        overlap=config.embeddings.artifact_chunk_overlap,
                    )
                    if img_chunks:
                        img_embeddings = await embed_texts(img_chunks)
                        total_chunks += await crud.store_artifact_chunks(
                            artifact_id, img_chunks, img_embeddings,
                        )
        finally:
            # Clean up temp image files
            if image_paths:
                tmp_dir = image_paths[0].parent
                shutil.rmtree(tmp_dir, ignore_errors=True)

    if total_chunks > 0:
        logger.info("Embedded artifact {} ({} chunks)", artifact_id, total_chunks)

    if not text and total_chunks == 0:
        return None, warnings

    refetched = await crud.get_artifact(artifact_id)
    assert refetched is not None
    return refetched, warnings


async def _require_project(project_id: str) -> dict[str, Any]:
    project = await crud.get_project(project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


@router.post("", response_model=ArtifactResponse, status_code=201)
async def create_artifact(project_id: str, body: ArtifactCreate) -> ArtifactResponse:
    await _require_project(project_id)
    return ArtifactResponse(**await crud.create_artifact(project_id, **body.model_dump()))


@router.post("/upload", response_model=ArtifactResponse, status_code=201)
async def upload_artifact(
    project_id: str,
    file: UploadFile,
    artifact_type: ArtifactType = Form(...),
    description: str = Form(...),
) -> ArtifactResponse:
    """Upload a file and create an artifact record.

    Accepts multipart/form-data with:
    - ``file``: the file to upload (required)
    - ``artifact_type``: one of the 7 artifact types (required)
    - ``description``: what this artifact is and why you're uploading it (required)
    """
    project = await _require_project(project_id)
    research_context: str | None = project.get("research_question") or None

    file_name = PurePath(file.filename or "unnamed").name  # strip directory components
    suffix = PurePath(file_name).suffix.lower()

    # Validate format for this artifact type
    allowed = ALLOWED_FORMATS[artifact_type.value]
    if suffix not in allowed:
        allowed_str = ", ".join(sorted(allowed))
        raise HTTPException(
            415,
            f"File type '{suffix}' is not allowed for artifact type '{artifact_type.value}'. "
            f"Allowed: {allowed_str}",
        )

    config = load_config()
    max_bytes = config.storage.max_upload_bytes

    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(
            413,
            f"File too large: {len(data)} bytes (max {max_bytes})",
        )

    mime_type = (
        file.content_type or mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    )

    # Create DB record first to get the artifact ID
    artifact = await crud.create_artifact(
        project_id,
        artifact_type=artifact_type.value,
        file_name=file_name,
        file_path="",  # placeholder, updated after save
        mime_type=mime_type,
        file_size_bytes=len(data),
        description=description,
    )

    # Save file to disk
    dest = await save_artifact_file(project_id, artifact["id"], file_name, data)

    # Update record with actual path
    result = await crud.update_artifact(artifact["id"], file_path=str(dest))
    assert result is not None
    final: dict[str, Any] = result

    # Extract text, chunk, and embed (best-effort — don't fail the upload)
    warnings: list[str] = []
    try:
        refetched, embed_warnings = await _extract_and_embed(
            artifact["id"], str(dest), research_context,
            artifact_type=artifact_type.value,
            artifact_description=description,
        )
        warnings.extend(embed_warnings)
        if refetched is not None:
            final = refetched
    except Exception:
        logger.opt(exception=True).warning(
            "Text extraction/embedding failed for artifact {}", artifact["id"],
        )

    # Warn if vision is enabled with local backend but model not downloaded
    if config.vision.enabled and config.vision.backend == "local":
        from research_mentor.vision import is_local_model_cached

        if not is_local_model_cached():
            warnings.append(
                "Image interpretation is not available — the local vision model "
                "is not downloaded. Go to Settings → Vision to download it, "
                "or switch to a different backend."
            )

    return ArtifactResponse(**final, warnings=warnings)


@router.get("", response_model=PaginatedArtifacts)
async def list_artifacts(
    project_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
) -> PaginatedArtifacts:
    await _require_project(project_id)
    result = await crud.list_artifacts(project_id, page=page, page_size=page_size)
    return PaginatedArtifacts(
        data=[ArtifactResponse(**a) for a in result["data"]],
        pagination=_pagination_info(page, page_size, result["total"]),
    )


@router.get("/{artifact_id}", response_model=ArtifactResponse)
async def get_artifact(project_id: str, artifact_id: str) -> ArtifactResponse:
    await _require_project(project_id)
    artifact = await crud.get_artifact(artifact_id)
    if artifact is None or artifact["project_id"] != project_id:
        raise HTTPException(404, "Artifact not found")
    return ArtifactResponse(**artifact)


@router.get("/{artifact_id}/download")
async def download_artifact(project_id: str, artifact_id: str) -> FileResponse:
    """Download the artifact file."""
    await _require_project(project_id)
    artifact = await crud.get_artifact(artifact_id)
    if artifact is None or artifact["project_id"] != project_id:
        raise HTTPException(404, "Artifact not found")

    path = get_artifact_path(project_id, artifact_id, artifact["file_name"])
    if path is None:
        raise HTTPException(404, "Artifact file not found on disk")

    return FileResponse(
        path=path,
        filename=artifact["file_name"],
        media_type=artifact["mime_type"],
    )


@router.patch("/{artifact_id}", response_model=ArtifactResponse)
async def update_artifact(
    project_id: str, artifact_id: str, body: ArtifactUpdate,
) -> ArtifactResponse:
    await _require_project(project_id)
    existing = await crud.get_artifact(artifact_id)
    if existing is None or existing["project_id"] != project_id:
        raise HTTPException(404, "Artifact not found")
    fields = body.model_dump(exclude_unset=True)
    result = await crud.update_artifact(artifact_id, **fields)
    if result is None:
        raise HTTPException(404, "Artifact not found")
    return ArtifactResponse(**result)


@router.post("/{artifact_id}/reinterpret", response_model=ArtifactResponse)
async def reinterpret_artifact(
    project_id: str, artifact_id: str,
) -> ArtifactResponse:
    """Re-extract text from an artifact using the currently configured backend.

    Deletes old embedding chunks and re-embeds the new extracted text.
    """
    project = await _require_project(project_id)
    artifact = await crud.get_artifact(artifact_id)
    if artifact is None or artifact["project_id"] != project_id:
        raise HTTPException(404, "Artifact not found")

    path = get_artifact_path(project_id, artifact_id, artifact["file_name"])
    if path is None:
        raise HTTPException(404, "Artifact file not found on disk")

    research_context: str | None = project.get("research_question") or None

    # Clear old chunks before re-embedding
    await crud.delete_artifact_chunks(artifact_id)

    refetched, embed_warnings = await _extract_and_embed(
        artifact_id, str(path), research_context,
        artifact_type=artifact.get("artifact_type"),
        artifact_description=artifact.get("description"),
    )
    if refetched is not None:
        refetched.pop("warnings", None)
        return ArtifactResponse(**refetched, warnings=embed_warnings)

    # Extraction produced no text (or embeddings disabled) — clear old text, return as-is
    await crud.update_artifact(artifact_id, extracted_text=None)
    updated = await crud.get_artifact(artifact_id)
    assert updated is not None
    return ArtifactResponse(**updated)


@router.post("/{artifact_id}/analyze", response_model=ArtifactResponse)
async def analyze_artifact(
    project_id: str, artifact_id: str,
) -> ArtifactResponse:
    """Run (or re-run) statistical analysis on a data artifact.

    Only works for CSV/TSV artifacts. Re-runs LLM test selection + scipy
    execution, useful after updating the artifact description.
    """
    project = await _require_project(project_id)
    artifact = await crud.get_artifact(artifact_id)
    if artifact is None or artifact["project_id"] != project_id:
        raise HTTPException(404, "Artifact not found")

    if artifact.get("artifact_type") != "data":
        raise HTTPException(
            400, "Statistical analysis is only available for data artifacts (CSV/TSV)",
        )

    path = get_artifact_path(project_id, artifact_id, artifact["file_name"])
    if path is None:
        raise HTTPException(404, "Artifact file not found on disk")

    from research_mentor.data_analysis import analyze_dataframe

    df = _read_dataframe(str(path))
    if df is None or df.empty:
        raise HTTPException(400, "Could not read data from file")

    profile = await crud.get_student_profile()
    research_context: str | None = project.get("research_question") or None
    analysis_text, _results = await analyze_dataframe(
        df,
        description=artifact.get("description"),
        research_question=research_context,
        student_profile=profile,
    )

    # Preserve pandas summary, replace statistical analysis section
    current_analysis = artifact.get("analysis") or ""
    stats_header = "## Statistical Analysis"
    if stats_header in current_analysis:
        # Replace existing stats section — split on first occurrence
        pandas_part = current_analysis.split(stats_header)[0].rstrip()
    else:
        pandas_part = current_analysis.rstrip()

    combined = f"{pandas_part}\n\n{analysis_text}" if pandas_part else analysis_text
    await crud.update_artifact(artifact_id, analysis=combined)

    # Re-embed the stats chunk
    if analysis_text:
        config = load_config()
        if config.embeddings.enabled:
            from research_mentor.embeddings import embed_texts

            stats_embeddings = await embed_texts([analysis_text])
            await crud.store_artifact_chunks(
                artifact_id, [analysis_text], stats_embeddings,
            )

    updated = await crud.get_artifact(artifact_id)
    assert updated is not None
    return ArtifactResponse(**updated)


@router.delete("/{artifact_id}", status_code=204)
async def delete_artifact(project_id: str, artifact_id: str) -> None:
    await _require_project(project_id)
    existing = await crud.get_artifact(artifact_id)
    if existing is None or existing["project_id"] != project_id:
        raise HTTPException(404, "Artifact not found")
    # Delete embeddings, file from disk, then DB record
    await crud.delete_artifact_chunks(artifact_id)
    delete_artifact_files(project_id, artifact_id)
    await crud.delete_artifact(artifact_id)
