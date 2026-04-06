"""Artifacts API — nested under projects.

Provides metadata CRUD plus file upload/download via local filesystem storage.
Uploaded files are automatically text-extracted, chunked, and embedded for
semantic search (if embeddings are enabled).
"""

from __future__ import annotations

import asyncio
import mimetypes
import shutil
from pathlib import PurePath
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from loguru import logger

from research_mentor.api.schemas import (
    ArtifactCreate,
    ArtifactResponse,
    ArtifactUpdate,
    PaginatedArtifacts,
    _pagination_info,
)
from research_mentor.config import load_config
from research_mentor.db import crud
from research_mentor.storage import (
    delete_artifact_files,
    get_artifact_path,
    save_artifact_file,
)

router = APIRouter(prefix="/api/projects/{project_id}/artifacts", tags=["artifacts"])

_ALLOWED_EXTENSIONS = frozenset({
    ".pdf", ".docx",
    ".csv", ".tsv",
    ".txt", ".md", ".markdown", ".rst",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg",
})


async def _extract_and_embed(
    artifact_id: str,
    file_path: str,
    research_context: str | None,
) -> dict[str, Any] | None:
    """Extract text (and images) from an artifact, chunk, and embed.

    For PDFs and DOCX, also extracts embedded images via Docling and sends
    each to the vision backend for description. Both text and image
    descriptions are chunked and embedded for semantic search.

    Returns the re-fetched artifact row if extraction happened, or None if
    embeddings are disabled or extraction produced no text.
    """
    config = load_config()
    if not config.embeddings.enabled:
        return None

    from research_mentor.embeddings import embed_texts
    from research_mentor.text_extraction import (
        chunk_text,
        extract_images,
        extract_text_or_describe,
    )

    text = await extract_text_or_describe(
        file_path, config, research_context=research_context,
    )

    total_chunks = 0

    if text:
        await crud.update_artifact(artifact_id, extracted_text=text)
        chunks = chunk_text(
            text,
            chunk_size=config.embeddings.artifact_chunk_size,
            overlap=config.embeddings.artifact_chunk_overlap,
        )
        if chunks:
            embeddings = await embed_texts(chunks)
            total_chunks += await crud.store_artifact_chunks(
                artifact_id, chunks, embeddings,
            )

    # Extract and describe embedded images (PDFs and DOCX only)
    if config.vision.enabled:
        image_paths = await asyncio.to_thread(extract_images, file_path)
        try:
            for img_path in image_paths:
                description = await extract_text_or_describe(
                    img_path, config, research_context=research_context,
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
        return None

    refetched = await crud.get_artifact(artifact_id)
    assert refetched is not None
    return refetched


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
    artifact_type: str = Form(default="document"),
    description: str | None = Form(default=None),
) -> ArtifactResponse:
    """Upload a file and create an artifact record.

    Accepts multipart/form-data with:
    - ``file``: the file to upload (required)
    - ``artifact_type``: type classification (default: "document")
    - ``description``: optional description
    """
    project = await _require_project(project_id)
    research_context: str | None = project.get("research_question") or None

    file_name = PurePath(file.filename or "unnamed").name  # strip directory components
    suffix = PurePath(file_name).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
        raise HTTPException(
            415,
            f"Unsupported file type '{suffix}'. Allowed: {allowed}",
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
        artifact_type=artifact_type,
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
    try:
        refetched = await _extract_and_embed(
            artifact["id"], str(dest), research_context,
        )
        if refetched is not None:
            final = refetched
    except Exception:
        logger.opt(exception=True).warning(
            "Text extraction/embedding failed for artifact {}", artifact["id"],
        )

    return ArtifactResponse(**final)


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

    refetched = await _extract_and_embed(
        artifact_id, str(path), research_context,
    )
    if refetched is not None:
        return ArtifactResponse(**refetched)

    # Extraction produced no text (or embeddings disabled) — clear old text, return as-is
    await crud.update_artifact(artifact_id, extracted_text=None)
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
