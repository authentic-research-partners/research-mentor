"""Local filesystem storage for artifact files.

Files are stored under ``~/.research-mentor/artifacts/<project>/<artifact>/<file>``.
This mirrors the hosted GCS layout but on the local filesystem.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from research_mentor.config import load_config


def _artifacts_root() -> Path:
    """Return the resolved artifacts root directory (creates if needed)."""
    return load_config().storage.get_artifacts_path()


def artifact_dir(project_id: str, artifact_id: str) -> Path:
    """Return the directory for a specific artifact (creates if needed)."""
    d = _artifacts_root() / project_id / artifact_id
    d.mkdir(parents=True, exist_ok=True)
    return d


async def save_artifact_file(
    project_id: str,
    artifact_id: str,
    file_name: str,
    data: bytes,
) -> Path:
    """Save artifact file to local filesystem. Returns the absolute path."""
    safe_name = Path(file_name).name  # strip any directory components
    dest = artifact_dir(project_id, artifact_id) / safe_name
    await asyncio.to_thread(dest.write_bytes, data)
    logger.info(
        "Saved artifact file: {} ({} bytes)", dest, len(data),
    )
    return dest


def get_artifact_path(project_id: str, artifact_id: str, file_name: str) -> Path | None:
    """Return the path to an artifact file, or None if it doesn't exist."""
    safe_name = Path(file_name).name
    p = artifact_dir(project_id, artifact_id) / safe_name
    return p if p.is_file() else None


def delete_artifact_files(project_id: str, artifact_id: str) -> None:
    """Delete all files for an artifact and clean up empty directories."""
    d = _artifacts_root() / project_id / artifact_id
    if not d.exists():
        return
    for f in d.iterdir():
        f.unlink()
        logger.debug("Deleted artifact file: {}", f)
    d.rmdir()
    # Clean up project dir if empty
    project_dir = d.parent
    if project_dir.exists() and not any(project_dir.iterdir()):
        project_dir.rmdir()
