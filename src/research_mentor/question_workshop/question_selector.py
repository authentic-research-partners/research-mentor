"""Universal question selection: quality filter + similarity dedup.

Every pipeline workshop generates multiple candidate questions, then calls
``select_questions`` as its final step. This module enforces a uniform
output contract:

- Up to ``max_results`` questions (default 3)
- Each passes a quality threshold
- Each is sufficiently different from higher-scored selections

The module is scoring-agnostic: workshops normalize their own scores to
0-1 before calling ``select_questions``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(slots=True)
class ScoredCandidate:
    """A candidate question with a normalized score and opaque data payload."""

    question_text: str
    """The question string (used for similarity dedup)."""

    overall_score: float
    """Quality score normalized to 0-1.  Each workshop normalizes its own scale."""

    data: dict[str, Any]
    """The full question dict — passed through untouched to the output."""


# ---------------------------------------------------------------------------
# Embedding helpers (reuse the project's sentence-transformers singleton)
# ---------------------------------------------------------------------------

def _embed_questions(texts: list[str]) -> NDArray[np.float32]:
    """Embed question texts and return raw numpy vectors (not sqlite-vec blobs).

    Uses the same model singleton as ``embeddings.py`` but returns the raw
    float32 array so we can compute cosine similarity directly.
    """
    from research_mentor.embeddings import _get_model

    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, batch_size=32)


def _cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine similarity between two L2-normalized vectors (= dot product)."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_questions(
    candidates: list[ScoredCandidate],
    *,
    quality_threshold: float = 0.6,
    similarity_threshold: float = 0.85,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Filter by quality, deduplicate by semantic similarity, return up to *max_results*.

    Algorithm:
    1. Sort by ``overall_score`` descending.
    2. Drop candidates below ``quality_threshold``.
    3. For each remaining candidate (top-down), skip if cosine similarity
       to any already-selected question exceeds ``similarity_threshold``.
    4. Return up to ``max_results``.
    5. If nothing passes the threshold, return the single best candidate
       (always return at least one).

    Parameters
    ----------
    candidates:
        Scored candidates from a pipeline's scoring stage.
    quality_threshold:
        Minimum normalized score (0-1) to be eligible.
    similarity_threshold:
        Maximum cosine similarity allowed between any two selected questions.
        Higher = more permissive (closer questions allowed).
    max_results:
        Maximum number of questions to return.

    Returns
    -------
    list[dict[str, Any]]
        Selected question dicts (the ``data`` field of each candidate),
        ordered by score descending.
    """
    if not candidates:
        return []

    # 1. Sort by score descending
    ranked = sorted(candidates, key=lambda c: c.overall_score, reverse=True)

    # 2. Quality filter
    passing = [c for c in ranked if c.overall_score >= quality_threshold]

    # 5. Guarantee: always return at least one
    if not passing:
        logger.info(
            "No candidates passed quality threshold {:.2f} (best: {:.2f}), returning top 1",
            quality_threshold,
            ranked[0].overall_score,
        )
        return [ranked[0].data]

    # Short-circuit: if only 1 passes, return it
    if len(passing) == 1:
        return [passing[0].data]

    # 3. Embed all passing question texts for similarity comparison
    texts = [c.question_text for c in passing]
    embeddings = _embed_questions(texts)

    selected: list[dict[str, Any]] = []
    selected_indices: list[int] = []

    for i, candidate in enumerate(passing):
        if len(selected) >= max_results:
            break

        # Check similarity against all already-selected questions
        too_similar = False
        for j in selected_indices:
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim > similarity_threshold:
                logger.debug(
                    "Skipping question (sim={:.3f} > {:.2f}): {}",
                    sim,
                    similarity_threshold,
                    candidate.question_text[:80],
                )
                too_similar = True
                break

        if not too_similar:
            selected.append(candidate.data)
            selected_indices.append(i)

    # Safety: if dedup removed everything (all questions nearly identical),
    # return the top one
    if not selected:
        return [passing[0].data]

    return selected
