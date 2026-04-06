"""Venue matching — filter and score curated venues.

Pure Python logic (no LLM calls) for the first two steps of venue matching:
1. Filter by hard constraints (age, field, level, format, exclude ISEF)
2. Score by soft factors (cost, quality tier, format preference)

The third step (LLM reasoning) happens in the venue_discovery node.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from importlib import resources
from typing import Any

from loguru import logger


@lru_cache(maxsize=1)
def load_curated_venues() -> list[dict[str, Any]]:
    """Load curated venues from the bundled TOML file.

    Returns:
        List of venue dicts with all attributes from curated_venues.toml.
    """
    try:
        seed_data = resources.files("research_mentor") / "seed_data" / "curated_venues.toml"
        raw = seed_data.read_text(encoding="utf-8")
    except Exception:
        logger.warning("curated_venues.toml: importlib.resources failed, trying file path")
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "seed_data" / "curated_venues.toml"
        raw = path.read_text(encoding="utf-8")

    data = tomllib.loads(raw)
    venues: list[dict[str, Any]] = data.get("venues", [])
    logger.info("Loaded {} curated venues", len(venues))
    return venues


def filter_venues(
    venues: list[dict[str, Any]],
    *,
    age: int | None = None,
    education_level: str | None = None,
    field: str | None = None,
    format_type: str | None = None,
    exclude_isef: bool = True,
    active_only: bool = True,
) -> list[dict[str, Any]]:
    """Filter venues by hard constraints.

    Args:
        venues: List of venue dicts (from load_curated_venues or API).
        age: Student's age. Venues outside age_range are excluded.
        education_level: 'middle_school', 'high_school', 'university'.
        field: 'natural_science' or 'social_science'.
        format_type: 'paper', 'poster', 'oral', 'video', etc.
        exclude_isef: Always True per philosophy doc.
        active_only: Only include active venues.

    Returns:
        Filtered list of venues matching all hard constraints.
    """
    filtered: list[dict[str, Any]] = []

    for venue in venues:
        # Active filter
        if active_only and not venue.get("active", True):
            continue

        # ISEF exclusion (always on)
        if exclude_isef and venue.get("is_isef_affiliated", False):
            continue

        # Age range filter
        if age is not None:
            age_range = venue.get("age_range", [0, 99])
            if len(age_range) >= 2 and not (age_range[0] <= age <= age_range[1]):
                continue

        # Education level filter
        if education_level is not None:
            venue_levels = venue.get("education_level", ["any"])
            if "any" not in venue_levels and education_level not in venue_levels:
                continue

        # Field filter
        if field is not None:
            venue_fields = venue.get("fields", ["all"])
            if "all" not in venue_fields and field not in venue_fields:
                continue

        # NOTE: format_type is NOT a hard filter. A student wanting a poster
        # should still see journals like JEI — format mismatch is penalized
        # in score_venues() via format_preference, not excluded here.

        filtered.append(venue)

    return filtered


def score_venues(
    venues: list[dict[str, Any]],
    *,
    cost_preference: str = "any",
    quality_target: str | None = None,
    format_preference: str | None = None,
) -> list[dict[str, Any]]:
    """Score and sort venues by soft factors.

    Higher score = better match. Scores are for relative ordering,
    not absolute quality judgment.

    Args:
        venues: Pre-filtered venue list.
        cost_preference: 'free', 'low', 'any'.
        quality_target: 'selective', 'moderate', 'open', or None.
        format_preference: Preferred output format, or None.

    Returns:
        Venues sorted by descending match score, with 'match_score' added.
    """
    scored: list[dict[str, Any]] = []

    for venue in venues:
        score = 50.0  # base score

        # Cost scoring
        venue_cost = venue.get("cost", "free")
        if cost_preference == "free":
            if venue_cost == "free":
                score += 20
            elif venue_cost == "low":
                score += 5
        elif cost_preference == "low":
            if venue_cost in ("free", "low"):
                score += 15

        # Quality tier scoring
        venue_tier = venue.get("quality_tier", "moderate")
        if quality_target:
            if venue_tier == quality_target:
                score += 15
            elif quality_target == "selective" and venue_tier == "moderate":
                score += 5
        else:
            # Default: prefer selective > moderate > open
            if venue_tier == "selective":
                score += 10
            elif venue_tier == "moderate":
                score += 5

        # Format match bonus
        if format_preference:
            venue_format = venue.get("format", "")
            if venue_format == format_preference:
                score += 10
            elif venue_format == "multiple":
                score += 5

        # Vanity press penalty
        if venue.get("vanity_press_risk", False):
            score -= 30

        # Mentor requirement (slight penalty if requires mentor — more barrier)
        if venue.get("requires_mentor", False):
            score -= 5

        entry = dict(venue)
        entry["match_score"] = score
        scored.append(entry)

    scored.sort(key=lambda v: v["match_score"], reverse=True)
    return scored
