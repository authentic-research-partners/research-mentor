"""Claims config accessor — avoids circular imports."""

from __future__ import annotations

from research_mentor.config import ClaimsConfig, load_config


def claims_config() -> ClaimsConfig:
    """Load Claims config from TOML (cached at config level)."""
    return load_config().question_workshop.claims
