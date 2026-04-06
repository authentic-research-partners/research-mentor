"""Config accessor for the Questioned workshop."""

from __future__ import annotations

from research_mentor.config import QuestionedConfig, load_config


def questioned_config() -> QuestionedConfig:
    """Load Questioned config from TOML (cached at config level)."""
    return load_config().question_workshop.questioned
