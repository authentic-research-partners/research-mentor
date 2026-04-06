"""Questioned workshop — surfaces papers under scrutiny.

Non-interactive pipeline: student provides a field, system surfaces
published papers with expressions of concern, corrections, or critiques.
Teaches scientific skepticism: published ≠ true.
"""

from research_mentor.question_workshop.questioned.orchestrator import (
    surface_scrutinized_papers,
)

__all__ = ["surface_scrutinized_papers"]
