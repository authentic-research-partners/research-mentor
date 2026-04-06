"""Question Workshop dispatcher — routes to the appropriate workshop type.

Backward-compatible: ``generate_research_question`` and ``generate_best_of_n``
are re-exported from the phenomenon sub-module so existing imports keep working.
"""

from __future__ import annotations

# Re-export phenomenon functions for backward compatibility.
# Callers can keep importing from ``research_mentor.question_workshop.orchestrator``.
from research_mentor.question_workshop.phenomenon.orchestrator import (  # noqa: F401
    generate_best_of_n,
    generate_research_question,
)

__all__ = ["generate_research_question", "generate_best_of_n"]
