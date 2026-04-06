"""Phenomenon question workshop — generates original research questions from scratch."""

from research_mentor.question_workshop.phenomenon.orchestrator import (
    generate_best_of_n,
    generate_research_question,
)

__all__ = ["generate_research_question", "generate_best_of_n"]
