"""Retractions workshop — pipeline question generation from retracted papers.

5-stage batch pipeline: case retriever → failure analyzer →
impact assessor → question generator → feasibility scorer.
"""

from research_mentor.question_workshop.retractions.pipeline import (
    generate_retraction_questions,
)

__all__ = ["generate_retraction_questions"]
