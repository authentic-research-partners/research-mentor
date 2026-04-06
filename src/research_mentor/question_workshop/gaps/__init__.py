"""Gaps — Literature-driven research question workshop (pipeline).

4-stage pipeline: landscape explorer, gap analyzer (parallel strategy modes),
question generator, FUNDAMENTAL 11-criteria scorer. Natural science only.
"""

from research_mentor.question_workshop.gaps.pipeline import (
    generate_best_of_n,
    generate_gaps_questions,
)

__all__ = ["generate_gaps_questions", "generate_best_of_n"]
