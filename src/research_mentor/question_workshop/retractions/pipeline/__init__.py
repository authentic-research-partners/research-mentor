"""Retractions pipeline — batch question generation from retracted papers.

5-stage sequential pipeline:
1. Case Retriever — find retracted papers by field or DOI
2. Failure Analyzer — classify failure type + methodology issues
3. Impact Assessor — citation analysis + reopened gap identification
4. Question Generator — produce research questions from reopened gaps
5. Feasibility Scorer — rank questions by feasibility dimensions
"""

from research_mentor.question_workshop.retractions.pipeline.orchestrator import (
    generate_retraction_questions,
)

__all__ = ["generate_retraction_questions"]
