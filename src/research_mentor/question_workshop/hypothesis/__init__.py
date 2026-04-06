"""Hypothesis — Research question workshop agent.

Autonomous LangGraph architecture for guiding students through 9 stages of
research design: variable identification, literature review, hypothesis formation,
dataset search, confounds, operationalization, analysis plan, and resources.
"""

from research_mentor.question_workshop.hypothesis.graph import (
    get_hypothesis_graph,
    reset_hypothesis_graph,
)

__all__ = ["get_hypothesis_graph", "reset_hypothesis_graph"]
