"""Question Workshop — research question generation pipelines.

Supports multiple workshop types:
- phenomenon: Generate original research questions from scratch (4-stage pipeline)
- claims: Transform media claims into research projects (5-stage pipeline)
- hypothesis: Interactive 9-stage research design workshop (18-node LangGraph)
- gaps: Literature-driven fundamental question discovery (5-node LangGraph)
"""

from research_mentor.question_workshop.types import WorkshopType

__all__ = ["WorkshopType"]
