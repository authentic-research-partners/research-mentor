"""Strategy prompts and descriptions for the Gaps pipeline.

Re-exports from the shared gaps/utils/prompts.py to keep a single source
of truth for strategy definitions.
"""

from research_mentor.question_workshop.gaps.utils.prompts import (
    STRATEGY_DESCRIPTIONS,
    STRATEGY_MAP,
)

__all__ = ["STRATEGY_MAP", "STRATEGY_DESCRIPTIONS"]
