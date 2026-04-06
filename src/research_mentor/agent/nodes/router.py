"""Router node — determines which guide or expert handles the request.

Pure conditional routing function (no LLM, no DB calls).
"""

from __future__ import annotations

from loguru import logger

from research_mentor.agent.state import MentorState


def route_to_node(state: MentorState) -> str:
    """Determine which node should handle this request.

    Routing priority:
    1. specialist_needed → route to expert
    2. guidance_type → route to guide
    """
    specialist_needed = state.get("specialist_needed")
    guidance_type = state.get("guidance_type", "scaffolding").lower()

    if specialist_needed:
        if specialist_needed == "safety":
            logger.info("Router: Routing to safety_expert")
            return "safety_expert"
        elif specialist_needed == "ethics":
            logger.info("Router: Routing to ethics_expert")
            return "ethics_expert"
        elif specialist_needed == "communication":
            logger.info("Router: Routing to communication_expert")
            return "communication_expert"

    if guidance_type == "scaffolding":
        if state.get("student_has_provided_initial_thoughts", False):
            logger.info("Router: Routing to scaffolding_guide (diagnosis complete)")
            return "scaffolding_guide"
        else:
            logger.info("Router: Routing to understand_starting_point (diagnostic phase)")
            return "understand_starting_point"
    elif guidance_type == "guided_discovery":
        logger.info("Router: Routing to guided_discovery")
        return "guided_discovery"
    elif guidance_type == "pure_socratic":
        logger.info("Router: Routing to pure_socratic")
        return "pure_socratic"
    elif guidance_type == "reflection":
        logger.info("Router: Routing to reflection")
        return "reflection"

    raise ValueError(f"Invalid guidance_type: {guidance_type}")
