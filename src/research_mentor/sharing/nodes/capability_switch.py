"""Capability Switch Handler — transition between phases.

Generates a brief bridging response and updates the phase. Does NOT
reference research_summary (extraction artifacts would leak as tool leakage).
The next phase node has full state access for context.
"""

from typing import Any

from loguru import logger

from research_mentor.sharing.state import SharingState

_VALID_PHASES = frozenset({
    "writing_support",
    "venue_discovery",
    "communication_guidance",
    "completion",
})


async def capability_switch(state: SharingState) -> dict[str, Any]:
    """Handle phase transitions — update phase metadata, then chain to target node.

    Does NOT generate a response. The target phase node (routed via graph
    conditional edges) generates the actual contextual LLM response. This
    avoids canned transition strings that ignore what the student just said.
    """
    requested = state.get("requested_capability")
    current_phase = state.get("current_phase", "context_gathering")

    logger.info(
        "Capability switch: {} → {}",
        current_phase, requested,
    )

    if not requested or requested not in _VALID_PHASES:
        logger.warning("Invalid capability requested: {}", requested)
        return {
            "last_ai_response": (
                "I can help with writing, finding venues, "
                "or presentation guidance. Which would you like?"
            ),
            "invalid_capability": True,
        }

    phases_visited = list(state.get("phases_visited", []))
    if requested not in phases_visited:
        phases_visited.append(requested)

    return {
        "current_phase": requested,
        "phases_visited": phases_visited,
        "requested_capability": None,
    }
