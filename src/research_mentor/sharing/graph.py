"""Sharing & Publication LangGraph Definition.

Feed-forward graph: every ainvoke() executes exactly ONE node and exits.

Architecture:
    START → entry_validator → intent_detector → (conditional) → [phase_node] → END
                                                  ↓ (if error)
                                                 END

Key difference from Gaps: non-linear phase routing. After context_gathering,
the student can access any capability phase in any order. The intent_detector
detects capability-switch signals and routes to capability_switch handler.

Phases:
    1. context_gathering — understand research + communication goals (mandatory first)
    2. writing_support — writing mentorship (not ghostwriting)
    3. venue_discovery — venue matching + evaluation + predatory detection
    4. communication_guidance — presentations, digital media, science art, etc.
    5. completion — summary + new questions feedback loop
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from loguru import logger

from research_mentor.db.checkpointer import get_checkpointer

from .nodes.capability_switch import capability_switch
from .nodes.communication_guidance import communication_guidance
from .nodes.completion_handler import completion_handler
from .nodes.context_gathering import context_gathering
from .nodes.entry_validator import entry_validator
from .nodes.intent_detector import intent_detector
from .nodes.venue_discovery import venue_discovery
from .nodes.writing_support import writing_support
from .state import SharingState

# ---------------------------------------------------------------------------
# Routing map: current_phase value -> graph node name
# ---------------------------------------------------------------------------

_PHASE_MAP: dict[str, str] = {
    "context_gathering": "context_gathering",
    "writing_support": "writing_support",
    "venue_discovery": "venue_discovery",
    "communication_guidance": "communication_guidance",
    "completion": "completion_handler",
}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def _route_after_capability_switch(state: SharingState) -> str:
    """Route after capability switch: to target phase node or END (if invalid)."""
    if state.get("invalid_capability"):
        return END
    phase = state.get("current_phase", "context_gathering")
    node = _PHASE_MAP.get(phase, "context_gathering")
    logger.info("Capability switch → chaining to: {}", node)
    return node


def _route_after_validation(state: SharingState) -> str:
    """Route after entry validation: to intent_detector or END."""
    if state.get("error"):
        logger.info("Entry validation failed -- routing to END")
        return END
    return "intent_detector"


def _route_after_intent(state: SharingState) -> str:
    """Route after intent detection: to phase node or capability switch.

    Three routing decisions:
    1. capability_switch intent + context gathered → capability_switch handler
    2. capability_switch intent + no context → stay in context_gathering
    3. ready_to_complete → completion_handler
    4. Otherwise → current phase node
    """
    intent = state.get("user_intent", "general")

    # Capability switch (only if context has been gathered)
    if intent == "capability_switch":
        if state.get("context_gathered", False):
            logger.info(
                "Capability switch requested → {}",
                state.get("requested_capability"),
            )
            return "capability_switch"
        logger.info("Capability switch requested but context not gathered — staying in context")
        return "context_gathering"

    # Ready to complete
    if intent == "ready_to_complete":
        logger.info("Ready to complete → completion_handler")
        return "completion_handler"

    # Normal routing: go to current phase
    phase = state.get("current_phase", "context_gathering")
    node = _PHASE_MAP.get(phase, "context_gathering")
    logger.info("Routing to phase: {} -> node: {}", phase, node)
    return node


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


async def create_sharing_graph(*, checkpointer: Any | None = None) -> Any:
    """Create the Sharing & Publication graph.

    Architecture:
        entry_validator → intent_detector → (conditional) → phase_node → END
    """
    logger.info("Creating Sharing graph...")

    graph = StateGraph(SharingState)

    # --- Add nodes ---
    graph.add_node("entry_validator", entry_validator)
    graph.add_node("intent_detector", intent_detector)
    graph.add_node("capability_switch", capability_switch)
    graph.add_node("context_gathering", context_gathering)
    graph.add_node("writing_support", writing_support)
    graph.add_node("venue_discovery", venue_discovery)
    graph.add_node("communication_guidance", communication_guidance)
    graph.add_node("completion_handler", completion_handler)

    # --- Entry point ---
    graph.set_entry_point("entry_validator")

    # --- entry_validator → intent_detector (or END if error) ---
    graph.add_conditional_edges(
        "entry_validator",
        _route_after_validation,
        {
            "intent_detector": "intent_detector",
            END: END,
        },
    )

    # --- intent_detector → phase node (based on current_phase + intent) ---
    graph.add_conditional_edges(
        "intent_detector",
        _route_after_intent,
        {
            "capability_switch": "capability_switch",
            "context_gathering": "context_gathering",
            "writing_support": "writing_support",
            "venue_discovery": "venue_discovery",
            "communication_guidance": "communication_guidance",
            "completion_handler": "completion_handler",
        },
    )

    # --- capability_switch chains to target phase node ---
    graph.add_conditional_edges(
        "capability_switch",
        _route_after_capability_switch,
        {
            "context_gathering": "context_gathering",
            "writing_support": "writing_support",
            "venue_discovery": "venue_discovery",
            "communication_guidance": "communication_guidance",
            "completion_handler": "completion_handler",
            END: END,
        },
    )

    # --- All phase nodes exit to END ---
    graph.add_edge("context_gathering", END)
    graph.add_edge("writing_support", END)
    graph.add_edge("venue_discovery", END)
    graph.add_edge("communication_guidance", END)
    graph.add_edge("completion_handler", END)

    # --- Compile with checkpointer ---
    if checkpointer is None:
        checkpointer = await get_checkpointer()
    compiled_graph = graph.compile(checkpointer=checkpointer)

    logger.info("Sharing graph created successfully")
    return compiled_graph


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_sharing_graph_instance = None


async def get_sharing_graph() -> Any:
    """Get or create the Sharing graph singleton."""
    global _sharing_graph_instance

    if _sharing_graph_instance is None:
        logger.info("Initializing Sharing graph singleton...")
        _sharing_graph_instance = await create_sharing_graph()

    return _sharing_graph_instance


def reset_sharing_graph() -> None:
    """Reset the Sharing graph singleton (for tests)."""
    global _sharing_graph_instance
    _sharing_graph_instance = None
