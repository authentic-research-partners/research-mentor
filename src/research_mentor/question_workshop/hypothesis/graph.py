"""Hypothesis LangGraph Definition — Research Question Workshop Agent.

Feed-forward graph: every ainvoke() executes exactly ONE stage and exits.

Architecture:
    START → input_validator → (if error → END)
         → security_check → (if error → END)
         → [stage_node] → END

Stage progression is managed via `current_stage` in checkpointed state.
Each stage node may update `current_stage` for the NEXT invocation
(never re-entering the graph within the same call).

Stages:
    1. Discovery (variables)
    2. Literature search (first entry) / Literature review (subsequent)
    3A. Hypothesis (theory + scope)
    3B. Data (datasets + feasibility)
    3C. Refinement (scope adjustment)
    3.5. Confounds
    4. Operationalization
    5A. Analysis plan
    5B-Ethics. Research ethics
    5B-Software. Software selection
    5B-Timeline. Timeline planning
    Completion
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from loguru import logger

from research_mentor.db.checkpointer import get_checkpointer

from .nodes.completion_handler import completion_handler
from .nodes.input_validator import input_validator
from .nodes.security_check import security_check
from .nodes.stage_1_discovery import stage_1_discovery
from .nodes.stage_2_review import stage_2_review
from .nodes.stage_2_search import stage_2_search
from .nodes.stage_3_5_confounds import stage_3_5_confounds
from .nodes.stage_3a_hypothesis import stage_3a_hypothesis
from .nodes.stage_3b_data import stage_3b_data
from .nodes.stage_3c_refinement import stage_3c_refinement
from .nodes.stage_4_operationalization import stage_4_operationalization
from .nodes.stage_5a_analysis import stage_5a_analysis
from .nodes.stage_5b_ethics import stage_5b_ethics
from .nodes.stage_5b_software import stage_5b_software
from .nodes.stage_5b_timeline import stage_5b_timeline
from .state import HypothesisState

# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def _route_after_input_validation(state: HypothesisState) -> str:
    """Route after input validation: to security_check or END (if rejected)."""
    if state.get("error"):
        logger.info("Input validation failed — routing to END")
        return END
    return "security_check"


def _route_after_security_check(state: HypothesisState) -> str:
    """Route after security check: to stage node or END (if threat detected).

    For stage_2_literature, sub-routes to stage_2_search (no papers) or
    stage_2_review (papers already found).
    """
    if state.get("error"):
        logger.info("Security check failed — routing to END")
        return END

    stage = state.get("current_stage", "stage_1_discovery")

    # Stage 2 sub-routing: search vs review
    if stage == "stage_2_literature":
        papers = state.get("papers_reviewed", [])
        node = "stage_2_review" if papers else "stage_2_search"
        logger.info("Routing to stage: {} → node: {}", stage, node)
        return node

    node = _STAGE_MAP.get(stage, "stage_1_discovery")
    logger.info("Routing to stage: {} → node: {}", stage, node)
    return node


# ---------------------------------------------------------------------------
# Routing map: current_stage value → graph node name
# ---------------------------------------------------------------------------

_STAGE_MAP: dict[str, str] = {
    "stage_1_discovery": "stage_1_discovery",
    # stage_2_literature is handled by sub-routing in _route_after_security_check
    "stage_3a_hypothesis": "stage_3a_hypothesis",
    "stage_3b_data": "stage_3b_data",
    "stage_3c_refinement": "stage_3c_refinement",
    "stage_3_5_confounds": "stage_3_5_confounds",
    "stage_4_operationalization": "stage_4_operationalization",
    "stage_5a_analysis": "stage_5a_analysis",
    "stage_5b_ethics": "stage_5b_ethics",
    "stage_5b_software": "stage_5b_software",
    "stage_5b_timeline": "stage_5b_timeline",
    "completion": "completion_handler",
}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


async def create_hypothesis_graph(*, checkpointer: Any | None = None) -> Any:
    """Create the Hypothesis agent graph.

    Args:
        checkpointer: Optional checkpointer override. If None, uses the
            production AsyncSqliteSaver. Pass MemorySaver() for evals/tests.

    Architecture:
        input_validator → security_check → (conditional) → stage_node → END
        Every ainvoke() executes exactly one stage and produces one response.
    """
    logger.info("Creating Hypothesis graph...")

    graph = StateGraph(HypothesisState)

    # --- Add nodes ---
    graph.add_node("input_validator", input_validator)
    graph.add_node("security_check", security_check)
    graph.add_node("stage_1_discovery", stage_1_discovery)
    graph.add_node("stage_2_search", stage_2_search)
    graph.add_node("stage_2_review", stage_2_review)
    graph.add_node("stage_3a_hypothesis", stage_3a_hypothesis)
    graph.add_node("stage_3b_data", stage_3b_data)
    graph.add_node("stage_3c_refinement", stage_3c_refinement)
    graph.add_node("stage_3_5_confounds", stage_3_5_confounds)
    graph.add_node("stage_4_operationalization", stage_4_operationalization)
    graph.add_node("stage_5a_analysis", stage_5a_analysis)
    graph.add_node("stage_5b_ethics", stage_5b_ethics)
    graph.add_node("stage_5b_software", stage_5b_software)
    graph.add_node("stage_5b_timeline", stage_5b_timeline)
    graph.add_node("completion_handler", completion_handler)

    # --- Entry point ---
    graph.set_entry_point("input_validator")

    # --- Input validation → security check or END ---
    graph.add_conditional_edges(
        "input_validator",
        _route_after_input_validation,
        {
            "security_check": "security_check",
            END: END,
        },
    )

    # --- Security check → stage routing or END ---
    graph.add_conditional_edges(
        "security_check",
        _route_after_security_check,
        {
            "stage_1_discovery": "stage_1_discovery",
            "stage_2_search": "stage_2_search",
            "stage_2_review": "stage_2_review",
            "stage_3a_hypothesis": "stage_3a_hypothesis",
            "stage_3b_data": "stage_3b_data",
            "stage_3c_refinement": "stage_3c_refinement",
            "stage_3_5_confounds": "stage_3_5_confounds",
            "stage_4_operationalization": "stage_4_operationalization",
            "stage_5a_analysis": "stage_5a_analysis",
            "stage_5b_ethics": "stage_5b_ethics",
            "stage_5b_software": "stage_5b_software",
            "stage_5b_timeline": "stage_5b_timeline",
            "completion_handler": "completion_handler",
            END: END,
        },
    )

    # --- All stage nodes exit to END ---
    graph.add_edge("stage_1_discovery", END)
    graph.add_edge("stage_2_search", END)
    graph.add_edge("stage_2_review", END)
    graph.add_edge("stage_3a_hypothesis", END)
    graph.add_edge("stage_3b_data", END)
    graph.add_edge("stage_3c_refinement", END)
    graph.add_edge("stage_3_5_confounds", END)
    graph.add_edge("stage_4_operationalization", END)
    graph.add_edge("stage_5a_analysis", END)
    graph.add_edge("stage_5b_ethics", END)
    graph.add_edge("stage_5b_software", END)
    graph.add_edge("stage_5b_timeline", END)
    graph.add_edge("completion_handler", END)

    # --- Compile with checkpointer ---
    if checkpointer is None:
        checkpointer = await get_checkpointer()
    compiled_graph = graph.compile(checkpointer=checkpointer)

    logger.info("Hypothesis graph created successfully")
    return compiled_graph


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_hypothesis_graph_instance = None


async def get_hypothesis_graph() -> Any:
    """Get or create the Hypothesis graph singleton.

    Uses lazy initialization to handle async setup.
    Thread-safe for single-threaded async applications.
    """
    global _hypothesis_graph_instance

    if _hypothesis_graph_instance is None:
        logger.info("Initializing Hypothesis graph singleton...")
        _hypothesis_graph_instance = await create_hypothesis_graph()

    return _hypothesis_graph_instance


def reset_hypothesis_graph() -> None:
    """Reset the Hypothesis graph singleton (for tests)."""
    global _hypothesis_graph_instance
    _hypothesis_graph_instance = None
