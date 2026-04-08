"""Main LangGraph agent graph definition.

Complete workflow: [progress_assessor + input_safety_review + student_discovery
                    + component_recommender]
    -> [planner + info_gathering] -> expert_router -> guide/expert
    -> presenter -> memory_writer -> END

Parallel fan-outs maximize GPU utilization via vLLM continuous batching:
- Entry: 4 concurrent LLM calls (assessment + safety + discovery + recommender)
- Planning: planner + info_gathering in parallel (no data dependency)
- Info gathering internals: up to 5 concurrent assistants

No input_validator node — local single-user system, low jailbreak risk.
Input safety review handles pedagogical concerns (dangerous experiments, ethics).
Prompt injection defense: untrusted content (uploaded docs, memories, user inputs)
is wrapped in <user_content> delimiters + anti-injection instructions in guide prompts.
See docs/SECURITY_REVIEW.md Finding 9.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from loguru import logger

from research_mentor.agent.nodes import (
    apply_presenter,
    assess_guidance_type,
    component_recommender_node,
    create_research_guidance_plan,
    extract_student_attributes_node,
    gather_information,
    guide_discovery,
    handle_communication_concern,
    handle_ethics_concern,
    handle_physical_safety,
    handle_psychological_safety,
    handle_safety_concern,
    provide_reflection_guidance,
    provide_scaffolding_guide,
    provide_socratic_guidance,
    review_input_safety,
    route_to_expert,
    route_to_node,
    understand_student_starting_point,
    validate_output_safety,
    write_memories,
)
from research_mentor.agent.state import MentorState
from research_mentor.utils.memory_profiler import memory_profile_node


async def create_mentor_graph() -> object:
    """Create the complete mentor agent graph.

    Flow:
    0. [PARALLEL] progress_assessor + input_safety_review + student_discovery + component_recommender
    1. post_assessment_router — Route based on combined assessment results
    2. create_research_guidance_plan — Plan methodology using persona
    3. gather_information — Autonomous information gathering (0-3 assistants)
    4. expert_router — Analyze gathered info for expert routing needs
    5. route_to_node — Route to guide (if no expert needed)
    6. Guide — Generate teaching content
    7. output_safety_validator — Validate content safety
    8. presenter — Apply persona voice and translation
    9. memory_writer — Extract and store conversation memories (sqlite-vec)
    """
    logger.info("Creating mentor graph...")

    graph = StateGraph(MentorState)

    # Add all nodes
    graph.add_node("progress_assessor", assess_guidance_type)
    graph.add_node("planner", create_research_guidance_plan)
    graph.add_node("info_gathering", gather_information)
    graph.add_node("expert_router", route_to_expert)
    graph.add_node("input_safety_review", review_input_safety)

    # Guide nodes
    graph.add_node("understand_starting_point", understand_student_starting_point)
    graph.add_node("scaffolding_guide", provide_scaffolding_guide)
    graph.add_node("guided_discovery", guide_discovery)
    graph.add_node("pure_socratic", provide_socratic_guidance)
    graph.add_node("reflection", provide_reflection_guidance)

    # Expert nodes
    graph.add_node("safety_expert", handle_safety_concern)
    graph.add_node("ethics_expert", handle_ethics_concern)
    graph.add_node("communication_expert", handle_communication_concern)

    # Safety handlers
    graph.add_node("output_safety_validator", validate_output_safety)
    graph.add_node("physical_safety_handler", handle_physical_safety)
    graph.add_node("psychological_safety_handler", handle_psychological_safety)

    # Presenter
    graph.add_node("presenter", apply_presenter)

    # Memory writer (extracts + stores memories after each turn)
    graph.add_node("memory_writer", write_memories)

    # Background nodes (fire-and-forget, route to END)
    graph.add_node("student_discovery", extract_student_attributes_node)
    graph.add_node("component_recommender", component_recommender_node)

    # Join node for parallel assessment branches (no LLM call)
    @memory_profile_node("post_assessment_router")
    async def _join_assessments(state: MentorState) -> dict[str, Any]:
        return {}

    graph.add_node("post_assessment_router", _join_assessments)

    # --- Fan-out: 4 concurrent LLM calls to maximize GPU utilization ---
    graph.add_edge(START, "progress_assessor")
    graph.add_edge(START, "input_safety_review")
    graph.add_edge(START, "student_discovery")
    graph.add_edge(START, "component_recommender")

    # --- Parallel branches converge ---
    graph.add_edge("progress_assessor", "post_assessment_router")
    graph.add_edge("input_safety_review", "post_assessment_router")
    graph.add_edge("student_discovery", END)
    graph.add_edge("component_recommender", END)

    # --- Conditional: post_assessment_router → expert or [planner + info_gathering] ---
    def route_after_assessments(state: MentorState) -> str | list[str]:
        specialist_needed = state.get("specialist_needed")
        if specialist_needed:
            logger.info("Assessment: Expert needed ({})", specialist_needed)
            return f"{specialist_needed}_expert"
        # Fan-out: planner + info_gathering have no data dependency on each other
        return ["planner", "info_gathering"]

    graph.add_conditional_edges(
        "post_assessment_router",
        route_after_assessments,
        {
            "safety_expert": "safety_expert",
            "ethics_expert": "ethics_expert",
            "communication_expert": "communication_expert",
            "planner": "planner",
            "info_gathering": "info_gathering",
        },
    )

    # --- planner + info_gathering converge → expert_router ---
    graph.add_edge("planner", "expert_router")
    graph.add_edge("info_gathering", "expert_router")

    # --- Conditional: expert_router → expert or guide ---
    def route_after_expert_analysis(state: MentorState) -> str:
        expert = state.get("route_to_expert")
        if expert:
            logger.info(f"Expert Router: Expert needed ({expert})")
            return f"{expert}_expert"
        guide = route_to_node(state)
        logger.info(f"Expert Router: No expert needed, routing to {guide}")
        return guide

    graph.add_conditional_edges(
        "expert_router",
        route_after_expert_analysis,
        {
            "safety_expert": "safety_expert",
            "ethics_expert": "ethics_expert",
            "communication_expert": "communication_expert",
            "understand_starting_point": "understand_starting_point",
            "scaffolding_guide": "scaffolding_guide",
            "guided_discovery": "guided_discovery",
            "pure_socratic": "pure_socratic",
            "reflection": "reflection",
        },
    )

    # --- Guides → output_safety_validator ---
    guide_names = [
        "understand_starting_point",
        "scaffolding_guide",
        "guided_discovery",
        "pure_socratic",
        "reflection",
    ]
    for guide_name in guide_names:
        graph.add_edge(guide_name, "output_safety_validator")

    # --- Experts → output_safety_validator ---
    graph.add_edge("safety_expert", "output_safety_validator")
    graph.add_edge("ethics_expert", "output_safety_validator")
    graph.add_edge("communication_expert", "output_safety_validator")

    # --- Conditional: output_safety_validator → handler or presenter ---
    def route_after_safety_validation(state: MentorState) -> str:
        if not state.get("content_safety_flagged", False):
            return "presenter"
        concern_type = state.get("output_safety_review", {}).get("concern_type")
        if concern_type == "psychological":
            return "psychological_safety_handler"
        return "physical_safety_handler"

    graph.add_conditional_edges(
        "output_safety_validator",
        route_after_safety_validation,
        {
            "physical_safety_handler": "physical_safety_handler",
            "psychological_safety_handler": "psychological_safety_handler",
            "presenter": "presenter",
        },
    )

    # --- Safety handlers → presenter ---
    graph.add_edge("physical_safety_handler", "presenter")
    graph.add_edge("psychological_safety_handler", "presenter")

    # --- Presenter → Memory Writer → END ---
    graph.add_edge("presenter", "memory_writer")
    graph.add_edge("memory_writer", END)

    # Compile with SQLite checkpointer for multi-turn conversation persistence
    from research_mentor.db.checkpointer import get_checkpointer

    checkpointer = await get_checkpointer()
    compiled_graph = graph.compile(checkpointer=checkpointer)

    logger.info("Mentor graph created successfully (with checkpointer)")
    return compiled_graph


# Lazy singleton
_mentor_graph_instance = None


async def get_mentor_graph() -> object:
    """Get or create the mentor graph singleton."""
    global _mentor_graph_instance
    if _mentor_graph_instance is None:
        logger.info("Initializing mentor graph singleton...")
        _mentor_graph_instance = await create_mentor_graph()
    return _mentor_graph_instance
