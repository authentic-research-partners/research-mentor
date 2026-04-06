"""LangGraph state schema for the research mentor agent.

Single-user adaptation of the hosted MentorState. Removed multi-tenant fields
(user_id, student_id, organization_id, supervisor_id, conversation_id).
project_id is the primary key for all data.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

TeachingPersona = str
GuidanceType = Literal["scaffolding", "guided_discovery", "pure_socratic", "reflection"]
SpecialistType = Literal["safety", "ethics", "communication", "learning_pathway"]


class MentorState(TypedDict, total=False):
    """State shared across all nodes in the mentor agent graph."""

    # ===== Input (provided by API / CLI) =====
    messages: Annotated[list[BaseMessage], add_messages]
    project_id: str
    persona: TeachingPersona
    language: str
    response_length: str  # "normal" or "concise"
    # "adaptive", "beginner", "intermediate", "advanced", "expert"
    user_requested_guidance_level: str
    student_progress: dict[str, Any]
    student_skills: dict[str, Any]
    project_context: str
    student_demographics: dict[str, Any]

    # ===== Assessment (progress_assessor) =====
    guidance_type: GuidanceType
    confidence: float
    struggle_type: Literal["productive", "unproductive", "not_struggling"]
    needs_scaffolding_override: bool
    input_clarity: Literal["clear", "vague"]
    student_profile_text: str
    question_domain: str
    diagnostic_instruction: str
    guidance_level_mismatch_warning: str | None

    # ===== Adaptive Diagnostics =====
    student_has_provided_initial_thoughts: bool
    diagnostic_question_count: int

    # ===== Planning (planner) =====
    research_guidance_plan: Annotated[dict[str, Any] | None, lambda x, y: y]

    # ===== Information Gathering (info_gathering) =====
    gathered_information: dict[str, Any] | None
    conversation_info_cache: list[dict[str, Any]]
    retrieved_memories: list[dict[str, Any]]
    tool_warnings: list[dict[str, Any]]

    # ===== Expert Routing (expert_router) =====
    route_to_expert: SpecialistType | None
    expert_routing_review: dict[str, Any]

    # ===== Safety Review (input_safety_review) =====
    has_safety_concern: bool
    safety_review: dict[str, Any]
    specialist_needed: SpecialistType | None

    # ===== Content Generation (guide/expert nodes) =====
    response: str
    specialist_type: str | None
    content_metadata: dict[str, Any]

    # ===== Output Safety (output_safety_validator) =====
    content_safety_flagged: bool
    output_safety_review: dict[str, Any]

    # ===== Presentation (presenter) =====
    final_response: str

    # ===== Component Recommendation (component_recommender) =====
    component_recommendation: str | None

    # ===== Knowledge =====
    student_knowledge_profile: dict[str, Any]
