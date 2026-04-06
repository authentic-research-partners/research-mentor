"""Sharing & Publication State Schema — TypedDict for LangGraph checkpointer.

All Sharing workflow state stored in checkpointer. No manual persistence needed.
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SharingState(TypedDict, total=False):
    """Sharing workflow state — all fields optional for partial updates."""

    # Core tracking
    user_id: str
    session_id: str
    language: str
    current_phase: str  # 'context_gathering', 'writing_support', 'venue_discovery',
    #                     'communication_guidance', 'completion'
    student_demographics: dict[str, Any]  # age, country, gradeLevel, firstName

    # Conversation (auto-pruned to last MAX_MESSAGE_HISTORY messages)
    messages: Annotated[list[BaseMessage], add_messages]
    last_user_message: str
    last_ai_response: str
    error: str | None

    # Intent detection (runs before phase node)
    user_intent: str | None  # 'engaged', 'frustrated', 'asking_how', 'general',
    #                          'capability_switch', 'ready_to_complete'
    requested_capability: str | None  # target phase when intent='capability_switch'

    # Context gathering (mandatory first phase output)
    research_summary: str | None  # what the student's research is about
    output_type: str | None  # 'paper', 'poster', 'presentation', 'video', 'blog', etc.
    target_audience: str | None  # 'peers', 'general_public', 'experts', 'teachers'
    student_level: str | None  # 'middle_school', 'high_school', 'university', 'adult'
    research_field: str | None  # 'natural_science', 'social_science', etc.
    context_gathered: bool  # gate for allowing phase switches

    # Writing support
    writing_focus: str | None  # 'structure', 'feedback', 'revision', 'simulated_review'
    draft_sections_discussed: list[str]  # e.g. ['abstract', 'methods', 'introduction']

    # Venue discovery
    matched_venues: list[dict[str, Any]]
    venue_search_done: bool
    selected_venues: list[dict[str, Any]]  # venues the student expressed interest in
    venue_evaluation_done: bool

    # Communication guidance (sub-routing within the phase)
    communication_topic: str | None  # 'poster', 'presentation', 'digital_media',
    #                                  'science_art', 'authorship', 'open_science'

    # Progress tracking (tracks which phases student has used)
    phases_visited: list[str]
    workflow_complete: bool

    # Routing
    invalid_capability: bool  # set by capability_switch when requested phase is invalid

    # Tool usage tracking
    info_gathering_history: list[dict[str, Any]]
    last_tool_calls: list[str]


# Constants
MAX_MESSAGE_HISTORY = 30
MAX_VENUE_RESULTS = 20


def prune_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Prune messages to prevent unbounded growth."""
    if len(messages) <= MAX_MESSAGE_HISTORY:
        return messages
    return messages[-MAX_MESSAGE_HISTORY:]


def initialize_sharing_state(
    user_id: str,
    session_id: str,
    language: str = "en",
    student_demographics: dict[str, Any] | None = None,
) -> SharingState:
    """Initialize Sharing state for new session."""
    return {
        "user_id": user_id,
        "session_id": session_id,
        "language": language,
        "student_demographics": student_demographics or {},
        "current_phase": "context_gathering",
        "context_gathered": False,
        "messages": [],
        "draft_sections_discussed": [],
        "matched_venues": [],
        "venue_search_done": False,
        "selected_venues": [],
        "venue_evaluation_done": False,
        "phases_visited": [],
        "workflow_complete": False,
        "info_gathering_history": [],
        "last_tool_calls": [],
    }
