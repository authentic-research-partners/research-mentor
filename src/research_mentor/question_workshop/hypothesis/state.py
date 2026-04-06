"""Hypothesis State Schema — TypedDict for LangGraph checkpointer.

All Hypothesis workflow state stored in checkpointer. No manual persistence needed.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from research_mentor.config import HypothesisConfig, load_config


class HypothesisState(TypedDict, total=False):
    """Hypothesis workflow state — all fields optional for partial updates."""

    # Core tracking
    user_id: str
    session_id: str
    language: str
    current_stage: str  # 'stage_1_discovery', 'stage_2_literature', etc.
    student_demographics: dict[str, Any]  # age, country, gradeLevel, firstName

    # Conversation (auto-pruned to last MAX_MESSAGE_HISTORY messages)
    messages: Annotated[list[BaseMessage], add_messages]
    last_user_message: str
    last_ai_response: str

    # Stage 1: Variables
    independent_var: str | None  # X variable
    dependent_var: str | None  # Y variable

    # Stage 2: Literature
    papers_reviewed: list[dict[str, Any]]  # OpenAlex results
    papers_discussed_count: int
    papers_found_count: int
    gap_identified: bool
    gap_description: str | None
    ideal_measurements: dict[str, str]  # {'X': '...', 'Y': '...'}
    review_mode: str | None  # 'guided' or 'independent'

    # Stage 3A: Hypothesis
    scope_defined: bool
    scope_details: dict[str, Any]  # {'when': '...', 'where': '...', 'population': '...'}
    hypothesis: str | None
    hypothesis_direction: str | None  # 'increase', 'decrease', 'u-shaped'
    theory: str | None  # Mechanism explaining X -> Y

    # Stage 3B: Datasets
    datasets_searched: bool
    datasets_found: list[dict[str, Any]]
    datasets_count: int

    # Stage 3C: Refinement
    selected_dataset: dict[str, Any] | None
    scope_adjusted: bool

    # Stage 3.5: Confounds
    confounds: list[dict[str, Any]]  # [{'name': '...', 'affects_x': bool, 'affects_y': bool}]
    confounds_count: int

    # Stage 4: Operationalization
    data_alignment_evaluated: bool
    measurement_quality_notes: str | None
    x_measurement: str | None
    y_measurement: str | None

    # Stage 5A: Analysis
    statistical_method: str | None  # 'regression', 't-test', 'ANOVA', etc.
    h0_defined: bool
    h1_defined: bool

    # Stage 5B: Resources
    ethics_discussed: bool
    software_discussed: bool
    timeline_discussed: bool
    resources_planned: list[str]

    # Info gathering tracking
    info_gathering_history: list[dict[str, Any]]

    # Completion tracking
    pdf_requested: bool
    workflow_complete: bool


def hypothesis_config() -> HypothesisConfig:
    """Load Hypothesis config from TOML (cached at config level).

    Returns a HypothesisConfig with max_message_history, max_papers_reviewed,
    max_datasets_found, max_confounds, min_confounds_for_transition,
    conversational_temperature, extraction_temperature.
    """
    return load_config().question_workshop.hypothesis


def prune_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Prune messages to prevent unbounded growth."""
    limit = hypothesis_config().max_message_history
    if len(messages) <= limit:
        return messages
    return messages[-limit:]


def initialize_hypothesis_state(
    user_id: str,
    session_id: str,
    language: str = "en",
    student_demographics: dict[str, Any] | None = None,
) -> HypothesisState:
    """Initialize Hypothesis state for new session."""
    return {
        "user_id": user_id,
        "session_id": session_id,
        "language": language,
        "student_demographics": student_demographics or {},
        "current_stage": "stage_1_discovery",
        "messages": [],
        "papers_reviewed": [],
        "papers_discussed_count": 0,
        "papers_found_count": 0,
        "gap_identified": False,
        "scope_defined": False,
        "datasets_searched": False,
        "datasets_found": [],
        "datasets_count": 0,
        "confounds": [],
        "confounds_count": 0,
        "data_alignment_evaluated": False,
        "h0_defined": False,
        "h1_defined": False,
        "ethics_discussed": False,
        "software_discussed": False,
        "timeline_discussed": False,
        "resources_planned": [],
        "info_gathering_history": [],
    }
