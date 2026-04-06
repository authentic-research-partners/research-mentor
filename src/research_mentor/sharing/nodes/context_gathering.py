"""Context Gathering Node — understand the student's research and communication goals.

Mandatory first phase. Gathers:
1. What research the student did (research_summary)
2. What they want to create (output_type)
3. Who their audience is (target_audience)
4. Education level and field (detected from demographics + conversation)

When enough context is gathered, sets context_gathered=True and offers
the available capabilities.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.sharing.schemas import ContextGatheringExtraction
from research_mentor.sharing.state import SharingState
from research_mentor.sharing.utils import build_conversation_context, build_sharing_system_prompt
from research_mentor.sharing.utils.prompts import CONTEXT_GATHERING_PROMPT


async def context_gathering(state: SharingState) -> dict[str, Any]:
    """Gather research context and communication goals from the student."""
    logger.info("Context Gathering: Processing...")

    messages = state.get("messages", [])
    demographics = state.get("student_demographics", {})

    # Detect student level from demographics if not already set
    student_level = state.get("student_level")
    if not student_level and demographics:
        student_level = _detect_level_from_demographics(demographics)

    # Extract context from conversation so far
    context_text = build_conversation_context(messages)
    try:
        extraction = await structured_call(
            ContextGatheringExtraction,
            [
                SystemMessage(content=(
                    "Extract research context from this conversation. "
                    "Identify what the student's research is about, what output they want "
                    "to create, who their audience is, and their education level."
                )),
                HumanMessage(content=f"Conversation so far:\n{context_text}"),
            ],
            thinking="medium",
            temperature=0.0,
        )
    except Exception as e:
        logger.warning("Context extraction failed: {}", e)
        extraction = None

    # Update state with extracted context
    updates: dict[str, Any] = {}

    if extraction:
        if extraction.research_summary and not state.get("research_summary"):
            updates["research_summary"] = extraction.research_summary
        if extraction.output_type and not state.get("output_type"):
            updates["output_type"] = extraction.output_type
        if extraction.target_audience and not state.get("target_audience"):
            updates["target_audience"] = extraction.target_audience
        if extraction.student_level:
            student_level = extraction.student_level
        if extraction.research_field and not state.get("research_field"):
            updates["research_field"] = extraction.research_field

    if student_level:
        updates["student_level"] = student_level

    # Check if context is complete enough to proceed
    has_summary = bool(state.get("research_summary") or updates.get("research_summary"))
    has_goal = bool(
        state.get("output_type") or updates.get("output_type")
        or (extraction and extraction.context_complete)
    )
    context_complete = has_summary and has_goal

    if context_complete and not state.get("context_gathered"):
        updates["context_gathered"] = True
        logger.info("Context gathering complete — capabilities unlocked")

    # Choose prompt based on whether context is now gathered
    already_gathered = state.get("context_gathered", False) or updates.get("context_gathered", False)
    if already_gathered:
        # Context is gathered — use a prompt that offers capabilities
        from research_mentor.sharing.utils.prompts import CONTEXT_DONE_PROMPT

        system_prompt = build_sharing_system_prompt(CONTEXT_DONE_PROMPT, state)
    else:
        system_prompt = build_sharing_system_prompt(CONTEXT_GATHERING_PROMPT, state)

    llm = get_chat_llm(temperature=0.7, max_tokens=100)
    response = await llm.ainvoke(messages + [SystemMessage(content=system_prompt)])
    response_text = str(response.content)

    updates["last_ai_response"] = response_text
    return updates


def _detect_level_from_demographics(demographics: dict[str, Any]) -> str | None:
    """Infer education level from student demographics."""
    age = demographics.get("age")
    grade = demographics.get("gradeLevel", "")
    education = demographics.get("educationLevel", "")

    if education:
        edu_lower = education.lower()
        if "middle" in edu_lower:
            return "middle_school"
        if "high" in edu_lower:
            return "high_school"
        if any(x in edu_lower for x in ("undergrad", "college", "university", "graduate")):
            return "university"

    if isinstance(age, int):
        if age < 14:
            return "middle_school"
        if age < 18:
            return "high_school"
        return "university"

    if grade:
        grade_lower = str(grade).lower()
        if any(x in grade_lower for x in ("6", "7", "8", "middle")):
            return "middle_school"
        if any(x in grade_lower for x in ("9", "10", "11", "12", "high")):
            return "high_school"

    return None
