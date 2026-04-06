"""Physical Safety Handler — gathers safety resource information.

When physical safety concerns are detected in OUTPUT, this node gathers information
about the student's available safety resources.

Purpose:
- Validator flagged PHYSICAL safety concerns → need to understand student's context
- Ask about: equipment available, supervision, work location
- Student responds on NEXT turn
- Guide can then provide better, contextual safety guidance

This node does NOT apply persona voice or translation.
That's the presenter's job. This node generates the information request only.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.agent.prompts.shared import (
    build_student_profile_context,
    get_response_length_instruction,
    get_token_limit_instruction,
)
from research_mentor.agent.state import MentorState
from research_mentor.llm import get_chat_llm
from research_mentor.utils.memory_profiler import memory_profile_node

SAFETY_RESOURCE_CHECK_INSTRUCTION = """You are a research mentor gathering safety information from a student.

**Student Profile:**
{student_profile_text}

**Original unsafe content was flagged with these concerns:**
{safety_concerns}

**Student's Original Question:**
{student_question}

**Your Job:**
Generate a message that gathers safety resource information for the NEXT turn.

**Why we're asking:**
The student's question involves potentially unsafe procedures. Before providing guidance on the NEXT turn, we need to understand:
1. What safety equipment they have available
2. Whether adult/supervisor will be present
3. Where they'll be doing this work (lab, home, school)

**Message structure:**
1. **Acknowledge their interest** - Don't ignore what they asked
2. **Explain why you're asking** - "To help you safely with this..."
3. **Ask specific questions:**
   - What safety equipment do you have available?
   - Will an adult or supervisor be present when you do this?
   - Where will you be doing this work (school lab, home, etc.)?
4. **Set expectation** - "Once you let me know, I'll provide specific guidance based on your resources"
5. **STOP THERE** - Do NOT provide recommendations yet

**Tone:**
- Supportive, helpful (not scolding or dismissive)
- Educational (explain we need this info to give better guidance)
- Forward-looking (I WILL help you on next turn, just need context first)

**CRITICAL:**
- Write in CLEAR, DIRECT ENGLISH
- DO NOT apply persona voice (that's presenter's job)
- DO NOT translate (that's presenter's job)
- Focus ONLY on gathering information for the NEXT turn

**Output:** Clear information request in plain English that prepares for next turn.
"""


@memory_profile_node("physical_safety_handler")
async def handle_physical_safety(state: MentorState) -> dict[str, Any]:
    """Generate safety resource information request."""
    logger.info("Physical safety handler: Gathering safety info...")

    messages = state.get("messages", [])
    output_safety_review = state.get("output_safety_review", {})
    demographics = state.get("student_demographics", {})
    student_question = messages[-1].content if messages else ""

    student_profile_text = build_student_profile_context(demographics)

    # Extract safety concerns
    concerns = output_safety_review.get("concerns", [])
    severity = output_safety_review.get("severity", "unknown")
    concern_type = output_safety_review.get("concern_type", "unknown")
    reasoning = output_safety_review.get("reasoning", "Safety concerns detected")

    safety_concerns_text = f"""**Severity:** {severity}
**Type:** {concern_type}
**Specific Concerns:**
{chr(10).join(f"  - {concern}" for concern in concerns)}
**Reasoning:** {reasoning}"""

    formatted_instruction = SAFETY_RESOURCE_CHECK_INSTRUCTION.format(
        student_profile_text=student_profile_text or "Student demographics not available",
        safety_concerns=safety_concerns_text,
        student_question=student_question,
    )

    token_limit = get_token_limit_instruction()
    response_length_inst = get_response_length_instruction(
        "handler", state.get("response_length", "normal"),
    )
    system_content = formatted_instruction + "\n\n" + token_limit + "\n\n" + response_length_inst

    llm = get_chat_llm()

    result = await llm.ainvoke([
        SystemMessage(content=system_content),
        HumanMessage(
            content="Generate a safety resource information request that acknowledges "
            "the student's question, explains why we need this information, asks about "
            "their available safety equipment, supervision, and work location, and sets "
            "the expectation that guidance will come after they respond. "
            "Do NOT provide recommendations or alternatives yet. "
            "Write in plain English — presenter will apply persona voice and translation later."
        ),
    ])

    return {
        "response": str(result.content),
        "content_metadata": {
            "node_executed": "physical_safety_handler",
            "physical_safety_check": True,
            "original_severity": severity,
            "original_concern_type": concern_type,
            "concerns_flagged": concerns,
        },
    }
