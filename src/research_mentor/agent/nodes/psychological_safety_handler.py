"""Psychological Safety Handler — handles psychological concerns.

When psychological safety concerns are detected in OUTPUT (guide proposed unhealthy timeline,
rude content, etc.), this node handles them appropriately.

Purpose:
- Validator flagged psychological safety concerns in GUIDE'S proposal (not student's request)
- Dual behavior: BLOCKS rude/offensive content OR asks about planning resources
- For planning concerns: Asks about teacher/supervisor, deadline, available time
- For rude/offensive: Blocks with generic safe message (does NOT regenerate)

Different from physical_safety_handler:
- Physical concerns → Ask about equipment/supervision (physical_safety_handler)
- Psychological concerns → Ask about planning resources OR block (this node)

IMPORTANT: This is OUTPUT validation (what guide proposed), NOT input validation (what student asked).
Input psychological concerns are handled by communication_expert.

This node does NOT apply persona voice or translation.
That's the presenter's job. This node generates the information request or block message only.
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

PSYCHOLOGICAL_SAFETY_INSTRUCTION = """You are a research mentor gathering planning resource information from a student.

**Student Profile:**
{student_profile_text}

**Original content (from guide) was flagged with psychological safety concerns:**
{safety_concerns}

**Student's Original Question:**
{student_question}

**Your Job:**
Generate a message that gathers planning resource information for the NEXT turn.

**Why we're asking:**
The guide's proposed approach involves potentially unhealthy timelines or pressure. Before providing better planning guidance on the NEXT turn, we need to understand:
1. Do they have a teacher/supervisor they can talk to about planning their project?
2. What is their actual deadline (if they have one)?
3. How much time do they have available per week?

**Message structure:**
1. **Acknowledge their goal** - Don't dismiss what they're trying to achieve
2. **Explain concern briefly** - "The timeline suggested could lead to burnout/stress..."
3. **Ask specific questions:**
   - Do you have a teacher, supervisor, or mentor you work with on this project who could help with planning?
   - When is your project deadline?
   - How much time can you realistically work on this per week?
4. **Set expectation** - "Once I know this, I'll help you create a sustainable plan that produces better results"
5. **STOP THERE** - Do NOT provide planning recommendations yet

**CRITICAL SAFETY:**
- ⚠️ NEVER suggest getting advice from "adults" or "someone you trust" (too vague, could be unsafe)
- ✅ ONLY ask about specific roles: teacher, supervisor, mentor, parent (institutional/family roles only)
- If they don't have a supervisor, we'll provide planning guidance ourselves on next turn

**Tone:**
- Supportive, helpful (not scolding or dismissive)
- Educational (explain we need this info to give better planning guidance)
- Forward-looking (I WILL help you plan on next turn, just need context first)

**CRITICAL:**
- Write in CLEAR, DIRECT ENGLISH
- DO NOT apply persona voice (that's presenter's job)
- DO NOT translate (that's presenter's job)
- Focus ONLY on gathering information for the NEXT turn

**Output:** Clear information request about planning resources in plain English.
"""


@memory_profile_node("psychological_safety_handler")
async def handle_psychological_safety(state: MentorState) -> dict[str, Any]:
    """Handle psychological safety concerns with dual behavior.

    Dual behavior: BLOCKS rude/offensive OR asks about planning resources.
    For planning concerns: Asks about teacher/supervisor, deadline, available time.
    For rude/offensive: Returns generic safe message (does NOT regenerate).
    """
    logger.info("Psychological safety handler: Processing...")

    token_limit = get_token_limit_instruction()
    response_length_inst = get_response_length_instruction(
        "handler", state.get("response_length", "normal"),
    )

    output_safety_review = state.get("output_safety_review", {})
    concerns = output_safety_review.get("concerns", [])
    severity = output_safety_review.get("severity", "unknown")
    concern_type = output_safety_review.get("concern_type", "psychological")

    # Check if the concern is about rude/offensive content in the response
    concern_text = " ".join(concerns).lower()
    is_rude_offensive = (
        "rude" in concern_text
        or "offensive" in concern_text
        or "condescending" in concern_text
        or "dismissive" in concern_text
    )

    if is_rude_offensive:
        logger.warning(
            "Psychological safety handler: Blocking rude/offensive content "
            "(severity: {}, concerns: {})", severity, concerns,
        )
        return {
            "response": "I want to support your research project effectively. "
            "Let me think about the best way to guide you on this. "
            "Could you tell me more about what specific aspect you need help with?",
            "content_metadata": {
                "node_executed": "psychological_safety_handler",
                "psychological_safety_check": True,
                "content_blocked": True,
                "original_severity": severity,
                "original_concern_type": concern_type,
                "concerns_flagged": concerns,
            },
        }

    # For timeline/burnout concerns, ask about planning resources
    logger.info(
        "Psychological safety handler: Planning concern - gathering resources (severity: {})",
        severity,
    )

    messages = state.get("messages", [])
    demographics = state.get("student_demographics", {})
    student_question = messages[-1].content if messages else ""
    student_profile_text = build_student_profile_context(demographics)
    reasoning = output_safety_review.get("reasoning", "Psychological safety concerns detected")

    safety_concerns_text = f"""**Severity:** {severity}
**Type:** {concern_type}
**Specific Concerns:**
{chr(10).join(f"  - {concern}" for concern in concerns)}
**Reasoning:** {reasoning}"""

    formatted_instruction = PSYCHOLOGICAL_SAFETY_INSTRUCTION.format(
        student_profile_text=student_profile_text or "Student demographics not available",
        safety_concerns=safety_concerns_text,
        student_question=student_question,
    )
    system_content = formatted_instruction + "\n\n" + token_limit + "\n\n" + response_length_inst

    llm = get_chat_llm()

    result = await llm.ainvoke([
        SystemMessage(content=system_content),
        HumanMessage(
            content="Generate a planning resource information request that acknowledges "
            "the student's goal, explains briefly why we're concerned about the "
            "timeline/approach, asks about teacher/supervisor for planning help "
            "(be specific: teacher, supervisor, mentor, parent only), asks about "
            "deadline and available time, and sets expectation that guidance will "
            "come after they respond. Do NOT provide planning recommendations yet. "
            "Write in plain English — presenter will apply persona voice and translation later."
        ),
    ])

    return {
        "response": str(result.content),
        "content_metadata": {
            "node_executed": "psychological_safety_handler",
            "psychological_safety_check": True,
            "original_severity": severity,
            "original_concern_type": concern_type,
            "concerns_flagged": concerns,
        },
    }
