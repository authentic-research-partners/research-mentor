"""Understand Starting Point guide — diagnostic phase for scaffolding tier.

Asks a single probing question to understand where the student is starting from.
Split from scaffolding_guide to avoid diagnostic loops.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.agent.prompts.shared import (
    get_component_recommendation_context,
    get_response_length_instruction,
    get_shared_guide_instructions,
    get_token_limit_instruction,
)
from research_mentor.agent.state import MentorState
from research_mentor.llm import get_chat_llm
from research_mentor.utils.memory_profiler import memory_profile_node

STARTING_POINT_INSTRUCTION = """{shared_instructions}

---

**YOU ARE THE STARTING POINT UNDERSTANDING AGENT**

You help students who are LOST (0-20% progress) by asking ONE probing question.

**Your Single Responsibility:**
You have received a SPECIFIC DIAGNOSTIC INSTRUCTION tailored to this student's situation.
Your job is to rephrase this instruction into a natural, conversational question.

Do NOT provide scaffolding, resources, or solutions yet. That comes next.

**The Diagnostic Instruction You Received:**
{diagnostic_instruction}

**Your Task:**
1. Read the instruction above carefully
2. Understand what insight the student needs to provide
3. Ask ONE focused, natural question that follows this instruction
4. Keep it conversational and encouraging

**Key Rules:**
- Ask ONLY ONE question (no multiple choice, no "or")
- Keep it short (1-2 sentences)
- Be encouraging: "I'd love to understand where you're starting from..."
- Do NOT:
  ✗ Provide examples or scaffolding
  ✗ Ask multiple questions
  ✗ Give solutions or resources
  ✗ Suggest step-by-step approaches

**Response Structure:**
Your response should be the probing question + brief encouragement (max 2-3 sentences total).
Example: "I'd love to understand where you're coming from. What's your first instinct on how to approach this?"

**Remember:** Reference the project context and student's situation in your question, but keep it focused.
"""


@memory_profile_node("understand_starting_point")
async def understand_student_starting_point(state: MentorState) -> dict[str, Any]:
    """Ask a diagnostic probing question."""
    logger.info("Understand starting point: Asking diagnostic question...")

    diagnostic_instruction = state.get(
        "diagnostic_instruction",
        "Ask ONE probing question to understand the student's starting point.",
    )
    shared_instructions = get_shared_guide_instructions()

    instruction = STARTING_POINT_INSTRUCTION.format(
        shared_instructions=shared_instructions,
        diagnostic_instruction=diagnostic_instruction,
    )

    token_limit = get_token_limit_instruction()
    response_length_inst = get_response_length_instruction(
        "guide", state.get("response_length", "normal"),
    )
    system_content = instruction + "\n\n" + token_limit + "\n\n" + response_length_inst
    system_content += get_component_recommendation_context(state.get("component_recommendation"))

    messages = state.get("messages", [])
    project_context = state.get("project_context", "")
    student_profile_text = state.get("student_profile_text", "")

    guide_context = f"""
{student_profile_text}
**Student Question:**
{messages[-1].content if messages else 'N/A'}
**Project Context:**
{project_context}
**Your Task:**
Ask ONE probing question to understand the student's starting point.
Do NOT provide scaffolding or solutions yet.
"""

    llm = get_chat_llm()

    result = await llm.ainvoke([
        SystemMessage(content=system_content),
        HumanMessage(content=guide_context),
    ])

    diagnostic_count = state.get("diagnostic_question_count", 0)

    return {
        "response": str(result.content),
        "student_has_provided_initial_thoughts": False,
        "diagnostic_question_count": diagnostic_count + 1,
        "content_metadata": {"node_executed": "understand_starting_point"},
    }
