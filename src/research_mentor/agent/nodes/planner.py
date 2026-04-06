"""Planner node — creates a teaching methodology plan using the selected persona."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from research_mentor.agent.prompts.shared import (
    get_response_length_instruction,
    get_token_limit_instruction,
)
from research_mentor.agent.state import MentorState
from research_mentor.db.crud import get_persona_by_id
from research_mentor.llm import structured_call
from research_mentor.utils.memory_profiler import memory_profile_node


class ResearchGuidancePlanOutput(BaseModel):
    """Structured output for the research guidance plan."""

    approach: str = Field(max_length=800, description="How this scientist would approach this guidance type")
    first_step: str = Field(max_length=500, description="First action in their methodology")
    teaching_strategy: str = Field(max_length=500, description="Why this strategy fits their research approach")
    persona: str = Field(max_length=150, description="Scientist/approach name")


PLANNER_INSTRUCTION = """You are a Research Methodology Planning Agent.

**IMPORTANT CONTEXT:**
- Current date: {current_date}
{persona_context}

**Your Role:** {role_description}

**CRITICAL: Guidance type is ALREADY DECIDED**
- Create a plan for HOW this persona would deliver the specified guidance_type
- Do NOT decide the guidance type - it's provided to you

**We Combine Modern Pedagogy with Research Approach:**

- **Guidance type** (scaffolding/guided_discovery/pure_socratic/reflection) comes from
  modern research mentoring pedagogy - what we know works for student development
- **Methodology** (HOW to deliver that guidance) comes from {methodology_source}

**Your Task:**

Draw on your knowledge of:
{task_guidance}

From their research approach, infer how they would deliver the SPECIFIED guidance_type to
mentor a student on THIS specific research project in {current_year}.

**PLANNING APPROACH:**
- Use modern language and contemporary teaching methods adapted to {current_year}
- Apply the persona's THINKING METHODOLOGY (how they investigated), not historical limitations
- Account for modern tools (computers, databases, software, internet)
- Adapt systematic research approach to current practices (e.g., use Python for modeling, not hand-calculation)

**Student State:**
- Engagement: Thoughtful questions (engaged) vs one-word responses (minimal) vs demands (disrespectful)
- Knowledge: Fundamental gaps (lost) vs specific questions (partial) vs thoughtful analysis (progressing)

**Output (reference student's specific project in ALL fields):**
- **APPROACH**: 1-2 sentences applying persona's methodology to this project
- **FIRST STEP**: Specific, actionable ("Search for...", "Delegate to [scaffolding/guided_discovery/pure_socratic/reflection] to...", or direct questions)
- **TEACHING STRATEGY**: Why this fits student state and persona's approach
"""

# Role descriptions
ROLE_HISTORICAL = "Think AS the specified historical scientist and create a plan for how to apply their research methodology to the specified educational guidance type."
ROLE_MODERN = "Think using the specified contemporary research approach and create a plan for how to apply that methodology to the specified educational guidance type."

# Persona-specific context
PLANNER_CONTEXT_HISTORICAL = """- You are planning as a historical scientist mentoring a MODERN student in {current_year}
- The student has access to modern tools (computers, databases, software, internet)
- Apply the historical scientist's research methodology while understanding modern context
- The historical scientist is acting as a mentor in the present day, not planning from their historical period"""

PLANNER_CONTEXT_MODERN = """- You are planning using the research approach of leading contemporary researchers
- The persona represents how world-class current researchers in that field work
- Draw on your deep knowledge of how leading researchers in this field approach problems and guide students"""

# Methodology source descriptions
METHODOLOGY_SOURCE_HISTORICAL = "the historical scientist's research approach documented in their published works, papers, correspondence, and historical records"
METHODOLOGY_SOURCE_MODERN = "contemporary research methodology and thinking approach in that field"

# Task guidance descriptions
TASK_GUIDANCE_HISTORICAL = """- Research methodology and approach to discovery (from their published works, papers, and records)
- Problem-solving style and investigative methods
- How they built knowledge and structured their inquiries"""

TASK_GUIDANCE_MODERN = """- Contemporary research methodology and approach to discovery in that field
- Problem-solving style and investigative methods used by leading current researchers
- How they build knowledge and structure their inquiries"""


@memory_profile_node("planner")
async def create_research_guidance_plan(state: MentorState) -> dict[str, Any]:
    """Create a research guidance plan based on persona and guidance type."""
    logger.info("Planner: Creating research guidance plan...")

    persona_id = state.get("persona", "newton")
    guidance_type = state.get("guidance_type", "guided_discovery")
    confidence = state.get("confidence", 0.0)
    student_progress = state.get("student_progress", {})
    project_context = state.get("project_context", "")

    persona_details = await get_persona_by_id(persona_id)
    if not persona_details:
        raise ValueError(
            f"Persona '{persona_id}' not found in database. Run 'research-mentor init'."
        )

    persona_type = persona_details["personaType"]
    persona_full_name = persona_details["fullName"]

    # Get current date and year for temporal context
    now = datetime.now()
    current_date = now.strftime("%B %d, %Y")
    current_year = now.year

    # Select context, role, methodology, and guidance based on persona type
    if persona_type == "modern":
        persona_context = PLANNER_CONTEXT_MODERN
        role_description = ROLE_MODERN
        methodology_source = METHODOLOGY_SOURCE_MODERN
        task_guidance = TASK_GUIDANCE_MODERN
    else:
        persona_context = PLANNER_CONTEXT_HISTORICAL.format(current_year=current_year)
        role_description = ROLE_HISTORICAL
        methodology_source = METHODOLOGY_SOURCE_HISTORICAL
        task_guidance = TASK_GUIDANCE_HISTORICAL

    formatted_instruction = PLANNER_INSTRUCTION.format(
        current_date=current_date,
        current_year=current_year,
        persona_context=persona_context,
        role_description=role_description,
        methodology_source=methodology_source,
        task_guidance=task_guidance,
    )

    token_limit = get_token_limit_instruction()
    response_length_inst = get_response_length_instruction(
        "planner", state.get("response_length", "normal"),
    )
    formatted_instruction = formatted_instruction + "\n\n" + token_limit + "\n\n" + response_length_inst

    messages = state.get("messages", [])
    latest_msg = messages[-1].content if messages else "No message"

    planning_context = f"""
**Persona:** {persona_full_name}
**Guidance Type (ALREADY DECIDED):** {guidance_type}
**Confidence in Guidance Type:** {confidence:.2f}

**Student Question:**
"{latest_msg}"

**Student Progress:**
{student_progress}

**Project Context:**
{project_context}

**Your Task:**
Create a research methodology guidance plan for how {persona_full_name} would deliver {guidance_type} guidance to this student.
Return a structured ResearchGuidancePlanOutput object.
"""

    plan = await structured_call(
        ResearchGuidancePlanOutput,
        [
            SystemMessage(content=formatted_instruction),
            HumanMessage(content=planning_context),
        ],
        thinking="medium",
    )

    plan_dict = plan.model_dump()
    logger.info("Plan created: approach={}", plan_dict.get("approach", "")[:80])
    return {"research_guidance_plan": plan_dict}
