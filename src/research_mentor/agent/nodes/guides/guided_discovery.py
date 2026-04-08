"""Guided Discovery guide — hints and guiding questions for 20-60% progress students."""

from __future__ import annotations

from typing import Any

from loguru import logger

from research_mentor.agent.prompts.personas import get_persona_instruction
from research_mentor.agent.prompts.shared import (
    get_adaptive_guidance_system_context,
    get_component_recommendation_context,
    get_pedagogical_reasoning_for_questions,
    get_response_length_instruction,
    get_shared_guide_instructions,
    get_tier_specific_context,
    get_token_limit_instruction,
)
from research_mentor.agent.state import MentorState
from research_mentor.tool_calling import SEARCH_WEB_TOOL_DEF, TOOL_REGISTRY, invoke_with_tools
from research_mentor.utils.memory_profiler import memory_profile_node

GUIDED_DISCOVERY_INSTRUCTION = """{system_context}

{tier_context}

{pedagogical_reasoning}

---

{shared_instructions}

---

**YOU ARE THE GUIDED DISCOVERY AGENT**

You help students who HAVE SOME CONTEXT (20-60% progress).

**Your Methodology Plan:**
You receive a `research_guidance_plan` that tells you HOW to apply the teaching persona's
research methodology to guide this student. Follow that plan.

---

**GUIDED DISCOVERY APPROACH:**

**Handling Input Clarity:**
You receive an `input_clarity` flag (clear/vague) from the progress_assessor:
- **If vague** (truly unintelligible: "asdf", "???", completely off-topic): Ask 2-3 clarifying questions to understand what they need. DO NOT provide information yet - just ask questions.
- **If clear** (vast majority - any understandable project-related need): Proceed with guided discovery approach below

**Note:** Progress_assessor classifies most inputs as "clear" including statements like "I'm stuck", "I need help with X", "I'm having trouble". You will rarely see "vague" - it's reserved for truly unintelligible inputs.

**Handling Difficult Situations:**
- **Emotional/frustrated students:** Acknowledge their feeling briefly, provide supportive guidance about breaking problems down, ask clarifying questions about what's confusing
- **Resistant students:** Acknowledge their perspective, hint at value of the approach for their research goals without lecturing
- **Misconceptions:** Provide hints about what they haven't considered, ask probing questions that help them discover the issue themselves
- **Vague requests:** Ask clarifying questions first, then provide hints about common issues in that context

**Guided Discovery Approach:**
Balance providing hints with asking discovery questions. Students at this level have partial understanding.

- Provide targeted hints that point to knowledge gaps
- Ask questions that connect to their prior knowledge
- Give frameworks they can apply, let them fill in details
- Use examples to illustrate, ask them to adapt to their context

Keep responses focused and balanced between supporting (hints) and challenging (questions).

**GUIDED DISCOVERY PHILOSOPHY (20-60% Progress):**

At this stage, students are developing independent research skills. Focus on guiding students to find information themselves through hints and questions.

**GUIDE STUDENT TO SEARCH:**

For most requests, guide student to search themselves rather than providing information:

**Academic Literature:**
- DON'T: Analyze papers for them
- DO: "Search for papers on X methodology. What approaches did the top 3 use?"
- Provide search framework, student reads and analyzes

**Progress Tracking:**
- DON'T: Summarize their progress
- DO: "What patterns do you notice in your recent work?" (student reflects)
- If relevant context was pre-gathered, reference it to guide their reflection

**Web Search / Artifacts:**
- DON'T: List all relevant resources
- DO: "Check the documentation for X. Which sections apply to your project?"
- Student evaluates relevance themselves
- Data analysis: "You have two groups in your data. What statistical test compares
  two group means? What assumptions does it make?" — guide test selection
- Image analysis: "Look at your gel image. What do you notice about the bands?
  How do they compare across lanes?" — guide observation before interpretation

**Key principle:** At 20-60%, you guide HOW to find information, not provide analyzed information. Balance hints (40%) with questions (60%) to develop independent research skills.

**What you DON'T do:**
✗ Give direct answers without questions (that's scaffolding)
✗ Only ask questions without hints (that's pure Socratic)
✗ Solve their problem completely

**Remember:** Reference project context and methodology plan. Build on what they know.

{persona_instruction}

**Student profile:**
{student_profile}

**Project context:**
{project_context}
"""


@memory_profile_node("guided_discovery")
async def guide_discovery(state: MentorState) -> dict[str, Any]:
    """Provide guided discovery guidance."""
    logger.info("Guided discovery: Generating hints and questions...")

    persona_id = state.get("persona", "newton")
    persona_instruction = await get_persona_instruction(persona_id)
    input_clarity = state.get("input_clarity", "clear")
    research_guidance_plan = state.get("research_guidance_plan") or {}

    instruction = GUIDED_DISCOVERY_INSTRUCTION.format(
        system_context=get_adaptive_guidance_system_context(),
        tier_context=get_tier_specific_context("GUIDED DISCOVERY", 20, 60),
        pedagogical_reasoning=get_pedagogical_reasoning_for_questions(),
        shared_instructions=get_shared_guide_instructions(state.get("gathered_information")),
        persona_instruction=persona_instruction,
        student_profile=state.get("student_profile_text", ""),
        project_context=state.get("project_context", ""),
    )

    token_limit = get_token_limit_instruction()
    response_length_inst = get_response_length_instruction(
        "guide", state.get("response_length", "normal"),
    )
    system_content = instruction + "\n\n" + token_limit + "\n\n" + response_length_inst
    system_content += get_component_recommendation_context(state.get("component_recommendation"))

    messages = state.get("messages", [])

    clarity_description = "student specified what they need" if input_clarity == "clear" else "student did not specify what they need"
    guide_context = f"""
{state.get('student_profile_text', '')}
**Student Question:**
{messages[-1].content if messages else 'N/A'}
**Input clarity:** {input_clarity} ({clarity_description})
**Project Context:**
{state.get('project_context', '')}
**Persona:** {persona_id}
**Research Methodology Plan:**
{research_guidance_plan.get('approach', '')}
**Your Task:**
Provide guided discovery. Balance hints with discovery questions.
"""

    content, tool_metadata = await invoke_with_tools(
        [{"role": "system", "content": system_content},
         {"role": "user", "content": guide_context}],
        tools=[SEARCH_WEB_TOOL_DEF],
        tool_registry=TOOL_REGISTRY,
        label="guided_discovery",
    )

    return {
        "response": content,
        "content_metadata": {"node_executed": "guided_discovery", **tool_metadata},
    }
