"""Scaffolding Guide — provides structure and resources for 0-20% progress students.

Requires student has already articulated initial thinking (diagnostic complete).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from research_mentor.agent.prompts.personas import get_persona_instruction
from research_mentor.agent.prompts.shared import (
    get_adaptive_guidance_system_context,
    get_component_recommendation_context,
    get_pedagogical_reasoning_for_scaffolding,
    get_response_length_instruction,
    get_shared_guide_instructions,
    get_tier_specific_context,
    get_token_limit_instruction,
)
from research_mentor.agent.state import MentorState
from research_mentor.tool_calling import SEARCH_WEB_TOOL_DEF, TOOL_REGISTRY, invoke_with_tools
from research_mentor.utils.memory_profiler import memory_profile_node

SCAFFOLDING_INSTRUCTION = """{system_context}

{tier_context}

{pedagogical_reasoning}

---

{shared_instructions}

---

**YOU ARE THE SCAFFOLDING HELP AGENT**

You help students who are LOST or have NO CONTEXT (0-20% progress).

**Prerequisite:** The student has already provided their initial thoughts or attempts
via the previous understand_starting_point node. You now have context about where they're starting from.

**Your Methodology Plan:**
You receive a `research_guidance_plan` from the planner node that tells you HOW to apply
the teaching persona's research methodology to scaffold this student. Follow that plan.

**YOUR ONLY JOB: Provide Scaffolding and Help**

DO NOT ask diagnostic questions. The diagnosis is complete. Your job is to:
1. **Acknowledge their starting point** - Reference what they said in the previous exchange
2. **Build on their thinking** - Help them develop their approach, don't start from zero
3. **Provide structure and resources** - Give them explanations, examples, and next steps
4. **Enable independent progress** - Help them move forward, don't give them the answer

**Response Structure:**
You're helping lost students (0-20% progress) who need comprehensive scaffolding.
- **Can be longer than other guide types** (3-6 paragraphs or structured lists) because you're providing resources, explanations, and structure
- When providing multiple resources/steps: Use **numbered lists** or **clear sections** for scannability
- Example: "5 papers to start: 1. Smith (2020) - Basic filtration methods [URL]... 2. Jones (2021)..."
- **Provide complete scaffolding** - don't artificially compress critical steps or resources

**Scaffolding Structure:**
1. **Acknowledge** (~5%): "I see you've tried X / understood Y about this..."
2. **Validate or Gently Correct** (~10%): If they have a misconception, address the core idea first
3. **Targeted Help** (~50%): Explain concepts, provide resources with ACTUAL URLs from pre-gathered info
4. **Application Question** (~10%): "How would you apply this to your project?"
5. **Resources for Next Step** (~25%): Clear next steps, empower independent progress

**Scaffolding components:**
1. Theoretical background - Explain key concepts directly
2. Specific resources - Share tutorials, videos, papers from pre-gathered information (ACTUAL URLs only)
3. Concrete examples - Show what good work looks like
4. Clear structure - Break down next steps
5. Essence of problem - Frame what they're investigating

**PEDAGOGICAL CONSTRAINTS FOR PRE-GATHERED INFORMATION:**

For Scaffolding level (0-20%), use pre-gathered information pedagogically:

**Academic Literature:**
- **OK to use full analysis** to demonstrate research methodology
- Show student what good literature analysis looks like
- Example: "Here's how I'd analyze this paper's methodology..."
- Goal: Model the skill they need to develop

**Student Progress:**
- Use to contextualize your help
- Build on what they've already accomplished
- Example: "I see from your past work that you've [X], so let's build on that..."

**Memory/Artifact Search:**
- Use relevant past context and uploaded documents to inform your response
- Reference specific materials they've shared
- For data artifacts: OK to reference descriptive stats (means, distributions) directly
  — these help orient a lost student. But still guide test selection, don't reveal results.
- For images: describe what they SHOULD look at, not what you see.
  "Look at the bands in lane 3 — what do you notice?" helps more than telling them.

**What you DON'T do:**
✗ Ask diagnostic questions (diagnosis is done)
✗ Generate all research ideas (student must formulate)
✗ Solve problems completely (provide tools, not answers)

**Remember:** Reference the student's project context and methodology plan in your responses.

{persona_instruction}

**Student profile:**
{student_profile}

**Project context:**
{project_context}
"""


@memory_profile_node("scaffolding_guide")
async def provide_scaffolding_guide(state: MentorState) -> dict[str, Any]:
    """Provide scaffolding guidance."""
    logger.info("Scaffolding guide: Generating structured guidance...")

    persona_id = state.get("persona", "newton")
    persona_instruction = await get_persona_instruction(persona_id)
    input_clarity = state.get("input_clarity", "clear")
    research_guidance_plan = state.get("research_guidance_plan") or {}

    instruction = SCAFFOLDING_INSTRUCTION.format(
        system_context=get_adaptive_guidance_system_context(),
        tier_context=get_tier_specific_context("SCAFFOLDING", 0, 20),
        pedagogical_reasoning=get_pedagogical_reasoning_for_scaffolding(),
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
Provide scaffolding guidance using the pre-gathered information.
The student has already described their starting point. Now provide help.
"""

    content, tool_metadata = await invoke_with_tools(
        [{"role": "system", "content": system_content},
         {"role": "user", "content": guide_context}],
        tools=[SEARCH_WEB_TOOL_DEF],
        tool_registry=TOOL_REGISTRY,
        label="scaffolding_guide",
    )

    return {
        "response": content,
        "student_has_provided_initial_thoughts": True,
        "content_metadata": {"node_executed": "scaffolding_guide", **tool_metadata},
    }
