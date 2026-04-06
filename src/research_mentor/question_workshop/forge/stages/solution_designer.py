"""Stage 3: Solution Designer

Single responsibility: Design 3-5 tool concepts with minimum viable specifications.

Takes bottlenecks from Stage 2 and designs concrete tool concepts, each with
a name, problem description, solution design, MVP spec, required skills, and
required resources.

Pattern: Chat LLM + Structured Extraction
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.forge.schemas import (
    TOOL_CATEGORY_LABELS,
    ToolConcepts,
)

SOLUTION_DESIGN_PROMPT = """You are a research tool designer. You design practical, buildable tools that solve specific problems in scientific workflows.

## Your Task

For each identified bottleneck, design a concrete tool that a student could build. Each tool concept must include a MINIMUM VIABLE VERSION — the simplest thing that would actually work.

## Design Principles

1. **Minimum viable version first.** The MVP should be something a student can build in weeks, not months. Strip away all nice-to-haves. What is the absolute simplest version that solves the core problem?

2. **The student's contribution must be NOVEL.** The tool must do something that no existing free tool does. Building blocks (libraries, APIs) are fine, but the assembled tool must create a NEW capability. "A Python script that does what ImageJ already does" is NOT a valid tool concept. "A Python script that connects ImageJ output to a field-specific statistical model that doesn't exist as a package" IS.

3. **Match the student's level.** A middle schooler's MVP looks different from a university student's:
   - Middle school: Simple measurement device, data collection app, field identification tool
   - High school: Arduino sensor with field-specific calibration, protocol optimizer, niche data converter
   - University: Specialized software package, instrument modification, domain-specific pipeline, integration tool

4. **Concrete specifications.** Don't say "a tool that analyzes data." Say "a Python script that reads CSV files from the Vernier sensor, calculates the rolling average, and plots temperature vs. time with error bars."

5. **Validation built in.** Every tool concept must have a clear way to prove it works — comparison with existing methods, calibration against known standards, or user testing.

## CRITICAL: Novelty Check

Before finalizing each tool concept, ask yourself:
- "Can I Google this and find a free tool that does it?" → If yes, REDESIGN.
- "Is this just a wrapper around an existing library with no new logic?" → If yes, REDESIGN.
- "Would a researcher say 'I already use X for this'?" → If yes, REDESIGN.

Valid novelty comes from:
- **New combinations:** Connecting two existing tools that don't talk to each other
- **Domain specialization:** A generic tool exists but not one tuned to this specific subfield's needs
- **Accessibility shift:** A capability exists in $10K commercial software but not as a free/open tool
- **Workflow integration:** Individual steps have tools, but the pipeline between them is manual
- **Novel measurement:** A physical quantity that researchers want to measure but have no affordable device for

## Tool Naming

Give each tool a memorable, descriptive name (2-5 words). Good names:
- Describe what it does: "ColonySnap Counter", "FieldNote Digitizer"
- Are specific enough to distinguish from generic tools
- Are catchy enough that a student would be excited to build it

## Required Skills & Resources

Be specific and realistic:
- Skills: "Python programming", "basic soldering", "3D printing", "microscopy"
- Resources: "Raspberry Pi ($35)", "3D printer access", "Arduino Uno", "smartphone camera"
"""


async def design_solutions(
    field: str,
    tool_category: str,
    bottleneck_data: dict[str, Any],
    student_profile_text: str = "",
    current_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Stage 3: Design tool concepts for identified bottlenecks.

    Args:
        field: Scientific field.
        tool_category: Type of tool to focus on.
        bottleneck_data: Output from Stage 2 (bottleneck analysis).
        student_profile_text: Pre-built student profile context.
        current_tools: Existing tools in the field (from Stage 1) for novelty checks.

    Returns:
        Dict with tools list (3-5 ToolConcept dicts).
    """
    logger.info(
        "Stage 3: Solution Design - field={}, category={}, bottlenecks={}",
        field, tool_category, len(bottleneck_data.get("bottlenecks", [])),
    )

    tool_category_label = TOOL_CATEGORY_LABELS.get(tool_category, TOOL_CATEGORY_LABELS["any"])

    prompt_text = SOLUTION_DESIGN_PROMPT
    if student_profile_text:
        prompt_text = f"{student_profile_text}\n\n{prompt_text}"

    # Format bottlenecks
    bottleneck_text = ""
    for i, b in enumerate(bottleneck_data.get("bottlenecks", []), 1):
        bottleneck_text += (
            f"\n### Bottleneck {i}: {b['name']}\n"
            f"- **Description:** {b['description']}\n"
            f"- **Affected stage:** {b['affected_stage']}\n"
            f"- **Severity:** {b['severity']}\n"
        )

    existing_tools_text = ""
    if current_tools:
        tools_list = ", ".join(current_tools)
        existing_tools_text = f"""

## Existing Tools Already Used by Researchers
{tools_list}

These tools ALREADY EXIST and are FREE. Your proposed tools MUST NOT duplicate any of these. If a bottleneck is already addressed by one of these tools, your solution must target a specific gap that the existing tool does NOT cover."""

    user_prompt_text = f"""Field: {field.capitalize()}
Tool Category: {tool_category_label}

## Bottlenecks to Address
{bottleneck_text}{existing_tools_text}

Design a concrete tool concept for each bottleneck (3-5 total). Each tool must have:
1. A memorable name
2. Clear problem description
3. High-level solution design
4. Minimum viable version specification
5. Required skills to build it
6. Required resources/materials"""

    # Step 1: Creative design with chat LLM
    llm = get_chat_llm()
    response = await llm.ainvoke([
        SystemMessage(content=prompt_text),
        HumanMessage(content=user_prompt_text),
    ])
    raw_response = response.content

    logger.debug("Solution design response length: {} chars", len(raw_response))

    # Step 2: Extract structured data
    extraction_msg = SystemMessage(
        content=(
            "Extract ALL tool concepts from the design below into structured format. "
            "Every tool must have ALL 6 fields fully populated: name, problem_addressed, "
            "solution_design, mvp_spec, required_skills, required_resources. "
            "Do NOT truncate or skip any tool — extract every single one completely."
        ),
    )
    extraction_user = HumanMessage(content=f"""Extract ALL structured tool concepts from this design:

{raw_response}""")

    concepts = await structured_call(
        ToolConcepts, [extraction_msg, extraction_user],
        thinking="medium",
    )

    result: dict[str, Any] = {
        "tools": [t.model_dump() for t in concepts.tools],
        "raw_response": raw_response,
    }

    logger.info(
        "Stage 3 complete: {} tool concepts designed",
        len(result["tools"]),
    )
    return result
