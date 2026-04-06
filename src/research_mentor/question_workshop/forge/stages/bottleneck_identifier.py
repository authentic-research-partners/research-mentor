"""Stage 2: Bottleneck Identifier

Single responsibility: Identify 3-5 concrete capability gaps from the workflow map.

Takes the workflow map from Stage 1 and identifies specific bottlenecks where
a new tool would have the highest impact, prioritizing gaps that students
could realistically address.

Pattern: Chat LLM + Structured Extraction
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.forge.schemas import (
    TOOL_CATEGORY_LABELS,
    BottleneckAnalysis,
)

BOTTLENECK_PROMPT = """You are a research infrastructure analyst. You identify specific, concrete capability gaps in scientific workflows — places where a new tool would save researchers significant time, money, or effort.

## Your Task

Given a mapped research workflow with identified pain points, drill down to identify 3-5 SPECIFIC bottlenecks that a student could address by building a new tool.

## What Makes a Good Bottleneck

A good bottleneck for tool development is:
- **Specific:** Not "data analysis is hard" but "converting between FASTQ and FASTA formats requires manual command-line steps that non-computational biologists can't do"
- **Measurable:** You can quantify the problem (hours wasted, error rates, cost)
- **Addressable:** A student could realistically build something that helps
- **Impactful:** Multiple researchers face this problem, not just one lab

## What to Avoid

- Problems that require massive infrastructure (cloud computing platforms, enterprise software)
- Problems that are already well-solved by mature FREE tools (ImageJ, R, BLAST, Python/scipy, Arduino IDE)
- Problems that require domain expertise beyond what a student would have
- Problems where the bottleneck is fundamentally about funding, not tools
- GENERIC bottlenecks like "data analysis takes time" or "visualization is hard" — these have hundreds of existing tools

## PRIORITY: Gaps Where NO Good Tool Exists

The most valuable bottleneck is one where researchers currently use WORKAROUNDS — not where they use a tool they wish were better, but where they have NO tool and resort to:
- Manual spreadsheets and hand-counting
- Copy-pasting between incompatible software
- Custom one-off scripts that every lab rewrites from scratch
- Expensive commercial tools with no free alternative for student labs
- Paper-based records that should be digital

Ask: "If I searched for a tool that does this, would I find one?" If yes, it's not a real bottleneck. If the answer is "researchers just do it manually" or "every lab writes their own hacky script" — THAT is a real bottleneck.

## Severity Assessment

- **HIGH:** This bottleneck blocks research entirely or wastes >10 hours per project
- **MEDIUM:** This bottleneck slows research noticeably (2-10 hours per project)
- **LOW:** This is an inconvenience but researchers work around it (<2 hours per project)

## Demographics Adaptation

Consider the student's level when assessing which bottlenecks they could address:
- Middle/high school students can build simple measurement tools, data collection apps, basic scripts
- University students can build software packages, instrument modifications, computational pipelines
- The bottleneck should be matched to what the student can realistically tackle
"""


async def identify_bottlenecks(
    field: str,
    tool_category: str,
    workflow_data: dict[str, Any],
    student_profile_text: str = "",
) -> dict[str, Any]:
    """Stage 2: Identify specific capability gaps from the workflow map.

    Args:
        field: Scientific field.
        tool_category: Type of tool to focus on.
        workflow_data: Output from Stage 1 (workflow map).
        student_profile_text: Pre-built student profile context.

    Returns:
        Dict with bottlenecks list.
    """
    logger.info(
        "Stage 2: Bottleneck Identification - field={}, category={}",
        field, tool_category,
    )

    tool_category_label = TOOL_CATEGORY_LABELS.get(tool_category, TOOL_CATEGORY_LABELS["any"])

    prompt_text = BOTTLENECK_PROMPT
    if student_profile_text:
        prompt_text = f"{student_profile_text}\n\n{prompt_text}"

    # Format workflow data for the LLM
    stages_text = "\n".join(f"- {s}" for s in workflow_data.get("research_stages", []))
    pain_text = "\n".join(f"- {p}" for p in workflow_data.get("pain_points", []))
    tools_text = ", ".join(workflow_data.get("current_tools", []))
    context = workflow_data.get("field_context", "")

    user_prompt_text = f"""Field: {field.capitalize()}
Tool Category: {tool_category_label}

## Research Workflow
{stages_text}

## Identified Pain Points
{pain_text}

## Current Tools in Use
{tools_text}

## Field Context
{context}

Identify 3-5 SPECIFIC bottlenecks in this workflow where a new {tool_category_label} tool would have the highest impact. For each bottleneck, be concrete about what capability is missing and why it matters."""

    # Step 1: Creative analysis with chat LLM
    llm = get_chat_llm()
    response = await llm.ainvoke([
        SystemMessage(content=prompt_text),
        HumanMessage(content=user_prompt_text),
    ])
    raw_response = response.content

    logger.debug("Bottleneck analysis response length: {} chars", len(raw_response))

    # Step 2: Extract structured data
    extraction_msg = SystemMessage(
        content="Extract the structured bottleneck analysis from the research workflow analysis.",
    )
    extraction_user = HumanMessage(
        content=f"Extract structured bottleneck data from this analysis:\n\n{raw_response}",
    )

    analysis = await structured_call(
        BottleneckAnalysis, [extraction_msg, extraction_user],
        thinking="medium",
    )

    result: dict[str, Any] = {
        "bottlenecks": [b.model_dump() for b in analysis.bottlenecks],
        "raw_response": raw_response,
    }

    logger.info(
        "Stage 2 complete: {} bottlenecks identified",
        len(result["bottlenecks"]),
    )
    return result
