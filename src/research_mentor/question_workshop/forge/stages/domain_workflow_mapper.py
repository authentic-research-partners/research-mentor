"""Stage 1: Domain Workflow Mapper

Single responsibility: Map the research workflow in a given field and identify pain points.

Given a field and tool category, identifies the key stages of the research workflow,
current tools in use, and bottlenecks where time/effort/money concentrates.

Pattern: Chat LLM + Structured Extraction
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.forge.schemas import (
    TOOL_CATEGORY_LABELS,
    WorkflowMap,
)

WORKFLOW_MAPPING_PROMPT = """You are a research methodology expert who deeply understands the day-to-day workflows of scientific researchers.

Your task: Given a scientific field and a tool category, map the full research workflow — what researchers actually DO, step by step — and identify where time, effort, or money concentrates disproportionately.

## What to Map

For each research field, trace the complete workflow from question to publication:
1. **Literature review & planning** — how researchers find and organize prior work
2. **Sample/data collection** — how they gather raw materials or data
3. **Preparation & processing** — how samples are prepared or data cleaned
4. **Measurement & acquisition** — how key variables are measured or recorded
5. **Analysis** — how raw data becomes results
6. **Visualization & interpretation** — how results are presented and understood
7. **Writing & communication** — how findings are shared

Not every field uses every stage. Omit irrelevant stages, add field-specific ones.

## Pain Point Identification

For each workflow stage, ask:
- How long does this take relative to its intellectual value?
- What is done manually that could be automated?
- What requires expensive equipment that could be replaced with simpler alternatives?
- What is error-prone due to human factors?
- What creates bottlenecks that slow the entire pipeline?
- What prevents students or under-resourced labs from doing this research?

## Tool Category Focus

Focus your pain point identification on bottlenecks addressable by: {tool_category_label}

If the category is "any type of research tool", identify the highest-impact bottleneck regardless of tool type.

## Key Principle

Think about what researchers ACTUALLY struggle with — not what textbooks describe as the process. The gap between the textbook version and reality is where tool opportunities live.

Focus especially on:
- **Tedious manual work** that researchers complain about
- **Expensive steps** that exclude under-resourced labs
- **Error-prone processes** where human judgment introduces variability
- **Format/compatibility issues** between different tools or stages
- **Missing standardization** that makes results hard to compare across labs

## PRIORITY: Underserved Niches Over Obvious Gaps

**IMPORTANT:** Do NOT list the most well-known pain points that every methods review already describes. Those have existing tools. Instead, look for:

- **Niche sub-workflows** that major tool developers ignore because the market is too small (e.g., not "image analysis" but "counting trichomes on leaf cross-sections under polarized light")
- **Interface gaps** between two existing tools where researchers waste time on manual format conversion, copy-pasting, or re-entry
- **Field-specific quirks** that generic tools handle badly (e.g., a general-purpose timer doesn't account for temperature-dependent reaction rates)
- **Student/teaching lab gaps** — tools designed for professional labs that don't have simplified versions for educational settings
- **Emerging method bottlenecks** — new techniques (CRISPR screens, single-cell RNA-seq, environmental DNA) where the tooling hasn't caught up to the method

**Avoid:** Generic pain points that already have mature solutions: "data visualization" (matplotlib, R), "statistical analysis" (SPSS, R), "literature search" (Google Scholar, Scopus), "image analysis" (ImageJ/FIJI), "DNA sequence analysis" (BLAST). If a well-known free tool solves it, it's not a real bottleneck.
"""


async def map_domain_workflow(
    field: str,
    tool_category: str,
    seed_idea: str | None = None,
    student_profile_text: str = "",
) -> dict[str, Any]:
    """Stage 1: Map the research workflow and identify pain points.

    Args:
        field: Scientific field (physics, chemistry, biology).
        tool_category: Type of tool to focus on.
        seed_idea: Optional user seed idea.
        student_profile_text: Pre-built student profile context.

    Returns:
        Dict with research_stages, pain_points, current_tools, field_context.
    """
    logger.info(
        "Stage 1: Domain Workflow Mapping - field={}, category={}, seed={}",
        field, tool_category, seed_idea[:50] if seed_idea else None,
    )

    tool_category_label = TOOL_CATEGORY_LABELS.get(tool_category, TOOL_CATEGORY_LABELS["any"])
    prompt_text = WORKFLOW_MAPPING_PROMPT.format(tool_category_label=tool_category_label)
    if student_profile_text:
        prompt_text = f"{student_profile_text}\n\n{prompt_text}"

    # Build user prompt
    field_cap = field.capitalize()
    if seed_idea:
        user_prompt_text = f"""Field: {field_cap}
Tool Category: {tool_category_label}
User's seed idea: "{seed_idea}"

Map the research workflow in {field}, paying special attention to the area related to the user's idea. Identify 3-5 pain points where a new {tool_category_label} tool would have the highest impact."""
    else:
        user_prompt_text = f"""Field: {field_cap}
Tool Category: {tool_category_label}

Map the complete research workflow in {field}. Identify 3-5 pain points where a new {tool_category_label} tool would have the highest impact for researchers."""

    # Step 1: Creative exploration with chat LLM
    llm = get_chat_llm()
    response = await llm.ainvoke([
        SystemMessage(content=prompt_text),
        HumanMessage(content=user_prompt_text),
    ])
    raw_response = response.content

    logger.debug("Workflow mapping response length: {} chars", len(raw_response))

    # Step 2: Extract structured data
    extraction_msg = SystemMessage(
        content="Extract the structured workflow map from the research workflow analysis.",
    )
    extraction_user = HumanMessage(content=f"""Extract structured data from this workflow analysis:

{raw_response}

Field: {field_cap}
Tool focus: {tool_category_label}""")

    workflow_map = await structured_call(
        WorkflowMap, [extraction_msg, extraction_user],
        thinking="medium",
    )

    result: dict[str, Any] = {
        "research_stages": workflow_map.research_stages,
        "pain_points": workflow_map.pain_points,
        "current_tools": workflow_map.current_tools,
        "field_context": workflow_map.field_context,
        "raw_response": raw_response,
    }

    logger.info(
        "Stage 1 complete: stages={}, pain_points={}",
        len(result["research_stages"]), len(result["pain_points"]),
    )
    return result
