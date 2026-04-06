"""Stage 4: Feasibility & Impact Scorer

Single responsibility: Score each tool concept on buildability, novelty, impact, and
validation path. Identify the recommended starting point.

Pattern: Structured LLM (direct extraction)
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.forge.schemas import ScoredTools

SCORING_PROMPT = """You are an expert at evaluating research tool proposals for student projects.

Your task: Score each tool concept on 4 dimensions (1-10 scale) and recommend a starting point.

## Scoring Dimensions

### 1. Student Buildable (1-10)
Can a student at this level actually build this?

- **10:** Can be built in an afternoon with basic skills (simple measurement jig, data form)
- **7-9:** Can be built in 1-4 weeks with moderate skills (Python script, Arduino sensor)
- **4-6:** Requires 1-3 months and specialized skills (software package, instrument modification)
- **1-3:** Requires expert-level skills or resources beyond student access

### 2. Novelty (1-10)
Does this tool already exist, or is it genuinely new?

- **10:** Nothing like this exists — completely new capability
- **7-9:** Similar tools exist but not for this specific use case or at this price point
- **4-6:** Commercial tools exist but are expensive/inaccessible; this is an accessible alternative
- **1-3:** Free, mature, widely-used tools already solve this problem well

### 3. Impact (1-10)
How many researchers would benefit if this tool existed?

- **10:** Every researcher in this field would use it daily
- **7-9:** Many labs would benefit; solves a widely-recognized pain point
- **4-6:** A subset of researchers would benefit significantly
- **1-3:** Very niche; only a few specific labs would use it

### 4. Validation Path (1-10)
How easy is it to prove the tool works correctly?

- **10:** Simple comparison with gold standard (count manually vs. tool count)
- **7-9:** Clear validation protocol (compare accuracy, speed, cost with existing method)
- **4-6:** Validation possible but requires effort (user study, expert review)
- **1-3:** No clear way to validate correctness; success is subjective

## Overall Feasibility

Based on scores:
- **HIGH:** Student buildable >=7, validation path >=7, no scores below 4
- **MEDIUM:** Mixed scores, some dimensions 4-6
- **LOW:** Student buildable <4 OR validation path <4

## Recommendation

- **YES:** At least one tool has HIGH feasibility and impact >=6
- **WITH_MODIFICATIONS:** Tools have potential but need scope adjustments
- **NO:** All tools are beyond student capability or already exist

## Recommended Starting Point

Choose the single best tool to build first — the one with the best combination of:
1. Student buildable (can they actually do it?)
2. Validation path (can they prove it works?)
3. Impact (will it matter?)

Explain WHY this is the best starting point in 1-2 sentences.
"""


async def score_feasibility_impact(
    field: str,
    tool_concepts: dict[str, Any],
    existing_titles: list[str],
    student_profile_text: str = "",
) -> dict[str, Any]:
    """Stage 4: Score tool concepts on feasibility and impact.

    Args:
        field: Scientific field.
        tool_concepts: Output from Stage 3 (tool concepts).
        existing_titles: Previously generated tool names (for novelty awareness).
        student_profile_text: Pre-built student profile context.

    Returns:
        Dict with scored_tools, recommended_starting_point, recommended.
    """
    logger.info(
        "Stage 4: Feasibility & Impact Scoring - field={}, tools={}",
        field, len(tool_concepts.get("tools", [])),
    )

    prompt_text = SCORING_PROMPT
    if student_profile_text:
        prompt_text = f"{student_profile_text}\n\n{prompt_text}"

    # Format tool concepts for scoring
    tools_text = ""
    for i, t in enumerate(tool_concepts.get("tools", []), 1):
        tools_text += (
            f"\n### Tool {i}: {t['name']}\n"
            f"- **Problem:** {t['problem_addressed']}\n"
            f"- **Solution:** {t['solution_design']}\n"
            f"- **MVP:** {t['mvp_spec']}\n"
            f"- **Skills needed:** {', '.join(t.get('required_skills', []))}\n"
            f"- **Resources needed:** {', '.join(t.get('required_resources', []))}\n"
        )

    existing_note = ""
    if existing_titles:
        existing_note = (
            f"\n\n**Previously generated tools** (check novelty against these): "
            f"{', '.join(existing_titles)}"
        )

    user_prompt_text = f"""Field: {field.capitalize()}

## Tool Concepts to Score
{tools_text}{existing_note}

Score each tool on all 4 dimensions (1-10), assess overall feasibility, and recommend the best starting point."""

    messages = [
        SystemMessage(content=prompt_text),
        HumanMessage(content=user_prompt_text),
    ]

    scored = await structured_call(ScoredTools, messages, thinking="medium")

    result: dict[str, Any] = {
        "scored_tools": [s.model_dump() for s in scored.scored_tools],
        "recommended_starting_point": scored.recommended_starting_point,
        "recommended": scored.recommended,
    }

    logger.info(
        "Stage 4 complete: {} tools scored, recommended={}",
        len(result["scored_tools"]), result["recommended"],
    )
    return result
