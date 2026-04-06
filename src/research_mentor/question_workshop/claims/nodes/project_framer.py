"""Stage 4: Project Framer Node

Single responsibility: Frame an open-ended research direction for the student.

Given the claim and gap analysis, creates:
- A catchy title
- Description connecting claim to investigable phenomenon
- Open-ended investigation directive (NOT a procedure)
- Project scope and type

Pattern: Direct structured_call (single-step generation)
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.claims._config import claims_config
from research_mentor.question_workshop.claims.schemas import (
    EvidenceLevel,
    ResearchDirection,
)

PROJECT_FRAMING_PROMPT = """You are an expert at framing research directions \
for students.

## Your Goal

Transform a media claim into an open-ended research direction that:
- Connects to the original claim but investigates the UNDERLYING PHENOMENON
- Is appropriate for a 2-6 month student project
- Is open-ended (student decides methodology)
- Does NOT prescribe specific procedures

## Output Format

### Title (2-4 words)
Catchy, memorable, hints at the investigation.

GOOD: "Lemon Water Chemistry", "Screen Light Effects", "Organic Nutrient Comparison"
BAD: "Testing the Lemon Water Claim" (prescriptive), "Investigation of Health Effects" \
(generic)

### Description (2-4 sentences)
Structure:
1. What media claims (the contested idea)
2. The scientific gap or question
3. What's actually investigable

### Investigation Directive
MUST start with: "Investigate...", "Explore...", "Study...", or "Examine..."

Focus on PHENOMENA and PARAMETERS, not procedures:
GOOD: "Investigate the chemistry of lemon water and its interaction with acids."
BAD: "Step 1: Get pH strips. Step 2: Measure the pH." (procedure, not direction)

### Project Type
- **experimental:** Lab-based investigation
- **data_analysis:** Analysis of public datasets
- **survey:** Survey-based correlation study

### Underlying Phenomenon
The real scientific phenomenon that can be investigated, even if the original \
claim can't be directly tested.

## Important

- Frame what's INVESTIGABLE, not what's claimed
- Student decides HOW to investigate — you frame WHAT
- 2-6 month timeline, high school appropriate equipment
- **Stay in the same field as the original claim.** If the claim is about \
physics, the project MUST investigate a physics phenomenon (not pivot to \
psychology or surveys). If the claim is about chemistry, frame a chemistry \
investigation. The project type should match: physics → experimental \
(measuring physical quantities), chemistry → experimental (measuring chemical \
properties), biology → experimental or data_analysis.
"""


async def project_framer(
    claim_data: dict[str, Any],
    evidence_result: dict[str, Any],
    gap_analysis: dict[str, Any],
    student_profile_text: str = "",
) -> dict[str, Any]:
    """Stage 4: Frame the research direction.

    Args:
        claim_data: Output from claim_extractor.
        evidence_result: Output from evidence_searcher.
        gap_analysis: Output from gap_analyzer.
        student_profile_text: Rendered student profile context.

    Returns:
        Dict with keys: research_direction.
    """
    logger.info("Stage 4: Project Framing")

    original_text = claim_data.get("original_text", "")
    field = claim_data.get("field", "unknown")
    independent_var = claim_data.get("independent_var", "")
    dependent_var = claim_data.get("dependent_var", "")

    evidence_level = evidence_result.get("level", EvidenceLevel.NOT_SUPPORTED)
    what_science_says = evidence_result.get("what_science_says", "")
    related_science = evidence_result.get("related_science", "")

    why_hard = gap_analysis.get("why_hard_to_test", [])
    key_insight = gap_analysis.get("key_insight", "")

    profile_section = f"\n\n{student_profile_text}" if student_profile_text else ""
    user_prompt = f"""Create a research direction for a student project based \
on this claim:{profile_section}

ORIGINAL CLAIM: "{original_text}"
FIELD: {field}
VARIABLES: {independent_var} -> {dependent_var}

EVIDENCE LEVEL: {evidence_level}
WHAT SCIENCE SAYS: {what_science_says}
RELATED SCIENCE THAT EXISTS: {related_science or 'None identified'}

WHY THE EXACT CLAIM IS HARD TO TEST:
{chr(10).join(f'- {w}' for w in why_hard)}

KEY INSIGHT: {key_insight}

Based on this, frame a research direction that:
1. Connects to the original claim
2. Investigates an UNDERLYING PHENOMENON students can actually study
3. Is open-ended (doesn't prescribe specific methods)
4. Is appropriate for 2-6 months of work
5. Uses equipment accessible to high school students
6. **STAYS IN THE {field.upper()} FIELD** — the project must investigate a \
{field} phenomenon using {field} methods and measurements. Do NOT pivot to \
psychology, surveys, or a different scientific discipline.

Remember: Students decide HOW to investigate. You frame WHAT is worth investigating.
"""

    research_direction = await structured_call(
        ResearchDirection,
        [
            SystemMessage(content=PROJECT_FRAMING_PROMPT),
            HumanMessage(content=user_prompt),
        ],
        thinking="medium",
        temperature=claims_config().framing_temperature,
    )

    result = {
        "research_direction": research_direction.model_dump(),
    }

    logger.info(
        "Stage 4 complete: title='{}', type={}, duration={}",
        research_direction.title, research_direction.project_type,
        research_direction.project_duration,
    )

    return result
