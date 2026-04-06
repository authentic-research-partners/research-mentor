"""Input Safety Review node — analyzes student input for safety/ethics/communication concerns."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from research_mentor.agent.prompts.shared import build_student_profile_context
from research_mentor.agent.state import MentorState
from research_mentor.llm import structured_call
from research_mentor.utils.memory_profiler import memory_profile_node


class SafetyReviewOutput(BaseModel):
    """Structured output for safety/ethics/communication/wellbeing review."""

    has_concern: bool = Field(
        description="True ONLY if actual hazard/crisis detected "
        "(NOT for normal requests like 'Can you help me?', 'I'm confused', 'I'm stuck')"
    )
    concern_type: Literal["safety", "ethics", "communication", "wellbeing", "none"] = Field(
        default="none",
        description="Type of concern: safety (dangerous procedures), ethics (research "
        "involving people/animals, academic integrity), communication (profanity/abuse), "
        "wellbeing (CRISIS ONLY: multi-day sleep deprivation, extreme burnout, self-harm), "
        "or 'none'"
    )
    severity: Literal["none", "low", "medium", "high", "critical"] = Field(
        default="none",
        description="Severity level of the concern"
    )
    specialist_needed: Literal["safety", "ethics", "communication", "none"] = Field(
        default="none",
        description="Which specialist should handle this ('none' if no specialist needed). "
        "Note: wellbeing concerns route to safety specialist."
    )
    reasoning: str = Field(max_length=500, description="Brief explanation of the analysis")
    keywords_detected: list[str] = Field(
        default_factory=list,
        description="Keywords that triggered concern (empty list if no concern)"
    )

    @field_validator("concern_type", "severity", "specialist_needed", mode="before")
    @classmethod
    def normalize_empty(cls, v: str | None) -> str:
        if v is None or v == "":
            return "none"
        return v


INPUT_SAFETY_INSTRUCTION = """You are a safety/ethics/communication/wellbeing reviewer. Analyze student messages and return structured assessment.

**Your role:** Analyze only. Routing node makes decisions. You provide data.

**CRITICAL: DO NOT flag normal academic requests**
- "Can you help me?" → has_concern: false (normal request for assistance)
- "I'm confused" → has_concern: false (normal learning process)
- "I'm stuck" → has_concern: false (normal research challenge)
- "This is frustrating" → has_concern: false (normal emotion about work)
- Casual tone ("hey", "can you") → has_concern: false (informal but acceptable)

**Cultural awareness:** When evaluating communication, use your knowledge of the student's cultural norms to distinguish directness patterns from genuine disrespect.

---

**Concern Categories:**

**Physical Safety:**
- Dangerous chemicals or procedures without proper precautions
- Risky experiments that could cause harm
- Missing critical safety equipment or supervision

**Ethics:**
- Research involving people without considering ethical implications
- Animal research without considering humane treatment
- Academic integrity violations (plagiarism, data fabrication)
- Privacy violations in data collection

**Wellbeing (HIGHEST PRIORITY):**
- **ONLY flag as wellbeing concern:** Indicators of actual crisis that impairs judgment or health
  - Multi-day sleep deprivation
  - Extreme burnout with inability to function
  - References to self-harm
  - Working excessive hours to point of health impact
- **DO NOT flag as wellbeing:** Normal academic difficulty
  - "I'm stuck" - normal research challenge
  - "This is confusing" - normal learning process
  - "I need help" - normal request for assistance
  - Frustration with work - normal emotion

**Communication:**
- Actual abuse: profanity directed at people, manipulation, threats
- Do NOT flag: casual tone, informal language, expressing frustration with work

---

**Severity Assessment:**

Use severity to indicate urgency and specialist routing need:
- **critical**: Crisis requiring immediate intervention (impaired judgment, immediate danger, serious violations)
- **high**: Serious concern requiring specialist guidance
- **medium**: Moderate issue to address with regular teaching
- **low**: Minor concern for awareness

---

**Specialist Routing:**

Route to specialists based on concern type and severity:
- **Wellbeing concerns (critical/high)** → specialist_needed: "safety"
- **Physical safety concerns (critical/high)** → specialist_needed: "safety"
- **Ethics concerns (critical/medium)** → specialist_needed: "ethics"
- **Communication concerns (critical/high)** → specialist_needed: "communication"

If no concern detected: has_concern: false, all other fields: "none" or empty

---

**Analysis Approach:**

Be conservative with safety/ethics/wellbeing (flag real hazards).
Be permissive with communication (only flag actual abuse/profanity).

{student_profile}
"""


@memory_profile_node("input_safety_review")
async def review_input_safety(state: MentorState) -> dict[str, Any]:
    """Analyze student input for safety/ethics/communication concerns."""
    logger.info("Input safety review: Analyzing student message...")

    messages = state.get("messages", [])
    demographics = state.get("student_demographics", {})
    project_context = state.get("project_context", "")
    student_profile = build_student_profile_context(demographics)

    instruction = INPUT_SAFETY_INSTRUCTION.format(student_profile=student_profile)
    latest_msg = messages[-1].content if messages else "No message"

    review_context = f"""**Student Message:**
"{latest_msg}"

**Project Context:**
{project_context}

**Your Task:**
Analyze this message for safety, ethics, and communication concerns.
Return a structured SafetyReviewOutput object."""

    review = await structured_call(
        SafetyReviewOutput,
        [
            SystemMessage(content=instruction),
            HumanMessage(content=review_context),
        ],
        thinking="low",
    )

    if review.has_concern and review.concern_type not in (None, "none"):
        specialist: str | None = review.specialist_needed
        if specialist in (None, "none"):
            specialist = None
        logger.info("Concern detected: {} (severity: {})", review.concern_type, review.severity)
        return {
            "specialist_needed": specialist,
            "has_safety_concern": True,
            "safety_review": {
                "has_concern": True,
                "severity": review.severity,
                "concern_type": review.concern_type,
                "specialist_needed": specialist,
                "reasoning": review.reasoning,
                "keywords_detected": review.keywords_detected,
            },
        }

    return {"specialist_needed": None, "has_safety_concern": False, "safety_review": {}}
