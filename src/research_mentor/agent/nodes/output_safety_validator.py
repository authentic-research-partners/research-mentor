"""Output Safety Validator node — validates generated content for physical/psychological safety."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from research_mentor.agent.state import MentorState
from research_mentor.llm import structured_call
from research_mentor.utils.memory_profiler import memory_profile_node


class OutputSafetyResult(BaseModel):
    """Structured output for content safety validation."""

    is_safe: bool = Field(description="Whether the content is safe for this student")
    concerns: list[str] = Field(description="Specific safety issues identified")
    severity: Literal["none", "low", "medium", "high", "critical"] = Field(
        default="none",
        description="Severity level, or 'none' if safe"
    )
    concern_type: Literal["physical", "psychological", "both", "none"] = Field(
        default="none",
        description="Type of concern, or 'none' if safe"
    )
    reasoning: str = Field(max_length=500, description="Brief reasoning")

    @field_validator("severity", "concern_type", mode="before")
    @classmethod
    def normalize_empty(cls, v: str | None) -> str:
        if v is None or v == "":
            return "none"
        return v


OUTPUT_SAFETY_INSTRUCTION = """You validate AI-generated research guidance for student safety.

{student_profile}

**Your ONLY job:** Analyze content for safety risks. Flag issues. DO NOT create new content.

---

**PHYSICAL SAFETY CHECKS:**

🚨 **CRITICAL (always flag):**
- Dangerous chemicals/materials for home setting without extensive safety protocols
- High-risk procedures for young students without "requires adult supervision"
- Explosive/flammable materials without explicit warnings
- High voltage electricity (>50V) without safety training requirements
- Toxic substances without proper PPE and ventilation requirements
- **And other similar immediate physical dangers**

⚠️ **HIGH (flag):**
- Heat/fire without safety equipment requirements (fire extinguisher, ventilation)
- Sharp tools/power tools without supervision requirements for young students
- Chemicals without MSDS review mentioned
- Biological hazards without proper disposal instructions
- Equipment student likely doesn't have access to (fume hood, autoclave)
- **And other similar serious safety concerns**

---

**PSYCHOLOGICAL SAFETY CHECKS:**

🚨 **CRITICAL (always flag):**
- Timelines causing sleep deprivation (work through the night, skip sleep)
- Excessive work hours (10+ hours straight) without breaks
- Potentially traumatic content for this age (graphic dissection details for young students)
- Rude, disrespectful, or culturally insensitive language toward student
- Dismissive or condescending tone that undermines student confidence
- **And other similar severe psychological harm risks**

⚠️ **HIGH (flag):**
- Unrealistic deadlines creating unhealthy pressure
- Lack of breaks in intensive work periods
- Content that could be distressing without proper framing
- Harsh or overly critical feedback without constructive elements
- **And other similar psychological stress concerns**

---

**WHAT IS SAFE (do NOT flag):**
- Age-appropriate experiments with proper safety warnings
- Procedures explicitly stating supervision requirements
- Realistic timelines with appropriate breaks
- Challenging but achievable goals
- Normal research frustration or difficulty
- **And other similar safe, well-structured guidance**

---

**Decision Logic:**

**SAFE (is_safe=true):**
- Content appropriate for student's age
- Safety warnings present where needed
- Supervision requirements stated
- Reasonable timelines and expectations

**UNSAFE (is_safe=false):**
- Missing critical safety warnings
- Dangerous for this age without proper safeguards
- Could cause physical harm
- Could cause psychological distress

**Severity Guidelines:**
- critical: Immediate physical danger or severe psychological harm
- high: Serious risk without proper precautions
- medium: Moderate concern, should be addressed
- low: Minor issue, good to note

---

**Output Format:**

Safe content:
{{"is_safe": true, "concerns": [], "severity": "none", "concern_type": "none", "reasoning": "Content includes appropriate safety warnings for student's age"}}

Unsafe content:
{{"is_safe": false, "concerns": ["heating sealed container without pressure relief", "no mention of adult supervision"], "severity": "critical", "concern_type": "physical", "reasoning": "Recommends dangerous procedure without safety requirements"}}
"""


@memory_profile_node("output_safety_validator")
async def validate_output_safety(state: MentorState) -> dict[str, Any]:
    """Validate generated content for safety."""
    logger.info("Output safety validator: Checking content...")

    response = state.get("response", "")
    if not response:
        return {"content_safety_flagged": False, "output_safety_review": {"is_safe": True}}

    from research_mentor.agent.prompts.shared import build_student_profile_context

    demographics = state.get("student_demographics", {})
    profile_text = build_student_profile_context(demographics)

    instruction = OUTPUT_SAFETY_INSTRUCTION.format(
        student_profile=profile_text or "Student profile not available",
    )

    validation = await structured_call(
        OutputSafetyResult,
        [
            SystemMessage(content=instruction),
            HumanMessage(
                content=f"**Content to Validate:**\n\n{response[:2000]}\n\n"
                "**Task:** Analyze this content for physical and psychological safety "
                "concerns for this student. Return structured OutputSafetyResult."
            ),
        ],
        thinking="low",
    )

    if not validation.is_safe:
        logger.warning(
            "Content flagged: type={}, severity={}",
            validation.concern_type,
            validation.severity,
        )

    return {
        "content_safety_flagged": not validation.is_safe,
        "output_safety_review": {
            "is_safe": validation.is_safe,
            "concerns": validation.concerns,
            "severity": validation.severity,
            "concern_type": validation.concern_type,
            "reasoning": validation.reasoning,
        },
    }
