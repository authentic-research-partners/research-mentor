"""Student Discovery node — extracts student attributes from conversation."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from research_mentor.agent.state import MentorState
from research_mentor.config import get_llm_model_info
from research_mentor.db.crud import update_student_attributes, upsert_student_profile
from research_mentor.llm import structured_call
from research_mentor.utils.memory_profiler import memory_profile_node


class DiscoveredAttributes(BaseModel):
    """Extracted student attributes."""

    courses_taken: list[str] = Field(default_factory=list, description="Courses mentioned")
    technical_skills: list[str] = Field(default_factory=list, description="Skills mentioned")
    research_interests: list[str] = Field(default_factory=list, description="Interests mentioned")
    institution: str | None = Field(
        default=None,
        max_length=200,
        description="University or institution explicitly mentioned (e.g. 'MIT', 'Stanford')",
    )
    location: str | None = Field(
        default=None,
        max_length=200,
        description="City or region explicitly mentioned (e.g. 'Ann Arbor', 'California')",
    )
    confidence: str = Field(
        max_length=50,
        description="YOUR confidence that student EXPLICITLY stated attributes "
        "(NOT student's confidence in their abilities): 'high' or 'low'"
    )
    reasoning: str = Field(
        default="",
        max_length=500,
        description="Brief explanation of what was discovered and why YOU are confident/not confident"
    )


EXTRACTION_INSTRUCTION = """Extract courses, skills, interests, and location from student messages.

**OUTPUT FORMAT:**
Array fields MUST be arrays: courses_taken=[], technical_skills=[], research_interests=[]
String fields: institution=null or string, location=null or string

**EXTRACTION PRINCIPLES:**
- Preserve student's language (don't expand or normalize names)
- Extract specific mentions (not generic categories)
- Focus on explicitly stated current attributes (not future intent or vague preferences)
- For institution: extract university, college, or research institute name if mentioned
- For location: extract city, region, or state if mentioned

**confidence="high" - Use when student DIRECTLY states:**
- Taking a course
- Knowing a specific skill
- Wanting to study a specific topic
- Being at a specific institution ("I'm at MIT", "I study at Stanford")
- Being in a specific location ("I'm in Ann Arbor", "here in California")

**confidence="low" - Use when:**
- Statement is vague or generic
- Refers to future intent ("I should learn...")
- Discusses project topic (not personal background)

**CRITICAL: "confidence" reflects YOUR confidence in the extraction, NOT the student's skill level.**

**Student's message:**"""


@memory_profile_node("student_discovery")
async def extract_student_attributes_node(state: MentorState) -> dict[str, Any]:
    """Background node: extract and save student attributes."""
    logger.debug("Student discovery: Extracting attributes...")

    messages = state.get("messages", [])
    project_id = state.get("project_id")

    if not messages or not project_id:
        return {}

    recent_messages = messages[-4:]
    conversation = "\n".join(
        f"{'Student' if isinstance(m, HumanMessage) else 'Mentor'}: {m.content}"
        for m in recent_messages
    )

    try:
        attrs = await structured_call(
            DiscoveredAttributes,
            [
                SystemMessage(content=EXTRACTION_INSTRUCTION),
                HumanMessage(content=f"Extract attributes from:\n\n{conversation}"),
            ],
            thinking="low",
        )

        if attrs.confidence == "high" and any(
            [attrs.courses_taken, attrs.technical_skills, attrs.research_interests]
        ):
            await update_student_attributes(
                project_id=project_id,
                courses_taken=attrs.courses_taken or None,
                technical_skills=attrs.technical_skills or None,
                research_interests=attrs.research_interests or None,
                llm_info=get_llm_model_info(),
            )

        # Store institution/location to student profile if discovered
        if attrs.confidence == "high" and (attrs.institution or attrs.location):
            await upsert_student_profile(
                institution=attrs.institution,
                city=attrs.location,
            )

    except Exception as e:
        logger.debug(f"Student discovery failed (non-critical): {e}")

    return {}
