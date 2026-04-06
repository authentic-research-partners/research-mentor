"""Learning Pathway Advisor Tool.

Recommends systematic learning paths (books, courses, sequences) for long-term
mastery when students want to go deeper into a field beyond their current
research project.

CRITICAL BOUNDARIES:
- This is OPTIONAL enrichment (non-blocking)
- Student can complete research without this
- Triggered when student shows sustained interest + asks about future learning
- NOT for immediate prerequisite gaps (use concept_explainer for that)

Ported from progress_mentor's educational_pathway_advisor.py with adaptations:
- Uses structured_call() instead of get_structured_llm()
- No silent fallbacks — let errors propagate
- No LangChain StructuredTool wrapper — exports async function directly
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field


class ImmediateResource(BaseModel):
    """Resource to start learning immediately (this month)."""

    resource: str = Field(
        max_length=200,
        description="Name of resource (e.g., 'Khan Academy: Fluid Mechanics')",
    )
    time_commitment: str = Field(
        max_length=100,
        description="How long it takes (e.g., '8 hours', '2 weeks')",
    )
    cost: str = Field(max_length=100, description="Cost ('Free', '$49', 'Library book')")
    why: str = Field(
        max_length=300,
        description="Why start here (builds on their project knowledge)",
    )


class HighSchoolCourse(BaseModel):
    """High school course recommendation."""

    course: str = Field(max_length=200, description="Course name (e.g., 'AP Physics C: Mechanics')")
    when: str = Field(max_length=100, description="When to take it (e.g., 'Next school year')")
    prerequisite: str | None = Field(
        default=None, max_length=200, description="What's needed before this",
    )
    why: str = Field(
        max_length=300, description="Why this course matters for the field",
    )
    alternative: str | None = Field(
        default=None, max_length=200, description="Alternative if not available",
    )


class Book(BaseModel):
    """Book recommendation."""

    title: str = Field(max_length=300, description="Book title with author")
    level: str = Field(max_length=50, description="Level (e.g., 'High school', 'Undergraduate')")
    why: str = Field(max_length=300, description="Why this book (what makes it good)")
    chapters: str | None = Field(
        default=None, max_length=200, description="Specific chapters if not reading whole book"
    )


class OnlineCourse(BaseModel):
    """Online course recommendation."""

    platform: str = Field(
        max_length=100,
        description="Platform (Coursera, edX, MIT OCW, Khan Academy)",
    )
    course: str = Field(max_length=200, description="Course name")
    duration: str = Field(max_length=100, description="How long (e.g., '4 weeks', '10 hours')")
    when: str = Field(max_length=150, description="When to take it (after what prerequisites)")
    why: str = Field(max_length=300, description="Why this course specifically")


class CareerPath(BaseModel):
    """Career path that connects to the field."""

    title: str = Field(max_length=150, description="Career title")
    connection: str = Field(
        max_length=300,
        description="How their research project relates to this career",
    )


class LearningPathway(BaseModel):
    """Complete structured learning pathway for long-term mastery."""

    field: str = Field(max_length=150, description="Field of study")
    start_here: list[ImmediateResource] = Field(
        description="1-2 resources to start immediately",
        max_length=2,
    )
    high_school_courses: list[HighSchoolCourse] = Field(
        description="Recommended high school courses (2-4)",
        max_length=4,
    )
    books: list[Book] = Field(
        description="2-4 books in progressive difficulty",
        max_length=4,
    )
    online_courses: list[OnlineCourse] = Field(
        description="2-4 online courses for self-paced learning",
        max_length=4,
    )
    learning_sequence: list[str] = Field(
        description="Step-by-step timeline (4-8 steps) from now through college",
        max_length=8,
    )
    career_paths: list[CareerPath] = Field(
        description="2-4 career paths connected to this field",
        max_length=4,
    )
    builds_on_project: str = Field(
        max_length=800,
        description="How this pathway builds on their research project (2-3 sentences)"
    )
    message_to_student: str = Field(
        max_length=800,
        description="Encouraging message (2-3 sentences) about pursuing this path"
    )


PATHWAY_ADVISOR_INSTRUCTION = """You are an Educational Pathway Advisor that creates \
structured learning paths for students who want to go deeper into a field beyond \
their current research project.

**CRITICAL: This is OPTIONAL enrichment (non-blocking).**

Students DON'T need this to complete their research. This is for when they discover a \
passion and want systematic mastery for future careers/college.

**Your Goal:**
Create realistic, actionable, cost-conscious learning pathways that:
1. Start where student is NOW (build on their project knowledge)
2. Progress gradually (respect school constraints)
3. Prioritize free/accessible resources (Khan Academy, MIT OCW, library books)
4. Sequence appropriately (prerequisites → intermediate → advanced)
5. Connect to real careers (maintain motivation)
6. **ADAPT TO COUNTRY'S EDUCATION SYSTEM** (country-specific qualifications and pathways)

**Output Structure:**
- start_here: 1-2 immediate resources (this month)
- high_school_courses: 2-4 recommended courses
- books: 2-4 books in progressive difficulty
- online_courses: 2-4 self-paced courses
- learning_sequence: 4-8 step timeline through college
- career_paths: 2-4 connected careers
- builds_on_project: How pathway connects to their research
- message_to_student: Encouraging but realistic message

**CRITICAL PRINCIPLES:**
1. Free First: Prioritize free resources
2. Build Gradually: Don't jump from high school to graduate level
3. Respect Prerequisites: Don't recommend calculus-based physics before calculus
4. Respect Time: Students have limited time
5. Connect to Project: Every recommendation ties back to their research
6. Be Specific: "Khan Academy: Fluid Mechanics" not "watch videos online"
7. Realistic Timeline: Learning takes years, not weeks
8. Country-Specific: Adapt to their education system (AP, A-Levels, IB, etc.)"""


async def recommend_learning_pathway(
    field_of_interest: str,
    project_context: str,
    student_demographics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recommend systematic learning pathway for long-term mastery of a field.

    This is for students who want to go DEEPER into a field after discovering
    passion through their research project. NOT for immediate prerequisite gaps
    (use concept_explainer for those).

    Args:
        field_of_interest: Field student wants to master
        project_context: Description of their research project
        student_demographics: Optional dict with age, gradeLevel, country

    Returns:
        Dictionary with pathway sections (start_here, courses, books, etc.)
    """
    from research_mentor.llm import structured_call

    logger.info("Pathway advisor invoked: field='{}'", field_of_interest)

    from research_mentor.agent.prompts.shared import build_student_profile_context

    demographics = student_demographics or {}
    profile_text = build_student_profile_context(demographics)

    context = (
        f"Field of Interest: {field_of_interest}\n"
        f"Current Project: {project_context}\n\n"
    )
    if profile_text:
        context += f"{profile_text}\n\n"
    context += (
        "Create a structured learning pathway for this student that:\n"
        "1. Respects their country's educational system\n"
        "2. Builds on their project knowledge and existing expertise\n"
        "3. Connects to their research interests\n"
        "4. Skips material they already know based on their domain expertise"
    )

    pathway = await structured_call(
        LearningPathway,
        [
            SystemMessage(content=PATHWAY_ADVISOR_INSTRUCTION),
            HumanMessage(content=context),
        ],
        thinking="medium",
    )

    logger.info(
        "Generated pathway for {}: {} steps",
        field_of_interest, len(pathway.learning_sequence),
    )

    return {
        "field": pathway.field,
        "start_here": [r.model_dump() for r in pathway.start_here],
        "high_school_courses": [c.model_dump() for c in pathway.high_school_courses],
        "books": [b.model_dump() for b in pathway.books],
        "online_courses": [c.model_dump() for c in pathway.online_courses],
        "learning_sequence": pathway.learning_sequence,
        "career_paths": [c.model_dump() for c in pathway.career_paths],
        "builds_on_project": pathway.builds_on_project,
        "message_to_student": pathway.message_to_student,
    }
