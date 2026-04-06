"""Component Recommender — parallel background node.

Runs alongside progress_assessor, input_safety_review, and student_discovery.
Routes to END directly — never blocks the main flow. If the student's message
suggests they'd benefit from a specific workshop, stores a recommendation in
state that guide nodes can optionally surface.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from research_mentor.agent.state import MentorState
from research_mentor.components_registry import get_catalog_prompt
from research_mentor.llm import structured_call
from research_mentor.utils.memory_profiler import memory_profile_node


class ComponentRecommendation(BaseModel):
    """Whether the student would benefit from a specific workshop."""

    recommended_component: str | None = Field(
        default=None,
        max_length=50,
        description="Component id (e.g. 'hypothesis', 'gaps') or null if Office is the right place",
    )
    confidence: str = Field(
        default="low",
        max_length=50,
        description="'high' if clearly matches a workshop, 'low' if uncertain",
    )


RECOMMENDER_INSTRUCTION = """Does this student message suggest they'd benefit from a specific workshop?

{catalog}

WHEN TO RETURN NULL (stay in Office):
- Student is asking about their ongoing project (data, results, p-values, analysis)
- Student is asking a general methodology question
- Student is greeting or being vague
- You are unsure

Return the component id (the short key like "hypothesis", "gaps", "sharing") — NOT the
full name. Return null if the Office is the right place, or if you are unsure.

Student's message:"""


@memory_profile_node("component_recommender")
async def component_recommender_node(state: MentorState) -> dict[str, Any]:
    """Background node: detect if a workshop would serve the student better."""
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        return {}

    user_text = str(last_message.content).strip()
    if len(user_text) < 10:
        return {}

    catalog = get_catalog_prompt(exclude="office", include_ids=True)
    if not catalog:
        return {}

    try:
        result = await structured_call(
            ComponentRecommendation,
            [
                SystemMessage(content=RECOMMENDER_INSTRUCTION.format(catalog=catalog)),
                HumanMessage(content=user_text),
            ],
            thinking="off",
            temperature=0.0,
        )

        if result.recommended_component and result.confidence == "high":
            logger.info(
                "Component recommender: suggest '{}' (confidence: {})",
                result.recommended_component, result.confidence,
            )
            return {"component_recommendation": result.recommended_component}

        logger.debug("Component recommender: no recommendation (confidence: {})", result.confidence)

    except Exception:
        logger.debug("Component recommender: extraction failed, skipping")

    return {}
