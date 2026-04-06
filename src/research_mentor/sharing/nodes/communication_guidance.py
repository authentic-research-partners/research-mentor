"""Communication Guidance Node — presentations, media, art, authorship.

Consolidates 5 capabilities from the philosophy doc:
- Poster & Presentation Support
- Digital & Social Media Dissemination
- Science Art & Visualization
- Collaborative Authorship
- Open Science Practices

Sub-routing via the communication_topic state field.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.sharing.schemas import CommunicationSubTopic
from research_mentor.sharing.state import SharingState
from research_mentor.sharing.utils import build_sharing_system_prompt
from research_mentor.sharing.utils.prompts import COMMUNICATION_GUIDANCE_PROMPT


async def communication_guidance(state: SharingState) -> dict[str, Any]:
    """Guide student on science communication (posters, presentations, media, etc.)."""
    logger.info("Communication Guidance: Processing...")

    messages = state.get("messages", [])
    user_msg = state.get("last_user_message", "")
    current_topic = state.get("communication_topic")

    # Detect sub-topic from the user's message
    try:
        sub = await structured_call(
            CommunicationSubTopic,
            [
                SystemMessage(content=(
                    "Classify the student's communication question. "
                    "Current topic: " + (current_topic or "not yet determined")
                )),
                HumanMessage(content=f"Student's message: {user_msg}"),
            ],
            thinking="medium",
            temperature=0.0,
        )
        communication_topic = sub.sub_topic
    except Exception as e:
        logger.warning("Communication sub-topic detection failed: {}", e)
        communication_topic = current_topic or "presentation"

    # Generate response
    prompt = COMMUNICATION_GUIDANCE_PROMPT.format(
        communication_topic=communication_topic,
        student_level=state.get("student_level", "unknown"),
    )
    system_prompt = build_sharing_system_prompt(prompt, state)
    llm = get_chat_llm(temperature=0.7, max_tokens=120)
    response = await llm.ainvoke(messages + [SystemMessage(content=system_prompt)])
    response_text = str(response.content)

    # Track phase visit
    phases_visited = list(state.get("phases_visited", []))
    if "communication_guidance" not in phases_visited:
        phases_visited.append("communication_guidance")

    return {
        "last_ai_response": response_text,
        "communication_topic": communication_topic,
        "phases_visited": phases_visited,
    }
