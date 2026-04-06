"""Input Validator Node — basic input validation.

Validates user input before security check:
1. Messages list not empty
2. Latest message is a HumanMessage
3. Content is not empty/whitespace
4. Set last_user_message for downstream nodes
"""

from typing import Any

from langchain_core.messages import HumanMessage
from loguru import logger

from research_mentor.question_workshop.hypothesis.state import HypothesisState


async def input_validator(state: HypothesisState) -> dict[str, Any]:
    """Validate basic input requirements."""
    logger.info("Input Validator: Checking message...")

    messages = state.get("messages", [])
    if not messages:
        logger.warning("No messages to validate")
        return {"error": "No message provided"}

    latest_message = messages[-1]
    if not isinstance(latest_message, HumanMessage):
        logger.warning("Latest message is not from user")
        return {"error": "Invalid message type"}

    user_input = str(latest_message.content)
    logger.debug("Validating input: {}...", user_input[:100])

    if not user_input or not user_input.strip():
        logger.warning("Empty or whitespace-only input")
        return {
            "error": "Empty input",
            "last_ai_response": (
                "I didn't receive any message. Could you please share "
                "what you're thinking about or what you need help with?"
            ),
        }

    logger.info("Input validated — passing to security check")
    return {"last_user_message": user_input, "error": None}
