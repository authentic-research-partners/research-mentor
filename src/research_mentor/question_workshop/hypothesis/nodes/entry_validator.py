"""Entry Validator Node — security and input validation.

Validates user input before processing:
1. Security: Detect prompt injection attempts
2. Basic validation: Not empty, not just whitespace
3. Update state with last_user_message
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.hypothesis.schemas import InputValidationResult
from research_mentor.question_workshop.hypothesis.state import HypothesisState, hypothesis_config
from research_mentor.question_workshop.hypothesis.utils.prompts import VALIDATOR_INSTRUCTION


async def entry_validator(state: HypothesisState) -> dict[str, Any]:
    """Validate user input for security and basic requirements."""
    logger.info("Entry Validator: Validating user input...")

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

    result = await structured_call(
        InputValidationResult,
        [
            SystemMessage(content=VALIDATOR_INSTRUCTION),
            HumanMessage(
                content=(
                    "Analyze this student message for security threats:"
                    f"\n\n{user_input}"
                ),
            ),
        ],
        thinking="low",
        temperature=hypothesis_config().extraction_temperature,
    )

    logger.info("Threat detected: {}, Type: {}", result.is_security_threat, result.threat_type)

    if result.is_security_threat:
        logger.warning(
            "Input blocked - {}: {}", result.threat_type, result.threat_explanation,
        )
        return {
            "error": "Security threat detected",
            "last_ai_response": (
                "I'm here to help with your research question development. "
                "Let's focus on your research project - what aspect would "
                "you like to work on?"
            ),
        }

    logger.info("Input validated successfully")
    return {"last_user_message": user_input, "error": None}
