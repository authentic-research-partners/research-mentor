"""Security Check Node — two-layer prompt injection detection.

Layer 1: Deterministic regex pre-check catches known injection patterns
without relying on the LLM (which may itself be susceptible).

Layer 2: LLM-based structured_call for subtler attacks that don't
match known patterns.

Runs after input_validator. Fail-fast: exceptions propagate.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.hypothesis.schemas import InputValidationResult
from research_mentor.question_workshop.hypothesis.state import HypothesisState, hypothesis_config
from research_mentor.question_workshop.hypothesis.utils.prompts import VALIDATOR_INSTRUCTION

# ---------------------------------------------------------------------------
# Layer 1: Deterministic pattern matching (can't be fooled by the LLM)
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"(reveal|show|display|print|output)\s+(me\s+)?(your\s+)?system\s+prompt",
        r"what\s+(model|ai|llm)\s+are\s+you",
        r"what\s+are\s+you\s+(running|based|built)\s+on",
        r"who\s+(made|created|built)\s+you",
        r"are\s+you\s+(chat\s*gpt|gpt|claude|gemma|llama|mistral)",
        r"enter\s+(developer|admin|debug)\s+mode",
        r"pretend\s+you('re|\s+are)\s+(a\s+)?(different|general|unrestricted)",
        r"forget\s+(your\s+)?(limitations|instructions|rules|constraints)",
        r"(disregard|override|bypass)\s+(all\s+)?(your\s+)?instructions",
        r"you\s+are\s+now\s+a\s+(general|unrestricted|different)",
        r"jailbreak",
        r"dan\s+mode",
        r"do\s+anything\s+now",
    ]
]

_BLOCKED_RESPONSES = [
    (
        "I'm here to help with your research question development. "
        "Let's focus on your research project — what aspect would "
        "you like to work on?"
    ),
    (
        "Let's get back to your research. Is there a topic or question "
        "you've been curious about? I can help you turn it into a hypothesis."
    ),
    (
        "I'd love to help you with a research question! "
        "What's something you've noticed or wondered about recently?"
    ),
]


def _matches_injection_pattern(text: str) -> str | None:
    """Check text against known injection patterns.

    Returns the matched pattern description or None.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def _pick_blocked_response(state: HypothesisState) -> str:
    """Pick a varied blocked response based on conversation length."""
    messages = state.get("messages", [])
    idx = len(messages) % len(_BLOCKED_RESPONSES)
    return _BLOCKED_RESPONSES[idx]


async def security_check(state: HypothesisState) -> dict[str, Any]:
    """Check user input for security threats (deterministic + LLM)."""
    user_input = state.get("last_user_message", "")
    logger.info("Security Check: Analyzing input for threats...")

    # Layer 1: Deterministic pattern matching (fast, reliable)
    matched = _matches_injection_pattern(user_input)
    if matched:
        logger.warning("Input blocked by deterministic check — pattern: {}", matched)
        return {
            "error": "Security threat detected",
            "last_ai_response": _pick_blocked_response(state),
        }

    # Layer 2: LLM-based check for subtler attacks
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
            "last_ai_response": _pick_blocked_response(state),
        }

    logger.info("Security check passed")
    return {"error": None}
