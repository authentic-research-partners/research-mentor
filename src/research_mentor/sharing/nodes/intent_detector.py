"""Intent Detector Node — classify user intent + detect capability switches.

Runs after entry_validator, before the phase node. Detects:
- engaged: normal conversation within current phase
- frustrated: user is stuck
- asking_how: specific guidance request
- capability_switch: user wants a different capability
- ready_to_complete: user wants to wrap up
- general: other conversation

Uses keyword pre-classification for obvious capability switches (reliable),
then falls back to LLM for ambiguous cases. This hybrid approach compensates
for smaller models struggling with routing classification.
"""

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.sharing.schemas import SharingIntentDetection
from research_mentor.sharing.state import SharingState
from research_mentor.sharing.utils.prompts import INTENT_DETECTION_PROMPT

# Keyword patterns for capability detection (checked before LLM call)
_WRITING_KEYWORDS = re.compile(
    r"\b(writ\w*|draft\w*|abstract|introduction|methods?\s+section|results?\s+section|"
    r"paper\s+structure|imrad|revis\w*|reviewer?\s+feedback|peer\s+review|"
    r"structure\s+(my|the|a)\s+(paper|report|article))\b",
    re.IGNORECASE,
)
_VENUE_KEYWORDS = re.compile(
    r"\b(journal|venue|publish|where\s+(should|can|could)\s+i\s+(share|publish|submit|present)|"
    r"preprint|predatory|legitimate|submit\s+(to|my)|red\s+flags?|doaj)\b",
    re.IGNORECASE,
)
_COMM_KEYWORDS = re.compile(
    r"\b(poster|presentation|slides?|tiktok|youtube|video|blog|podcast|"
    r"infographic|science\s+art|illustration|authorship|open\s+science|"
    r"preregistration|data\s+sharing|credit)\b",
    re.IGNORECASE,
)
_COMPLETION_KEYWORDS = re.compile(
    r"\b(wrap\s+up|that'?s\s+all|we'?re\s+done|i'?m\s+done|i\s+have\s+what\s+i\s+need|"
    r"let'?s\s+finish|thank\s+you\s+that'?s\s+enough)\b",
    re.IGNORECASE,
)


def _keyword_detect(
    msg: str, current_phase: str, context_gathered: bool,
) -> tuple[str, str | None] | None:
    """Fast keyword-based capability detection. Returns (intent, target) or None."""
    if not context_gathered:
        return None

    # Completion signals (always checked)
    if _COMPLETION_KEYWORDS.search(msg):
        return ("ready_to_complete", None)

    # Check for capability mismatch with current phase.
    # Writing and communication overlap (poster structure = writing_support).
    # Only switch if the match is clearly for a DIFFERENT phase.
    if current_phase not in ("writing_support", "communication_guidance"):
        if _WRITING_KEYWORDS.search(msg):
            return ("capability_switch", "writing_support")
    if current_phase != "venue_discovery" and _VENUE_KEYWORDS.search(msg):
        return ("capability_switch", "venue_discovery")
    if current_phase not in ("communication_guidance", "writing_support"):
        if _COMM_KEYWORDS.search(msg):
            return ("capability_switch", "communication_guidance")

    # Within-phase keyword match: if the message matches the CURRENT phase's
    # keywords, the student is engaged in that phase (not switching away).
    # Without this, the LLM fallback often misclassifies follow-up questions.
    _PHASE_KEYWORDS = {
        "venue_discovery": _VENUE_KEYWORDS,
        "writing_support": _WRITING_KEYWORDS,
        "communication_guidance": _COMM_KEYWORDS,
    }
    phase_pattern = _PHASE_KEYWORDS.get(current_phase)
    if phase_pattern and phase_pattern.search(msg):
        return ("asking_how", None)

    return None  # Ambiguous — fall back to LLM


async def intent_detector(state: SharingState) -> dict[str, Any]:
    """Detect user intent and capability switch signals.

    Uses keyword pre-classification for obvious switches, LLM for ambiguous.
    """
    user_msg = state.get("last_user_message", "")
    current_phase = state.get("current_phase", "context_gathering")
    context_gathered = state.get("context_gathered", False)

    if not user_msg:
        return {"user_intent": "general"}

    # Fast path: keyword-based detection for obvious capability switches
    kw_result = _keyword_detect(user_msg, current_phase, context_gathered)
    if kw_result is not None:
        intent, target = kw_result
        logger.info(
            "Intent (keyword): {} → {} (current: {})",
            intent, target or "none", current_phase,
        )
        updates: dict[str, Any] = {"user_intent": intent}
        if target:
            updates["requested_capability"] = target
        return updates

    # Slow path: LLM classification for ambiguous cases
    try:
        result = await structured_call(
            SharingIntentDetection,
            [
                SystemMessage(content=INTENT_DETECTION_PROMPT.format(
                    user_message=user_msg,
                    current_phase=current_phase,
                    context_gathered=context_gathered,
                )),
                HumanMessage(content="Classify this user's intent."),
            ],
            thinking="medium",
            temperature=0.0,
        )

        logger.info(
            "Intent (LLM): {} (requested_capability: {})",
            result.intent,
            result.requested_capability or "none",
        )

        updates = {"user_intent": result.intent}
        if result.requested_capability:
            updates["requested_capability"] = result.requested_capability

        return updates

    except Exception as e:
        logger.warning("Intent detection failed: {}", e)
        return {"user_intent": "general"}
