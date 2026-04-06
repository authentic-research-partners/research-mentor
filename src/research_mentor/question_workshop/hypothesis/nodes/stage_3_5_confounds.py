"""Stage 3.5 Confounds Node — confound identification.

Guides student through identifying confounding variables:
1. Format system prompt with current confound count
2. Generate conversational response with chat LLM
3. Extract ConfoundExtraction with structured_call
4. Merge with existing confounds (dedup by lowercase name)
5. Validate each confound with validate_confound()
6. If >= 3 valid confounds -> stage_4_operationalization
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.hypothesis.schemas import ConfoundExtraction
from research_mentor.question_workshop.hypothesis.state import HypothesisState, hypothesis_config
from research_mentor.question_workshop.hypothesis.utils import (
    build_conversation_context,
    build_hypothesis_system_prompt,
)
from research_mentor.question_workshop.hypothesis.utils.prompts import (
    CONFOUND_EXTRACTION_PROMPT,
    STAGE_3_5_CONFOUNDS_PROMPT,
)
from research_mentor.question_workshop.hypothesis.utils.validation import (
    validate_confound,
)


def _merge_confounds(
    existing: list[dict[str, Any]],
    new_confounds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge new confounds into existing list, deduplicating by name."""
    max_confounds = hypothesis_config().max_confounds
    seen = {c["name"].lower().strip() for c in existing}
    merged = list(existing)
    for c in new_confounds:
        key = c["name"].lower().strip()
        if key not in seen:
            seen.add(key)
            merged.append(c)
    return merged[:max_confounds]


async def stage_3_5_confounds(state: HypothesisState) -> dict[str, Any]:
    """Stage 3.5: Guide student through confound identification."""
    logger.info("Stage 3.5 Confounds: Processing...")

    messages = state.get("messages", [])
    independent_var = state.get("independent_var", "")
    dependent_var = state.get("dependent_var", "")
    existing_confounds = state.get("confounds", [])
    confounds_count = state.get("confounds_count", 0)

    # Step 1: Format system prompt
    system_prompt = STAGE_3_5_CONFOUNDS_PROMPT.format(
        confounds_count=confounds_count,
    )

    # Build message list with system prompt + recent conversation
    system_prompt = build_hypothesis_system_prompt(system_prompt, state)
    recent_messages: list[Any] = [SystemMessage(content=system_prompt)]
    for msg in messages[-10:]:
        recent_messages.append(msg)

    # Step 2: Generate conversational response
    cfg = hypothesis_config()
    llm = get_chat_llm(agent="hypothesis", temperature=cfg.conversational_temperature)
    response_msg = await llm.ainvoke(recent_messages)
    response_text = str(response_msg.content)

    logger.debug(
        "Stage 3.5 response length: {} chars", len(response_text),
    )

    # Step 3: Extract confounds with structured_call
    conversation_text = build_conversation_context(messages)
    extraction_prompt = CONFOUND_EXTRACTION_PROMPT.format(
        independent_var=independent_var or "unknown",
        dependent_var=dependent_var or "unknown",
        conversation=conversation_text,
    )

    extracted = await structured_call(
        ConfoundExtraction,
        [
            SystemMessage(content=extraction_prompt),
            HumanMessage(
                content="Extract confounding variables from the conversation.",
            ),
        ],
        thinking="medium",
        temperature=cfg.extraction_temperature,
    )

    logger.info(
        "Extracted {} confounds from conversation",
        len(extracted.confounds),
    )

    # Step 4: Validate and merge confounds
    new_confounds = []
    for c in extracted.confounds:
        dumped = c.model_dump()
        if await validate_confound(dumped):
            new_confounds.append(dumped)

    merged = _merge_confounds(existing_confounds, new_confounds)
    valid_count = len(merged)

    logger.info(
        "Confounds — existing: {}, new valid: {}, total: {}",
        len(existing_confounds), len(new_confounds), valid_count,
    )

    # Build state updates
    updates: dict[str, Any] = {
        "last_ai_response": response_text,
        "confounds": merged,
        "confounds_count": valid_count,
    }

    # Step 5: Check for backward jump or forward transition
    if extracted.hypothesis_undermined:
        logger.info(
            "Stage 3.5 → Stage 3A: confound undermines hypothesis",
        )
        updates["current_stage"] = "stage_3a_hypothesis"
    elif valid_count >= hypothesis_config().min_confounds_for_transition:
        logger.info(
            "Stage 3.5 complete ({} confounds) → Stage 4",
            valid_count,
        )
        updates["current_stage"] = "stage_4_operationalization"

    return updates
