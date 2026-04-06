"""Stage 3C Refinement Node — scope adjustment with anti-p-hacking guard.

Guides student through adjusting scope to match data availability:
1. Format system prompt with datasets and current scope
2. Generate conversational response with chat LLM
3. Extract Stage3CProgress with structured_call
4. RED FLAG: If hypothesis_changed -> block with corrective response
5. If scope_adjusted and ready_for_confounds -> stage_3_5_confounds
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.hypothesis.schemas import Stage3CProgress
from research_mentor.question_workshop.hypothesis.state import HypothesisState, hypothesis_config
from research_mentor.question_workshop.hypothesis.utils import (
    build_conversation_context,
    build_hypothesis_system_prompt,
)
from research_mentor.question_workshop.hypothesis.utils.prompts import (
    STAGE_3C_PROGRESS_EXTRACTION_PROMPT,
    STAGE_3C_REFINEMENT_PROMPT,
)

_P_HACKING_WARNING = (
    "I noticed you're considering changing your hypothesis direction. "
    "This is a critical concern in research methodology — changing your "
    "hypothesis AFTER seeing the data is called **p-hacking** and "
    "undermines scientific integrity.\n\n"
    "We can adjust **scope** (time period, location, population) to "
    "match available data, but the **direction** of your hypothesis "
    "(whether X increases or decreases Y) must stay the same.\n\n"
    "Let's focus on adjusting scope instead. What aspect of your "
    "scope could we narrow to better match the available data?"
)


async def stage_3c_refinement(state: HypothesisState) -> dict[str, Any]:
    """Stage 3C: Adjust scope to match data, guard against p-hacking."""
    logger.info("Stage 3C Refinement: Processing...")

    messages = state.get("messages", [])
    datasets_found = state.get("datasets_found", [])
    scope_details = state.get("scope_details", {})

    # Step 1: Format system prompt
    system_prompt = STAGE_3C_REFINEMENT_PROMPT.format(
        datasets_found=datasets_found,
        scope_details=scope_details or "Not yet defined",
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
        "Stage 3C response length: {} chars", len(response_text),
    )

    # Step 3: Extract structured progress
    conversation_text = build_conversation_context(messages)
    extraction_prompt = STAGE_3C_PROGRESS_EXTRACTION_PROMPT.format(
        conversation=conversation_text,
    )

    progress = await structured_call(
        Stage3CProgress,
        [
            SystemMessage(content=extraction_prompt),
            HumanMessage(
                content="Extract scope adjustment progress.",
            ),
        ],
        thinking="medium",
        temperature=cfg.extraction_temperature,
    )

    logger.info(
        "Stage 3C progress — scope_adjusted: {}, "
        "hypothesis_changed: {}, ready_for_confounds: {}",
        progress.scope_adjusted,
        progress.hypothesis_changed,
        progress.ready_for_confounds,
    )

    # Build state updates
    updates: dict[str, Any] = {"last_ai_response": response_text}

    # Step 4: RED FLAG CHECK — hypothesis direction changed
    if progress.hypothesis_changed:
        logger.warning("RED FLAG: Hypothesis direction change detected!")
        updates["last_ai_response"] = _P_HACKING_WARNING
        return updates

    # Update scope if adjusted
    if progress.scope_adjusted:
        updates["scope_adjusted"] = True
        new_scope = dict(scope_details) if scope_details else {}
        if progress.adjusted_when:
            new_scope["when"] = progress.adjusted_when
        if progress.adjusted_where:
            new_scope["where"] = progress.adjusted_where
        if progress.adjusted_population:
            new_scope["population"] = progress.adjusted_population
        updates["scope_details"] = new_scope

    # Step 5: Transition if ready
    if (
        progress.scope_adjusted
        and progress.ready_for_confounds
        and not progress.hypothesis_changed
    ):
        logger.info(
            "Stage 3C complete — transitioning to stage_3_5_confounds",
        )
        updates["current_stage"] = "stage_3_5_confounds"

    return updates
