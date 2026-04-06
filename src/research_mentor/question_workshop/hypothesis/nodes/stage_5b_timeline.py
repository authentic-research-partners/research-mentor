"""Stage 5B Timeline Node — research timeline planning.

Guides student through timeline estimation:
1. Format system prompt with current timeline status
2. Generate conversational response with chat LLM
3. Extract Stage5BTimelineProgress with structured_call
4. Store timeline in resources_planned
5. If timeline_discussed -> set should_transition=True, current_stage="completion"
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.hypothesis.schemas import (
    Stage5BTimelineProgress,
)
from research_mentor.question_workshop.hypothesis.state import HypothesisState, hypothesis_config
from research_mentor.question_workshop.hypothesis.utils import (
    build_conversation_context,
    build_hypothesis_system_prompt,
)
from research_mentor.question_workshop.hypothesis.utils.prompts import (
    STAGE_5B_TIMELINE_EXTRACTION_PROMPT,
    STAGE_5B_TIMELINE_PROMPT,
)


async def stage_5b_timeline(state: HypothesisState) -> dict[str, Any]:
    """Stage 5B Timeline: Help student plan research timeline."""
    logger.info("Stage 5B Timeline: Processing...")

    messages = state.get("messages", [])
    timeline_discussed = state.get("timeline_discussed", False)
    resources_planned = list(state.get("resources_planned", []))

    # Step 1: Format system prompt
    system_prompt = STAGE_5B_TIMELINE_PROMPT.format(
        timeline_discussed=timeline_discussed,
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
        "Stage 5B Timeline response length: {} chars",
        len(response_text),
    )

    # Step 3: Extract timeline progress
    conversation_text = build_conversation_context(messages)
    extraction_prompt = STAGE_5B_TIMELINE_EXTRACTION_PROMPT.format(
        conversation=conversation_text,
    )

    progress = await structured_call(
        Stage5BTimelineProgress,
        [
            SystemMessage(content=extraction_prompt),
            HumanMessage(
                content="Extract timeline planning progress.",
            ),
        ],
        thinking="medium",
        temperature=cfg.extraction_temperature,
    )

    logger.info(
        "Stage 5B Timeline progress — discussed: {}, total_weeks: {}",
        progress.timeline_discussed, progress.total_weeks,
    )

    # Build state updates
    updates: dict[str, Any] = {"last_ai_response": response_text}

    if progress.timeline_discussed:
        updates["timeline_discussed"] = True

        # Build timeline summary for resources_planned
        parts = []
        if progress.data_collection_weeks:
            parts.append(
                f"Data collection: {progress.data_collection_weeks}w",
            )
        if progress.analysis_weeks:
            parts.append(f"Analysis: {progress.analysis_weeks}w")
        if progress.writeup_weeks:
            parts.append(f"Write-up: {progress.writeup_weeks}w")
        if progress.total_weeks:
            parts.append(f"Total: {progress.total_weeks}w")

        if parts:
            timeline_info = "Timeline: " + ", ".join(parts)
            if timeline_info not in resources_planned:
                resources_planned.append(timeline_info)
            updates["resources_planned"] = resources_planned

        # Timeline is the last 5B sub-stage — transition to completion
        logger.info(
            "Stage 5B Timeline complete — "
            "transitioning to completion",
        )
        updates["current_stage"] = "completion"

    return updates
