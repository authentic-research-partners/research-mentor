"""Stage 5B Software Node — statistical software selection.

Guides student through choosing analysis software:
1. Format system prompt with statistical method
2. Generate conversational response with chat LLM
3. Extract Stage5BSoftwareProgress with structured_call
4. Store software choice in resources_planned
5. NO transition flags — hard graph edge routes to timeline node
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.hypothesis.schemas import (
    Stage5BSoftwareProgress,
)
from research_mentor.question_workshop.hypothesis.state import HypothesisState, hypothesis_config
from research_mentor.question_workshop.hypothesis.utils import (
    build_conversation_context,
    build_hypothesis_system_prompt,
)
from research_mentor.question_workshop.hypothesis.utils.prompts import (
    STAGE_5B_SOFTWARE_EXTRACTION_PROMPT,
    STAGE_5B_SOFTWARE_PROMPT,
)


async def stage_5b_software(state: HypothesisState) -> dict[str, Any]:
    """Stage 5B Software: Help student choose analysis software."""
    logger.info("Stage 5B Software: Processing...")

    messages = state.get("messages", [])
    statistical_method = state.get("statistical_method")
    software_discussed = state.get("software_discussed", False)
    resources_planned = list(state.get("resources_planned", []))

    # Step 1: Format system prompt
    system_prompt = STAGE_5B_SOFTWARE_PROMPT.format(
        statistical_method=statistical_method or "Not yet selected",
        software_discussed=software_discussed,
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
        "Stage 5B Software response length: {} chars",
        len(response_text),
    )

    # Step 3: Extract software progress
    conversation_text = build_conversation_context(messages)
    extraction_prompt = STAGE_5B_SOFTWARE_EXTRACTION_PROMPT.format(
        conversation=conversation_text,
    )

    progress = await structured_call(
        Stage5BSoftwareProgress,
        [
            SystemMessage(content=extraction_prompt),
            HumanMessage(
                content="Extract software selection progress.",
            ),
        ],
        thinking="medium",
        temperature=cfg.extraction_temperature,
    )

    logger.info(
        "Stage 5B Software progress — discussed: {}, selected: {}",
        progress.software_discussed, progress.selected_software,
    )

    # Build state updates
    updates: dict[str, Any] = {"last_ai_response": response_text}

    if progress.software_discussed:
        updates["software_discussed"] = True

        # Store software choice in resources_planned
        if progress.selected_software:
            sw_info = f"Software: {progress.selected_software}"
            if progress.rationale:
                sw_info += f" — {progress.rationale}"
            if sw_info not in resources_planned:
                resources_planned.append(sw_info)
            updates["resources_planned"] = resources_planned

    # Transition to timeline when software discussion is complete
    if progress.software_discussed:
        updates["current_stage"] = "stage_5b_timeline"

    return updates
