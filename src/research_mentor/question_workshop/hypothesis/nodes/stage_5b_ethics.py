"""Stage 5B Ethics Node — research ethics discussion.

Guides student through ethical considerations:
1. Format system prompt with hypothesis and selected dataset
2. Generate conversational response with chat LLM
3. Extract Stage5BEthicsProgress with structured_call
4. Store ethics info in resources_planned
5. NO transition flags — hard graph edge routes to software node
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.hypothesis.schemas import Stage5BEthicsProgress
from research_mentor.question_workshop.hypothesis.state import HypothesisState, hypothesis_config
from research_mentor.question_workshop.hypothesis.utils import (
    build_conversation_context,
    build_hypothesis_system_prompt,
)
from research_mentor.question_workshop.hypothesis.utils.prompts import (
    STAGE_5B_ETHICS_EXTRACTION_PROMPT,
    STAGE_5B_ETHICS_PROMPT,
)


async def stage_5b_ethics(state: HypothesisState) -> dict[str, Any]:
    """Stage 5B Ethics: Discuss research ethics considerations."""
    logger.info("Stage 5B Ethics: Processing...")

    messages = state.get("messages", [])
    hypothesis = state.get("hypothesis", "Not yet formed")
    selected_dataset = state.get("selected_dataset")
    ethics_discussed = state.get("ethics_discussed", False)
    resources_planned = list(state.get("resources_planned", []))

    dataset_info = (
        selected_dataset.get("name", "Unknown")
        if selected_dataset
        else "No dataset selected"
    )

    # Step 1: Format system prompt
    system_prompt = STAGE_5B_ETHICS_PROMPT.format(
        hypothesis=hypothesis,
        selected_dataset=dataset_info,
        ethics_discussed=ethics_discussed,
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
        "Stage 5B Ethics response length: {} chars", len(response_text),
    )

    # Step 3: Extract ethics progress
    conversation_text = build_conversation_context(messages)
    extraction_prompt = STAGE_5B_ETHICS_EXTRACTION_PROMPT.format(
        conversation=conversation_text,
    )

    progress = await structured_call(
        Stage5BEthicsProgress,
        [
            SystemMessage(content=extraction_prompt),
            HumanMessage(
                content="Extract ethics discussion progress.",
            ),
        ],
        thinking="medium",
        temperature=cfg.extraction_temperature,
    )

    logger.info(
        "Stage 5B Ethics progress — discussed: {}",
        progress.ethics_discussed,
    )

    # Build state updates
    updates: dict[str, Any] = {"last_ai_response": response_text}

    if progress.ethics_discussed:
        updates["ethics_discussed"] = True

        # Store ethics info in resources_planned
        ethics_info = "Ethics: "
        if progress.ethics_notes:
            ethics_info += progress.ethics_notes
        else:
            ethics_info += "Discussed"

        if ethics_info not in resources_planned:
            resources_planned.append(ethics_info)
        updates["resources_planned"] = resources_planned

    # Transition to software when ethics discussion is complete
    if progress.ethics_discussed:
        updates["current_stage"] = "stage_5b_software"

    return updates
