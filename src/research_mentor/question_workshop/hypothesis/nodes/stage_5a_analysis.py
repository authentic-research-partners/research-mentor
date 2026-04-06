"""Stage 5A Analysis Node — statistical analysis planning.

Guides student through choosing statistical methods:
1. Format system prompt with current method status
2. Generate conversational response with chat LLM
3. Extract AnalysisProgress with structured_call
4. If statistical_method selected -> stage_5b_resources
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.hypothesis.schemas import AnalysisProgress
from research_mentor.question_workshop.hypothesis.state import HypothesisState, hypothesis_config
from research_mentor.question_workshop.hypothesis.utils import (
    build_conversation_context,
    build_hypothesis_system_prompt,
)
from research_mentor.question_workshop.hypothesis.utils.prompts import (
    ANALYSIS_EXTRACTION_PROMPT,
    STAGE_5A_ANALYSIS_PROMPT,
)


async def stage_5a_analysis(state: HypothesisState) -> dict[str, Any]:
    """Stage 5A: Plan statistical analysis with student."""
    logger.info("Stage 5A Analysis: Processing...")

    messages = state.get("messages", [])
    statistical_method = state.get("statistical_method")

    # Step 1: Format system prompt
    system_prompt = STAGE_5A_ANALYSIS_PROMPT.format(
        independent_var=state.get("independent_var") or "N/A",
        dependent_var=state.get("dependent_var") or "N/A",
        x_measurement=state.get("x_measurement") or "Not yet specified",
        y_measurement=state.get("y_measurement") or "Not yet specified",
        hypothesis=state.get("hypothesis") or "Not yet defined",
        statistical_method=statistical_method or "Not yet selected",
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
        "Stage 5A response length: {} chars", len(response_text),
    )

    # Step 3: Extract analysis progress
    conversation_text = build_conversation_context(messages)
    extraction_prompt = ANALYSIS_EXTRACTION_PROMPT.format(
        conversation=conversation_text,
    )

    progress = await structured_call(
        AnalysisProgress,
        [
            SystemMessage(content=extraction_prompt),
            HumanMessage(
                content="Extract analysis planning progress.",
            ),
        ],
        thinking="medium",
        temperature=cfg.extraction_temperature,
    )

    logger.info(
        "Stage 5A progress — method: {}, H0: {}, H1: {}",
        progress.statistical_method,
        progress.h0_defined,
        progress.h1_defined,
    )

    # Build state updates
    updates: dict[str, Any] = {"last_ai_response": response_text}

    if progress.statistical_method:
        updates["statistical_method"] = progress.statistical_method
    if progress.h0_defined:
        updates["h0_defined"] = True
    if progress.h1_defined:
        updates["h1_defined"] = True

    # Step 4: Transition if statistical method selected
    if progress.statistical_method:
        logger.info(
            "Stage 5A complete — transitioning to stage_5b_resources",
        )
        updates["current_stage"] = "stage_5b_ethics"

    return updates
