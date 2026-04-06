"""Completion Handler Node — final stage of Hypothesis workflow.

Handles the completion flow after all stages are done:
1. Classify user intent with structured_call(CompletionIntent)
2. Generate conversational response with chat LLM
3. If wants_pdf -> set pdf_requested=True
4. If has_question -> stay, re-offer PDF
5. If wants_to_end -> set workflow_complete=True
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.components_registry import get_completion_suggestions_prompt
from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.hypothesis.schemas import CompletionIntent
from research_mentor.question_workshop.hypothesis.state import HypothesisState, hypothesis_config
from research_mentor.question_workshop.hypothesis.utils.prompts import (
    COMPLETION_HANDLER_PROMPT,
    INTENT_CLASSIFICATION_PROMPT,
)


async def completion_handler(state: HypothesisState) -> dict[str, Any]:
    """Handle completion: classify intent, respond, route."""
    logger.info("Completion Handler: Processing...")

    cfg = hypothesis_config()
    messages = state.get("messages", [])
    user_message = state.get("last_user_message", "")

    # Step 1: Classify user intent
    intent_prompt = INTENT_CLASSIFICATION_PROMPT.format(
        user_message=user_message,
    )

    intent = await structured_call(
        CompletionIntent,
        [
            SystemMessage(content=intent_prompt),
            HumanMessage(content=user_message),
        ],
        thinking="off",
        temperature=cfg.extraction_temperature,
    )

    logger.info(
        "Completion intent — pdf: {}, question: {}, end: {}",
        intent.wants_pdf, intent.has_question, intent.wants_to_end,
    )

    # Step 2: Generate conversational response
    response_prompt = COMPLETION_HANDLER_PROMPT.format(
        user_message=user_message,
        independent_var=state.get("independent_var") or "Not set",
        dependent_var=state.get("dependent_var") or "Not set",
        hypothesis=state.get("hypothesis") or "Not set",
        statistical_method=state.get("statistical_method") or "Not set",
        next_steps=get_completion_suggestions_prompt("hypothesis"),
    )

    # Build message list: system prompt + recent conversation
    recent_messages: list[Any] = [
        SystemMessage(content=response_prompt),
    ]
    for msg in messages[-6:]:
        recent_messages.append(msg)

    llm = get_chat_llm(agent="hypothesis", temperature=cfg.conversational_temperature)
    response_msg = await llm.ainvoke(recent_messages)
    response_text = str(response_msg.content)

    logger.debug(
        "Completion response length: {} chars", len(response_text),
    )

    # Build state updates
    updates: dict[str, Any] = {"last_ai_response": response_text}

    # Step 3: Route based on intent
    if intent.wants_pdf:
        logger.info("Student wants PDF summary")
        updates["pdf_requested"] = True

    if intent.wants_to_end:
        logger.info("Student wants to end — workflow complete")
        updates["workflow_complete"] = True

    # If has_question: stay in completion (no transition flags set),
    # the response already re-offers PDF per the prompt instructions

    if intent.has_question:
        logger.info(
            "Student has follow-up question about: {}",
            intent.question_topic or "general",
        )

    return updates
