"""Stage 3A: Hypothesis Formation Node.

Guides student through hypothesis development:
1. Format system prompt with current state
2. Build conversation context, generate response
3. Extract Stage3AProgress with structured_call
4. Update scope_details, hypothesis, theory
5. Transition to stage_3b_info_gathering when hypothesis is complete
   and student is requesting dataset search
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.hypothesis.schemas import Stage3AProgress
from research_mentor.question_workshop.hypothesis.state import HypothesisState, hypothesis_config
from research_mentor.question_workshop.hypothesis.utils import (
    build_conversation_context,
    build_hypothesis_system_prompt,
)
from research_mentor.question_workshop.hypothesis.utils.prompts import (
    STAGE_3A_HYPOTHESIS_PROMPT,
    STAGE_3A_PROGRESS_EXTRACTION_PROMPT,
)


async def stage_3a_hypothesis(
    state: HypothesisState,
) -> dict[str, Any]:
    """Stage 3A: Guide student through hypothesis formation."""
    logger.info("Stage 3A Hypothesis: Processing...")

    messages = state.get("messages", [])
    independent_var = state.get("independent_var") or "N/A"
    dependent_var = state.get("dependent_var") or "N/A"
    scope_defined = state.get("scope_defined", False)
    hypothesis = state.get("hypothesis")
    theory = state.get("theory")

    # Step 1: Format system prompt with current state
    system_prompt = STAGE_3A_HYPOTHESIS_PROMPT.format(
        independent_var=independent_var,
        dependent_var=dependent_var,
        scope_defined=scope_defined,
        hypothesis=hypothesis or "Not yet defined",
        theory=theory or "Not yet defined",
    )

    # Step 2: Build conversation context and generate response
    system_prompt = build_hypothesis_system_prompt(system_prompt, state)
    recent_messages: list[Any] = [
        SystemMessage(content=system_prompt),
    ]
    for msg in messages[-10:]:
        recent_messages.append(msg)

    cfg = hypothesis_config()
    llm = get_chat_llm(agent="hypothesis", temperature=cfg.conversational_temperature)
    response_msg = await llm.ainvoke(recent_messages)
    response_text = str(response_msg.content)

    logger.debug(
        "Stage 3A response length: {} chars", len(response_text),
    )

    updates: dict[str, Any] = {
        "last_ai_response": response_text,
    }

    # Step 3: Extract Stage3AProgress with structured_call
    conversation_text = build_conversation_context(messages)
    extraction_prompt = STAGE_3A_PROGRESS_EXTRACTION_PROMPT.format(
        conversation=conversation_text,
    )

    progress = await structured_call(
        Stage3AProgress,
        [
            SystemMessage(content=extraction_prompt),
            HumanMessage(
                content="Extract Stage 3A hypothesis progress "
                "from the conversation above.",
            ),
        ],
        thinking="medium",
        temperature=cfg.extraction_temperature,
    )

    logger.info(
        "Stage 3A progress — scope: {}, hypothesis: {}, "
        "theory: {}, ready: {}",
        progress.scope_defined,
        progress.hypothesis is not None,
        progress.theory is not None,
        progress.ready_for_datasets,
    )

    # Step 4: Update scope_details, hypothesis, theory
    if progress.scope_defined:
        updates["scope_defined"] = True
        scope_details: dict[str, Any] = state.get(
            "scope_details", {},
        )
        if progress.scope_when:
            scope_details["when"] = progress.scope_when
        if progress.scope_where:
            scope_details["where"] = progress.scope_where
        if progress.scope_population:
            scope_details["population"] = progress.scope_population
        updates["scope_details"] = scope_details

    if progress.hypothesis:
        updates["hypothesis"] = progress.hypothesis
    if progress.hypothesis_direction:
        updates["hypothesis_direction"] = progress.hypothesis_direction
    if progress.theory:
        updates["theory"] = progress.theory

    # Step 5: Transition to stage_3b_info_gathering
    # when hypothesis is complete AND student is requesting datasets
    hypothesis_complete = (
        progress.scope_defined
        and progress.hypothesis is not None
        and progress.theory is not None
    )

    last_user_message = state.get("last_user_message", "")
    dataset_keywords = {
        "dataset", "data set", "search for data", "find data",
        "yes", "sure", "okay", "let's do it", "search",
    }
    student_requesting_datasets = any(
        kw in last_user_message.lower() for kw in dataset_keywords
    )
    response_mentions_datasets = "dataset" in response_text.lower()

    if hypothesis_complete and (
        student_requesting_datasets
        or progress.ready_for_datasets
        or response_mentions_datasets
    ):
        logger.info(
            "Stage 3A complete — transitioning to stage_3b_data",
        )
        updates["current_stage"] = "stage_3b_data"

    return updates
