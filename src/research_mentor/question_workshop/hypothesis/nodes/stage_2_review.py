"""Stage 2 Review Node — guided/independent paper review.

Runs on subsequent entries to Stage 2 (papers already found):
1. Detect review mode choice (guided vs independent) via LLM
2. Generate review response with chat LLM
3. Extract progress (gap, measurements) with structured_call
4. Transition to Stage 3A when gap + ideal measurements identified
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.hypothesis.schemas import (
    ReviewModeClassification,
    Stage2Progress,
)
from research_mentor.question_workshop.hypothesis.state import HypothesisState, hypothesis_config
from research_mentor.question_workshop.hypothesis.utils import (
    build_conversation_context,
    build_hypothesis_system_prompt,
)
from research_mentor.question_workshop.hypothesis.utils.prompts import (
    STAGE_2_DEFAULT,
    STAGE_2_GUIDED_AFTER_CHOICE,
    STAGE_2_GUIDED_REVIEW,
    STAGE_2_INDEPENDENT_AFTER_CHOICE,
    STAGE_2_INDEPENDENT_REVIEW,
    STAGE_2_PROGRESS_EXTRACTION_PROMPT,
)

_PAPERS_ALREADY_SHOWN = (
    "\n\nIMPORTANT: You already found and displayed {papers_found_count} papers "
    "to the student. DO NOT offer to search again, DO NOT say 'let me search', "
    "DO NOT reference 'the search' or 'searching'. The papers are already visible. "
    "Discuss them directly."
)


async def _detect_review_mode(user_message: str) -> str | None:
    """Detect review mode choice from user message using LLM classification."""
    result = await structured_call(
        ReviewModeClassification,
        [
            SystemMessage(
                content=(
                    "The student was just shown research papers and offered two ways "
                    "to review them:\n"
                    "1. **Guided** — the mentor walks through papers one by one together\n"
                    "2. **Independent** — the student reviews papers on their own\n\n"
                    "Classify whether the student's message expresses a preference. "
                    "If the message is about something else entirely, return null."
                ),
            ),
            HumanMessage(content=f"Student message: {user_message}"),
        ],
        thinking="off",
        temperature=hypothesis_config().extraction_temperature,
    )
    mode = result.mode
    if mode and mode not in ("guided", "independent"):
        logger.warning("Unexpected review mode from LLM: {!r}, ignoring", mode)
        return None
    return mode


def _format_system_prompt(
    state: HypothesisState, review_mode: str | None,
) -> str:
    """Select and format the appropriate system prompt."""
    papers_found_count = state.get("papers_found_count", 0)
    format_vars = {
        "independent_var": state.get("independent_var") or "N/A",
        "dependent_var": state.get("dependent_var") or "N/A",
        "papers_found_count": papers_found_count,
        "papers_discussed_count": state.get(
            "papers_discussed_count", 0,
        ),
        "gap_identified": state.get("gap_identified", False),
    }

    if review_mode == "guided":
        prompt = STAGE_2_GUIDED_REVIEW.format(**format_vars)
    elif review_mode == "independent":
        prompt = STAGE_2_INDEPENDENT_REVIEW.format(**format_vars)
    else:
        prompt = STAGE_2_DEFAULT.format(**format_vars)

    if papers_found_count > 0:
        prompt += _PAPERS_ALREADY_SHOWN.format(
            papers_found_count=papers_found_count,
        )

    return prompt


async def stage_2_review(state: HypothesisState) -> dict[str, Any]:
    """Stage 2 Review: guided/independent paper review conversation."""
    logger.info("Stage 2 Review: Processing...")

    messages = state.get("messages", [])
    last_user_message = state.get("last_user_message", "")
    review_mode = state.get("review_mode")
    papers_found_count = state.get("papers_found_count", 0)
    gap_identified = state.get("gap_identified", False)

    updates: dict[str, Any] = {}

    # Detect review mode choice
    if not review_mode:
        detected = await _detect_review_mode(last_user_message)
        if detected:
            logger.info("Review mode detected: {}", detected)
            updates["review_mode"] = detected
            review_mode = detected

            if detected == "guided":
                updates["last_ai_response"] = (
                    STAGE_2_GUIDED_AFTER_CHOICE
                )
            else:
                updates["last_ai_response"] = (
                    STAGE_2_INDEPENDENT_AFTER_CHOICE
                )
            return updates

    # Generate review response
    system_prompt = _format_system_prompt(state, review_mode)
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
    updates["last_ai_response"] = response_text

    logger.debug(
        "Stage 2 review response length: {} chars",
        len(response_text),
    )

    # Extract progress
    conversation_text = build_conversation_context(messages)
    extraction_prompt = STAGE_2_PROGRESS_EXTRACTION_PROMPT.format(
        conversation=conversation_text,
    )

    progress = await structured_call(
        Stage2Progress,
        [
            SystemMessage(content=extraction_prompt),
            HumanMessage(
                content="Extract Stage 2 progress from the "
                "conversation above.",
            ),
        ],
        thinking="medium",
        temperature=cfg.extraction_temperature,
    )

    logger.info(
        "Stage 2 progress — gap: {} (desc: {}), mode: {}, "
        "X meas: {!r}, Y meas: {!r}",
        progress.gap_identified,
        progress.gap_description[:60] if progress.gap_description else None,
        progress.review_mode,
        progress.ideal_measurements_x,
        progress.ideal_measurements_y,
    )

    # Update state from extraction
    if progress.gap_identified:
        updates["gap_identified"] = True
    if progress.gap_description:
        updates["gap_description"] = progress.gap_description
    if progress.review_mode and not review_mode:
        updates["review_mode"] = progress.review_mode

    ideal_measurements = state.get("ideal_measurements", {})
    if progress.ideal_measurements_x:
        ideal_measurements["X"] = progress.ideal_measurements_x
    if progress.ideal_measurements_y:
        ideal_measurements["Y"] = progress.ideal_measurements_y
    if ideal_measurements:
        updates["ideal_measurements"] = ideal_measurements

    # Transition check
    has_papers = papers_found_count > 0
    has_gap = progress.gap_identified or gap_identified
    has_ideal = bool(
        ideal_measurements.get("X") and ideal_measurements.get("Y")
    )

    logger.info(
        "Stage 2 transition check — has_papers: {}, has_gap: {}, "
        "has_ideal: {} (X={!r}, Y={!r})",
        has_papers, has_gap, has_ideal,
        ideal_measurements.get("X"), ideal_measurements.get("Y"),
    )

    if has_papers and has_gap and has_ideal:
        logger.info(
            "Stage 2 complete — next invocation routes to stage_3a_hypothesis",
        )
        updates["current_stage"] = "stage_3a_hypothesis"

    return updates
