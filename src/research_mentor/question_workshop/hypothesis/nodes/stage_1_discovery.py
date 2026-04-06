"""Stage 1: Discovery Node — variable identification.

Guides student through identifying X (independent) and Y (dependent) variables.

1. Format system prompt with current variables
2. Build conversation context (last 10 messages)
3. Generate response with chat LLM
4. Extract variables with structured_call
5. Validate with validate_stage_1_completion()
6. If valid: transition to stage_2_literature
7. If invalid with feedback: replace response with feedback
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.hypothesis.schemas import VariableExtraction
from research_mentor.question_workshop.hypothesis.state import HypothesisState, hypothesis_config
from research_mentor.question_workshop.hypothesis.utils import (
    build_conversation_context,
    build_hypothesis_system_prompt,
)
from research_mentor.question_workshop.hypothesis.utils.prompts import (
    STAGE_1_DISCOVERY_PROMPT,
    VARIABLE_EXTRACTION_PROMPT,
)
from research_mentor.question_workshop.hypothesis.utils.validation import (
    validate_stage_1_completion,
)


async def stage_1_discovery(state: HypothesisState) -> dict[str, Any]:
    """Stage 1: Guide student through variable identification."""
    logger.info("Stage 1 Discovery: Processing...")

    messages = state.get("messages", [])
    independent_var = state.get("independent_var")
    dependent_var = state.get("dependent_var")

    # Step 1: Format system prompt with current variables
    system_prompt = STAGE_1_DISCOVERY_PROMPT.format(
        independent_var=independent_var or "Not yet identified",
        dependent_var=dependent_var or "Not yet identified",
    )

    # Step 2: Build conversation context (last 10 messages)
    system_prompt = build_hypothesis_system_prompt(system_prompt, state)
    recent_messages: list[Any] = [SystemMessage(content=system_prompt)]
    for msg in messages[-10:]:
        recent_messages.append(msg)

    # Step 3: Generate response with chat LLM
    cfg = hypothesis_config()
    llm = get_chat_llm(agent="hypothesis", temperature=cfg.conversational_temperature)
    response_msg = await llm.ainvoke(recent_messages)
    response_text = str(response_msg.content)

    logger.debug(
        "Stage 1 response length: {} chars", len(response_text),
    )

    # Step 4: Extract variables with structured_call
    conversation_text = build_conversation_context(messages)
    extraction_prompt = VARIABLE_EXTRACTION_PROMPT.format(
        conversation=conversation_text,
    )

    extracted = await structured_call(
        VariableExtraction,
        [
            SystemMessage(content=extraction_prompt),
            HumanMessage(
                content="Extract variables from the conversation above.",
            ),
        ],
        thinking="medium",
        temperature=cfg.extraction_temperature,
    )

    logger.info(
        "Extracted variables — X: {}, Y: {}, confirmed: {}",
        extracted.independent_var, extracted.dependent_var, extracted.student_confirmed,
    )

    # Update state with extracted variables
    updates: dict[str, Any] = {"last_ai_response": response_text}

    if extracted.independent_var:
        updates["independent_var"] = extracted.independent_var
    if extracted.dependent_var:
        updates["dependent_var"] = extracted.dependent_var

    # Step 5: Validate with validate_stage_1_completion
    # Build a temp state with updated variables for validation
    validation_state = {**state, **updates}
    is_valid, feedback = await validate_stage_1_completion(validation_state)

    # Step 6: If valid, transition to stage_2_literature
    # Require at least 2 student messages AND explicit student confirmation.
    # Vague agreement ("yeah", "sure") is not confirmation — the student
    # must name or clearly reference specific variables themselves.
    human_messages = [
        m for m in messages if isinstance(m, HumanMessage)
    ]
    if is_valid and extracted.student_confirmed and len(human_messages) >= 2:
        logger.info("Stage 1 complete — transitioning to stage_2_literature")
        updates["current_stage"] = "stage_2_literature"
        return updates
    elif is_valid and not extracted.student_confirmed:
        logger.info(
            "Stage 1 variables valid but student hasn't explicitly confirmed — "
            "staying to get clear confirmation",
        )
    elif is_valid:
        logger.info(
            "Stage 1 variables valid but only {} student message(s) — "
            "staying to confirm",
            len(human_messages),
        )

    # Step 7: If invalid and feedback, replace response with feedback
    if feedback:
        logger.info("Stage 1 validation failed — returning feedback")
        updates["last_ai_response"] = feedback

    return updates
