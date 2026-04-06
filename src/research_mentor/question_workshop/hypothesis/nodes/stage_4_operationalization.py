"""Stage 4 Operationalization Node — measurement quality evaluation.

Guides student through evaluating measurement quality:
1. Format system prompt with selected dataset info
2. Generate conversational response with chat LLM
3. Extract MeasurementExtraction with structured_call
4. Extract OperationalizationProgress with structured_call
5. If data_alignment_evaluated -> stage_5a_analysis
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.hypothesis.schemas import (
    MeasurementExtraction,
    OperationalizationProgress,
)
from research_mentor.question_workshop.hypothesis.state import HypothesisState, hypothesis_config
from research_mentor.question_workshop.hypothesis.utils import (
    build_conversation_context,
    build_hypothesis_system_prompt,
)
from research_mentor.question_workshop.hypothesis.utils.prompts import (
    MEASUREMENT_EXTRACTION_PROMPT,
    OPERATIONALIZATION_EXTRACTION_PROMPT,
    STAGE_4_OPERATIONALIZATION_PROMPT,
)


async def stage_4_operationalization(
    state: HypothesisState,
) -> dict[str, Any]:
    """Stage 4: Evaluate measurement quality and data alignment."""
    logger.info("Stage 4 Operationalization: Processing...")

    messages = state.get("messages", [])
    independent_var = state.get("independent_var", "")
    dependent_var = state.get("dependent_var", "")
    selected_dataset = state.get("selected_dataset")

    # Step 1: Format system prompt
    dataset_info = (
        selected_dataset.get("name", "Unknown")
        if selected_dataset
        else "No dataset selected"
    )
    system_prompt = STAGE_4_OPERATIONALIZATION_PROMPT.format(
        selected_dataset=dataset_info,
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
        "Stage 4 response length: {} chars", len(response_text),
    )

    # Step 3: Extract measurement specifications
    conversation_text = build_conversation_context(messages)
    measurement_prompt = MEASUREMENT_EXTRACTION_PROMPT.format(
        independent_var=independent_var or "unknown",
        dependent_var=dependent_var or "unknown",
        conversation=conversation_text,
    )

    measurements = await structured_call(
        MeasurementExtraction,
        [
            SystemMessage(content=measurement_prompt),
            HumanMessage(
                content="Extract measurement specifications.",
            ),
        ],
        thinking="medium",
        temperature=cfg.extraction_temperature,
    )

    logger.info(
        "Measurements — X: {}, Y: {}",
        measurements.x_measurement, measurements.y_measurement,
    )

    # Step 4: Extract operationalization progress
    op_prompt = OPERATIONALIZATION_EXTRACTION_PROMPT.format(
        conversation=conversation_text,
    )

    progress = await structured_call(
        OperationalizationProgress,
        [
            SystemMessage(content=op_prompt),
            HumanMessage(
                content="Extract operationalization progress.",
            ),
        ],
        thinking="medium",
        temperature=cfg.extraction_temperature,
    )

    logger.info(
        "Stage 4 progress — data_alignment_evaluated: {}",
        progress.data_alignment_evaluated,
    )

    # Build state updates
    updates: dict[str, Any] = {"last_ai_response": response_text}

    if measurements.x_measurement:
        updates["x_measurement"] = measurements.x_measurement
    if measurements.y_measurement:
        updates["y_measurement"] = measurements.y_measurement
    if progress.data_alignment_evaluated:
        updates["data_alignment_evaluated"] = True
    if progress.measurement_quality_notes:
        updates["measurement_quality_notes"] = (
            progress.measurement_quality_notes
        )

    # Step 5: Check for backward jump or forward transition
    if progress.alignment_too_poor:
        logger.info(
            "Stage 4 → Stage 3C: data alignment too poor, need scope revision",
        )
        updates["current_stage"] = "stage_3c_refinement"
    elif progress.data_alignment_evaluated:
        logger.info(
            "Stage 4 complete → Stage 5A",
        )
        updates["current_stage"] = "stage_5a_analysis"

    return updates
