"""Stage 3B Data Node — dataset search + feasibility discussion.

Merged node handling the complete Stage 3B flow:
1. First entry (no datasets): search web for datasets → END
2. Subsequent entries: discuss feasibility, extract progress,
   transition to refinement or confounds when ready → END
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.hypothesis.schemas import Stage3BProgress
from research_mentor.question_workshop.hypothesis.state import HypothesisState, hypothesis_config
from research_mentor.question_workshop.hypothesis.utils import (
    build_conversation_context,
    build_hypothesis_system_prompt,
)
from research_mentor.question_workshop.hypothesis.utils.prompts import (
    STAGE_3B_DATASETS_PROMPT,
    STAGE_3B_PROGRESS_EXTRACTION_PROMPT,
)
from research_mentor.tools.web_search import search_web


def _format_datasets_for_display(
    results: list[dict[str, Any]],
) -> str:
    """Format search results as a readable dataset list."""
    if not results:
        return "No relevant datasets found."

    lines = [f"**I found {len(results)} potential datasets:**\n"]
    for i, r in enumerate(results[:10], 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        description = r.get("description", "No description")
        desc_preview = (
            description[:200] + "..." if len(description) > 200
            else description
        )
        lines.append(
            f"**{i}. {title}**\n"
            f"   {desc_preview}\n"
            + (f"   {url}\n" if url else "")
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Search sub-step (runs on first entry only)
# ---------------------------------------------------------------------------


async def _search_datasets(state: HypothesisState) -> dict[str, Any]:
    """Search web for datasets matching the hypothesis."""
    independent_var = state.get("independent_var", "")
    dependent_var = state.get("dependent_var", "")
    hypothesis = state.get("hypothesis", "")

    query_parts = []
    if independent_var:
        query_parts.append(independent_var)
    if dependent_var:
        query_parts.append(dependent_var)
    if not query_parts and hypothesis:
        query_parts.append(hypothesis)
    query = " ".join(query_parts) + " dataset"

    logger.info("Searching for datasets with query: '{}'", query)

    search_result = await search_web(query, num_results=10)

    if not search_result.get("success"):
        error = search_result.get("error", "Unknown search error")
        logger.warning("Dataset search failed: {}", error)
        return {
            "datasets_searched": True,
            "last_ai_response": (
                "I tried to search for datasets but encountered an issue. "
                "Let's discuss what kind of data you might need."
            ),
            "info_gathering_history": [
                *state.get("info_gathering_history", []),
                {"action": "dataset_search", "success": False, "error": error},
            ],
        }

    raw_results = search_result.get("results", [])
    datasets = [
        {
            "title": r.get("title", "Untitled"),
            "url": r.get("url", ""),
            "description": r.get("description", ""),
            "source": r.get("source", "web"),
        }
        for r in raw_results
    ]

    logger.info("Found {} dataset results", len(datasets))

    formatted = _format_datasets_for_display(raw_results)
    offer = (
        "\n\nWhich of these datasets looks most promising for testing "
        "your hypothesis? Consider the variables, time period, and "
        "population coverage."
    )

    return {
        "datasets_searched": True,
        "datasets_found": datasets,
        "datasets_count": len(datasets),
        "last_ai_response": formatted + offer,
        "info_gathering_history": [
            *state.get("info_gathering_history", []),
            {
                "action": "dataset_search",
                "success": True,
                "query": query,
                "count": len(datasets),
            },
        ],
    }


# ---------------------------------------------------------------------------
# Discussion sub-step (runs on subsequent entries)
# ---------------------------------------------------------------------------


async def _discuss_datasets(state: HypothesisState) -> dict[str, Any]:
    """Handle the dataset feasibility discussion."""
    messages = state.get("messages", [])
    hypothesis = state.get("hypothesis", "Not yet formed")
    scope_details = state.get("scope_details", {})
    datasets_count = state.get("datasets_count", 0)

    system_prompt = STAGE_3B_DATASETS_PROMPT.format(
        hypothesis=hypothesis,
        scope_details=scope_details or "Not yet defined",
        datasets_searched=True,
        datasets_count=datasets_count,
    )

    system_prompt = build_hypothesis_system_prompt(system_prompt, state)
    recent_messages: list[Any] = [SystemMessage(content=system_prompt)]
    for msg in messages[-10:]:
        recent_messages.append(msg)

    cfg = hypothesis_config()
    llm = get_chat_llm(agent="hypothesis", temperature=cfg.conversational_temperature)
    response_msg = await llm.ainvoke(recent_messages)
    response_text = str(response_msg.content)

    logger.debug(
        "Stage 3B response length: {} chars", len(response_text),
    )

    # Extract progress
    conversation_text = build_conversation_context(messages)
    extraction_prompt = STAGE_3B_PROGRESS_EXTRACTION_PROMPT.format(
        conversation=conversation_text,
    )

    progress = await structured_call(
        Stage3BProgress,
        [
            SystemMessage(content=extraction_prompt),
            HumanMessage(
                content="Extract dataset evaluation progress.",
            ),
        ],
        thinking="medium",
        temperature=cfg.extraction_temperature,
    )

    logger.info(
        "Stage 3B progress — dataset_selected: {}, "
        "needs_scope_adjustment: {}, ready_for_confounds: {}",
        progress.dataset_selected,
        progress.needs_scope_adjustment,
        progress.ready_for_confounds,
    )

    updates: dict[str, Any] = {"last_ai_response": response_text}

    if progress.dataset_selected and progress.selected_dataset_name:
        updates["selected_dataset"] = {
            "name": progress.selected_dataset_name,
            "feasibility_concerns": progress.feasibility_concerns,
        }

    # Transition — backward jumps checked first
    if progress.needs_variable_rethink:
        logger.info(
            "Stage 3B → Stage 1: no viable datasets, variables need rethinking",
        )
        updates["current_stage"] = "stage_1_discovery"
    elif progress.needs_scope_adjustment and progress.dataset_selected:
        logger.info(
            "Stage 3B → Stage 3C: scope adjustment needed",
        )
        updates["current_stage"] = "stage_3c_refinement"
    elif progress.ready_for_confounds and progress.dataset_selected:
        logger.info(
            "Stage 3B → Stage 3.5: ready for confounds",
        )
        updates["current_stage"] = "stage_3_5_confounds"

    return updates


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def stage_3b_data(state: HypothesisState) -> dict[str, Any]:
    """Stage 3B: Dataset search + feasibility discussion.

    First entry (no datasets): searches web, shows results.
    Subsequent entries: feasibility discussion with progress extraction.
    """
    logger.info("Stage 3B Data: Processing...")

    datasets_searched = state.get("datasets_searched", False)

    if not datasets_searched:
        return await _search_datasets(state)

    return await _discuss_datasets(state)
