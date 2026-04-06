"""Venue Discovery Node — data-driven venue matching + evaluation.

Three-step matching:
1. Filter curated venues by hard constraints (age, field, level, format)
2. Score remaining venues by soft factors (cost, quality, format preference)
3. LLM reasoning to produce personalized recommendations

Also handles venue evaluation and predatory detection teaching.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.sharing.schemas import VenueSearchParameters
from research_mentor.sharing.state import MAX_VENUE_RESULTS, SharingState
from research_mentor.sharing.utils import build_sharing_system_prompt
from research_mentor.sharing.utils.prompts import (
    VENUE_DISCOVERY_PROMPT,
    VENUE_EVALUATION_PROMPT,
)
from research_mentor.sharing.utils.venue_matching import (
    filter_venues,
    load_curated_venues,
    score_venues,
)


async def venue_discovery(state: SharingState) -> dict[str, Any]:
    """Match student to appropriate venues using curated data + LLM reasoning."""
    logger.info("Venue Discovery: Processing...")

    messages = state.get("messages", [])
    user_msg = state.get("last_user_message", "")
    venue_search_done = state.get("venue_search_done", False)

    # If we haven't searched yet, do the initial venue matching
    if not venue_search_done:
        return await _initial_venue_search(state)

    # Subsequent turns: discuss venues, evaluate specific ones, refine
    return await _venue_discussion(state, messages, user_msg)


async def _initial_venue_search(state: SharingState) -> dict[str, Any]:
    """First entry: search curated venues, optionally query OpenAlex, present results."""
    user_msg = state.get("last_user_message", "")
    messages = state.get("messages", [])

    # Extract search parameters from context
    try:
        params = await structured_call(
            VenueSearchParameters,
            [
                SystemMessage(content=(
                    "Extract venue search parameters from the student's context. "
                    "Use what's known from the conversation."
                )),
                HumanMessage(content=(
                    f"Research: {state.get('research_summary', 'unknown')}\n"
                    f"Output type: {state.get('output_type', 'unknown')}\n"
                    f"Level: {state.get('student_level', 'unknown')}\n"
                    f"Field: {state.get('research_field', 'unknown')}\n"
                    f"Latest message: {user_msg}"
                )),
            ],
            thinking="medium",
            temperature=0.0,
        )
    except Exception as e:
        logger.warning("Venue parameter extraction failed: {}", e)
        params = None

    # Determine filter parameters
    demographics = state.get("student_demographics", {})
    age = None
    if params and params.student_age:
        age = params.student_age
    elif demographics.get("age"):
        age = demographics["age"]

    education_level = (
        (params.education_level if params else None)
        or state.get("student_level")
    )
    field = (
        (params.field if params else None)
        or state.get("research_field")
    )
    format_type = (
        (params.output_format if params else None)
        or state.get("output_type")
    )
    cost_preference = (params.cost_preference if params else None) or "any"

    # Step 1: Filter curated venues
    all_venues = load_curated_venues()
    filtered = filter_venues(
        all_venues,
        age=age,
        education_level=education_level,
        field=field,
        format_type=format_type,
        exclude_isef=True,
    )
    logger.info(
        "Venue filter: {} total → {} after constraints (age={}, level={}, field={}, format={})",
        len(all_venues), len(filtered), age, education_level, field, format_type,
    )

    # Step 2: Score venues (format_type as soft preference, not hard filter)
    scored = score_venues(
        filtered,
        cost_preference=cost_preference,
        format_preference=format_type,
    )[:MAX_VENUE_RESULTS]

    # Step 3: Query OpenAlex for professional venues (university level only)
    api_venues: list[dict[str, Any]] = []
    if education_level == "university" and state.get("research_summary"):
        try:
            from research_mentor.tools.venue_discovery import search_openalex_sources

            summary = state.get("research_summary") or ""
            api_venues = await search_openalex_sources(
                str(summary),
                source_type="journal",
                is_oa=True,
                max_results=10,
            )
            # Filter out error results
            api_venues = [v for v in api_venues if "error" not in v]
            logger.info("OpenAlex sources: {} results", len(api_venues))
        except Exception as e:
            logger.warning("OpenAlex sources search failed: {}", e)

    # Format venue list for the LLM
    venue_list = _format_venue_list(scored, api_venues)

    # Generate response with venue recommendations
    # Use student_level/output_type as short labels, not research_summary
    # (research_summary from extraction may contain third-person artifacts
    # that leak as tool leakage when echoed by the LLM)
    prompt = VENUE_DISCOVERY_PROMPT.format(
        student_level=state.get("student_level") or "student",
        research_field=state.get("research_field") or "science",
        output_type=state.get("output_type") or "research work",
        research_summary="(see conversation above)",
        venue_list=venue_list,
    )
    system_prompt = build_sharing_system_prompt(prompt, state)
    llm = get_chat_llm(temperature=0.7, max_tokens=150)
    response = await llm.ainvoke(messages + [SystemMessage(content=system_prompt)])
    response_text = str(response.content)

    # Track that venue_discovery was visited
    phases_visited = list(state.get("phases_visited", []))
    if "venue_discovery" not in phases_visited:
        phases_visited.append("venue_discovery")

    return {
        "last_ai_response": response_text,
        "matched_venues": scored,
        "venue_search_done": True,
        "phases_visited": phases_visited,
        "info_gathering_history": state.get("info_gathering_history", []) + [{
            "tool": "venue_matching",
            "curated_count": len(scored),
            "api_count": len(api_venues),
        }],
    }


async def _venue_discussion(
    state: SharingState,
    messages: list[Any],
    user_msg: str,
) -> dict[str, Any]:
    """Subsequent turns: discuss venues, evaluate specific ones, refine search."""
    # Check if student is asking about a specific venue (evaluation mode)
    user_lower = user_msg.lower()
    is_evaluation = any(
        kw in user_lower
        for kw in ("predatory", "legitimate", "trustworthy", "real", "scam", "quality", "safe")
    )

    if is_evaluation:
        prompt = VENUE_EVALUATION_PROMPT.format(venue_name=user_msg[:200])
    else:
        matched = state.get("matched_venues", [])
        venue_list = _format_venue_list(matched, [])
        prompt = VENUE_DISCOVERY_PROMPT.format(
            student_level=state.get("student_level", "unknown"),
            research_field=state.get("research_field", "unknown"),
            output_type=state.get("output_type", "unknown"),
            research_summary=state.get("research_summary", "unknown"),
            venue_list=venue_list,
        )

    system_prompt = build_sharing_system_prompt(prompt, state)
    llm = get_chat_llm(temperature=0.7, max_tokens=200)
    response = await llm.ainvoke(messages + [SystemMessage(content=system_prompt)])

    return {
        "last_ai_response": str(response.content),
        "venue_evaluation_done": is_evaluation or state.get("venue_evaluation_done", False),
    }


def _format_venue_list(
    curated: list[dict[str, Any]],
    api_venues: list[dict[str, Any]],
) -> str:
    """Format venues as a compact list — name + one-line reason only.

    Deliberately minimal to prevent the model from elaborating on metadata.
    """
    lines: list[str] = []

    for v in curated[:8]:
        notes = v.get("notes", "")
        # One-line: just name and the first sentence of notes
        brief = notes.split(".")[0] if notes else v.get("type", "")
        lines.append(f"- {v['name']}: {brief}")

    for v in api_venues[:3]:
        oa_tag = "open access" if v.get("is_oa") else ""
        lines.append(f"- {v['display_name']}: {oa_tag} journal")

    if not lines:
        lines.append("(No matches found — suggest school presentations or local events)")

    return "\n".join(lines)
