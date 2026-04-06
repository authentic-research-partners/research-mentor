"""Completion Handler Node — summary + new questions feedback loop.

Wraps up the Sharing session by:
1. Summarizing what was discussed
2. Asking if communication surfaced new research questions
3. Suggesting next steps (including taking new questions to Question Workshop)

Key principle: communication feeds back into new discovery.
"""

from typing import Any

from langchain_core.messages import SystemMessage
from loguru import logger

from research_mentor.components_registry import get_completion_suggestions_prompt
from research_mentor.llm import get_chat_llm
from research_mentor.sharing.state import SharingState
from research_mentor.sharing.utils import build_sharing_system_prompt
from research_mentor.sharing.utils.prompts import COMPLETION_PROMPT


async def completion_handler(state: SharingState) -> dict[str, Any]:
    """Generate session summary and identify new questions."""
    logger.info("Completion Handler: Wrapping up session...")

    messages = state.get("messages", [])
    phases_visited = state.get("phases_visited", [])
    research_summary = state.get("research_summary") or "your research"

    # Generate completion response
    prompt = COMPLETION_PROMPT.format(
        phases_visited=", ".join(phases_visited) if phases_visited else "context gathering",
        research_summary=research_summary,
        next_steps=get_completion_suggestions_prompt("sharing"),
    )
    system_prompt = build_sharing_system_prompt(prompt, state)
    llm = get_chat_llm(temperature=0.7, max_tokens=250)
    response = await llm.ainvoke(messages + [SystemMessage(content=system_prompt)])
    response_text = str(response.content)

    # Track completion
    phases_visited_updated = list(phases_visited)
    if "completion" not in phases_visited_updated:
        phases_visited_updated.append("completion")

    return {
        "last_ai_response": response_text,
        "current_phase": "completion",
        "phases_visited": phases_visited_updated,
        "workflow_complete": True,
    }
