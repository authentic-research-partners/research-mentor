"""Writing Support Node — writing mentorship (NOT ghostwriting).

Provides:
- Structure guidance (organizing documents)
- Feedback on drafts (critique, not rewriting)
- Revision coaching (interpreting and responding to feedback)
- Simulated peer review (role-playing as a reviewer)

Critical constraint: never produce text the student could submit directly.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.sharing.schemas import WritingFocusDetection
from research_mentor.sharing.state import SharingState
from research_mentor.sharing.utils import build_sharing_system_prompt
from research_mentor.sharing.utils.prompts import WRITING_SUPPORT_PROMPT


async def writing_support(state: SharingState) -> dict[str, Any]:
    """Provide writing mentorship — advice about writing, not replacement."""
    logger.info("Writing Support: Processing...")

    messages = state.get("messages", [])
    user_msg = state.get("last_user_message", "")
    current_focus = state.get("writing_focus") or "structure"

    # Detect what kind of writing help the student needs
    try:
        focus_result = await structured_call(
            WritingFocusDetection,
            [
                SystemMessage(content=(
                    "Classify what kind of writing help the student needs "
                    "based on their latest message. Current focus: "
                    + current_focus
                )),
                HumanMessage(content=f"Student's message: {user_msg}"),
            ],
            thinking="medium",
            temperature=0.0,
        )
        writing_focus: str = focus_result.writing_focus
        draft_section = focus_result.draft_section
    except Exception as e:
        logger.warning("Writing focus detection failed: {}", e)
        writing_focus = current_focus
        draft_section = None

    # Track sections discussed
    sections_discussed = list(state.get("draft_sections_discussed", []))
    if draft_section and draft_section not in sections_discussed:
        sections_discussed.append(draft_section)

    # Generate response
    prompt = WRITING_SUPPORT_PROMPT.format(writing_focus=writing_focus)
    system_prompt = build_sharing_system_prompt(prompt, state)
    llm = get_chat_llm(temperature=0.7, max_tokens=120)
    response = await llm.ainvoke(messages + [SystemMessage(content=system_prompt)])
    response_text = str(response.content)

    updates: dict[str, Any] = {
        "last_ai_response": response_text,
        "writing_focus": writing_focus,
        "draft_sections_discussed": sections_discussed,
    }

    # Track that writing_support was visited
    phases_visited = list(state.get("phases_visited", []))
    if "writing_support" not in phases_visited:
        phases_visited.append("writing_support")
        updates["phases_visited"] = phases_visited

    return updates
