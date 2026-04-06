"""Auto-generate and periodically review session titles.

Titles are generated on the first message and reviewed at message 5 and 15
(when the conversation topic may have shifted). User-set titles (title_source='user')
are never overwritten.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.db import crud
from research_mentor.llm import get_chat_llm, set_usage_context

# Message counts at which to re-evaluate the auto-generated title.
_REVIEW_AT_MESSAGES = {5, 15}

_TITLE_SYSTEM_PROMPT = """\
Your ONLY job is to generate a short title (2-8 words) for the conversation below.

Rules:
- Output ONLY the title, nothing else
- 2-8 words, no full sentences
- No quotes, no punctuation, no explanation
- Capture the main topic

Examples of good titles:
- Water Filtration Experiment Design
- Understanding P-Values
- Soil pH and Plant Growth
- Getting Unstuck on Research Project"""

_REVIEW_SYSTEM_PROMPT = """\
You are reviewing the title of an ongoing conversation. The current title may no longer \
reflect the main topic because the conversation has evolved.

Rules:
- Output ONLY the new title, nothing else
- 2-8 words, no full sentences
- No quotes, no punctuation, no explanation
- If the current title is still accurate, output it unchanged
- If the conversation has shifted focus, output a better title that captures the overall theme"""


async def _generate_title(prompt: str, system_prompt: str) -> str:
    """Call LLM to produce a title. Raises on failure."""
    llm = get_chat_llm()
    result = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ])
    content = result.content
    raw = (content if isinstance(content, str) else "").strip()
    # Take only the first line to guard against verbose LLM output
    title = raw.split("\n")[0].strip().strip("\"'")
    if len(title) > 80:
        title = title[:77] + "..."
    return title


async def generate_session_title(user_message: str, assistant_response: str) -> str:
    """Generate a concise title from the first exchange."""
    prompt = (
        f"Generate a 2-8 word title for this conversation:\n\n"
        f"Student: {user_message}\n\n"
        f"Mentor: {assistant_response[:500]}"
    )
    return await _generate_title(prompt, _TITLE_SYSTEM_PROMPT)


async def review_session_title(
    current_title: str,
    user_message: str,
    assistant_response: str,
) -> str:
    """Re-evaluate the title given the latest exchange and current title."""
    prompt = (
        f"Current title: {current_title}\n\n"
        f"Latest exchange:\n"
        f"Student: {user_message}\n\n"
        f"Mentor: {assistant_response[:500]}\n\n"
        f"Is the current title still the best fit, or should it change?"
    )
    return await _generate_title(prompt, _REVIEW_SYSTEM_PROMPT)


async def maybe_generate_title(
    session_id: str,
    user_message: str,
    assistant_response: str,
    message_count: int,
) -> None:
    """Generate or review the session title.

    - First message (no title yet): generate a new title.
    - Messages 5 and 15: review the title if it was auto-generated.
    - User-set titles (title_source='user') are never changed.

    Background task entry point — never raises.
    """
    try:
        set_usage_context(purpose="title", session_id=session_id)

        session = await crud.get_session(session_id)
        if session is None:
            return

        title_source = session.get("title_source", "auto")

        # Never touch user-set titles
        if title_source == "user":
            return

        current_title = session.get("title")

        if current_title is None:
            # First message — generate initial title
            title = await generate_session_title(user_message, assistant_response)
            if title:
                await crud.update_session(session_id, title=title, title_source="auto")
                logger.info("Auto-titled session {}: {!r}", session_id, title)
            return

        # Review at specific message counts
        if message_count in _REVIEW_AT_MESSAGES:
            new_title = await review_session_title(
                current_title, user_message, assistant_response,
            )
            if new_title and new_title != current_title:
                await crud.update_session(session_id, title=new_title, title_source="auto")
                logger.info(
                    "Revised title for session {} (msg {}): {!r} → {!r}",
                    session_id, message_count, current_title, new_title,
                )
    except Exception:
        logger.exception("Failed to generate/review title for session {}", session_id)
