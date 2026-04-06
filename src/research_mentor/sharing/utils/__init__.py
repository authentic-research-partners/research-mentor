"""Sharing utilities — prompts, venue matching, and shared helpers."""

from __future__ import annotations

from typing import Any

from research_mentor.agent.prompts.shared import build_student_profile_context
from research_mentor.components_registry import get_catalog_prompt
from research_mentor.sharing.state import SharingState


def build_sharing_system_prompt(base_prompt: str, state: SharingState) -> str:
    """Wrap a phase's base system prompt with student context and language.

    Prepends student demographics (age, country, grade) so the LLM adjusts
    complexity. Appends language instruction when language != "en".
    """
    parts: list[str] = []

    # Student context
    demographics: dict[str, Any] = state.get("student_demographics", {})
    profile_text = build_student_profile_context(demographics)
    if profile_text:
        parts.append(profile_text)

    # Age-specific vocabulary adaptation
    age = demographics.get("age")
    if age and isinstance(age, int) and age < 18:
        parts.append(
            "**VOCABULARY:** This is a young student. Use SIMPLE everyday language. "
            "When you must use a scientific term, IMMEDIATELY explain it with an "
            "analogy or simple definition in parentheses. Avoid jargon."
        )

    # Component catalog (other available workshops)
    parts.append(get_catalog_prompt(exclude="sharing"))

    # Base phase prompt
    parts.append(base_prompt)

    # NOTE: research_summary and output_type are NOT injected here.
    # They live in the conversation history (which the LLM sees via messages).
    # Injecting extraction artifacts like "The student conducted a study on..."
    # causes tool leakage when the LLM echoes them verbatim.
    # Nodes that need research context (venue_discovery) format it themselves.

    # Highlight latest user message
    last_msg = state.get("last_user_message", "")
    if last_msg:
        parts.append(
            f'The student just said: "{last_msg}"\n'
            f"Address what they said."
        )

    # Language instruction
    language = state.get("language", "en")
    if language and language != "en":
        parts.append(
            f"\n**LANGUAGE:** Respond ENTIRELY in {language}. "
            f"Use appropriate scientific terminology in {language}. "
            f"Do NOT mix languages."
        )

    return "\n\n".join(parts)


def build_conversation_context(messages: list[Any], max_messages: int = 10) -> str:
    """Convert message list to text for LLM extraction.

    Formats recent messages as "User: ... / Sharing: ..." plain text
    for structured_call extraction prompts.
    """
    from langchain_core.messages import HumanMessage

    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    lines = []
    for msg in recent:
        role = "User" if isinstance(msg, HumanMessage) else "Sharing"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)
