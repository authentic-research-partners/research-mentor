"""Component registry — central catalog of Research Mentor flows.

All other modules should use functions here to access component metadata,
redirect phrases, and prompt fragments. Never import from
components_registry_data directly.
"""

from __future__ import annotations

from research_mentor.components_registry_data import (
    COMPONENTS,
    REDIRECT_PHRASES,
    ComponentInfo,
)

__all__ = [
    "ComponentInfo",
    "get_all_components",
    "get_catalog_prompt",
    "get_completion_suggestions",
    "get_completion_suggestions_prompt",
    "get_component",
    "get_redirect_phrase",
]


def get_component(component_id: str) -> ComponentInfo:
    """Return metadata for a single component. Raises ``KeyError`` if unknown."""
    return COMPONENTS[component_id]


def get_all_components() -> dict[str, ComponentInfo]:
    """Return the full component catalog."""
    return COMPONENTS


def get_catalog_prompt(
    exclude: str | set[str] | None = None,
    *,
    include_ids: bool = False,
) -> str:
    """Build a short prompt fragment listing available components.

    Args:
        exclude: component id(s) to omit (typically the current flow).
        include_ids: if True, include internal IDs (for component_recommender
            routing). Default False — IDs in student-facing prompts cause
            tool leakage when the LLM echoes them.

    Returns a ready-to-inject string, or ``""`` if nothing to list.
    """
    if exclude is None:
        skip: set[str] = set()
    elif isinstance(exclude, str):
        skip = {exclude}
    else:
        skip = set(exclude)

    lines: list[str] = []
    for cid, info in COMPONENTS.items():
        if cid in skip:
            continue
        if include_ids:
            lines.append(f"- {info.name} (id: {cid}): {info.one_liner}")
        else:
            lines.append(f"- {info.name}: {info.one_liner}")

    if not lines:
        return ""

    return "OTHER AVAILABLE RESOURCES (mention only if relevant):\n" + "\n".join(lines)


def get_redirect_phrase(from_id: str, to_id: str) -> str | None:
    """Return the pre-computed redirect phrase for a specific pair.

    Returns ``None`` if no redirect phrase exists for this pair.
    """
    return REDIRECT_PHRASES.get((from_id, to_id))


def get_completion_suggestions(
    from_id: str,
) -> list[tuple[ComponentInfo, str | None]]:
    """Return suggested next-step components for completion handlers.

    Returns a list of ``(ComponentInfo, redirect_phrase_or_None)`` tuples,
    ordered by the component's ``completion_suggestions`` field.
    """
    info = COMPONENTS.get(from_id)
    if not info:
        return []

    result: list[tuple[ComponentInfo, str | None]] = []
    for target_id in info.completion_suggestions:
        target = COMPONENTS.get(target_id)
        if target:
            phrase = REDIRECT_PHRASES.get((from_id, target_id))
            result.append((target, phrase))
    return result


def get_completion_suggestions_prompt(from_id: str) -> str:
    """Build a prompt fragment for completion handlers listing next steps.

    Returns a ready-to-inject string, or ``""`` if no suggestions exist.
    """
    suggestions = get_completion_suggestions(from_id)
    if not suggestions:
        return ""

    lines: list[str] = []
    for comp, phrase in suggestions:
        if phrase:
            lines.append(f"- {comp.name}: {phrase}")
        else:
            lines.append(f"- {comp.name}: {comp.one_liner}")

    return "SUGGESTED NEXT STEPS you can mention to the student:\n" + "\n".join(lines)
