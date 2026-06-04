"""Domain scope for Question Workshops.

Shared validator that workshops call to check whether a field/topic
is in the default supported set. All lists — including refused fields —
come from config.toml.
"""

from __future__ import annotations

from functools import lru_cache

from research_mentor.config import load_config


def _load_fields_config() -> dict[str, list[str]]:
    """Load the [question_workshop.fields] config section."""
    cfg = load_config()
    return cfg.question_workshop.fields


def get_refused_fields() -> list[str]:
    """Return the configured refused field list (config.toml is the source of truth)."""
    return _load_fields_config()["refused"]


def get_natural_science_fields() -> list[str]:
    """Return the configured natural science field list."""
    return _load_fields_config()["natural_science"]


def get_social_science_fields() -> list[str]:
    """Return the configured social science field list."""
    return _load_fields_config()["social_science"]


def get_allowed_categories(workshop: str) -> list[str]:
    """Return which field categories a workshop accepts.

    E.g., ``get_allowed_categories("phenomenon")`` → ``["natural_science"]``
    """
    fields_cfg = _load_fields_config()
    return fields_cfg.get(workshop, [])


def get_allowed_fields(workshop: str) -> list[str]:
    """Return the flat list of allowed field names for a workshop.

    Expands category references (``natural_science``, ``social_science``)
    into concrete field names from config.
    """
    categories = get_allowed_categories(workshop)
    fields_cfg = _load_fields_config()
    result: list[str] = []
    for cat in categories:
        result.extend(fields_cfg.get(cat, []))
    return result


def is_field_refused(field: str) -> bool:
    """Check if a field is in the refused list."""
    normalized = field.lower().replace(" ", "_")
    return normalized in {f.lower().replace(" ", "_") for f in get_refused_fields()}


@lru_cache(maxsize=32)
def _allowed_set(workshop: str) -> frozenset[str]:
    """Cached set of allowed fields for fast lookup."""
    return frozenset(get_allowed_fields(workshop))


def is_field_allowed(workshop: str, field: str) -> bool:
    """Check if a field is allowed for a specific workshop.

    Returns False if the field is in the refusal list or not in the
    workshop's allowed list.
    """
    normalized = field.lower().replace(" ", "_")
    if is_field_refused(normalized):
        return False
    return normalized in _allowed_set(workshop)


def get_domain_classification_prompt() -> str:
    """Build the domain classification prompt dynamically from config.

    All three lists (natural science, social science, refused) come from
    config.toml so they stay in sync with the field definitions.
    """
    natural = ", ".join(f.replace("_", " ") for f in get_natural_science_fields())
    social = ", ".join(f.replace("_", " ") for f in get_social_science_fields())
    refused = ", ".join(f.replace("_", " ") for f in get_refused_fields())

    return f"""Classify this research topic.

Topic: "{{topic}}"

Classify as ONE of:
- natural_science: {natural}
- social_science: {social}
- refused: {refused}
- off_topic: not a research topic (casual conversation, personal questions, non-academic)

If the topic goes beyond typical science and involves social, political, ethical, \
or clinical considerations, classify as "refused".

confidence: "high" if the topic clearly belongs to one category, "low" if ambiguous"""


def get_refused_fields_display() -> str:
    """Comma-separated refused fields for use in prompts."""
    return ", ".join(f.replace("_", " ") for f in get_refused_fields())


def get_natural_science_fields_display() -> str:
    """Comma-separated natural science fields for use in prompts."""
    return ", ".join(f.replace("_", " ") for f in get_natural_science_fields())


def get_social_science_fields_display() -> str:
    """Comma-separated social science fields for use in prompts."""
    return ", ".join(f.replace("_", " ") for f in get_social_science_fields())
