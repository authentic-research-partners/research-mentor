"""Prompt-side length hints derived from Pydantic field caps (layer 1).

**Layer 1** of the structured-output bounds architecture:

1. **Prompt soft cap** (this module) — guide the model toward the intended
   length so it self-limits and the layer-3 truncator rarely fires.
2. **Wire schema stripped** (``vllm_schema.strip_vllm_banned_keys``) — caps never
   reach vLLM (they trigger xgrammar's slow path).
3. **Post-decode truncate + validate** (``vllm_schema.truncate_to_model``) — the
   safety net that clips any residual overrun before validation.

**Target vs ceiling.** ``Field(max_length=...)`` is a *generous safety ceiling*
(~1.5-2x natural output) — it is NOT the length the prompt should ask for. Asking
the model to write up to the ceiling makes output drift long. So a field may also
declare a natural :class:`Target` (the length the prompt aims at); the hint uses
the target when present and otherwise falls back to deriving a looser figure from
the ceiling::

    description: Annotated[str, Field(max_length=800), Target(words=80)]
    #            ceiling 800 chars (validation/truncation), prompt aims at ~80 words

Either way the number is derived from the model, never hardcoded in a prompt
string (hardcoded numbers drift: phenomenon prompts once said "max 50/500/300
chars" while the schema was 100/800/500).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, get_origin

from pydantic import BaseModel

from research_mentor.vllm_schema import pydantic_max, unwrap_optional

# English averages ~5 chars/word; add the trailing space and round up to 8 to
# stay deliberately conservative. A ceiling-derived hint of ``cap / 8`` words has
# the model write ≈ ``cap * 6/8 = 0.75 * cap`` chars — inside the ceiling. This
# is only the *fallback*; prefer an explicit Target for the intended length.
_CHARS_PER_WORD = 8


@dataclass(frozen=True)
class Target:
    """Natural target length for a field's prompt hint, distinct from its
    ``Field(max_length=...)`` ceiling.

    Attach via ``Annotated`` alongside the ``Field``. Use ``words`` for ``str``
    fields and ``items`` for ``list`` fields. The cap remains the generous
    validation/truncation ceiling; the target is what prompts aim at.
    """

    words: int | None = None
    items: int | None = None


def chars_to_word_hint(chars: int | None) -> int:
    """Convert a character cap to a round word target (nearest 5).

    Returns 0 for ``None`` or non-positive input, so callers can interpolate the
    result unconditionally without branching on whether a cap exists.
    """
    if chars is None or chars <= 0:
        return 0
    words = chars // _CHARS_PER_WORD
    return ((words + 4) // 5) * 5


def _field_metadata(model: type[BaseModel], field: str) -> list[Any]:
    finfo = model.model_fields.get(field)
    return list(finfo.metadata) if finfo is not None else []


def _target(model: type[BaseModel], field: str) -> Target | None:
    """The field's declared :class:`Target`, if any."""
    for m in _field_metadata(model, field):
        if isinstance(m, Target):
            return m
    return None


def _is_list_field(model: type[BaseModel], field: str) -> bool:
    """True when ``field``'s (optional-unwrapped) annotation is a ``list``."""
    finfo = model.model_fields.get(field)
    if finfo is None:
        return False
    return get_origin(unwrap_optional(finfo.annotation)) is list


def _word_figure(model: type[BaseModel], field: str) -> int | None:
    """Word target for a ``str`` field: the explicit Target, else cap-derived."""
    target = _target(model, field)
    if target is not None and target.words is not None:
        return target.words
    cap = pydantic_max(model, field)
    return chars_to_word_hint(cap) if cap is not None else None


def _item_figure(model: type[BaseModel], field: str) -> int | None:
    """Item target for a ``list`` field: the explicit Target, else the cap."""
    target = _target(model, field)
    if target is not None and target.items is not None:
        return target.items
    return pydantic_max(model, field)


def length_hint(model: type[BaseModel], field: str) -> str:
    """Soft length phrase for one field.

    ``"~80 words"`` for ``str`` fields, ``"at most 5 items"`` for ``list``
    fields, and ``""`` when the field is absent or has neither a Target nor a
    cap (so it can be dropped into a prompt unconditionally). Prefers an explicit
    :class:`Target`; otherwise derives from the ``max_length`` ceiling.
    """
    if model.model_fields.get(field) is None:
        return ""
    if _is_list_field(model, field):
        items = _item_figure(model, field)
        return f"at most {items} items" if items is not None else ""
    words = _word_figure(model, field)
    return f"~{words} words" if words is not None else ""


def length_limits_block(model: type[BaseModel], *fields: str) -> str:
    """Render a length-guidance block for a schema's fields.

    ``str`` fields render as ``- field: aim for ~N words (hard limit M characters)``
    (the hard-limit clause is dropped when the field has no cap); ``list`` fields
    render as ``- field: at most N items``. Fields with neither a Target nor a cap
    are skipped. With no ``fields`` given, all fields are considered, in
    declaration order. The "aim for" figure is the natural target; the "hard
    limit" is the ceiling — keeping the two visibly distinct.
    """
    names = fields or tuple(model.model_fields)
    lines: list[str] = []
    for name in names:
        if model.model_fields.get(name) is None:
            continue
        if _is_list_field(model, name):
            items = _item_figure(model, name)
            if items is not None:
                lines.append(f"- {name}: at most {items} items")
            continue
        words = _word_figure(model, name)
        if words is None:
            continue
        cap = pydantic_max(model, name)
        if cap is not None:
            lines.append(f"- {name}: aim for ~{words} words (hard limit {cap} characters)")
        else:
            lines.append(f"- {name}: aim for ~{words} words")
    return "\n".join(lines)
