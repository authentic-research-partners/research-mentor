"""vLLM wire-schema sanitisation for constrained decoding.

**Why this module exists.** Pydantic ``Field(max_length=...)`` and list-length
bounds surface in ``model_json_schema()`` as ``maxLength`` / ``maxItems`` (plus
``minLength``, ``pattern``, numeric ranges, ...). Sending any of those over the
wire to vLLM triggers xgrammar's *slow path*: even a single ``maxLength`` field
can collapse throughput under concurrency, with every request hitting the
client-side timeout. The slowdown is per-request, not a one-time grammar
compile, so it bites exactly the high-concurrency paths (eval runners,
workshop pipelines).

**The fix.** Strip those keys from the JSON schema we send to vLLM/the API
(:func:`strip_vllm_banned_keys`), but keep the constraints on the Pydantic model
for post-decode validation. Because the wire no longer enforces the caps, the
model may emit slightly over-length output, so :func:`truncate_to_model` clips
decoded data back to the model's declared bounds *before* ``model_validate`` —
turning a would-be ``ValidationError`` (and retry) into a silent, contractual
trim.
"""

from __future__ import annotations

from types import UnionType
from typing import Any, Union, get_args, get_origin

from annotated_types import MaxLen
from pydantic import BaseModel

# Constraint keys stripped from the wire schema. These trigger xgrammar's slow
# path on the vLLM/xgrammar stack we run. The Pydantic model still carries them
# for post-decode validation + truncation — they just never go over the wire.
_VLLM_BANNED_SCHEMA_KEYS = frozenset(
    {
        "maxLength",
        "minLength",
        "maxItems",
        "minItems",
        "pattern",
        "multipleOf",
        "maximum",
        "minimum",
        "exclusiveMaximum",
        "exclusiveMinimum",
    }
)


def strip_vllm_banned_keys(schema: Any) -> Any:
    """Recursively drop xgrammar-slow constraint keys from a JSON schema tree.

    Returns a new structure; does not mutate the input.
    """
    if isinstance(schema, dict):
        return {
            k: strip_vllm_banned_keys(v)
            for k, v in schema.items()
            if k not in _VLLM_BANNED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [strip_vllm_banned_keys(v) for v in schema]
    return schema


def unwrap_optional(annotation: Any) -> Any:
    """Return the sole non-``None`` member of an ``X | None`` union, else as-is."""
    if get_origin(annotation) in (UnionType, Union):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def _max_length_from_metadata(metadata: list[Any]) -> int | None:
    """Pull a ``max_length`` constraint out of a Pydantic field's metadata."""
    for m in metadata:
        if isinstance(m, MaxLen):
            return int(m.max_length)
    return None


def pydantic_max(model: type[BaseModel], field: str) -> int | None:
    """Read a field's declared ``max_length`` cap (the single source of truth).

    Returns the character cap for ``str`` fields and the item cap for ``list``
    fields (Pydantic emits both as the same ``max_length`` constraint), or
    ``None`` when the field is absent or uncapped. This is the one reader used
    by both post-decode truncation and prompt-side length hints
    (``research_mentor.prompt_limits``).
    """
    finfo = model.model_fields.get(field)
    if finfo is None:
        return None
    return _max_length_from_metadata(finfo.metadata)


def _coerce_to_bounds(value: Any, annotation: Any, cap: int | None) -> Any:
    """Truncate one value against its declared annotation + ``max_length``.

    Strings clip to ``cap`` chars; lists clip to ``cap`` items and recurse into
    nested ``BaseModel`` items; standalone nested ``BaseModel`` dicts recurse so
    their own fields get capped. Other types pass through untouched.
    """
    annotation = unwrap_optional(annotation)
    if isinstance(value, str):
        return value[:cap] if cap is not None else value
    if isinstance(value, list):
        truncated = value[:cap] if cap is not None else value
        if get_origin(annotation) is list:
            args = get_args(annotation)
            if args:
                inner = unwrap_optional(args[0])
                if isinstance(inner, type) and issubclass(inner, BaseModel):
                    return [
                        truncate_to_model(inner, item) if isinstance(item, dict) else item
                        for item in truncated
                    ]
        return truncated
    if (
        isinstance(value, dict)
        and isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
    ):
        return truncate_to_model(annotation, value)
    return value


def truncate_to_model(model: type[BaseModel], data: Any) -> Any:
    """Clip an LLM-output dict to a Pydantic model's declared bounds.

    Walks the model's fields: top-level strings clip to their ``max_length``,
    lists clip to their item cap, nested submodels (standalone or in lists)
    recurse. Unknown keys and non-dict input pass through. Call this *before*
    ``model.model_validate(data)`` so a minor over-run trims instead of raising.
    Returns a new structure; does not mutate the input.
    """
    if not isinstance(data, dict):
        return data
    out: dict[str, Any] = {}
    for k, v in data.items():
        finfo = model.model_fields.get(k)
        if finfo is None:
            out[k] = v
            continue
        out[k] = _coerce_to_bounds(
            v, finfo.annotation, _max_length_from_metadata(finfo.metadata)
        )
    return out
