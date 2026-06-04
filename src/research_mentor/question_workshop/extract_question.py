"""Extract a saveable research question from interactive workshop state.

Each workshop stores its progress in LangGraph checkpointed state. This module
provides per-workshop extraction functions that read the state dict and return
a ``generated_problems``-compatible record, or ``None`` if the session has not
yet produced a question worth saving.

The two-step pipeline:
1. **Deterministic extraction** — read state fields into a draft record
2. **LLM polish** — ``polish_question()`` rewrites title/description/investigation
   into concise, student-facing prose via ``structured_call``
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_mentor.prompt_limits import Target, length_hint


def extract_hypothesis(state: dict[str, Any]) -> dict[str, Any] | None:
    """Extract from a Hypothesis workshop session."""
    hypothesis = state.get("hypothesis")
    if not hypothesis:
        return None

    iv = state.get("independent_var", "")
    dv = state.get("dependent_var", "")
    theory = state.get("theory", "")
    scope = state.get("scope_details") or {}
    gap = state.get("gap_description", "")
    method = state.get("statistical_method", "")
    x_meas = state.get("x_measurement", "")
    y_meas = state.get("y_measurement", "")
    dataset = state.get("selected_dataset") or {}

    description_parts = [hypothesis]
    if theory:
        description_parts.append(f"Mechanism: {theory}")
    if gap:
        description_parts.append(f"Gap: {gap}")
    description = "\n\n".join(description_parts)

    investigation_parts = []
    if method:
        investigation_parts.append(f"Statistical method: {method}")
    if x_meas:
        investigation_parts.append(f"X measurement: {x_meas}")
    if y_meas:
        investigation_parts.append(f"Y measurement: {y_meas}")
    if dataset.get("name"):
        investigation_parts.append(f"Dataset: {dataset['name']}")
    if scope:
        scope_str = ", ".join(f"{k}: {v}" for k, v in scope.items() if v)
        if scope_str:
            investigation_parts.append(f"Scope: {scope_str}")
    investigation = "\n".join(investigation_parts) or "To be determined"

    title = hypothesis
    if iv and dv:
        title = f"Effect of {iv} on {dv}"

    return {
        "id": uuid.uuid4().hex,
        "field": _infer_field(state),
        "problem_type": "hypothesis",
        "title": title,
        "description": description,
        "investigation": investigation,
        "feasibility": "Developed through guided workshop",
        "recommended": "YES",
        "workshop_type": "hypothesis",
    }


def extract_gaps(state: dict[str, Any]) -> dict[str, Any] | None:
    """Extract from a Gaps workshop session."""
    validated = state.get("validated_questions") or []
    candidates = state.get("candidate_questions") or []

    questions = validated or candidates
    if not questions:
        return None

    best = max(questions, key=lambda q: q.get("score", 0))
    question_text = best.get("question", "")
    if not question_text:
        return None

    focus = state.get("focus_area", "")
    mode = state.get("selected_mode", "")
    domain = state.get("domain_topic", "")

    description_parts = [question_text]
    if focus:
        description_parts.append(f"Focus area: {focus}")
    if mode:
        description_parts.append(f"Discovery mode: {mode}")

    score = best.get("score")

    return {
        "id": uuid.uuid4().hex,
        "field": _infer_field(state),
        "problem_type": "literature_gap",
        "title": question_text[:200],
        "description": "\n\n".join(description_parts),
        "investigation": (
            f"Literature-gap approach via {mode or 'deep dive'} in {domain or 'the field'}"
        ),
        "feasibility": "Validated through FUNDAMENTAL criteria",
        "recommended": "YES" if (score and score >= 0.8) else "WITH_MODIFICATIONS",
        "workshop_type": "gaps",
        **({"overall_quality": round(score * 10, 1)} if score else {}),
    }


def extract_theory(state: dict[str, Any]) -> dict[str, Any] | None:
    """Extract from a Theory workshop session (legacy — theory is now pipeline-only)."""
    framework = state.get("framework_draft")
    explanation = state.get("preferred_explanation")
    if not framework and not explanation:
        return None

    domain = state.get("domain_topic", "")
    fw_type = state.get("framework_type", "")
    formal = state.get("formal_model", "")
    categories = state.get("framework_categories") or []
    relationships = state.get("framework_relationships") or []

    title = f"Theory: {domain}" if domain else "Constructed Theory"
    if fw_type:
        title = f"{fw_type.replace('_', ' ').title()}: {domain or 'theory'}"

    description_parts = []
    if explanation:
        description_parts.append(f"Core explanation: {explanation}")
    if framework:
        description_parts.append(f"Framework: {framework}")
    if formal:
        description_parts.append(f"Formal model: {formal}")

    investigation_parts = []
    if categories:
        investigation_parts.append(f"Categories: {', '.join(categories)}")
    if relationships:
        investigation_parts.append(f"Relationships: {', '.join(relationships)}")

    consilience_success = state.get("consilience_successes", 0)
    consilience_fail = state.get("consilience_failures", 0)
    if consilience_success or consilience_fail:
        investigation_parts.append(
            f"Consilience: {consilience_success} supporting, {consilience_fail} contradicting"
        )

    return {
        "id": uuid.uuid4().hex,
        "field": _infer_field(state),
        "problem_type": "theoretical",
        "title": title[:200],
        "description": "\n\n".join(description_parts) or "Theory constructed through workshop",
        "investigation": "\n".join(investigation_parts) or "Theory construction approach",
        "feasibility": "Developed through theory construction workshop",
        "recommended": "YES",
        "workshop_type": "theory",
    }


def extract_modeling(state: dict[str, Any]) -> dict[str, Any] | None:
    """Extract from a Modeling workshop session."""
    model_spec = state.get("model_specification")
    system_desc = state.get("system_description")
    if not model_spec and not system_desc:
        return None

    domain = state.get("domain_topic", "")
    components = state.get("components") or []
    simplifications = state.get("simplifications") or []

    title = f"Computational Model: {domain}" if domain else "Computational Model"

    description_parts = []
    if system_desc:
        description_parts.append(system_desc)
    if model_spec:
        description_parts.append(f"Model specification: {model_spec}")

    investigation_parts = []
    if components:
        investigation_parts.append(f"Components: {', '.join(str(c) for c in components[:10])}")
    if simplifications:
        investigation_parts.append(
            f"Simplifications: {', '.join(str(s) for s in simplifications[:10])}"
        )

    return {
        "id": uuid.uuid4().hex,
        "field": _infer_field(state),
        "problem_type": "simulation",
        "title": title[:200],
        "description": "\n\n".join(description_parts) or "Model developed through workshop",
        "investigation": "\n".join(investigation_parts) or "Computational modeling approach",
        "feasibility": "Developed through modeling workshop",
        "recommended": "YES",
        "workshop_type": "modeling",
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

_Extractor = Callable[[dict[str, Any]], dict[str, Any] | None]

EXTRACTORS: dict[str, _Extractor] = {
    "hypothesis": extract_hypothesis,
    "gaps": extract_gaps,
    "theory": extract_theory,
    "modeling": extract_modeling,
}


def extract_question(workshop_type: str, state: dict[str, Any]) -> dict[str, Any] | None:
    """Dispatch to the correct extractor.  Returns ``None`` if no question yet."""
    extractor = EXTRACTORS.get(workshop_type)
    if extractor is None:
        return None
    return extractor(state)


# ── LLM Polish ───────────────────────────────────────────────────────────────

class PolishedQuestion(BaseModel):
    """Clean, student-facing version of a saved research question."""

    title: str = Field(
        max_length=120,
        description="Short, specific research question title. "
        "Written as a question when possible.",
    )
    description: Annotated[str, Target(words=45)] = Field(
        max_length=500,
        description="1-3 sentence summary of what this research question "
        "investigates and why it matters. Written in clear prose, not bullet "
        "points or labeled fragments.",
    )
    investigation: Annotated[str, Target(words=45)] = Field(
        max_length=500,
        description="1-3 sentence description of how this question could be "
        "investigated — methods, data sources, measurements. Written in "
        "clear prose.",
    )
    field: str = Field(
        max_length=50,
        description="Scientific field this question belongs to. Use a concise, "
        "standard name (e.g. 'biology', 'ecology', 'physics', 'psychology').",
    )


_POLISH_SYSTEM = f"""\
You are editing a research question card for a student's collection. \
Rewrite the draft into clean, concise, student-facing prose.

Rules:
- Title: a clear research question ({length_hint(PolishedQuestion, "title")}).
- Description: 1-3 sentences ({length_hint(PolishedQuestion, "description")}). \
No labels like "Mechanism:" or "Gap:" or \
"Focus area:" or "Inspired by retracted paper:" — rewrite as flowing prose. \
CRITICAL: Every specific fact, name, number, mechanism, and detail from the \
draft MUST appear in your output. Do not summarize away specifics. If the \
draft says "enzymatic reactions" or "temperatures above 40°C" or names a \
specific paper, those details must be in your output.
- Investigation: 1-3 sentences ({length_hint(PolishedQuestion, "investigation")}) \
on methods, data, measurements. No bullet \
points or key-value labels like "Statistical method:" or "X measurement:".
- Field: one concise standard field name.
- Do NOT invent new facts, studies, author names, or claims.
- Write at a level appropriate for a motivated high-school science student."""


async def polish_question(draft: dict[str, Any]) -> dict[str, Any]:
    """Use an LLM to rewrite a draft question record into clean prose.

    Takes a ``generated_problems``-shaped dict, rewrites title/description/
    investigation/field via ``structured_call``, and returns the updated dict.
    Other fields (id, workshop_type, etc.) pass through unchanged.
    """
    from research_mentor.llm import structured_call

    user_content = (
        f"Workshop type: {draft.get('workshop_type', 'unknown')}\n\n"
        f"Draft title: {draft.get('title', '')}\n\n"
        f"Draft description:\n{draft.get('description', '')}\n\n"
        f"Draft investigation:\n{draft.get('investigation', '')}\n\n"
        f"Draft field: {draft.get('field', '')}"
    )

    polished = await structured_call(
        PolishedQuestion,
        [
            SystemMessage(content=_POLISH_SYSTEM),
            HumanMessage(content=user_content),
        ],
        thinking="medium",
    )

    result = dict(draft)
    result["title"] = polished.title[:200]
    result["description"] = polished.description
    result["investigation"] = polished.investigation
    result["field"] = polished.field
    return result


def _infer_field(state: dict[str, Any]) -> str:
    """Best-effort field inference from workshop state."""
    for key in ("field", "browse_field", "domain_topic"):
        val: str | None = state.get(key)
        if val and isinstance(val, str):
            return val
    return "interdisciplinary"
