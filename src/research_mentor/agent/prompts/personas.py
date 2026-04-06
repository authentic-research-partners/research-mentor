"""Persona adaptation instructions for guide nodes.

Ported from hosted progress_mentor_langgraph/nodes/guides/_persona_instructions.py.
"""

from __future__ import annotations

from research_mentor.db.crud import get_persona_by_id

PERSONA_ADAPTATION_INSTRUCTION_BASE = """
---

**PERSONA ADAPTATION (METHODOLOGY, NOT VOICE):**

**CRITICAL: We combine modern research mentoring pedagogy with {approach_type}**

You receive {persona_description}.

Apply {methodology_source}. From how they conduct research, infer how they would guide a student:
- What investigative steps would they suggest?
- What order of inquiry matches their research style?
- How do they structure their knowledge-building?

**Modern pedagogy + {approach_label}:**
- The GUIDANCE TYPE (scaffolding/guided_discovery/pure_socratic/reflection) comes from
  modern research mentoring pedagogy (what we know works for student development)
- The METHODOLOGY (how to deliver that guidance) comes from {methodology_detail}

**IMPORTANT:** Adapt methodology (what steps, what order), NOT voice (how to say it).
Voice styling is handled by Presenter agent.

When no persona specified, use balanced approach (guided discovery style).

---
"""

PERSONA_ADAPTATION_INSTRUCTION_HISTORICAL = PERSONA_ADAPTATION_INSTRUCTION_BASE.format(
    approach_type="historical research approach",
    persona_description="persona context (e.g., [PERSONA] Isaac Newton)",
    methodology_source=(
        "that scientist's RESEARCH METHODOLOGY - their approach to discovery "
        "documented in their published works, papers, correspondence, and historical records"
    ),
    approach_label="Historical approach",
    methodology_detail=(
        "the historical scientist's documented research approach "
        "(from their published works, papers, and records)"
    ),
)

PERSONA_ADAPTATION_INSTRUCTION_MODERN = PERSONA_ADAPTATION_INSTRUCTION_BASE.format(
    approach_type="contemporary research approach",
    persona_description="research approach context (e.g., [PERSONA] World-class modern physicist)",
    methodology_source="that field's contemporary research methodology and thinking approach",
    approach_label="Contemporary approach",
    methodology_detail="contemporary research practices in that field",
)


async def get_persona_instruction(persona_id: str) -> str:
    """Get the appropriate persona adaptation instruction based on persona type."""
    persona_details = await get_persona_by_id(persona_id)
    if not persona_details:
        return PERSONA_ADAPTATION_INSTRUCTION_HISTORICAL

    persona_type = persona_details.get("personaType", "historical")
    if persona_type == "modern":
        return PERSONA_ADAPTATION_INSTRUCTION_MODERN
    return PERSONA_ADAPTATION_INSTRUCTION_HISTORICAL
