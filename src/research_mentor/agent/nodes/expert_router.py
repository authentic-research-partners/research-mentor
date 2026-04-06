"""Expert Router node — analyzes gathered info for expert consultation needs.

Runs AFTER info_gathering and BEFORE guides. Analyzes complete context
(student question + project + gathered information) to detect safety,
ethics, or communication concerns requiring specialist handling.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from research_mentor.agent.state import MentorState
from research_mentor.llm import structured_call
from research_mentor.utils.memory_profiler import memory_profile_node


class ExpertRoutingDecision(BaseModel):
    """Structured output for expert routing analysis."""

    needs_expert: bool = Field(
        description="Whether expert consultation is needed based on gathered information and student question"
    )
    expert_type: Literal["safety", "ethics", "communication", "none"] = Field(
        default="none",
        description="Which expert to route to, or 'none' if none needed"
    )
    severity: Literal["none", "low", "medium", "high", "critical"] = Field(
        default="none",
        description="Severity level if expert needed, or 'none'"
    )
    reasoning: str = Field(max_length=500, description="Brief explanation of routing decision")
    keywords_detected: list[str] = Field(
        default_factory=list,
        description="Keywords that triggered expert routing (empty if no expert needed)"
    )

    @field_validator("expert_type", "severity", mode="before")
    @classmethod
    def normalize_empty(cls, v: str | None) -> str:
        if v is None or v == "":
            return "none"
        return v


EXPERT_ROUTING_INSTRUCTION = """You are an Expert Routing Coordinator. Analyze gathered information + student question to determine if expert consultation is needed.

**Your role:** Analyze context and decide routing. You do NOT provide guidance — you only determine if an expert is needed.

**Available experts:**
1. **safety** — Physical safety, dangerous procedures, lab safety, equipment hazards
2. **ethics** — IRB/IACUC requirements, human/animal subjects, data privacy, research integrity
3. **communication** — Profanity, abusive language, manipulation (NOT casual tone or frustration)

**When to route to safety:**
- Dangerous chemicals without proper safety measures
- Risky experimental procedures, lab equipment hazards
- NOT: General "how do I do this safely" questions (guides handle these)

**When to route to ethics:**
- Human subjects research without IRB mentioned
- Animal research without IACUC mentioned
- Data privacy concerns with personal data
- NOT: "Is this ethical?" general questions (guides handle these)

**When to route to communication:**
- Profanity, abusive or manipulative language
- NOT: Casual language, frustration, confusion, directness

**Severity guide:**
- **critical**: Immediate attention (e.g., human subjects without IRB, active dangerous procedure)
- **high**: Serious concern (e.g., planning unsafe experiment)
- **medium**: Moderate concern, expert guidance helpful
- **low**: Minor concern, quick expert guidance

**Decision priority (check in order, route to first match):**
1. Safety (critical/high)
2. Ethics (critical/high)
3. Communication (critical/high)

Most questions (90%+) do NOT need expert routing. Be conservative with safety/ethics (route if uncertain), permissive with communication (only route for actual abuse).
"""


def _format_gathered_info(gathered_information: dict[str, Any] | None) -> str:
    """Format gathered information for expert routing analysis.

    Produces a structured view of each assistant's results with previews,
    giving the LLM full context for routing decisions.
    """
    if not gathered_information:
        return "No information was pre-gathered."

    results = gathered_information.get("results", [])
    if not results:
        return "No information was pre-gathered."

    formatted_parts: list[str] = []
    for i, item in enumerate(results, 1):
        assistant = item.get("assistant", "unknown")

        if item.get("error"):
            formatted_parts.append(
                f"{i}. [{assistant}] ERROR: {str(item['error'])[:100]}"
            )
            continue

        summary = item.get("summary", "")
        sub_results = item.get("results", [])

        if sub_results:
            # Show individual result previews (papers, web results, etc.)
            previews: list[str] = []
            for r in sub_results[:5]:
                title = r.get("title", "")
                if title:
                    extra = ""
                    if r.get("year"):
                        extra += f", {r['year']}"
                    if r.get("source"):
                        extra += f", {r['source']}"
                    previews.append(f"     - {title}{extra}")
            detail = "\n".join(previews) if previews else "     (no details)"
            formatted_parts.append(
                f"{i}. [{assistant}]\n{detail}"
            )
        elif summary:
            formatted_parts.append(
                f"{i}. [{assistant}]\n     {summary[:300]}"
                f"{'...' if len(summary) > 300 else ''}"
            )
        else:
            formatted_parts.append(f"{i}. [{assistant}] (no results)")

    return "\n\n".join(formatted_parts)


@memory_profile_node("expert_router")
async def route_to_expert(state: MentorState) -> dict[str, Any]:
    """Analyze gathered info + student question for expert routing needs."""
    logger.info("Expert router: Analyzing for expert needs...")

    messages = state.get("messages", [])
    student_question = messages[-1].content if messages else "No message"
    project_context = state.get("project_context", "")
    gathered_information = state.get("gathered_information")

    gathered_info_text = _format_gathered_info(gathered_information)

    routing_context = f"""**Student Question:**
"{student_question}"

**Project Context:**
{project_context or "No project context available."}

**Gathered Information:**
{gathered_info_text}

Analyze this complete context to determine if expert consultation is needed."""

    decision = await structured_call(
        ExpertRoutingDecision,
        [
            SystemMessage(content=EXPERT_ROUTING_INSTRUCTION),
            HumanMessage(content=routing_context),
        ],
        thinking="low",
    )

    logger.info(
        "Expert routing: needs_expert={}, type={}, severity={}",
        decision.needs_expert, decision.expert_type, decision.severity,
    )
    if decision.keywords_detected:
        logger.debug("Keywords detected: {}", decision.keywords_detected)

    review: dict[str, Any] = {
        "needs_expert": decision.needs_expert,
        "expert_type": decision.expert_type,
        "severity": decision.severity,
        "reasoning": decision.reasoning,
        "keywords_detected": decision.keywords_detected,
    }

    if decision.needs_expert and decision.expert_type not in (None, "none"):
        return {
            "route_to_expert": decision.expert_type,
            "expert_routing_review": review,
        }

    return {"route_to_expert": None, "expert_routing_review": review}
