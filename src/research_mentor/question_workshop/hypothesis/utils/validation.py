"""Hypothesis Validation Utilities — semantic validation via structured_call().

Validation functions for variables, hypotheses, and confounds.
Uses LLM-based semantic checks as the authoritative path,
with basic structural checks (empty, too short) as fast rejections.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.hypothesis.schemas import VariableSubstantiveness
from research_mentor.question_workshop.hypothesis.state import HypothesisState, hypothesis_config


async def validate_variable(var_value: str | None, var_name: str) -> bool:
    """Validate that a variable is meaningful (not garbage or trivial).

    Uses structural fast-path for obvious rejections, then LLM-based
    semantic check as the authoritative decision.

    Args:
        var_value: The variable value to validate.
        var_name: Name of variable for logging (e.g., "independent_var", "dependent_var").

    Returns:
        True if variable is valid, False otherwise.
    """
    if not var_value:
        logger.debug("Validation failed for {}: empty or None", var_name)
        return False

    stripped = var_value.strip()
    if len(stripped) < 3:
        logger.debug("Validation failed for {}: too short (< 3 chars): '{}'", var_name, stripped)
        return False

    # LLM-based semantic check: is this a real research variable?
    result = await structured_call(
        VariableSubstantiveness,
        [
            SystemMessage(
                content=(
                    "You are validating whether a student's proposed research variable "
                    "is a meaningful, specific concept that could be studied. "
                    "Accept any legitimate research variable or measurable concept, "
                    "even single words (age, income, pollution, crime, happiness). "
                    "Reject: trivial conversational responses (yes, no, ok, maybe, idk), "
                    "vague placeholders (things, stuff, something, it, everything, anything), "
                    "and random/meaningless text. "
                    "The variable must name a SPECIFIC concept, not a catch-all word."
                ),
            ),
            HumanMessage(content=f"Is this a substantive research variable?\n\n{stripped}"),
        ],
        thinking="off",
        temperature=hypothesis_config().extraction_temperature,
    )

    if not result.is_substantive:
        logger.debug("Validation failed for {}: LLM classified as not substantive: '{}'",
                      var_name, stripped)
        return False

    logger.debug("Validation passed for {}: '{}'", var_name, stripped)
    return True


async def validate_stage_1_completion(
    state: HypothesisState | dict[str, Any],
) -> tuple[bool, str | None]:
    """Validate Stage 1 completion with quality checks.

    Returns:
        (is_valid, feedback_message) — if invalid, feedback explains why.
    """
    x = state.get("independent_var")
    y = state.get("dependent_var")

    if not x or not y:
        return False, None

    x_valid = await validate_variable(x, "independent_var (X)")
    y_valid = await validate_variable(y, "dependent_var (Y)")

    if not x_valid or not y_valid:
        feedback_parts = []
        if not x_valid:
            feedback_parts.append(
                f"**Independent variable (X)**: '{x}' seems too vague or unclear.",
            )
        if not y_valid:
            feedback_parts.append(
                f"**Dependent variable (Y)**: '{y}' seems too vague or unclear.",
            )

        feedback = "I want to make sure I understand your variables correctly:\n\n"
        feedback += "\n".join(feedback_parts)
        feedback += "\n\nCould you describe your variables in a bit more detail? For example:"
        feedback += "\n- What specific aspect are you interested in?"
        feedback += "\n- How would you explain this to someone unfamiliar with your topic?"

        logger.info("Stage 1 validation failed - X valid: {}, Y valid: {}", x_valid, y_valid)
        return False, feedback

    logger.info("Stage 1 validation passed - X: '{}', Y: '{}'", x, y)
    return True, None


async def validate_hypothesis(hypothesis: str | None) -> bool:
    """Validate that hypothesis is meaningful (not garbage).

    Uses LLM semantic check for substantiveness.
    """
    if not hypothesis:
        return False

    stripped = hypothesis.strip()
    if len(stripped) < 10:
        logger.debug("Validation failed for hypothesis: too short: '{}'", stripped)
        return False

    result = await structured_call(
        VariableSubstantiveness,
        [
            SystemMessage(
                content=(
                    "You are validating whether a student's proposed hypothesis "
                    "is a meaningful research statement. Accept any legitimate "
                    "hypothesis that proposes a relationship between variables. "
                    "Reject trivial responses (yes, no, ok, maybe, idk) "
                    "and text that is not a hypothesis."
                ),
            ),
            HumanMessage(content=f"Is this a substantive research hypothesis?\n\n{stripped}"),
        ],
        thinking="off",
        temperature=hypothesis_config().extraction_temperature,
    )

    if not result.is_substantive:
        logger.debug("Validation failed for hypothesis: not substantive: '{}'", stripped[:50])
        return False

    logger.debug("Validation passed for hypothesis: '{}'", stripped[:50])
    return True


async def validate_confound(confound: dict[str, Any]) -> bool:
    """Validate that a confound is meaningful.

    Uses LLM semantic check for substantiveness.
    """
    name = confound.get("name", "")

    if len(name.strip()) < 3:
        return False

    result = await structured_call(
        VariableSubstantiveness,
        [
            SystemMessage(
                content=(
                    "You are validating whether a proposed confounding variable "
                    "is a meaningful concept. Accept any legitimate variable that "
                    "could plausibly confound a research relationship. "
                    "Reject trivial responses (yes, no, ok) and meaningless text."
                ),
            ),
            HumanMessage(
                content=f"Is this a substantive confounding variable?\n\n{name.strip()}",
            ),
        ],
        thinking="off",
        temperature=hypothesis_config().extraction_temperature,
    )

    if not result.is_substantive:
        logger.debug("Validation failed for confound: not substantive: '{}'", name.strip())
        return False

    return True
