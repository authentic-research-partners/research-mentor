"""Stage 1: Field Validation — validate field is in scope for Questioned.

Uses domain.py for config-driven field checking, plus a quick LLM check
that the field name maps to a real academic discipline.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from research_mentor.question_workshop.domain import is_field_allowed, is_field_refused
from research_mentor.question_workshop.questioned.schemas import (
    FieldValidationResult,
    IsValidField,
)


async def field_validator(field: str) -> dict[str, Any]:
    """Validate that the field is real and in scope for Questioned.

    Returns:
        Dict with ``is_valid``, ``field_validation``, and optionally
        ``rejection_reason``.
    """
    logger.info("Stage 1/4: Field Validation — field='{}'", field)

    normalized = field.strip().lower().replace(" ", "_")

    # Fast check: refused domain?
    if is_field_refused(normalized):
        reason = (
            f"'{field}' involves clinical, medical, or policy considerations "
            "that require specialized mentoring. We recommend working with "
            "a faculty advisor in that area."
        )
        logger.warning("Field refused: {}", field)
        return {
            "is_valid": False,
            "field_validation": FieldValidationResult(
                is_valid=False,
                field=normalized,
                rejection_reason=reason,
            ),
        }

    # Fast check: allowed for questioned workshop?
    if not is_field_allowed("questioned", normalized):
        # The field might be valid but use a non-standard name.
        # Use LLM to normalize and re-check.
        from langchain_core.messages import HumanMessage, SystemMessage

        from research_mentor.llm import structured_call
        from research_mentor.question_workshop.questioned._config import (
            questioned_config,
        )

        cfg = questioned_config()
        result = await structured_call(
            IsValidField,
            [
                SystemMessage(
                    content="You are classifying academic fields. Determine if the "
                    "user's input refers to a real academic field of study, and if "
                    "so, provide its canonical name and broad category.",
                ),
                HumanMessage(content=f"Is '{field}' a real academic field?"),
            ],
            thinking="low",
            temperature=cfg.extraction_temperature,
        )

        if not result.is_real_field:
            return {
                "is_valid": False,
                "field_validation": FieldValidationResult(
                    is_valid=False,
                    field=normalized,
                    rejection_reason=(
                        f"'{field}' does not appear to be a "
                        "recognized academic field."
                    ),
                ),
            }

        # Re-check with the normalized name
        norm2 = (result.normalized_name or field).lower().replace(" ", "_")
        if is_field_refused(norm2):
            return {
                "is_valid": False,
                "field_validation": FieldValidationResult(
                    is_valid=False,
                    field=norm2,
                    rejection_reason=(
                        f"'{field}' maps to a domain that requires specialized mentoring. "
                        "We recommend working with a faculty advisor."
                    ),
                ),
            }

        if not is_field_allowed("questioned", norm2):
            # Still not in allowed list, but it's a real field — accept it
            # if the broad category matches (natural or social science)
            if result.broad_category in ("natural_science", "social_science"):
                logger.info(
                    "Field '{}' normalized to '{}' ({}), accepting",
                    field, result.normalized_name, result.broad_category,
                )
                return {
                    "is_valid": True,
                    "field_validation": FieldValidationResult(
                        is_valid=True,
                        field=result.normalized_name or field,
                        field_category=result.broad_category or "",
                    ),
                }

            return {
                "is_valid": False,
                "field_validation": FieldValidationResult(
                    is_valid=False,
                    field=norm2,
                    rejection_reason=f"'{field}' is not in the supported scope for this workshop.",
                ),
            }

        return {
            "is_valid": True,
            "field_validation": FieldValidationResult(
                is_valid=True,
                field=result.normalized_name or field,
                field_category=result.broad_category or "",
            ),
        }

    # Field is directly in allowed list
    from research_mentor.question_workshop.domain import get_allowed_categories
    cats = get_allowed_categories("questioned")
    category = "natural_science" if "natural_science" in cats else "social_science"

    logger.info("Field validated: {} (category={})", field, category)
    return {
        "is_valid": True,
        "field_validation": FieldValidationResult(
            is_valid=True,
            field=field,
            field_category=category,
        ),
    }
