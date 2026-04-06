"""Token usage budget enforcement.

Only enforced for the "api" backend (pay-per-token).
Claude CLI (subscription) and vLLM (local) skip budget checks.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from research_mentor.config import load_config


class BudgetExceededError(Exception):
    """Raised when a token budget limit is exceeded."""


async def check_budget(backend: str) -> None:
    """Check if the current backend is within budget.

    Only enforced for the "api" backend. No-op for others.
    Raises BudgetExceededError if any limit is exceeded.
    """
    if backend != "api":
        return

    config = load_config()
    budget = config.budget

    if not budget.enabled:
        return

    from research_mentor.db.crud import get_period_usage_for_budget

    period = budget.period
    provider = config.api.provider
    model = config.api.model

    # Check model-specific limit (most specific)
    model_limit = budget.models.get(model, 0)
    if model_limit > 0:
        model_usage = await get_period_usage_for_budget(period, model=model)
        if model_usage >= model_limit:
            raise BudgetExceededError(
                f"Model budget exceeded: {model} used {model_usage:,} tokens "
                f"({period} limit: {model_limit:,})"
            )

    # Check provider limit
    provider_limit = budget.providers.get(provider, 0)
    if provider_limit > 0:
        provider_usage = await get_period_usage_for_budget(period, provider=provider)
        if provider_usage >= provider_limit:
            raise BudgetExceededError(
                f"Provider budget exceeded: {provider} used {provider_usage:,} tokens "
                f"({period} limit: {provider_limit:,})"
            )

    # Check global limit
    if budget.global_limit > 0:
        global_usage = await get_period_usage_for_budget(period)
        if global_usage >= budget.global_limit:
            raise BudgetExceededError(
                f"Global budget exceeded: used {global_usage:,} tokens "
                f"({period} limit: {budget.global_limit:,})"
            )

    logger.debug("Budget check passed for backend={}", backend)


async def get_budget_status() -> dict[str, Any]:
    """Return current budget utilization for the UI.

    Returns utilization percentages for global, provider, and model limits.
    """
    config = load_config()
    budget = config.budget

    if not budget.enabled or config.backend != "api":
        return {"enabled": False, "backend": config.backend}

    from research_mentor.db.crud import get_period_usage_for_budget

    period = budget.period
    provider = config.api.provider
    model = config.api.model

    status: dict[str, Any] = {
        "enabled": True,
        "backend": config.backend,
        "period": period,
        "provider": provider,
        "model": model,
    }

    # Global
    if budget.global_limit > 0:
        global_usage = await get_period_usage_for_budget(period)
        status["global"] = {
            "usage": global_usage,
            "limit": budget.global_limit,
            "percent": round(global_usage / budget.global_limit * 100, 1),
        }

    # Provider
    provider_limit = budget.providers.get(provider, 0)
    if provider_limit > 0:
        provider_usage = await get_period_usage_for_budget(period, provider=provider)
        status["provider_budget"] = {
            "usage": provider_usage,
            "limit": provider_limit,
            "percent": round(provider_usage / provider_limit * 100, 1),
        }

    # Model
    model_limit = budget.models.get(model, 0)
    if model_limit > 0:
        model_usage = await get_period_usage_for_budget(period, model=model)
        status["model_budget"] = {
            "usage": model_usage,
            "limit": model_limit,
            "percent": round(model_usage / model_limit * 100, 1),
        }

    return status
