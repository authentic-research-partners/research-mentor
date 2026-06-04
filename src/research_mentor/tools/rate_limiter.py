"""Async rate limiter and 429 retry utilities.

``MonotonicRateLimiter`` — token-bucket rate limiter using ``time.monotonic()``.
``retry_on_429`` — tenacity retry for HTTP 429 with exponential backoff.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, TypeVar

from loguru import logger
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_F = TypeVar("_F", bound=Callable[..., Any])


class MonotonicRateLimiter:
    """Token-bucket rate limiter using ``time.monotonic()``.

    Args:
        max_rate: Maximum number of tokens (requests) in the bucket.
        time_period: Period in seconds over which tokens refill.

    Usage::

        limiter = MonotonicRateLimiter(2, 3)  # 2 requests per 3 seconds
        async with limiter:
            await do_request()
    """

    def __init__(self, max_rate: float, time_period: float = 1.0) -> None:
        self.max_rate = max_rate
        self.time_period = time_period
        self._tokens = max_rate
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * (self.max_rate / self.time_period)
        self._tokens = min(self.max_rate, self._tokens + new_tokens)
        self._last_refill = now

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Calculate wait time for next token
                wait = (1.0 - self._tokens) * (self.time_period / self.max_rate)
            await asyncio.sleep(wait)

    async def __aenter__(self) -> MonotonicRateLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


# ---------------------------------------------------------------------------
# 429 retry via tenacity
# ---------------------------------------------------------------------------


class RateLimitedError(Exception):
    """Raised when an HTTP API returns 429. Triggers tenacity retry."""

    def __init__(
        self, service: str, detail: str = "", retry_after: float | None = None,
    ) -> None:
        self.service = service
        self.detail = detail
        self.retry_after = retry_after
        super().__init__(f"{service} rate limited (429){': ' + detail if detail else ''}")


def parse_retry_after(header: str | None) -> float | None:
    """Parse an HTTP ``Retry-After`` header into seconds.

    Supports both delay-seconds (``"5"``) and HTTP-date formats.
    Returns *None* if the header is missing or unparseable.
    """
    if not header:
        return None
    # Try integer/float seconds first (most common for 429s)
    try:
        value = float(header)
        return max(value, 0.0)
    except ValueError:
        logger.debug(f"Retry-After not a float: {header!r} — trying HTTP-date")
    # Try HTTP-date format (RFC 7231 §7.1.1.1)
    from email.utils import parsedate_to_datetime
    try:
        dt = parsedate_to_datetime(header)
        from datetime import UTC, datetime
        delta = (dt - datetime.now(UTC)).total_seconds()
        return max(delta, 0.0)
    except (ValueError, TypeError):
        return None


def _log_retry(retry_state: RetryCallState) -> None:
    """Tenacity before-sleep callback — logs each retry via loguru."""
    exc = retry_state.outcome and retry_state.outcome.exception()
    service = getattr(exc, "service", "unknown")
    retry_after = getattr(exc, "retry_after", None)
    attempt = retry_state.attempt_number
    max_attempts = retry_state.retry_object.stop.max_attempt_number  # type: ignore[union-attr]
    extra = f", Retry-After: {retry_after:.1f}s" if retry_after else ""
    logger.warning(
        "{} 429, retry {}/{}{}", service, attempt, max_attempts - 1, extra,
    )


_MAX_RETRY_AFTER = 30.0  # Cap Retry-After to prevent a rogue header from blocking the pipeline


class _WaitRetryAfterOrExponential(wait_exponential):
    """Wait strategy: use ``Retry-After`` from the exception when available,
    otherwise fall back to exponential backoff.  Capped at ``_MAX_RETRY_AFTER``
    seconds to prevent unbounded stalls."""

    def __call__(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome and retry_state.outcome.exception()
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None and retry_after > 0:
            return min(float(retry_after), _MAX_RETRY_AFTER)
        return super().__call__(retry_state)


def retry_on_429(**overrides: Any) -> Callable[[_F], _F]:
    """Return a tenacity ``@retry`` decorator for 429 rate-limit errors.

    Default: 3 attempts, respects ``Retry-After`` header when available,
    falls back to exponential backoff (1s → 2s → 4s).  Logs each retry
    via loguru.  Pass keyword arguments to override any tenacity setting::

        @retry_on_429(stop=stop_after_attempt(5))
        async def call_api(...): ...
    """
    defaults: dict[str, Any] = {
        "retry": retry_if_exception_type(RateLimitedError),
        "stop": stop_after_attempt(3),
        "wait": _WaitRetryAfterOrExponential(multiplier=1, min=1, max=4),
        "before_sleep": _log_retry,
        "reraise": True,
    }
    defaults.update(overrides)
    return retry(**defaults)  # type: ignore[no-any-return]
