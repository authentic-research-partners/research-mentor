"""Unpaywall DOI → OA URL lookup.

Unpaywall finds legal open-access copies of papers by DOI. It returns
a URL where the student can read the paper — we never fetch or parse
the target page ourselves.

API docs: https://unpaywall.org/products/api
Rate limits: 100K requests/day (generous). Requires email in query params.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger

from research_mentor.config import load_config
from research_mentor.tools.rate_limiter import (
    MonotonicRateLimiter,
    RateLimitedError,
    parse_retry_after,
    retry_on_429,
)
from research_mentor.tools.status import record_tool_usage

_UNPAYWALL_BASE = "https://api.unpaywall.org/v2"

# ---------------------------------------------------------------------------
# Rate limiter (lazy init)
# ---------------------------------------------------------------------------

_unpaywall_limiter: MonotonicRateLimiter | None = None


def _ensure_limiter() -> MonotonicRateLimiter:
    global _unpaywall_limiter
    if _unpaywall_limiter is None:
        cfg = load_config()
        rl = cfg.rate_limits.unpaywall
        _unpaywall_limiter = MonotonicRateLimiter(rl[0], rl[1])
    return _unpaywall_limiter


# ---------------------------------------------------------------------------
# DOI → OA URL lookup
# ---------------------------------------------------------------------------


async def lookup_unpaywall(doi: str) -> dict[str, Any] | None:
    """Look up a DOI on Unpaywall to find a legal OA copy.

    Args:
        doi: The paper DOI (e.g., "10.1038/nature12373").

    Returns:
        Dict with oa_url, pdf_url, is_oa — or None if no OA copy found
        or the request fails. The URLs are for the student to click,
        not for us to fetch/parse.
    """
    cfg = load_config()
    email = cfg.services.polite_email
    if not email:
        # Unpaywall requires an email in every request (TOS)
        logger.debug("Unpaywall skipped: no polite_email configured")
        return None

    # Strip doi.org prefix if present
    clean_doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")

    import time as _time

    limiter = _ensure_limiter()
    t0 = _time.monotonic()
    success = False

    try:
        @retry_on_429()
        async def _request() -> httpx.Response:
            async with limiter:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"{_UNPAYWALL_BASE}/{clean_doi}",
                        params={"email": email},
                    )
                if resp.status_code == 429:
                    raise RateLimitedError(
                        "Unpaywall", f"DOI: {clean_doi}",
                        retry_after=parse_retry_after(resp.headers.get("Retry-After")),
                    )
                return resp

        try:
            resp = await _request()
        except RateLimitedError:
            return None

        if resp.status_code == 404:
            success = True  # API worked, DOI just not found
            return None

        resp.raise_for_status()
        data = resp.json()

        success = True

        if not data.get("is_oa"):
            return None

        # Find the best OA location
        best_oa = data.get("best_oa_location") or {}
        oa_url = best_oa.get("url_for_landing_page") or best_oa.get("url")
        pdf_url = best_oa.get("url_for_pdf")

        if not oa_url and not pdf_url:
            return None

        return {
            "oa_url": oa_url,
            "pdf_url": pdf_url,
            "is_oa": True,
            "source": "Unpaywall",
        }

    except TimeoutError:
        logger.debug("Unpaywall timeout for DOI: {}", clean_doi)
        return None
    except Exception as e:
        logger.debug("Unpaywall error for DOI {}: {}", clean_doi, e)
        return None
    finally:
        asyncio.create_task(record_tool_usage(
            "unpaywall",
            query=clean_doi,
            success=success,
            error_category=None if success else "network",
            elapsed=_time.monotonic() - t0,
        ))
