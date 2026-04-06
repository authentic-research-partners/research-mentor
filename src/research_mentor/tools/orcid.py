"""ORCID public profile enrichment for researcher search results.

Fetches verified researcher data from ORCID's public API v3.0: biography,
employment history, education, and external identifiers. Complements
OpenAlex's bibliometric data (citations, h-index) with the researcher's
own self-curated profile.

Free, no API key needed for public data. CC0 licensed.
"""

from __future__ import annotations

import asyncio
import re
import time as _time
from typing import Any

import httpx
from loguru import logger

from research_mentor.config import load_config
from research_mentor.tools.rate_limiter import MonotonicRateLimiter
from research_mentor.tools.status import record_tool_usage

_ORCID_BASE = "https://pub.orcid.org/v3.0"

# ---------------------------------------------------------------------------
# Rate limiter (lazy init)
# ---------------------------------------------------------------------------

_limiter: MonotonicRateLimiter | None = None

# Concurrency cap for parallel enrichment
_ENRICH_SEMAPHORE = asyncio.Semaphore(5)


def _ensure_limiter() -> MonotonicRateLimiter:
    global _limiter
    if _limiter is None:
        cfg = load_config()
        rl = cfg.rate_limits.orcid
        _limiter = MonotonicRateLimiter(rl[0], rl[1])
    return _limiter




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_orcid_id(orcid: str) -> str:
    """Extract bare ORCID ID from URL or bare format.

    OpenAlex returns ``https://orcid.org/0000-0001-2345-6789``; the API
    expects the bare ``0000-0001-2345-6789``.
    """
    match = re.search(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", orcid)
    return match.group(0) if match else orcid


def _parse_employments(data: dict[str, Any]) -> list[dict[str, str]]:
    """Extract employment history from ORCID record."""
    employments: list[dict[str, str]] = []
    groups = (
        data
        .get("activities-summary", {})
        .get("employments", {})
        .get("affiliation-group", [])
    )
    for group in groups:
        for summary in group.get("summaries", []):
            emp = summary.get("employment-summary", {})
            org = emp.get("organization", {})
            employments.append({
                "institution": org.get("name", ""),
                "role": emp.get("role-title", ""),
                "department": emp.get("department-name", ""),
            })
    return employments


def _parse_education(data: dict[str, Any]) -> list[dict[str, str]]:
    """Extract education history from ORCID record."""
    education: list[dict[str, str]] = []
    groups = (
        data
        .get("activities-summary", {})
        .get("educations", {})
        .get("affiliation-group", [])
    )
    for group in groups:
        for summary in group.get("summaries", []):
            edu = summary.get("education-summary", {})
            org = edu.get("organization", {})
            education.append({
                "institution": org.get("name", ""),
                "degree": edu.get("role-title", ""),
                "department": edu.get("department-name", ""),
            })
    return education


def _parse_external_ids(data: dict[str, Any]) -> dict[str, str | None]:
    """Extract external identifiers (Scopus, ResearcherID, etc.)."""
    ids: dict[str, str | None] = {"scopus": None, "researcher_id": None}
    ext = (
        data
        .get("person", {})
        .get("external-identifiers", {})
        .get("external-identifier", [])
    )
    for eid in ext:
        id_type = (eid.get("external-id-type") or "").lower()
        id_value = eid.get("external-id-value")
        if "scopus" in id_type:
            ids["scopus"] = id_value
        elif "researcherid" in id_type or "researcher" in id_type:
            ids["researcher_id"] = id_value
    return ids


def _parse_biography(data: dict[str, Any]) -> str | None:
    """Extract biography text."""
    bio = data.get("person", {}).get("biography", {})
    content = bio.get("content") if isinstance(bio, dict) else None
    return content.strip() if content else None


def _parse_keywords(data: dict[str, Any]) -> list[str]:
    """Extract researcher-chosen keywords."""
    kw_section = data.get("person", {}).get("keywords", {})
    kw_list = kw_section.get("keyword", []) if isinstance(kw_section, dict) else []
    return [k.get("content", "") for k in kw_list if k.get("content")]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def lookup_orcid_profile(orcid_id: str) -> dict[str, Any] | None:
    """Fetch a public profile from ORCID.

    Args:
        orcid_id: ORCID identifier (bare ``0000-0001-2345-6789`` or full URL).

    Returns:
        Dict with biography, employments, education, external_ids, keywords.
        None on failure (404, timeout, network error).
    """
    bare_id = _extract_orcid_id(orcid_id)
    t0 = _time.monotonic()

    try:
        async with _ensure_limiter():
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{_ORCID_BASE}/{bare_id}/record",
                    headers={"Accept": "application/json"},
                )
        if resp.status_code == 404:
            logger.debug("ORCID profile not found: {}", bare_id)
            return None
        resp.raise_for_status()
        data = resp.json()
    except (httpx.TimeoutException, httpx.HTTPError, TimeoutError) as e:
        logger.debug("ORCID lookup failed for {}: {}", bare_id, e)
        asyncio.create_task(
            record_tool_usage(
                "orcid", query=bare_id, success=False, elapsed=_time.monotonic() - t0,
            ),
        )
        return None

    result = {
        "orcid_id": bare_id,
        "biography": _parse_biography(data),
        "employments": _parse_employments(data),
        "education": _parse_education(data),
        "external_ids": _parse_external_ids(data),
        "keywords": _parse_keywords(data),
    }
    asyncio.create_task(
        record_tool_usage("orcid", query=bare_id, success=True, elapsed=_time.monotonic() - t0),
    )
    return result


async def enrich_researchers_with_orcid(
    researchers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich researcher dicts that have an ``orcid`` field.

    Adds ``orcid_profile`` key with biography, employments, education,
    external_ids, and keywords. Researchers without a valid ORCID are
    returned unchanged. Individual lookup failures are non-fatal.

    Args:
        researchers: List of researcher dicts (e.g. from ``search_researchers``).

    Returns:
        Same list with ``orcid_profile`` added where available.
    """
    if not researchers:
        return researchers

    async def _enrich_one(researcher: dict[str, Any]) -> dict[str, Any]:
        orcid = researcher.get("orcid", "")
        if not orcid:
            return researcher

        async with _ENRICH_SEMAPHORE:
            profile = await lookup_orcid_profile(orcid)

        if profile:
            researcher["orcid_profile"] = profile
        return researcher

    return list(await asyncio.gather(*[_enrich_one(r) for r in researchers]))
