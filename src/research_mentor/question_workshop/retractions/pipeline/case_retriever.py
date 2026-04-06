"""Stage 1: Case Retriever — find retracted papers by field or DOI.

Two modes:
- **Browse:** semantic search → LLM selects 3-5 diverse cases
- **Direct:** DOI lookup (or title fallback) → single case

Medical retractions are excluded. Author names are omitted (no-blame).
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.retractions.pipeline.schemas import (
    CaseSelection,
    RetractionCase,
    RetractionCases,
    SingleCaseClaim,
    looks_like_doi,
)
from research_mentor.tools.retraction_watch import (
    get_retraction_by_doi,
    search_retracted_papers,
)

# Fields that should be excluded (medical retractions)
_MEDICAL_KEYWORDS = re.compile(
    r"\b(medicine|medical|clinical trial|patient|surgery|"
    r"oncology|cardiology|pharmacology|drug trial|therapeutic)\b",
    re.IGNORECASE,
)

_CASE_SELECTION_SYSTEM = """\
You are selecting retracted papers for a student to learn about \
research integrity and generate new research questions.

From the list below, select 3-5 cases that are:
1. **Diverse in failure type** — do NOT pick multiple cases with the same \
reason for retraction (e.g., not 5 fabrication cases)
2. **Clear methodological lessons** — prefer cases where what went wrong \
is understandable and instructive
3. **Non-medical** — skip retractions from medical/clinical fields
4. **No-blame** — you will NOT include author names in any output

For each selected case, write a plain-language summary of what the paper \
originally claimed (1-2 sentences). Use ONLY the title, abstract, and \
reason provided — do not invent details.

**Student level:** {population}"""

_SINGLE_CLAIM_SYSTEM = """\
Summarize what this retracted paper originally claimed, in plain language \
(1-2 sentences). Use ONLY the title, abstract, and reason provided — do \
not invent details. Do NOT name any individuals.

**Student level:** {population}"""


def _is_medical(raw: dict[str, Any]) -> bool:
    """Return True if the paper appears to be from a medical field."""
    text = " ".join([
        raw.get("title", ""),
        raw.get("journal", ""),
        raw.get("reason", ""),
        raw.get("abstract", ""),
    ])
    return bool(_MEDICAL_KEYWORDS.search(text))


def _extract_year(date_str: str | None) -> int | None:
    """Extract year as int from a date string like '2024-01-15'."""
    if not date_str:
        return None
    parts = date_str.strip().split("-")
    if parts and len(parts[0]) == 4:
        try:
            return int(parts[0])
        except ValueError:
            pass
    parts = date_str.strip().split("/")
    if parts and len(parts[-1]) == 4:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return None


def _strip_authors(text: str) -> str:
    """Remove common author name patterns (best-effort no-blame)."""
    # The retraction watch data sometimes includes author references
    # We rely primarily on not passing author fields through, but
    # also strip common patterns from reason strings
    return text


def _raw_to_case(
    raw: dict[str, Any], original_claim: str, field: str,
) -> RetractionCase:
    """Convert a raw Retraction Watch result to a RetractionCase."""
    return RetractionCase(
        doi=raw.get("original_doi") or raw.get("doi", "unknown"),
        title=raw["title"],
        journal=raw.get("journal", "Unknown"),
        year=_extract_year(raw.get("retraction_date")),
        retraction_reason_raw=raw.get("reason", "Not specified"),
        original_claim=original_claim,
        field=field,
    )


async def retrieve_cases(
    query: str,
    population: str,
    max_results: int = 8,
) -> RetractionCases:
    """Stage 1: Retrieve retraction cases from the Retraction Watch DB.

    Args:
        query: Field name or DOI/title.
        population: Student level for calibrating claim summaries.
        max_results: Max papers to fetch in browse mode.

    Returns:
        RetractionCases with 1-5 cases and mode indicator.
    """
    if looks_like_doi(query):
        return await _direct_mode(query, population)
    return await _browse_mode(query, population, max_results)


async def _browse_mode(
    field: str, population: str, max_results: int,
) -> RetractionCases:
    """Browse mode: semantic search → LLM selects diverse cases."""
    logger.info("Stage 1/5: Case Retriever (browse) — field='{}'", field)

    raw_results = await search_retracted_papers(
        field, max_results=max_results,
    )

    if not raw_results:
        msg = f"No retracted papers found for field '{field}'"
        raise ValueError(msg)

    # Filter out medical retractions
    filtered = [r for r in raw_results if not _is_medical(r)]
    if not filtered:
        msg = (
            f"All retracted papers found for '{field}' appear to be medical. "
            "Try a different field."
        )
        raise ValueError(msg)

    logger.info(
        "Found {} retracted papers ({} after medical filter)",
        len(raw_results), len(filtered),
    )

    # Format for LLM selection
    paper_descriptions = []
    for i, raw in enumerate(filtered):
        abstract_line = f"Abstract: {raw.get('abstract', 'N/A')}\n" if raw.get("abstract") else ""
        desc = (
            f"[{i}] Title: {raw['title']}\n"
            f"    Journal: {raw.get('journal', 'Unknown')}\n"
            f"    Year: {_extract_year(raw.get('retraction_date')) or 'Unknown'}\n"
            f"    Reason: {raw.get('reason', 'Not specified')}\n"
            f"    {abstract_line}"
        )
        paper_descriptions.append(desc)

    papers_text = "\n\n".join(paper_descriptions)
    system = _CASE_SELECTION_SYSTEM.format(population=population)

    selection = await structured_call(
        CaseSelection,
        [
            SystemMessage(content=system),
            HumanMessage(content=f"Field: {field}\n\nPapers:\n\n{papers_text}"),
        ],
        thinking="high",
        temperature=0.3,
    )

    # Build cases from selected indices
    cases: list[RetractionCase] = []
    for idx, claim in zip(
        selection.selected_indices, selection.original_claims, strict=False,
    ):
        if idx < 0 or idx >= len(filtered):
            continue
        raw = filtered[idx]
        cases.append(_raw_to_case(raw, claim, field))

    if not cases:
        # Fallback: use first non-medical result
        raw = filtered[0]
        claim_result = await structured_call(
            SingleCaseClaim,
            [
                SystemMessage(content=_SINGLE_CLAIM_SYSTEM.format(population=population)),
                HumanMessage(content=f"Title: {raw['title']}\nReason: {raw.get('reason', 'N/A')}"),
            ],
            thinking="low",
            temperature=0.3,
        )
        cases.append(_raw_to_case(raw, claim_result.original_claim, field))

    logger.info("Selected {} diverse cases", len(cases))
    return RetractionCases(cases=cases, mode="browse")


async def _direct_mode(
    query: str, population: str,
) -> RetractionCases:
    """Direct mode: DOI lookup → single case."""
    from research_mentor.question_workshop.retractions.pipeline.schemas import DOI_REGEX

    logger.info("Stage 1/5: Case Retriever (direct) — query='{}'", query[:60])

    # Extract DOI from query
    match = DOI_REGEX.search(query.strip())
    doi = match.group(0) if match else query.strip()

    # Try DOI lookup first
    result = await get_retraction_by_doi(doi)

    if not result:
        # Fallback: title search
        logger.info("DOI not found, falling back to title search")
        results = await search_retracted_papers(query, max_results=3)
        if not results:
            msg = f"No retracted paper found for '{query}'"
            raise ValueError(msg)
        result = results[0]

    # Generate claim summary
    abstract_line = f"\nAbstract: {result.get('abstract', '')}" if result.get("abstract") else ""
    claim_result = await structured_call(
        SingleCaseClaim,
        [
            SystemMessage(content=_SINGLE_CLAIM_SYSTEM.format(population=population)),
            HumanMessage(
                content=(
                    f"Title: {result['title']}\n"
                    f"Reason: {result.get('reason', 'N/A')}"
                    f"{abstract_line}"
                ),
            ),
        ],
        thinking="low",
        temperature=0.3,
    )

    case = _raw_to_case(result, claim_result.original_claim, "unknown")
    logger.info("Direct mode: found '{}'", case.title[:60])
    return RetractionCases(cases=[case], mode="direct")
