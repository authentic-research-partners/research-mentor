"""Shared citation formatting utilities for question workshops.

All citation formatting goes through code — the LLM never generates URLs
or formats citations. It outputs {text, papers_cited: [1, 3]} via
CitedResponse, and these functions handle the rest deterministically.
"""

from __future__ import annotations

from typing import Any


def _best_access_url(paper: dict[str, Any]) -> tuple[str, str] | None:
    """Pick the best access URL for a paper, with a label.

    Priority: arXiv PDF > CORE download > Unpaywall OA > OpenAlex OA.
    Returns (label, url) or None if no OA link available.
    """
    # arXiv PDF — always the best (direct, free, reliable)
    pdf_url = paper.get("pdf_url")
    if pdf_url and "arxiv.org" in pdf_url:
        return ("PDF", pdf_url)

    # CORE download URL
    download_url = paper.get("download_url")
    if download_url:
        return ("Read", download_url)

    # Unpaywall OA URL
    unpaywall_url = paper.get("unpaywall_url")
    if unpaywall_url:
        return ("Read", unpaywall_url)

    # OpenAlex OA URL (pdf_url from oa_url field)
    if pdf_url:
        return ("PDF", pdf_url)

    return None


def format_cited_response(
    text: str, papers_cited: list[int], papers: list[dict[str, Any]],
) -> str:
    """Build final response text with appended citations and access links.

    The LLM produces text + paper numbers. This function appends
    real citations deterministically — the LLM never touches citation
    formatting or URLs.

    Citation format:
        Author (Year), "Title" [PDF](url) DOI
        Author (Year), "Title" [Read](url) DOI
        Author (Year), "Title" DOI (no free full text found)
    """
    if not papers_cited:
        return text

    citations = []
    for num in papers_cited:
        idx = num - 1
        if 0 <= idx < len(papers):
            p = papers[idx]
            authors = p.get("authors", ["?"])
            first = authors[0] if authors else "?"
            cite = f'{first} ({p.get("year", "?")}), "{p.get("title", "?")}"'

            # Access link (code-generated, never from LLM)
            access = _best_access_url(p)
            if access:
                label, url = access
                cite += f" [{label}]({url})"

            doi = p.get("doi")
            if doi:
                cite += f" {doi}"

            # No OA link — tell the student
            if not access:
                cite += " (no free full text found)"

            citations.append(cite)

    if citations:
        return f"{text}\n\n*References: {'; '.join(citations)}*"
    return text


def format_papers_for_prompt(papers: list[dict[str, Any]], limit: int = 10) -> str:
    """Format papers as a numbered list for CitedResponse prompts.

    Returns e.g. '[1] "Title" — Author (Year)' so the LLM can reference
    by number and code appends the real citation.
    """
    lines = []
    for i, p in enumerate(papers[:limit], 1):
        title = p.get("title", "Untitled")
        authors = p.get("authors", ["?"])
        first = authors[0] if authors else "?"
        year = p.get("year", "?")
        lines.append(f'[{i}] "{title}" — {first} ({year})')
    return "\n".join(lines)
