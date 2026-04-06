"""Trending science news search tool.

Thin wrapper around ``research_mentor.tools.web_search.search_web``
with science/health filtering.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from research_mentor.tools.web_search import search_web

# Keywords indicating science/health content
_SCIENCE_KEYWORDS = [
    "study", "research", "scientists", "researchers", "health", "medical",
    "clinical", "trial", "found", "discovered", "linked", "causes",
    "effects", "benefits",
]

# Keywords indicating political/social content (filter out)
_POLITICAL_KEYWORDS = [
    "election", "vote", "congress", "senate", "republican", "democrat",
    "trump", "biden", "policy", "legislation", "immigration", "border",
]


def _is_science_relevant(result: dict[str, Any]) -> bool:
    """Check if a search result is science-relevant (not political)."""
    title = result.get("title", "").lower()
    description = result.get("description", "").lower()
    combined = f"{title} {description}"

    for keyword in _POLITICAL_KEYWORDS:
        if keyword in combined:
            return False

    for keyword in _SCIENCE_KEYWORDS:
        if keyword in combined:
            return True

    # Default: accept (might be science-related)
    return True


async def search_trending_science_news(
    topic: str | None = None,
    num_results: int = 10,
) -> dict[str, Any]:
    """Search for trending science/health news.

    Args:
        topic: Optional topic to search for (e.g., "nutrition", "health").
        num_results: Number of results to return (default: 10).

    Returns:
        Dict with keys: results, query.
    """
    if topic:
        query = f"{topic} health science study research"
    else:
        query = "health science study new research"

    logger.info("Searching trending science news: query='{}'", query)

    result = await search_web(
        query=query,
        num_results=num_results * 2,  # Fetch extra for filtering
        search_type="news",
    )

    filtered_results = [
        r for r in result.get("results", [])
        if _is_science_relevant(r)
    ][:num_results]

    logger.info(
        "Science news search: {} total, {} after filtering",
        len(result.get("results", [])), len(filtered_results),
    )

    return {
        "results": filtered_results,
        "query": query,
    }
