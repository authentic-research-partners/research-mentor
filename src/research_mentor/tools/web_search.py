"""Web search: Brave Search API or Tavily Search API.

Two backends available — configure whichever you prefer:

Brave Search API — sign up at https://brave.com/search/api/
  $5/mo free credits (~1,000 queries). Paid tiers available.
  Supports web, news, images, videos.

Tavily Search API — sign up at https://tavily.com/
  Free: 1,000 credits/mo, no card required. Paid plans available.
  Returns extracted content snippets (not just links).

Configure ONE by placing the API key in a file:
  echo "BSA..." > ~/.research-mentor/brave_search_api_key    # Brave
  echo "tvly-..." > ~/.research-mentor/tavily_api_key        # Tavily

If both are configured, Brave is used.
If neither is configured, web search is disabled (agent still works).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
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
from research_mentor.tools.status import ToolStatus, record_tool_usage

# ---------------------------------------------------------------------------
# Rate limiters (initialized lazily from config on first use)
# ---------------------------------------------------------------------------

_brave_limiters_initialized = False
_brave_per_second_limiter: MonotonicRateLimiter
_brave_monthly_limiter: MonotonicRateLimiter

_tavily_limiter_initialized = False
_tavily_per_second_limiter: MonotonicRateLimiter


def _ensure_brave_limiters() -> None:
    global _brave_limiters_initialized, _brave_per_second_limiter, _brave_monthly_limiter
    if _brave_limiters_initialized:
        return
    rl = load_config().rate_limits
    _brave_per_second_limiter = MonotonicRateLimiter(
        rl.brave_search_per_second[0], rl.brave_search_per_second[1],
    )
    _brave_monthly_limiter = MonotonicRateLimiter(
        rl.brave_search_monthly[0], rl.brave_search_monthly[1],
    )
    _brave_limiters_initialized = True


def _ensure_tavily_limiter() -> None:
    global _tavily_limiter_initialized, _tavily_per_second_limiter
    if _tavily_limiter_initialized:
        return
    rl = load_config().rate_limits
    _tavily_per_second_limiter = MonotonicRateLimiter(
        rl.tavily_per_second[0], rl.tavily_per_second[1],
    )
    _tavily_limiter_initialized = True




# ---------------------------------------------------------------------------
# API key loaders
# ---------------------------------------------------------------------------


def _read_brave_api_key() -> str | None:
    """Read Brave Search API key from file. Returns None if not configured."""
    cfg = load_config()
    key_path = Path(cfg.services.brave_search.api_key_file).expanduser()
    if not key_path.exists():
        return None
    text = key_path.read_text().strip()
    return text if text else None


def _read_tavily_api_key() -> str | None:
    """Read Tavily API key from file. Returns None if not configured."""
    cfg = load_config()
    key_path = Path(cfg.services.tavily.api_key_file).expanduser()
    if not key_path.exists():
        return None
    text = key_path.read_text().strip()
    return text if text else None


# ===========================================================================
# Brave Search
# ===========================================================================

_BRAVE_TOOL_NAME = "brave_search"

_ENDPOINTS: dict[str | None, str] = {
    "news": "https://api.search.brave.com/res/v1/news/search",
    "images": "https://api.search.brave.com/res/v1/images/search",
    "videos": "https://api.search.brave.com/res/v1/videos/search",
    None: "https://api.search.brave.com/res/v1/web/search",
}


async def _search_brave_internal(
    query: str,
    num_results: int,
    search_type: str | None,
    api_key: str,
) -> dict[str, Any]:
    """Execute Brave Search API call with rate limiting."""
    endpoint = _ENDPOINTS.get(search_type, _ENDPOINTS[None])

    params: dict[str, str | int] = {"q": query, "count": num_results}
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    logger.info(
        "Brave Search: query='{}', type={}",
        query, search_type or "web",
    )

    try:
        @retry_on_429()
        async def _request() -> httpx.Response:
            async with _brave_per_second_limiter:
                async with _brave_monthly_limiter:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        resp = await client.get(
                            endpoint, params=params, headers=headers,
                        )
                    if resp.status_code == 429:
                        raise RateLimitedError(
                            "Brave Search",
                            retry_after=parse_retry_after(
                                resp.headers.get("Retry-After"),
                            ),
                        )
                    return resp

        try:
            response = await _request()
        except RateLimitedError:
            logger.warning(
                "Brave Search rate limit exceeded for query: {}", query,
            )
            msg = (
                "Rate limit exceeded. "
                "Check your Brave Search quota."
            )
            return {
                "success": False, "results": [],
                "error": msg,
                "tool_status": ToolStatus.error(
                    "Brave Search", "rate_limit", msg,
                ),
            }

        if response.status_code in (401, 403):
            logger.error("Brave Search authentication failed")
            msg = (
                "Authentication failed. "
                "Check your Brave Search API key."
            )
            return {
                "success": False, "results": [],
                "error": msg,
                "tool_status": ToolStatus.error(
                    "Brave Search", "auth", msg,
                ),
            }

        if response.status_code != 200:
            logger.error(
                "Brave Search API error: {} - {}",
                response.status_code, response.text,
            )
            msg = f"Search API error: HTTP {response.status_code}"
            return {
                "success": False, "results": [],
                "error": msg,
                "tool_status": ToolStatus.error(
                    "Brave Search", "network", msg,
                ),
            }

        data = response.json()
        results = _extract_brave_results(data, search_type)
        query_info = data.get("query", {})

        logger.info(
            "Brave Search successful: query='{}', type={}, results={}",
            query, search_type or "web", len(results),
        )
        return {
            "success": True,
            "results": results,
            "query": query_info.get("original", query),
            "search_type": search_type or "web",
            "tool_status": ToolStatus.ok("Brave Search"),
        }

    except httpx.TimeoutException:
        logger.error("Timeout during Brave Search for query: {}", query)
        msg = "Search request timed out"
        return {
            "success": False, "results": [],
            "error": msg,
            "tool_status": ToolStatus.error("Brave Search", "timeout", msg),
        }
    except Exception as e:
        logger.exception("Unexpected error during Brave Search: {}", e)
        msg = f"Search failed: {e}"
        return {
            "success": False, "results": [],
            "error": msg,
            "tool_status": ToolStatus.error("Brave Search", "network", msg),
        }


def _extract_brave_results(
    data: dict[str, Any], search_type: str | None,
) -> list[dict[str, Any]]:
    """Extract results from Brave Search API response."""
    results: list[dict[str, Any]] = []

    if search_type == "news":
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
                "age": item.get("age", ""),
                "source": item.get("meta_url", {}).get("hostname", ""),
            })
    elif search_type == "images":
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "thumbnail": item.get("thumbnail", {}).get("src", ""),
                "source": item.get("source", ""),
            })
    elif search_type == "videos":
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
                "thumbnail": item.get("thumbnail", {}).get("src", ""),
                "age": item.get("age", ""),
            })
    else:
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
                "age": item.get("age", ""),
                "language": item.get("language", ""),
                "family_friendly": item.get("family_friendly", True),
            })

    return results


# ===========================================================================
# Tavily Search
# ===========================================================================

_TAVILY_TOOL_NAME = "tavily_search"


async def _search_tavily_internal(
    query: str,
    num_results: int,
    search_type: str | None,
    api_key: str,
) -> dict[str, Any]:
    """Execute Tavily Search API call with rate limiting."""
    from tavily import AsyncTavilyClient

    # Map search_type to Tavily topic
    topic = "news" if search_type == "news" else "general"

    logger.info(
        "Tavily Search: query='{}', topic={}", query, topic,
    )

    @retry_on_429()
    async def _request() -> Any:
        async with _tavily_per_second_limiter:
            try:
                client = AsyncTavilyClient(api_key=api_key)
                return await client.search(
                    query,
                    max_results=num_results,
                    topic=topic,
                )
            except Exception as e:
                err_name = type(e).__name__
                err_str = str(e).lower()
                # Non-retryable errors — re-raise wrapped to distinguish them
                if "UsageLimitExceeded" in err_name:
                    raise
                if "InvalidAPIKey" in err_name:
                    raise
                if "Timeout" in err_name:
                    raise
                # 429 / rate-limit — convert to RateLimitedError for tenacity
                if "429" in err_str or "rate" in err_str:
                    raise RateLimitedError("Tavily Search") from e
                raise

    try:
        response = await _request()
    except RateLimitedError:
        msg = "Tavily rate limited (429). Try again later."
        return {
            "success": False, "results": [],
            "error": msg,
            "tool_status": ToolStatus.error(
                "Tavily Search", "rate_limit", msg,
            ),
        }
    except Exception as e:
        err_name = type(e).__name__
        if "UsageLimitExceeded" in err_name:
            logger.warning("Tavily quota exceeded for query: {}", query)
            msg = "Tavily credit quota exceeded"
            return {
                "success": False, "results": [],
                "error": msg,
                "tool_status": ToolStatus.error(
                    "Tavily Search", "rate_limit", msg,
                ),
            }
        if "InvalidAPIKey" in err_name:
            logger.error("Tavily authentication failed")
            msg = "Authentication failed. Check your Tavily API key."
            return {
                "success": False, "results": [],
                "error": msg,
                "tool_status": ToolStatus.error(
                    "Tavily Search", "auth", msg,
                ),
            }
        if "Timeout" in err_name:
            logger.error("Tavily timeout for query: {}", query)
            msg = "Tavily search request timed out"
            return {
                "success": False, "results": [],
                "error": msg,
                "tool_status": ToolStatus.error(
                    "Tavily Search", "timeout", msg,
                ),
            }
        logger.exception("Tavily search failed: {}", e)
        msg = f"Tavily search failed: {e}"
        return {
            "success": False, "results": [],
            "error": msg,
            "tool_status": ToolStatus.error("Tavily Search", "network", msg),
        }

    # Normalize Tavily results to match our standard format
    results: list[dict[str, Any]] = []
    for item in response.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("content", ""),
            "score": item.get("score", 0.0),
        })

    logger.info(
        "Tavily Search successful: query='{}', results={}",
        query, len(results),
    )
    return {
        "success": True,
        "results": results,
        "query": query,
        "search_type": search_type or "web",
        "tool_status": ToolStatus.ok("Tavily Search"),
    }


# ===========================================================================
# Public API
# ===========================================================================


async def search_web(
    query: str,
    num_results: int = 5,
    search_type: str | None = None,
) -> dict[str, Any]:
    """Search the web using a configured search backend.

    Checks for API keys in order: Brave Search, then Tavily.
    Uses whichever is configured. If both are configured, Brave is used.

    Args:
        query: Search query string
        num_results: Number of results (1-20, default 5)
        search_type: Optional type ('news', 'images', 'videos', None for web).
                     Tavily supports 'news' topic; images/videos are web-only.

    Returns:
        {"success": bool, "results": list, "error"?: str,
         "query"?: str, "search_type"?: str}
    """
    if not query or not query.strip():
        msg = "Empty query provided"
        return {
            "success": False, "results": [],
            "error": msg,
            "tool_status": ToolStatus.error("Web Search", "network", msg),
        }

    num_results = max(1, min(20, num_results))

    # Safety review: check for accidental PII leaks before sending externally
    from research_mentor.tools.query_safety_review import review_outgoing_query

    if not await review_outgoing_query(query, "search_web"):
        return {
            "success": False, "results": [],
            "error": "Query blocked by safety review (may contain sensitive data)",
            "tool_status": ToolStatus.error(
                "Web Search", "safety", "Query blocked by safety review",
            ),
        }

    brave_key = _read_brave_api_key()
    tavily_key = _read_tavily_api_key() if brave_key is None else None

    import time as _time

    t0 = _time.monotonic()

    if brave_key is not None:
        _ensure_brave_limiters()
        tool_name = _BRAVE_TOOL_NAME
        try:
            result = await asyncio.wait_for(
                _search_brave_internal(
                    query, num_results, search_type, brave_key,
                ),
                timeout=10.0,
            )
        except TimeoutError:
            logger.error(
                "Brave Search timeout after 10s for query: {}", query,
            )
            msg = (
                "Search timed out after 10 seconds. "
                "Rate limit quota may be exhausted."
            )
            result = {
                "success": False, "results": [],
                "error": msg,
                "tool_status": ToolStatus.error(
                    "Brave Search", "timeout", msg,
                ),
            }
    elif tavily_key is not None:
        if search_type in ("images", "videos"):
            logger.debug(
                "Tavily does not support search_type='{}', "
                "performing web search instead", search_type,
            )
        _ensure_tavily_limiter()
        tool_name = _TAVILY_TOOL_NAME
        try:
            result = await asyncio.wait_for(
                _search_tavily_internal(
                    query, num_results, search_type, tavily_key,
                ),
                timeout=15.0,
            )
        except TimeoutError:
            logger.error(
                "Tavily Search timeout after 15s for query: {}", query,
            )
            msg = "Tavily search timed out after 15 seconds"
            result = {
                "success": False, "results": [],
                "error": msg,
                "tool_status": ToolStatus.error(
                    "Tavily Search", "timeout", msg,
                ),
            }
    else:
        msg = (
            "Web search not configured. Set up one of:\n"
            "  Brave Search: echo 'BSA...' > "
            "~/.research-mentor/brave_search_api_key\n"
            "    Sign up at https://brave.com/search/api/\n"
            "  Tavily: echo 'tvly-...' > "
            "~/.research-mentor/tavily_api_key\n"
            "    Sign up at https://tavily.com/"
        )
        return {
            "success": False, "results": [],
            "error": msg,
            "tool_status": ToolStatus.error(
                "Web Search", "unavailable", msg,
            ),
        }

    elapsed = _time.monotonic() - t0
    ts = result.get("tool_status")
    asyncio.create_task(record_tool_usage(
        tool_name,
        query=query,
        success=result.get("success", False),
        error_category=ts.category if ts and ts.level != "ok" else None,
        elapsed=elapsed,
    ))
    return result
