"""Academic search tools — arXiv, Semantic Scholar, PubMed, OpenAlex, Europe PMC.

All five APIs are free and require no API keys. Rate limiting is client-side
using MonotonicRateLimiter. arXiv uses the `arxiv` library (delegated to thread
pool); the rest use httpx directly (no wrapper libs).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from xml.etree.ElementTree import ParseError as _XMLParseError

import httpx
from defusedxml.common import DefusedXmlException as _DefusedXmlError
from defusedxml.ElementTree import fromstring as _safe_fromstring
from loguru import logger
from tenacity import stop_after_attempt

from research_mentor.config import load_config
from research_mentor.tools.rate_limiter import (
    MonotonicRateLimiter,
    RateLimitedError,
    parse_retry_after,
    retry_on_429,
)
from research_mentor.tools.status import ToolStatus, make_error_result, record_tool_usage


def decode_inverted_abstract(inverted: dict[str, list[int]]) -> str:
    """Reconstruct abstract text from an OpenAlex inverted index."""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)

# ---------------------------------------------------------------------------
# Rate limiters (initialized lazily from config on first use)
# ---------------------------------------------------------------------------

_limiters_initialized = False
_arxiv_limiter: MonotonicRateLimiter
_semantic_scholar_limiter: MonotonicRateLimiter
_pubmed_limiter: MonotonicRateLimiter
_openalex_limiter: MonotonicRateLimiter


def _ensure_limiters() -> None:
    """Initialize module-level rate limiters from config (once).

    Uses ``MonotonicRateLimiter`` which persists token-bucket state
    across event-loop boundaries.

    Uses polite-pool limits for PubMed and OpenAlex when a courtesy email
    is configured, otherwise falls back to conservative anonymous limits.
    """
    global _limiters_initialized, _arxiv_limiter, _semantic_scholar_limiter
    global _pubmed_limiter, _openalex_limiter

    if _limiters_initialized:
        return

    cfg = load_config()
    rl = cfg.rate_limits
    has_email = bool(cfg.services.polite_email)

    _arxiv_limiter = MonotonicRateLimiter(rl.arxiv[0], rl.arxiv[1])
    _semantic_scholar_limiter = MonotonicRateLimiter(
        rl.semantic_scholar[0], rl.semantic_scholar[1],
    )

    pubmed_rl = rl.pubmed_polite if has_email else rl.pubmed
    _pubmed_limiter = MonotonicRateLimiter(pubmed_rl[0], pubmed_rl[1])

    openalex_rl = rl.openalex_polite if has_email else rl.openalex
    _openalex_limiter = MonotonicRateLimiter(openalex_rl[0], openalex_rl[1])

    if has_email:
        logger.info("Academic rate limits: polite pool (email configured)")
    else:
        logger.info("Academic rate limits: anonymous (no courtesy email)")

    _limiters_initialized = True


def get_openalex_limiter() -> MonotonicRateLimiter:
    """Return the OpenAlex rate limiter, initializing if needed."""
    _ensure_limiters()
    return _openalex_limiter



# ---------------------------------------------------------------------------
# arXiv (uses `arxiv` library — only sync lib we keep, it's well-behaved)
# ---------------------------------------------------------------------------

_arxiv_client: Any = None
_arxiv_lock = asyncio.Lock()


async def _get_arxiv_client() -> Any:
    global _arxiv_client
    if _arxiv_client is None:
        async with _arxiv_lock:
            if _arxiv_client is None:
                import arxiv
                _arxiv_client = arxiv.Client(
                    page_size=100, delay_seconds=3, num_retries=3,
                )
    return _arxiv_client


def _search_arxiv_sync(
    query: str, max_results: int, category: str | None, client: Any,
) -> list[dict[str, Any]]:
    """Synchronous arXiv search (runs in thread pool)."""
    import arxiv

    try:
        search_query = f"{query} AND cat:{category}" if category else query
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results: list[dict[str, Any]] = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "abstract": paper.summary.replace("\n", " ").strip(),
                "url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "published": paper.published.strftime("%Y-%m-%d"),
                "categories": paper.categories,
                "source": "arXiv",
            })
        return results
    except Exception as e:
        msg = f"arXiv search failed: {e}"
        return [{"error": msg, "source": "arXiv",
                 "tool_status": ToolStatus.error("arXiv", "network", msg)}]


async def search_arxiv(
    query: str,
    max_results: int = 5,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Search arXiv for research papers (physics, CS, math, etc.).

    Args:
        query: Search query (e.g., "projectile motion drag coefficient")
        max_results: Maximum results to return (default 5)
        category: Optional arXiv category filter (e.g., "physics", "cs", "math")

    Returns:
        List of paper dicts with title, authors, abstract, url, pdf_url,
        published, categories, source.
    """
    _ensure_limiters()
    import time as _time

    t0 = _time.monotonic()
    try:
        async def _with_rate_limit() -> list[dict[str, Any]]:
            async with _arxiv_limiter:
                client = await _get_arxiv_client()
                return await asyncio.to_thread(
                    _search_arxiv_sync, query, max_results, category, client,
                )

        results = await asyncio.wait_for(_with_rate_limit(), timeout=15.0)
    except TimeoutError:
        logger.error("arXiv search timeout after 15s for query: {}", query)
        results = make_error_result("arXiv", "timeout", "arXiv search timed out after 15 seconds.")
    asyncio.create_task(
        record_tool_usage("arxiv", query=query, results=results, elapsed=_time.monotonic() - t0),
    )
    return results


# ---------------------------------------------------------------------------
# Semantic Scholar (direct httpx — library hangs on 429)
# ---------------------------------------------------------------------------

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_SEARCH_FIELDS = (
    "title,authors,abstract,year,citationCount,"
    "fieldsOfStudy,url,externalIds,openAccessPdf"
)


async def search_semantic_scholar(
    query: str,
    max_results: int = 5,
    year_filter: str | None = None,
    fields_of_study: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search Semantic Scholar for papers across all disciplines.

    Args:
        query: Search query
        max_results: Maximum results (default 5)
        year_filter: Year range (e.g., "2020-" for 2020+, "2015-2020")
        fields_of_study: Field filters (e.g., ["Biology", "Physics"])

    Returns:
        List of paper dicts with title, authors, abstract, url, year,
        citation_count, fields_of_study, source.
    """
    _ensure_limiters()

    async def _with_rate_limit() -> list[dict[str, Any]]:
        params: dict[str, str | int] = {
            "query": query,
            "limit": max_results,
            "fields": _S2_SEARCH_FIELDS,
        }
        if year_filter:
            params["year"] = year_filter
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        @retry_on_429()
        async def _request() -> dict[str, Any]:
            async with _semantic_scholar_limiter:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"{_S2_BASE}/paper/search", params=params,
                    )
                if resp.status_code == 429:
                    raise RateLimitedError(
                        "Semantic Scholar",
                        retry_after=parse_retry_after(resp.headers.get("Retry-After")),
                    )
                resp.raise_for_status()
                result: dict[str, Any] = resp.json()
                return result

        try:
            data = await _request()
        except RateLimitedError:
            msg = "Semantic Scholar rate limited (429). Try again later."
            return [{
                "error": msg, "source": "Semantic Scholar",
                "tool_status": ToolStatus.error(
                    "Semantic Scholar", "rate_limit", msg,
                ),
            }]

        results: list[dict[str, Any]] = []
        for paper in data.get("data", []):
            if not paper.get("abstract"):
                continue

            arxiv_id = None
            ext_ids = paper.get("externalIds") or {}
            if ext_ids.get("ArXiv"):
                arxiv_id = ext_ids["ArXiv"]

            authors = [
                a.get("name", "") for a in (paper.get("authors") or [])
            ]
            paper_id = paper.get("paperId", "")
            entry: dict[str, Any] = {
                "title": paper.get("title", ""),
                "authors": authors,
                "abstract": paper["abstract"],
                "url": (
                    paper.get("url")
                    or f"https://www.semanticscholar.org/paper/{paper_id}"
                ),
                "year": paper.get("year"),
                "citation_count": paper.get("citationCount") or 0,
                "fields_of_study": paper.get("fieldsOfStudy") or [],
                "source": "Semantic Scholar",
            }
            if arxiv_id:
                entry["arxiv_id"] = arxiv_id
                entry["pdf_url"] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                entry["arxiv_abs_url"] = f"https://arxiv.org/abs/{arxiv_id}"

            s2_pdf = (paper.get("openAccessPdf") or {}).get("url")
            if s2_pdf:
                entry["s2_pdf_url"] = s2_pdf

            results.append(entry)

        return results

    import time as _time

    t0 = _time.monotonic()
    try:
        results = await asyncio.wait_for(_with_rate_limit(), timeout=15.0)
    except TimeoutError:
        logger.error("Semantic Scholar search timeout after 15s for query: {}", query)
        results = make_error_result(
            "Semantic Scholar", "timeout", "Semantic Scholar search timed out after 15 seconds.",
        )
    except Exception as e:
        msg = f"Semantic Scholar search failed: {e}"
        results = make_error_result("Semantic Scholar", "network", msg)
    asyncio.create_task(
        record_tool_usage(
            "semantic_scholar", query=query, results=results, elapsed=_time.monotonic() - t0,
        ),
    )
    return results


async def enrich_with_semantic_scholar_retry(
    doi: str, max_retries: int = 3,
) -> dict[str, Any]:
    """Get Semantic Scholar data (TLDRs, citations, arXiv IDs) by DOI with retry.

    Uses tenacity exponential backoff on 429. Returns dict with 'success' flag.
    """
    _ensure_limiters()
    _fail = {"tldr": None, "citation_count": 0, "arxiv_id": None, "success": False}

    @retry_on_429(stop=stop_after_attempt(max_retries))
    async def _request() -> dict[str, Any]:
        async with _semantic_scholar_limiter:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{_S2_BASE}/paper/DOI:{doi}",
                    params={
                        "fields": "title,abstract,tldr,externalIds,citationCount",
                    },
                )
            if resp.status_code == 429:
                raise RateLimitedError(
                    "Semantic Scholar", f"DOI: {doi}",
                    retry_after=parse_retry_after(resp.headers.get("Retry-After")),
                )
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            return result

    try:
        data = await _request()
    except RateLimitedError:
        return {**_fail, "error": "Rate limited (429)"}
    except (httpx.TimeoutException, httpx.HTTPError) as e:
        return {**_fail, "error": str(e)}

    arxiv_id = None
    ext_ids = data.get("externalIds", {})
    if ext_ids.get("ArXiv"):
        arxiv_id = ext_ids["ArXiv"]

    tldr_obj = data.get("tldr")
    tldr = (
        tldr_obj.get("text")
        if isinstance(tldr_obj, dict) else None
    )

    return {
        "tldr": tldr,
        "citation_count": data.get("citationCount", 0),
        "arxiv_id": arxiv_id,
        "success": True,
    }


# ---------------------------------------------------------------------------
# PubMed (direct httpx via NCBI E-utilities — pymed hangs on errors)
# ---------------------------------------------------------------------------

_PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


async def search_pubmed(
    query: str,
    max_results: int = 5,
    year_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Search PubMed for biomedical research papers.

    Args:
        query: Search query
        max_results: Maximum results (default 5)
        year_filter: Year range in Entrez format (e.g., "2020:2025")

    Returns:
        List of paper dicts with title, authors, abstract, url,
        pubmed_id, publication_date, journal, source.
    """
    _ensure_limiters()

    async def _with_rate_limit() -> list[dict[str, Any]]:
        cfg = load_config()
        email = cfg.services.polite_email or "not-set@example.com"

        search_query = (
            f"{query} AND {year_filter}[pdat]" if year_filter else query
        )

        # Step 1: ESearch — get PMIDs
        search_params: dict[str, str | int] = {
            "db": "pubmed",
            "term": search_query,
            "retmax": max_results,
            "retmode": "json",
            "tool": "ResearchMentorLocal",
            "email": email,
        }

        async with _pubmed_limiter:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(_PUBMED_SEARCH, params=search_params)
                resp.raise_for_status()
                search_data = resp.json()

        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []

        # Step 2: EFetch — get article details as XML
        fetch_params: dict[str, str] = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
            "tool": "ResearchMentorLocal",
            "email": email,
        }

        async with _pubmed_limiter:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(_PUBMED_FETCH, params=fetch_params)
                resp.raise_for_status()
                xml_text = resp.text

        return _parse_pubmed_xml(xml_text)

    import time as _time

    t0 = _time.monotonic()
    try:
        results = await asyncio.wait_for(_with_rate_limit(), timeout=15.0)
    except TimeoutError:
        logger.error("PubMed search timeout after 15s for query: {}", query)
        results = make_error_result(
            "PubMed", "timeout", "PubMed search timed out after 15 seconds.",
        )
    except Exception as e:
        results = make_error_result("PubMed", "network", f"PubMed search failed: {e}")
    asyncio.create_task(
        record_tool_usage("pubmed", query=query, results=results, elapsed=_time.monotonic() - t0),
    )
    return results


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse PubMed EFetch XML into structured dicts."""
    results: list[dict[str, Any]] = []
    try:
        root = _safe_fromstring(xml_text)
    except (_XMLParseError, _DefusedXmlError):
        return make_error_result("PubMed", "network", "Failed to parse PubMed XML")

    for article in root.findall(".//PubmedArticle"):
        medline = article.find(".//MedlineCitation")
        if medline is None:
            continue

        pmid_el = medline.find("PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        art = medline.find("Article")
        if art is None:
            continue

        title_el = art.find("ArticleTitle")
        title = title_el.text if title_el is not None else "Untitled"

        abstract_el = art.find(".//AbstractText")
        abstract = abstract_el.text if abstract_el is not None else "No abstract available"

        journal_el = art.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else "Unknown"

        # Publication date
        pub_date_el = art.find(".//PubDate")
        pub_date = "Unknown"
        if pub_date_el is not None:
            year_el = pub_date_el.find("Year")
            month_el = pub_date_el.find("Month")
            day_el = pub_date_el.find("Day")
            parts = []
            if year_el is not None and year_el.text:
                parts.append(year_el.text)
            if month_el is not None and month_el.text:
                parts.append(month_el.text)
            if day_el is not None and day_el.text:
                parts.append(day_el.text)
            if parts:
                pub_date = "-".join(parts)

        # Authors
        authors: list[str] = []
        for author in art.findall(".//Author"):
            last = author.find("LastName")
            first = author.find("ForeName")
            name_parts = []
            if last is not None and last.text:
                name_parts.append(last.text)
            if first is not None and first.text:
                name_parts.append(first.text)
            if name_parts:
                authors.append(" ".join(name_parts))

        results.append({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "pubmed_id": pmid,
            "publication_date": pub_date,
            "journal": journal,
            "source": "PubMed",
        })

    return results


# ---------------------------------------------------------------------------
# OpenAlex (direct httpx)
# ---------------------------------------------------------------------------


async def search_openalex(
    query: str,
    max_results: int = 20,
    mailto: str | None = None,
) -> list[dict[str, Any]]:
    """Search OpenAlex for research papers (270M+ papers, all disciplines).

    Args:
        query: Search query (e.g., "water droplet optics")
        max_results: Maximum results (default 20)
        mailto: Email for polite pool. If None, uses config value.

    Returns:
        List of paper dicts with title, authors, abstract, url, doi,
        pdf_url, year, citation_count, source, openalex_id, arxiv_id.
    """
    _ensure_limiters()
    if mailto is None:
        mailto = load_config().services.polite_email or None

    async def _with_rate_limit() -> list[dict[str, Any]]:
        sanitized = re.sub(r"[^\w\s-]", "", query).strip()
        if not sanitized:
            return []

        params: dict[str, str | int] = {
            "filter": f"title.search:{sanitized}",
            "per_page": max_results,
            "select": (
                "id,doi,title,authorships,abstract_inverted_index,"
                "publication_year,cited_by_count,primary_location,"
                "ids,open_access"
            ),
        }
        if mailto:
            params["mailto"] = mailto

        async with _openalex_limiter:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    "https://api.openalex.org/works", params=params,
                )
                resp.raise_for_status()
                data = resp.json()

        results: list[dict[str, Any]] = []
        for work in data.get("results", []):
            authors = [
                authorship.get("author", {}).get("display_name", "")
                for authorship in work.get("authorships", [])
                if authorship.get("author", {}).get("display_name")
            ]

            # Reconstruct abstract from inverted index
            abstract_text = None
            inverted = work.get("abstract_inverted_index")
            if inverted:
                abstract_text = decode_inverted_abstract(inverted)

            pdf_url = work.get("open_access", {}).get("oa_url")
            primary_url = (
                work.get("primary_location", {}).get("landing_page_url")
            )

            # Extract arXiv ID if present
            ids = work.get("ids", {})
            arxiv_id = None
            openalex_id_str = str(ids.get("openalex", ""))
            if "arxiv.org/" in openalex_id_str:
                match = re.search(
                    r"arxiv\.org/(\d+\.\d+)", openalex_id_str,
                )
                if match:
                    arxiv_id = match.group(1)

            results.append({
                "title": work.get("title", "Unknown"),
                "authors": authors,
                "abstract": abstract_text or "No abstract available",
                "url": primary_url or work.get("id", ""),
                "doi": work.get("doi"),
                "pdf_url": pdf_url,
                "year": work.get("publication_year"),
                "citation_count": work.get("cited_by_count", 0),
                "source": "OpenAlex",
                "openalex_id": work.get("id"),
                "arxiv_id": arxiv_id,
            })
        return results

    import time as _time

    t0 = _time.monotonic()
    try:
        results = await asyncio.wait_for(_with_rate_limit(), timeout=15.0)
    except TimeoutError:
        logger.error("OpenAlex search timeout after 15s for query: {}", query)
        results = make_error_result(
            "OpenAlex", "timeout", "OpenAlex search timed out after 15 seconds.",
        )
    except Exception as e:
        results = make_error_result("OpenAlex", "network", f"OpenAlex search failed: {e}")
    asyncio.create_task(
        record_tool_usage("openalex", query=query, results=results, elapsed=_time.monotonic() - t0),
    )
    return results


# ---------------------------------------------------------------------------
# Enrichment helper
# ---------------------------------------------------------------------------


async def enrich_paper_with_arxiv(paper: dict[str, Any]) -> dict[str, Any]:
    """Enrich a paper with arXiv PDF link via DOI -> Semantic Scholar."""
    if paper.get("arxiv_id") or paper.get("pdf_url") or "error" in paper:
        return paper

    doi = paper.get("doi")
    if doi:
        enrichment = await enrich_with_semantic_scholar_retry(
            doi, max_retries=2,
        )
        if enrichment.get("success") and enrichment.get("arxiv_id"):
            arxiv_id = enrichment["arxiv_id"]
            paper["arxiv_id"] = arxiv_id
            paper["pdf_url"] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            paper["arxiv_abs_url"] = f"https://arxiv.org/abs/{arxiv_id}"
    return paper


# ---------------------------------------------------------------------------
# Parallel multi-source search
# ---------------------------------------------------------------------------


async def search_papers(
    query: str,
    sources: list[str] | None = None,
    max_results_per_source: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """Search multiple academic databases in parallel and combine results.

    Args:
        query: Search query
        sources: Databases to search. Default: ["openalex", "semantic_scholar"].
            Options: "openalex", "semantic_scholar", "pubmed", "arxiv", "europepmc".
        max_results_per_source: Max results from each source (default 3)
        **kwargs: Filters forwarded to individual search functions:
            mailto, year_filter, category, fields_of_study.

    Returns:
        {"query": str, "total_results": int, "results": list[dict]}
    """
    if sources is None:
        sources = ["openalex", "semantic_scholar"]

    tasks: list[asyncio.Task[list[dict[str, Any]]]] = []
    loop = asyncio.get_running_loop()

    if "openalex" in sources:
        tasks.append(loop.create_task(search_openalex(
            query,
            max_results=max_results_per_source,
            mailto=kwargs.get("mailto"),
        )))
    if "arxiv" in sources:
        tasks.append(loop.create_task(search_arxiv(
            query,
            max_results=max_results_per_source,
            category=kwargs.get("category"),
        )))
    if "semantic_scholar" in sources:
        tasks.append(loop.create_task(search_semantic_scholar(
            query,
            max_results=max_results_per_source,
            year_filter=kwargs.get("year_filter"),
            fields_of_study=kwargs.get("fields_of_study"),
        )))
    if "pubmed" in sources:
        tasks.append(loop.create_task(search_pubmed(
            query,
            max_results=max_results_per_source,
            year_filter=kwargs.get("year_filter"),
        )))
    if "europepmc" in sources:
        from research_mentor.tools.europepmc_search import search_europepmc

        tasks.append(loop.create_task(search_europepmc(
            query,
            max_results=max_results_per_source,
            year_filter=kwargs.get("year_filter"),
        )))

    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    all_results: list[dict[str, Any]] = []
    tool_statuses: list[ToolStatus] = []
    for results in results_list:
        if isinstance(results, Exception):
            status = ToolStatus.error("unknown", "network", str(results))
            all_results.append({
                "error": str(results), "source": "unknown",
                "tool_status": status,
            })
            tool_statuses.append(status)
        elif isinstance(results, list):
            for item in results:
                if "tool_status" in item:
                    tool_statuses.append(item["tool_status"])
            all_results.extend(results)

    return {
        "query": query,
        "total_results": len(all_results),
        "results": all_results,
        "tool_statuses": tool_statuses,
    }
