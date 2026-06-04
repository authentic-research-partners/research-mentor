"""Fetch paper introductions from open-access PDFs for richer gap mining.

Downloads PDFs for top-cited papers, extracts the introduction section
via GROBID/pdfplumber, and caches results to disk. Sources are tried in
priority order — trusted, high-quality sources first; less reliable last.

Used by Gaps Phase 2 (deep dive) to give the mining strategy LLM richer
context about research gaps, motivation, and what's missing in the field.

Sources (in priority order):
1. arXiv          — direct PDF via arxiv_id
2. S2 openAccess  — URL pre-acquired from Semantic Scholar search
3. OpenAlex OA    — URL pre-acquired from OpenAlex search
4. PubMed Central — DOI → PMCID lookup, then PDF download
5. Europe PMC     — DOI search → PDF URL
6. Unpaywall      — URL pre-acquired from enrichment
7. bioRxiv/medRxiv— direct PDF for 10.1101/ DOIs
8. CORE           — URL pre-acquired from CORE search
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from research_mentor.config import get_data_dir, load_config
from research_mentor.text_extraction import extract_text
from research_mentor.tools.rate_limiter import MonotonicRateLimiter
from research_mentor.tools.status import record_tool_usage

# Max PDF download size: 100 MB.  Academic PDFs are typically 1–20 MB;
# anything larger is likely a malicious or corrupt response.
_MAX_PDF_BYTES = 100 * 1024 * 1024

# ---------------------------------------------------------------------------
# Rate limiters (lazy init)
# ---------------------------------------------------------------------------

_pdf_limiter: MonotonicRateLimiter | None = None
_pmc_limiter: MonotonicRateLimiter | None = None
_europepmc_limiter: MonotonicRateLimiter | None = None

_PMC_ID_CONVERTER = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
_EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"

_MIN_PDF_SIZE = 5000  # bytes — reject anything smaller


def _ensure_pdf_limiter() -> MonotonicRateLimiter:
    """Initialize PDF download rate limiter from arXiv config."""
    global _pdf_limiter
    if _pdf_limiter is None:
        cfg = load_config()
        rl = cfg.rate_limits.arxiv  # [max_rate, time_period]
        _pdf_limiter = MonotonicRateLimiter(rl[0], rl[1])
    return _pdf_limiter


def _ensure_pmc_limiter() -> MonotonicRateLimiter:
    global _pmc_limiter
    if _pmc_limiter is None:
        cfg = load_config()
        rl = cfg.rate_limits.pmc
        _pmc_limiter = MonotonicRateLimiter(rl[0], rl[1])
    return _pmc_limiter


def _ensure_europepmc_limiter() -> MonotonicRateLimiter:
    global _europepmc_limiter
    if _europepmc_limiter is None:
        cfg = load_config()
        rl = cfg.rate_limits.europepmc
        _europepmc_limiter = MonotonicRateLimiter(rl[0], rl[1])
    return _europepmc_limiter


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _get_cache_dir() -> Path:
    """Get or create the paper intro cache directory."""
    cache_dir = get_data_dir() / "cache" / "paper_intros"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_key_for_paper(paper: dict[str, Any]) -> str:
    """Generate a stable cache key for a paper.

    Uses arXiv ID if available (backward-compatible with existing cache),
    otherwise DOI hash, otherwise title hash.
    """
    arxiv_id = _extract_arxiv_id(paper)
    if arxiv_id:
        return arxiv_id.replace("/", "_")

    doi = paper.get("doi", "")
    if doi:
        clean = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        return "doi_" + hashlib.sha256(clean.encode()).hexdigest()[:16]

    title = paper.get("title", "")
    if title:
        return "title_" + hashlib.sha256(title.encode()).hexdigest()[:16]

    return ""


# ---------------------------------------------------------------------------
# PDF validation
# ---------------------------------------------------------------------------


def _is_valid_pdf(data: bytes) -> bool:
    """Check if bytes start with PDF magic header and meet minimum size."""
    return len(data) >= _MIN_PDF_SIZE and data[:5] == b"%PDF-"


# ---------------------------------------------------------------------------
# arXiv URL helpers
# ---------------------------------------------------------------------------


def build_pdf_url(paper: dict[str, Any]) -> str | None:
    """Build arXiv PDF URL from paper dict.

    Returns the URL if the paper has an arXiv source (pdf_url pointing to
    arxiv.org, or an arxiv_id field). Returns None otherwise.
    """
    pdf_url: str = paper.get("pdf_url", "")
    if pdf_url and "arxiv.org" in pdf_url:
        return pdf_url

    arxiv_id: str = paper.get("arxiv_id", "")
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return None


# ---------------------------------------------------------------------------
# Introduction extraction
# ---------------------------------------------------------------------------

# Heading patterns for introduction sections in GROBID/pdfplumber markdown output.
_INTRO_PATTERNS = [
    r"^#{1,3}\s+\d*\.?\s*Introduction",
    r"^#{1,3}\s+I\.\s+INTRODUCTION",
    r"^\d+\.\s+Introduction",
    r"^I\.\s+INTRODUCTION",
    r"^Introduction\s*$",
]

# Patterns for any next section heading (where intro ends).
_NEXT_SECTION_PATTERNS = [
    r"^#{1,3}\s+\d*\.?\s*[A-Z]",
    r"^#{1,3}\s+[IVX]+\.\s+[A-Z]",
    r"^\d+\.\s+[A-Z]",
    r"^[IVX]+\.\s+[A-Z]",
]


def extract_intro_section(full_text: str, max_words: int = 400) -> str:
    """Extract the introduction section from converted PDF text.

    Looks for common heading patterns (markdown, numbered, roman numeral)
    and extracts text until the next section heading. Falls back to the
    first ``max_words`` of the document if no introduction heading is found.
    """
    if not full_text or not full_text.strip():
        return ""

    lines = full_text.split("\n")

    # Find intro start
    intro_start: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        for pattern in _INTRO_PATTERNS:
            if re.match(pattern, stripped, re.IGNORECASE):
                intro_start = i + 1
                break
        if intro_start is not None:
            break

    if intro_start is None:
        # Fallback: first max_words of full text
        words = full_text.split()
        return " ".join(words[:max_words])

    # Find intro end (next section heading)
    intro_end = len(lines)
    for i in range(intro_start, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        for pattern in _NEXT_SECTION_PATTERNS:
            if re.match(pattern, stripped, re.IGNORECASE):
                intro_end = i
                break
        if intro_end != len(lines):
            break

    intro_text = "\n".join(lines[intro_start:intro_end]).strip()

    # Truncate to max_words
    words = intro_text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."

    return intro_text


def _extract_arxiv_id(paper: dict[str, Any]) -> str | None:
    """Extract arXiv ID from paper dict or PDF URL."""
    arxiv_id: str = paper.get("arxiv_id", "")
    if arxiv_id:
        return arxiv_id

    pdf_url: str = paper.get("pdf_url", "")
    match = re.search(r"arxiv\.org/pdf/([^/]+?)(?:\.pdf)?$", pdf_url)
    if match:
        return match.group(1)

    return None


# ---------------------------------------------------------------------------
# NCBI API key helper
# ---------------------------------------------------------------------------


def _read_ncbi_key() -> str | None:
    """Read NCBI API key from configured file path."""
    cfg = load_config()
    key_path = Path(cfg.services.ncbi.api_key_file).expanduser()
    if not key_path.exists():
        return None
    text = key_path.read_text().strip()
    return text if text else None


# ---------------------------------------------------------------------------
# Source download functions (each returns bytes | None)
# ---------------------------------------------------------------------------


def _toggle_www(url: str) -> str | None:
    """Return the www/non-www alternate of *url*, or None if not applicable."""
    if "://www." in url:
        return url.replace("://www.", "://", 1)
    if "://" in url:
        return url.replace("://", "://www.", 1)
    return None


# Errors that indicate DNS or connection failure (no HTTP redirect to follow).
_DNS_CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)


async def _download_url(url: str, *, timeout: int = 30) -> bytes | None:
    """Download a URL and return validated PDF bytes, or None.

    On DNS / connection failure, retries with the www/non-www alternate of the
    URL (e.g. ``theseedsofscience.pub`` → ``www.theseedsofscience.pub``).
    """
    result, dns_fail = await _try_download(url, timeout=timeout)
    if result is not None:
        return result

    if dns_fail:
        alt = _toggle_www(url)
        if alt is not None:
            logger.debug("Retrying with www-toggled URL: {}", alt)
            alt_result, _ = await _try_download(alt, timeout=timeout)
            return alt_result
    return None


async def _try_download(
    url: str, *, timeout: int = 30,
) -> tuple[bytes | None, bool]:
    """Single-attempt download with size cap.

    Returns ``(pdf_bytes_or_none, was_dns_connect_failure)``.
    Uses streaming with a byte-count guard to enforce the size cap without
    loading oversized responses into memory.  The Content-Length pre-check
    is skipped when Content-Encoding is set (e.g. gzip) because the header
    reports the compressed wire size, not the decompressed size — comparing
    it against the cap is unreliable and causes false rejections.
    """
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True,
        ) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                # Only trust Content-Length when there's no Content-Encoding,
                # otherwise the header reports compressed size.
                if not resp.headers.get("content-encoding"):
                    cl = resp.headers.get("content-length")
                    if cl and int(cl) > _MAX_PDF_BYTES:
                        logger.debug("PDF too large (Content-Length {}): {}", cl, url)
                        return None, False
                # Stream with byte-count guard
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > _MAX_PDF_BYTES:
                        logger.debug("PDF exceeded {} B: {}", _MAX_PDF_BYTES, url)
                        return None, False
                    chunks.append(chunk)
            content = b"".join(chunks)
            if _is_valid_pdf(content):
                return content, False
            logger.debug("Invalid PDF from {}: size={}", url, len(content))
            return None, False
    except _DNS_CONNECT_ERRORS as e:
        logger.debug("DNS/connect failure for {}: {}", url, e)
        return None, True
    except Exception as e:
        logger.debug("Download failed for {}: {}", url, e)
        return None, False


async def _download_arxiv(paper: dict[str, Any], timeout: int) -> bytes | None:
    """Download PDF from arXiv."""
    url = build_pdf_url(paper)
    if not url:
        return None
    limiter = _ensure_pdf_limiter()
    async with limiter:
        return await _download_url(url, timeout=timeout)


async def _download_s2_pdf(paper: dict[str, Any], timeout: int) -> bytes | None:
    """Download from Semantic Scholar openAccessPdf URL."""
    url = paper.get("s2_pdf_url")
    if not url:
        return None
    return await _download_url(url, timeout=timeout)


async def _download_oa_url(paper: dict[str, Any], timeout: int) -> bytes | None:
    """Download from OpenAlex OA URL (non-arXiv pdf_url)."""
    url = paper.get("pdf_url", "")
    if not url or "arxiv.org" in url:
        return None  # arXiv handled by _download_arxiv
    return await _download_url(url, timeout=timeout)


async def _download_pmc(paper: dict[str, Any], timeout: int) -> bytes | None:
    """Look up DOI → PMCID via NCBI, then download from PubMed Central."""
    import time as _time

    doi = paper.get("doi")
    if not doi:
        return None

    clean_doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")

    # Step 1: DOI → PMCID
    limiter = _ensure_pmc_limiter()
    params: dict[str, str] = {
        "ids": clean_doi, "format": "json", "tool": "research-mentor",
    }
    ncbi_key = _read_ncbi_key()
    if ncbi_key:
        params["api_key"] = ncbi_key

    pmcid: str | None = None
    t0 = _time.monotonic()
    success = False
    try:
        async with limiter:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(_PMC_ID_CONVERTER, params=params)
                if resp.status_code != 200:
                    return None
                success = True  # API call worked (even if no PMCID found)
                records = resp.json().get("records") or []
                for rec in records:
                    pmcid = rec.get("pmcid")
                    if pmcid:
                        break
    except Exception as e:
        logger.debug("PMC ID lookup failed for {}: {}", clean_doi, e)
        return None
    finally:
        asyncio.create_task(record_tool_usage(
            "pmc",
            query=clean_doi,
            success=success,
            error_category=None if success else "network",
            elapsed=_time.monotonic() - t0,
        ))

    if not pmcid:
        return None

    # Step 2: download PDF
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
    async with limiter:
        result = await _download_url(url, timeout=timeout)
    success = result is not None
    return result


async def _download_europepmc(paper: dict[str, Any], timeout: int) -> bytes | None:
    """Look up DOI on Europe PMC, then download PDF."""
    import time as _time

    doi = paper.get("doi")
    if not doi:
        return None

    clean_doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")

    limiter = _ensure_europepmc_limiter()
    pdf_url: str | None = None
    t0 = _time.monotonic()
    success = False
    try:
        async with limiter:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{_EUROPEPMC_BASE}/search",
                    params={
                        "query": f"DOI:{clean_doi}",
                        "format": "json",
                        "resultType": "core",
                    },
                )
                if resp.status_code != 200:
                    return None
                success = True  # API call worked
                results = (
                    (resp.json().get("resultList") or {}).get("result") or []
                )
                for result in results:
                    for url_info in (
                        result.get("fullTextUrlList", {}).get("fullTextUrl", [])
                    ):
                        if url_info.get("documentStyle") == "pdf":
                            pdf_url = url_info.get("url")
                            break
                    if not pdf_url:
                        epmc_pmcid = result.get("pmcid")
                        if epmc_pmcid:
                            pdf_url = (
                                "https://europepmc.org/backend/ptpmcrender.fcgi"
                                f"?accid={epmc_pmcid}&blobtype=pdf"
                            )
                    if pdf_url:
                        break
    except Exception as e:
        logger.debug("Europe PMC lookup failed for {}: {}", clean_doi, e)
        return None
    finally:
        asyncio.create_task(record_tool_usage(
            "europepmc",
            query=clean_doi,
            success=success,
            error_category=None if success else "network",
            elapsed=_time.monotonic() - t0,
        ))

    if not pdf_url:
        return None
    return await _download_url(pdf_url, timeout=timeout)


async def _download_unpaywall(paper: dict[str, Any], timeout: int) -> bytes | None:
    """Download from Unpaywall PDF URL (pre-acquired during enrichment)."""
    url = paper.get("unpaywall_url")
    if not url:
        return None
    return await _download_url(url, timeout=timeout)


async def _download_biorxiv(paper: dict[str, Any], timeout: int) -> bytes | None:
    """Download from bioRxiv/medRxiv for 10.1101/ DOIs."""
    doi = paper.get("doi")
    if not doi:
        return None

    clean_doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    if not clean_doi.startswith("10.1101/"):
        return None

    # Try bioRxiv first
    url = f"https://www.biorxiv.org/content/{clean_doi}v1.full.pdf"
    result = await _download_url(url, timeout=timeout)
    if result:
        return result

    # Try medRxiv
    url = f"https://www.medrxiv.org/content/{clean_doi}v1.full.pdf"
    return await _download_url(url, timeout=timeout)


async def _download_core(paper: dict[str, Any], timeout: int) -> bytes | None:
    """Download from CORE repository URL (pre-acquired during enrichment)."""
    url = paper.get("download_url")
    if not url:
        return None
    return await _download_url(url, timeout=timeout)


# ---------------------------------------------------------------------------
# Source chain — ordered list of (name, download_fn) pairs
# ---------------------------------------------------------------------------

_SOURCES: list[tuple[str, Any]] = [
    ("arXiv", _download_arxiv),
    ("S2 openAccessPdf", _download_s2_pdf),
    ("OpenAlex OA", _download_oa_url),
    ("PubMed Central", _download_pmc),
    ("Europe PMC", _download_europepmc),
    ("Unpaywall", _download_unpaywall),
    ("bioRxiv/medRxiv", _download_biorxiv),
    ("CORE", _download_core),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def has_any_pdf_source(paper: dict[str, Any]) -> bool:
    """Check if a paper has any field that could lead to a PDF download."""
    if paper.get("error"):
        return False
    return bool(
        paper.get("arxiv_id")
        or paper.get("doi")
        or (paper.get("pdf_url") and "arxiv.org" in paper.get("pdf_url", ""))
        or paper.get("s2_pdf_url")
        or paper.get("unpaywall_url")
        or paper.get("download_url")
        # Non-arXiv pdf_url (e.g. OpenAlex OA URL)
        or paper.get("pdf_url")
    )


async def fetch_paper_intro(
    paper: dict[str, Any],
    *,
    timeout: int = 30,
    max_words: int = 400,
) -> str | None:
    """Fetch and extract the introduction from a paper's PDF.

    Tries multiple open-access sources in priority order. Returns the
    intro text, or None if no source has a downloadable PDF or any
    step fails. Results are cached to disk.
    """
    cache_key = _cache_key_for_paper(paper)
    if not cache_key:
        return None

    # Check cache
    cache_dir = _get_cache_dir()
    cache_path = cache_dir / f"{cache_key}.txt"

    if cache_path.exists():
        cached = (await asyncio.to_thread(cache_path.read_text, "utf-8")).strip()
        if cached:
            logger.debug("Paper intro cache hit: {}", cache_key)
            words = cached.split()
            if len(words) > max_words:
                return " ".join(words[:max_words]) + "..."
            return cached

    # Try each source in priority order
    pdf_bytes: bytes | None = None
    source_name = ""
    for name, download_fn in _SOURCES:
        pdf_bytes = await download_fn(paper, timeout)
        if pdf_bytes:
            source_name = name
            break

    if not pdf_bytes:
        logger.debug("No PDF found for paper: {}", paper.get("title", "?")[:60])
        return None

    logger.debug("Downloaded PDF via {} for {}", source_name, cache_key)

    # Extract text via GROBID/pdfplumber (sync — run in thread)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)

        full_text = await extract_text(tmp_path)
    except Exception as e:
        logger.debug("Text extraction failed for {}: {}", cache_key, e)
        return None
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)

    if not full_text:
        logger.debug("No text extracted from PDF for {}", cache_key)
        return None

    # Extract intro section
    intro = extract_intro_section(full_text, max_words=max_words)
    if not intro:
        return None

    # Cache full intro (up to 2000 words) so config changes take effect at read time
    try:
        full_intro = extract_intro_section(full_text, max_words=2000)
        await asyncio.to_thread(cache_path.write_text, full_intro, "utf-8")
        logger.debug(
            "Cached paper intro for {} ({} words, source: {})",
            cache_key, len(full_intro.split()), source_name,
        )
    except Exception as e:
        logger.debug("Failed to cache intro for {}: {}", cache_key, e)

    return intro


async def enrich_papers_with_intros(
    papers: list[dict[str, Any]],
    *,
    max_papers: int = 2,
    max_words: int = 400,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Enrich top-cited papers with introduction text from open-access PDFs.

    Sorts papers by citation_count descending, fetches intros for the top
    N that have any PDF source (arXiv, DOI, OA URLs). Adds an ``intro_text``
    field to enriched paper dicts. Papers without sources or where fetching
    fails are left unchanged.
    """
    if not papers or max_papers <= 0:
        return papers

    # Find candidates with any PDF source, sorted by citation count
    candidates = [
        (i, p) for i, p in enumerate(papers)
        if has_any_pdf_source(p)
    ]
    candidates.sort(key=lambda x: x[1].get("citation_count", 0), reverse=True)
    candidates = candidates[:max_papers]

    if not candidates:
        logger.debug("No papers with PDF sources for intro enrichment")
        return papers

    logger.info(
        "Fetching intros for {} papers (top by citation count)",
        len(candidates),
    )

    # Fetch intros with concurrency limit
    sem = asyncio.Semaphore(3)

    async def _fetch(paper: dict[str, Any]) -> str | None:
        async with sem:
            return await fetch_paper_intro(
                paper, timeout=timeout, max_words=max_words,
            )

    tasks = [_fetch(p) for _, p in candidates]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    enriched_count = 0
    for (idx, _), result in zip(candidates, results, strict=False):
        if isinstance(result, BaseException):
            logger.debug("Intro fetch exception for paper {}: {}", idx, result)
            continue
        if result:
            papers[idx]["intro_text"] = result
            enriched_count += 1

    logger.info("Enriched {}/{} papers with introduction text", enriched_count, len(candidates))
    return papers
