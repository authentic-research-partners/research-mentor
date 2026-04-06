"""GROBID client for structured academic PDF extraction.

Communicates with a running GROBID service via its REST API to extract
structured content from academic PDFs. Returns markdown with proper heading
hierarchy, which is superior to Docling's flat extraction for papers.

Health checks are cached to avoid probing on every PDF extraction.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.etree.ElementTree import ParseError as _XMLParseError

import httpx
from defusedxml.common import DefusedXmlException as _DefusedXmlError
from defusedxml.ElementTree import fromstring as _safe_fromstring
from loguru import logger

# Cached health check state
_last_check_time: float = 0.0
_last_check_result: bool = False
_CACHE_TTL_SECONDS = 60.0

# GROBID API timeout for large academic PDFs
_REQUEST_TIMEOUT = 120.0
_HEALTH_TIMEOUT = 5.0

# Shared async client for connection reuse across concurrent requests.
_shared_client: httpx.AsyncClient | None = None

# TEI namespace
_TEI_NS = "http://www.tei-c.org/ns/1.0"


async def _get_client() -> httpx.AsyncClient:
    """Return a shared async client, creating it on first use."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT)
    return _shared_client


async def close_client() -> None:
    """Close the shared async client. Call on application shutdown."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
        _shared_client = None


async def is_grobid_available(url: str) -> bool:
    """Check if GROBID service is responding. Result is cached for 60 seconds."""
    global _last_check_time, _last_check_result

    now = time.monotonic()
    if now - _last_check_time < _CACHE_TTL_SECONDS:
        return _last_check_result

    try:
        client = await _get_client()
        resp = await client.get(f"{url}/api/isalive", timeout=_HEALTH_TIMEOUT)
        _last_check_result = resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        _last_check_result = False

    _last_check_time = now
    logger.debug("GROBID health check ({}): {}", url, _last_check_result)
    return _last_check_result


def invalidate_grobid_cache() -> None:
    """Force next ``is_grobid_available`` call to re-check."""
    global _last_check_time
    _last_check_time = 0.0


async def extract_pdf_via_grobid(path: Path, url: str) -> str | None:
    """Extract structured text from a PDF using GROBID's fulltext endpoint.

    Posts the PDF to ``/api/processFulltextDocument`` and converts the
    TEI XML response to markdown with proper heading hierarchy.

    Returns markdown text, or None if extraction fails.
    """
    try:
        client = await _get_client()
        with open(path, "rb") as f:
            resp = await client.post(
                f"{url}/api/processFulltextDocument",
                files={"input": (path.name, f, "application/pdf")},
                data={"consolidateHeader": "1"},
            )
        if resp.status_code != 200:
            logger.warning(
                "GROBID returned {} for {}", resp.status_code, path.name,
            )
            return None
        return tei_to_markdown(resp.text)
    except (httpx.ConnectError, httpx.TimeoutException, OSError) as e:
        logger.warning("GROBID request failed for {}: {}", path.name, e)
        return None


def tei_to_markdown(tei_xml: str) -> str:
    """Convert GROBID TEI XML to markdown with heading hierarchy.

    Extracts title, abstract, body sections with proper heading levels,
    and references.
    """
    try:
        root = _safe_fromstring(tei_xml)
    except (_XMLParseError, _DefusedXmlError):
        logger.warning("Failed to parse TEI XML from GROBID")
        return ""
    parts: list[str] = []

    # Title
    title_el = root.find(f".//{{{_TEI_NS}}}titleStmt/{{{_TEI_NS}}}title")
    if title_el is not None and title_el.text:
        parts.append(f"# {title_el.text.strip()}")

    # Abstract
    abstract_el = root.find(f".//{{{_TEI_NS}}}profileDesc/{{{_TEI_NS}}}abstract")
    if abstract_el is not None:
        abstract_text = _extract_text(abstract_el)
        if abstract_text:
            parts.append("## Abstract")
            parts.append(abstract_text)

    # Body sections
    body_el = root.find(f".//{{{_TEI_NS}}}body")
    if body_el is not None:
        _convert_body(body_el, parts, heading_level=2)

    # References
    refs_el = root.find(f".//{{{_TEI_NS}}}listBibl")
    if refs_el is not None:
        ref_items = refs_el.findall(f"{{{_TEI_NS}}}biblStruct")
        if ref_items:
            parts.append("## References")
            for i, ref in enumerate(ref_items, 1):
                ref_text = _format_reference(ref)
                if ref_text:
                    parts.append(f"{i}. {ref_text}")

    result = "\n\n".join(parts)
    return result if result.strip() else ""


def _convert_body(element: ET.Element, parts: list[str], heading_level: int) -> None:
    """Recursively convert body divs to markdown sections."""
    for child in element:
        tag = child.tag.removeprefix(f"{{{_TEI_NS}}}")

        if tag == "div":
            # Check for section heading
            head = child.find(f"{{{_TEI_NS}}}head")
            if head is not None:
                head_text = (head.text or "").strip()
                # Strip leading numbers like "1." or "1.1"
                n = head.get("n", "")
                if n and head_text:
                    heading = f"{'#' * heading_level} {n} {head_text}"
                elif head_text:
                    heading = f"{'#' * heading_level} {head_text}"
                else:
                    heading = None
                if heading:
                    parts.append(heading)
            # Recurse into nested divs
            _convert_body(child, parts, heading_level=min(heading_level + 1, 6))

        elif tag == "p":
            text = _extract_text(child)
            if text:
                parts.append(text)

        elif tag == "figure":
            # Include figure captions
            caption = child.find(f"{{{_TEI_NS}}}figDesc")
            if caption is not None:
                cap_text = _extract_text(caption)
                if cap_text:
                    parts.append(f"*Figure: {cap_text}*")


def _extract_text(element: ET.Element) -> str:
    """Extract all text content from an element, including nested elements."""
    return "".join(element.itertext()).strip()


def _format_reference(bib: ET.Element) -> str:
    """Format a biblStruct element as a simple reference string."""
    parts: list[str] = []

    # Authors
    authors: list[str] = []
    for author in bib.findall(f".//{{{_TEI_NS}}}author/{{{_TEI_NS}}}persName"):
        surname = author.find(f"{{{_TEI_NS}}}surname")
        forename = author.find(f"{{{_TEI_NS}}}forename")
        if surname is not None and surname.text:
            name = surname.text
            if forename is not None and forename.text:
                name = f"{forename.text} {name}"
            authors.append(name)
    if authors:
        parts.append(", ".join(authors))

    # Title
    title = bib.find(f".//{{{_TEI_NS}}}title")
    if title is not None and title.text:
        parts.append(f'"{title.text.strip()}"')

    # Year
    date = bib.find(f".//{{{_TEI_NS}}}date")
    if date is not None:
        year = date.get("when", "")
        if year:
            parts.append(f"({year[:4]})")

    return ". ".join(parts)
