"""Tools for the Personal Research Mentor agent.

External service integrations:
- search_and_enrich: Multi-source discovery + enrichment (primary API for workshops)
- academic_search: arXiv, Semantic Scholar, PubMed, OpenAlex, Europe PMC (free, no API keys)
- europepmc_search: Europe PMC full-text search (biomedical + life science)
- core_api: CORE API (abstract enrichment, optional API key)
- unpaywall: DOI → OA URL lookup (no API key, requires polite_email)
- orcid: ORCID public profile enrichment for researcher search
- web_search: Brave Search API (optional, requires API key)

Pedagogical tools:
- concept_explainer: Prerequisite concept explanations with research-question guard
- learning_pathway: Long-term learning pathway recommendations

Status & tracking:
- status: ToolStatus model for standardized error/warning reporting
- Every external API call is recorded in the tool_usage DB table (fire-and-forget)
- Tool errors flow to UI via MentorState.tool_warnings → SSE stream
- GET /api/tools/status returns per-tool health, config, and usage stats

Full docs: docs/tools.md
"""

from research_mentor.tools.academic_search import (
    enrich_paper_with_arxiv,
    enrich_with_semantic_scholar_retry,
    search_arxiv,
    search_openalex,
    search_papers,
    search_pubmed,
    search_semantic_scholar,
)
from research_mentor.tools.concept_explainer import explain_concept
from research_mentor.tools.core_api import search_core_by_doi
from research_mentor.tools.europepmc_search import search_europepmc
from research_mentor.tools.learning_pathway import recommend_learning_pathway
from research_mentor.tools.orcid import enrich_researchers_with_orcid, lookup_orcid_profile
from research_mentor.tools.search_and_enrich import search_and_enrich
from research_mentor.tools.status import ToolStatus
from research_mentor.tools.unpaywall import lookup_unpaywall
from research_mentor.tools.web_search import search_web

__all__ = [
    "ToolStatus",
    "enrich_paper_with_arxiv",
    "enrich_researchers_with_orcid",
    "enrich_with_semantic_scholar_retry",
    "explain_concept",
    "lookup_orcid_profile",
    "lookup_unpaywall",
    "recommend_learning_pathway",
    "search_and_enrich",
    "search_arxiv",
    "search_core_by_doi",
    "search_europepmc",
    "search_openalex",
    "search_papers",
    "search_pubmed",
    "search_semantic_scholar",
    "search_web",
]
