"""Stage 2: Evidence Searcher Node

Single responsibility: Search for peer-reviewed evidence and assess evidence level.

Given a claim, searches academic databases and determines:
- Evidence level (WELL_SUPPORTED, WEAKLY_SUPPORTED, NOT_SUPPORTED, CONTRADICTED)
- Relevant papers found
- What science actually says about this claim

Pattern: Tool search + direct structured_call (single-step classification)
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.claims._config import claims_config
from research_mentor.question_workshop.claims.schemas import (
    EvidenceResult,
    PaperSummary,
)
from research_mentor.question_workshop.claims.tools.evidence_search import (
    search_claim_evidence,
)

EVIDENCE_ANALYSIS_PROMPT = """You are a scientific evidence evaluator.

Your task: Assess the level of peer-reviewed evidence supporting a claim.

## Evidence Levels — Decision Tree

Follow this decision tree IN ORDER to classify:

### Step 1: Has this topic been directly studied in peer-reviewed research?
- YES → go to Step 2
- NO (zero relevant studies) → **NOT_SUPPORTED**

### Step 2: Do the studies' findings support or contradict the claim?
- Studies ACTIVELY TESTED THIS SPECIFIC CLAIM and found it FALSE → **CONTRADICTED**
  This requires: RCTs or systematic reviews that directly tested the claimed effect and found null or negative results. "Magnetic bracelets reduce pain" is CONTRADICTED because sham-controlled RCTs found no difference.
  NOT sufficient for CONTRADICTED: mechanism seeming implausible, the claim being vague or unfalsifiable, absence of supporting evidence, or related science making it unlikely. Those are NOT_SUPPORTED.
- Studies show MIXED or WEAK support → go to Step 3
- Studies CONSISTENTLY SUPPORT the claim → go to Step 4

### Step 3: How strong is the supporting evidence?
- Only 1-2 small/preliminary studies, not replicated, methodological concerns → **WEAKLY_SUPPORTED**

### Step 4: How robust is the supporting evidence?
- Multiple independent RCTs (3+), systematic reviews or meta-analyses, consistent results, reputable journals → **WELL_SUPPORTED**
- Fewer studies but generally supportive → **WEAKLY_SUPPORTED**

## Critical Distinctions

**CONTRADICTED vs NOT_SUPPORTED:**
- CONTRADICTED = scientists TESTED this and found it FALSE (active disproof)
- NOT_SUPPORTED = no one has properly studied this (absence of evidence)
- The mechanism seeming physiologically implausible (e.g., "the body regulates pH so alkaline water can't change systemic pH") is NOT sufficient for CONTRADICTED — it means the mechanism is implausible, but no RCT specifically tested and disproved the health outcome claim. That is NOT_SUPPORTED.
- CONTRADICTED requires DIRECT experimental disproof of the OUTCOME, not just theoretical implausibility of the mechanism.

**WELL_SUPPORTED vs WEAKLY_SUPPORTED:**
- WELL_SUPPORTED = scientific consensus backed by robust evidence (meta-analyses, many RCTs)
- WEAKLY_SUPPORTED = some evidence exists but insufficient for consensus

**Compound claims — evaluate BOTH parts:**
- If claim says "X does A and B", assess evidence for BOTH A and B separately
- If components have different evidence levels, the WEAKEST component determines the overall level
- Explain the distinction in your summary

**Use your knowledge:** The papers found via search may be incomplete. Use your scientific knowledge to assess the broader evidence base. If you know a topic is well-studied with strong consensus, say so even if the search returned few papers.
"""


async def evidence_searcher(
    claim_data: dict[str, Any],
) -> dict[str, Any]:
    """Stage 2: Search for evidence and assess evidence level.

    Args:
        claim_data: Output from claim_extractor with claim details.

    Returns:
        Dict with keys: evidence_result, papers_found.
    """
    independent_var = claim_data.get("independent_var", "")
    dependent_var = claim_data.get("dependent_var", "")
    original_text = claim_data.get("original_text", "")

    logger.info(
        "Stage 2: Evidence Search — X='{}', Y='{}'",
        independent_var, dependent_var,
    )

    # Step 1: Search for academic papers
    search_result = await search_claim_evidence(
        claim_text=original_text,
        independent_var=independent_var,
        dependent_var=dependent_var,
        max_papers=claims_config().max_papers,
    )

    papers = search_result.get("papers", [])
    queries_used = search_result.get("queries_used", [])
    tool_statuses = search_result.get("tool_statuses", [])

    logger.info("Found {} papers from {} queries", len(papers), len(queries_used))

    # Step 2: Classify evidence directly via structured_call
    papers_summary = _format_papers_for_analysis(papers)

    user_prompt = f"""Analyze the scientific evidence for this claim:

CLAIM: "{original_text}"
NORMALIZED: {independent_var} -> {dependent_var}
FIELD: {claim_data.get('field', 'unknown')}

SEARCH QUERIES USED:
{chr(10).join(f'- {q}' for q in queries_used)}

PAPERS FOUND ({len(papers)}):
{papers_summary if papers_summary else "No relevant peer-reviewed papers found."}

Follow the decision tree to determine the evidence level. Explain what science \
actually says about this claim.
"""

    evidence_result = await structured_call(
        EvidenceResult,
        [
            SystemMessage(content=EVIDENCE_ANALYSIS_PROMPT),
            HumanMessage(content=user_prompt),
        ],
        thinking="medium",
        temperature=claims_config().analysis_temperature,
    )

    # Add paper details from search
    evidence_result.papers_found = len(papers)
    evidence_result.search_queries_used = queries_used
    evidence_result.relevant_papers = [
        PaperSummary(
            title=p.get("title", "Untitled"),
            authors=p.get("authors", "Unknown"),
            year=p.get("year"),
            journal=p.get("journal"),
            relevance_summary=p.get("abstract") or "No abstract",
            doi=p.get("doi"),
            url=p.get("url"),
        )
        for p in papers[:5]
    ]

    tool_warnings = [
        ts.model_dump() for ts in tool_statuses if ts.level != "ok"
    ]
    result = {
        "evidence_result": evidence_result.model_dump(),
        "papers_found": len(papers),
        "tool_warnings": tool_warnings,
    }

    logger.info(
        "Stage 2 complete: level={}, papers={}",
        evidence_result.level, len(papers),
    )

    return result


def _format_papers_for_analysis(papers: list[dict[str, Any]]) -> str:
    """Format papers for LLM analysis."""
    if not papers:
        return ""

    lines = []
    for i, paper in enumerate(papers[:10], 1):
        title = paper.get("title", "Untitled")
        authors = paper.get("authors", "Unknown")
        year = paper.get("year", "N/A")
        journal = paper.get("journal", "Unknown journal")
        citations = paper.get("citation_count", 0)
        abstract = paper.get("abstract") or "No abstract"

        lines.append(
            f"\n{i}. {title}\n"
            f"   Authors: {authors}\n"
            f"   Year: {year} | Journal: {journal} | Citations: {citations}\n"
            f"   Abstract: {abstract}\n"
        )

    return "\n".join(lines)
