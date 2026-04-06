"""Stage 3: Impact Assessor — citation analysis and reopened gap identification.

For each case:
1. Search OpenAlex for citing papers (parallel)
2. LLM-based impact analysis (parallel)
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.retractions.pipeline.schemas import (
    AnalyzedCase,
    AnalyzedCases,
    ImpactAnalysis,
    ImpactedCase,
    ImpactedCases,
)

_LLM_SEMAPHORE = asyncio.Semaphore(20)  # vLLM batches; cap only pathological fanout

_IMPACT_SYSTEM = """\
You are assessing the impact of a retracted paper on its field.

Given the paper metadata and citing papers found, produce:

1. **field_impact** — How this retraction affected the field (1-2 sentences). \
If citing papers are provided, use them to ground your assessment. \
If no citing papers found, assess based on journal prestige and claim importance.

2. **affected_conclusions** — What field conclusions were compromised by this \
retraction (1-3 items). Be specific about what was believed and is now uncertain.

3. **reopened_gap** — What scientific question is now unanswered again because \
the evidence was retracted? This is the KEY output — it becomes the basis for \
new research questions. Frame as a clear gap statement (1-2 sentences).

Use ONLY the information provided. Do NOT invent citation details."""


async def _search_citations(doi: str) -> list[dict[str, Any]]:
    """Search OpenAlex for papers citing the given DOI."""
    try:
        from research_mentor.tools.academic_search import search_openalex

        results = await search_openalex(f"cites:{doi}", max_results=15)
        return [r for r in results if "error" not in r]
    except Exception:
        logger.debug("Citation search failed for DOI {}", doi)
        return []


async def _assess_one(
    analyzed: AnalyzedCase,
    citing_papers: list[dict[str, Any]],
) -> ImpactedCase:
    """Assess impact for a single case."""
    async with _LLM_SEMAPHORE:
        # Format citing papers for the prompt
        if citing_papers:
            citations_text = "\n".join(
                f"- {p.get('title', 'Untitled')} ({p.get('year', '?')}), "
                f"cited {p.get('citation_count', '?')} times"
                for p in citing_papers[:10]
            )
            citations_section = (
                f"\n\nCiting papers found ({len(citing_papers)}):\n{citations_text}"
            )
        else:
            citations_section = "\n\nNo citing papers found in OpenAlex."

        case = analyzed.case
        case_info = (
            f"Title: {case.title}\n"
            f"Journal: {case.journal}\n"
            f"Year: {case.year or 'Unknown'}\n"
            f"Original claim: {case.original_claim}\n"
            f"Failure type: {analyzed.analysis.failure_type}\n"
            f"What went wrong: {analyzed.analysis.what_went_wrong}"
            f"{citations_section}"
        )

        impact = await structured_call(
            ImpactAnalysis,
            [
                SystemMessage(content=_IMPACT_SYSTEM),
                HumanMessage(content=f"Assess impact:\n\n{case_info}"),
            ],
            thinking="high",
            temperature=0.3,
        )

        return ImpactedCase(
            case=case,
            analysis=analyzed.analysis,
            impact=impact,
        )


async def assess_impact(analyzed_cases: AnalyzedCases) -> ImpactedCases:
    """Stage 3: Assess citation impact and identify reopened gaps (parallel).

    Args:
        analyzed_cases: Stage 2 output.

    Returns:
        ImpactedCases with impact analysis attached.
    """
    logger.info("Stage 3/5: Impact Assessor — {} cases", len(analyzed_cases.cases))

    # Step 1: Parallel citation search for all cases
    citation_tasks = [
        _search_citations(ac.case.doi) for ac in analyzed_cases.cases
    ]
    all_citations = await asyncio.gather(*citation_tasks)

    logger.info(
        "Citation search complete: {}",
        [len(c) for c in all_citations],
    )

    # Step 2: Parallel impact assessment
    assess_tasks = [
        _assess_one(ac, citations)
        for ac, citations in zip(analyzed_cases.cases, all_citations, strict=True)
    ]
    results = await asyncio.gather(*assess_tasks)

    logger.info("Impact assessment complete for {} cases", len(results))
    return ImpactedCases(cases=list(results))
