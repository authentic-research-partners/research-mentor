"""Stage 2: Failure Analyzer — classify failure type and methodology issues.

Runs parallel structured_call for each case. Uses a 9-type failure taxonomy.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.retractions.pipeline.schemas import (
    AnalyzedCase,
    AnalyzedCases,
    FailureAnalysis,
    RetractionCase,
    RetractionCases,
)

# Cap concurrent LLM calls to prevent pathological memory growth when
# many retraction cases arrive.  Set high enough to let vLLM batch.
_LLM_SEMAPHORE = asyncio.Semaphore(20)

_FAILURE_ANALYSIS_SYSTEM = """\
You are analyzing a retracted scientific paper to classify what went wrong.

**Failure taxonomy** — classify as exactly ONE of:
- fabrication: data or results were invented
- falsification: data or results were manipulated
- honest_error: genuine mistake in methods, analysis, or interpretation
- statistical_manipulation: p-hacking, selective reporting, HARKing
- image_manipulation: altered or duplicated images/figures
- plagiarism: copied from others without attribution
- irreproducible: results could not be replicated by others
- methodology_flawed: fundamental design flaws (wrong controls, etc.)
- peer_review_failure: problems in the review process itself

Use the retraction reason, title, and any abstract provided. \
If multiple types apply, pick the PRIMARY cause.

**No-blame constraint:** NEVER name individual researchers. \
Focus on methodology and processes, not people.

Identify systemic factors from this catalog (pick 1-3 that apply):
- Publish-or-perish pressure
- Lack of pre-registration
- Inadequate peer review
- No independent replication
- Insufficient oversight / mentoring
- Perverse incentive structures
- Lack of data sharing requirements
- Inadequate statistical training

**Student level:** {population}
Calibrate ``what_went_wrong`` to this level — simpler language for \
younger students, more technical detail for advanced."""


async def _analyze_one(
    case: RetractionCase, population: str,
) -> AnalyzedCase:
    """Analyze a single retraction case."""
    async with _LLM_SEMAPHORE:
        system = _FAILURE_ANALYSIS_SYSTEM.format(population=population)
        case_info = (
            f"Title: {case.title}\n"
            f"Journal: {case.journal}\n"
            f"Year: {case.year or 'Unknown'}\n"
            f"Retraction reason: {case.retraction_reason_raw}\n"
            f"Original claim: {case.original_claim}\n"
            f"Field: {case.field}"
        )

        analysis = await structured_call(
            FailureAnalysis,
            [
                SystemMessage(content=system),
                HumanMessage(content=f"Analyze this retraction:\n\n{case_info}"),
            ],
            thinking="high",
            temperature=0.2,
        )

        return AnalyzedCase(case=case, analysis=analysis)


async def analyze_failures(
    cases: RetractionCases, population: str,
) -> AnalyzedCases:
    """Stage 2: Analyze failure type for each retraction case (parallel).

    Args:
        cases: Stage 1 output.
        population: Student level for calibrating explanations.

    Returns:
        AnalyzedCases with failure analysis attached to each case.
    """
    logger.info("Stage 2/5: Failure Analyzer — {} cases", len(cases.cases))

    tasks = [_analyze_one(case, population) for case in cases.cases]
    results = await asyncio.gather(*tasks)

    logger.info(
        "Failure analysis complete: types={}",
        [r.analysis.failure_type for r in results],
    )
    return AnalyzedCases(cases=list(results))
