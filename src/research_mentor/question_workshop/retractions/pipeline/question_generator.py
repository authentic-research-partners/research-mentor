"""Stage 4: Question Generator — produce research questions from reopened gaps.

For each impacted case, generates 1-2 research questions that:
- Address the gap left by the retraction
- Avoid the methodological mistakes that caused the retraction
- Include specific methodology improvements
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.retractions.pipeline.schemas import (
    CaseQuestions,
    ImpactedCase,
    ImpactedCases,
    RetractionQuestion,
    RetractionQuestions,
)

_LLM_SEMAPHORE = asyncio.Semaphore(20)

_QUESTION_GEN_SYSTEM = """\
You are generating research questions from a gap re-opened by a retracted paper.

The retraction invalidated certain findings, leaving an unanswered question. \
Your job: produce 1-2 research questions that:

1. **Address the reopened gap** — directly investigate what the retracted paper \
claimed but could not reliably establish
2. **Avoid the original mistakes** — build in safeguards SPECIFIC to the \
failure type (see below)
3. **Include methodology improvements** — 2-4 improvements that MATCH the \
failure type. Generic improvements that don't address the actual failure score poorly.
4. **Are scientifically interesting** — explain why re-answering this matters

**CRITICAL: Match improvements to the failure type:**
- fabrication → raw data archiving with open access, independent data audit, \
provenance tracking, institutional oversight, mandatory data deposition
- falsification → independent replication by separate lab, raw data sharing, \
automated data integrity checks
- statistical_manipulation → pre-registration, registered report format, \
blind analysis, power analysis, no optional stopping
- image_manipulation → raw image archiving, standardized acquisition protocols, \
explicit limits on post-processing, independent image verification
- honest_error → orthogonal analytical methods, independent replication, \
enhanced controls, cross-validation with different techniques
- irreproducible → multi-site replication, detailed protocol sharing, \
standardized materials and reagents
- methodology_flawed → redesigned controls, confound elimination, \
validated measurement instruments

Do NOT suggest pre-registration as a fix for fabrication (it prevents \
selective reporting, not intentional fraud). Do NOT suggest blinding as \
a fix for fabrication (it prevents bias, not data invention).

**No-blame constraint:** NEVER name individual researchers. \
Frame everything around methodology and knowledge gaps.

**Student level:** {population}
Calibrate question complexity to this level. For middle_school, suggest \
questions doable with basic equipment. For high_school, standard lab \
equipment. For university/adult, specialized instruments are acceptable."""


async def _generate_for_case(
    impacted: ImpactedCase, population: str,
) -> list[RetractionQuestion]:
    """Generate questions from one impacted case."""
    async with _LLM_SEMAPHORE:
        return await _generate_for_case_inner(impacted, population)


async def _generate_for_case_inner(
    impacted: ImpactedCase, population: str,
) -> list[RetractionQuestion]:
    """Generate questions from one impacted case (under semaphore)."""
    case = impacted.case
    case_info = (
        f"Title: {case.title}\n"
        f"Field: {case.field}\n"
        f"Original claim: {case.original_claim}\n"
        f"Failure type: {impacted.analysis.failure_type}\n"
        f"What went wrong: {impacted.analysis.what_went_wrong}\n"
        f"Methodology issues: {', '.join(impacted.analysis.methodology_issues)}\n"
        f"Reopened gap: {impacted.impact.reopened_gap}\n"
        f"Field impact: {impacted.impact.field_impact}\n"
        f"Affected conclusions: {', '.join(impacted.impact.affected_conclusions)}"
    )

    system = _QUESTION_GEN_SYSTEM.format(population=population)

    result = await structured_call(
        CaseQuestions,
        [
            SystemMessage(content=system),
            HumanMessage(
                content=(
                    f"Generate research questions from this retraction case "
                    f"(DOI: {case.doi}):\n\n{case_info}"
                ),
            ),
        ],
        thinking="high",
        temperature=0.7,
    )

    return result.questions


async def generate_questions(
    impacted_cases: ImpactedCases, population: str,
) -> RetractionQuestions:
    """Stage 4: Generate research questions from reopened gaps (parallel).

    Args:
        impacted_cases: Stage 3 output.
        population: Student level for calibrating question complexity.

    Returns:
        RetractionQuestions with deduplicated questions across all cases.
    """
    logger.info("Stage 4/5: Question Generator — {} cases", len(impacted_cases.cases))

    tasks = [
        _generate_for_case(ic, population) for ic in impacted_cases.cases
    ]
    results = await asyncio.gather(*tasks)

    # Flatten and deduplicate by question text (case-insensitive)
    all_questions: list[RetractionQuestion] = []
    seen: set[str] = set()
    for case_questions in results:
        for q in case_questions:
            key = q.question.strip().lower()
            if key not in seen:
                seen.add(key)
                all_questions.append(q)

    logger.info(
        "Generated {} unique questions from {} cases",
        len(all_questions), len(impacted_cases.cases),
    )
    return RetractionQuestions(questions=all_questions)
