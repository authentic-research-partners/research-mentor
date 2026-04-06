"""Stage 5: Question Generator — produce theory-grounded research questions.

Generates 5 candidate questions (one per call, in parallel) that test
predictions, explain failures, extend the framework, or refine boundaries.
Individual calls avoid vLLM constrained decoding issues with nested schemas.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.theory.pipeline.schemas import (
    ConsilienceResult,
    FrameworkResult,
    PhenomenonResult,
    TheoryQuestion,
    TheoryQuestions,
)

_LLM_SEMAPHORE = asyncio.Semaphore(20)

QUESTION_GENERATION_PROMPT = """\
You are generating ONE research question grounded in a theoretical framework \
built through abductive reasoning and consilience testing.

Framework:
- Type: {framework_type}
- Description: {framework_description}
- Predictions: {predictions}
- Historical analogy: {historical_analogy}

Best explanation: {best_explanation}

Consilience results:
{consilience_summary}

Domain summary: {domain_summary}

Surprising findings:
{surprising_findings}

Unexplained phenomena:
{unexplained_phenomena}

Papers from literature search:
{papers_text}

Student level: {population}

{question_instruction}

Requirements:
- The question must be SPECIFIC to this framework — not a generic science question
- The question must be TESTABLE: it must be possible to design an experiment, \
observation, or analysis that could answer it. Specify measurable variables. \
Avoid compound questions that ask multiple things at once.
- Calibrate complexity to {population} level
- literature_support: cite SPECIFIC FINDINGS from the numbered papers above \
(e.g., "Paper [3] found that Y increases with Z" or "Paper [7] showed \
mechanism A operates at temperature B"). Reference papers by their number. \
Do NOT cite the framework's own predictions — cite independent empirical \
results from the papers listed above. Do NOT invent papers or findings \
not present in the list.
- why_interesting: explain what we'd LEARN about the underlying mechanism, \
not just what we'd DO experimentally"""


# Each instruction targets a different question type — exactly one per type
# to maximize type diversity in the final output (max 3 selected).
_QUESTION_INSTRUCTIONS = [
    (
        "Generate a test_prediction question: one that would directly test "
        "the framework's MOST IMPORTANT prediction. The framework predicts "
        "X — how would you verify or falsify it? Specify the independent "
        "and dependent variables. Set question_type to 'test_prediction'."
    ),
    (
        "Generate an explain_failure question: one that investigates WHY "
        "the framework failed or only partially succeeded in consilience "
        "testing. The framework didn't fully work in some domain — what "
        "mechanism is missing? What's different about that domain? "
        "Set question_type to 'explain_failure'."
    ),
    (
        "Generate an extend question: one that would extend the framework "
        "to a NEW domain or phenomenon not yet tested. If the framework is "
        "right, what else should we expect to observe? Set question_type "
        "to 'extend'."
    ),
    (
        "Generate a refine question: one that would sharpen the framework's "
        "boundaries by identifying a specific condition or variable that "
        "determines where the framework applies vs. breaks down. The question "
        "MUST specify a measurable dependent variable and a concrete "
        "comparison (e.g., 'Does X hold when Y exceeds Z?'). "
        "Set question_type to 'refine'."
    ),
]


async def generate_questions(
    framework_result: FrameworkResult,
    consilience: ConsilienceResult,
    phenomenon: PhenomenonResult,
    population: str,
    student_profile_text: str = "",
) -> TheoryQuestions:
    """Stage 5: Generate theory-grounded research questions.

    Generates 4 questions in parallel (one per type: test_prediction,
    explain_failure, extend, refine) to maximize type diversity.

    Args:
        framework_result: Output from Stage 3.
        consilience: Output from Stage 4.
        phenomenon: Output from Stage 1.
        population: Student level for calibration.
        student_profile_text: Pre-formatted student profile context.

    Returns:
        TheoryQuestions with up to 5 candidate questions.
    """
    logger.info("Theory Stage 5: Generating research questions")

    framework = framework_result.framework
    predictions = "\n".join(f"- {p}" for p in framework.predictions)
    surprising = "\n".join(
        f"- {s}" for s in phenomenon.surprising_findings
    ) or "- None"
    unexplained = "\n".join(
        f"- {u}" for u in phenomenon.unexplained_phenomena
    ) or "- None"

    # Build consilience summary
    consilience_lines = []
    for t in consilience.domains_tested:
        consilience_lines.append(
            f"- {t.domain}: {t.fit} — {t.what_it_explains}",
        )
        if t.fit in ("weak", "contradicts"):
            consilience_lines.append(f"  FAILURE: {t.what_it_fails}")
    consilience_text = "\n".join(consilience_lines) or "- No domains tested"
    if consilience.revision_notes:
        consilience_text += f"\nRevision notes: {consilience.revision_notes}"

    # Format papers so the model can cite specific findings by number
    papers_text = _format_papers(phenomenon.papers) if phenomenon.papers else "(no papers)"

    base_kwargs = {
        "framework_type": framework.framework_type,
        "framework_description": framework.description,
        "predictions": predictions,
        "historical_analogy": framework.historical_analogy or "None",
        "best_explanation": framework_result.best_explanation,
        "consilience_summary": consilience_text,
        "domain_summary": phenomenon.domain_summary,
        "surprising_findings": surprising,
        "unexplained_phenomena": unexplained,
        "papers_text": papers_text,
        "population": population,
    }

    # Generate 4 questions in parallel — one flat schema per call
    async def _generate_one(instr: str) -> TheoryQuestion:
        async with _LLM_SEMAPHORE:
            return await structured_call(
                TheoryQuestion,
                [
                    SystemMessage(content=QUESTION_GENERATION_PROMPT.format(
                        question_instruction=instr,
                        **base_kwargs,
                    )),
                    HumanMessage(content="Generate one research question."),
                ],
                thinking="high",
                temperature=0.7,
            )

    tasks = [_generate_one(instr) for instr in _QUESTION_INSTRUCTIONS]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    questions: list[TheoryQuestion] = [
        r for r in results if isinstance(r, TheoryQuestion)
    ]
    for r in results:
        if isinstance(r, BaseException):
            logger.warning("Stage 5: Question generation failed: {}", r)

    if not questions:
        raise RuntimeError("All question generations failed")

    # Log type distribution
    type_counts: dict[str, int] = {}
    for q in questions:
        type_counts[q.question_type] = type_counts.get(q.question_type, 0) + 1

    logger.info(
        "Stage 5 complete: {} questions (types: {})",
        len(questions),
        ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())),
    )

    return TheoryQuestions(questions=questions)


def _format_papers(papers: list[dict]) -> str:  # type: ignore[type-arg]
    """Format papers for the question generator — numbered for citation."""
    from typing import Any
    lines = []
    for i, p in enumerate(papers[:15], 1):
        p_dict: dict[str, Any] = p
        title = p_dict.get("title", "Untitled")
        year = p_dict.get("year", "")
        abstract = p_dict.get("abstract", "No abstract")
        lines.append(f"[{i}] {title} ({year})\nAbstract: {abstract}\n")
    return "\n".join(lines)
