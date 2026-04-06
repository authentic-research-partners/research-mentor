"""Stage 3: Question Generator — synthesize gaps into candidate research questions.

Takes landscape context and gap analysis findings, produces 5-8 candidate
research questions linked to their source gaps and strategic modes.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.gaps.pipeline.schemas import (
    CandidateQuestions,
    GapAnalysisResult,
    LandscapeResult,
)

QUESTION_GENERATION_PROMPT = """\
You are a research question generator specializing in fundamental science.

Generate 5-8 candidate research questions based on the literature gaps and \
findings below. Each question must be linked to the specific gap that \
inspired it and the analytical mode that surfaced it.

## Field context
{domain_summary}

## Landscape patterns
Consensus: {consensus}
Debates: {debates}
Gaps: {gaps}
Frontiers: {frontiers}

## Gap analysis findings
{strategy_findings}

## Candidate questions from gap analysis
{candidate_questions}

## Student level
{population}

## MODE DIVERSITY REQUIREMENT (MANDATORY)
{num_modes} strategies were applied: {mode_list}.
You MUST generate AT LEAST ONE question from EACH strategy listed above. \
Set mode_origin to the strategy that surfaced the gap. \
If a strategy found findings but no suggested questions, create a question \
from its findings. After covering all modes, add more questions from any mode.

## Instructions
1. Generate 5-8 research questions. Each must:
   - Be specific and testable (not vague or too broad)
   - Address a real gap identified above (cite the gap)
   - Name the strategic mode that surfaced the gap in mode_origin
   - List paper titles that support the question
   - Explain in 1-2 sentences why it's a fundamental question
   - Be a SINGLE question (no compound questions joined by "and")

2. Depth calibration: adjust question complexity and language for {population} \
level — but don't simplify the science, just the framing. Use precise \
scientific terminology, not informal paraphrases.

3. Each question should be phrased as a clear interrogative sentence ending \
with "?"."""


async def generate_questions(
    landscape: LandscapeResult,
    gaps: GapAnalysisResult,
    population: str,
    config: Any | None = None,
) -> CandidateQuestions:
    """Generate candidate research questions from landscape and gap analysis."""
    logger.info("Gaps Pipeline Stage 3: Generating candidate questions")

    # Format strategy findings for prompt
    strategy_text = _format_strategy_findings(gaps)
    candidate_text = "\n".join(
        f"- {q}" for q in gaps.all_candidate_questions
    ) or "(none yet)"

    mode_names = [s.strategy_name for s in gaps.strategies_applied]
    prompt = QUESTION_GENERATION_PROMPT.format(
        domain_summary=landscape.domain_summary or "Not available",
        consensus="; ".join(landscape.consensus) or "None identified",
        debates="; ".join(landscape.debates) or "None identified",
        gaps="; ".join(landscape.gaps) or "None identified",
        frontiers="; ".join(landscape.frontiers) or "None identified",
        strategy_findings=strategy_text,
        candidate_questions=candidate_text,
        population=population,
        num_modes=len(mode_names),
        mode_list=", ".join(mode_names),
    )

    result = await structured_call(
        CandidateQuestions,
        [
            SystemMessage(content=prompt),
            HumanMessage(content="Generate candidate research questions."),
        ],
        thinking="high",
        temperature=0.7,
    )

    # Post-process: enrich literature_support from strategy papers (code, not LLM)
    mode_papers = {
        sf.strategy_name: sf.papers_referenced
        for sf in gaps.strategies_applied
    }
    for q in result.questions:
        strategy_papers = mode_papers.get(q.mode_origin, [])
        if strategy_papers and not q.literature_support:
            q.literature_support = strategy_papers[:3]

    logger.info(
        "Stage 3 complete: {} candidate questions generated",
        len(result.questions),
    )

    return result


def _format_strategy_findings(gaps: GapAnalysisResult) -> str:
    """Format gap analysis findings for the generation prompt."""
    sections = []
    for sf in gaps.strategies_applied:
        lines = [f"### {sf.strategy_name}"]
        if sf.findings:
            lines.append("Findings:")
            for f in sf.findings:
                lines.append(f"  - {f}")
        if sf.potential_questions:
            lines.append("Suggested questions:")
            for q in sf.potential_questions:
                lines.append(f"  - {q}")
        if sf.papers_referenced:
            lines.append(f"Papers: {', '.join(sf.papers_referenced[:5])}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections) if sections else "(no strategy findings)"
