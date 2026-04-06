"""Stage 2: Gap Analyzer — apply strategic modes to find research gaps.

Selects 3-5 strategic modes based on field maturity and runs them in
parallel via asyncio.gather. Each mode independently analyzes the
literature corpus for gaps, contradictions, and opportunities.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.gaps.pipeline.schemas import (
    GapAnalysisResult,
    LandscapeResult,
    StrategyFindings,
)
from research_mentor.question_workshop.gaps.pipeline.strategy_prompts import (
    STRATEGY_MAP,
)
from research_mentor.question_workshop.gaps.schemas import StrategyOutput

# Which modes to weight by field maturity
_MATURITY_MODES: dict[str, list[str]] = {
    "emerging": [
        "explicit_mining",
        "integration_synthesis",
        "theoretical_probing",
    ],
    "established": [
        "contradiction_detection",
        "theoretical_probing",
        "explicit_mining",
        "mechanistic_dissection",
    ],
    "mature": [
        "mechanistic_dissection",
        "theoretical_probing",
        "contradiction_detection",
        "integration_synthesis",
    ],
}

# Fallback: run at least these 3 modes for any maturity level
_DEFAULT_MODES = [
    "explicit_mining",
    "contradiction_detection",
    "theoretical_probing",
]


def select_modes(field_maturity: str) -> list[str]:
    """Select strategic modes based on field maturity.

    Always returns at least 3 modes, up to 5 for mature fields.
    """
    modes = _MATURITY_MODES.get(field_maturity, _DEFAULT_MODES)
    # Ensure at least 3 modes
    if len(modes) < 3:
        for m in _DEFAULT_MODES:
            if m not in modes:
                modes.append(m)
            if len(modes) >= 3:
                break
    return modes


async def analyze_gaps(
    landscape: LandscapeResult,
    population: str,
    config: Any | None = None,
) -> GapAnalysisResult:
    """Run strategic analysis modes in parallel to find research gaps.

    Returns deduplicated findings and candidate questions across all modes.
    """
    modes = select_modes(landscape.field_maturity)
    logger.info(
        "Gaps Pipeline Stage 2: Running {} modes: {}",
        len(modes), ", ".join(modes),
    )

    papers_text = _format_papers_for_strategy(landscape.papers)

    # Build focus area from landscape context
    focus_area = _derive_focus_area(landscape)

    # Run all modes in parallel
    tasks = [
        _run_strategy(mode, focus_area, papers_text)
        for mode in modes
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect successful results
    strategies: list[StrategyFindings] = []
    for mode, result in zip(modes, results):
        if isinstance(result, BaseException):
            logger.warning("Strategy {} failed: {}", mode, result)
            continue
        strategies.append(result)

    # Deduplicate candidate questions across all modes
    seen_questions: set[str] = set()
    all_questions: list[str] = []
    for sf in strategies:
        for q in sf.potential_questions:
            normalized = q.strip().lower()
            if normalized not in seen_questions:
                seen_questions.add(normalized)
                all_questions.append(q)

    logger.info(
        "Stage 2 complete: {} strategies succeeded, {} unique candidate questions",
        len(strategies), len(all_questions),
    )

    return GapAnalysisResult(
        strategies_applied=strategies,
        all_candidate_questions=all_questions,
    )


async def _run_strategy(
    mode: str,
    focus_area: str,
    papers_text: str,
) -> StrategyFindings:
    """Run a single strategic analysis mode."""
    prompt_template = STRATEGY_MAP[mode]
    prompt = prompt_template.format(
        focus_area=focus_area,
        papers_text=papers_text,
    )

    output = await structured_call(
        StrategyOutput,
        [
            SystemMessage(content=prompt),
            HumanMessage(content="Analyze these papers using this strategy."),
        ],
        thinking="medium",
        temperature=0.3,
    )

    return StrategyFindings(
        strategy_name=mode,
        findings=output.findings,
        potential_questions=output.potential_questions,
        papers_referenced=output.papers_referenced,
    )


def _derive_focus_area(landscape: LandscapeResult) -> str:
    """Derive a focus area description from the landscape."""
    if landscape.domain_summary:
        return landscape.domain_summary
    # Fall back to combining key patterns
    parts: list[str] = []
    if landscape.debates:
        parts.append(f"Debates: {'; '.join(landscape.debates[:2])}")
    if landscape.gaps:
        parts.append(f"Gaps: {'; '.join(landscape.gaps[:2])}")
    return " | ".join(parts) if parts else "general field analysis"


def _format_papers_for_strategy(papers: list[dict[str, Any]]) -> str:
    """Format papers for strategy analysis."""
    lines = []
    for i, paper in enumerate(papers[:15], 1):
        title = paper.get("title", "Untitled")
        year = paper.get("year", "")
        abstract = paper.get("abstract", "No abstract available")
        lines.append(f"[{i}] {title} ({year})\nAbstract: {abstract}\n")
    return "\n".join(lines)
