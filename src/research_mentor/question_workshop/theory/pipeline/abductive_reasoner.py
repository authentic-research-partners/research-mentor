"""Stage 2: Abductive Reasoner — generate and compare candidate explanations.

Applies Peirce's abduction (inference to the best explanation) to
generate competing explanations for observed patterns, then performs
head-to-head comparison to select the strongest.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.theory.pipeline.schemas import (
    AbductionResult,
    CandidateExplanation,
    ExplanationComparison,
    PhenomenonResult,
)

CANDIDATE_GENERATION_PROMPT = """\
You are applying abductive reasoning (Peirce's inference to the best \
explanation) to generate a candidate explanation for surprising scientific \
observations.

Domain: {domain_summary}

Surprising findings from literature:
{surprising_findings}

Unexplained phenomena:
{unexplained_phenomena}

Cross-domain connections:
{cross_domain_connections}

Student's observations:
{observations_text}

{candidate_instruction}

Rules:
1. The explanation must address the FULL set of observations, not just a subset.
2. Clearly state what evidence supports it AND what it fails to explain.
3. source_domain: which field does this reasoning come from?
4. Be specific — "the system is complex" is not an explanation."""

COMPARISON_PROMPT = """\
Compare these candidate explanations HEAD-TO-HEAD. Do NOT evaluate each \
independently — compare them against each other.

Observations to explain:
{all_observations}

Candidate explanations:
{candidates_text}

Select the BEST explanation based on:
1. Explanatory scope — which one explains the MOST observations?
2. Parsimony — which one makes the fewest assumptions?
3. Testability — which one makes the most concrete, falsifiable predictions?

Provide:
- best_explanation: The winning explanation (1-2 sentences)
- runner_up: Second-best if it's close, null if clearly outclassed
- comparison_reasoning: WHY the best wins over the others (2-3 sentences)
- explanatory_scope: What the best explanation covers
- key_weakness: What the best explanation STILL cannot explain"""


async def generate_explanations(
    phenomenon: PhenomenonResult,
    observations_text: str,
    population: str,
    student_profile_text: str = "",
) -> AbductionResult:
    """Stage 2: Generate and compare candidate explanations.

    Generates 3 candidates in parallel (one structured_call each) to avoid
    vLLM constrained decoding issues with deeply nested schemas.

    Args:
        phenomenon: Output from Stage 1.
        observations_text: Original student observations.
        population: Student level for calibration.
        student_profile_text: Pre-formatted student profile context.

    Returns:
        AbductionResult with candidates and comparison.
    """
    logger.info("Theory Stage 2: Generating candidate explanations")

    # Format inputs
    none = "- None identified"
    surprising = "\n".join(
        f"- {s}" for s in phenomenon.surprising_findings
    ) or none
    unexplained = "\n".join(
        f"- {u}" for u in phenomenon.unexplained_phenomena
    ) or none
    cross_domain = (
        "\n".join(
            f"- {c}" for c in phenomenon.cross_domain_connections
        ) or none
    )

    # Step 1: Generate 3 candidate explanations in parallel
    # Each is a flat schema (no nested list-of-objects) — fast on vLLM.
    instructions = [
        (
            "Generate the ESTABLISHED explanation — the one grounded in "
            "the dominant theory or framework in this domain's published "
            "literature. Name the specific theory (e.g., classical "
            "nucleation theory, BCS theory, natural selection). This must "
            "be a real, citable scientific theory, not a generic statement."
        ),
        (
            "Generate a COMPETING explanation from an alternative or "
            "newer theory in this domain that challenges the established "
            "one. It should address observations the established theory "
            "struggles with. Name the specific theory or mechanism."
        ),
        (
            "Generate a NON-OBVIOUS explanation — cross-domain transfer, "
            "mechanism reversal, or novel synthesis from another field. "
            "This must come from a DIFFERENT scientific field than the "
            "primary domain."
        ),
    ]

    base_kwargs = {
        "domain_summary": phenomenon.domain_summary,
        "surprising_findings": surprising,
        "unexplained_phenomena": unexplained,
        "cross_domain_connections": cross_domain,
        "observations_text": observations_text[:500],
    }

    tasks = [
        structured_call(
            CandidateExplanation,
            [
                SystemMessage(content=CANDIDATE_GENERATION_PROMPT.format(
                    candidate_instruction=instr,
                    **base_kwargs,
                )),
                HumanMessage(content="Generate one candidate explanation."),
            ],
            thinking="high",
            temperature=0.7,
        )
        for instr in instructions
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates: list[CandidateExplanation] = [
        r for r in results if isinstance(r, CandidateExplanation)
    ]
    for r in results:
        if isinstance(r, BaseException):
            logger.warning("Stage 2: Candidate generation failed: {}", r)

    if not candidates:
        raise RuntimeError("All candidate explanation generations failed")

    logger.info("Stage 2: Generated {} candidate explanations", len(candidates))

    # Step 2: Head-to-head comparison
    all_observations = (
        surprising + "\n" + unexplained + "\n"
        + "\n".join(f"- {o}" for o in phenomenon.key_observations)
    )

    candidates_text = ""
    for i, c in enumerate(candidates, 1):
        supports = "; ".join(c.supporting_evidence)
        fails = "; ".join(c.fails_to_explain)
        candidates_text += (
            f"\nExplanation {i}: {c.explanation}\n"
            f"  Supports: {supports}\n"
            f"  Fails to explain: {fails}\n"
            f"  Source domain: {c.source_domain}\n"
        )

    comparison = await structured_call(
        ExplanationComparison,
        [
            SystemMessage(content=COMPARISON_PROMPT.format(
                all_observations=all_observations,
                candidates_text=candidates_text,
            )),
            HumanMessage(content="Compare these explanations head-to-head."),
        ],
        thinking="high",
        temperature=0.0,
    )

    logger.info(
        "Stage 2 complete: best='{}', runner_up='{}'",
        comparison.best_explanation[:60],
        (comparison.runner_up or "none")[:60],
    )

    return AbductionResult(
        candidates=candidates,
        comparison=comparison,
    )
