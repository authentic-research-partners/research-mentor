"""Stage 4: Consilience Tester — test the framework across domains.

Identifies 2-3 external domains where the framework might apply,
searches for evidence in parallel, and evaluates fit in each domain.
Key improvement over interactive: tests multiple domains simultaneously.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.theory.pipeline.schemas import (
    ConsilienceDomains,
    ConsilienceResult,
    ConsilienceTest,
    FrameworkResult,
    PhenomenonResult,
)
from research_mentor.tools.search_and_enrich import search_and_enrich

_LLM_SEMAPHORE = asyncio.Semaphore(20)

DOMAIN_IDENTIFICATION_PROMPT = """\
You are testing a theoretical framework's consilience — whether it explains \
phenomena across multiple domains (Whewell's concept).

Framework type: {framework_type}
Framework description: {framework_description}
Primary domain: {domain}
Framework predictions:
{predictions}

Identify 2-3 domains OUTSIDE the primary domain where this framework might \
apply. For each domain, provide a specific search query to find evidence.

Rules:
1. Each domain MUST be a different scientific discipline from the primary \
domain AND from each other. "Different discipline" means a different \
department at a university (e.g., chemistry → biology is OK, crystal \
nucleation → mineral crystallization is NOT — both are materials science).
2. The framework's MECHANISM must physically or causally apply in the test \
domain — not just as a metaphor. If the primary domain involves physical \
nucleation, the test domain must also involve a physical phase transition \
or aggregation process, not a social phenomenon called "nucleation" by analogy.
3. AVOID: (a) subfields of the primary domain, (b) domains where the \
framework is already standard textbook material, (c) domains connected \
only by metaphorical similarity (e.g., "idea spreading" for a disease \
transmission framework).
4. Search queries should include the key concept from the framework combined \
with the test domain's specific terminology."""

CONSILIENCE_TEST_PROMPT = """\
Test whether this theoretical framework explains phenomena in a different domain.

Framework:
- Type: {framework_type}
- Description: {framework_description}
- Key predictions: {predictions}
- Primary domain: {primary_domain}

Testing domain: {test_domain}

Papers found in {test_domain}:
{papers_text}

Evaluate:
1. evidence_found: Which papers or findings support or contradict the \
framework's applicability? (2-4 items)
2. fit: How well does the framework apply? 'strong' (explains core phenomena), \
'partial' (explains some), 'weak' (barely applies), 'contradicts' (evidence \
against).
3. what_it_explains: What the framework successfully explains here (1 sentence).
4. what_it_fails: What the framework cannot explain in this domain (1 sentence).

Be honest — partial fit is more useful than forced strong fit."""


async def test_consilience(
    framework_result: FrameworkResult,
    phenomenon: PhenomenonResult,
    domain: str,
    population: str,
    student_profile_text: str = "",
) -> ConsilienceResult:
    """Stage 4: Test framework consilience across domains.

    Args:
        framework_result: Output from Stage 3.
        phenomenon: Output from Stage 1.
        domain: Primary research domain.
        population: Student level for calibration.
        student_profile_text: Pre-formatted student profile context.

    Returns:
        ConsilienceResult with per-domain tests and overall strength.
    """
    logger.info("Theory Stage 4: Testing consilience")

    framework = framework_result.framework
    predictions = "\n".join(f"- {p}" for p in framework.predictions)

    # Step 1: Identify consilience domains
    domains_result = await structured_call(
        ConsilienceDomains,
        [
            SystemMessage(content=DOMAIN_IDENTIFICATION_PROMPT.format(
                framework_type=framework.framework_type,
                framework_description=framework.description,
                domain=domain,
                predictions=predictions,
            )),
            HumanMessage(content="Identify domains for consilience testing."),
        ],
        thinking="high",
        temperature=0.7,
    )

    domains = domains_result.domains[:3]
    queries = domains_result.search_queries[:3]

    logger.info("Stage 4: Testing {} domains: {}", len(domains), domains)

    # Step 2: Search all domains in parallel
    search_tasks = [
        search_and_enrich(query=q, max_results=8)
        for q in queries
    ]
    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Step 3: Test consilience in each domain (parallel)
    test_tasks = []
    for _i, (test_domain, papers_result) in enumerate(zip(domains, search_results, strict=False)):
        if isinstance(papers_result, BaseException):
            logger.warning("Stage 4: Search failed for '{}': {}", test_domain, papers_result)
            papers_result = []
        test_tasks.append(
            _test_single_domain(
                framework=framework,
                primary_domain=domain,
                test_domain=test_domain,
                papers=papers_result,
            )
        )

    test_results = await asyncio.gather(*test_tasks, return_exceptions=True)

    # Collect successful tests
    tests: list[ConsilienceTest] = []
    for r in test_results:
        if isinstance(r, ConsilienceTest):
            tests.append(r)
        elif isinstance(r, BaseException):
            logger.warning("Stage 4: Consilience test failed: {}", r)

    # Tally results
    successes = sum(1 for t in tests if t.fit == "strong")
    partial = sum(1 for t in tests if t.fit == "partial")
    failures = sum(1 for t in tests if t.fit in ("weak", "contradicts"))

    # Determine framework strength
    if successes >= 2:
        strength = "strong"
    elif successes >= 1 or partial >= 2:
        strength = "moderate"
    else:
        strength = "weak"

    # Generate revision notes for any non-strong results
    revision_notes = None
    imperfect = [t for t in tests if t.fit != "strong"]
    if imperfect:
        notes_parts = []
        for t in imperfect:
            notes_parts.append(f"{t.domain} ({t.fit}): {t.what_it_fails}")
        revision_notes = (
            f"Framework has gaps in {len(imperfect)} domain(s): "
            f"{'; '.join(notes_parts)}. "
            f"Consider specifying boundary conditions or adding "
            f"domain-specific qualifications."
        )

    logger.info(
        "Stage 4 complete: {} tested, {} strong, {} partial, {} weak/contradicts → {}",
        len(tests), successes, partial, failures, strength,
    )

    return ConsilienceResult(
        domains_tested=tests,
        successes=successes,
        partial=partial,
        failures=failures,
        framework_strength=strength,
        revision_notes=revision_notes,
    )


async def _test_single_domain(
    framework: Any,
    primary_domain: str,
    test_domain: str,
    papers: list[dict[str, Any]],
) -> ConsilienceTest:
    """Test consilience in a single domain."""
    predictions = "\n".join(f"- {p}" for p in framework.predictions)

    if not papers:
        return ConsilienceTest(
            domain=test_domain,
            evidence_found=["No papers found for this domain"],
            fit="weak",
            what_it_explains="Cannot assess — no evidence found.",
            what_it_fails="No evidence available to test framework applicability.",
        )

    papers_text = _format_papers(papers)

    async with _LLM_SEMAPHORE:
        result = await structured_call(
            ConsilienceTest,
            [
                SystemMessage(content=CONSILIENCE_TEST_PROMPT.format(
                    framework_type=framework.framework_type,
                    framework_description=framework.description,
                    predictions=predictions,
                    primary_domain=primary_domain,
                    test_domain=test_domain,
                    papers_text=papers_text,
                )),
                HumanMessage(content=f"Test framework consilience in {test_domain}."),
            ],
            thinking="high",
            temperature=0.0,
        )

        logger.debug("Consilience test for '{}': fit={}", test_domain, result.fit)
        return result


def _format_papers(papers: list[dict[str, Any]]) -> str:
    """Format papers for LLM extraction."""
    lines = []
    for i, p in enumerate(papers[:8], 1):
        title = p.get("title", "Untitled")
        year = p.get("year", "")
        abstract = p.get("abstract", "No abstract")
        lines.append(f"[{i}] {title} ({year})\nAbstract: {abstract}\n")
    return "\n".join(lines)
