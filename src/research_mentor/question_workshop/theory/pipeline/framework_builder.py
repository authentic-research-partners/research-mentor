"""Stage 3: Framework Builder — construct a theoretical framework from the best explanation.

Selects a framework type (classification, taxonomy, conceptual_model, causal_model)
and defines categories, relationships, predictions, and historical analogies.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.theory.pipeline.schemas import (
    AbductionResult,
    FrameworkResult,
    PhenomenonResult,
    TheoreticalFramework,
)

FRAMEWORK_PROMPT = """\
You are constructing a theoretical framework to organize understanding of \
a scientific phenomenon. Build on the best explanation from abductive reasoning.

Best explanation:
{best_explanation}

Runner-up explanation (if any):
{runner_up}

Key weakness of best explanation:
{key_weakness}

Domain summary:
{domain_summary}

Key observations:
{key_observations}

Cross-domain connections:
{cross_domain_connections}

Student level: {population}

Construct a framework that organizes these findings. The framework MUST \
engage with established theories in this domain — not ignore them. If the \
best explanation modifies or extends an existing theory, the framework \
should show how it relates to that theory.

Choose the framework type that best fits the domain:

- causal_model: Cause-effect chains with mechanisms. USE THIS when the \
domain has known physical/chemical/biological mechanisms. Most natural \
science topics should use this type.
- classification: Distinct categories with clear boundaries. Use when \
phenomena fall into discrete types without causal ordering.
- taxonomy: Hierarchical organization. Use when there's a natural \
parent-child hierarchy.
- conceptual_model: Relationships between abstract concepts. Use for \
social science or interdisciplinary topics where causal mechanisms \
are debated.

Requirements:
1. The framework must make TESTABLE predictions (2-4 items). Each \
prediction must specify what you would observe if the framework is \
correct AND what you would observe if it is wrong.
2. Categories/components must be specific to this domain — not generic. \
Name the actual scientific concepts, variables, or entities.
3. Relationships must be directional and mechanistic — state WHAT causes \
WHAT and through what mechanism.
4. Include a historical_analogy ONLY if a genuine parallel exists. \
Set to null otherwise — do not force it."""


async def build_framework(
    abduction: AbductionResult,
    phenomenon: PhenomenonResult,
    population: str,
    student_profile_text: str = "",
) -> FrameworkResult:
    """Stage 3: Build a theoretical framework from the best explanation.

    Args:
        abduction: Output from Stage 2.
        phenomenon: Output from Stage 1.
        population: Student level for calibration.
        student_profile_text: Pre-formatted student profile context.

    Returns:
        FrameworkResult with framework and best explanation.
    """
    logger.info("Theory Stage 3: Building theoretical framework")

    comparison = abduction.comparison

    observations = "\n".join(f"- {o}" for o in phenomenon.key_observations) or "- None"
    cross_domain = "\n".join(f"- {c}" for c in phenomenon.cross_domain_connections) or "- None"

    framework = await structured_call(
        TheoreticalFramework,
        [
            SystemMessage(content=FRAMEWORK_PROMPT.format(
                best_explanation=comparison.best_explanation,
                runner_up=comparison.runner_up or "None",
                key_weakness=comparison.key_weakness,
                domain_summary=phenomenon.domain_summary,
                key_observations=observations,
                cross_domain_connections=cross_domain,
                population=population,
            )),
            HumanMessage(content="Construct a theoretical framework."),
        ],
        thinking="high",
        temperature=0.7,
    )

    logger.info(
        "Stage 3 complete: type={}, {} categories, {} relationships, {} predictions",
        framework.framework_type,
        len(framework.categories),
        len(framework.relationships),
        len(framework.predictions),
    )

    return FrameworkResult(
        framework=framework,
        best_explanation=comparison.best_explanation,
    )
