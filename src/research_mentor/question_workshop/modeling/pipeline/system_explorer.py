"""Stage 1: System Explorer — search published models, map the modeling landscape.

Finds what computational models exist for the student's system,
identifies key variables, simplification trade-offs, and validation data.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.modeling.pipeline.schemas import (
    MODELING_APPROACHES,
    ModelingLandscape,
)
from research_mentor.tools.search_and_enrich import search_and_enrich

EXPLORATION_PROMPT = """\
You are a computational modeling expert. Analyze these published papers \
about modeling {system_topic} and extract information about how THIS \
SPECIFIC system is modeled computationally.

CRITICAL: All extracted information must be about "{system_topic}". \
Do NOT substitute a different system or domain.

Student's preferred approach: {approach}
{approach_grounding}

Papers found:
{papers_text}

Extract:
1. published_models: What computational models are used for \
{system_topic} in these papers? Include the model type. 2-5 items.
2. key_variables: What are the key variables or quantities tracked \
in models of {system_topic}? 3-6 items.
3. simplification_tradeoffs: What simplification decisions do modelers \
of {system_topic} face? What to include vs. ignore? 2-4 items.
4. available_data: What public datasets or benchmarks exist for \
validating models of {system_topic}? 1-3 items. Empty list if none.
5. domain_summary: One sentence on how computational modeling is used \
to study {system_topic}."""


async def explore_system(
    system_topic: str,
    approach: str = "any",
    student_profile_text: str = "",
) -> tuple[ModelingLandscape, list[dict[str, Any]]]:
    """Search for published models and map the modeling landscape.

    Returns (landscape, papers) — the papers are passed to later stages
    for citation grounding.
    """
    logger.info("Modeling Stage 1: Exploring system '{}'", system_topic[:80])

    # Build search query focused on computational models
    search_query = f"{system_topic} computational model simulation"
    if approach and approach != "any":
        approach_terms = {
            "agent_based": "agent-based model",
            "numerical_simulation": "numerical simulation differential equation",
            "monte_carlo": "Monte Carlo simulation stochastic",
            "molecular_dynamics": "molecular dynamics simulation",
            "dft": "density functional theory DFT",
            "cfd": "computational fluid dynamics CFD",
            "fea": "finite element analysis",
            "ml_based": "machine learning physics-informed neural network",
            "system_dynamics": "system dynamics stock flow",
            "network": "network model graph dynamics",
            "cellular_automata": "cellular automaton simulation",
            "discrete_event": "discrete event simulation",
        }
        search_query = f"{system_topic} {approach_terms.get(approach, 'model')}"

    papers = await search_and_enrich(query=search_query, max_results=15)

    if not papers:
        logger.warning("Stage 1: No papers found for '{}'", system_topic)
        # Return a minimal landscape based on model knowledge alone
        return ModelingLandscape(
            published_models=["(no published models found in search)"],
            key_variables=["(to be determined from system analysis)"],
            simplification_tradeoffs=[
                "What to include vs. ignore in the model",
            ],
            available_data=[],
            domain_summary=(
                f"No published computational models found for "
                f"'{system_topic}' — the model will need to be "
                f"designed from first principles."
            ),
        ), []

    # Format papers for extraction
    papers_text = _format_papers(papers)

    # Add approach grounding if a specific approach was selected
    approach_grounding = ""
    if approach != "any" and approach in MODELING_APPROACHES:
        info = MODELING_APPROACHES[approach]
        approach_grounding = (
            f"\nApproach details ({info['label']}):\n"
            f"{info['description']}\n"
            f"Examples: {info['examples']}\n"
            f"Tools: {info['tools']}"
        )

    landscape = await structured_call(
        ModelingLandscape,
        [
            SystemMessage(content=EXPLORATION_PROMPT.format(
                system_topic=system_topic,
                approach=approach,
                approach_grounding=approach_grounding,
                papers_text=papers_text,
            )),
            HumanMessage(content="Analyze the modeling landscape."),
        ],
        thinking="medium",
        temperature=0.0,
    )

    logger.info(
        "Stage 1 complete: {} models, {} variables, {} tradeoffs",
        len(landscape.published_models),
        len(landscape.key_variables),
        len(landscape.simplification_tradeoffs),
    )

    return landscape, papers


def _format_papers(papers: list[dict[str, Any]]) -> str:
    """Format papers for LLM extraction."""
    lines = []
    for i, p in enumerate(papers[:15], 1):
        title = p.get("title", "Untitled")
        year = p.get("year", "")
        abstract = p.get("abstract", "No abstract")
        lines.append(f"[{i}] {title} ({year})\nAbstract: {abstract}\n")
    return "\n".join(lines)
