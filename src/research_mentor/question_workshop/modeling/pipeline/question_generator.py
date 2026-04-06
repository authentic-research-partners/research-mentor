"""Stage 2: Question Generator — produce ONE modeling research question.

Takes the system exploration landscape and generates a single research
question investigable through computational modeling. Multiple candidates
are generated in parallel by the orchestrator (best-of-n pattern).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.modeling.pipeline.schemas import (
    MODELING_APPROACHES,
    ModelingLandscape,
    ModelingResearchQuestion,
)

QUESTION_GENERATION_PROMPT = """\
You are a research mentor helping a student discover an interesting research \
question that can be investigated through computational modeling.

System/phenomenon: {system_topic}
Student's preferred modeling approach: {approach}
Student level: {population}

Published models in this domain:
{published_models}

Key variables tracked:
{key_variables}

Important simplification trade-offs:
{simplification_tradeoffs}

Available validation data:
{available_data}

Domain context: {domain_summary}

{approach_context}

CRITICAL: Your question MUST be about "{system_topic}" specifically. \
Do NOT generate a question about a different system or domain. \
If the system is about ice sheets, the question must be about ice sheets. \
If the system is about protein folding, the question must be about proteins.

Generate ONE research question that:
1. Starts with "What would happen if..." or "How does X affect Y in the model?"
2. Is DIRECTLY about "{system_topic}" — mentions the specific system by name
3. Can be investigated by building and running a computational model
4. Involves an interesting simplification decision specific to this system
5. Would teach the student something that intuition alone cannot predict
6. Is appropriate for a {population} student

Provide:
- question: The research question itself
- approach: Which modeling approach fits best
- key_simplification: The most important simplification decision
- why_interesting: What the model might reveal that intuition cannot
- suggested_tools: 1-2 implementation tools appropriate for a {population} student"""


async def generate_question(
    system_topic: str,
    approach: str,
    population: str,
    landscape: ModelingLandscape,
    student_profile_text: str = "",
) -> ModelingResearchQuestion:
    """Generate ONE modeling research question from the landscape."""
    logger.info("Modeling Stage 2: Generating question for '{}'", system_topic[:80])

    # Format landscape data
    published = "\n".join(f"- {m}" for m in landscape.published_models)
    variables = "\n".join(f"- {v}" for v in landscape.key_variables)
    tradeoffs = "\n".join(f"- {t}" for t in landscape.simplification_tradeoffs)
    data = "\n".join(f"- {d}" for d in landscape.available_data) or "- None found"

    # Add approach-specific context with full grounding + constraints
    if approach != "any" and approach in MODELING_APPROACHES:
        info = MODELING_APPROACHES[approach]
        constraints = info.get("question_constraints", "")
        constraint_block = f"\n\nQUALITY CONSTRAINTS:\n{constraints}" if constraints else ""
        approach_context = (
            f"The student selected {info['label']}. "
            f"The question MUST use this approach.\n"
            f"Description: {info['description']}\n"
            f"Examples: {info['examples']}\n"
            f"Tools: {info['tools']}\n"
            f"Use ONLY the tools listed above — do not suggest tools "
            f"from other approaches."
            f"{constraint_block}"
        )
    else:
        approach_context = (
            "The student hasn't selected a specific approach. "
            "Choose the most natural approach for the question. "
            "Use tools appropriate for that specific approach."
        )

    question = await structured_call(
        ModelingResearchQuestion,
        [
            SystemMessage(content=QUESTION_GENERATION_PROMPT.format(
                system_topic=system_topic,
                approach=approach,
                population=population,
                published_models=published,
                key_variables=variables,
                simplification_tradeoffs=tradeoffs,
                available_data=data,
                domain_summary=landscape.domain_summary,
                approach_context=approach_context,
            )),
            HumanMessage(content="Generate a modeling research question."),
        ],
        thinking="medium",
        temperature=0.7,
    )

    logger.info("Stage 2 complete: '{}'", question.question[:80])

    return question
