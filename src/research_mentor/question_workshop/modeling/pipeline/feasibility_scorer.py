"""Stage 3: Feasibility Scorer — assess a question's feasibility.

Scores computational feasibility, data availability, student accessibility,
and scientific value for the generated question.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.modeling.pipeline.schemas import (
    ModelingResearchQuestion,
    QuestionFeasibility,
)

FEASIBILITY_PROMPT = """\
Assess the feasibility of this modeling research question for a {population} \
student.

Question: {question}
Approach: {approach}
Tools: {suggested_tools}
Key simplification: {key_simplification}

Score each dimension (1-10):
1. computational_feasibility: Can this run on a laptop (10) or needs a \
supercomputer (1)?
2. data_availability: Is validation data publicly available (10) or \
nonexistent (1)?
3. student_accessibility: Can a {population} student implement this (10) \
or does it need graduate training (1)?
4. scientific_value: Will this produce novel insight (10) or reproduce \
well-known results (1)?
5. overall_score: Overall recommendation considering all factors.

Be honest — a middle schooler cannot do molecular dynamics, and a DFT \
calculation of a novel material has genuine scientific value even if the \
tool exists."""


async def score_feasibility(
    question: ModelingResearchQuestion,
    population: str,
) -> QuestionFeasibility:
    """Score feasibility for a single question."""
    logger.info("Modeling Stage 3: Scoring feasibility")

    result = await structured_call(
        QuestionFeasibility,
        [
            SystemMessage(content=FEASIBILITY_PROMPT.format(
                population=population,
                question=question.question,
                approach=question.approach,
                suggested_tools=question.suggested_tools,
                key_simplification=question.key_simplification,
            )),
            HumanMessage(content="Assess feasibility."),
        ],
        thinking="medium",
        temperature=0.0,
    )

    logger.info("Stage 3 complete: overall={}", result.overall_score)

    return result
