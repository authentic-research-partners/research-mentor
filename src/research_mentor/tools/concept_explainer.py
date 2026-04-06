"""Concept Explainer Tool.

Provides pedagogically sound explanations for prerequisite concepts students
haven't learned yet.

CRITICAL BOUNDARIES:
- ONLY explains prerequisite knowledge (concepts blocking research)
- NEVER explains research answers (what student is supposed to discover)
- ALWAYS provides context-bound explanations (ties to student's project)
- ALWAYS includes verification questions (checks understanding)

Ported from progress_mentor with adaptations:
- Uses structured_call() instead of get_structured_llm()
- No silent fallbacks — let errors propagate
- No LangChain StructuredTool wrapper — exports async function directly
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field, field_validator


class PrerequisiteClassification(BaseModel):
    """Classification of whether a concept is a prerequisite or research question."""

    is_prerequisite: bool = Field(
        description="True if concept is prerequisite knowledge, False if it's a research question"
    )
    confidence: float = Field(
        description="Confidence in classification (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        max_length=500,
        description="Brief explanation of classification decision"
    )
    refusal_guidance: str | None = Field(
        default=None,
        max_length=500,
        description="Socratic question to redirect student if this is a research question",
    )


class ConceptExplanation(BaseModel):
    """Structured prerequisite concept explanation."""

    explanation: str = Field(
        description="Simple explanation (3-5 sentences MAX, ~75-100 words)",
        max_length=600,
    )
    why_matters: str = Field(
        description="Why this matters for student's project (1 sentence)",
        max_length=200,
    )
    analogy: str | None = Field(
        default=None,
        description="Optional concrete analogy at student's level",
        max_length=300,
    )
    verification_question: str = Field(
        max_length=300,
        description="Application-based question to check understanding"
    )
    application_prompt: str = Field(
        max_length=300,
        description="Specific next step using this concept"
    )

    @field_validator("explanation")
    @classmethod
    def explanation_not_essay(cls, v: str) -> str:
        """Prevent verbose explanations."""
        word_count = len(v.split())
        if word_count > 150:
            msg = (
                f"Explanation too long ({word_count} words) "
                "— keep to 3-5 sentences (~75-100 words)"
            )
            raise ValueError(msg)
        return v


PREREQUISITE_CLASSIFICATION_INSTRUCTION = """You classify whether a concept is \
PREREQUISITE KNOWLEDGE or a RESEARCH QUESTION.

**PREREQUISITE KNOWLEDGE** = Foundational concepts students need to HAVE before doing research:
- Physics principles: "What's Darcy's Law?", "How do forces work?"
- Chemistry basics: "What does pH mean?", "What's a chemical reaction?"
- Math techniques: "How do you calculate standard deviation?"
- Methodology: "What's a control variable?", "What's a p-value?"
- Technical terms: "What's turbulent flow?", "What's a buffer solution?"

**RESEARCH QUESTIONS** = What students are INVESTIGATING (what they should discover):
- Experimental observations: "Why does my filter work better at pH 7?"
- Optimization questions: "What launch angle gives maximum range?"
- Cause-effect relationships: "How does spin affect trajectory?"
- Project-specific discoveries: "What's the best material for my design?"

**KEY DISTINCTION:**
- Prerequisites are GENERAL concepts applicable to many projects
- Research questions are SPECIFIC to the student's experiment/project

**IF RESEARCH QUESTION:** Provide refusal_guidance with Socratic redirection."""


CONCEPT_EXPLANATION_INSTRUCTION = """You provide simple, context-bound explanations \
for PREREQUISITE concepts.

**EXPLANATION STRUCTURE:**

1. **Simple Definition** (2-3 sentences, student's level)
   - Build on concepts student already knows
   - Use concrete examples from their experience
   - NO jargon unless necessary, and define it if used

2. **Why It Matters** (1 sentence, tied to research_context)
   - Direct connection to their specific project

3. **Concrete Example/Analogy** (OPTIONAL, 1-2 sentences if helpful)
   - Age-appropriate
   - From their research domain when possible

4. **Verification Question** (1 sentence)
   - Application-based (not just recall)
   - Related to their project when possible

5. **Application Prompt** (1-2 sentences)
   - Specific next step using this concept
   - Tied to their research

**CRITICAL RULES:**
1. **BREVITY**: 3-5 sentences for explanation (75-100 words MAX)
2. **CONTEXT-BOUND**: Always tie to student's project
3. **AGE-APPROPRIATE**: Match student's level"""


async def classify_prerequisite(
    concept: str,
    research_context: str,
) -> PrerequisiteClassification:
    """Classify whether concept is prerequisite or research question."""
    from research_mentor.llm import structured_call

    result = await structured_call(
        PrerequisiteClassification,
        [
            SystemMessage(content=PREREQUISITE_CLASSIFICATION_INSTRUCTION),
            HumanMessage(
                content=f"Concept/Question: {concept}\n"
                f"Research Context: {research_context}\n\n"
                "Classify as prerequisite or research question."
            ),
        ],
        thinking="low",
    )
    logger.info(
        "Prerequisite classification: {} → {} (confidence: {})",
        concept, result.is_prerequisite, result.confidence,
    )
    return result


async def explain_concept(
    concept: str,
    research_context: str,
    student_demographics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate pedagogically sound prerequisite concept explanation.

    Two-stage process:
    1. Classifies whether concept is genuinely prerequisite (vs research question)
    2. If research question: Returns refusal with Socratic redirection
    3. If prerequisite: Generates simple, context-bound explanation

    Args:
        concept: Prerequisite concept to explain
        research_context: Student's research project description
        student_demographics: Optional student info (age, gradeLevel, country)

    Returns:
        Dictionary containing:
        - If prerequisite: explanation, why_matters, verification_question, etc.
        - If research question: is_research_question=True, refusal with Socratic guidance
    """
    from research_mentor.llm import structured_call

    logger.info("Concept explainer invoked: concept='{}'", concept)

    # Step 1: Classify prerequisite vs research question
    classification = await classify_prerequisite(concept, research_context)

    # Step 2: If research question, refuse and provide Socratic redirection
    if not classification.is_prerequisite:
        logger.info("Refusing to explain research question: {}", concept)
        return {
            "is_research_question": True,
            "refusal": classification.refusal_guidance or (
                "This is your research question to explore. What do you think "
                "might explain it? What experiments could help you investigate this?"
            ),
            "classification_reasoning": classification.reasoning,
            "confidence": classification.confidence,
        }

    # Step 3: Generate prerequisite explanation
    demo_summary = _summarize_demographics(student_demographics or {})

    explanation = await structured_call(
        ConceptExplanation,
        [
            SystemMessage(content=CONCEPT_EXPLANATION_INSTRUCTION),
            HumanMessage(
                content=f"Concept: {concept}\n"
                f"Research Context: {research_context}\n"
                f"Student: {demo_summary}\n\n"
                "Provide prerequisite explanation following the structure.\n"
                "Remember: 3-5 sentences MAX for explanation itself."
            ),
        ],
        thinking="high",
    )

    logger.info("Generated explanation for {}: {} chars", concept, len(explanation.explanation))

    return {
        "is_research_question": False,
        "concept": concept,
        "explanation": explanation.explanation,
        "why_matters": explanation.why_matters,
        "analogy": explanation.analogy,
        "verification_question": explanation.verification_question,
        "application_prompt": explanation.application_prompt,
        "classification_confidence": classification.confidence,
    }


def _summarize_demographics(demographics: dict[str, Any]) -> str:
    """Summarize student demographics for context.

    Delegates to the shared ``build_student_profile_context`` so that
    education level, domain expertise, and professional experience are
    included alongside the population-aware guidance instruction.
    """
    from research_mentor.agent.prompts.shared import build_student_profile_context

    result = build_student_profile_context(demographics)
    return result if result else "No profile available"
