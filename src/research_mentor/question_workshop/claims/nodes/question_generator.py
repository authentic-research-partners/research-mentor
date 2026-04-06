"""Stage 5: Question Generator Node

Single responsibility: Generate critical thinking questions for the student's journey.

Creates questions that develop:
- Media literacy (about the journalist's "research")
- Scientific thinking (about evidence and proof)
- Self-awareness (about the student's project limitations)

Pattern: Direct structured_call (eliminates truncation from chat→extract two-step)
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.claims._config import claims_config
from research_mentor.question_workshop.claims.schemas import (
    CriticalQuestions,
    EvidenceLevel,
)

QUESTION_GENERATION_PROMPT = """You are an expert at developing critical \
thinking through questions.

## Your Goal

Generate questions that help students think critically about:
1. The journalist's "research" process
2. What scientific evidence actually means
3. Their own project's capabilities and limitations

## Question Quality Standards

- Each question MUST be a complete sentence ending with a question mark
- Keep each question to 1-2 sentences max
- Questions must be SPECIFIC to the claim and project described — never generic
- Make questions thought-provoking, not leading
- Questions should be answerable through reflection
- Focus on transferable critical thinking skills

## Question Categories

### About the Journalist's "Research" (3 questions)
Questions that expose the gap between journalism and science. Target the specific
source type, the specific shortcuts taken, and the specific expertise that was
missing. Avoid generic media literacy questions.

### About Evidence and Proof (3 questions)
Questions about what scientific evidence means for THIS specific claim. Target
the specific evidence level, the specific type of studies needed, and the specific
methodological challenges. At least one question should address the specific
quantitative claims or mechanisms in the original claim.

### About Your Project (3 questions)
Questions about the student's own investigation and its relationship to the
original claim. Target what the project CAN tell them, what it CANNOT prove,
and how their months of work differs from the journalist's days of research.
"""


async def question_generator(
    claim_data: dict[str, Any],
    evidence_result: dict[str, Any],
    gap_analysis: dict[str, Any],
    research_direction: dict[str, Any],
    student_profile_text: str = "",
) -> dict[str, Any]:
    """Stage 5: Generate critical thinking questions.

    Args:
        claim_data: Output from claim_extractor.
        evidence_result: Output from evidence_searcher.
        gap_analysis: Output from gap_analyzer.
        research_direction: Output from project_framer.
        student_profile_text: Rendered student profile context.

    Returns:
        Dict with keys: critical_questions.
    """
    logger.info("Stage 5: Question Generation")

    original_text = claim_data.get("original_text", "")
    source_name = claim_data.get("source_name", "Unknown source")
    field = claim_data.get("field", "unknown")

    evidence_level = evidence_result.get("level", EvidenceLevel.NOT_SUPPORTED)
    papers_found = evidence_result.get("papers_found", 0)

    journalist_did = gap_analysis.get("journalist_did", [])
    time_spent = gap_analysis.get("time_spent", "a few days")

    project_title = research_direction.get("title", "")
    project_duration = research_direction.get("project_duration", "2-4 months")
    project_investigation = research_direction.get("investigation", "")

    profile_section = f"\n\n{student_profile_text}" if student_profile_text else ""
    user_prompt = f"""Generate critical thinking questions for a student \
investigating this claim:{profile_section}

ORIGINAL CLAIM: "{original_text}"
SOURCE: {source_name}
FIELD: {field}

EVIDENCE LEVEL: {evidence_level}
PEER-REVIEWED PAPERS FOUND: {papers_found}

WHAT THE JOURNALIST LIKELY DID:
{chr(10).join(f'- {j}' for j in journalist_did)}
TIME JOURNALIST SPENT: {time_spent}

STUDENT'S PROJECT:
- Title: {project_title}
- Duration: {project_duration}
- Investigation: {project_investigation}

Generate 3 questions per category, specific to this claim and project.
"""

    critical_questions = await structured_call(
        CriticalQuestions,
        [
            SystemMessage(content=QUESTION_GENERATION_PROMPT),
            HumanMessage(content=user_prompt),
        ],
        thinking="medium",
        temperature=claims_config().questions_temperature,
    )

    result = {
        "critical_questions": critical_questions.model_dump(),
    }

    logger.info(
        "Stage 5 complete: journalist_qs={}, evidence_qs={}, project_qs={}",
        len(critical_questions.about_journalist),
        len(critical_questions.about_evidence),
        len(critical_questions.about_project),
    )

    return result
