"""Claims orchestrator — 5-stage sequential pipeline.

Transforms a media claim into a student research project:
1. Claim Extractor — extract and validate the scientific claim
2. Evidence Searcher — search OpenAlex for peer-reviewed evidence
3. Gap Analyzer — expose journalist vs real research gap
4. Project Framer — frame an open-ended research direction
5. Question Generator — generate critical thinking questions

~60-90 seconds total.  No checkpointing (re-run on failure).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from loguru import logger

from research_mentor.db import crud
from research_mentor.question_workshop.claims.nodes.claim_extractor import (
    claim_extractor,
)
from research_mentor.question_workshop.claims.nodes.evidence_searcher import (
    evidence_searcher,
)
from research_mentor.question_workshop.claims.nodes.gap_analyzer import (
    gap_analyzer,
)
from research_mentor.question_workshop.claims.nodes.project_framer import (
    project_framer,
)
from research_mentor.question_workshop.claims.nodes.question_generator import (
    question_generator,
)
from research_mentor.question_workshop.question_selector import (
    ScoredCandidate,
    select_questions,
)


async def investigate_claim(
    claim_text: str | None = None,
    claim_url: str | None = None,
    article_text: str | None = None,
    field_filter: str | None = None,
    project_id: str | None = None,
    student_demographics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Investigate a claim through the full 5-stage pipeline.

    Must provide at least one of claim_text, claim_url, or article_text.

    Args:
        claim_text: Direct claim text to investigate.
        claim_url: URL to article containing the claim.
        article_text: Full article text to extract claims from.
        field_filter: Optional field filter (physics/chemistry/biology).
        project_id: Optional project to link the investigation to.
        student_demographics: Optional student profile for adaptation.

    Returns:
        Complete generated problem dict (stored in generated_problems table),
        or a rejection dict if the claim is not valid natural science.
    """
    if not claim_text and not claim_url and not article_text:
        raise ValueError("Must provide claim_text, claim_url, or article_text")

    from research_mentor.agent.prompts.shared import build_student_profile_context

    profile_text = build_student_profile_context(student_demographics or {})

    # Determine input text
    if claim_text:
        input_text = claim_text
    elif article_text:
        input_text = article_text
    else:
        input_text = f"[Article URL: {claim_url}]"

    logger.info(
        "Claims: starting investigation (input_length={}, url={})",
        len(input_text), claim_url,
    )

    # ===== Stage 1: Claim Extraction =====
    logger.info("Stage 1/5: Claim Extraction")
    result_1 = await claim_extractor(
        input_text=input_text,
        source_url=claim_url,
    )

    if not result_1.get("is_valid"):
        logger.warning("Claim rejected: {}", result_1.get("rejection_reason"))
        return {
            "success": False,
            "rejection_reason": result_1.get("rejection_reason"),
            "stage_reached": "claim_extraction",
        }

    claim_data = result_1["claim_data"]

    # ===== Stage 2: Evidence Search =====
    logger.info("Stage 2/5: Evidence Search")
    result_2 = await evidence_searcher(claim_data=claim_data)
    evidence_result = result_2["evidence_result"]
    tool_warnings = result_2.get("tool_warnings", [])

    # ===== Stage 3: Gap Analysis =====
    logger.info("Stage 3/5: Gap Analysis")
    result_3 = await gap_analyzer(
        claim_data=claim_data,
        evidence_result=evidence_result,
    )
    gap_analysis_data = result_3["gap_analysis"]

    # ===== Stage 4: Project Framing =====
    logger.info("Stage 4/5: Project Framing")
    result_4 = await project_framer(
        claim_data=claim_data,
        evidence_result=evidence_result,
        gap_analysis=gap_analysis_data,
        student_profile_text=profile_text,
    )
    research_direction = result_4["research_direction"]

    # ===== Stage 5: Question Generation =====
    logger.info("Stage 5/5: Question Generation")
    result_5 = await question_generator(
        claim_data=claim_data,
        evidence_result=evidence_result,
        gap_analysis=gap_analysis_data,
        research_direction=research_direction,
        student_profile_text=profile_text,
    )
    critical_questions = result_5["critical_questions"]

    # ===== Assemble & persist =====
    record = {
        "id": uuid.uuid4().hex,
        "project_id": project_id,
        "field": claim_data.get("field", "biology"),
        "problem_type": research_direction.get("project_type", "experimental"),
        "workshop_type": "claims",
        "domains": json.dumps([]),
        "user_suggestion": claim_text or article_text,
        "title": research_direction["title"],
        "description": research_direction["description"],
        "investigation": research_direction["investigation"],
        "core_concepts": json.dumps([]),
        "materials": json.dumps([]),
        "feasibility": "MEDIUM",
        "recommended": "YES",
        "safety_level": "SAFE",
        "complexity_score": None,
        "engagement_score": None,
        "overall_quality": None,
        "metadata": json.dumps({
            "claim_data": claim_data,
            "evidence_result": evidence_result,
            "gap_analysis": gap_analysis_data,
            "critical_questions": critical_questions,
            "underlying_phenomenon": research_direction.get("underlying_phenomenon"),
            "project_duration": research_direction.get("project_duration"),
            "source_url": claim_url,
        }),
    }

    stored = await crud.create_generated_problem(**record)

    logger.info(
        "Claims complete: title='{}' field={} evidence={}",
        stored["title"], stored["field"],
        evidence_result.get("level", "?"),
    )

    return {
        "success": True,
        "tool_warnings": tool_warnings,
        **stored,
    }


async def generate_best_of_n(
    claim_text: str | None = None,
    claim_url: str | None = None,
    article_text: str | None = None,
    field_filter: str | None = None,
    project_id: str | None = None,
    candidates: int = 3,
    student_demographics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate N investigations in parallel, select up to 3 diverse results.

    All candidates run concurrently and are stored in DB.
    Returns up to 3 quality-filtered, deduplicated investigation directions.
    """
    import asyncio

    logger.info(
        "Claims: generating best of {} investigations",
        candidates,
    )

    tasks = [
        investigate_claim(
            claim_text=claim_text,
            claim_url=claim_url,
            article_text=article_text,
            field_filter=field_filter,
            project_id=project_id,
            student_demographics=student_demographics,
        )
        for _ in range(candidates)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    good: list[dict[str, Any]] = [
        r for r in results
        if isinstance(r, dict) and r.get("success")
    ]

    if not good:
        for r in results:
            if isinstance(r, BaseException):
                raise r
        for r in results:
            if isinstance(r, dict):
                return [r]
        raise RuntimeError("All candidates failed")

    # Claims has no scoring — use a flat score of 0.7 for all
    # (quality filtering won't drop any, similarity dedup is the main value)
    scored_candidates = [
        ScoredCandidate(
            question_text=p.get("title", "") + " " + p.get("description", ""),
            overall_score=0.7,
            data=p,
        )
        for p in good
    ]
    selected = select_questions(
        scored_candidates,
        quality_threshold=0.5,
        similarity_threshold=0.85,
        max_results=3,
    )

    logger.info(
        "Claims best of {}: {} selected from {} candidates",
        candidates, len(selected), len(good),
    )

    return selected
