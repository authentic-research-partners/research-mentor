"""Project and student assessment generators.

Enhanced adaptation of the hosted multi-call assessment architecture.
Uses available local data (project info, milestones, artifacts, memories,
student profile, previous assessments, engagement metrics) for a 4-call
project assessment pipeline and cross-project student assessment.

Architecture (4 LLM calls for project assessment):
1. Trajectory analysis — trend from previous assessments (short-circuits if < 2)
2. Project health + engagement — health score, issues, stuck detection
3. Skills development — 8 per-domain skill scores
4. Narrative synthesis — human-readable assessment text
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from research_mentor.db import crud
from research_mentor.llm import get_chat_llm, set_usage_context, structured_call

# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------


class AssessmentIssue(BaseModel):
    type: str = Field(
        max_length=50,
        description="Category: project, methodology, timeline, scope",
    )
    severity: str = Field(max_length=50, description="low, medium, or high")
    description: str = Field(
        max_length=500, description="Concise description of the issue",
    )


class AssessmentAction(BaseModel):
    priority: str = Field(
        max_length=50, description="urgent, important, or monitor",
    )
    action: str = Field(
        max_length=500, description="Specific actionable recommendation",
    )


class TrajectoryAnalysis(BaseModel):
    trend: str = Field(
        max_length=50,
        description="improving, stable, declining, or insufficient_data",
    )
    trend_details: str = Field(
        max_length=800, description="Explanation of the trend",
    )
    recurring_issues: list[str] = Field(
        default_factory=list,
        description="Issues that appeared in multiple assessments",
    )
    intervention_effectiveness: str = Field(
        default="",
        max_length=800,
        description="How well past recommendations were followed",
    )


class ProjectHealthAnalysis(BaseModel):
    health_score: int = Field(
        description="Overall project health 0-100 (0=critical, 100=excellent)",
        ge=0, le=100,
    )
    status: str = Field(
        max_length=50,
        description="green (80-100), yellow (60-79), or red (0-59)",
    )
    issues: list[AssessmentIssue] = Field(
        default_factory=list,
        description="Identified issues (0-5)",
    )
    recommended_actions: list[AssessmentAction] = Field(
        default_factory=list,
        description="Recommended next steps (0-5)",
    )
    stuck_detected: bool = Field(
        default=False,
        description="Whether the student appears stuck or disengaged",
    )
    stuck_indicators: list[str] = Field(
        default_factory=list,
        description="Evidence of being stuck (e.g. no activity, same issues recurring)",
    )


class SkillDimensionScore(BaseModel):
    dimension: str = Field(
        max_length=150, description="Skill dimension name",
    )
    score: int = Field(description="0-100 skill score", ge=0, le=100)
    evidence: str = Field(
        max_length=500,
        description="Evidence from project data supporting this score",
    )


class SkillsDevelopment(BaseModel):
    skills: list[SkillDimensionScore] = Field(
        description="Scores for 8 skill dimensions",
    )
    overall_skill_trajectory: str = Field(
        max_length=50,
        description="Overall skill trajectory: improving, stable, declining",
    )


# Keep for backward compatibility (tests reference this)
StructuredAssessment = ProjectHealthAnalysis


# ---------------------------------------------------------------------------
# Data gathering helpers
# ---------------------------------------------------------------------------


async def _gather_project_context(project_id: str) -> dict[str, Any]:
    """Collect all available data for a project to feed into assessment."""
    project = await crud.get_project(project_id)
    if project is None:
        return {"error": "Project not found"}

    milestones_result = await crud.list_milestones(project_id, page_size=100)
    artifacts_result = await crud.list_artifacts(project_id, page_size=20)
    memories_result = await crud.get_memories_for_project(
        project_id, page_size=50,
    )
    assessments_result = await crud.list_assessments(project_id, page_size=20)
    student_profile = await crud.get_student_profile()
    engagement = await crud.compute_engagement_metrics(project_id)

    return {
        "project": project,
        "milestones": milestones_result["data"],
        "artifacts": artifacts_result["data"],
        "memories": memories_result["data"],
        "previous_assessments": assessments_result["data"],
        "student_profile": student_profile,
        "engagement": engagement,
    }


def _format_context_for_prompt(ctx: dict[str, Any]) -> str:
    """Format gathered context into a structured prompt section."""
    project = ctx["project"]
    parts: list[str] = []

    # Project info
    parts.append(f"**Project: {project['title']}**")
    parts.append(f"Research Question: {project['research_question']}")
    parts.append(f"Status: {project['status']}")

    # Student profile — use shared profile context builder for full demographics
    from research_mentor.agent.prompts.shared import build_student_profile_context

    profile = ctx.get("student_profile")
    if profile:
        demographics: dict[str, Any] = {
            "firstName": profile.get("name") or "",
            "age": profile.get("age"),
            "gradeLevel": profile.get("grade") or profile.get("college_year") or "",
            "country": profile.get("country") or "",
            "educationLevel": profile.get("education_level") or "",
            "domainExpertise": profile.get("domain_expertise", []),
            "professionalExperience": profile.get("professional_experience", []),
            "backgroundNotes": profile.get("background_notes") or "",
        }
        profile_context = build_student_profile_context(demographics)
        if profile_context:
            parts.append(f"\n{profile_context}")

    # Milestones
    milestones = ctx.get("milestones", [])
    if milestones:
        parts.append("\n**Milestones:**")
        overdue_count = 0
        for m in milestones:
            status = m.get("status", "pending")
            due = m.get("due_date", "no date")
            parts.append(f"- {m['title']} ({status}, due: {due})")
            if status == "pending" and due and due != "no date":
                try:
                    due_date = datetime.fromisoformat(due).date()
                    if due_date < date.today():
                        overdue_count += 1
                except ValueError:
                    pass
        if overdue_count:
            parts.append(f"  ({overdue_count} overdue)")
    else:
        parts.append("\n**Milestones:** None defined")

    # Artifacts
    artifacts = ctx.get("artifacts", [])
    if artifacts:
        parts.append(f"\n**Artifacts Uploaded:** {len(artifacts)} files")
        for a in artifacts[:5]:
            desc = a.get("description", "")
            parts.append(f"- {a['file_name']} ({a['artifact_type']})"
                         + (f": {desc}" if desc else ""))
    else:
        parts.append("\n**Artifacts:** None uploaded")

    # Engagement metrics
    engagement = ctx.get("engagement")
    if engagement:
        parts.append("\n**Engagement Metrics:**")
        parts.append(f"- Sessions: {engagement['session_count']}")
        parts.append(f"- Chat interactions: {engagement['chat_interaction_count']}")
        parts.append(f"- Artifacts uploaded: {engagement['artifact_count']}")
        parts.append(
            f"- Milestones: {engagement['milestone_completed']}/{engagement['milestone_total']}"
            f" completed, {engagement['overdue_count']} overdue"
        )
        parts.append(f"- Memories recorded: {engagement['memory_count']}")
        if engagement.get("days_since_last_activity") is not None:
            parts.append(
                f"- Days since last activity: {engagement['days_since_last_activity']}"
            )

    # Conversation memories (recent insights about the student)
    memories = ctx.get("memories", [])
    if memories:
        parts.append(f"\n**Conversation Insights ({len(memories)} memories):**")
        for mem in memories[:10]:
            parts.append(f"- [{mem['memory_type']}] {mem['memory_text']}")
    else:
        parts.append("\n**Conversation Insights:** No conversation history yet")

    # Previous assessments (trajectory)
    prev = ctx.get("previous_assessments", [])
    if prev:
        parts.append(f"\n**Previous Assessments ({len(prev)} total):**")
        for a in prev[:5]:
            score = a.get("health_score")
            score_str = f"{score}/100" if score is not None else "N/A"
            issues_count = len(a.get("issues", []))
            parts.append(
                f"- {a['assessment_date']}: Health={score_str}, {issues_count} issues"
            )
    else:
        parts.append("\n**Previous Assessments:** First assessment")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call prompts
# ---------------------------------------------------------------------------


TRAJECTORY_SYSTEM_PROMPT = """You are an AI research project assessor analyzing assessment trajectory.

Review the previous assessments for this project and identify:
1. The overall trend (improving, stable, declining, or insufficient_data if < 2 assessments)
2. Issues that keep recurring across assessments
3. Whether past recommended actions were effective

Be concise and evidence-based. If there are fewer than 2 previous assessments, set trend to "insufficient_data"."""


HEALTH_SYSTEM_PROMPT = """You are an AI research project assessor. Analyze the project data and produce a health assessment.

Consider:
1. Is the research question well-defined and realistic for the student's level?
2. Is the project making adequate progress? (Check milestone completion, artifacts, engagement)
3. Are there methodology or scope concerns?
4. Is the timeline viable? (Check for overdue milestones)
5. What are the most impactful next steps?
6. Is the student stuck or disengaged? (Check engagement metrics: days since last activity, session count, low interaction count)

Stuck indicators to watch for:
- No activity for 3+ days
- Very few chat interactions relative to project timeline
- Milestones consistently overdue
- No new artifacts uploaded

Scoring guide:
- 80-100 (green): Project on track, minor improvements possible
- 60-79 (yellow): Some concerns, guidance needed
- 0-59 (red): Significant issues, intervention recommended

If this is a new project with minimal data, score conservatively (50-60) and recommend establishing foundation."""


SKILLS_SYSTEM_PROMPT = """You are an AI research skills assessor. Analyze the project data and rate the student's skills across 8 dimensions.

Score each dimension 0-100 based on evidence from the project data:
1. critical_reading — Ability to find, read, and critique research literature
2. data_analysis_skill — Ability to collect, organize, and analyze data
3. experimental_design_skill — Ability to design valid experiments/studies
4. writing_skill — Quality of written communication and documentation
5. critical_thinking_skill — Ability to reason logically and evaluate evidence
6. time_management_skill — Project pacing, meeting deadlines, planning
7. collaboration_skill — Communication with mentors, seeking help appropriately
8. research_maturity — Overall research sophistication and independence

For each dimension, cite specific evidence. If evidence is insufficient, score conservatively (30-40).
Provide exactly 8 skill scores, one per dimension listed above."""


NARRATIVE_SYSTEM_PROMPT = """You are an encouraging research mentor writing a project assessment for a student.

Using the structured analysis results and project data, write a concise assessment (3-4 paragraphs):

1. **Strengths** — What's going well, cite specific evidence (milestones completed, artifacts uploaded, conversation insights)
2. **Areas for Growth** — Frame concerns as learning opportunities, be specific
3. **Recommended Next Steps** — Prioritized, actionable, concrete
4. **Encouragement** — Acknowledge effort, remind them research is iterative

Be supportive but honest. Reference actual data. Keep it under 500 words."""


# ---------------------------------------------------------------------------
# 4-call project assessment pipeline
# ---------------------------------------------------------------------------

# Dimension name mapping: schema field name → human-readable name
_SKILL_DIMENSIONS = {
    "critical_reading": "avg_critical_reading",
    "data_analysis_skill": "avg_data_analysis_skill",
    "experimental_design_skill": "avg_experimental_design_skill",
    "writing_skill": "avg_writing_skill",
    "critical_thinking_skill": "avg_critical_thinking_skill",
    "time_management_skill": "avg_time_management_skill",
    "collaboration_skill": "avg_collaboration_skill",
    "research_maturity": "avg_research_maturity",
}


async def generate_assessment(project_id: str) -> dict[str, Any]:
    """Generate a comprehensive project assessment via 4-call pipeline.

    Returns dict with keys matching project_assessments table columns:
    health_score, issues, recommended_actions, general_notes, metadata.
    """
    logger.info("Generating assessment for project {}", project_id)
    set_usage_context(purpose="assessment", project_id=project_id)

    # Gather all available context
    ctx = await _gather_project_context(project_id)
    if "error" in ctx:
        raise ValueError(f"Cannot generate assessment: {ctx['error']}")

    context_text = _format_context_for_prompt(ctx)

    # --- Call 1: Trajectory analysis ---
    prev_assessments = ctx.get("previous_assessments", [])
    if len(prev_assessments) >= 2:
        # Format previous assessments for trajectory analysis
        prev_text = "\n".join(
            f"- {a['assessment_date']}: score={a.get('health_score', 'N/A')}, "
            f"issues={len(a.get('issues', []))}"
            for a in prev_assessments[:10]
        )
        trajectory = await structured_call(
            TrajectoryAnalysis,
            [
                SystemMessage(content=TRAJECTORY_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Previous assessments (most recent first):\n{prev_text}"
                ),
            ],
            thinking="medium",
        )
        logger.info(
            "Trajectory analysis: trend={}, recurring_issues={}",
            trajectory.trend, len(trajectory.recurring_issues),
        )
    else:
        trajectory = TrajectoryAnalysis(
            trend="insufficient_data",
            trend_details="Fewer than 2 previous assessments available.",
            recurring_issues=[],
            intervention_effectiveness="Not enough data to evaluate.",
        )
        logger.info("Trajectory: short-circuit (< 2 previous assessments)")

    # --- Call 2: Project health + engagement ---
    trajectory_context = (
        f"\n\n**Trajectory Analysis:**\n"
        f"Trend: {trajectory.trend}\n"
        f"Details: {trajectory.trend_details}\n"
    )
    if trajectory.recurring_issues:
        trajectory_context += (
            f"Recurring issues: {', '.join(trajectory.recurring_issues)}\n"
        )

    health = await structured_call(
        ProjectHealthAnalysis,
        [
            SystemMessage(content=HEALTH_SYSTEM_PROMPT),
            HumanMessage(content=context_text + trajectory_context),
        ],
        thinking="medium",
    )
    logger.info(
        "Health analysis: score={}, status={}, issues={}, stuck={}",
        health.health_score, health.status,
        len(health.issues), health.stuck_detected,
    )

    # --- Call 3: Skills development ---
    skills_result = await structured_call(
        SkillsDevelopment,
        [
            SystemMessage(content=SKILLS_SYSTEM_PROMPT),
            HumanMessage(content=context_text),
        ],
        thinking="medium",
    )
    logger.info(
        "Skills assessment: {} dimensions, trajectory={}",
        len(skills_result.skills), skills_result.overall_skill_trajectory,
    )

    # Map skill scores to DB column names
    skill_scores: dict[str, int] = {}
    for s in skills_result.skills:
        col_name = _SKILL_DIMENSIONS.get(s.dimension)
        if col_name:
            skill_scores[col_name] = s.score

    # --- Call 4: Narrative synthesis ---
    synthesis_context = (
        f"{context_text}\n\n"
        f"---\n\n"
        f"**Trajectory Analysis:**\n"
        f"Trend: {trajectory.trend} — {trajectory.trend_details}\n\n"
        f"**Structured Health Analysis:**\n"
        f"Health Score: {health.health_score}/100 ({health.status})\n"
    )
    if health.stuck_detected:
        synthesis_context += (
            f"STUCK DETECTED: {', '.join(health.stuck_indicators)}\n"
        )
    if health.issues:
        synthesis_context += "Issues:\n"
        for issue in health.issues:
            synthesis_context += f"- [{issue.severity}] {issue.description}\n"
    if health.recommended_actions:
        synthesis_context += "Recommended Actions:\n"
        for action in health.recommended_actions:
            synthesis_context += f"- [{action.priority}] {action.action}\n"
    synthesis_context += (
        f"\n**Skills Assessment:**\n"
        f"Trajectory: {skills_result.overall_skill_trajectory}\n"
    )
    for s in skills_result.skills:
        synthesis_context += f"- {s.dimension}: {s.score}/100 ({s.evidence})\n"

    llm = get_chat_llm()
    narrative_response = await llm.ainvoke([
        SystemMessage(content=NARRATIVE_SYSTEM_PROMPT),
        HumanMessage(content=synthesis_context),
    ])
    narrative = narrative_response.content or ""

    logger.info("Assessment narrative generated ({} chars)", len(narrative))

    return {
        "health_score": health.health_score,
        "issues": [i.model_dump() for i in health.issues],
        "recommended_actions": [a.model_dump() for a in health.recommended_actions],
        "general_notes": narrative,
        "metadata": {
            "architecture": "local_v2",
            "llm_calls": 3 + (1 if len(prev_assessments) >= 2 else 0),
            "generated_at": datetime.now().isoformat(),
            "trajectory_trend": trajectory.trend,
            "stuck_detected": health.stuck_detected,
            "stuck_indicators": health.stuck_indicators,
            "skill_scores": skill_scores,
            "engagement": ctx.get("engagement"),
        },
    }


# ---------------------------------------------------------------------------
# Student assessment generator (cross-project)
# ---------------------------------------------------------------------------


class StudentSkillAssessment(BaseModel):
    """Structured output for cross-project student assessment."""

    overall_skill_level: str = Field(
        description="Overall level: beginner, developing, intermediate, advanced, expert"
    )
    research_maturity_score: int = Field(
        description="0-100 score for research maturity", ge=0, le=100
    )
    growth_trajectory: str = Field(
        description="Growth trend: accelerating, steady, plateauing, declining"
    )
    key_strengths: list[str] = Field(description="2-5 key strengths")
    growth_areas: list[str] = Field(description="2-5 areas for growth")
    cross_project_patterns: str = Field(
        description="Patterns observed across projects (2-3 sentences)"
    )
    recommendations: list[str] = Field(description="2-5 actionable recommendations")
    avg_critical_reading: int = Field(
        description="0-100 critical reading skill", ge=0, le=100
    )
    avg_data_analysis_skill: int = Field(
        description="0-100 data analysis skill", ge=0, le=100
    )
    avg_experimental_design_skill: int = Field(
        description="0-100 experimental design skill", ge=0, le=100
    )
    avg_writing_skill: int = Field(
        description="0-100 writing skill", ge=0, le=100
    )
    avg_critical_thinking_skill: int = Field(
        description="0-100 critical thinking skill", ge=0, le=100
    )
    avg_time_management_skill: int = Field(
        description="0-100 time management skill", ge=0, le=100
    )
    avg_collaboration_skill: int = Field(
        description="0-100 collaboration skill", ge=0, le=100
    )
    avg_research_maturity: int = Field(
        description="0-100 research maturity", ge=0, le=100
    )


STUDENT_ASSESSMENT_SYSTEM_PROMPT = """You are a research skills assessor. Analyze the student's work across all their research projects and produce a cross-project assessment of their research skills, maturity, and growth trajectory.

Score each of the 8 skill dimensions 0-100:
1. avg_critical_reading — Ability to find, read, and critique research literature
2. avg_data_analysis_skill — Data collection, organization, and analysis
3. avg_experimental_design_skill — Designing valid experiments/studies
4. avg_writing_skill — Written communication and documentation quality
5. avg_critical_thinking_skill — Logical reasoning and evidence evaluation
6. avg_time_management_skill — Project pacing, meeting deadlines
7. avg_collaboration_skill — Communication with mentors, seeking help
8. avg_research_maturity — Overall research sophistication and independence

Be specific and constructive. Base your assessment on evidence from the project data."""


async def generate_student_assessment() -> dict[str, Any]:
    """Generate cross-project student assessment with per-domain skill scores.

    Returns the created assessment dict from the database.
    """
    from research_mentor.config import get_llm_model_info

    logger.info("Generating cross-project student assessment")
    set_usage_context(purpose="assessment")

    # Gather data from all projects
    projects_result = await crud.list_projects(page=1, page_size=100)
    projects = projects_result["data"]

    if not projects:
        raise ValueError("No projects found — create a project first")

    # Build context from each project
    project_summaries: list[str] = []
    for p in projects:
        ctx = await crud.fetch_project_context(p["id"])
        if ctx:
            project_summaries.append(ctx)

    # Get student profile and build demographics context
    from research_mentor.agent.prompts.shared import build_student_profile_context

    profile = await crud.get_student_profile()
    demographics: dict[str, Any] = {}
    if profile:
        demographics = {
            "firstName": profile.get("name") or "",
            "age": profile.get("age"),
            "gradeLevel": profile.get("grade") or profile.get("college_year") or "",
            "country": profile.get("country") or "",
            "educationLevel": profile.get("education_level") or "",
            "domainExpertise": profile.get("domain_expertise", []),
            "professionalExperience": profile.get("professional_experience", []),
            "backgroundNotes": profile.get("background_notes") or "",
        }
    profile_text = build_student_profile_context(demographics)

    context = (
        f"{profile_text or 'Student: No profile available'}\n\n"
        f"Projects ({len(projects)}):\n\n"
        + "\n---\n".join(project_summaries)
    )

    result = await structured_call(
        StudentSkillAssessment,
        [
            SystemMessage(content=STUDENT_ASSESSMENT_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ],
        thinking="medium",
    )

    assessment = await crud.create_student_assessment(
        assessment_date=date.today().isoformat(),
        overall_skill_level=result.overall_skill_level,
        research_maturity_score=result.research_maturity_score,
        growth_trajectory=result.growth_trajectory,
        key_strengths=result.key_strengths,
        growth_areas=result.growth_areas,
        cross_project_patterns=result.cross_project_patterns,
        recommendations=result.recommendations,
        metadata=get_llm_model_info(),
        avg_critical_reading=result.avg_critical_reading,
        avg_data_analysis_skill=result.avg_data_analysis_skill,
        avg_experimental_design_skill=result.avg_experimental_design_skill,
        avg_writing_skill=result.avg_writing_skill,
        avg_critical_thinking_skill=result.avg_critical_thinking_skill,
        avg_time_management_skill=result.avg_time_management_skill,
        avg_collaboration_skill=result.avg_collaboration_skill,
        avg_research_maturity=result.avg_research_maturity,
    )

    logger.info(
        "Student assessment generated: score={}, level={}",
        result.research_maturity_score, result.overall_skill_level,
    )

    return assessment


# ---------------------------------------------------------------------------
# Auto-trigger (called from chat endpoints)
# ---------------------------------------------------------------------------


async def maybe_trigger_assessment(project_id: str) -> None:
    """Check if project needs a daily assessment and generate if due.

    Called from the chat endpoint. Non-blocking — errors are logged, not raised.
    Generates at most one assessment per project per day.
    """
    try:
        # Check last assessment date
        assessments = await crud.list_assessments(project_id, page_size=1)
        today = date.today().isoformat()

        if assessments["data"]:
            last_date = assessments["data"][0].get("assessment_date", "")
            if last_date == today:
                logger.debug(
                    "Project {} already assessed today, skipping", project_id,
                )
                return

        # Check project has some activity (milestones or artifacts or memories)
        milestones = await crud.list_milestones(project_id, page_size=1)
        artifacts = await crud.list_artifacts(project_id, page_size=1)
        memories = await crud.get_memories_for_project(project_id, page_size=1)

        has_activity = (
            milestones["total"] > 0
            or artifacts["total"] > 0
            or memories["total"] > 0
        )
        if not has_activity:
            logger.debug(
                "Project {} has no activity data, skipping auto-assessment",
                project_id,
            )
            return

        # Generate assessment
        logger.info("Auto-triggering assessment for project {}", project_id)
        result = await generate_assessment(project_id)

        from research_mentor.config import get_llm_model_info

        await crud.create_assessment(
            project_id=project_id,
            assessment_date=today,
            health_score=result["health_score"],
            issues=result["issues"],
            recommended_actions=result["recommended_actions"],
            general_notes=result["general_notes"],
            metadata={**result["metadata"], "auto_generated": True, **get_llm_model_info()},
        )
        logger.info(
            "Auto-assessment completed for project {}: health={}",
            project_id, result["health_score"],
        )

    except Exception:
        # Don't let assessment errors block chat
        logger.exception("Error during auto-assessment for project {}", project_id)


async def maybe_trigger_student_assessment() -> None:
    """Auto-trigger daily student assessment. Non-blocking, errors logged.

    Pattern identical to maybe_trigger_assessment():
    - Check if already assessed today
    - Check student has projects
    - Call generate_student_assessment()
    - Swallow exceptions (logged, not raised)
    """
    try:
        today = date.today().isoformat()

        # Check if already assessed today
        latest = await crud.get_latest_student_assessment()
        if latest and latest.get("assessment_date") == today:
            logger.debug("Student already assessed today, skipping")
            return

        # Check student has projects
        projects = await crud.list_projects(page_size=1)
        if projects["total"] == 0:
            logger.debug("No projects found, skipping student auto-assessment")
            return

        logger.info("Auto-triggering student assessment")
        await generate_student_assessment()
        logger.info("Auto student assessment completed")

    except Exception:
        # Don't let assessment errors block chat
        logger.exception("Error during auto student assessment")
