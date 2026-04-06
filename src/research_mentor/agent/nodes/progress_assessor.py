"""Progress Assessor node — determines pedagogical guidance approach.

Analyzes student progress/skills to route to the right guide tier:
scaffolding (0-20%), guided_discovery (20-60%), pure_socratic (60-80%), reflection (80-100%).
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from research_mentor.agent.prompts.shared import build_student_profile_context
from research_mentor.agent.state import MentorState
from research_mentor.llm import structured_call
from research_mentor.utils.memory_profiler import memory_profile_node

MAX_DIAGNOSTIC_QUESTIONS = 3

# Manual override: user_requested_guidance_level → guidance_type
# "adaptive" means use LLM-based skill routing (no override)
USER_REQUESTED_LEVEL_TO_GUIDANCE_TYPE: dict[str, str | None] = {
    "adaptive": None,
    "beginner": "scaffolding",
    "intermediate": "guided_discovery",
    "advanced": "pure_socratic",
    "expert": "reflection",
}


class GuidanceTypeOutput(BaseModel):
    """Structured output for guidance type assessment."""

    guidance_type: Literal["scaffolding", "guided_discovery", "pure_socratic", "reflection"] = Field(
        description="Guidance approach to use"
    )
    confidence: float = Field(description="Confidence in this assessment (0.0-1.0)")
    struggle_type: Literal["productive", "unproductive", "not_struggling"] = Field(
        description="Type of struggle the student is experiencing"
    )
    needs_scaffolding_override: bool = Field(
        default=False,
        description="True if student needs temporary scaffolding despite higher progress level",
    )
    input_clarity: Literal["clear", "vague"] = Field(
        description="Whether the student's input is clear or vague"
    )
    question_domain: Literal[
        "literature_search", "experimental_design", "data_analysis",
        "scientific_writing", "critical_thinking", "project_management", "general",
    ] = Field(description="Domain of the student's question")
    reasoning: str = Field(max_length=500, description="Brief reasoning for the assessment")

    @field_validator("confidence", mode="before")
    @classmethod
    def convert_confidence_to_float(cls, v: object) -> object:
        """Convert integer confidence (e.g., 90) to float (0.90)."""
        if isinstance(v, int):
            return v / 100.0 if v > 1 else float(v)
        return v


ASSESSOR_INSTRUCTION = """You determine guidance type. Return ONE of: scaffolding, guided_discovery, pure_socratic, reflection.

**Guidance types:**
- scaffolding: Structure, resources, examples (low skill in domain)
- guided_discovery: Hints, guiding questions (developing skill in domain)
- pure_socratic: Challenging questions (high skill in domain)
- reflection: Metacognition, philosophy (high skill + philosophical question)

---

**STEP 1: Domain Detection (classify what aspect of research this is about)**

**Question domains:**
- literature_search: Finding papers, citations, search strategies, databases
- experimental_design: Methods, variables, controls, procedures, equipment
- data_analysis: Statistics, graphing, data interpretation, patterns
- scientific_writing: Research questions, hypotheses, paper structure, citations
- critical_thinking: Logic, reasoning, methodology critique, epistemology
- project_management: Timeline, milestones, planning, organization
- general: Multiple aspects or not clearly domain-specific

**Milestone context:** Project context may include milestones the student set for themselves. Use them as engagement signals only:
- No milestones after multiple sessions → student may need help learning to plan (project_management domain)
- All milestones stale (pending for weeks) → possible disengagement signal
- Active milestones → student is self-regulating, reference them naturally
Do NOT critique or review milestone quality — they are the student's own planning tool.

**CRITICAL:** Domain detection drives routing. A student might have high overall progress but low skill in a specific domain (e.g., 80% overall but struggling with statistics). Route based on domain-specific skill, NOT overall progress.

---

**STEP 2: Input Clarity (classify INTENT, not phrasing)**

**CLEAR** - The user's goal is understandable, even if general. The vast majority of inputs are CLEAR.
- Any question about the project: "How do I start?", "What should I do next?"
- Any statement of being stuck: "I'm stuck on my research question.", "I can't find papers."
- Polite requests: "Can you help me with data analysis?", "Could you explain what a control group is?"
- Direct commands: "Find me papers on water filtration."

**VAGUE** - The user's intent is impossible to understand. Use this classification sparingly.
- Unintelligible text: "asdfasdf", "????"
- Empty or near-empty messages: "...", "hi"
- Completely off-topic questions unrelated to the project or learning.

**Key distinction:** The goal is to understand if the user is making a good-faith effort to communicate a need related to their project. Do not penalize politeness ("Can you help...") or general statements of confusion ("I'm lost"). If you can understand what they are confused about, the input is CLEAR.

---

**STEP 3: Struggle Type (EVALUATE AFTER STEP 4 routing)**

**Productive** (ANY signal present → productive):
- Describes attempts: "I tested X, Y, Z"
- Explains results: "When I did X, this happened"
- Proposes next steps: "Should I try A or B?"
→ struggle_type: "productive", needs_scaffolding_override: False

**Unproductive** (ALL signals present → unproductive):
- Stuck without attempts described
- No evidence of trying anything
- Pure frustration: "Nothing works, I give up"
→ struggle_type: "unproductive", needs_scaffolding_override: True (ONLY if STEP 4 routing gave guided_discovery or pure_socratic)

**Not struggling** → struggle_type: "not_struggling", needs_scaffolding_override: False

**IMPORTANT - Override Logic:**
- FIRST apply STEP 4 routing based on skill threshold
- THEN check if unproductive struggle detected
- IF unproductive AND (STEP 4 result was guided_discovery OR pure_socratic):
  → Override to "scaffolding" temporarily
  → Set needs_scaffolding_override: True
- IF unproductive BUT STEP 4 result was already "scaffolding":
  → Keep "scaffolding" (no override needed, already at correct level)
  → Set needs_scaffolding_override: False (not overriding, already there)

---

**STEP 4: Skill-Based Routing (PRIMARY)**

**CRITICAL - Route on DOMAIN-SPECIFIC SKILL, not overall progress:**

**You will receive:**
- Student skills (16 dimensions): avg_critical_reading, avg_data_analysis_skill, avg_experimental_design_skill, etc.
- Overall progress: Average task completion across project

**Domain → Skill Mapping:**
- literature_search → avg_critical_reading
- experimental_design → avg_experimental_design_skill
- data_analysis → avg_data_analysis_skill
- scientific_writing → avg_writing_skill
- critical_thinking → avg_critical_thinking_skill
- project_management → avg_time_management_skill
- general → Use overall_progress (fallback when no specific domain)

**CRITICAL: If the mapped skill is not available (missing or empty), use overall_progress instead.**

**Routing Thresholds:**

**CRITICAL: Apply thresholds in STEP order. Do NOT skip to struggle detection.**

**STEP 4A: Check for Reflection (philosophical questions at 70%+):**

IF BOTH conditions TRUE:
  1. Skill >= 70% AND
  2. Philosophical, interpretive, or metacognitive question about the **meaning, significance, or implications** of the research.
     Examples: "What does this result mean for my hypothesis?", "Should I pursue this direction?", "Why is this methodology important?", "What are the broader implications of these findings?", "How has my thinking evolved?"

THEN → "reflection" (STOP - do not check STEP 4B)

**STEP 4B: Apply skill-based routing thresholds (PRIMARY ROUTING):**

**USE THIS TABLE:**

| Skill Range | Guidance Type     |
|-------------|-------------------|
| 0 to 19     | scaffolding       |
| 20 to 59    | guided_discovery  |
| 60 to 69    | pure_socratic     |
| 70 to 100   | pure_socratic*    |

*If skill 70 to 100 AND philosophical question → use "reflection"

**Examples:**
- Skill 15 → 0 to 19 → scaffolding
- Skill 20 → 20 to 59 → guided_discovery
- Skill 40 → 20 to 59 → guided_discovery
- Skill 60 → 60 to 69 → pure_socratic
- Skill 65 → 60 to 69 → pure_socratic
- Skill 70 → 70 to 100, not philosophical → pure_socratic
- Skill 85 → 70 to 100, philosophical → reflection

**Examples of PHILOSOPHICAL questions (route to reflection at 70%+):**
- "What does this finding mean for the field?"
- "Should I pursue this direction or pivot?"
- "How has my understanding evolved?"
- "What are the implications of this methodology?"

**Examples of NON-PHILOSOPHICAL questions (route to pure_socratic even at 80%+):**
- "How do I analyze this data?" (data_analysis question, 85% skill → pure_socratic)
- "What statistical test should I use?" (technical question, 75% skill → pure_socratic)
- "How do I design this control?" (experimental_design, 80% skill → pure_socratic)

---

**Examples (showing STEP 4 routing then STEP 3 struggle check order):**

1. Low skill (15 percent): Check STEP 4 rules. Skill is below 20, so use "scaffolding"
   `{"guidance_type": "scaffolding", "struggle_type": "not_struggling", "needs_scaffolding_override": false, ...}`

2. Mid skill (40 percent): Check STEP 4 rules. Skill is twenty or above but below sixty, so use "guided_discovery"
   `{"guidance_type": "guided_discovery", "struggle_type": "productive", "needs_scaffolding_override": false, ...}`

3. Mid skill (40 percent) with unproductive struggle: STEP 4 gives "guided_discovery", then STEP 3 detects unproductive, override to "scaffolding"
   `{"guidance_type": "scaffolding", "struggle_type": "unproductive", "needs_scaffolding_override": true, ...}`

4. High skill (65 percent): Check STEP 4 rules. Skill is sixty or above but below seventy, so use "pure_socratic"
   `{"guidance_type": "pure_socratic", "struggle_type": "not_struggling", "needs_scaffolding_override": false, ...}`

5. Very high skill (80 percent) with philosophical question: Check STEP 4A first. Skill is seventy or above AND philosophical, so use "reflection"
   `{"guidance_type": "reflection", "struggle_type": "not_struggling", "needs_scaffolding_override": false, ...}`

6. Very high skill (85 percent) WITHOUT philosophical question: Check STEP 4A (not philosophical). Check STEP 4B. Skill is seventy or above, so use "pure_socratic"
   `{"guidance_type": "pure_socratic", "struggle_type": "not_struggling", "needs_scaffolding_override": false, ...}`

**Boundary Examples (specific numbers):**
- skill is exactly 20: twenty or above, below sixty means "guided_discovery"
- skill is exactly 60: sixty or above, below seventy means "pure_socratic"
- skill is exactly 70 with philosophical question: seventy or above AND philosophical means "reflection"
- skill is exactly 70 WITHOUT philosophical question: seventy or above, not philosophical means "pure_socratic"

---

Return structured output with ALL fields: guidance_type, confidence, reasoning, struggle_type, needs_scaffolding_override, input_clarity, question_domain.
"""


def _generate_diagnostic_instruction(
    question_domain: str,
    struggle_type: str,
    student_message: str,
    project_context: str,
) -> str:
    """Generate a context-aware diagnostic instruction for the diagnostics node.

    This instruction tells the diagnostics node what kind of question to ask
    based on the student's domain, struggle type, and actual question.
    """
    domain_guidance = {
        "literature_search": "Focus on understanding their search strategy and what databases/sources they've tried so far.",
        "experimental_design": "Focus on understanding what they currently understand about variables, controls, methodology, and what equipment/materials they have access to.",
        "data_analysis": "Focus on understanding their current data, what they're trying to find, and what they've attempted.",
        "scientific_writing": "Focus on understanding their current understanding of research questions or hypothesis structure.",
        "critical_thinking": "Focus on understanding their reasoning process and how they're approaching the problem.",
        "project_management": "Focus on understanding their current timeline understanding and planning approach.",
        "general": "Focus on understanding their overall approach and what aspect confuses them most.",
    }

    struggle_guidance = {
        "productive": "The student is already engaged and trying. Ask what they've learned so far from their attempts.",
        "unproductive": "The student is stuck without clear attempts. Ask them to articulate their first instinct or understanding.",
        "not_struggling": "The student is progressing normally. Ask about their current thinking on the next step.",
    }

    domain_msg = domain_guidance.get(question_domain, domain_guidance["general"])
    struggle_msg = struggle_guidance.get(struggle_type, "Ask to understand their starting point.")

    return f"""
Generate a SINGLE, focused diagnostic question for this student.

**Domain:** {question_domain}
**Student's Struggle Type:** {struggle_type}
**Their Question:** {student_message}
**Project Context:** {project_context}

**Your Instruction:**
{domain_msg}
{struggle_msg}

The diagnostic node will use this instruction to ask a natural, conversational probing question.
Remember: ONE question only, keep it focused and encouraging. The goal is to extract their
initial thinking before providing scaffolding help.
"""


@memory_profile_node("progress_assessor")
async def assess_guidance_type(state: MentorState) -> dict[str, Any]:
    """Determine the guidance approach based on student progress and skills."""
    logger.info("Progress assessor: Evaluating guidance type...")

    messages = state.get("messages", [])
    student_progress = state.get("student_progress", {})
    overall_progress = student_progress.get("overall", 0)
    student_skills = state.get("student_skills", {})
    demographics = state.get("student_demographics", {})

    student_profile = build_student_profile_context(demographics)

    # Check if user has manually selected guidance level (skip LLM routing)
    user_requested = state.get("user_requested_guidance_level", "adaptive")
    forced_type = USER_REQUESTED_LEVEL_TO_GUIDANCE_TYPE.get(user_requested)
    guidance_level_mismatch_warning = None

    if forced_type is not None:
        logger.info(
            "User requested guidance level '{}' → forcing guidance_type='{}'",
            user_requested, forced_type,
        )

        # Check for mismatch with actual skills
        valid_skills = [v for v in student_skills.values() if v is not None]
        if valid_skills:
            actual_avg_skill = sum(valid_skills) / len(valid_skills)
            guidance_type_to_skill_range = {
                "scaffolding": (0, 20),
                "guided_discovery": (20, 60),
                "pure_socratic": (60, 100),
            }
            expected_min, expected_max = guidance_type_to_skill_range.get(
                forced_type, (0, 100)
            )
            if actual_avg_skill < expected_min - 20 or actual_avg_skill > expected_max + 20:
                logger.warning(
                    "Guidance level mismatch: user_requested='{}' but actual skills ~{:.0f}%",
                    user_requested, actual_avg_skill,
                )
                guidance_level_mismatch_warning = (
                    "Based on your progress so far, the system recommends using 'Adaptive' "
                    "guidance for better learning results. You can change this in your "
                    "project settings."
                )

        return {
            "guidance_type": forced_type,
            "confidence": 1.0,
            "struggle_type": "not_struggling",
            "needs_scaffolding_override": False,
            "input_clarity": "clear",
            "question_domain": "general",
            "diagnostic_instruction": "",
            "student_profile_text": student_profile,
            "student_has_provided_initial_thoughts": True,
            "guidance_level_mismatch_warning": guidance_level_mismatch_warning,
        }

    # Get latest message and conversation history
    latest_message = str(messages[-1].content) if messages else ""

    conversation_history = ""
    if len(messages) >= 2:
        recent_messages = messages[-6:] if len(messages) >= 6 else messages
        conversation_history = "\n".join([
            f"{'Student' if i % 2 == 0 else 'Mentor'}: {msg.content[:200]}..."
            for i, msg in enumerate(recent_messages)
        ])

    # Format student skills for display
    skills_display = (
        "\n".join([f"  - {skill}: {value}%" for skill, value in student_skills.items()])
        if student_skills
        else "  No skill assessments available"
    )

    # Build evaluation context
    evaluation_context = f"""
**Student State Analysis:**

Overall Progress: {overall_progress}% (task completion across project)
Project Context: {state.get('project_context', 'N/A')}

**Student Skill Assessments (16 dimensions):**
{skills_display}

**CRITICAL:** Route based on domain-specific skill (from above), NOT overall progress.
Example: If question is about data analysis, use avg_data_analysis_skill for routing threshold.
If no skills available, fall back to overall_progress.

**Recent Conversation History:**
{conversation_history if conversation_history else "First message in conversation"}

**Latest Student Message (ANALYZE FOR DOMAIN, STRUGGLE TYPE, CLARITY):**
"{latest_message}"

**Your Task:**
1. FIRST: Detect question domain (literature_search, experimental_design, data_analysis, etc.)
2. SECOND: Map domain to skill dimension (see STEP 4 in instructions)
3. THIRD: Analyze struggle type from the latest message
   - Look for: attempts described, learning articulated, next steps proposed
   - Classify as: productive, unproductive, or not_struggling
4. FOURTH: Determine guidance type using domain-specific skill threshold
5. Return structured GuidanceTypeOutput with ALL fields (guidance_type, confidence, reasoning, struggle_type, needs_scaffolding_override, input_clarity, question_domain)
"""

    assessment = await structured_call(
        GuidanceTypeOutput,
        [
            SystemMessage(content=ASSESSOR_INSTRUCTION),
            HumanMessage(content=evaluation_context),
        ],
        thinking="medium",
    )

    logger.info(
        "Assessment: type={}, confidence={:.2f}, domain={}, struggle={}, override={}",
        assessment.guidance_type,
        assessment.confidence,
        assessment.question_domain,
        assessment.struggle_type,
        assessment.needs_scaffolding_override,
    )
    logger.debug("Assessment reasoning: {}", assessment.reasoning)

    # Check diagnostic question count guardrail
    diagnostic_count = state.get("diagnostic_question_count", 0)
    guidance_type_final = assessment.guidance_type.lower()

    # Default to skipping diagnostics, unless we are in a scaffolding case below the limit
    student_has_provided_thoughts = True

    if guidance_type_final == "scaffolding":
        if diagnostic_count >= MAX_DIAGNOSTIC_QUESTIONS:
            logger.warning(
                "Diagnostic limit reached ({} >= {}). Forcing route to scaffolding_guide.",
                diagnostic_count, MAX_DIAGNOSTIC_QUESTIONS,
            )
            guidance_type_final = "scaffolding"
            student_has_provided_thoughts = True  # Skip the diagnostic node
        else:
            # This is the only case where we want to run the diagnostic node
            student_has_provided_thoughts = False

    # Generate context-aware diagnostic instruction
    diagnostic_instruction = ""
    if guidance_type_final == "scaffolding" and diagnostic_count < MAX_DIAGNOSTIC_QUESTIONS:
        diagnostic_instruction = _generate_diagnostic_instruction(
            question_domain=assessment.question_domain,
            struggle_type=assessment.struggle_type,
            student_message=latest_message,
            project_context=state.get("project_context", ""),
        )
        logger.debug(
            "Generated diagnostic instruction for {} domain", assessment.question_domain
        )

    return {
        "guidance_type": guidance_type_final,
        "confidence": assessment.confidence,
        "struggle_type": assessment.struggle_type.lower(),
        "needs_scaffolding_override": assessment.needs_scaffolding_override,
        "input_clarity": assessment.input_clarity.lower(),
        "question_domain": assessment.question_domain.lower(),
        "diagnostic_instruction": diagnostic_instruction,
        "student_profile_text": student_profile,
        "student_has_provided_initial_thoughts": student_has_provided_thoughts,
        "guidance_level_mismatch_warning": guidance_level_mismatch_warning,
    }
