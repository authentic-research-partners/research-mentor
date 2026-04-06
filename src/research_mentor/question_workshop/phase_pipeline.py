"""Phase pipeline — split the monolithic CitedResponse call into focused sub-steps.

Replaces the pattern where one structured_call(CitedResponse) must simultaneously
track context, apply pedagogy, avoid pre-loading, ground in literature, and
generate natural prose. Instead, each step has ONE job:

Step 1 (PARALLEL): ContextSummary + ProgressExtraction
Step 2 (SEQUENTIAL): PedagogicalDecision — what to probe and how
Step 3 (SEQUENTIAL): CitedResponse — write 2-3 sentences executing the decision
Step 4 (OPTIONAL): PreloadingCheck — regenerate if response answers its own question

vLLM batches the parallel calls automatically via asyncio.gather().
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel

from research_mentor.llm import structured_call
from research_mentor.question_workshop.citation_utils import (
    format_cited_response,
    format_papers_for_prompt,
)
from research_mentor.question_workshop.pipeline_schemas import (
    CitedResponse,
    ContextSummary,
    PedagogicalDecision,
    PreloadingCheck,
)

# ==================== Configuration ====================


@dataclass
class PipelineConfig:
    """Per-phase configuration for the pipeline.

    Each phase node creates this with its specific rules and context,
    then passes it to run_phase_pipeline().
    """

    phase_name: str
    """Human-readable phase name (e.g. 'phase_2_simplification')."""

    pedagogical_rules: str
    """Phase-specific pedagogical rules. Focused text describing what
    pedagogical moves are appropriate and what to avoid. Extracted from
    the original monolithic prompts."""

    context_fields: dict[str, str]
    """Workshop-specific state summary for context extraction.
    Keys are field names, values are formatted strings.
    E.g. {'components': 'predators; prey', 'assumptions_count': '2'}"""

    papers: list[dict[str, Any]] = field(default_factory=list)
    """Papers list for citation formatting."""

    student_demographics: dict[str, Any] = field(default_factory=dict)
    """Student profile (age, grade, country, etc.)."""

    language: str = "en"
    """Response language."""

    last_user_message: str = ""
    """The student's latest message — highlighted in generation prompt."""

    workshop_name: str = ""
    """Workshop identifier for logging (e.g. 'modeling', 'theory')."""

    enable_preloading_check: bool = True
    """Whether to run the optional Step 4 guardrail."""

    state_context: ContextSummary | None = None
    """Pre-built context summary from existing state fields. When provided,
    skips the ContextSummary LLM call (Step 1a) and uses this instead.
    This is preferred when the state already tracks what was established
    (e.g., Modeling's system_components, system_interactions, change_drivers)."""

    pedagogical_override: PedagogicalDecision | None = None
    """Pre-built pedagogical decision from code logic. When provided,
    skips the LLM PedagogicalDecision call (Step 2) entirely.
    See docs/philosophy/code-decides-model-writes.md for rationale.

    CAUTION: This is computed BEFORE extraction runs, so it uses stale
    state from the previous turn. Prefer pedagogical_decision_fn instead,
    which receives fresh extraction results."""

    pedagogical_decision_fn: Callable[[BaseModel], PedagogicalDecision] | None = None
    """Code-based decision function called AFTER extraction (Step 1) completes.
    Receives the fresh progress extraction result as its argument, so the
    decision can account for what the student just said — not just what
    was in state from the previous turn.

    This is the preferred approach over pedagogical_override. If both are
    set, decision_fn takes priority."""


@dataclass
class PipelineResult:
    """Output from the pipeline."""

    response_text: str
    """Final formatted response with citations appended."""

    papers_cited: list[int]
    """Paper numbers referenced in the response."""

    context_summary: ContextSummary
    """For logging and debugging — never shown to student."""

    pedagogical_decision: PedagogicalDecision
    """For logging and debugging."""

    preloading_flagged: bool = False
    """Whether the guardrail triggered and response was regenerated."""


# ==================== Pipeline Orchestrator ====================


async def run_phase_pipeline[T: BaseModel](
    config: PipelineConfig,
    messages: list[BaseMessage],
    progress_schema: type[T],
    progress_prompt: str,
) -> tuple[PipelineResult, T]:
    """Execute the 4-step pipeline.

    Returns (pipeline_result, progress_extraction) so the phase node
    can use progress for state updates and forward transitions.
    """
    prefix = f"Pipeline [{config.workshop_name}/{config.phase_name}]"

    # ---- Step 1: Context summary + progress extraction ----
    progress_messages = [
        SystemMessage(content=progress_prompt),
        HumanMessage(content="Analyze the student's progress."),
    ]

    if config.state_context is not None:
        # Use pre-built context from state (zero LLM calls for context)
        logger.debug("{}: Step 1 — using state-built context + progress extraction", prefix)
        context_summary = config.state_context
        progress_result = await structured_call(
            progress_schema, progress_messages, thinking="high", temperature=0.0,
        )
    else:
        # Fall back to LLM extraction (parallel with progress)
        logger.debug("{}: Step 1 — parallel LLM context + progress extraction", prefix)
        context_messages = _build_context_messages(messages, config)
        context_summary, progress_result = await asyncio.gather(
            structured_call(ContextSummary, context_messages, thinking="medium", temperature=0.0),
            structured_call(progress_schema, progress_messages, thinking="high", temperature=0.0),
        )

    logger.info(
        "{}: Context — {} contributions, stage={}, latest='{}'",
        prefix,
        len(context_summary.student_contributions),
        context_summary.conversation_stage,
        context_summary.latest_student_point[:80],
    )

    # ---- Step 2: Pedagogical decision (code-based or LLM fallback) ----
    if config.pedagogical_decision_fn is not None:
        decision = config.pedagogical_decision_fn(progress_result)
        logger.debug("{}: Step 2 — using code-based decision fn (fresh extraction)", prefix)
    elif config.pedagogical_override is not None:
        decision = config.pedagogical_override
        logger.debug("{}: Step 2 — using pre-built pedagogical decision", prefix)
    else:
        decision_messages = _build_decision_messages(
            context_summary, config, progress_result,
        )
        logger.debug("{}: Step 2 — LLM pedagogical decision", prefix)
        decision = await structured_call(
            PedagogicalDecision, decision_messages, thinking="medium", temperature=0.0,
        )

    logger.info(
        "{}: Decision — move={}, focus='{}', avoid='{}'",
        prefix,
        decision.pedagogical_move,
        decision.focus_point[:80],
        decision.avoid[:60],
    )

    # ---- Step 3: SEQUENTIAL — generate response ----
    gen_messages = _build_generation_messages(decision, config, messages)

    logger.debug("{}: Step 3 — response generation", prefix)
    response = await structured_call(CitedResponse, gen_messages, thinking="high", temperature=0.7)

    # ---- Step 4: OPTIONAL — preloading check ----
    preloading_flagged = False
    if config.enable_preloading_check:
        logger.debug("{}: Step 4 — preloading check", prefix)
        check = await structured_call(
            PreloadingCheck,
            _build_preloading_messages(response.text, context_summary),
            thinking="off",
            temperature=0.0,
        )
        if check.contains_preloading:
            preloading_flagged = True
            logger.warning(
                "{}: Preloading detected — regenerating. Suggestion: {}",
                prefix, check.suggestion[:100],
            )
            gen_messages_constrained = _build_generation_messages(
                decision, config, messages,
                extra_constraint=f"CRITICAL: {check.suggestion}",
            )
            response = await structured_call(
                CitedResponse, gen_messages_constrained, thinking="high", temperature=0.7,
            )

    # Format citations from the decision's paper references
    cited_papers = decision.papers_to_reference or response.papers_cited
    final_text = format_cited_response(response.text, cited_papers, config.papers)

    return (
        PipelineResult(
            response_text=final_text,
            papers_cited=cited_papers,
            context_summary=context_summary,
            pedagogical_decision=decision,
            preloading_flagged=preloading_flagged,
        ),
        progress_result,
    )


# ==================== Message Builders ====================


def _build_context_messages(
    messages: list[BaseMessage],
    config: PipelineConfig,
) -> list[BaseMessage]:
    """Build messages for Step 1: ContextSummary extraction."""
    # Format conversation for extraction
    recent = messages[-10:] if len(messages) > 10 else messages
    conversation_lines = []
    for msg in recent:
        role = "Student" if isinstance(msg, HumanMessage) else "Mentor"
        conversation_lines.append(f"{role}: {msg.content}")
    conversation_text = "\n".join(conversation_lines)

    # Include workshop-specific context fields
    context_info = "\n".join(
        f"- {key}: {value}" for key, value in config.context_fields.items()
    )

    prompt = (
        f"Analyze this conversation and extract a summary of what has been "
        f"established so far.\n\n"
        f"Phase: {config.phase_name}\n"
        f"Current state:\n{context_info}\n\n"
        f"Conversation:\n{conversation_text}\n\n"
        f"Extract ONLY what the STUDENT has contributed (not the mentor's "
        f"suggestions). List the mentor's questions to avoid repetition."
    )

    return [
        SystemMessage(content=prompt),
        HumanMessage(content="Summarize the conversation context."),
    ]


def _build_decision_messages(
    context: ContextSummary,
    config: PipelineConfig,
    progress: BaseModel,
) -> list[BaseMessage]:
    """Build messages for Step 2: PedagogicalDecision."""
    # Format context summary
    contributions = "\n".join(
        f"  - {c}" for c in context.student_contributions
    ) or "  (none yet)"
    asked = "\n".join(
        f"  - {q}" for q in context.questions_already_asked
    ) or "  (none yet)"

    # Format progress as key-value pairs
    progress_lines = []
    for field_name, field_value in progress.model_dump().items():
        progress_lines.append(f"  {field_name}: {field_value}")
    progress_text = "\n".join(progress_lines)

    # Paper list for grounding decisions
    paper_list = format_papers_for_prompt(config.papers) if config.papers else "none"

    prompt = (
        f"Decide what to do next in this mentoring conversation.\n\n"
        f"Phase: {config.phase_name}\n"
        f"Conversation stage: {context.conversation_stage}\n\n"
        f"What the student has contributed:\n{contributions}\n\n"
        f"Student's latest point: {context.latest_student_point}\n\n"
        f"Questions already asked (DO NOT repeat these):\n{asked}\n\n"
        f"Progress extraction:\n{progress_text}\n\n"
        f"Available papers:\n{paper_list}\n\n"
        f"PHASE-SPECIFIC RULES:\n{config.pedagogical_rules}\n\n"
        f"Based on all of the above, decide:\n"
        f"1. focus_point: The ONE most important thing to probe next\n"
        f"2. pedagogical_move: How to probe it\n"
        f"3. papers_to_reference: Which papers (by number) are relevant (0-2)\n"
        f"4. avoid: What NOT to do (especially: do not repeat asked questions)"
    )

    return [
        SystemMessage(content=prompt),
        HumanMessage(content="Make your pedagogical decision."),
    ]


def _build_generation_messages(
    decision: PedagogicalDecision,
    config: PipelineConfig,
    messages: list[BaseMessage],
    extra_constraint: str = "",
) -> list[BaseMessage]:
    """Build messages for Step 3: CitedResponse generation.

    The generation prompt is deliberately simple — the hard reasoning
    was already done in Step 2. This call just writes natural prose.

    State-to-prompt isolation: this prompt receives the PedagogicalDecision
    (focus_point + move) but NOT the raw ContextSummary. The student-facing
    text is generated from the decision + recent messages only.
    """
    # Paper list for citation references
    paper_list = format_papers_for_prompt(config.papers) if config.papers else ""

    # Demographics adaptation
    demo_parts: list[str] = []
    age = config.student_demographics.get("age")
    if age and isinstance(age, int) and age < 18:
        demo_parts.append(
            "This is a young student. Use SIMPLE everyday language. "
            "When you must use a scientific term, immediately explain it."
        )

    # Language instruction
    lang_parts: list[str] = []
    if config.language and config.language != "en":
        lang_parts.append(
            f"Respond ENTIRELY in {config.language}. "
            f"Use appropriate scientific terminology in {config.language}."
        )

    # Build the simple generation prompt
    avoid_text = f"\nAVOID: {decision.avoid}" if decision.avoid else ""
    constraint_text = f"\n{extra_constraint}" if extra_constraint else ""
    papers_text = f"\n\nPapers (reference by number):\n{paper_list}" if paper_list else ""

    prompt = (
        f"You are a research mentor having a conversation with a student.\n\n"
        f"YOUR TASK: Write a 2-3 sentence response that does this:\n"
        f"- Focus: {decision.focus_point}\n"
        f"- Move: {decision.pedagogical_move}\n"
        f"{avoid_text}{constraint_text}\n\n"
        f"STRICT RULES:\n"
        f"- EXACTLY 2-3 sentences. No headers, no bullet points, no lists.\n"
        f"- Ask ONE question. Address the student directly.\n"
        f"- Do NOT include paper titles or author names in your text.\n"
        f"- Do NOT provide answers, examples, or lists — only ask.\n"
        f"{''.join(demo_parts)}\n"
        f"{''.join(lang_parts)}"
        f"{papers_text}"
    )

    # Include last 8 messages for conversational context — enough for the model
    # to write coherently while the PedagogicalDecision provides strategic direction
    recent = messages[-8:] if len(messages) > 8 else messages
    result: list[BaseMessage] = [SystemMessage(content=prompt)]
    result.extend(recent)

    return result


def _build_preloading_messages(
    response_text: str,
    context: ContextSummary,
) -> list[BaseMessage]:
    """Build messages for Step 4: PreloadingCheck."""
    contributions = "\n".join(
        f"  - {c}" for c in context.student_contributions
    ) or "  (none yet)"

    prompt = (
        f"Check if this mentor response pre-loads answers.\n\n"
        f"What the STUDENT has contributed so far:\n{contributions}\n\n"
        f"Mentor's response to check:\n\"{response_text}\"\n\n"
        f"Pre-loading means the mentor provides information, examples, "
        f"specific concepts, or answers that the student has NOT articulated. "
        f"Asking questions is NOT pre-loading. Acknowledging what the student "
        f"already said is NOT pre-loading.\n\n"
        f"contains_preloading: TRUE only if the response introduces NEW "
        f"factual content the student did not say.\n"
        f"suggestion: If pre-loading found, how to rephrase as a question."
    )

    return [
        SystemMessage(content=prompt),
        HumanMessage(content="Check for pre-loading."),
    ]
