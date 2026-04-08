"""Info Gathering node — autonomously decides which assistants to call.

Analyzes conversation context and decides which 0-3 assistant tools to invoke.
academic_literature calls real search APIs (arXiv, Semantic Scholar, PubMed, OpenAlex).
Brave Search is used if configured.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from research_mentor.agent.state import MentorState
from research_mentor.db.crud import (
    get_student_progress,
    search_similar_artifacts,
    search_similar_memories,
)
from research_mentor.llm import structured_call
from research_mentor.tools.academic_search import search_papers
from research_mentor.tools.concept_explainer import explain_concept
from research_mentor.tools.learning_pathway import recommend_learning_pathway
from research_mentor.tools.status import ToolStatus
from research_mentor.tools.web_search import search_web
from research_mentor.utils.memory_profiler import memory_profile_node

# Collaboration search tools (lazy-imported in executor to avoid circular deps)


class GatheringDecision(BaseModel):
    """Structured output for gathering decision."""

    assistants_to_call: list[str] = Field(description="List of assistant names to call")
    academic_sources: list[str] = Field(
        default_factory=list,
        description=(
            "Extra academic databases to search IN ADDITION to OpenAlex (always included). "
            "Options: 'arxiv', 'pubmed', 'semantic_scholar'. Empty = OpenAlex only."
        ),
    )
    collaboration_gap_type: str | None = Field(
        default=None,
        max_length=50,
        description=(
            "If collaboration_search is in assistants_to_call, specify the gap type: "
            "'data', 'instrument', 'methods', 'domain', or 'field_labor'. "
            "Null if collaboration_search is not called."
        ),
    )
    reasoning: str = Field(max_length=500, description="Brief explanation of why these assistants are needed")


GATHERING_INSTRUCTION = """You are an Autonomous Information Gathering Coordinator.

**Your Role:**
Analyze the student's question and context, then autonomously decide which assistants (0-3) to call
to gather information that will help the downstream guide provide effective teaching.

**Available Assistants:**
1. **academic_literature** — Search for academic papers and research articles.
   OpenAlex (270M+ papers, all disciplines) is ALWAYS searched.
   You may add extra sources via `academic_sources`:
   - **arxiv** — Preprints in physics, CS, math, biology. Best for cutting-edge work not yet published.
   - **pubmed** — Biomedical and life sciences. Essential for health, medicine, biology, neuroscience.
   - **semantic_scholar** — Strong in CS/AI, provides citation counts and TLDRs.
   Only add extras when the topic clearly benefits from that source's specialty.
2. **web_search** — Search the web for current information, datasets, tutorials
3. **student_progress** — Look up the student's project progress and milestones
4. **memory_search** — Search past conversation memories for relevant context
5. **artifact_search** — Search uploaded documents and files for relevant content
6. **concept_explanation** — Explain a prerequisite concept the student asks about \
(trigger: student explicitly asks "What is X?" / "Explain X" / "How does X work?")
7. **learning_pathway** — Recommend a learning pathway for a field \
(trigger: student explicitly asks about career/course/learning path, e.g. "How can I become \
an expert in X?" / "What should I study after this project?")
8. **collaboration_search** — Search for potential collaborators, labs, or facilities \
(trigger: student's project clearly has a specific gap — needs data they don't have, methods \
they can't do, equipment they lack access to, or domain expertise outside their field). \
If called, you MUST also set collaboration_gap_type to one of: 'data', 'instrument', \
'methods', 'domain', 'field_labor'. \
DO NOT call unless a clear, specific gap is evident from the conversation.

**Decision Guidelines by Student Tier:**

**Scaffolding (0-20% progress):**
- May need background literature when student asks for it
- May need web_search for tutorials, explanations, beginner resources
- Student needs foundational understanding
- Example: Literature question → Call academic_literature
- Example: "How do I start?" → Call web_search for tutorials

**Guided Discovery (20-60% progress):**
- Targeted literature + progress checks
- Student building on existing knowledge
- Example: Experimental design question → Call student_progress + academic_literature
- Example: "What papers should I read?" → Call academic_literature

**Pure Socratic (60-80% progress):**
- Minimal gathering - student should discover answers
- Maybe progress check to understand their journey
- Example: Usually 0-1 assistant calls

**Reflection (80-100% progress):**
- Usually no gathering - student is synthesizing
- Exception: Progress check to help with reflection on their journey
- Example: Usually 0 calls, maybe student_progress

**Decision Guidelines by Question Domain:**
- Literature search questions → academic_literature
- Progress/journey questions → student_progress (milestones, status, timeline)
- Current events, tools, datasets, tutorials → web_search
- **memory_search → WHEN student references what was previously discussed:**
  - The student asks YOU to remember, recall, or revisit something from a past conversation
  - The student references what "we" discussed, what "you said", or what happened "last time"
  - Key distinction: memory_search retrieves CONVERSATION HISTORY (what was said); \
student_progress retrieves PROJECT DATA (milestones, status). \
If the student says "remember my project" or "what was my hypothesis" — \
that information lives in past conversations, so use memory_search.
  - DO NOT call for: new questions, general project work, first messages
- **artifact_search → ONLY IF:**
  - Student EXPLICITLY references uploaded documents, files, or attached materials
  - Student asks about content in their uploaded files
  - Student asks about their data, results, images, figures, or analysis \
(e.g. "what test should I run?", "can you look at my data?", \
"what do you see in my gel/chart/image?")
  - DO NOT call for: general questions, topics not related to uploads or data
- **concept_explanation → ONLY IF:**
  - Student EXPLICITLY asks to explain a concept: "What is X?", "Explain X", "How does X work?"
  - The concept is prerequisite knowledge (not the student's research question)
  - DO NOT call for: general questions, research-specific queries
- **learning_pathway → ONLY IF:**
  - Student EXPLICITLY asks about career paths, courses, or long-term learning
  - E.g. "How can I become an expert in X?", "What should I study?"
  - DO NOT call for: immediate project questions, prerequisite concepts

**Check Previously Gathered Information First:**
You will see "Previously Gathered Information" from earlier turns with summaries.
DON'T call assistants for information you already have - reuse the cache!
Check SEMANTICALLY, not just by name — if a previous academic_literature search \
already covered the topic, don't search again even if the exact query differs.

**Your Decision Process:**
1. Analyze: What is the student asking about? What tier are they at?
2. Check cache: Do we already have this information from earlier?
3. Decide: Which assistants (0-3) would provide useful context for the guide?
4. Execute: Call assistants in parallel

**Important:**
- You gather information - the GUIDE handles how to present it pedagogically
- Call 0-3 assistants (not more) - be selective
- Reuse cached information when possible

{cache_summary}

**Guidance type:** {guidance_type}
**Question domain:** {question_domain}

If no information gathering is needed, return empty assistants_to_call list.
"""


# ---------------------------------------------------------------------------
# Assistant executors
# ---------------------------------------------------------------------------


async def _run_academic_literature(
    query: str,
    extra_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Search academic databases for papers related to the query.

    OpenAlex is always included. ``extra_sources`` adds additional databases
    (``arxiv``, ``pubmed``, ``semantic_scholar``) chosen by the LLM.
    """
    valid_extras = {"arxiv", "pubmed", "semantic_scholar"}
    sources = ["openalex"]
    for s in extra_sources or []:
        if s in valid_extras and s not in sources:
            sources.append(s)

    result = await search_papers(
        query,
        sources=sources,
        max_results_per_source=3,
    )
    tool_statuses: list[ToolStatus] = result.get("tool_statuses", [])
    papers = [r for r in result.get("results", []) if "error" not in r]
    if not papers:
        return {
            "assistant": "academic_literature",
            "results": [],
            "summary": "No papers found.",
            "tool_statuses": tool_statuses,
        }

    summaries = []
    for p in papers[:6]:
        title = p.get("title", "Unknown")
        year = p.get("year") or p.get("published", "")
        source = p.get("source", "")
        abstract = p.get("abstract") or ""
        entry = f"- **{title}** ({year}, {source})"
        if abstract:
            entry += f"\n  {abstract}"
        summaries.append(entry)

    note = (
        "\n\n*These are titles and abstracts from academic databases. "
        "For deeper analysis of a specific paper, you can upload the "
        "PDF via the Artifacts page, or paste the relevant section "
        "into this chat.*"
    )
    return {
        "assistant": "academic_literature",
        "results": papers[:6],
        "summary": f"Found {len(papers)} papers:\n" + "\n".join(summaries) + note,
        "tool_statuses": tool_statuses,
    }


async def _run_web_search(query: str) -> dict[str, Any]:
    """Search the web via Brave Search API."""
    result = await search_web(query, num_results=5)
    tool_status: ToolStatus = result.get(
        "tool_status", ToolStatus.ok("Brave Search"),
    )
    if not result.get("success"):
        error = result.get("error", "Web search failed")
        return {
            "assistant": "web_search",
            "results": [],
            "summary": f"Web search: {error}",
            "tool_statuses": [tool_status],
        }

    items = result.get("results", [])
    summaries = [
        f"- {r.get('title', '')}: {r.get('description', '')[:120]}"
        for r in items[:5]
    ]
    return {
        "assistant": "web_search",
        "results": items[:5],
        "summary": f"Found {len(items)} web results:\n" + "\n".join(summaries),
        "tool_statuses": [tool_status],
    }


async def _run_memory_search(query: str, project_id: str) -> dict[str, Any]:
    """Search conversation memories for relevant past context."""
    from research_mentor.embeddings import embed_text

    if not project_id:
        return {
            "assistant": "memory_search",
            "results": [],
            "summary": "No project context for memory search.",
        }

    from research_mentor.config import load_config

    config = load_config()
    query_embedding = await embed_text(query)
    memories = await search_similar_memories(
        project_id,
        query_embedding,
        threshold=config.embeddings.similarity_threshold,
        limit=config.embeddings.max_results,
    )

    if not memories:
        return {
            "assistant": "memory_search",
            "results": [],
            "summary": "No relevant past memories found.",
        }

    summaries = []
    for m in memories:
        sim = m.get("similarity", 0)
        text = m["memory_text"][:150]
        summaries.append(f"- [{sim:.2f}] <user_content>{text}</user_content>")

    return {
        "assistant": "memory_search",
        "results": memories,
        "summary": f"Found {len(memories)} relevant memories:\n" + "\n".join(summaries),
    }


async def _run_artifact_search(query: str, project_id: str) -> dict[str, Any]:
    """Search uploaded artifact content for relevant chunks."""
    from research_mentor.embeddings import embed_text

    if not project_id:
        return {
            "assistant": "artifact_search",
            "results": [],
            "summary": "No project context for artifact search.",
        }

    from research_mentor.config import load_config

    config = load_config()
    query_embedding = await embed_text(query)
    results = await search_similar_artifacts(
        project_id,
        query_embedding,
        threshold=config.embeddings.similarity_threshold,
        limit=config.embeddings.max_results,
    )

    if not results:
        return {
            "assistant": "artifact_search",
            "results": [],
            "summary": "No relevant content found in uploaded artifacts.",
        }

    # Types where the full analysis is embedded as one chunk (not a fragment)
    _FULL_ANALYSIS_TYPES = {
        "data", "photograph", "scientific_image", "chart", "handwriting",
    }

    summaries = []
    for r in results:
        sim = r.get("similarity", 0)
        fname = r["file_name"]
        atype = r.get("artifact_type", "")
        desc = r.get("description", "")
        text = r["chunk_text"][:500]
        is_full = atype in _FULL_ANALYSIS_TYPES
        label = "full analysis" if is_full else "extracted chunk"
        header = f"[{sim:.2f}] {fname} ({atype}, {label})"
        if desc:
            header += f" — <user_content>{desc}</user_content>"
        summaries.append(f"- {header}\n  <user_content>{text}</user_content>")

    return {
        "assistant": "artifact_search",
        "results": results,
        "summary": f"Found {len(results)} relevant artifact chunks:\n" + "\n".join(summaries),
    }


async def _run_student_progress(project_id: str) -> dict[str, Any]:
    """Look up student's project progress from the database."""
    progress = await get_student_progress(project_id)

    if not progress or "error" in progress:
        return {
            "assistant": "student_progress",
            "results": [],
            "summary": "No project progress found.",
        }
    return {
        "assistant": "student_progress",
        "results": [progress],
        "summary": (
            f"Project: {progress.get('title', 'Untitled')} — "
            f"{len(progress.get('milestones', []))} milestones"
        ),
    }


async def _run_concept_explanation(
    query: str,
    research_context: str,
    student_demographics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Explain a prerequisite concept the student asks about."""
    result = await explain_concept(query, research_context, student_demographics)

    if result.get("is_research_question"):
        return {
            "assistant": "concept_explanation",
            "results": [result],
            "summary": f"Research question detected — Socratic redirect: {result.get('refusal', '')}",
        }

    return {
        "assistant": "concept_explanation",
        "results": [result],
        "summary": (
            f"Concept '{result.get('concept', query)}': "
            f"{result.get('explanation', '')[:200]}"
        ),
    }


async def _run_learning_pathway(
    query: str,
    project_context: str,
    student_demographics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recommend a learning pathway for a field of interest."""
    result = await recommend_learning_pathway(query, project_context, student_demographics)

    if "error" in result:
        return {
            "assistant": "learning_pathway",
            "results": [result],
            "summary": f"Learning pathway error: {result['error']}",
        }

    steps = result.get("learning_sequence", [])
    return {
        "assistant": "learning_pathway",
        "results": [result],
        "summary": (
            f"Learning pathway for {result.get('field', query)}: "
            f"{len(steps)} steps — {result.get('message_to_student', '')[:150]}"
        ),
    }


async def _run_collaboration_search(
    query: str,
    gap_type: str,
    project_id: str,
    student_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Search for potential collaborators, labs, or facilities.

    Uses researcher_search and institution_search tools based on gap type.
    Checks DB cache first, stores new results for future reference.
    """
    from research_mentor.config import load_config
    from research_mentor.db.crud import (
        get_collaboration_opportunities,
        store_collaboration_opportunity,
    )
    from research_mentor.tools.institution_search import search_institutions
    from research_mentor.tools.researcher_search import search_researchers

    cfg = load_config()
    cooldown_minutes = cfg.collaboration.search_cooldown_minutes

    # Check for cached (unexpired) results for this project + gap type
    cached = await get_collaboration_opportunities(
        project_id, gap_type=gap_type,
    )
    if cached:
        # Cooldown: check if most recent result was stored within cooldown window
        from research_mentor.db.connection import get_db

        async with get_db() as db:
            cursor = await db.execute(
                """
                SELECT 1 FROM collaboration_opportunities
                WHERE project_id = ? AND gap_type = ?
                  AND created_at > datetime('now', ?)
                LIMIT 1
                """,
                (project_id, gap_type, f"-{cooldown_minutes} minutes"),
            )
            within_cooldown = await cursor.fetchone() is not None

        # Filter out sentinel "no results" rows for display
        real_results = [r for r in cached if r["resource_type"] != "none"]

        if within_cooldown:
            if not real_results:
                return {
                    "assistant": "collaboration_search",
                    "results": [],
                    "summary": (
                        f"No collaboration opportunities found for {gap_type} gap "
                        f"(searched recently, cooldown active)."
                    ),
                }
            summaries = [
                f"- {r['resource_name']} ({r['resource_type']}, "
                f"{r['resource_affiliation'] or 'unknown affiliation'})"
                for r in real_results[:5]
            ]
            return {
                "assistant": "collaboration_search",
                "results": real_results[:5],
                "summary": (
                    f"Found {len(real_results)} cached collaboration opportunities "
                    f"for {gap_type} gap:\n" + "\n".join(summaries)
                ),
            }

    # Extract location context from student profile
    institution_ror = None
    country = None
    if student_profile:
        institution_ror = student_profile.get("institution_ror")
        country = student_profile.get("country")

    all_results: list[dict[str, Any]] = []
    tool_statuses: list[ToolStatus] = []

    if gap_type in ("data", "methods", "domain"):
        researchers = await search_researchers(
            query, institution_ror=institution_ror, country=country, max_results=5,
        )
        for r in researchers:
            if "error" not in r:
                all_results.append({
                    "resource_type": "researcher",
                    "resource_name": r["name"],
                    "resource_affiliation": r.get("affiliation", ""),
                    "openalex_author_id": r.get("openalex_id", ""),
                    "proximity": "local" if country and r.get("institution_country") == country else "international",
                })
            else:
                ts = r.get("tool_status")
                if ts:
                    tool_statuses.append(ts)

    elif gap_type == "instrument":
        institutions = await search_institutions(
            query, type_filter="facility", country=country, max_results=5,
        )
        for inst in institutions:
            if "error" not in inst:
                all_results.append({
                    "resource_type": "facility",
                    "resource_name": inst["name"],
                    "resource_affiliation": inst.get("homepage_url", ""),
                    "openalex_institution_id": inst.get("openalex_id", ""),
                    "proximity": "local" if country and inst.get("country") == country else "international",
                })
            else:
                ts = inst.get("tool_status")
                if ts:
                    tool_statuses.append(ts)

    elif gap_type == "field_labor":
        researchers = await search_researchers(
            query, country=country, max_results=5,
        )
        for r in researchers:
            if "error" not in r:
                all_results.append({
                    "resource_type": "researcher",
                    "resource_name": r["name"],
                    "resource_affiliation": r.get("affiliation", ""),
                    "openalex_author_id": r.get("openalex_id", ""),
                    "proximity": "local" if country and r.get("institution_country") == country else "international",
                })
            else:
                ts = r.get("tool_status")
                if ts:
                    tool_statuses.append(ts)

    # Store results in DB for caching
    for r in all_results[:5]:
        await store_collaboration_opportunity(
            project_id,
            gap_type=gap_type,
            need_description=query[:200],
            resource_type=r["resource_type"],
            resource_name=r["resource_name"],
            resource_affiliation=r.get("resource_affiliation"),
            openalex_author_id=r.get("openalex_author_id"),
            openalex_institution_id=r.get("openalex_institution_id"),
            proximity=r.get("proximity"),
        )

    if not all_results:
        # Store sentinel row so cooldown applies even when nothing was found
        await store_collaboration_opportunity(
            project_id,
            gap_type=gap_type,
            need_description=query[:200],
            resource_type="none",
            resource_name="(no results)",
        )
        return {
            "assistant": "collaboration_search",
            "results": [],
            "summary": f"No collaboration opportunities found for {gap_type} gap.",
            "tool_statuses": tool_statuses,
        }

    summaries = [
        f"- {r['resource_name']} ({r['resource_type']}, {r.get('resource_affiliation', '')})"
        for r in all_results[:5]
    ]
    return {
        "assistant": "collaboration_search",
        "results": all_results[:5],
        "summary": (
            f"Found {len(all_results)} potential collaborators/resources "
            f"for {gap_type} gap:\n" + "\n".join(summaries)
        ),
        "tool_statuses": tool_statuses,
    }


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------


@memory_profile_node("info_gathering")
async def gather_information(state: MentorState) -> dict[str, Any]:
    """Autonomously gather information based on context."""
    logger.info("Info gathering: Deciding what to gather...")

    guidance_type = state.get("guidance_type", "guided_discovery")
    question_domain = state.get("question_domain", "general")
    messages = state.get("messages", [])
    cache = state.get("conversation_info_cache", [])
    project_id = state.get("project_id", "")

    if cache:
        cache_items: list[str] = []
        for i, item in enumerate(cache[-8:], start=max(1, len(cache) - 7)):
            assistant = item.get("assistant", "?")
            request = item.get("request", "?")[:60]
            response = item.get("response", "")[:120]
            cache_items.append(
                f"  Turn {i} — {assistant}: query='{request}'\n    → {response}"
            )
        cache_summary = "Previously gathered:\n" + "\n".join(cache_items)
    else:
        cache_summary = "No previous information gathered in this conversation."

    instruction = GATHERING_INSTRUCTION.format(
        cache_summary=cache_summary,
        guidance_type=guidance_type,
        question_domain=question_domain,
    )

    latest_msg = messages[-1].content if messages else "No message"
    query_text = (
        latest_msg[:200] if isinstance(latest_msg, str)
        else str(latest_msg)[:200]
    )
    t0 = time.monotonic()

    decision = await structured_call(
        GatheringDecision,
        [
            SystemMessage(content=instruction),
            HumanMessage(content=f"Student question:\n\n{latest_msg}"),
        ],
        thinking="medium",
    )

    assistants = decision.assistants_to_call
    decision_ms = int((time.monotonic() - t0) * 1000)

    if not assistants:
        logger.info("Info gathering: No assistants needed ({}ms)", decision_ms)
        return {"gathered_information": None}

    logger.info(
        "Info gathering: Calling {} assistants: {} (decision: {}ms)",
        len(assistants), assistants, decision_ms,
    )

    # Execute requested assistants in parallel
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    loop = asyncio.get_running_loop()

    for assistant in assistants:
        if assistant == "academic_literature":
            tasks.append(loop.create_task(
                _run_academic_literature(query_text, decision.academic_sources),
            ))
        elif assistant == "web_search":
            tasks.append(loop.create_task(
                _run_web_search(query_text),
            ))
        elif assistant == "student_progress":
            tasks.append(loop.create_task(
                _run_student_progress(project_id),
            ))
        elif assistant == "memory_search":
            tasks.append(loop.create_task(
                _run_memory_search(query_text, project_id),
            ))
        elif assistant == "artifact_search":
            tasks.append(loop.create_task(
                _run_artifact_search(query_text, project_id),
            ))
        elif assistant == "concept_explanation":
            project_context = state.get("project_context", "")
            demographics = state.get("student_demographics")
            tasks.append(loop.create_task(
                _run_concept_explanation(query_text, project_context, demographics),
            ))
        elif assistant == "learning_pathway":
            project_context = state.get("project_context", "")
            demographics = state.get("student_demographics")
            tasks.append(loop.create_task(
                _run_learning_pathway(query_text, project_context, demographics),
            ))
        elif assistant == "collaboration_search":
            gap_type = decision.collaboration_gap_type or "methods"
            student_profile = state.get("student_demographics")
            tasks.append(loop.create_task(
                _run_collaboration_search(
                    query_text, gap_type, project_id, student_profile,
                ),
            ))
        else:
            logger.warning(
                "Info gathering: Unknown assistant '{}', skipping",
                assistant,
            )

    if not tasks:
        return {"gathered_information": None}

    results_raw = await asyncio.gather(*tasks, return_exceptions=True)

    all_results: list[dict[str, Any]] = []
    summaries: list[str] = []
    tool_warnings: list[dict[str, Any]] = []
    for r in results_raw:
        if isinstance(r, BaseException):
            logger.error("Info gathering assistant failed: {}", r)
            all_results.append({"assistant": "unknown", "error": str(r)})
            tool_warnings.append(
                ToolStatus.error("unknown", "network", str(r)).model_dump(),
            )
        else:
            all_results.append(r)
            summaries.append(r.get("summary", ""))
            # Collect non-ok tool statuses as warnings
            for ts in r.get("tool_statuses", []):
                if isinstance(ts, ToolStatus) and ts.level != "ok":
                    tool_warnings.append(ts.model_dump())
            # Remove ToolStatus objects — already extracted into tool_warnings;
            # keeping them in results causes JSON serialization failures in
            # the LangGraph checkpointer.
            r.pop("tool_statuses", None)

    total_ms = int((time.monotonic() - t0) * 1000)

    gathered: dict[str, Any] = {
        "results": all_results,
        "summary": "\n\n".join(summaries),
        "rounds": 1,
        "total_requests": len(tasks),
        "academic_sources": decision.academic_sources,
        "decision_time_ms": decision_ms,
        "execution_time_ms": total_ms,
    }

    new_cache = list(cache)
    for result in all_results:
        new_cache.append({
            "assistant": result.get("assistant", "unknown"),
            "request": query_text,
            "response": result.get("summary", "")[:500],
        })

    return {
        "gathered_information": gathered,
        "conversation_info_cache": new_cache,
        "tool_warnings": tool_warnings,
    }
