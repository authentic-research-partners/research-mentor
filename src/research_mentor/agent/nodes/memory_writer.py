"""Memory Writer node — extracts and stores conversation memories after each turn.

Runs after presenter (final node before END). Uses a structured LLM call to
extract 0-3 memories worth remembering, embeds each, and stores in SQLite
via sqlite-vec for future semantic retrieval.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from research_mentor.agent.state import MentorState
from research_mentor.config import get_llm_model_info
from research_mentor.db.crud import store_memory
from research_mentor.embeddings import embed_text
from research_mentor.llm import structured_call
from research_mentor.utils.memory_profiler import memory_profile_node

MEMORY_EXTRACTION_INSTRUCTION = """You are a memory extraction agent for a research mentoring system.

Analyze the student's message and the mentor's response. Extract 0-3 memories
worth remembering for future conversations. Focus on:

1. **Student insights** — what the student knows, struggles with, or is interested in
2. **Pedagogical notes** — what teaching approach worked, what to avoid
3. **Key decisions** — methodology choices, research direction changes

**Rules:**
- Only extract memories that would be useful in FUTURE conversations
- Don't extract generic or obvious information
- Each memory should be a single, self-contained statement
- If the conversation is routine (greetings, simple Q&A), return 0 memories

**Student message:**
{student_message}

**Mentor response:**
{mentor_response}
"""


class ExtractedMemory(BaseModel):
    """A single extracted memory."""

    text: str = Field(max_length=500, description="The memory text — a self-contained statement")
    memory_type: str = Field(
        max_length=50,
        description="Type: student_insight, pedagogical_note, or key_decision"
    )
    importance: int = Field(description="Importance 1-10 (10 = critical)", ge=1, le=10)


class MemoryExtractionResult(BaseModel):
    """Result of memory extraction from a conversation turn."""

    memories: list[ExtractedMemory] = Field(
        description="List of 0-3 extracted memories",
    )


@memory_profile_node("memory_writer")
async def write_memories(state: MentorState) -> dict[str, Any]:
    """Extract and store memories from the current conversation turn."""
    project_id = state.get("project_id", "")
    if not project_id:
        logger.debug("Memory writer: No project_id, skipping")
        return {}

    messages = state.get("messages", [])
    final_response = state.get("final_response", "")

    if not messages or not final_response:
        logger.debug("Memory writer: No messages or response, skipping")
        return {}

    # Get the latest student message
    student_message = ""
    for msg in reversed(messages):
        if msg.type == "human":
            student_message = str(msg.content)
            break

    if not student_message:
        return {}

    # Truncate to avoid blowing up the context
    student_msg_truncated = student_message[:1000]
    response_truncated = final_response[:2000]

    instruction = MEMORY_EXTRACTION_INSTRUCTION.format(
        student_message=student_msg_truncated,
        mentor_response=response_truncated,
    )

    try:
        result = await structured_call(
            MemoryExtractionResult,
            [
                SystemMessage(content=instruction),
                HumanMessage(content="Extract memories from this conversation turn."),
            ],
            thinking="low",
        )
    except Exception:
        logger.exception("Memory writer: Failed to extract memories")
        return {}

    if not result.memories:
        logger.info("Memory writer: No memories to store")
        return {}

    stored_memories: list[dict[str, Any]] = []
    llm_info = get_llm_model_info()
    for mem in result.memories[:3]:
        embedding = await embed_text(mem.text)
        memory_id = await store_memory(
            project_id,
            mem.text,
            mem.memory_type,
            importance=mem.importance,
            embedding=embedding,
            conversation_context=llm_info,
        )
        stored_memories.append({
            "id": memory_id,
            "text": mem.text,
            "type": mem.memory_type,
            "importance": mem.importance,
        })

    logger.info("Memory writer: Stored {} memories", len(stored_memories))
    return {"retrieved_memories": stored_memories}
