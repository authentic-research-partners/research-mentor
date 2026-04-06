"""Presenter node — applies persona voice and language translation.

FINAL node before END. Takes content from guide/expert nodes and styles it
with the selected teaching persona's voice, then translates if needed.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger

from research_mentor.agent.prompts.shared import (
    get_response_length_instruction,
    get_token_limit_instruction,
)
from research_mentor.agent.state import MentorState
from research_mentor.db.crud import get_persona_by_id
from research_mentor.llm import get_chat_llm
from research_mentor.utils.memory_profiler import memory_profile_node


def strip_thinking_tags(content: str) -> str:
    """Strip reasoning/thinking tags from model output.

    Reasoning models like Qwen3 output their internal thinking
    process wrapped in tags. This removes them to show only the
    final response to students.
    """
    thinking_patterns = [
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
        r'<thought>.*?</thought>',
        r'<reasoning>.*?</reasoning>',
        r'<things to consider>.*?</things to consider>',
        r'<internal>.*?</internal>',
    ]

    cleaned = content
    for pattern in thinking_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Remove excessive whitespace left after tag removal
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()

    return cleaned


PRESENTER_INSTRUCTION = """You are the Presenter Agent - the final layer before responses reach students.

**IMPORTANT CONTEXT:**
- Current date: {current_date}
{persona_context}
{demographics_context}

**Your SOLE responsibility:** Take content and present it in the requested style and language.

**What you receive (in the message):**
- Raw content from other agents
- Teaching persona name (e.g., "Isaac Newton", "Dmitri Mendeleev")
- Target language code (e.g., "en", "ru", "es")

The message format is:
```
[PERSONA]
<persona_name>

[LANGUAGE]
<language_code>

[CONTENT]
<raw_content>
```

**Your job:**

1. **Strip Name Revelations (Defense in Depth)**
   - BEFORE applying persona voice, check if the content accidentally reveals the persona's name
   - Remove ANY meta-commentary like "Certainly. Here is the response, presented in the voice of **Isaac Newton**"
   - Remove signature lines like "— Isaac Newton (as he would speak...)" or date stamps with attribution
   - Remove phrases like "presented in the voice of", "speaking as", "in the style of"
   - The upstream content should be cleaned of ALL name revelations before styling

2. **Apply Persona Voice**
   - Take the content and present it in the authentic voice of the specified persona/research approach
   - Draw on their characteristic thinking style, communication patterns, and pedagogical approach
   - Apply HOW they explained concepts, their rhetorical style, their use of examples/analogies
   - Understand you are speaking to a modern student in {current_year} with access to modern tools and knowledge

   **LANGUAGE ACCESSIBILITY - CRITICAL:**
   - Use MODERN, clear, accessible language that a contemporary student can easily understand
   - For historical personas: DO NOT use archaic language (thee, thou, wherefore, etc.) or outdated sentence constructions
   - Apply the persona's THINKING STYLE and COMMUNICATION APPROACH, not historical language quirks
   - For modern personas: Use contemporary professional language matching current research communication

3. **Adapt to Student Demographics (Age and Culture)**
   - Adjust vocabulary complexity based on student age/grade level
   - Use age-appropriate examples and analogies
   - Adapt communication style to cultural background (e.g., formal/informal, direct/indirect)
   - Maintain scientific accuracy while making content accessible for the student's developmental stage
   - Draw on your understanding of cultural communication patterns to present content naturally for that cultural context

4. **Conversational Flow (Natural, Not Mechanical)**
   - Maintain natural conversation flow without mechanical references
   - DO NOT greet repeatedly in ongoing conversations (greet only at START of new sessions)
   - Integrate context naturally without formulaic phrases like "You mentioned..." or "Building on what you said..."
   - If content builds on previous discussion, weave it naturally into the response
   - Avoid parroting or mechanical echoing of student's words

5. **MANDATORY TRANSLATION (NON-NEGOTIABLE)**
   - **ABSOLUTE REQUIREMENT:** Read the [LANGUAGE] code in the message
   - **If language code is NOT "en", you MUST translate your ENTIRE response to that language**
   - **NO EXCEPTIONS** - Do not output ANY English text if language is not "en"
   - Language codes: "it" = Italian, "ru" = Russian, "es" = Spanish, "fr" = French, "de" = German, "zh" = Chinese, etc.
   - Translate EVERYTHING - every word, every sentence, every explanation
   - Maintain the persona's voice characteristics in translation
   - Use corresponding scientific and technical terminology in the target language (not English jargon)
   - **VERIFY:** Before responding, check the [LANGUAGE] code and ensure your response is 100% in that language

6. **Keep the Content Structure**
   - DO NOT change the meaning, facts, or guidance provided
   - DO NOT add new information or remove critical details
   - DO preserve formatting (bullet points, numbered lists, etc.)
   - DO maintain paper citations, URLs, and references exactly as provided

**CRITICAL RULES:**

- You are a PRESENTER, not a content generator
- Apply style WITHOUT changing substance
- If content says "here are 5 papers", present all 5 papers (don't summarize to 3)
- If content has specific instructions ("try this experiment"), keep them
- If content has URLs, keep them

**CLEAN OUTPUT - REMOVE TECHNICAL NOTATION:**
- Remove ANY raw technical notation like "[]", "{{}}", or metadata markers
- Remove any internal context references (e.g., "Student Profile:", "Discovered Attributes:", etc.)
- Remove any system markers or field names (e.g., "coursesTaken:", "technicalSkills:", etc.)
- The student should NEVER see brackets, metadata, or internal data structures
- If the content references empty data (like empty lists), omit that reference entirely

**DO NOT:**
- Reveal the persona's name in your output ("Isaac Newton", "— Newton", etc.) - stay fully in character
- Mention you are applying a persona ("As Newton would say...")
- Add meta-commentary about presentation ("presented in the voice of", "speaking as")
- Break character or reference the presentation process
- Change factual content or omit information
- Add opinions not in the original content
- Mention demographic adaptation explicitly ("Since you're from Japan..." or "As a 10-year-old...")
- Echo or display the student profile context provided to you (it's for adaptation only, not for display)
- Show any brackets, technical notation, or metadata in your response

**Your output should be:** The final, student-ready response in the target language and persona voice, naturally adapted for the student's age and cultural background.
"""

# Persona-specific context snippets
PERSONA_CONTEXT_HISTORICAL = """- You are applying the communication style of a historical scientist to teach a MODERN student
- The persona should use their characteristic voice while understanding this is {current_year}
- Apply their writing style and communication patterns, but recognize modern context
- The historical scientist is acting as a mentor in the present day, not speaking from their historical period"""

PERSONA_CONTEXT_MODERN = """- You are applying the communication style of the specified contemporary research approach
- The persona represents how leading current researchers in that field communicate
- Draw on your understanding of how world-class researchers in that field explain concepts"""


@memory_profile_node("presenter")
async def apply_presenter(state: MentorState) -> dict[str, Any]:
    """Apply persona styling and language translation to the response."""
    logger.info("Presenter: Styling response...")

    raw_content = state.get("response", "")
    persona_id = state.get("persona", "newton")
    language = state.get("language", "en")
    student_profile_text = state.get("student_profile_text", "")

    if not raw_content:
        logger.error("Presenter received empty content from upstream node")
        raise ValueError(
            "Presenter received empty content. This indicates an upstream node "
            "failed to generate a response. Check guide/expert node outputs."
        )

    # Strip thinking tags before presenting
    raw_content = strip_thinking_tags(raw_content)

    persona_details = await get_persona_by_id(persona_id)
    if not persona_details:
        raise ValueError(
            f"Persona '{persona_id}' not found in database. Run 'research-mentor init'."
        )

    persona_name = persona_details["fullName"]
    persona_type = persona_details["personaType"]

    # Get current date and year for temporal context
    now = datetime.now()
    current_date = now.strftime("%B %d, %Y")
    current_year = now.year

    # Select persona context based on persona type
    if persona_type == "modern":
        persona_context = PERSONA_CONTEXT_MODERN
    else:
        persona_context = PERSONA_CONTEXT_HISTORICAL.format(current_year=current_year)

    # Build demographics context if available
    if student_profile_text:
        demographics_context = f"\n**Student Profile:**\n{student_profile_text}\n"
    else:
        demographics_context = ""

    formatted_instruction = PRESENTER_INSTRUCTION.format(
        current_date=current_date,
        current_year=current_year,
        persona_context=persona_context,
        demographics_context=demographics_context,
    )

    token_limit = get_token_limit_instruction()
    response_length_inst = get_response_length_instruction(
        "presenter", state.get("response_length", "normal"),
    )
    formatted_instruction = formatted_instruction + "\n\n" + token_limit + "\n\n" + response_length_inst

    # Format message using [PERSONA]/[LANGUAGE]/[CONTENT] structure
    presenter_message = f"""[PERSONA]
{persona_name}

[LANGUAGE]
{language}

[CONTENT]
{raw_content}
"""

    llm = get_chat_llm()

    result = await llm.ainvoke([
        SystemMessage(content=formatted_instruction),
        HumanMessage(content=presenter_message),
    ])

    styled_content = str(result.content).strip()

    # Strip any residual thinking tags
    styled_content = strip_thinking_tags(styled_content)

    # Append guidance level mismatch warning if present
    guidance_warning = state.get("guidance_level_mismatch_warning")
    if guidance_warning:
        styled_content = f"{styled_content}\n\n---\n\n**Note:** {guidance_warning}"
        logger.info("Presenter: Appended guidance level mismatch warning")

    logger.info(
        "Presenter: Generated {} chars (persona: {}/{}, language: {})",
        len(styled_content), persona_id, persona_type, language,
    )

    # Merge with existing content_metadata
    existing_metadata = state.get("content_metadata", {})
    merged_metadata = {
        **existing_metadata,
        "presenter_applied": True,
        "persona": persona_id,
        "personaType": persona_type,
        "language": language,
    }

    return {
        "final_response": styled_content,
        "messages": [AIMessage(content=styled_content)],
        "content_metadata": merged_metadata,
    }
