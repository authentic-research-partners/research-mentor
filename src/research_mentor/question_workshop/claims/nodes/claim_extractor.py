"""Stage 1: Claim Extractor Node

Single responsibility: Extract and structure a scientific claim from input.

Given text (URL content, article text, or direct claim), identifies:
- The testable scientific claim
- Independent and dependent variables
- Scientific field
- Source type and credibility

Pattern: Chat LLM + Structured Extraction
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.claims._config import claims_config
from research_mentor.question_workshop.claims.schemas import (
    ClaimData,
    IsScienceClaim,
)
from research_mentor.question_workshop.domain import (
    get_natural_science_fields_display,
    get_refused_fields_display,
    get_social_science_fields_display,
)

CLAIM_EXTRACTION_PROMPT = """You are a scientific claim extractor focused on \
natural and social science.

Your task: Extract the main testable scientific claim from the provided text.

## What IS a Scientific Claim (extract these)

- **Causal:** "X causes Y", "X leads to Y", "X results in Y"
- **Correlational:** "X is linked to Y", "X is associated with Y"
- **Property:** "X has property Y", "X contains Y"
- **Process:** "X works by doing Y", "X prevents Y"

## What is NOT a Scientific Claim (REJECT these)

- Political opinions or policy debates
- Economic predictions or financial advice
- Social/psychological claims without biological basis
- Pure speculation without testable prediction
- Ethical/moral statements
- Opinions or editorials

## Field Classification

Classify into ONE of:
- **physics:** Energy, radiation, waves, forces, electromagnetic fields
- **chemistry:** Chemical reactions, pH, nutrients, toxins, molecules
- **biology:** Health, disease, metabolism, genes, organs, body systems

## Source Type Classification

- **mainstream_media:** NYT, BBC, CNN, major newspapers
- **wellness_blog:** Health/wellness blogs, lifestyle sites
- **social_media:** TikTok, Instagram, Twitter posts
- **influencer:** Celebrity or influencer content
- **press_release:** Company or institution announcements
- **unknown:** Can't determine

## Output Requirements

1. **original_text:** Quote the exact claim from the text (or close paraphrase)
2. **normalized_form:** Standardize as "X -> Y"
3. **independent_var:** The cause/input (X)
4. **dependent_var:** The effect/outcome (Y)
5. **field:** the natural science field (e.g., physics, chemistry, biology)
6. **claim_type:** causal, correlational, property, or process
7. **source_type:** From the list above
8. **source_name:** Name of the source if identifiable

## Compound Claims

Many media claims contain MULTIPLE sub-claims. Preserve ALL sub-claims:
- "X detoxifies the liver AND boosts metabolism by 30%" → normalize as \
"X -> liver detoxification + metabolism boost (30%)"
- "Magnetic bracelets realign electromagnetic field AND improve circulation 60%" \
→ normalize as "magnetic bracelet use -> electromagnetic field realignment + \
circulation improvement (60%)"
- Include ALL dependent variables, especially the most scientifically \
problematic ones (unfalsifiable concepts like "detoxification", physically \
incoherent claims like "realigning electromagnetic field")

## Field Classification — Borderline Cases

When a claim involves concepts from multiple fields, classify by the PRIMARY \
mechanism being claimed:
- pH, acid-base chemistry, molecular reactions → **chemistry** (even if the \
claimed effect is on health)
- Electromagnetic fields, forces, radiation → **physics** (even if applied to body)
- Metabolism, organ function, disease → **biology**
- Example: "Alkaline water (pH > 8.5) prevents disease" → **chemistry** \
(the core mechanism is about pH and acid-base chemistry)

## Important

- If the text is political/social without natural science basis, indicate this \
is not extractable
"""

def _build_pre_check_prompt(text: str) -> str:
    natural = get_natural_science_fields_display()
    social = get_social_science_fields_display()
    refused = get_refused_fields_display()
    return f"""Determine whether the following text contains a testable \
science claim — either natural science ({natural}) or social science ({social}).

Accept claims about the natural world OR about human behavior and society.
Reject: {refused}. Also reject political opinions that are not testable.

IMPORTANT: Wellness claims and pseudoscience claims about food, supplements, \
herbal remedies, or health products are NOT "nutrition science" or "medicine." \
They are testable biology or chemistry claims. ACCEPT them. Examples:
- "Lemon water detoxifies the liver" → ACCEPT (biology/chemistry claim)
- "Garlic cures the common cold" → ACCEPT (biology claim)
- "Alkaline water prevents disease" → ACCEPT (chemistry claim)
- "Magnetic bracelets improve circulation" → ACCEPT (physics claim)

Only reject actual clinical research proposals or medical treatment evaluations.

TEXT:
{text}
"""


async def claim_extractor(
    input_text: str,
    source_url: str | None = None,
) -> dict[str, Any]:
    """Stage 1: Extract scientific claim from input text.

    Args:
        input_text: Text containing the claim (article, claim text, etc.).
        source_url: Optional URL of the source.

    Returns:
        Dict with keys: claim_data, is_valid, rejection_reason, raw_response.
    """
    logger.info("Stage 1: Claim Extraction — input_length={}", len(input_text))

    # Pre-check: is this a testable science claim? (LLM-based, not keyword)
    pre_check = await structured_call(
        IsScienceClaim,
        [
            SystemMessage(content="You classify whether text contains testable science claims."),
            HumanMessage(content=_build_pre_check_prompt(input_text[:2000])),
        ],
        thinking="off",
    )

    if not pre_check.is_valid:
        logger.warning("Claim rejected (pre-check): {}", pre_check.rejection_reason)
        return {
            "claim_data": None,
            "is_valid": False,
            "rejection_reason": pre_check.rejection_reason,
            "raw_response": "",
        }

    # Step 1: Generate analysis with chat LLM
    user_prompt = f"""Analyze this text and extract the main scientific claim:

---
SOURCE URL: {source_url or 'Not provided'}

TEXT:
{input_text[:3000]}
---

Extract the scientific claim following the format specified.
If this is not a natural science claim, indicate that.
"""

    llm = get_chat_llm(temperature=claims_config().extraction_temperature)
    response = await llm.ainvoke([
        SystemMessage(content=CLAIM_EXTRACTION_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    raw_response = str(response.content)

    logger.debug("Claim extraction response length: {} chars", len(raw_response))

    # Note: rejection is handled by the structured IsScienceClaim
    # pre-check above. We do NOT use text-based heuristics on the chat
    # response because LLMs discussing pseudoscience claims naturally use
    # phrases like "not supported by science" which cause false rejections.

    # Step 2: Extract structured data
    extraction_prompt = """Extract structured claim data from the analysis.

Return the claim in the exact format specified:
- original_text: The exact claim quote
- normalized_form: "X -> Y" format
- independent_var: The X variable
- dependent_var: The Y variable
- field: the natural science field
- claim_type: causal, correlational, property, or process
- source_type: mainstream_media, wellness_blog, social_media, influencer, \
press_release, or unknown
- source_name: Name of source
"""

    claim_data = await structured_call(
        ClaimData,
        [
            SystemMessage(content=extraction_prompt),
            HumanMessage(
                content=f"Extract structured claim data from this analysis:\n\n"
                f"{raw_response}\n\nSource URL: {source_url or 'Not provided'}",
            ),
        ],
        thinking="medium",
    )

    if source_url:
        claim_data.source_url = source_url

    result = {
        "claim_data": claim_data.model_dump(),
        "is_valid": True,
        "rejection_reason": None,
        "raw_response": raw_response,
    }

    logger.info(
        "Stage 1 complete: field={}, type={}, X='{}', Y='{}'",
        claim_data.field, claim_data.claim_type,
        claim_data.independent_var, claim_data.dependent_var,
    )

    return result
