"""Stage 3: Gap Analyzer Node

Single responsibility: Analyze the gap between journalist "research" and real research.

Given the claim and evidence level, determines:
- What the journalist likely did (shallow research)
- What real research would require
- Why the claim is hard to test properly
- The key educational insight

Pattern: Direct structured_call (single-step generation)
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.claims._config import claims_config
from research_mentor.question_workshop.claims.schemas import (
    EvidenceLevel,
    GapAnalysis,
)

GAP_ANALYSIS_PROMPT = """You are an expert at exposing the gap between \
journalist "research" and real scientific research.

## The Core Contrast

| Journalist "Research" | Real Scientific Research |
|----------------------|-------------------------|
| 2-3 days of Googling | Months to years of work |
| Quote other articles | Collect original data |
| No methodology | Rigorous methodology |
| No peer review | Peer-reviewed publication |
| Confident headline | Careful uncertainty |

## Your Task

Analyze why this claim exists in media despite weak/no evidence, and what \
real research would require.

### 1. What the Journalist Likely Did
Infer what "research" led to this claim:
- Read other blogs/articles making similar claims
- Found quotes from non-experts ("health coaches", influencers)
- Misunderstood or exaggerated a real study
- Copied from press releases without verification
- Cited no peer-reviewed sources

### 2. What Real Research Would Require
Explain what proper scientific testing would need:
- Measurable definitions of vague terms
- Controlled experimental conditions
- Appropriate sample sizes
- Proper controls and blinding
- Statistical analysis
- Ethics approval (for human subjects)
- Peer review before publication

### 3. Why It's Hard to Test
List specific barriers to direct testing:
- Equipment costs ($X)
- Time requirements (months/years)
- Sample size requirements
- Ethical constraints
- Unmeasurable concepts

### 4. The Key Insight
The educational punchline — why this matters for understanding science vs. media.

## Important

- Be specific about what "real research" would look like
- Contrast the effort: "3 days" vs "3 years"
- Make the gap visceral and educational
- **Include field-specific barriers and requirements.** For physics claims: \
what instruments, what measurements, what physical quantities need measuring. \
For chemistry claims: what reagents, what analytical techniques, what safety \
considerations. For biology claims: what biological assays, what sample sizes, \
what ethical approvals. Generic barriers ("more research needed") are \
insufficient — name the specific scientific challenges.
- **The key insight must be specific to THIS claim**, not generic media \
literacy. Name the specific scientific mechanism or barrier that makes this \
particular claim fail — why THIS claim doesn't hold up, not just "media \
is unreliable." Bad: "This is why you can't trust what you read online."
"""


async def gap_analyzer(
    claim_data: dict[str, Any],
    evidence_result: dict[str, Any],
) -> dict[str, Any]:
    """Stage 3: Analyze the gap between journalist and real research.

    Args:
        claim_data: Output from claim_extractor.
        evidence_result: Output from evidence_searcher.

    Returns:
        Dict with keys: gap_analysis.
    """
    logger.info("Stage 3: Gap Analysis")

    original_text = claim_data.get("original_text", "")
    source_type = claim_data.get("source_type", "unknown")
    source_name = claim_data.get("source_name", "Unknown source")
    field = claim_data.get("field", "unknown")
    independent_var = claim_data.get("independent_var", "")
    dependent_var = claim_data.get("dependent_var", "")

    evidence_level = evidence_result.get("level", EvidenceLevel.NOT_SUPPORTED)
    what_science_says = evidence_result.get("what_science_says", "")
    papers_found = evidence_result.get("papers_found", 0)

    user_prompt = f"""Analyze the gap between journalist "research" and real \
science for this claim:

CLAIM: "{original_text}"
SOURCE: {source_name} ({source_type})
FIELD: {field}
VARIABLES: {independent_var} -> {dependent_var}

EVIDENCE LEVEL: {evidence_level}
PAPERS FOUND: {papers_found}
WHAT SCIENCE SAYS: {what_science_says}

Based on this information:
1. What did the journalist likely do to "research" this claim?
2. What would real scientific research require to test this?
3. Why is this claim hard to test properly?
4. What's the key insight about science vs. media?

Remember: The journalist probably spent 2-3 days on this. Real research takes \
months to years.
"""

    gap_analysis = await structured_call(
        GapAnalysis,
        [
            SystemMessage(content=GAP_ANALYSIS_PROMPT),
            HumanMessage(content=user_prompt),
        ],
        thinking="medium",
        temperature=claims_config().gap_temperature,
    )

    result = {
        "gap_analysis": gap_analysis.model_dump(),
    }

    logger.info(
        "Stage 3 complete: journalist_actions={}, requirements={}",
        len(gap_analysis.journalist_did),
        len(gap_analysis.real_research_requires),
    )

    return result
