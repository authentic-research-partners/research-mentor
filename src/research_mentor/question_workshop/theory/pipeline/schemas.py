"""Schemas for the Theory question pipeline.

Each stage produces structured output validated by these models.
The final output is a set of theory-grounded research questions with
framework, consilience evidence, and feasibility scores.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Domain Classification (Stage 1 input)
# ---------------------------------------------------------------------------


class DomainClassification(BaseModel):
    """Classify topic domain and extract search query."""

    domain_type: str = Field(
        max_length=50,
        description="One of: 'natural_science', 'social_science', 'off_topic'",
    )
    search_query: str = Field(
        max_length=150,
        description=(
            "A clean 3-5 word academic search query. Remove ALL filler words. "
            "Use only core scientific nouns."
        ),
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "domain_type": "natural_science",
            "search_query": "crystal formation symmetry breaking",
        },
    })


# ---------------------------------------------------------------------------
# Stage 1: Phenomenon Mapper
# ---------------------------------------------------------------------------


class PhenomenonAnalysis(BaseModel):
    """Patterns extracted from papers + student observations."""

    surprising_findings: list[str] = Field(
        description="Results that contradict expectations (2-4 items).",
    )
    cross_domain_connections: list[str] = Field(
        description="Links to other fields found in papers or observations (0-3 items).",
    )
    unexplained_phenomena: list[str] = Field(
        description="Patterns without established explanation (1-3 items).",
    )
    domain_summary: str = Field(
        max_length=500,
        description="One-sentence overview of the domain based on papers.",
    )
    key_observations: list[str] = Field(
        description="Student's observations enriched with literature context (2-5 items).",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "surprising_findings": [
                "Crystal growth rate increases at lower temperatures in some polymers",
            ],
            "cross_domain_connections": [
                "Similar symmetry breaking in both crystal formation and early embryo development",
            ],
            "unexplained_phenomena": [
                "Certain mineral crystals form in patterns "
                "not predicted by classical nucleation theory",
            ],
            "domain_summary": (
                "Crystal formation is well-studied in classical nucleation theory "
                "but recent work reveals anomalous pathways in nanoscale systems."
            ),
            "key_observations": [
                "Student noticed crystals growing faster near impurities — "
                "literature confirms impurity-driven nucleation is an active research area",
            ],
        },
    })


class PhenomenonResult(BaseModel):
    """Complete output of Stage 1: Phenomenon Mapper."""

    papers: list[dict] = Field(  # type: ignore[type-arg]
        description="Papers found via search_and_enrich.",
    )
    surprising_findings: list[str]
    cross_domain_connections: list[str]
    unexplained_phenomena: list[str]
    domain_summary: str
    key_observations: list[str]


# ---------------------------------------------------------------------------
# Stage 2: Abductive Reasoner
# ---------------------------------------------------------------------------


class CandidateExplanation(BaseModel):
    """One candidate explanation for observed patterns."""

    explanation: str = Field(
        max_length=500,
        description="What this explanation claims (1-2 sentences).",
    )
    supporting_evidence: list[str] = Field(
        description="Evidence from papers/observations that supports this (2-3 items).",
    )
    fails_to_explain: list[str] = Field(
        description="Observations this explanation cannot account for (1-2 items).",
    )
    source_domain: str = Field(
        max_length=150,
        description="The field or domain this reasoning originates from.",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "explanation": (
                "Impurities lower the energy barrier for nucleation by "
                "providing heterogeneous nucleation sites."
            ),
            "supporting_evidence": [
                "Classical nucleation theory predicts lower barriers at surfaces",
                "Experiments show faster growth near dust particles",
            ],
            "fails_to_explain": [
                "Why some pure solutions still show anomalous rapid crystallization",
            ],
            "source_domain": "materials science",
        },
    })


class ExplanationComparison(BaseModel):
    """Head-to-head comparison of candidate explanations."""

    best_explanation: str = Field(
        max_length=500,
        description="The explanation with the broadest explanatory scope (1-2 sentences).",
    )
    runner_up: str | None = Field(
        default=None,
        max_length=500,
        description="Second-best explanation, if close (1-2 sentences).",
    )
    comparison_reasoning: str = Field(
        max_length=800,
        description="Why the best explanation was selected over others (2-3 sentences).",
    )
    explanatory_scope: str = Field(
        max_length=300,
        description="What the best explanation covers (1 sentence).",
    )
    key_weakness: str = Field(
        max_length=300,
        description="What remains unexplained by the best explanation (1 sentence).",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "best_explanation": (
                "Two-step nucleation: amorphous precursors form first, "
                "then crystallize — explaining both fast and anomalous growth."
            ),
            "runner_up": "Impurity-driven heterogeneous nucleation.",
            "comparison_reasoning": (
                "Two-step nucleation explains both impurity-enhanced AND "
                "pure-solution anomalous crystallization, while heterogeneous "
                "nucleation only covers the impurity case."
            ),
            "explanatory_scope": (
                "Covers classical, impurity-driven, and anomalous crystallization "
                "pathways under a single mechanism."
            ),
            "key_weakness": (
                "Does not explain why certain crystal polymorphs are selected "
                "over others during the amorphous-to-crystalline transition."
            ),
        },
    })


class AbductionResult(BaseModel):
    """Complete output of Stage 2: Abductive Reasoner."""

    candidates: list[CandidateExplanation]
    comparison: ExplanationComparison


# ---------------------------------------------------------------------------
# Stage 3: Framework Builder
# ---------------------------------------------------------------------------


class TheoreticalFramework(BaseModel):
    """An organizing framework built from the best explanation."""

    framework_type: str = Field(
        max_length=50,
        description=(
            "One of: 'classification', 'taxonomy', 'conceptual_model', 'causal_model'."
        ),
    )
    description: str = Field(
        max_length=800,
        description="2-3 sentence summary of the framework.",
    )
    categories: list[str] = Field(
        description="Components or elements of the framework (3-6 items).",
    )
    relationships: list[str] = Field(
        description="How components connect to each other (2-4 items).",
    )
    predictions: list[str] = Field(
        description="Testable implications the framework makes (2-4 items).",
    )
    historical_analogy: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "A historical parallel if apt (e.g. Mendeleev's periodic table, "
            "Darwin's tree of life). None if no good analogy."
        ),
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "framework_type": "causal_model",
            "description": (
                "A two-pathway model of crystal nucleation where the pathway "
                "(classical vs. two-step) is selected by the degree of "
                "supersaturation and the presence of structural templates."
            ),
            "categories": [
                "Classical nucleation pathway",
                "Two-step nucleation pathway",
                "Supersaturation level",
                "Structural template availability",
            ],
            "relationships": [
                "High supersaturation favors two-step pathway",
                "Structural templates lower the barrier for classical pathway",
                "Both pathways compete; dominant pathway depends on conditions",
            ],
            "predictions": [
                "At intermediate supersaturation, both pathways should coexist",
                "Removing templates at high supersaturation should not slow growth",
            ],
            "historical_analogy": (
                "Like Wegener's continental drift — multiple independent lines "
                "of evidence (fossil, geological, climatic) converging on "
                "a single mechanism."
            ),
        },
    })


class FrameworkResult(BaseModel):
    """Complete output of Stage 3: Framework Builder."""

    framework: TheoreticalFramework
    best_explanation: str


# ---------------------------------------------------------------------------
# Stage 4: Consilience Tester
# ---------------------------------------------------------------------------


class ConsilienceTest(BaseModel):
    """Consilience test result for one domain."""

    domain: str = Field(
        max_length=150,
        description="The external domain tested.",
    )
    evidence_found: list[str] = Field(
        description="Papers or findings that support or contradict (2-4 items).",
    )
    fit: str = Field(
        max_length=50,
        description="One of: 'strong', 'partial', 'weak', 'contradicts'.",
    )
    what_it_explains: str = Field(
        max_length=300,
        description="What the framework explains in this domain (1 sentence).",
    )
    what_it_fails: str = Field(
        max_length=300,
        description="What the framework fails to explain (1 sentence).",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "domain": "protein folding",
            "evidence_found": [
                "Two-step folding via molten globule intermediate is well-documented",
                "Supersaturation analogy maps to protein concentration",
            ],
            "fit": "strong",
            "what_it_explains": (
                "Proteins also follow a two-step pathway through an "
                "amorphous intermediate before reaching native structure."
            ),
            "what_it_fails": (
                "Chaperone-assisted folding adds a biological layer "
                "not present in crystal nucleation."
            ),
        },
    })


class ConsilienceDomains(BaseModel):
    """Domains identified for consilience testing."""

    domains: list[str] = Field(
        description="2-3 domains outside the primary one where the framework might apply.",
    )
    search_queries: list[str] = Field(
        description="One search query per domain to find relevant evidence.",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "domains": ["protein folding", "atmospheric ice nucleation"],
            "search_queries": [
                "protein folding two-step nucleation intermediate",
                "atmospheric ice nucleation classical non-classical pathway",
            ],
        },
    })


class ConsilienceResult(BaseModel):
    """Complete output of Stage 4: Consilience Tester."""

    domains_tested: list[ConsilienceTest]
    successes: int
    partial: int
    failures: int
    framework_strength: str = Field(
        max_length=50,
        description="One of: 'strong', 'moderate', 'weak'.",
    )
    revision_notes: str | None = Field(
        default=None,
        max_length=800,
        description="If failures suggest framework needs revision, describe how.",
    )


# ---------------------------------------------------------------------------
# Stage 5: Question Generator
# ---------------------------------------------------------------------------


class TheoryQuestion(BaseModel):
    """A single theory-grounded research question."""

    question: str = Field(
        max_length=500,
        description="The research question (1-2 sentences).",
    )
    question_type: str = Field(
        max_length=50,
        description=(
            "One of: 'test_prediction', 'explain_failure', 'extend', 'refine'."
        ),
    )
    framework_connection: str = Field(
        max_length=500,
        description="How this question relates to the framework (1-2 sentences).",
    )
    literature_support: list[str] = Field(
        description="Key findings from literature that motivate this question (1-3 items).",
    )
    why_interesting: str = Field(
        max_length=500,
        description="Why this question is scientifically interesting (1-2 sentences).",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "question": (
                "Does the two-step nucleation pathway dominate at intermediate "
                "supersaturation levels, as the framework predicts?"
            ),
            "question_type": "test_prediction",
            "framework_connection": (
                "Directly tests the framework's core prediction that "
                "pathway selection depends on supersaturation."
            ),
            "literature_support": [
                "Two-step nucleation observed in protein crystallization",
                "Classical pathway dominates at low supersaturation",
            ],
            "why_interesting": (
                "Would validate whether a single mechanism explains "
                "crystallization across organic and inorganic systems."
            ),
        },
    })


class TheoryQuestions(BaseModel):
    """Output of Stage 5: Question Generator."""

    questions: list[TheoryQuestion]


# ---------------------------------------------------------------------------
# Stage 6: Feasibility Scorer
# ---------------------------------------------------------------------------


class TheoryFeasibility(BaseModel):
    """Feasibility assessment for a theory-grounded question."""

    methodological_feasibility: int = Field(
        ge=1, le=10,
        description=(
            "Can this be investigated with available methods? "
            "(10=standard methods, 1=no known method)."
        ),
    )
    data_availability: int = Field(
        ge=1, le=10,
        description=(
            "Is evidence or data accessible? "
            "(10=public datasets, 1=none exist)."
        ),
    )
    student_accessibility: int = Field(
        ge=1, le=10,
        description=(
            "Appropriate for the student's level? "
            "(10=straightforward, 1=requires graduate training)."
        ),
    )
    scientific_value: int = Field(
        ge=1, le=10,
        description=(
            "Would answering this advance understanding? "
            "(10=major advance, 1=well-known result)."
        ),
    )
    framework_relevance: int = Field(
        ge=1, le=10,
        description=(
            "How well does it test or extend the framework? "
            "(10=directly tests core prediction, 1=tangential)."
        ),
    )
    overall_score: int = Field(
        ge=1, le=10,
        description="Overall recommendation score.",
    )
    reasoning: str = Field(
        max_length=800,
        description="Brief justification for the scores (2-3 sentences).",
    )


class ScoredTheoryQuestion(BaseModel):
    """A theory question with its feasibility assessment."""

    question: str
    question_type: str
    framework_connection: str
    literature_support: list[str]
    why_interesting: str
    feasibility: TheoryFeasibility


class ScoredTheoryQuestions(BaseModel):
    """Final output of the Theory pipeline."""

    questions: list[ScoredTheoryQuestion]
    framework: TheoreticalFramework
    consilience_summary: str = Field(
        max_length=500,
        description="Brief summary: tested N domains, M successes.",
    )
