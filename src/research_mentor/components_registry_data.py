"""Component registry data — metadata and redirect phrases for all Research Mentor flows.

This module holds the raw data. Consumers should use components_registry.py functions
instead of importing from here directly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComponentInfo:
    """Metadata for one Research Mentor component."""

    id: str
    name: str
    one_liner: str
    scope: str  # "social_science" | "natural_science" | "any" | "post_research"
    interactive: bool
    entry_point: str  # UI tab/route name
    completion_suggestions: tuple[str, ...]


COMPONENTS: dict[str, ComponentInfo] = {
    "office": ComponentInfo(
        id="office",
        name="Research Office",
        one_liner="free-form research mentoring on any topic",
        scope="any",
        interactive=True,
        entry_point="office",
        completion_suggestions=(),
    ),
    "phenomenon": ComponentInfo(
        id="phenomenon",
        name="Phenomenon Workshop",
        one_liner="generate natural science research problems from scratch",
        scope="natural_science",
        interactive=False,
        entry_point="phenomenon",
        completion_suggestions=("hypothesis", "gaps", "theory", "forge", "sharing"),
    ),
    "claims": ComponentInfo(
        id="claims",
        name="Claims Workshop",
        one_liner="turn media claims into structured research projects",
        scope="any",
        interactive=False,
        entry_point="claims",
        completion_suggestions=("hypothesis", "gaps", "sharing"),
    ),
    "hypothesis": ComponentInfo(
        id="hypothesis",
        name="Hypothesis Workshop",
        one_liner="develop causal X\u2192Y hypotheses with full experimental design",
        scope="social_science",
        interactive=True,
        entry_point="hypothesis",
        completion_suggestions=("sharing",),
    ),
    "gaps": ComponentInfo(
        id="gaps",
        name="Gaps Workshop",
        one_liner="discover fundamental questions through literature analysis",
        scope="natural_science",
        interactive=False,
        entry_point="gaps",
        completion_suggestions=("theory", "hypothesis", "sharing"),
    ),
    "questioned": ComponentInfo(
        id="questioned",
        name="Questioned Workshop",
        one_liner="surface papers under scrutiny in your field",
        scope="any",
        interactive=False,
        entry_point="questioned",
        completion_suggestions=("retractions",),
    ),
    "retractions": ComponentInfo(
        id="retractions",
        name="Retractions Workshop",
        one_liner="generate research questions from gaps re-opened by retracted papers",
        scope="any",
        interactive=False,
        entry_point="retractions",
        completion_suggestions=("gaps", "hypothesis", "theory", "sharing"),
    ),
    "theory": ComponentInfo(
        id="theory",
        name="Theory Workshop",
        one_liner="build explanatory frameworks through abduction, classification, and consilience",
        scope="any",
        interactive=True,
        entry_point="theory",
        completion_suggestions=("hypothesis", "gaps", "sharing"),
    ),
    "modeling": ComponentInfo(
        id="modeling",
        name="Modeling Workshop",
        one_liner=(
            "discover research questions investigable through"
            " computational modeling and simulation"
        ),
        scope="any",
        interactive=False,
        entry_point="modeling",
        completion_suggestions=("hypothesis", "theory", "forge", "sharing"),
    ),
    "forge": ComponentInfo(
        id="forge",
        name="Forge Workshop",
        one_liner="identify research workflow bottlenecks and design buildable tools",
        scope="natural_science",
        interactive=False,
        entry_point="forge",
        completion_suggestions=("hypothesis", "gaps", "sharing"),
    ),
    "sharing": ComponentInfo(
        id="sharing",
        name="Sharing Workshop",
        one_liner="writing support, venue discovery, and communication guidance",
        scope="post_research",
        interactive=True,
        entry_point="sharing",
        completion_suggestions=("hypothesis", "gaps", "phenomenon", "claims", "theory", "forge"),
    ),
}


# ---------------------------------------------------------------------------
# Pre-computed redirect phrases keyed by (from_id, to_id).
# Only pairs that make logical sense are included.
# ---------------------------------------------------------------------------

REDIRECT_PHRASES: dict[tuple[str, str], str] = {
    # --- Scope mismatches: hypothesis <-> gaps ---
    ("hypothesis", "gaps"): (
        "That sounds like a natural science topic! The Gaps workshop is designed for"
        " literature-driven fundamental question discovery \u2014 you can find it in the"
        " Gaps tab."
    ),
    ("gaps", "hypothesis"): (
        "That sounds like a causal relationship between variables \u2014 the Hypothesis"
        " workshop is built for developing X-causes-Y research designs. You can find it"
        " in the Hypothesis tab."
    ),
    # --- Scope mismatches: phenomenon -> hypothesis / gaps ---
    ("phenomenon", "hypothesis"): (
        "That seed idea sounds like a social science topic. The Hypothesis workshop is"
        " designed for causal research with X\u2192Y variables \u2014 try it in the"
        " Hypothesis tab."
    ),
    ("phenomenon", "gaps"): (
        "If you'd like to explore existing literature and discover open questions, the"
        " Gaps workshop might be a great fit \u2014 it guides you through literature"
        " analysis."
    ),
    ("phenomenon", "theory"): (
        "Interested in building an explanatory framework rather than testing a specific"
        " hypothesis? The Theory workshop guides you through abduction, classification,"
        " and consilience."
    ),
    # --- Reasoning mode: hypothesis <-> theory ---
    ("hypothesis", "theory"): (
        "It sounds like you want to build an explanatory framework rather than test a"
        " specific causal claim. The Theory workshop is designed for that \u2014"
        " abduction, classification, and consilience rather than hypothesis testing."
    ),
    ("theory", "hypothesis"): (
        "Your framework suggests a testable causal relationship. The Hypothesis workshop"
        " can help you turn that into a rigorous X\u2192Y experimental design."
    ),
    ("theory", "gaps"): (
        "Your framework points to areas where the literature is thin. The Gaps workshop"
        " can help you explore those gaps through systematic literature analysis."
    ),
    # --- Reasoning mode: gaps -> theory ---
    ("gaps", "theory"): (
        "The gap you found might need a new theoretical framework rather than a single"
        " experiment. The Theory workshop can help you build one through abduction and"
        " cross-domain synthesis."
    ),
    # --- Deeper exploration: questioned -> retractions ---
    ("questioned", "retractions"): (
        "Want to dig deeper into why that paper was questioned? The Retractions workshop"
        " guides you through failure analysis and helps you learn from methodological"
        " mistakes."
    ),
    # --- Deeper exploration: claims -> hypothesis / gaps ---
    ("claims", "hypothesis"): (
        "Want to develop that into a testable hypothesis? The Hypothesis workshop can"
        " guide you through causal variable identification and experimental design."
    ),
    ("claims", "gaps"): (
        "Curious about what the literature says? The Gaps workshop can help you find"
        " fundamental open questions through systematic literature analysis."
    ),
    # --- Deeper exploration: retractions -> gaps / hypothesis / theory ---
    ("retractions", "gaps"): (
        "That gap you spotted could become a real research question. The Gaps workshop"
        " can help you develop it through systematic literature analysis."
    ),
    ("retractions", "hypothesis"): (
        "Interesting \u2014 that could become a testable hypothesis. The Hypothesis"
        " workshop can help you design a causal study around that idea."
    ),
    ("retractions", "theory"): (
        "The failure pattern you identified might point to a deeper theoretical issue."
        " The Theory workshop can help you build a framework that accounts for it."
    ),
    # --- Completion transitions: any -> sharing ---
    ("hypothesis", "sharing"): (
        "Now that you have a research design, the Sharing workshop can help you write it"
        " up, find the right venue, and plan your communication strategy."
    ),
    ("gaps", "sharing"): (
        "You've identified a strong research question. The Sharing workshop can help you"
        " communicate your findings and find the right audience."
    ),
    ("phenomenon", "sharing"): (
        "Got your research problem? The Sharing workshop can help you plan how to"
        " communicate your work \u2014 from writing to venue selection."
    ),
    ("claims", "sharing"): (
        "Ready to share your research project? The Sharing workshop can help with"
        " writing support, venue discovery, and communication planning."
    ),
    ("retractions", "sharing"): (
        "If you're ready to communicate what you've learned, the Sharing workshop can"
        " help with writing, venue selection, and presentation strategy."
    ),
    ("theory", "sharing"): (
        "Your theory is taking shape. The Sharing workshop can help you write it up,"
        " find the right venue, and plan how to present your framework."
    ),
    # --- Sharing completion -> workshops (new questions emerged) ---
    ("sharing", "hypothesis"): (
        "That new question sounds like it involves a causal relationship. The Hypothesis"
        " workshop can help you develop it into a testable research design."
    ),
    ("sharing", "gaps"): (
        "That's a great new direction. The Gaps workshop can help you explore the"
        " literature and find where the open questions are."
    ),
    ("sharing", "phenomenon"): (
        "Want to generate a fresh natural science problem? The Phenomenon workshop can"
        " help you discover an interesting research question from scratch."
    ),
    ("sharing", "claims"): (
        "Spotted a claim worth investigating? The Claims workshop can help you turn it"
        " into a structured research project."
    ),
    ("sharing", "theory"): (
        "That sounds like it needs an explanatory framework. The Theory workshop can help"
        " you build one through abduction and cross-domain synthesis."
    ),
    # --- Office -> any workshop (contextual suggestions) ---
    ("office", "hypothesis"): (
        "It sounds like you're developing a causal research question. The Hypothesis"
        " workshop can walk you through the full process \u2014 from variables to"
        " experimental design."
    ),
    ("office", "gaps"): (
        "If you want to find open questions in the literature, the Gaps workshop guides"
        " you through systematic literature analysis to discover fundamental gaps."
    ),
    ("office", "phenomenon"): (
        "Looking for a natural science research problem? The Phenomenon workshop can"
        " generate interesting problems based on your interests."
    ),
    ("office", "claims"): (
        "Want to investigate a media claim? The Claims workshop turns news stories and"
        " popular claims into structured research projects."
    ),
    ("office", "questioned"): (
        "Curious about papers under scrutiny? The Questioned workshop surfaces research"
        " in your field that's being re-examined by the community."
    ),
    ("office", "retractions"): (
        "Interested in learning from scientific mistakes? The Retractions workshop guides"
        " you through analyzing retracted papers and understanding what went wrong."
    ),
    ("office", "theory"): (
        "It sounds like you want to build an explanatory framework. The Theory workshop"
        " guides you through abduction, classification, and consilience."
    ),
    ("office", "sharing"): (
        "Ready to communicate your research? The Sharing workshop helps with writing,"
        " finding the right venue, and planning presentations."
    ),
    ("office", "forge"): (
        "Want to build a tool for researchers? The Forge workshop identifies workflow"
        " bottlenecks and designs buildable tools \u2014 software, hardware, or protocols."
    ),
    # --- Forge connections ---
    ("forge", "hypothesis"): (
        "Have a tool idea you want to validate? The Hypothesis workshop can help you"
        " design a rigorous evaluation \u2014 does tool X improve Y compared to Z?"
    ),
    ("forge", "sharing"): (
        "Built your tool? The Sharing workshop can help you write it up as a methods"
        " paper and find the right venue \u2014 JOSS, HardwareX, or MethodsX."
    ),
    ("forge", "gaps"): (
        "Your tool addresses a gap in the field. The Gaps workshop can help you explore"
        " the broader landscape of open questions around it."
    ),
    ("phenomenon", "forge"): (
        "Interested in building a research tool rather than running an experiment?"
        " The Forge workshop designs buildable tools for scientific workflows."
    ),
    ("hypothesis", "forge"): (
        "Need a better measurement tool for your experiment? The Forge workshop can"
        " help you design and build one."
    ),
    ("gaps", "forge"): (
        "Noticed a capability gap rather than a knowledge gap? The Forge workshop"
        " designs buildable tools to address research workflow bottlenecks."
    ),
    ("sharing", "forge"): (
        "Want to contribute a research tool? The Forge workshop identifies workflow"
        " bottlenecks and designs buildable tools for your field."
    ),
    ("retractions", "forge"): (
        "Methodological failures often point to tool gaps. The Forge workshop can help"
        " you design a tool that prevents the kind of error you identified."
    ),
    ("theory", "forge"): (
        "Your framework suggests a need for new measurement or analysis tools."
        " The Forge workshop can help you design and build them."
    ),
    # --- Modeling connections ---
    ("office", "modeling"): (
        "It sounds like you want to explore 'what would happen if...' questions."
        " The Modeling workshop can generate research questions investigable"
        " through computational modeling and simulation."
    ),
    ("modeling", "hypothesis"): (
        "Your model predictions suggest a testable hypothesis. The Hypothesis"
        " workshop can help you design a rigorous experiment around it."
    ),
    ("modeling", "theory"): (
        "Your model's assumptions suggest a deeper theoretical question. The Theory"
        " workshop can help you build an explanatory framework."
    ),
    ("modeling", "forge"): (
        "Building the model might require new tools. The Forge workshop can help"
        " you design and build simulation tools for researchers."
    ),
    ("modeling", "sharing"): (
        "Your model is ready to communicate. The Sharing workshop can help you"
        " write it up and find the right venue."
    ),
    ("theory", "modeling"): (
        "Your framework could be formalized as a computational model. The Modeling"
        " workshop guides you through decomposition, specification, and validation."
    ),
    ("hypothesis", "modeling"): (
        "You might test that hypothesis with a computational model before running"
        " the experiment. The Modeling workshop can help you build one."
    ),
    ("gaps", "modeling"): (
        "That gap could be explored through computational modeling. The Modeling"
        " workshop helps you build and validate models of dynamic systems."
    ),
    ("retractions", "modeling"): (
        "The methodological failure might be investigated with a simulation. The"
        " Modeling workshop helps you build computational models and test assumptions."
    ),
    ("phenomenon", "modeling"): (
        "That phenomenon could be explored through computational modeling. The Modeling"
        " workshop helps you build a model and ask 'what if?' questions."
    ),
    ("sharing", "modeling"): (
        "Want to support your findings with a computational model? The Modeling"
        " workshop guides you through building and validating simulations."
    ),
    ("forge", "modeling"): (
        "Need to test your tool with a simulation? The Modeling workshop can help"
        " you build a computational model to validate your tool design."
    ),
}
