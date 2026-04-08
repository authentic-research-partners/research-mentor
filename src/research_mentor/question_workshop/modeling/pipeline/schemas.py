"""Schemas for the Modeling question pipeline.

Each stage produces structured output validated by these models.
The final output is a set of modeling research questions with
approach, tools, feasibility, and validation grounding.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Modeling Approaches — for input selection and display
# ---------------------------------------------------------------------------

# Groups for UI display. Each group contains approach keys.
APPROACH_GROUPS: dict[str, dict[str, str | list[str]]] = {
    "equations": {
        "label": "Equations & Simulation",
        "approaches": ["numerical_simulation", "cfd", "fea"],
    },
    "agents": {
        "label": "Agents & Rules",
        "approaches": ["agent_based", "cellular_automata", "discrete_event"],
    },
    "probability": {
        "label": "Probability & Statistics",
        "approaches": ["monte_carlo", "system_dynamics"],
    },
    "molecular": {
        "label": "Molecular & Materials",
        "approaches": ["molecular_dynamics", "dft"],
    },
    "networks": {
        "label": "Networks",
        "approaches": ["network"],
    },
    "ai_ml": {
        "label": "AI / Machine Learning",
        "approaches": ["ml_based"],
    },
}


MODELING_APPROACHES: dict[str, dict[str, str]] = {
    "numerical_simulation": {
        "label": "Numerical Simulation",
        "summary": (
            "Simulate how systems change over time using math equations "
            "solved by a computer — population growth, weather, chemical "
            "reactions, and more."
        ),
        "description": (
            "Solve differential equations (ODEs/PDEs) computationally. "
            "Covers a vast range: population dynamics, chemical kinetics, "
            "heat transfer, wave propagation, chaos and nonlinear dynamics, "
            "climate/geophysics (ice sheet models, ocean circulation, "
            "radiative transfer), astrophysics, structural mechanics. "
            "Key decisions: discretization scheme, time step, boundary "
            "conditions, which terms to include."
        ),
        "examples": (
            "Lotka-Volterra predator-prey, SIR epidemic models, "
            "Lorenz attractor (chaos), reaction-diffusion patterns, "
            "shallow-ice approximation for glaciers, Navier-Stokes for "
            "ocean currents, radiative-convective climate models"
        ),
        "tools": "Mathematica, MATLAB, Python (SciPy), Julia, Octave, FEniCS",
        "question_constraints": (
            "For climate/geophysics questions: (1) use the correct stress "
            "balance — SIA for interior ice, SSA or Stokes for grounding "
            "lines and ice streams, (2) include scenario comparison "
            "(e.g., RCP 2.6 vs 8.5) or parameter sensitivity, "
            "(3) address grid resolution effects where relevant, "
            "(4) standard sliding laws relate stress to velocity "
            "(Weertman, Coulomb), NOT velocity to melt rate."
        ),
    },
    "agent_based": {
        "label": "Agent-Based Modeling",
        "summary": (
            "Give individual actors simple rules and watch complex "
            "behavior emerge — like birds flocking or diseases spreading."
        ),
        "description": (
            "Individual agents follow simple rules; complex behavior "
            "emerges from interactions. Segregation, flocking, traffic, "
            "disease spread on networks."
        ),
        "examples": (
            "Schelling segregation, bird flocking, ant foraging, "
            "market dynamics"
        ),
        "tools": "NetLogo, Mesa (Python), Repast, GAMA, AnyLogic",
    },
    "monte_carlo": {
        "label": "Monte Carlo / Stochastic Simulation",
        "summary": (
            "Run thousands of random scenarios to estimate how likely "
            "different outcomes are — from flood risk to genetic drift."
        ),
        "description": (
            "Run thousands of random scenarios to estimate probabilities "
            "and distributions. Risk analysis, genetic drift, flood "
            "probability, financial modeling."
        ),
        "examples": (
            "Flood risk estimation, option pricing, neutron transport, "
            "Bayesian inference"
        ),
        "tools": "Python, R, Julia, MATLAB, Stan",
    },
    "molecular_dynamics": {
        "label": "Molecular Dynamics",
        "summary": (
            "Simulate how atoms and molecules move and interact — "
            "predict how proteins fold, how materials behave, or how "
            "drugs bind to their targets."
        ),
        "description": (
            "Simulate atomic and molecular interactions using force fields "
            "(e.g., AMBER, CHARMM, OPLS). Atoms move according to Newton's "
            "equations with femtosecond time steps. Predicts material "
            "properties, protein conformations, membrane dynamics, "
            "solvation effects. Key decisions: force field choice, "
            "system size, simulation length, thermostat/barostat."
        ),
        "examples": (
            "Protein folding pathways in explicit water, lipid bilayer "
            "self-assembly, nanoparticle coating stability, polymer "
            "glass transition temperature"
        ),
        "tools": "GROMACS, LAMMPS, NAMD, OpenMM, AMBER",
        "question_constraints": (
            "MD questions MUST: (1) specify which force field to use "
            "(AMBER ff19SB, CHARMM36m, OPLS-AA), (2) use EXPLICIT solvent "
            "if studying solvent/salt effects — never implicit for ion "
            "interactions, (3) include comparison with experimental data "
            "(NMR, SAXS, CD, or kinetics), (4) address system size and "
            "simulation length feasibility."
        ),
    },
    "dft": {
        "label": "Density Functional Theory (DFT)",
        "summary": (
            "Calculate the electronic structure of materials and molecules "
            "from quantum mechanics — predict properties of new materials, "
            "catalysts, or drugs before making them in a lab."
        ),
        "description": (
            "Quantum mechanical electronic structure calculations using "
            "approximate functionals (PBE, B3LYP, HSE06). Two major "
            "categories: PERIODIC calculations for crystals/surfaces/slabs "
            "(use VASP, Quantum ESPRESSO, CP2K with plane-wave basis) and "
            "MOLECULAR calculations for isolated molecules/clusters "
            "(use Gaussian, ORCA with localized basis sets). Key decisions: "
            "functional choice, basis set, k-point sampling, supercell size."
        ),
        "examples": (
            "Band gap engineering via doping (periodic, VASP), catalytic "
            "surface adsorption energies (periodic, Quantum ESPRESSO), "
            "molecular orbital analysis (molecular, Gaussian), reaction "
            "barrier calculations (molecular, ORCA)"
        ),
        "tools": (
            "Periodic: VASP, Quantum ESPRESSO, CP2K. "
            "Molecular: Gaussian, ORCA. "
            "Do NOT use molecular codes for periodic systems or vice versa."
        ),
        "question_constraints": (
            "DFT questions MUST: (1) compare at least two dopants, "
            "compositions, or structural variants — not single-variable, "
            "(2) connect computed properties to experimental observables "
            "(photocatalytic activity, reaction rates, spectroscopic data), "
            "(3) address the band gap problem — specify whether using PBE, "
            "PBE+U, HSE06, or GW and WHY, (4) use periodic codes (VASP, QE) "
            "for crystals/surfaces — NEVER Gaussian/ORCA for periodic systems."
        ),
    },
    "cfd": {
        "label": "Computational Fluid Dynamics (CFD)",
        "summary": (
            "Simulate how fluids (air, water, blood) flow through and "
            "around things — from airplane wings to blood vessels."
        ),
        "description": (
            "Simulate fluid flow by solving Navier-Stokes equations. "
            "Aerodynamics, blood flow, ocean currents, combustion."
        ),
        "examples": (
            "Wing design optimization, indoor ventilation, river "
            "sediment transport"
        ),
        "tools": "OpenFOAM, ANSYS Fluent, COMSOL, Star-CCM+",
    },
    "fea": {
        "label": "Finite Element Analysis (FEA)",
        "summary": (
            "Test how structures handle stress, heat, or vibration "
            "in a computer — before building them in real life."
        ),
        "description": (
            "Structural, thermal, and electromagnetic analysis by "
            "dividing systems into discrete elements."
        ),
        "examples": (
            "Bridge stress analysis, heat sink design, vibration modes "
            "of a guitar body"
        ),
        "tools": "ANSYS, COMSOL, FEniCS, Abaqus, FreeFEM",
    },
    "ml_based": {
        "label": "AI/ML-Based Modeling",
        "summary": (
            "Use AI and machine learning to model systems that are "
            "too complex for traditional equations — let the data "
            "reveal the patterns."
        ),
        "description": (
            "Data-driven models including physics-informed neural "
            "networks (PINNs), surrogate models, symbolic regression. "
            "Prediction without full mechanistic understanding."
        ),
        "examples": (
            "Predicting protein structures from sequences, discovering "
            "hidden equations governing a physical system, forecasting "
            "turbulence from sensor data"
        ),
        "tools": "PyTorch, TensorFlow, JAX, scikit-learn, PySR",
    },
    "system_dynamics": {
        "label": "System Dynamics",
        "summary": (
            "Model systems with feedback loops — how populations grow, "
            "resources get depleted, or economies cycle between boom "
            "and bust."
        ),
        "description": (
            "Stock-and-flow models with feedback loops. Population "
            "dynamics, economic cycles, resource depletion, "
            "organizational behavior."
        ),
        "examples": (
            "Limits to Growth world model, supply chain bullwhip "
            "effect, epidemic policy delay"
        ),
        "tools": "Stella, Vensim, Insight Maker, PySD",
    },
    "network": {
        "label": "Network / Graph-Based Models",
        "summary": (
            "Model systems where connections matter — how diseases "
            "spread through social networks, how food webs collapse, "
            "or how information goes viral."
        ),
        "description": (
            "Model systems where structure matters: social networks, "
            "epidemics on networks, food webs, transportation."
        ),
        "examples": (
            "Epidemic spread on scale-free networks, information "
            "cascades, ecological food web stability"
        ),
        "tools": "NetworkX, igraph, Gephi, graph-tool",
    },
    "cellular_automata": {
        "label": "Cellular Automata",
        "summary": (
            "Simple rules on a grid that produce surprisingly complex "
            "patterns — like how forest fires spread or cities grow."
        ),
        "description": (
            "Grid-based rule systems where local rules produce global "
            "patterns. Forest fires, crystal growth, urban expansion."
        ),
        "examples": (
            "Game of Life, forest fire percolation, traffic flow "
            "(Nagel-Schreckenberg)"
        ),
        "tools": "NetLogo, Mathematica, custom implementations",
    },
    "discrete_event": {
        "label": "Discrete Event Simulation",
        "summary": (
            "Model systems where things happen one at a time — "
            "patients arriving at a hospital, packages moving through "
            "a warehouse, cars at an intersection."
        ),
        "description": (
            "Model systems as sequences of events. Queuing theory, "
            "manufacturing, hospital workflows, logistics."
        ),
        "examples": (
            "Emergency room wait times, factory scheduling, airport "
            "baggage handling"
        ),
        "tools": "SimPy, AnyLogic, Arena, Simul8",
    },
}


# ---------------------------------------------------------------------------
# Approach Fitness Filter
# ---------------------------------------------------------------------------


class ApproachFitness(BaseModel):
    """Whether a modeling approach makes sense for a given system."""

    approach: str = Field(max_length=50, description="The approach key being evaluated.")
    fits: bool = Field(
        description=(
            "TRUE if this approach can meaningfully model this system. "
            "FALSE if it makes no sense (e.g., DFT for traffic flow)."
        ),
    )
    reason: str = Field(
        max_length=300,
        description="One sentence explaining why it fits or doesn't.",
    )


class ApproachFitnessSet(BaseModel):
    """Fitness check results for multiple approaches."""

    results: list[ApproachFitness] = Field(
        description="One result per approach evaluated.",
    )


# ---------------------------------------------------------------------------
# Stage 1: System Exploration
# ---------------------------------------------------------------------------


class ModelingLandscape(BaseModel):
    """Published models and approaches found for the student's system."""

    published_models: list[str] = Field(
        description=(
            "Names of computational models found in published literature "
            "for this system (2-5 items). Include the model type."
        ),
    )
    key_variables: list[str] = Field(
        description="Key variables or quantities tracked in these models (3-6 items).",
    )
    simplification_tradeoffs: list[str] = Field(
        description=(
            "Important simplification decisions modelers face in this domain "
            "(2-4 items). What to include vs. ignore."
        ),
    )
    available_data: list[str] = Field(
        description=(
            "Public datasets or benchmarks that exist for validating models "
            "of this system (1-3 items). Empty if none known."
        ),
        default_factory=list,
    )
    domain_summary: str = Field(
        max_length=500,
        description="One-sentence overview of how computational modeling is used in this domain.",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "published_models": [
                "Lotka-Volterra (ODE predator-prey)",
                "Individual-based elk-wolf model (agent-based)",
            ],
            "key_variables": [
                "wolf population", "elk population",
                "predation rate", "carrying capacity",
            ],
            "simplification_tradeoffs": [
                "Spatial structure vs. well-mixed assumption",
                "Age structure vs. treating all individuals equally",
            ],
            "available_data": [
                "Yellowstone wolf-elk count time series (NPS)",
            ],
            "domain_summary": (
                "Predator-prey systems are modeled with ODEs (Lotka-Volterra) "
                "or agent-based simulations depending on spatial effects."
            ),
        },
    })


# ---------------------------------------------------------------------------
# Stage 2: Question Generation
# ---------------------------------------------------------------------------


class ModelingResearchQuestion(BaseModel):
    """A single modeling research question with context."""

    question: str = Field(
        max_length=500,
        description=(
            "The research question, phrased as 'What would happen if...?' "
            "or 'How does X affect Y in the model?' or similar."
        ),
    )
    approach: str = Field(
        max_length=50,
        description=(
            "Recommended modeling approach: numerical_simulation, agent_based, "
            "monte_carlo, molecular_dynamics, dft, cfd, fea, ml_based, "
            "system_dynamics, network, cellular_automata, discrete_event"
        ),
    )
    key_simplification: str = Field(
        max_length=300,
        description=(
            "The most important simplification decision the student will face "
            "when building this model. One sentence."
        ),
    )
    why_interesting: str = Field(
        max_length=300,
        description=(
            "Why this question is scientifically interesting — what might "
            "the model reveal that intuition alone cannot? One sentence."
        ),
    )
    suggested_tools: str = Field(
        max_length=200,
        description="1-2 suggested tools for implementation.",
    )


    model_config = ConfigDict(json_schema_extra={
        "example": {
            "question": (
                "How does adding spatial structure (wolves cluster "
                "near rivers) change the predicted coexistence "
                "region compared to a well-mixed model?"
            ),
            "approach": "agent_based",
            "key_simplification": (
                "Whether to model space explicitly or assume "
                "all individuals mix randomly."
            ),
            "why_interesting": (
                "Spatial clustering can stabilize predator-prey "
                "systems that are unstable in well-mixed models."
            ),
            "suggested_tools": "NetLogo or Mesa (Python)",
        },
    })


# ---------------------------------------------------------------------------
# Stage 3: Feasibility Assessment
# ---------------------------------------------------------------------------


class QuestionFeasibility(BaseModel):
    """Feasibility assessment for a modeling question."""

    computational_feasibility: int = Field(
        ge=1, le=10,
        description="How feasible computationally (10=laptop, 1=supercomputer).",
    )
    data_availability: int = Field(
        ge=1, le=10,
        description="Validation data availability (10=public datasets, 1=none exist).",
    )
    student_accessibility: int = Field(
        ge=1, le=10,
        description=(
            "Accessible to this student level (10=straightforward, "
            "1=requires graduate training)."
        ),
    )
    scientific_value: int = Field(
        ge=1, le=10,
        description=(
            "Scientific value of the question (10=novel insight likely, "
            "1=well-known result)."
        ),
    )
    overall_score: int = Field(
        ge=1, le=10,
        description="Overall recommendation score.",
    )


