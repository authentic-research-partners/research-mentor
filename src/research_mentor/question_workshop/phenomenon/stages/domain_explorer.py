"""Stage 1: Domain Explorer

Single responsibility: Identify interesting scientific phenomena.

Given a domain or user suggestion, finds 2-3 phenomena worth investigating
and lists required materials. Supports physics, chemistry, and biology domains.

Pattern: Chat LLM + Structured Extraction
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.knowledge_base import (
    get_all_domain_ids,
    get_domain_description,
    get_domains_for_field,
)
from research_mentor.question_workshop.schemas import ExplorationData

DOMAIN_EXPLORATION_PROMPT = """You are a science educator helping to identify interesting phenomena for IYPT-style open-ended experimental problems.

**IMPORTANT:** While the examples below are from IYPT (physics), this EXACT format applies to chemistry and biology problems as well. The key is identifying observable, surprising, and measurable phenomena with open-ended investigation potential.

**Format works across all sciences:**
- Physics: Leidenfrost effect (droplet levitation on hot surface)
- Chemistry: Oscillating reactions (color changes in rhythmic cycles)
- Biology: Bacterial chemotaxis (cell migration toward nutrients)

Your task: Given a domain or user suggestion, identify 2-3 interesting phenomena that would make good open-ended experimental problems.

## IYPT Problem Characteristics

Good IYPT problems are:
- **Observable experimentally** - Can be demonstrated in a lab
- **Surprising or counterintuitive** - "Wow factor" that captures attention
- **Accessible to high school** - Materials and concepts within reach
- **Rich in physics** - Simple setup but complex physics underneath
- **Open-ended** - Multiple parameters to investigate, no single "right answer"
- **Measurable** - Variables can be quantified, not just observed

## Available Domains

{domains_list}

## Your Task

Identify 2-3 interesting phenomena and the materials needed.

**If given a specific domain:** Focus on that domain's phenomena.
**If given a user suggestion:** Identify which domain(s) it relates to and find similar phenomena.
**If given both:** Combine domain knowledge with user's idea.

For each phenomenon:
- Explain what makes it interesting (surprise factor)
- List observable effects
- Identify variables that can be measured
- Consider accessibility (can high school students do this?)

Be creative! Look for phenomena that are:
- Visually striking (creates patterns, motions, or transformations)
- Counterintuitive (defies common expectations)
- Relatable (students have seen similar things in daily life)
- Scalable (can be demonstrated with simple apparatus)

## PRIORITY: Understudied & Niche Areas

**IMPORTANT:** Prefer phenomena that are UNDERSTUDIED or in NICHE areas. High school projects can make genuine contributions where:

- **Edge cases** - Unusual parameter regimes that well-funded labs don't bother exploring (extreme temperatures, unusual materials, boundary conditions)
- **Non-ideal conditions** - Academia focuses on ideal conditions (cleaner data, more publishable). Real-world messy conditions are understudied
- **Low-hanging fruit** - Simple but unexplored variations (what if we use honey instead of water? what about non-spherical droplets?)
- **Cross-disciplinary gaps** - Phenomena at the intersection of fields that specialists overlook
- **"Too simple" for academia** - Effects that researchers consider beneath publication but aren't fully characterized
- **Practical utility gaps** - Questions that consumers, policymakers, or educators actually need answered but academia ignores
- **Local/regional phenomena** - Effects involving local materials, conditions, or species that global labs don't study
- **Replication gaps** - Classic experiments that were done once but never thoroughly replicated or extended

**Goal:** Meaningful novel contribution potential - not breakthrough research, but authentic investigation where the answer isn't already known. Nobody knows if it will work - that's real research.

**Avoid:** Well-funded hot topics (quantum computing, graphene, CRISPR mechanisms) where students can't meaningfully contribute. A student project on "graphene conductivity" will just repeat what billion-dollar labs have done better.

## Examples of Good IYPT Phenomena

**Leidenfrost Effect:**
- Phenomenon: Droplet levitates on hot surface without touching
- Surprise: Water floats on 300°C metal plate
- Observable: Droplet motion, lifetime, stability
- Materials: Water, metal plate, stove, thermometer
- Variables: Temperature, droplet size, surface texture

**Chladni Patterns:**
- Phenomenon: Sand forms intricate patterns on vibrating plate
- Surprise: Random sand organizes into beautiful geometric shapes
- Observable: Pattern formation, nodal lines
- Materials: Metal plate, sand, speaker, frequency generator
- Variables: Frequency, plate shape, sand type

**Kaye Effect:**
- Phenomenon: Liquid stream coils like a spring when poured
- Surprise: Viscous fluid behaves unexpectedly
- Observable: Coiling frequency, height
- Materials: Honey, shampoo, container, camera
- Variables: Viscosity, height, flow rate
"""


SIMULATION_EXPLORATION_PROMPT = """You are a science educator helping to identify interesting phenomena suitable for numerical simulation in Wolfram Mathematica.

**IMPORTANT:** Focus on phenomena that can be modeled mathematically and investigated through computational simulation. The key is identifying phenomena with clear mathematical models, rich parameter spaces, and compelling visualizations.

**Good simulation problems have:**
- **Clear mathematical model** - Can be described by ODEs, PDEs, optimization, or statistical models
- **Tractable in Mathematica** - Built-in solvers (NDSolve, DSolve, NMinimize, etc.) can handle the equations
- **Rich parameter space** - Multiple parameters to vary systematically (initial conditions, coefficients, boundary conditions)
- **Visual results** - Produces interesting plots, animations, or 3D visualizations
- **Physical intuition** - Connects mathematical results to real-world phenomena

**Format works across all sciences:**
- Physics: Double pendulum (chaotic motion from deterministic equations)
- Chemistry: Reaction-diffusion systems (pattern formation from coupled PDEs)
- Biology: Predator-prey dynamics (Lotka-Volterra oscillations)

## Available Domains

{domains_list}

## Simulation-Friendly Phenomena Examples

**Chaotic Double Pendulum:**
- Model: System of coupled second-order ODEs
- Mathematica: NDSolve with ParametricPlot animation
- Parameters: Mass ratios, length ratios, initial angles, damping
- Visualization: Phase space portraits, Poincaré sections, bifurcation diagrams
- Surprise: Tiny changes in initial conditions → completely different trajectories

**Heat Equation (Diffusion):**
- Model: 1D/2D parabolic PDE
- Mathematica: NDSolve with various boundary conditions
- Parameters: Diffusion coefficient, boundary conditions, initial temperature profile
- Visualization: Temperature evolution animation, steady-state patterns
- Surprise: How initial conditions "relax" to steady state

**Lotka-Volterra Predator-Prey:**
- Model: Coupled ODEs for population dynamics
- Mathematica: NDSolve with phase portraits
- Parameters: Growth rates, interaction coefficients, carrying capacity
- Visualization: Population cycles, phase space trajectories
- Surprise: Oscillations emerge from simple equations

**Damped Driven Oscillator:**
- Model: Second-order ODE with forcing
- Mathematica: NDSolve with frequency response
- Parameters: Damping coefficient, driving frequency, amplitude
- Visualization: Resonance curves, transient vs steady-state
- Surprise: Resonance phenomena, beating patterns

## Your Task

Identify 2-3 phenomena that would make excellent numerical simulation projects in Wolfram Mathematica.

For each phenomenon:
- What mathematical model describes it? (ODEs, PDEs, optimization)
- What Mathematica functions would solve it? (NDSolve, DSolve, NMinimize, etc.)
- What parameters can students vary?
- What visualizations would be compelling?
- What's the "surprise factor" that makes exploration worthwhile?

**IMPORTANT:** Ensure the mathematical complexity is appropriate for high school students who have basic calculus knowledge. Mathematica handles the numerical methods - students focus on understanding the physics/chemistry/biology.

## PRIORITY: Understudied & Niche Areas

**IMPORTANT:** Prefer systems that are UNDERSTUDIED or in NICHE parameter regimes. High school simulation projects can make genuine contributions where:

- **Unexplored parameter regimes** - What happens at extreme coupling strengths, unusual initial conditions, or boundary cases that papers don't cover?
- **Non-ideal conditions** - Academia models ideal cases. Add realistic complications: friction, noise, asymmetry, imperfect boundaries
- **Neglected model variations** - Add friction/damping/noise to classic models that are usually idealized
- **Cross-disciplinary gaps** - Apply well-known physics models to chemistry/biology contexts (or vice versa)
- **"Too simple" for journals** - Systematic parameter studies that researchers consider obvious but haven't actually done
- **Practical utility gaps** - Simulations that would help engineers, educators, or hobbyists but aren't academically prestigious
- **Numerical curiosities** - Strange behaviors in simulations that appear in specific parameter ranges
- **Model comparison** - Compare different mathematical models for the same phenomenon (which approximations matter?)

**Goal:** Meaningful novel contribution potential - not breakthrough research, but authentic investigation where the simulation results aren't already known. The parameter space is genuinely unexplored.

**Avoid:** Hot simulation topics where advanced researchers dominate (neural network architectures, climate models, protein folding) - students can't meaningfully contribute there.
"""


DATA_ANALYSIS_EXPLORATION_PROMPT = """You are a science educator helping to identify interesting datasets and statistical questions suitable for data analysis and machine learning projects.

**IMPORTANT:** Focus on questions that can be investigated through statistical analysis, data visualization, and potentially machine learning. The key is finding publicly available datasets with interesting patterns to discover.

**Good data analysis problems have:**
- **Accessible dataset** - Publicly available from reliable sources (Kaggle, government databases, scientific repositories)
- **Clear question** - Testable hypothesis or pattern to discover
- **Rich variables** - Multiple features to explore correlations, clusters, or predictions
- **Visual potential** - Meaningful charts, maps, or interactive visualizations
- **Real-world relevance** - Connects to current issues or student interests

**Format works across all sciences:**
- Physics: Exoplanet data (predicting habitability from orbital parameters)
- Chemistry: Drug compound properties (predicting solubility from molecular structure)
- Biology: Species distribution (correlating biodiversity with environmental factors)

## Available Domains

{domains_list}

## Data Analysis Project Examples

**Climate Temperature Analysis:**
- Dataset: NOAA historical temperature records
- Question: How do temperature anomalies correlate with CO2 levels?
- Methods: Time series analysis, correlation, trend fitting
- Visualization: Time series plots, scatter plots with regression, geographic heatmaps
- Surprise: Regional variations reveal complex patterns

**Particle Physics Classification:**
- Dataset: CERN open data (particle collision events)
- Question: Can we classify particle types from detector signatures?
- Methods: Classification algorithms, feature importance analysis
- Visualization: Confusion matrices, feature distributions, decision boundaries
- Surprise: Simple features achieve high classification accuracy

**Earthquake Pattern Analysis:**
- Dataset: USGS earthquake catalog
- Question: Are there temporal or spatial patterns in earthquake occurrence?
- Methods: Clustering, time series analysis, statistical tests
- Visualization: Geographic plots, frequency distributions, correlation matrices
- Surprise: Aftershock sequences follow predictable statistical laws

**Astronomical Object Classification:**
- Dataset: SDSS (Sloan Digital Sky Survey) spectroscopic data
- Question: Can we classify stars vs galaxies vs quasars from photometric data?
- Methods: Machine learning classification, dimensionality reduction
- Visualization: Color-color diagrams, t-SNE plots, feature importance
- Surprise: A few color indices contain most of the classification information

## Your Task

Identify 2-3 datasets and questions that would make excellent data analysis projects.

For each:
- What dataset is available? (source, size, format)
- What question can be investigated?
- What statistical or ML methods would apply?
- What visualizations would be meaningful?
- What's the "discovery factor" that makes exploration worthwhile?

**IMPORTANT:** Ensure datasets are freely accessible and appropriately sized for high school students. Focus on insight discovery, not just technical ML implementation.

## PRIORITY: Understudied & Niche Areas

**IMPORTANT:** Prefer datasets and questions that are UNDERSTUDIED. High school data analysis projects can make genuine contributions where:

- **Neglected datasets** - Public data that exists but nobody has analyzed thoroughly (local government data, old digitized records, niche scientific databases)
- **Messy real-world data** - Academia prefers clean datasets. Analyzing noisy, incomplete, or inconsistent data is understudied
- **Unusual correlations** - Questions nobody thought to ask (does moon phase correlate with traffic accidents? do local plants flower differently than global averages?)
- **Regional/local focus** - Data specific to your city, state, or region that global researchers don't care about
- **Practical utility gaps** - Analysis that consumers, policymakers, educators, or small businesses actually need but academia ignores
- **Time period gaps** - Historical data that hasn't been connected to modern datasets
- **Cross-dataset integration** - Combine two public datasets that haven't been linked before
- **Replication/validation** - Test whether published findings hold for different populations or time periods
- **Citizen science data** - Large volunteer-collected datasets (eBird, iNaturalist) with unexplored questions

**Goal:** Meaningful novel contribution potential - not breakthrough discoveries, but authentic analysis where the patterns aren't already documented. Your dataset combination or question is genuinely new.

**Avoid:** Hot ML topics (image recognition benchmarks, NLP leaderboards, stock prediction) where you're just reimplementing what tech companies do with vastly more resources.
"""


THEORETICAL_EXPLORATION_PROMPT = """You are a science educator helping to identify interesting phenomena suitable for mathematical/theoretical derivation using symbolic computation.

**IMPORTANT:** Focus on problems where students derive analytical solutions, prove relationships, or explore mathematical structures underlying physical phenomena. The key is finding problems with elegant mathematical treatments accessible to high school calculus.

**Good theoretical problems have:**
- **Analytical tractability** - Can be solved symbolically (not requiring numerical methods)
- **Physical insight** - Mathematical result reveals deep physical principle
- **Elegant mathematics** - Uses calculus, algebra, or geometry in satisfying ways
- **Verifiable predictions** - Derived results can be compared to known values or experiments
- **Extension potential** - Base case leads to interesting generalizations

**Format works across all sciences:**
- Physics: Derive period of simple pendulum from Lagrangian mechanics
- Chemistry: Derive ideal gas law from statistical mechanics assumptions
- Biology: Derive logistic growth curve and interpret parameters

## Available Domains

{domains_list}

## Theoretical Problem Examples

**Simple Harmonic Motion:**
- Goal: Derive period formula T = 2π√(m/k) from Newton's laws
- Mathematics: Second-order linear ODE, characteristic equation
- Tools: Mathematica's DSolve, symbolic manipulation
- Insight: Period independent of amplitude (for small oscillations)
- Extension: Add damping, add forcing, explore nonlinear regime

**Kepler's Laws from Newton:**
- Goal: Derive elliptical orbits from inverse-square force law
- Mathematics: Differential equations in polar coordinates, conservation laws
- Tools: Mathematica's DSolve, coordinate transformations
- Insight: Geometry of conic sections emerges from force law
- Extension: Explore precession, perturbations, three-body problem

**Blackbody Radiation:**
- Goal: Derive Planck distribution, Wien's law, Stefan-Boltzmann law
- Mathematics: Integration, series expansion, optimization
- Tools: Mathematica's Integrate, Series, FindMaximum
- Insight: Classical physics fails (ultraviolet catastrophe), quantum saves it
- Extension: Cosmic microwave background, star temperature estimation

**Wave Equation Solutions:**
- Goal: Derive standing wave patterns for vibrating string
- Mathematics: Separation of variables, Fourier series
- Tools: Mathematica's DSolve, eigenvalue problems
- Insight: Only discrete frequencies allowed (quantization from boundary conditions)
- Extension: 2D membranes, quantum particle in box

## Your Task

Identify 2-3 topics that would make excellent theoretical/derivation projects using symbolic computation.

For each:
- What is the goal of the derivation?
- What mathematical techniques are needed?
- What Mathematica symbolic functions would help? (DSolve, Integrate, Series, Simplify, etc.)
- What physical insight emerges from the mathematics?
- What extensions could students explore?

**IMPORTANT:** Ensure the mathematics is accessible to students with basic calculus (derivatives, integrals, simple ODEs). Mathematica handles symbolic manipulation - students focus on understanding the physics and interpreting results.

## PRIORITY: Understudied & Niche Areas

**IMPORTANT:** Prefer derivations and theoretical questions that are UNDERSTUDIED. High school theory projects can make genuine contributions where:

- **Missing intermediate cases** - Classic problems are solved for simple cases (spheres, infinite planes) but not realistic shapes
- **Non-ideal geometries** - Academia solves for perfect spheres, infinite planes, point masses. Real shapes are understudied
- **Approximation comparisons** - Which simplifying assumptions actually matter? Compare different approximation orders
- **Historical gaps** - Old papers derived results but didn't explore all consequences or parameter regimes
- **Cross-model connections** - Show how two different-looking equations describe the same phenomenon
- **Practical utility gaps** - Derivations that engineers, teachers, or hobbyists need but theorists consider trivial
- **Error propagation** - How do measurement uncertainties affect derived quantities? Often ignored in textbooks
- **Dimensional analysis surprises** - What non-obvious scaling laws emerge from the equations?
- **Limiting cases** - What happens when a parameter goes to zero or infinity? Often not explored thoroughly

**Goal:** Meaningful novel contribution potential - not breakthrough theorems, but authentic derivations where the specific result isn't already in textbooks. Your combination of assumptions or limiting case is genuinely unexplored.

**Avoid:** Cutting-edge theoretical physics (string theory, quantum gravity, advanced gauge theories) where the mathematics is beyond high school level and the field is saturated with experts.
"""

EXTRACTION_PROMPT = """Extract structured exploration data from the physics discussion.

Return:
- domains: 1-3 physics domains involved (use domain IDs like 'fluid_dynamics', 'mechanics', etc.)
- phenomena: 2-3 interesting phenomena identified (names/descriptions)
- materials: 3+ required materials for experiments
- feasibility_assessment: 'HIGH', 'MEDIUM', or 'LOW'
- surprise_factor: Brief description of what makes this surprising (optional)

Valid domain IDs: {valid_domains}
"""


async def explore_domain(
    field: str,
    problem_type: str,
    domain: str | None,
    user_suggestion: str | None,
    student_profile_text: str = "",
) -> dict[str, Any]:
    """
    Stage 1: Explore scientific domain and identify interesting phenomena.

    Single responsibility: Find 2-3 phenomena worth investigating.

    Args:
        field: Scientific field ('physics', 'chemistry', or 'biology')
        problem_type: Problem type:
            - 'experimental': Lab-based physical experiments
            - 'simulation': Numerical modeling (ODEs, PDEs, chaos)
            - 'data_analysis': Statistical analysis, ML on datasets
            - 'theoretical': Mathematical derivations, symbolic proofs
        domain: Optional domain within field (e.g., 'fluid_dynamics' for physics)
        user_suggestion: Optional user's custom idea (e.g., "spinning coin sounds")

    Returns:
        {
            "domains": list[str],  # ['fluid_dynamics', 'thermal_physics']
            "phenomena": list[str],  # ['Leidenfrost effect', 'droplet levitation']
            "materials": list[str],  # ['water', 'metal plate', 'thermometer'] OR ['Mathematica', 'NDSolve', ...]
            "feasibility_assessment": str,  # 'HIGH', 'MEDIUM', 'LOW'
            "surprise_factor": str | None,  # Optional description
            "raw_response": str  # Full LLM response for debugging
        }
    """
    logger.info(
        "Stage 1: Domain Exploration - field={}, problem_type={}, domain={}, suggestion={}",
        field, problem_type, domain, user_suggestion,
    )

    # Select prompt based on problem_type
    if problem_type == "simulation":
        base_prompt = SIMULATION_EXPLORATION_PROMPT
        problem_type_label = "numerical simulation (Wolfram Mathematica)"
    elif problem_type == "data_analysis":
        base_prompt = DATA_ANALYSIS_EXPLORATION_PROMPT
        problem_type_label = "data analysis (statistical/ML)"
    elif problem_type == "theoretical":
        base_prompt = THEORETICAL_EXPLORATION_PROMPT
        problem_type_label = "theoretical derivation (symbolic computation)"
    else:  # experimental (default)
        base_prompt = DOMAIN_EXPLORATION_PROMPT
        problem_type_label = "experimental (lab-based)"

    # Build context - use field-specific domains
    field_domains = get_domains_for_field(field)
    domains_list = "\n".join([
        f"- **{d['id']}**: {d['name']} - {d['description'][:80]}..."
        for d in field_domains
    ])

    # Build user prompt with field context
    field_capitalized = field.capitalize()

    if domain and user_suggestion:
        user_prompt_text = f"""Field: {field_capitalized}
Problem Type: {problem_type_label}
Domain: {domain}
User suggestion: "{user_suggestion}"

Please identify 2-3 interesting {field} phenomena related to this domain and suggestion, suitable for {problem_type_label} investigation."""
    elif domain:
        domain_info = get_domain_description(domain, field)
        if domain_info:
            user_prompt_text = f"""Field: {field_capitalized}
Problem Type: {problem_type_label}
Domain: {domain_info['name']}

Description: {domain_info['description']}

Example phenomena in this domain:
{chr(10).join(f'- {p}' for p in domain_info['phenomena_examples'][:3])}

Please identify 2-3 interesting phenomena in this domain suitable for {problem_type_label} investigation."""
        else:
            user_prompt_text = f"""Field: {field_capitalized}
Problem Type: {problem_type_label}
Domain: {domain}

Please identify 2-3 interesting phenomena in this {field} domain suitable for {problem_type_label} investigation."""
    elif user_suggestion:
        user_prompt_text = f"""Field: {field_capitalized}
Problem Type: {problem_type_label}
User suggestion: "{user_suggestion}"

Please identify which {field} domain(s) this relates to and find 2-3 interesting phenomena similar to this suggestion, suitable for {problem_type_label} investigation."""
    else:
        # No domain or suggestion - pick an interesting domain in the specified field
        user_prompt_text = f"""Field: {field_capitalized}
Problem Type: {problem_type_label}
No specific domain requested.

Please select an interesting {field} domain and identify 2-3 phenomena that would make engaging {problem_type_label} problems."""

    # Step 1: Generate exploration with chat LLM (creative, flexible)
    llm = get_chat_llm()

    prompt_text = base_prompt.format(domains_list=domains_list)
    if student_profile_text:
        prompt_text = f"{student_profile_text}\n\n{prompt_text}"
    system_msg = SystemMessage(content=prompt_text)
    user_msg = HumanMessage(content=user_prompt_text)

    response = await llm.ainvoke([system_msg, user_msg])
    raw_response = response.content

    logger.debug("Domain exploration response length: {} chars", len(raw_response))

    # Step 2: Extract structured data with structured_call (type-safe)
    extraction_system_msg = SystemMessage(content=EXTRACTION_PROMPT.format(
        valid_domains=", ".join(get_all_domain_ids(field))
    ))
    extraction_user_msg = HumanMessage(content=f"""Extract structured data from this exploration:

{raw_response}

Domain requested: {domain or 'None'}
User suggestion: {user_suggestion or 'None'}
""")

    exploration_data = await structured_call(
        ExplorationData, [extraction_system_msg, extraction_user_msg], thinking="high",
    )

    # Convert Pydantic model to dict
    result = {
        "domains": exploration_data.domains,
        "phenomena": exploration_data.phenomena,
        "materials": exploration_data.materials,
        "feasibility_assessment": exploration_data.feasibility_assessment,
        "surprise_factor": exploration_data.surprise_factor,
        "raw_response": raw_response,
    }

    phenomena = result["phenomena"]
    assert isinstance(phenomena, list)
    logger.info(
        "Stage 1 complete: domains={}, phenomena_count={}",
        result["domains"], len(phenomena),
    )

    return result
