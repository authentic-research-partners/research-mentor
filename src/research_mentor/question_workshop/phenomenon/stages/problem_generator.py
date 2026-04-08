"""Stage 2: Problem Generator

Single responsibility: Formulate IYPT-style problem.

Given exploration data (phenomena, materials), creates:
- Title (2-4 words, catchy)
- Description (2-4 sentences, setup + phenomenon)
- Investigation (starts with "Investigate...", parameter-focused)

Pattern: Chat LLM + Structured Extraction
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.schemas import ProblemComponents

PROBLEM_GENERATION_PROMPT = """You are an expert at creating IYPT-style open-ended experimental problems.

**IMPORTANT:** While the examples below are from IYPT (physics), this EXACT format applies to chemistry and biology problems as well. The key is the open-ended investigation structure ("Investigate how X depends on Y..."), not the specific domain.

**Format works across all sciences:**
- Physics: "Investigate how droplet velocity depends on surface temperature..."
- Chemistry: "Investigate how reaction rate depends on catalyst concentration..."
- Biology: "Investigate how bacterial growth depends on nutrient gradient..."

Your task: Create an IYPT-style problem based on scientific phenomena exploration.

## IYPT Problem Structure

### 1. TITLE (2-4 words):
- **Catchy and memorable**
- Hints at the phenomenon
- Evocative, not generic
- Examples: "Vapor Maze", "Whistling Mesh", "Dancing Slinky", "Autumn Coin", "Sweet Monochromator"

**Bad examples:**
- "Fluid Dynamics Experiment" (too generic)
- "Investigation of Heat Transfer" (too formal)
- "The Effect of Temperature on Droplet Behavior" (too long)

### 2. DESCRIPTION (2-4 sentences):
- **Sentence 1:** Describe the experimental setup using simple, accessible terms
- **Sentence 2:** Describe what happens (the phenomenon to observe)
- **Sentence 3 (optional):** Mention a specific surprising aspect or variation

**Requirements:**
- Concise and clear
- Everyday language, not overly technical
- Evoke curiosity and surprise
- Describe WHAT to do and WHAT happens

**Example:**
"Drop a tiny water droplet onto a scorching metal maze. Watch as it magically glides, levitated by its own vapor cushion. Can you guide the droplet through the intricate pathways without it touching the metal?"

### 3. INVESTIGATION DIRECTIVE (1-2 sentences):
- **Must start with:** "Investigate...", "Study...", "Explore...", or "Examine..."
- Ask about parameter dependencies: "how does X depend on Y?"
- Open-ended (no single answer)
- Focus on measurable variables

**Good examples:**
- "Investigate how the maze geometry, surface temperature, and droplet size affect the droplet's velocity, stability, and success rate."
- "Study how the mesh size, water flow rate, and pressure affect the frequency and amplitude of the sound produced."
- "Explore the conditions under which the Leidenfrost effect occurs and how it depends on surface material and temperature."

**Bad examples:**
- "What causes this effect?" (not parameter-focused)
- "Explain the phenomenon" (too vague)
- "Measure the temperature" (too narrow, not open-ended)

## IYPT Problem Examples

**Paper Boomerang**
Create a returning boomerang from a paper sheet. Investigate how the motion depends on the folding and cutting parameters of the paper.

**Whistling Mesh**
When a stream of water hits a metal mesh, a clear sound can be generated. Investigate the parameters that affect the sound and explain your observations.

**Lato Lato**
Two balls are attached to the ends of a string. The string is attached to a pivot that is moved up and down. As a result, the balls start hitting one another. Examine the system and investigate how the oscillation amplitude increases.

**Ring Fountain**
When a metal ring falls onto the surface of water, a fountain may be produced. Investigate the relationship between the relevant parameters and the height of the fountain produced.

## Your Task

Based on the exploration data provided, create an IYPT problem that is:
- **Accessible**: Uses materials high school students can obtain
- **Surprising**: Shows counterintuitive or visually striking effects
- **Deep**: Simple setup but complex physics underneath
- **Open-ended**: Multiple parameters to investigate
- **Experimental**: Can be tested, not purely theoretical

Be creative! Make the title memorable and the description engaging.
"""


SIMULATION_GENERATION_PROMPT = """You are an expert at creating numerical simulation problems for Wolfram Mathematica.

**IMPORTANT:** Focus on problems that can be investigated through computational simulation. The key is identifying phenomena with clear mathematical models that produce interesting visualizations and parameter studies.

**Format works across all sciences:**
- Physics: "Investigate how pendulum behavior depends on damping coefficient using phase portraits..."
- Chemistry: "Investigate how reaction pattern depends on diffusion rates using animated concentration plots..."
- Biology: "Investigate how population dynamics depend on growth parameters using phase space trajectories..."

Your task: Create a simulation-based problem based on scientific phenomena exploration.

## Simulation Problem Structure

### 1. TITLE (2-4 words):
- **Catchy and memorable**
- Hints at the phenomenon or mathematical structure
- Evocative, not generic
- Examples: "Chaotic Swing", "Diffusion Patterns", "Population Waves", "Resonance Dance", "Thermal Flow"

**Bad examples:**
- "ODE Simulation" (too technical)
- "Mathematical Model of Heat" (too generic)
- "Numerical Solution Investigation" (too formal)

### 2. DESCRIPTION (2-4 sentences):
- **Sentence 1:** Describe the physical/chemical/biological phenomenon and its mathematical model
- **Sentence 2:** Describe what emerges from simulation (patterns, behaviors, transitions)
- **Sentence 3 (optional):** Mention surprising aspects or parameter sensitivity

**Requirements:**
- Reference the mathematical structure (ODEs, PDEs, optimization)
- Mention what Mathematica visualization would show
- Connect math to physical intuition
- Evoke curiosity about parameter dependencies

**Example:**
"A double pendulum follows simple equations of motion, yet produces wildly unpredictable chaos. Use Mathematica to simulate this system and watch how tiny changes in initial conditions create dramatically different trajectories. Can you find the boundary between ordered and chaotic motion?"

### 3. INVESTIGATION DIRECTIVE (1-2 sentences):
- **Must start with:** "Investigate...", "Study...", "Explore...", or "Examine..."
- Ask about parameter dependencies through simulation
- Reference visualization types (phase portraits, animations, bifurcation diagrams)
- Focus on computational exploration

**Good examples:**
- "Investigate how the mass ratio and initial angle affect the onset of chaos, using phase space portraits and Poincare sections in Mathematica."
- "Study how diffusion coefficient and boundary conditions affect pattern formation, visualizing the time evolution with animated density plots."
- "Explore how predator-prey parameters affect oscillation amplitude and period, using phase portraits and parametric sweeps."

**Bad examples:**
- "Solve the equations" (not exploratory)
- "Calculate the result" (too narrow)
- "Explain the mathematics" (not parameter-focused)

## Simulation Problem Examples

**Chaotic Pendulum**
A double pendulum exhibits chaotic motion despite following deterministic equations. Use Mathematica's NDSolve to simulate the system and create phase space animations. Investigate how the mass ratio, length ratio, and initial conditions affect the transition to chaos.

**Reaction-Diffusion Patterns**
Chemical reaction-diffusion systems can spontaneously form stripes, spots, and spirals from uniform initial conditions. Model a Turing pattern system using PDEs in Mathematica. Study how the diffusion ratio and reaction parameters control which patterns emerge.

**Resonance Explorer**
A driven damped oscillator shows dramatic resonance behavior near its natural frequency. Simulate the system with varying driving frequencies and create frequency response curves. Investigate how damping coefficient affects resonance peak height and width.

**Heat Flow Visualization**
Temperature in a conducting material evolves according to the heat equation. Model heat conduction in various geometries using Mathematica's PDE solvers. Explore how boundary conditions and material properties affect the approach to thermal equilibrium.

## Your Task

Based on the exploration data provided, create a simulation problem that is:
- **Mathematically tractable**: Has clear ODE/PDE/optimization formulation
- **Visually compelling**: Produces interesting plots, animations, or 3D surfaces
- **Parameter-rich**: Multiple parameters to explore systematically
- **Physically intuitive**: Connects mathematical results to real-world understanding
- **Appropriately complex**: Challenging but accessible to students with basic calculus

Make the title memorable and emphasize the computational exploration aspect!
"""


DATA_ANALYSIS_GENERATION_PROMPT = """You are an expert at creating data analysis and statistical investigation problems.

**IMPORTANT:** Focus on problems that can be investigated through statistical analysis, data visualization, and potentially machine learning. The key is identifying interesting patterns in publicly available datasets.

**Format works across all sciences:**
- Physics: "Investigate how exoplanet properties correlate with orbital distance using Kepler mission data..."
- Chemistry: "Investigate how molecular properties predict drug efficacy using compound databases..."
- Biology: "Investigate how environmental factors correlate with species diversity using ecological surveys..."

Your task: Create a data analysis problem based on scientific phenomena exploration.

## Data Analysis Problem Structure

### 1. TITLE (2-4 words):
- **Catchy and memorable**
- Hints at the dataset or pattern to discover
- Evocative, not generic
- Examples: "Star Hunters", "Climate Patterns", "Gene Networks", "Earthquake Atlas", "Drug Discovery"

**Bad examples:**
- "Data Analysis Project" (too generic)
- "Statistical Investigation of Variables" (too formal)
- "Machine Learning Classification" (too technical)

### 2. DESCRIPTION (2-4 sentences):
- **Sentence 1:** Describe the dataset and what it contains
- **Sentence 2:** Describe the scientific question or pattern to investigate
- **Sentence 3 (optional):** Mention what surprising insights might emerge

**Requirements:**
- Name the specific dataset or data source
- Clearly state the scientific question
- Mention the type of analysis (correlation, classification, clustering, etc.)
- Connect to real-world significance

**Example:**
"NASA's Kepler mission has discovered thousands of exoplanets with detailed orbital and stellar data. Analyze this treasure trove to predict which planets might be habitable. Can you find patterns that astronomers might have missed?"

### 3. INVESTIGATION DIRECTIVE (1-2 sentences):
- **Must start with:** "Investigate...", "Study...", "Explore...", or "Examine..."
- Ask about correlations, predictions, or pattern discovery
- Reference visualization types (scatter plots, heatmaps, clustering diagrams)
- Focus on insight discovery

**Good examples:**
- "Investigate how planetary radius and orbital period correlate with stellar properties, using scatter plots and regression analysis."
- "Study how atmospheric CO2 levels relate to global temperature anomalies over time, visualizing trends and testing for correlation significance."
- "Explore whether earthquake magnitude can be predicted from precursor seismic activity, using time series analysis and feature engineering."

**Bad examples:**
- "Analyze the data" (too vague)
- "Run machine learning" (not insight-focused)
- "Calculate statistics" (too narrow)

## Data Analysis Problem Examples

**Stellar Census**
The Gaia space telescope has mapped over a billion stars with precise positions and velocities. Analyze the Milky Way's structure using this stellar census. Investigate how star density, velocity dispersion, and metallicity vary across different galactic regions.

**Climate Correlations**
NOAA maintains decades of global temperature, precipitation, and atmospheric data. Explore how climate variables correlate across space and time. Study whether local temperature anomalies can predict regional weather patterns.

**Biodiversity Predictors**
The Global Biodiversity Information Facility contains millions of species occurrence records. Analyze what environmental factors best predict species richness. Investigate how habitat fragmentation, climate, and human activity correlate with biodiversity hotspots.

**Compound Properties**
ChEMBL contains millions of bioactive compounds with measured properties. Use molecular descriptors to predict drug-likeness or toxicity. Explore which chemical features most strongly predict biological activity.

## Your Task

Based on the exploration data provided, create a data analysis problem that is:
- **Data accessible**: Uses freely available, well-documented datasets
- **Question-driven**: Has a clear scientific question to answer
- **Analytically rich**: Supports multiple statistical or ML approaches
- **Visually meaningful**: Produces interpretable charts and visualizations
- **Insight-focused**: Aims for scientific discovery, not just technical metrics

Make the title memorable and emphasize the discovery aspect!
"""


THEORETICAL_GENERATION_PROMPT = """You are an expert at creating theoretical derivation problems using symbolic computation.

**IMPORTANT:** Focus on problems where students derive analytical solutions, prove relationships, or explore mathematical structures. The key is finding elegant mathematical treatments with physical insight.

**Format works across all sciences:**
- Physics: "Derive the period formula for a simple pendulum from first principles..."
- Chemistry: "Derive the ideal gas law from statistical mechanics assumptions..."
- Biology: "Derive the logistic growth equation and interpret its parameters..."

Your task: Create a theoretical derivation problem based on scientific phenomena exploration.

## Theoretical Problem Structure

### 1. TITLE (2-4 words):
- **Catchy and memorable**
- Hints at the physical principle or mathematical elegance
- Evocative, not generic
- Examples: "Harmonic Secrets", "Orbital Dance", "Wave Mechanics", "Energy Flow", "Quantum Steps"

**Bad examples:**
- "Derivation Problem" (too generic)
- "Mathematical Proof" (too formal)
- "Symbolic Computation Exercise" (too technical)

### 2. DESCRIPTION (2-4 sentences):
- **Sentence 1:** Describe the physical phenomenon and what you're trying to derive
- **Sentence 2:** Describe the mathematical approach and key steps
- **Sentence 3 (optional):** Mention what physical insight the derivation reveals

**Requirements:**
- Clearly state what is being derived (formula, relationship, behavior)
- Mention the mathematical techniques involved
- Connect the mathematics to physical understanding
- Highlight the elegance or surprise in the result

**Example:**
"The simple pendulum seems straightforward, but deriving its period formula reveals deep connections between geometry and dynamics. Use Lagrangian mechanics and symbolic computation to derive T = 2pi*sqrt(L/g). What happens when you go beyond the small-angle approximation?"

### 3. INVESTIGATION DIRECTIVE (1-2 sentences):
- **Must start with:** "Derive...", "Prove...", "Show...", or "Establish..."
- Ask for derivation of specific relationships or formulas
- Reference mathematical techniques (calculus, ODEs, series expansion)
- Focus on understanding, not just calculation

**Good examples:**
- "Derive the relationship between period and length for a simple pendulum using Lagrangian mechanics, and explore how the result changes for large amplitude oscillations."
- "Prove that elliptical orbits follow from Newton's inverse-square law using conservation of angular momentum and energy, expressing the result in terms of orbital parameters."
- "Show that standing wave patterns on a vibrating string are quantized by boundary conditions, deriving the allowed frequencies and mode shapes."

**Bad examples:**
- "Calculate the answer" (not derivation-focused)
- "Solve the equation" (too narrow)
- "Use Mathematica" (process, not goal)

## Theoretical Problem Examples

**Pendulum Secrets**
The simple pendulum has been studied for centuries, yet its mathematics reveals surprising depth. Starting from Newton's laws or Lagrangian mechanics, derive the period formula and explore its dependence on amplitude. Prove that the period is independent of amplitude for small oscillations, then investigate the corrections needed for large swings.

**Orbital Mechanics**
Kepler discovered that planets move in ellipses, but Newton explained why. Starting from the inverse-square gravitational force law, derive that bound orbits are ellipses with the Sun at one focus. Show how energy and angular momentum determine the shape of the orbit.

**Wave Quantization**
A vibrating string produces musical notes at specific frequencies. Derive why only certain frequencies are allowed by solving the wave equation with fixed boundary conditions. Show that the allowed frequencies form a harmonic series and explain the connection to musical harmony.

**Thermal Equilibrium**
Heat flows from hot to cold, but what determines the final temperature distribution? Derive the steady-state solution of the heat equation for simple geometries. Show how boundary conditions uniquely determine the equilibrium temperature field.

## Your Task

Based on the exploration data provided, create a theoretical problem that is:
- **Analytically tractable**: Can be solved symbolically with calculus and basic ODEs
- **Physically insightful**: Reveals deep principles, not just formulas
- **Mathematically elegant**: Uses satisfying mathematical techniques
- **Extensible**: Has natural extensions for further exploration
- **Appropriately scoped**: Achievable by students with AP Calculus (NOT university math)

## MATHEMATICAL COMPLEXITY CONSTRAINT (CRITICAL)

The target audience has AP Calculus (basic derivatives, integrals, first-order ODEs).

**ALLOWED math:**
- Single-variable calculus (derivatives, integrals)
- First-order ODEs (separable, linear)
- Basic series expansions (Taylor/Maclaurin)
- Trigonometry and algebra
- Simple vector operations

**FORBIDDEN math (will make the problem infeasible):**
- Partial differential equations (PDEs)
- Bessel functions, Legendre polynomials, special functions
- Sturm-Liouville theory, eigenvalue problems
- Complex analysis, contour integrals
- Tensor calculus, differential geometry
- Variable-coefficient ODEs requiring Frobenius method

If your description includes a formula, verify it is CONSISTENT with the \
prose. Do not write "stiffer center" if the formula places max stiffness at \
the edge.

Make the title memorable and emphasize the elegance of the derivation!
"""

EXTRACTION_PROMPT = """Extract structured problem components from the IYPT-style problem formulation.

Return:
- title: 2-4 words, catchy problem title (max 50 characters)
- description: 2-4 sentences describing setup and phenomenon (max 500 characters - BE CONCISE)
- investigation: Investigation directive starting with "Investigate...", "Study...", "Explore...", or "Examine..." (max 300 characters - BE CONCISE)
- core_concepts: 2-6 core scientific concepts involved (physics/chemistry/biology)

CRITICAL CHARACTER LIMITS - responses will fail validation if exceeded:
- title: max 50 characters
- description: max 500 characters (2-3 SHORT sentences, ~80-100 words)
- investigation: max 300 characters (1-2 SHORT sentences, ~40-50 words)

Validation:
- Title must be 2-4 words (not a full sentence)
- Investigation must start with action verb
- Description must be clear and engaging
- ALL fields must respect character limits above
"""


async def generate_problem(
    field: str,
    problem_type: str,
    exploration_data: dict[str, Any],
    student_profile_text: str = "",
) -> dict[str, Any]:
    """Stage 2: Generate problem from exploration data.

    Single responsibility: Formulate problem components (title, description, investigation).

    Args:
        field: Scientific field ('physics', 'chemistry', 'biology')
        problem_type: Problem type:
            - 'experimental': Lab-based physical experiments
            - 'simulation': Numerical modeling (ODEs, PDEs, chaos)
            - 'data_analysis': Statistical analysis, ML on datasets
            - 'theoretical': Mathematical derivations, symbolic proofs
        exploration_data: Output from domain_explorer containing:
            - domains: list[str]
            - phenomena: list[str]
            - materials: list[str]
            - feasibility_assessment: str
            - surprise_factor: str | None

    Returns:
        {
            "title": str,  # "Vapor Maze" or "Chaotic Swing"
            "description": str,  # "Drop a tiny water droplet..." or "A double pendulum..."
            "investigation": str,  # "Investigate how..."
            "core_concepts": list[str],  # ['Leidenfrost effect', 'fluid dynamics']
            "raw_response": str  # Full LLM response for debugging
        }
    """
    logger.info(f"Stage 2: Problem Generation - field={field}, problem_type={problem_type}")

    # Select prompt based on problem_type
    if problem_type == "simulation":
        base_prompt = SIMULATION_GENERATION_PROMPT
        problem_type_label = "numerical simulation (Wolfram Mathematica)"
    elif problem_type == "data_analysis":
        base_prompt = DATA_ANALYSIS_GENERATION_PROMPT
        problem_type_label = "data analysis (statistical/ML)"
    elif problem_type == "theoretical":
        base_prompt = THEORETICAL_GENERATION_PROMPT
        problem_type_label = "theoretical derivation (symbolic computation)"
    else:  # experimental (default)
        base_prompt = PROBLEM_GENERATION_PROMPT
        problem_type_label = "experimental (lab-based)"

    # Build context from exploration (wrap in delimiters — content originates
    # from user suggestions and prior LLM output, both untrusted for prompt injection)
    domains = ", ".join(exploration_data.get("domains", []))
    phenomena_raw = "\n".join(f"- {p}" for p in exploration_data.get("phenomena", []))
    phenomena = f"<user_content>\n{phenomena_raw}\n</user_content>" if phenomena_raw else ""
    materials = ", ".join(exploration_data.get("materials", []))
    surprise_factor = exploration_data.get("surprise_factor", "")

    # Build user prompt based on problem_type
    if problem_type == "simulation":
        user_prompt_text = f"""Based on this {field} exploration, create a numerical simulation problem for Wolfram Mathematica:

**Field:** {field.capitalize()}
**Problem Type:** {problem_type_label}

**Domain(s):** {domains}

**Interesting Phenomena Identified:**
{phenomena}

**Mathematical Tools/Methods:** {materials}

**Surprise Factor:** {surprise_factor if surprise_factor else 'Look for counterintuitive behavior or emergent patterns!'}

**Task:** Create a complete {field} simulation problem following the format shown above:
1. A catchy 2-4 word title
2. A 2-4 sentence description (mathematical model + what simulation reveals)
3. An investigation directive starting with "Investigate..." that asks about parameter dependencies through simulation

Make it engaging, mathematically tractable, and visually compelling!
"""
    elif problem_type == "data_analysis":
        user_prompt_text = f"""Based on this {field} exploration, create a data analysis problem:

**Field:** {field.capitalize()}
**Problem Type:** {problem_type_label}

**Domain(s):** {domains}

**Interesting Phenomena/Questions Identified:**
{phenomena}

**Data Sources/Methods:** {materials}

**Discovery Potential:** {surprise_factor if surprise_factor else 'Look for hidden patterns and surprising correlations!'}

**Task:** Create a complete {field} data analysis problem following the format shown above:
1. A catchy 2-4 word title
2. A 2-4 sentence description (dataset + scientific question)
3. An investigation directive starting with "Investigate..." that asks about correlations, patterns, or predictions

Make it engaging, data-accessible, and insight-focused!
"""
    elif problem_type == "theoretical":
        user_prompt_text = f"""Based on this {field} exploration, create a theoretical derivation problem:

**Field:** {field.capitalize()}
**Problem Type:** {problem_type_label}

**Domain(s):** {domains}

**Interesting Phenomena/Principles Identified:**
{phenomena}

**Mathematical Tools/Methods:** {materials}

**Physical Insight:** {surprise_factor if surprise_factor else 'Look for elegant mathematical structures with deep physical meaning!'}

**Task:** Create a complete {field} theoretical problem following the format shown above:
1. A catchy 2-4 word title
2. A 2-4 sentence description (what to derive + mathematical approach)
3. An investigation directive starting with "Derive...", "Prove...", or "Show..." that asks for specific derivations

Make it engaging, mathematically elegant, and physically insightful!
"""
    else:  # experimental (default)
        user_prompt_text = f"""Based on this {field} exploration, create an IYPT-style experimental problem:

**Field:** {field.capitalize()}
**Problem Type:** {problem_type_label}

**Domain(s):** {domains}

**Interesting Phenomena Identified:**
{phenomena}

**Available Materials:** {materials}

**Surprise Factor:** {surprise_factor if surprise_factor else 'Create something counterintuitive!'}

**Task:** Create a complete {field} problem following the IYPT format shown above:
1. A catchy 2-4 word title
2. A 2-4 sentence description (setup + what happens)
3. An investigation directive starting with "Investigate..." that asks about parameter dependencies

Make it engaging, accessible, and surprising!
"""

    # Step 1: Generate problem with chat LLM (creative, flexible)
    llm = get_chat_llm()

    prompt_text = f"{student_profile_text}\n\n{base_prompt}" if student_profile_text else base_prompt
    system_msg = SystemMessage(content=prompt_text)
    user_msg = HumanMessage(content=user_prompt_text)

    response = await llm.ainvoke([system_msg, user_msg])
    raw_response = response.content

    logger.debug(f"Problem generation response length: {len(raw_response)} chars")

    # Step 2: Extract structured components with structured_call (type-safe)
    extraction_messages = [
        SystemMessage(content=EXTRACTION_PROMPT),
        HumanMessage(content=f"""Extract structured problem components from this IYPT problem:

{raw_response}

STRICT REQUIREMENTS:
- Title: 2-4 words, max 50 characters
- Description: 2-3 SHORT sentences, max 500 characters (condense if needed)
- Investigation: 1-2 SHORT sentences starting with action verb, max 300 characters (condense if needed)
- If content is too long, SHORTEN it while preserving the core meaning
"""),
    ]

    problem_components = await structured_call(
        ProblemComponents, extraction_messages, thinking="medium",
    )

    # Convert Pydantic model to dict
    result = {
        "title": problem_components.title,
        "description": problem_components.description,
        "investigation": problem_components.investigation,
        "core_concepts": problem_components.core_concepts,
        "raw_response": raw_response,
    }

    logger.info(
        f"Stage 2 complete: title='{result['title']}', "
        f"concepts_count={len(result['core_concepts'])}"
    )

    return result
