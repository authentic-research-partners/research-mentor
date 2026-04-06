"""Stage 3: Feasibility Validator

Single responsibility: Assess problem feasibility.

For experimental problems: Evaluates whether high school students can actually do this experiment.
For wolfram problems: Evaluates whether the problem is tractable for numerical simulation.

Returns 5 scores (1-10 scale) + safety/tractability + recommendation.

Pattern: Structured LLM (direct extraction, no chat)
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from research_mentor.llm import structured_call
from research_mentor.question_workshop.schemas import FeasibilityScores

FEASIBILITY_VALIDATION_PROMPT = """You are an expert in evaluating IYPT (International Young Physicists' Tournament) problems for high school students.

Your task: Assess whether this problem is feasible for high school students to complete in 3-6 months with school lab resources.

## Evaluation Criteria (1-10 scale, 10 = best)

### 1. Material Accessibility (1-10)
**Question:** Can a high school lab obtain these materials?

**Scoring:**
- **10:** Common household items (water, paper, coins, magnets)
- **7-9:** Standard school lab equipment (beakers, thermometers, multimeter, springs)
- **4-6:** Specialized but accessible (oscilloscope, high-speed camera, frequency generator)
  - Can order online for <$500
  - May need special request from school
- **1-3:** Rare/expensive equipment (laser interferometer, vacuum chamber, thermal camera)
  - Requires university-level equipment
  - Cost >$1000 or requires special facilities

**Consider:** Cost, availability, substitutability

### 2. Experimental Difficulty (1-10)
**Question:** Can high school students build and execute this setup?

**Scoring:**
- **10:** Simple assembly, minimal precision (stack books, pour water, drop objects)
- **7-9:** Moderate complexity, standard lab techniques (build circuits, measure with sensors)
- **4-6:** Requires careful calibration, some technical skill (soldering, programming Arduino)
  - May need teacher guidance for setup
  - Multiple components to integrate
- **1-3:** Requires expert assistance, very precise measurements (optical alignment, vacuum systems)
  - University-level technical skills needed

**Consider:** Setup complexity, calibration needs, technical skills required, time to build

### 3. Measurability (1-10)
**Question:** Can the key variables be measured quantitatively?

**Scoring:**
- **10:** Easy direct measurement (ruler, stopwatch, basic sensors)
- **7-9:** Standard instruments (thermometer, multimeter, force sensor, light sensor)
- **4-6:** Requires data analysis (video tracking, image processing, oscilloscope data)
  - Software analysis needed
  - Indirect measurement via calculations
- **1-3:** Only qualitative observation, hard to quantify
  - Subjective measurements
  - No clear way to get numbers

**Consider:** Direct vs indirect measurement, measurement precision, repeatability, data analysis complexity

### 4. Complexity (1-10)
**Question:** Is this appropriate depth for high school research?

**Warning: IYPT TARGET: 5-7** (sweet spot for high school)

**Scoring:**
- **8-10:** TOO COMPLEX - Requires university-level physics
  - Quantum mechanics, general relativity, advanced thermodynamics
  - Multi-variable differential equations needed
- **5-7:** **PERFECT FOR IYPT** - Challenging but achievable
  - Multiple interacting physics concepts
  - Requires systematic investigation
  - 3-6 months of research time
  - Can build theoretical model
- **3-4:** TOO SIMPLE - Could be done in one lab session
  - Single physics concept
  - Obvious relationships
  - No real investigation needed
- **1-2:** TRIVIAL - High school lab exercise level
  - Basic verification
  - Cookbook experiment

**Consider:** Physics depth, parameter space, analysis sophistication, time to understand

### 5. Time Feasibility (1-10)
**Question:** Can this be completed in 3-6 months?

**Scoring:**
- **10:** Doable in 1-2 months
  - Quick setup
  - Fast data collection
  - Simple analysis
- **7-9:** 3-4 months (ideal IYPT timeline)
  - Moderate setup time
  - Multiple rounds of experiments
  - Iterative refinement
- **4-6:** 5-6 months (tight but manageable)
  - Complex setup
  - Long data collection
  - Extensive analysis
- **1-3:** Requires >6 months or waiting for rare events
  - Seasonal phenomena
  - Very slow processes
  - Extensive preliminary work

**Consider:** Setup time, data collection time, iteration time, analysis time

### 6. Safety Assessment
**Question:** Is this safe for high school students?

**Options:**
- **SAFE:** Safe for unsupervised student lab work
  - No hazardous materials
  - No high voltages
  - No sharp/hot objects
  - No special PPE required

- **REQUIRES_SUPERVISION:** Safe with teacher present
  - Minor hazards (hot plates, low voltage)
  - Eye protection needed
  - Teacher can manage risks
  - Standard lab safety sufficient

- **UNSAFE:** Not appropriate for high school
  - Dangerous chemicals (strong acids/bases)
  - High voltage (>50V AC, >100V DC)
  - Extreme temperatures
  - Toxic/radioactive materials
  - Requires specialized safety training

### 7. Overall Feasibility
**Based on all scores:**
- **HIGH:** All critical scores >=7, complexity 5-7, safe
  - Recommended for IYPT
  - Ready to proceed
- **MEDIUM:** Mixed scores (some 4-6)
  - Doable with effort
  - May need modifications
- **LOW:** Critical scores <4
  - Not recommended without major changes
  - Fundamental feasibility issues

**Critical scores:** material_accessibility, experimental_difficulty, time_feasibility

### 8. Recommendation
**Decision:**
- **YES:** Good IYPT problem, proceed
  - overall_feasibility='HIGH'
  - complexity_score 5-7
  - safety_level != 'UNSAFE'

- **WITH_MODIFICATIONS:** Good concept but needs changes
  - overall_feasibility='MEDIUM'
  - Fixable issues identified
  - See improvements_suggested

- **NO:** Not suitable even with modifications
  - overall_feasibility='LOW'
  - safety_level='UNSAFE'
  - Fundamental problems

### 9. Improvements Suggested
**If recommended != 'YES':**
List specific actionable improvements:
- "Simplify maze geometry to reduce setup time"
- "Use water instead of liquid nitrogen for safer operation"
- "Add video camera for better measurement of droplet motion"
- "Reduce parameter space to focus on 2-3 key variables"

**If recommended = 'YES':**
Return empty list []

## Your Task

Evaluate the IYPT problem provided and return structured feasibility assessment.

Be realistic about high school capabilities:
- They have standard lab equipment (not university facilities)
- They have 3-6 months (not unlimited time)
- They need teacher support (not expert physicists)
- They should be challenged (not given trivial tasks)

**Remember:** Complexity score 5-7 is IDEAL for IYPT!
"""


SIMULATION_FEASIBILITY_PROMPT = """You are an expert in evaluating numerical simulation problems for Wolfram Mathematica.

Your task: Assess whether this problem is feasible for high school students to complete in 3-6 months using Wolfram Mathematica.

## Evaluation Criteria (1-10 scale, 10 = best)

### 1. Model Clarity (1-10)
**Question:** Can this phenomenon be described by standard mathematical equations?

**Scoring:**
- **10:** Classic ODEs/PDEs (pendulum, heat equation, diffusion, Lotka-Volterra)
  - Well-known textbook equations
  - Direct implementation in Mathematica
- **7-9:** Well-established models with some complexity
  - May require simplifying assumptions
  - Standard physics/chemistry/biology models
- **4-6:** Requires significant simplification or approximation
  - Model derivation is part of the challenge
  - May need to justify assumptions
- **1-3:** No clear mathematical model exists
  - Highly empirical phenomena
  - Would require custom model development

**Consider:** Is the mathematical formulation clear? Can students write down the equations?

### 2. Mathematica Suitability (1-10)
**Question:** Can Mathematica's built-in solvers handle this problem?

**Scoring:**
- **10:** Direct use of NDSolve, DSolve, Integrate, NMinimize
  - Standard examples from Mathematica documentation
  - No special techniques required
- **7-9:** Standard numerical methods with minor customization
  - May need to adjust solver options
  - Well-documented approaches
- **4-6:** Needs custom implementation or advanced techniques
  - Requires understanding of numerical stability
  - May need specialized methods (finite elements, Monte Carlo)
- **1-3:** Requires specialized software beyond Mathematica
  - CFD packages, molecular dynamics codes
  - Beyond Mathematica's built-in capabilities

**Consider:** Will NDSolve/DSolve work? Are there Mathematica tutorials for similar problems?

### 3. Visualization Potential (1-10)
**Question:** Can the results be effectively visualized in Mathematica?

**Scoring:**
- **10:** Simple 2D/3D plots, animations easily created
  - Plot, ParametricPlot, Animate, Manipulate work directly
  - Results are visually intuitive
- **7-9:** Multiple visualization types possible
  - Phase portraits, parameter sweeps, heat maps
  - May need some plot customization
- **4-6:** High-dimensional data needs reduction
  - Requires creative visualization choices
  - Results may be hard to interpret visually
- **1-3:** Results are inherently hard to visualize
  - Purely numerical outputs without clear visual interpretation
  - Would need significant post-processing

**Consider:** Can students create compelling animations or interactive demos?

### 4. Complexity (1-10)
**Question:** Is this appropriate depth for high school computational research?

**Warning: TARGET: 5-7** (sweet spot for high school)

**Scoring:**
- **8-10:** TOO COMPLEX - Requires university-level mathematics
  - Advanced numerical analysis
  - Tensor calculus, advanced PDEs
  - Research-level complexity
- **5-7:** **PERFECT FOR HIGH SCHOOL** - Challenging but achievable
  - Requires understanding of calculus concepts
  - Multiple interacting parameters
  - Rich exploration possible
- **3-4:** TOO SIMPLE - Could be done in one session
  - Single parameter variation
  - Obvious relationships
- **1-2:** TRIVIAL - Textbook exercise level
  - Simple analytical solution exists
  - No real exploration needed

**Consider:** Does this challenge students without overwhelming them?

### 5. Development Time (1-10)
**Question:** Can this be completed in 3-6 months?

**Scoring:**
- **10:** Doable in 1-2 months
  - Quick model setup
  - Fast simulation runs
  - Simple analysis
- **7-9:** 3-4 months (ideal timeline)
  - Moderate coding time
  - Multiple parameter studies
  - Iterative refinement
- **4-6:** 5-6 months (tight but manageable)
  - Complex model development
  - Long simulation runs
  - Extensive analysis needed
- **1-3:** Requires >6 months
  - Very long computation times
  - Extensive debugging expected
  - Complex visualization development

**Consider:** Model setup time, simulation runtime, visualization development, analysis time

### 6. Computational Tractability
**Question:** Will the simulation converge and complete in reasonable time?

**Options:**
- **TRACTABLE:** Simulations complete quickly and reliably
  - Standard solver settings work
  - Results in seconds to minutes
  - Numerical stability not an issue

- **CONDITIONALLY_TRACTABLE:** May need parameter tuning
  - Some parameter ranges cause issues
  - May need to adjust step sizes or tolerances
  - Students need to learn about numerical stability

- **INTRACTABLE:** Significant computational challenges
  - Very long runtimes (hours per run)
  - Frequent convergence failures
  - Would require cluster computing

### 7. Overall Feasibility
**Based on all scores:**
- **HIGH:** All critical scores >=7, complexity 5-7, tractable
  - Recommended for simulation project
  - Ready to proceed
- **MEDIUM:** Mixed scores (some 4-6)
  - Doable with effort
  - May need simplifications
- **LOW:** Critical scores <4
  - Not recommended without major changes
  - Fundamental feasibility issues

**Critical scores:** model_clarity, mathematica_suitability, development_time

### 8. Recommendation
**Decision:**
- **YES:** Good simulation problem, proceed
  - overall_feasibility='HIGH'
  - complexity_score 5-7
  - tractability != 'INTRACTABLE'

- **WITH_MODIFICATIONS:** Good concept but needs changes
  - overall_feasibility='MEDIUM'
  - Fixable issues identified
  - See improvements_suggested

- **NO:** Not suitable even with modifications
  - overall_feasibility='LOW'
  - tractability='INTRACTABLE'
  - Fundamental problems

### 9. Improvements Suggested
**If recommended != 'YES':**
List specific actionable improvements:
- "Simplify to 1D instead of 2D to reduce computation time"
- "Use pre-built Mathematica examples as starting point"
- "Focus on one parameter at a time initially"
- "Consider quasi-steady approximation to avoid stiff ODEs"

**If recommended = 'YES':**
Return empty list []

## Your Task

Evaluate the simulation problem provided and return structured feasibility assessment.

Be realistic about high school capabilities:
- They have access to Mathematica (school or Wolfram Cloud)
- They have 3-6 months (not unlimited time)
- They have basic calculus knowledge (not advanced numerical analysis)
- They should be challenged (not given trivial problems)

**Remember:** Complexity score 5-7 is IDEAL! Not too simple, not too advanced.
"""


DATA_ANALYSIS_FEASIBILITY_PROMPT = """You are an expert in evaluating data analysis and statistical investigation problems.

Your task: Assess whether this problem is feasible for high school students to complete in 3-6 months using statistical software (Mathematica, Python, R, etc.).

## Evaluation Criteria (1-10 scale, 10 = best)

### 1. Data Accessibility (1-10)
**Question:** Can students easily access and use this dataset?

**Scoring:**
- **10:** Public datasets with easy download (Kaggle, government open data)
  - Clean, well-documented data
  - Simple CSV or similar format
- **7-9:** Publicly available but requires some effort
  - API access needed
  - May require registration
  - Moderate data cleaning required
- **4-6:** Available but challenging to obtain/process
  - Complex data format
  - Significant preprocessing needed
  - Requires domain-specific tools
- **1-3:** Restricted or very difficult to access
  - Requires special permissions
  - Paywalled or proprietary data
  - Extremely messy or incomplete

**Consider:** Can a high school student find and download this data in one afternoon?

### 2. Analysis Complexity (1-10)
**Question:** Can high school students perform the required analysis?

**Scoring:**
- **10:** Basic statistics (mean, median, correlation, bar charts)
  - Built-in spreadsheet functions work
  - Simple visualizations
- **7-9:** Standard statistical methods
  - Linear regression, t-tests, chi-square
  - May need software like Mathematica or Python
  - Well-documented techniques
- **4-6:** Moderate complexity
  - Machine learning classification
  - Time series analysis
  - Requires understanding of ML concepts
- **1-3:** Advanced techniques needed
  - Deep learning, complex neural networks
  - Bayesian inference
  - Requires graduate-level statistics

**Consider:** Can students learn the techniques in 1-2 months with online tutorials?

### 3. Interpretation Clarity (1-10)
**Question:** Can results be clearly interpreted and communicated?

**Scoring:**
- **10:** Results have obvious meaning
  - Clear correlation/prediction success
  - Easy to explain to non-experts
- **7-9:** Results require some explanation
  - Statistical significance needs context
  - Visualizations help interpretation
- **4-6:** Results need careful interpretation
  - Confounding factors to consider
  - Multiple possible explanations
- **1-3:** Results are hard to interpret meaningfully
  - Very noisy patterns
  - No clear scientific insight emerges

**Consider:** Can students present findings that non-experts understand?

### 4. Complexity (1-10)
**Question:** Is this appropriate depth for high school data analysis?

**Warning: TARGET: 5-7** (sweet spot for high school)

**Scoring:**
- **8-10:** TOO COMPLEX - Requires advanced statistics
  - Multiple confounding variables
  - Sophisticated model selection
  - Research-level complexity
- **5-7:** **PERFECT FOR HIGH SCHOOL** - Challenging but achievable
  - Interesting patterns to discover
  - Multiple approaches possible
  - Room for creative exploration
- **3-4:** TOO SIMPLE - Obvious patterns
  - One-variable analysis
  - Results predictable beforehand
- **1-2:** TRIVIAL - Cookbook exercise
  - No real discovery
  - Already well-known results

### 5. Project Time (1-10)
**Question:** Can this be completed in 3-6 months?

**Scoring:**
- **10:** Doable in 1-2 months
  - Quick data acquisition
  - Simple analysis pipeline
- **7-9:** 3-4 months (ideal timeline)
  - Moderate data cleaning
  - Multiple analysis iterations
- **4-6:** 5-6 months (tight but manageable)
  - Significant data preparation
  - Complex analysis required
- **1-3:** Requires >6 months
  - Very large datasets
  - Extensive preprocessing
  - Complex model development

### 6. Data Quality
**Question:** Is the data reliable and appropriate for the question?

**Options:**
- **HIGH_QUALITY:** Clean, complete, well-documented
  - From reputable sources
  - Appropriate for the analysis
  - Sufficient sample size

- **MODERATE_QUALITY:** Some issues but workable
  - Missing values manageable
  - May need cleaning
  - Sample size adequate

- **LOW_QUALITY:** Significant data issues
  - Many missing values
  - Unclear provenance
  - May bias results

### 7-9. Overall Feasibility, Recommendation, Improvements
Same framework as other problem types.

## Your Task

Evaluate the data analysis problem and return structured feasibility assessment.
Be realistic about high school capabilities with data analysis.
"""


THEORETICAL_FEASIBILITY_PROMPT = """You are an expert in evaluating theoretical derivation problems using symbolic computation.

Your task: Assess whether this problem is feasible for high school students to complete in 3-6 months using Wolfram Mathematica for symbolic computation.

## Evaluation Criteria (1-10 scale, 10 = best)

### 1. Mathematical Prerequisites (1-10)
**Question:** Does the student have the required math background?

**Scoring:**
- **10:** Basic algebra and introductory calculus
  - Simple derivatives and integrals
  - Quadratic equations, basic trig
- **7-9:** Standard calculus (AP Calculus level)
  - First-order ODEs
  - Basic integration techniques
  - Series expansions
- **4-6:** Advanced calculus topics
  - Second-order ODEs
  - Partial derivatives
  - Vector calculus basics
- **1-3:** University-level mathematics
  - PDEs, linear algebra
  - Complex analysis
  - Beyond typical high school curriculum

**Consider:** Can students in AP Calculus or equivalent handle the math?

### 2. Derivation Tractability (1-10)
**Question:** Can this be derived symbolically with reasonable effort?

**Scoring:**
- **10:** Straightforward symbolic manipulation
  - DSolve, Integrate work directly
  - Clear step-by-step derivation
- **7-9:** Standard techniques with some effort
  - May need substitutions
  - Well-known methods apply
  - Mathematica documentation covers it
- **4-6:** Requires insight or special techniques
  - Non-obvious approaches
  - May need approximations
  - Students need guidance
- **1-3:** Research-level difficulty
  - No closed-form solution
  - Very specialized techniques
  - Beyond symbolic computation

### 3. Physical Insight (1-10)
**Question:** Does the derivation reveal meaningful physical understanding?

**Scoring:**
- **10:** Clear physical interpretation
  - Result explains observable behavior
  - Connects math to intuition
- **7-9:** Good physical connection
  - Requires some interpretation
  - Reveals underlying principles
- **4-6:** Physical meaning needs work
  - Abstract result
  - Connection to physics not obvious
- **1-3:** Purely mathematical exercise
  - No clear physical significance
  - "So what?" problem

**Consider:** Will students understand WHY the result matters?

### 4. Complexity (1-10)
**Question:** Is this appropriate depth for high school theoretical work?

**Warning: TARGET: 5-7** (sweet spot for high school)

**Scoring:**
- **8-10:** TOO COMPLEX - University-level derivation
  - Advanced mathematical techniques
  - Long chain of reasoning
  - Expert knowledge assumed
- **5-7:** **PERFECT FOR HIGH SCHOOL** - Challenging but achievable
  - Reveals deep physical principles
  - Multiple steps but followable
  - Extensions possible
- **3-4:** TOO SIMPLE - One-step derivation
  - Trivial manipulation
  - Already in textbook
- **1-2:** TRIVIAL - Simple formula substitution
  - No real derivation
  - Plug and chug

### 5. Derivation Time (1-10)
**Question:** Can this be completed in 3-6 months?

**Scoring:**
- **10:** Doable in 1-2 months
  - Quick learning curve
  - Simple derivation chain
- **7-9:** 3-4 months (ideal timeline)
  - Time to understand physics
  - Practice with Mathematica symbolic tools
  - Explore extensions
- **4-6:** 5-6 months (tight but manageable)
  - Significant mathematical background needed
  - Complex derivation chain
- **1-3:** Requires >6 months
  - Would need to learn advanced math first
  - Very long derivation process

### 6. Verification Potential
**Question:** Can the derived result be verified or tested?

**Options:**
- **EASILY_VERIFIED:** Result matches known formulas or experiments
  - Can compare to textbook
  - Limiting cases work out
  - Physical intuition confirms

- **VERIFIABLE_WITH_EFFORT:** Needs some work to check
  - Numerical verification possible
  - Need to find reference values
  - Dimensional analysis works

- **HARD_TO_VERIFY:** Difficult to check correctness
  - No easy comparison
  - Students may not know if they're right
  - Could get wrong answer without knowing

### 7-9. Overall Feasibility, Recommendation, Improvements
Same framework as other problem types.

## Your Task

Evaluate the theoretical problem and return structured feasibility assessment.
Be realistic about high school mathematical capabilities.
Focus on whether the derivation provides genuine physical insight.
"""


async def validate_feasibility(
    problem: dict[str, Any],
    problem_type: str,
) -> dict[str, Any]:
    """Stage 3: Validate problem feasibility.

    Single responsibility: Assess all feasibility dimensions (1-10 scores).

    Criteria vary by problem_type:
    - experimental: material accessibility, experimental difficulty, measurability
    - simulation: model clarity, Mathematica suitability, visualization potential
    - data_analysis: data accessibility, analysis complexity, interpretation clarity
    - theoretical: mathematical prerequisites, derivation tractability, physical insight

    Args:
        problem: Output from problem_generator containing:
            - title: str
            - description: str
            - investigation: str
            - core_concepts: list[str]
        problem_type: Problem type:
            - 'experimental': Lab-based physical experiments
            - 'simulation': Numerical modeling (ODEs, PDEs, chaos)
            - 'data_analysis': Statistical analysis, ML on datasets
            - 'theoretical': Mathematical derivations, symbolic proofs

    Returns:
        {
            "material_accessibility": {"score": int},
            "experimental_difficulty": {"score": int},
            "measurability": {"score": int},
            "complexity": {"score": int},
            "time_feasibility": {"score": int},
            "safety": {"level": str},
            "overall_feasibility": str,  # 'HIGH', 'MEDIUM', 'LOW'
            "recommended": str,  # 'YES', 'WITH_MODIFICATIONS', 'NO'
            "improvements": list[str],
            "raw_response": dict  # Pydantic model dump for debugging
        }
    """
    logger.info(
        "Stage 3: Feasibility Validation - problem='{}', problem_type={}",
        problem.get("title"),
        problem_type,
    )

    # Select prompt based on problem_type
    if problem_type == "simulation":
        base_prompt = SIMULATION_FEASIBILITY_PROMPT
    elif problem_type == "data_analysis":
        base_prompt = DATA_ANALYSIS_FEASIBILITY_PROMPT
    elif problem_type == "theoretical":
        base_prompt = THEORETICAL_FEASIBILITY_PROMPT
    else:  # experimental (default)
        base_prompt = FEASIBILITY_VALIDATION_PROMPT

    # Build problem context based on problem_type
    core_concepts = problem.get("core_concepts", problem.get("physics_concepts", []))

    if problem_type == "simulation":
        user_prompt_text = f"""Evaluate this numerical simulation problem for feasibility in Wolfram Mathematica:

**Title:** {problem.get('title')}

**Description:** {problem.get('description')}

**Investigation:** {problem.get('investigation')}

**Core Concepts:** {', '.join(core_concepts)}

**Task:** Provide comprehensive feasibility assessment with all scores (1-10 scale).

Remember:
- Complexity score 5-7 is IDEAL (not too simple, not too complex)
- Be realistic about high school computational capabilities
- Consider computational tractability seriously
- Provide actionable improvements if needed
"""
    elif problem_type == "data_analysis":
        user_prompt_text = f"""Evaluate this data analysis problem for feasibility:

**Title:** {problem.get('title')}

**Description:** {problem.get('description')}

**Investigation:** {problem.get('investigation')}

**Core Concepts:** {', '.join(core_concepts)}

**Task:** Provide comprehensive feasibility assessment with all scores (1-10 scale).

Remember:
- Complexity score 5-7 is IDEAL (not too simple, not too complex)
- Be realistic about high school data analysis capabilities
- Consider data accessibility and quality seriously
- Provide actionable improvements if needed
"""
    elif problem_type == "theoretical":
        user_prompt_text = f"""Evaluate this theoretical derivation problem for feasibility:

**Title:** {problem.get('title')}

**Description:** {problem.get('description')}

**Investigation:** {problem.get('investigation')}

**Core Concepts:** {', '.join(core_concepts)}

**Task:** Provide comprehensive feasibility assessment with all scores (1-10 scale).

Remember:
- Complexity score 5-7 is IDEAL (not too simple, not too complex)
- Be realistic about high school mathematical capabilities
- Consider whether derivation provides genuine physical insight
- Provide actionable improvements if needed
"""
    else:  # experimental (default)
        user_prompt_text = f"""Evaluate this IYPT problem for feasibility:

**Title:** {problem.get('title')}

**Description:** {problem.get('description')}

**Investigation:** {problem.get('investigation')}

**Core Concepts:** {', '.join(core_concepts)}

**Task:** Provide comprehensive feasibility assessment with all scores (1-10 scale).

Remember:
- Complexity score 5-7 is IDEAL for IYPT (not too simple, not too complex)
- Be realistic about high school lab capabilities
- Consider safety seriously
- Provide actionable improvements if needed
"""

    # Use structured LLM for direct extraction (no chat step needed)
    messages = [
        SystemMessage(content=base_prompt),
        HumanMessage(content=user_prompt_text),
    ]

    feasibility_scores = await structured_call(FeasibilityScores, messages, thinking="medium")

    # Convert Pydantic model to dict with nested structure
    result = {
        "material_accessibility": {
            "score": feasibility_scores.material_accessibility_score,
        },
        "experimental_difficulty": {
            "score": feasibility_scores.experimental_difficulty_score,
        },
        "measurability": {
            "score": feasibility_scores.measurability_score,
        },
        "complexity": {
            "score": feasibility_scores.complexity_score,
        },
        "time_feasibility": {
            "score": feasibility_scores.time_feasibility_score,
        },
        "safety": {
            "level": feasibility_scores.safety_level,
        },
        "overall_feasibility": feasibility_scores.overall_feasibility,
        "recommended": feasibility_scores.recommended,
        "improvements": feasibility_scores.improvements_suggested,
        "raw_response": feasibility_scores.model_dump(),  # Full Pydantic model for debugging
    }

    complexity = result["complexity"]
    assert isinstance(complexity, dict)
    logger.info(
        "Stage 3 complete: overall={}, recommended={}, complexity={}",
        result["overall_feasibility"],
        result["recommended"],
        complexity["score"],
    )

    return result
