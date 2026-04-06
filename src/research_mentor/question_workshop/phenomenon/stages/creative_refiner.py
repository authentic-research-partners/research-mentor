"""Stage 4: Creative Refiner

Single responsibility: Enhance problem engagement and novelty.

Takes validated problem and refines it for:
- Title catchiness (evocative, memorable)
- Description vividness (sensory language)
- Investigation clarity (specific parameters)
- Overall engagement and "wow factor"

Pattern: Chat LLM + LLM-based Novelty Assessment + Structured Extraction
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import ValidationError

from research_mentor.llm import get_chat_llm, structured_call
from research_mentor.question_workshop.schemas import RefinedProblem

CREATIVE_REFINEMENT_PROMPT = """You are an expert at making IYPT-style open-ended experimental problems engaging and memorable.

**IMPORTANT:** While the examples below are from IYPT (physics), this EXACT format applies to chemistry and biology problems as well. The key is making problems engaging with vivid language, specific parameters, and strong "wow factor."

**Format works across all sciences:**
- Physics: "Vapor Maze" (not "Droplet on Hot Surface")
- Chemistry: "Color Clock" (not "Delayed Reaction Experiment")
- Biology: "Bacterial Maze" (not "Chemotaxis Study")

Your task: Refine an IYPT-style problem to maximize student engagement and "wow factor."

## What Makes an IYPT Problem Engaging?

### 1. Catchy Title (2-4 words)
**Goal:** Evocative imagery, memorable phrasing, hints at surprise

**Good examples:**
- ✅ "Vapor Maze" (not "Droplet on Hot Surface")
- ✅ "Whistling Mesh" (not "Water Through Metal Screen")
- ✅ "Dancing Slinky" (not "Spring Oscillation")
- ✅ "Autumn Coin" (not "Spinning Coin Sound")
- ✅ "Sweet Monochromator" (not "Sugar Crystal Light")

**What to change:**
- Generic → Specific ("Floating Object" → "Vapor Maze")
- Technical → Evocative ("Acoustic Resonance" → "Whistling Mesh")
- Boring → Dynamic ("Ball Movement" → "Dancing Slinky")

### 2. Vivid Description (2-4 sentences)
**Goal:** Sensory language, action verbs, surprising contrasts, engaging questions

**Enhancement techniques:**
- Add sensory verbs: "watch," "hear," "feel," "observe"
- Use action verbs: "glides," "dances," "bounces," "spirals"
- Emphasize contrasts: "scorching yet floating," "chaotic yet ordered"
- Add questions: "Can you...?", "What happens when...?"

**Before:**
"Put a water drop on a hot surface. The drop floats. Change the temperature."

**After:**
"Drop a tiny water droplet onto a scorching metal maze. Watch as it magically glides, levitated by its own vapor cushion. Can you guide the droplet through the intricate pathways without it touching the metal?"

**What to add:**
- Scale words: "tiny," "massive," "intricate," "delicate"
- Surprise words: "magically," "unexpectedly," "counterintuitively"
- Dramatic adjectives: "scorching," "freezing," "violent," "gentle"

### 3. Clear Investigation (1-2 sentences)
**Goal:** Specific parameter names, explicit relationships, still open-ended

**Enhancement techniques:**
- Name 2-4 independent variables explicitly
- Name 2-4 dependent variables explicitly
- Use "how X affects Y" structure
- Avoid vague terms like "relevant parameters"

**Before:**
"Investigate relevant parameters affecting droplet motion."

**After:**
"Investigate how the maze geometry, surface temperature, and droplet size affect the droplet's velocity, stability, and success rate in navigating the vapor-supported path."

**What to specify:**
- Independent variables: geometry, temperature, size, material, frequency
- Dependent variables: velocity, amplitude, frequency, pattern, efficiency
- Relationships: "how X affects Y," "dependence on Z"

## Engagement Scoring (1-10 scale)

**Question:** How exciting and intriguing is this problem for high school students?

**10 = Exceptional engagement:**
- Visually striking effect (patterns, motion, transformation)
- Strong "wow factor" (defies expectations)
- High "show-and-tell" potential (you'd demo this to friends)
- Relatable phenomenon (seen similar in daily life)
- Example: Leidenfrost droplet maze, Chladni patterns

**7-9 = Very engaging:**
- Clear surprising element
- Interesting visual or auditory effect
- Students can easily imagine it
- Example: Whistling mesh, liquid coiling

**4-6 = Moderately engaging:**
- Some interesting aspects
- Not obviously surprising
- Requires explanation to appreciate
- Example: Standard pendulum variations

**1-3 = Low engagement:**
- Dull or overly technical
- No obvious "hook"
- Students would say "so what?"
- Example: Generic measurement tasks

## Your Refinement Task

Given an IYPT problem (with validation feedback), refine it to maximize engagement.

**Refinement priorities:**
1. **Title:** Make it catchy and evocative (2-4 words)
2. **Description:** Add sensory language and surprise (2-4 sentences)
3. **Investigation:** Specify exact parameters (1-2 sentences)
4. **Engagement:** Aim for score ≥7 (engaging, not just technically sound)

**Constraints:**
- Keep the core physics concept (don't change the phenomenon)
- Maintain IYPT format requirements
- Stay realistic about feasibility (don't add impossible elements)
- Enhance, don't replace (build on what's already good)

**Consider validation feedback:**
If the problem has issues (low feasibility, safety concerns, needs modifications):
- Address concerns in refinement (suggest safer materials, simpler setup)
- Emphasize what DOES work
- Don't hide problems, but present positively

**Document your changes:**
List specific refinements made (for transparency and learning).
"""


SIMULATION_REFINEMENT_PROMPT = """You are an expert at making numerical simulation problems engaging and memorable for Wolfram Mathematica.

**IMPORTANT:** Focus on making simulation problems exciting through mathematical beauty, visual appeal, and parameter exploration potential.

**Format works across all sciences:**
- Physics: "Chaotic Swing" (not "Double Pendulum Simulation")
- Chemistry: "Pattern Birth" (not "Reaction-Diffusion System")
- Biology: "Population Pulse" (not "Predator-Prey Model")

Your task: Refine a simulation problem to maximize student engagement and computational exploration appeal.

## What Makes a Simulation Problem Engaging?

### 1. Catchy Title (2-4 words)
**Goal:** Evocative of mathematical beauty, emergent behavior, or surprising patterns

**Good examples:**
- ✅ "Chaotic Swing" (not "Double Pendulum Simulation")
- ✅ "Heat Whispers" (not "Diffusion Equation Solution")
- ✅ "Spiral Dance" (not "Pattern Formation Model")
- ✅ "Resonance Peak" (not "Driven Oscillator Study")
- ✅ "Population Pulse" (not "Lotka-Volterra Model")

**What to change:**
- Technical → Evocative ("ODE System" → "Chaotic Swing")
- Generic → Specific ("Simulation Study" → "Heat Whispers")
- Boring → Dynamic ("Model Solution" → "Spiral Dance")

### 2. Vivid Description (2-4 sentences)
**Goal:** Connect math to physical intuition, emphasize visual results, build anticipation

**Enhancement techniques:**
- Connect equations to real-world phenomena
- Describe what the simulation SHOWS (animations, patterns, transitions)
- Emphasize counterintuitive or emergent behavior
- Add questions: "Can you find...?", "What happens when...?"

**Before:**
"A double pendulum can be modeled using coupled ODEs. Solve the equations and plot the results."

**After:**
"A double pendulum follows simple equations of motion, yet produces wildly unpredictable chaos. Create animated phase portraits in Mathematica and watch deterministic equations spawn infinite variety. Can you find the boundary where order gives way to chaos?"

**What to add:**
- Mathematical beauty: "elegant," "emergent," "deterministic yet unpredictable"
- Visual imagery: "watch," "see," "animated," "visualize"
- Discovery language: "find," "discover," "explore," "uncover"

### 3. Clear Investigation (1-2 sentences)
**Goal:** Specific parameters, explicit visualization goals, clear exploration strategy

**Enhancement techniques:**
- Name simulation parameters explicitly (coefficients, initial conditions, boundaries)
- Name visualization outputs (phase portraits, bifurcation diagrams, animations)
- Use "how X affects Y" structure
- Reference Mathematica tools (NDSolve, Manipulate, ParametricPlot)

**Before:**
"Investigate the behavior of the system."

**After:**
"Investigate how initial angle and mass ratio affect the transition to chaos, using phase space animations and Poincaré sections to map the boundary between ordered and chaotic regimes."

## Engagement Scoring (1-10 scale)

**Question:** How exciting is this computational exploration for high school students?

**10 = Exceptional engagement:**
- Stunning visualizations (fractals, chaos, emergent patterns)
- Strong "discovery" potential (finding boundaries, transitions)
- Interactive exploration possible (Manipulate demos)
- Connects deep math to intuitive understanding
- Example: Chaotic pendulum, reaction-diffusion patterns

**7-9 = Very engaging:**
- Clear visual payoff (animations, phase portraits)
- Interesting parameter dependencies
- Students can explore "what if" questions
- Example: Resonance peaks, heat flow animations

**4-6 = Moderately engaging:**
- Some interesting visualizations
- Standard parameter studies
- Requires explanation to appreciate
- Example: Basic oscillator variations

**1-3 = Low engagement:**
- Mostly numerical output without visual appeal
- No clear exploration direction
- Students would say "just equations"
- Example: Routine textbook problems

## Your Refinement Task

Given a simulation problem (with validation feedback), refine it to maximize engagement.

**Refinement priorities:**
1. **Title:** Make it evocative of mathematical beauty (2-4 words)
2. **Description:** Connect math to intuition, emphasize visualizations (2-4 sentences)
3. **Investigation:** Specify parameters and visualization outputs (1-2 sentences)
4. **Engagement:** Aim for score ≥7 (exciting exploration, not just number crunching)

**Constraints:**
- Keep the core mathematical model (don't change the equations)
- Maintain tractability (don't add impossible computations)
- Stay realistic about high school capabilities
- Enhance, don't replace (build on what's already good)

**Consider validation feedback:**
If the problem has issues (computational complexity, tractability concerns):
- Address concerns in refinement (suggest simpler cases first)
- Emphasize what DOES work
- Suggest parameter ranges that converge reliably

**Document your changes:**
List specific refinements made (for transparency and learning).
"""


DATA_ANALYSIS_REFINEMENT_PROMPT = """You are an expert at making data analysis problems engaging and memorable.

**IMPORTANT:** Focus on making data analysis problems exciting through discovery potential, visualization impact, and real-world relevance.

**Format works across all sciences:**
- Physics: "Star Hunters" (not "Exoplanet Data Analysis")
- Chemistry: "Molecule Oracle" (not "Chemical Property Prediction")
- Biology: "Biodiversity Map" (not "Species Distribution Analysis")

Your task: Refine a data analysis problem to maximize student engagement and discovery appeal.

## What Makes a Data Analysis Problem Engaging?

### 1. Catchy Title (2-4 words)
**Goal:** Evocative of discovery, patterns, or real-world impact

**Good examples:**
- ✅ "Star Hunters" (not "Exoplanet Classification")
- ✅ "Climate Detective" (not "Temperature Correlation Study")
- ✅ "Gene Networks" (not "Expression Pattern Analysis")
- ✅ "Quake Mapper" (not "Earthquake Data Study")

**What to change:**
- Technical → Discovery-focused ("ML Classification" → "Star Hunters")
- Generic → Specific ("Data Analysis" → "Climate Detective")
- Boring → Adventurous ("Pattern Finding" → "Gene Networks")

### 2. Vivid Description (2-4 sentences)
**Goal:** Connect data to real-world significance, emphasize discovery moment

**Enhancement techniques:**
- Describe what the DATA represents (real observations, experiments, records)
- Emphasize the DISCOVERY potential (patterns hiding in data)
- Connect to broader significance (why this matters)
- Add questions: "Can you find...?", "What patterns...?"

**Before:**
"Analyze exoplanet data from Kepler telescope. Look for patterns in planetary properties."

**After:**
"NASA's Kepler telescope has cataloged thousands of distant worlds with detailed orbital data. Hunt through this cosmic census for patterns that predict habitability. Can you find the signatures of potentially Earth-like planets hiding in the numbers?"

### 3. Clear Investigation (1-2 sentences)
**Goal:** Specific variables, clear analysis approach, discovery-oriented

**Enhancement techniques:**
- Name specific features/variables in the dataset
- Name specific analysis techniques (correlation, classification, clustering)
- Use "how X relates to Y" or "what predicts Z" structure

**Before:**
"Investigate patterns in the data."

**After:**
"Investigate how planetary radius, orbital period, and stellar type correlate with planetary temperature estimates, using scatter plots and correlation matrices to identify habitable zone candidates."

## Engagement Scoring (1-10 scale)

**10 = Exceptional engagement:**
- Real-world data with significance
- Clear "Eureka!" discovery potential
- Visually striking patterns in data
- Students can share findings meaningfully

**7-9 = Very engaging:**
- Interesting dataset with clear question
- Good visualization potential
- Students can form hypotheses

**4-6 = Moderately engaging:**
- Standard analysis exercise
- Technical focus over discovery
- Requires explanation to appreciate

**1-3 = Low engagement:**
- Dry statistical exercise
- No clear discovery goal
- Students would say "so what?"

## Your Refinement Task

Given a data analysis problem, refine it to maximize engagement.

**Refinement priorities:**
1. **Title:** Make it evocative of discovery (2-4 words)
2. **Description:** Connect data to significance, emphasize discovery (2-4 sentences)
3. **Investigation:** Specify variables and analysis approach (1-2 sentences)
4. **Engagement:** Aim for score ≥7 (discovery-focused, not just technical)

**Document your changes:**
List specific refinements made (for transparency and learning).
"""


THEORETICAL_REFINEMENT_PROMPT = """You are an expert at making theoretical derivation problems engaging and memorable.

**IMPORTANT:** Focus on making derivation problems exciting through mathematical elegance, physical insight, and the satisfaction of proving fundamental truths.

**Format works across all sciences:**
- Physics: "Harmonic Secrets" (not "Pendulum Period Derivation")
- Chemistry: "Ideal Limits" (not "Gas Law Derivation")
- Biology: "Growth Curves" (not "Logistic Equation Derivation")

Your task: Refine a theoretical problem to maximize student engagement and derivation satisfaction.

## What Makes a Theoretical Problem Engaging?

### 1. Catchy Title (2-4 words)
**Goal:** Evocative of mathematical elegance or physical insight revealed

**Good examples:**
- ✅ "Harmonic Secrets" (not "Oscillation Period Derivation")
- ✅ "Orbital Dance" (not "Kepler's Laws Proof")
- ✅ "Wave Quantization" (not "Standing Wave Frequencies")
- ✅ "Entropy's Arrow" (not "Second Law Derivation")

**What to change:**
- Technical → Insightful ("ODE Solution" → "Harmonic Secrets")
- Generic → Evocative ("Period Formula" → "Orbital Dance")
- Boring → Mysterious ("Frequency Calculation" → "Wave Quantization")

### 2. Vivid Description (2-4 sentences)
**Goal:** Build anticipation for mathematical revelation, connect to physical intuition

**Enhancement techniques:**
- Describe the PUZZLE (why is this derivation interesting?)
- Emphasize what the RESULT reveals (physical insight)
- Connect abstract math to concrete phenomena
- Add questions: "Why does...?", "What connects...?"

**Before:**
"Derive the period formula for a simple pendulum using calculus."

**After:**
"The pendulum has fascinated scientists for centuries with its reliable rhythm. Why does the period depend on length but not mass? Use the elegant machinery of calculus to uncover this secret and reveal the deep connection between geometry and time."

### 3. Clear Investigation (1-2 sentences)
**Goal:** Specific derivation goal, clear mathematical approach, insight-oriented

**Enhancement techniques:**
- State what is being derived explicitly
- Name the mathematical techniques involved
- Reference what physical insight emerges

**Before:**
"Derive the formula and verify it."

**After:**
"Derive the pendulum period formula T = 2π√(L/g) from Newton's laws using small-angle approximation, then explore what happens to the derivation for large amplitude swings."

## Engagement Scoring (1-10 scale)

**10 = Exceptional engagement:**
- Elegant mathematical technique
- Profound physical insight revealed
- "Aha!" moment when result emerges
- Connects to observable phenomena

**7-9 = Very engaging:**
- Satisfying derivation with clear insight
- Well-known result but illuminating process
- Extensions possible

**4-6 = Moderately engaging:**
- Standard derivation exercise
- Result known beforehand
- Requires explanation to appreciate

**1-3 = Low engagement:**
- Mechanical calculation
- No insight emerges
- "Plug and chug" exercise

## Your Refinement Task

Given a theoretical problem, refine it to maximize engagement.

**Refinement priorities:**
1. **Title:** Make it evocative of mathematical beauty (2-4 words)
2. **Description:** Build anticipation, emphasize insight (2-4 sentences)
3. **Investigation:** Specify derivation goal and technique (1-2 sentences)
4. **Engagement:** Aim for score ≥7 (insight-focused, not mechanical)

**Document your changes:**
List specific refinements made (for transparency and learning).
"""

EXTRACTION_PROMPT = """Extract structured refinement from the creative enhancement.

Return:
- title: 2-4 words, enhanced for catchiness
- description: 2-4 sentences, enhanced for vividness
- investigation: 1-2 sentences, enhanced for clarity
- engagement_score: 1-10 (how exciting/intriguing is this problem?)
- changes_made: List of specific refinements

🚨 CRITICAL CHARACTER LIMITS (HARD CONSTRAINTS):
- title: MAXIMUM 50 characters (min: 3)
- description: MAXIMUM 500 characters (min: 50)
- investigation: MAXIMUM 300 characters (min: 30)

These are STRICT limits enforced by validation. Exceeding them will cause failure.

Guidelines for staying within limits:
- investigation (300 char max): Focus on 2-3 key parameters, use concise phrasing
  ✅ Good (178 chars): "Investigate how the maze geometry, surface temperature, and droplet size affect the droplet's velocity, stability, and success rate in navigating the vapor-supported path."
  ❌ Too long: Avoid excessive detail, multiple clauses, or verbose explanations

Validation:
- Title must be catchy, not generic
- Description must use sensory/action language
- Investigation must name specific parameters
- Engagement score must be realistic (consider wow factor)
- ALL fields must stay within character limits above
"""


async def refine_creatively(
    field: str,
    problem_type: str,
    problem: dict[str, Any],
    validation: dict[str, Any],
    existing_titles: list[str],
    student_profile_text: str = "",
) -> dict[str, Any]:
    """
    Stage 4: Refine problem for engagement and novelty.

    Single responsibility: Enhance problem presentation (title, description, investigation).

    Args:
        field: Scientific field ('physics', 'chemistry', 'biology')
        problem_type: Problem type:
            - 'experimental': Lab-based physical experiments
            - 'simulation': Numerical modeling (ODEs, PDEs, chaos)
            - 'data_analysis': Statistical analysis, ML on datasets
            - 'theoretical': Mathematical derivations, symbolic proofs
        problem: Output from problem_generator containing:
            - title: str
            - description: str
            - investigation: str
            - core_concepts: list[str]
        validation: Output from feasibility_validator containing:
            - overall_feasibility: str
            - recommended: str
            - improvements: list[str]
        existing_titles: Titles of previously generated problems (for novelty).

    Returns:
        {
            "title": str,
            "description": str,
            "investigation": str,
            "engagement_score": int,
            "novelty_assessment": str,
            "changes_made": list[str],
            "raw_response": str,
        }
    """
    logger.info(
        "Stage 4: Creative Refinement - field={}, problem_type={}, problem='{}'",
        field, problem_type, problem.get("title"),
    )

    # Select prompt based on problem_type
    if problem_type == "simulation":
        base_prompt = SIMULATION_REFINEMENT_PROMPT
        problem_type_label = "numerical simulation (Wolfram Mathematica)"
    elif problem_type == "data_analysis":
        base_prompt = DATA_ANALYSIS_REFINEMENT_PROMPT
        problem_type_label = "data analysis (statistical/ML)"
    elif problem_type == "theoretical":
        base_prompt = THEORETICAL_REFINEMENT_PROMPT
        problem_type_label = "theoretical derivation (symbolic computation)"
    else:  # experimental (default)
        base_prompt = CREATIVE_REFINEMENT_PROMPT
        problem_type_label = "experimental (lab-based)"

    # Build context from problem and validation
    feasibility_status = validation.get("overall_feasibility", "UNKNOWN")
    recommendation = validation.get("recommended", "UNKNOWN")
    improvements = validation.get("improvements", [])

    # Build refinement context based on problem_type
    if problem_type == "simulation":
        user_prompt_text = f"""Refine this {field} simulation problem for maximum engagement:

**Field:** {field.capitalize()}
**Problem Type:** {problem_type_label}

**Current Problem:**

**Title:** {problem.get('title')}

**Description:** {problem.get('description')}

**Investigation:** {problem.get('investigation')}

**Core Concepts:** {', '.join(problem.get('core_concepts', []))}

**Validation Feedback:**
- Feasibility: {feasibility_status}
- Recommendation: {recommendation}
"""
    elif problem_type == "data_analysis":
        user_prompt_text = f"""Refine this {field} data analysis problem for maximum engagement:

**Field:** {field.capitalize()}
**Problem Type:** {problem_type_label}

**Current Problem:**

**Title:** {problem.get('title')}

**Description:** {problem.get('description')}

**Investigation:** {problem.get('investigation')}

**Core Concepts:** {', '.join(problem.get('core_concepts', []))}

**Validation Feedback:**
- Feasibility: {feasibility_status}
- Recommendation: {recommendation}
"""
    elif problem_type == "theoretical":
        user_prompt_text = f"""Refine this {field} theoretical derivation problem for maximum engagement:

**Field:** {field.capitalize()}
**Problem Type:** {problem_type_label}

**Current Problem:**

**Title:** {problem.get('title')}

**Description:** {problem.get('description')}

**Investigation:** {problem.get('investigation')}

**Core Concepts:** {', '.join(problem.get('core_concepts', []))}

**Validation Feedback:**
- Feasibility: {feasibility_status}
- Recommendation: {recommendation}
"""
    else:  # experimental (default)
        user_prompt_text = f"""Refine this IYPT-style {field} problem for maximum engagement:

**Field:** {field.capitalize()}
**Problem Type:** {problem_type_label}

**Current Problem:**

**Title:** {problem.get('title')}

**Description:** {problem.get('description')}

**Investigation:** {problem.get('investigation')}

**Core Concepts:** {', '.join(problem.get('core_concepts', []))}

**Validation Feedback:**
- Feasibility: {feasibility_status}
- Recommendation: {recommendation}
"""

    # Add improvement suggestions if any
    if improvements:
        improvements_text = "\n".join(f"- {imp}" for imp in improvements)
        user_prompt_text += f"""
**Suggested Improvements:**
{improvements_text}
"""

    # Add task instructions based on problem_type
    if problem_type == "simulation":
        user_prompt_text += f"""
**Your Task:**
Refine the title, description, and investigation to make this {field} simulation problem MORE engaging while keeping the core mathematical model.

Focus on:
1. Making the title evocative of mathematical beauty (2-4 words)
2. Connecting math to intuition, emphasizing visualization potential
3. Specifying parameters and visualization outputs clearly
4. Boosting the computational exploration appeal

Provide a refined version that high school students will find exciting to simulate!
"""
    elif problem_type == "data_analysis":
        user_prompt_text += f"""
**Your Task:**
Refine the title, description, and investigation to make this {field} data analysis problem MORE engaging while keeping the core question.

Focus on:
1. Making the title evocative of discovery (2-4 words)
2. Connecting data to real-world significance, emphasizing discovery
3. Specifying variables and analysis approach clearly
4. Boosting the "Eureka!" discovery appeal

Provide a refined version that high school students will find exciting to explore!
"""
    elif problem_type == "theoretical":
        user_prompt_text += f"""
**Your Task:**
Refine the title, description, and investigation to make this {field} theoretical problem MORE engaging while keeping the core derivation.

Focus on:
1. Making the title evocative of mathematical elegance (2-4 words)
2. Building anticipation for the insight revealed
3. Specifying the derivation goal and technique clearly
4. Boosting the satisfaction of proving fundamental truths

Provide a refined version that high school students will find intellectually satisfying!
"""
    else:  # experimental (default)
        user_prompt_text += f"""
**Your Task:**
Refine the title, description, and investigation to make this {field} problem MORE engaging while keeping the core scientific concept.

Focus on:
1. Making the title catchy and evocative (2-4 words)
2. Adding sensory language and surprise to the description
3. Making the investigation specific (name exact parameters)
4. Boosting the "wow factor" and visual appeal

Provide a refined version that high school students will find exciting!
"""

    # Add LLM-based novelty assessment instructions
    if existing_titles:
        titles_list = ", ".join(f'"{t}"' for t in existing_titles)
        user_prompt_text += f"""
**Previously generated problems (avoid duplication):** {titles_list}
"""

    user_prompt_text += """
**Novelty Assessment:**
Assess whether this problem is novel or closely resembles well-known science competition problems (IYPT, Science Olympiad, etc.). If too similar, suggest how to make it more distinctive.
"""

    # Step 1: Generate refinement with chat LLM (creative, flexible)
    llm = get_chat_llm(temperature=0.7)

    prompt_text = f"{student_profile_text}\n\n{base_prompt}" if student_profile_text else base_prompt
    system_msg = SystemMessage(content=prompt_text)
    user_msg = HumanMessage(content=user_prompt_text)

    response = await llm.ainvoke([system_msg, user_msg])
    raw_response = response.content

    logger.debug("Creative refinement response length: {} chars", len(raw_response))

    # Step 2: Extract structured refinement with structured_call (type-safe)
    # Retry up to 3 times if validation fails (e.g., field too long)
    base_user_content = f"""Extract structured refinement from this enhanced problem:

{raw_response}

Ensure:
- Title is 2-4 words (catchy, not generic)
- Description is 2-4 sentences (sensory language)
- Investigation names specific parameters
- Engagement score reflects "wow factor" (1-10)
- Novelty assessment evaluates originality vs known competition problems
- Changes list specific refinements made
"""

    max_retries = 3
    refined_problem: RefinedProblem | None = None
    last_error: ValidationError | None = None

    for attempt in range(max_retries):
        try:
            user_content = base_user_content

            # Add specific guidance on retry attempts
            if attempt > 0:
                user_content += f"""

🚨 RETRY ATTEMPT {attempt + 1}/{max_retries}
Previous attempt failed validation. Common issues:
- investigation field exceeded 300 characters
- description field exceeded 500 characters
- title field exceeded 50 characters

Be MORE CONCISE. Focus on essential parameters only.
"""

            messages = [
                SystemMessage(content=EXTRACTION_PROMPT),
                HumanMessage(content=user_content),
            ]
            refined_problem = await structured_call(RefinedProblem, messages, thinking="high")

            # Success! Break out of retry loop
            logger.info(
                "Structured extraction successful on attempt {}/{}",
                attempt + 1, max_retries,
            )
            break

        except ValidationError as e:
            last_error = e
            error_details = str(e)

            # Parse which field failed (Pydantic error format)
            if "investigation" in error_details and "string_too_long" in error_details:
                field_name = "investigation"
                max_length = 300
            elif "description" in error_details and "string_too_long" in error_details:
                field_name = "description"
                max_length = 500
            elif "title" in error_details and "string_too_long" in error_details:
                field_name = "title"
                max_length = 50
            else:
                field_name = "unknown field"
                max_length = None

            logger.warning(
                "Validation failed on attempt {}/{}: {} exceeded limit{}",
                attempt + 1, max_retries, field_name,
                f" (max: {max_length} chars)" if max_length else "",
            )

            # Update base content for next retry with specific guidance
            if max_length:
                base_user_content += f"""

⚠️ CRITICAL: The '{field_name}' field MUST be under {max_length} characters.
Current attempt was too long. Make it MORE CONCISE:
- Remove unnecessary words
- Use shorter phrasing
- Focus on 2-3 key parameters only
"""

            if attempt == max_retries - 1:
                # Final attempt failed
                logger.error(
                    "All {} extraction attempts failed. Last error: {}",
                    max_retries, error_details,
                )
                raise last_error from None

    if refined_problem is None:
        # Should never happen (raised above), but for type safety
        raise RuntimeError("Failed to extract structured problem after retries")

    # Convert Pydantic model to dict
    result = {
        "title": refined_problem.title,
        "description": refined_problem.description,
        "investigation": refined_problem.investigation,
        "engagement_score": refined_problem.engagement_score,
        "novelty_assessment": refined_problem.novelty_assessment,
        "changes_made": refined_problem.changes_made,
        "raw_response": raw_response,
    }

    changes = result["changes_made"]
    assert isinstance(changes, list)
    logger.info(
        "Stage 4 complete: engagement={}/10, changes={}",
        result["engagement_score"], len(changes),
    )

    return result
