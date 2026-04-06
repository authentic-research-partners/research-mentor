"""Science research domains for question generation.

Comprehensive domains for physics, chemistry, and biology research problems.
Each domain includes description, example phenomena, and typical materials.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

# =============================================================================
# CONSTANTS
# =============================================================================

def _get_phenomenon_fields() -> set[str]:
    """Load allowed fields for the Phenomenon workshop from config.

    Falls back to the classic set if config is unavailable (e.g., during build).
    """
    try:
        from research_mentor.question_workshop.domain import get_allowed_fields
        allowed = get_allowed_fields("phenomenon")
        # Only return fields we actually have domain data for
        return {f for f in allowed if f in DOMAINS_BY_FIELD} if allowed else _DEFAULT_FIELDS
    except Exception:
        logger.debug("Failed to load allowed fields from config, using defaults")
        return _DEFAULT_FIELDS


_DEFAULT_FIELDS: set[str] = {"physics", "chemistry", "biology"}

# Public API — lazy-loaded from config
FIELDS: set[str] = _DEFAULT_FIELDS  # replaced at first use via get_fields()


def get_fields() -> set[str]:
    """Get the set of allowed fields for the Phenomenon workshop."""
    return _get_phenomenon_fields()

PROBLEM_TYPES: set[str] = {"experimental", "simulation", "data_analysis", "theoretical"}

# =============================================================================
# PHYSICS DOMAINS (13 domains)
# =============================================================================

PHYSICS_DOMAINS: list[dict[str, Any]] = [
    {
        "id": "fluid_dynamics",
        "name": "Fluid Dynamics",
        "description": (
            "Study of fluid flow, hydrodynamics, surface tension, "
            "viscosity, and turbulence"
        ),
        "phenomena_examples": [
            "Leidenfrost effect (droplet levitation on hot surface)",
            "Kaye effect (liquid stream coiling)",
            "Brazil nut effect (granular convection)",
            "Coffee ring effect",
            "Rayleigh-Plateau instability (jet breakup)",
        ],
        "typical_materials": ["water", "oil", "glycerin", "metal plate", "pipette", "thermometer"],
    },
    {
        "id": "mechanics",
        "name": "Classical Mechanics",
        "description": "Study of motion, forces, energy, momentum, and mechanical systems",
        "phenomena_examples": [
            "Chaotic pendulum oscillations",
            "Balancing objects",
            "Spinning tops and gyroscopes",
            "Elasticity and deformation",
            "Projectile motion with drag",
        ],
        "typical_materials": [
            "pendulum", "springs", "masses", "rulers",
            "stopwatch", "force sensors",
        ],
    },
    {
        "id": "rotational_dynamics",
        "name": "Rotational Dynamics",
        "description": "Study of angular motion, moment of inertia, torque, and spinning objects",
        "phenomena_examples": [
            "Spinning coin sounds",
            "Gyroscopic precession",
            "Rolling objects on inclines",
            "Angular momentum conservation",
            "Spinning tops",
        ],
        "typical_materials": [
            "coins", "gyroscopes", "rotating platforms",
            "cameras", "microphones",
        ],
    },
    {
        "id": "electromagnetism",
        "name": "Electromagnetism",
        "description": (
            "Study of electric and magnetic fields, "
            "electromagnetic induction, and EM waves"
        ),
        "phenomena_examples": [
            "Eddy current damping",
            "Magnetic levitation",
            "Electromagnetic induction",
            "Ferrofluids under magnetic fields",
            "Wireless power transfer",
        ],
        "typical_materials": [
            "magnets", "coils", "iron filings", "ferrofluid", "multimeter", "oscilloscope",
        ],
    },
    {
        "id": "wave_mechanics",
        "name": "Wave Mechanics",
        "description": "Study of wave propagation, interference, diffraction, and standing waves",
        "phenomena_examples": [
            "Standing waves in strings/plates",
            "Chladni patterns (nodal lines)",
            "Soliton waves",
            "Wave reflection and transmission",
            "Resonance phenomena",
        ],
        "typical_materials": [
            "string", "metal plates", "speakers", "frequency generator", "sand", "laser",
        ],
    },
    {
        "id": "acoustics",
        "name": "Acoustics & Sound",
        "description": "Study of sound generation, propagation, resonance, and vibrations",
        "phenomena_examples": [
            "Helmholtz resonance",
            "Whistling meshes",
            "Musical instrument physics",
            "Echo and reverberation",
            "Doppler effect",
        ],
        "typical_materials": [
            "speakers", "microphones", "tuning forks", "resonators", "tubes", "oscilloscope",
        ],
    },
    {
        "id": "optics",
        "name": "Optics & Light",
        "description": (
            "Study of light reflection, refraction, interference, diffraction, and polarization"
        ),
        "phenomena_examples": [
            "Rainbow formation",
            "Thin film interference",
            "Total internal reflection",
            "Laser speckle patterns",
            "Polarization effects",
        ],
        "typical_materials": [
            "laser", "prism", "lenses", "mirrors", "diffraction grating", "polarizers",
        ],
    },
    {
        "id": "thermal_physics",
        "name": "Thermal Physics",
        "description": (
            "Study of heat transfer, thermodynamics, phase transitions, and thermal expansion"
        ),
        "phenomena_examples": [
            "Convection patterns",
            "Thermal conductivity",
            "Phase transitions",
            "Thermal expansion",
            "Newton's law of cooling",
        ],
        "typical_materials": [
            "thermometers", "hot plate", "ice", "insulation materials", "thermal camera",
        ],
    },
    {
        "id": "granular_mechanics",
        "name": "Granular Mechanics",
        "description": "Study of granular materials, particle interactions, and flow dynamics",
        "phenomena_examples": [
            "Sand avalanches",
            "Granular segregation",
            "Sandpile angle of repose",
            "Jamming transitions",
            "Granular flow in hoppers",
        ],
        "typical_materials": [
            "sand", "rice", "beads", "containers", "inclined planes", "high-speed camera",
        ],
    },
    {
        "id": "elastic_mechanics",
        "name": "Elastic Mechanics",
        "description": "Study of elastic deformation, stress, strain, and material properties",
        "phenomena_examples": [
            "Spring oscillations",
            "Rubber band physics",
            "Beam bending",
            "Buckling instabilities",
            "Elastic collisions",
        ],
        "typical_materials": ["springs", "rubber bands", "beams", "force sensors", "strain gauges"],
    },
    {
        "id": "aerodynamics",
        "name": "Aerodynamics & Airflow",
        "description": "Study of air flow, lift, drag, and aerodynamic forces",
        "phenomena_examples": [
            "Paper airplane flight",
            "Boomerang aerodynamics",
            "Vortex shedding",
            "Bernoulli effect",
            "Drag reduction",
        ],
        "typical_materials": [
            "paper", "wind tunnel", "fans", "anemometer", "smoke", "high-speed camera",
        ],
    },
    {
        "id": "chaotic_systems",
        "name": "Chaotic & Nonlinear Systems",
        "description": (
            "Study of nonlinear dynamics, chaos, bifurcations, and sensitive dependence"
        ),
        "phenomena_examples": [
            "Double pendulum chaos",
            "Driven oscillator patterns",
            "Strange attractors",
            "Period-doubling cascades",
            "Butterfly effect demonstrations",
        ],
        "typical_materials": [
            "double pendulum", "driven oscillator", "magnets", "sensors", "data acquisition",
        ],
    },
    {
        "id": "pattern_formation",
        "name": "Pattern Formation",
        "description": (
            "Study of self-organization, emergent patterns, and spontaneous symmetry breaking"
        ),
        "phenomena_examples": [
            "Chladni patterns",
            "Rayleigh-B\u00e9nard convection cells",
            "Crystal growth patterns",
            "Reaction-diffusion patterns",
            "Turing patterns",
        ],
        "typical_materials": [
            "metal plates", "sand", "chemicals", "speakers", "cameras", "heating elements",
        ],
    },
]

# =============================================================================
# CHEMISTRY DOMAINS (12 domains)
# =============================================================================

CHEMISTRY_DOMAINS: list[dict[str, Any]] = [
    {
        "id": "reaction_kinetics",
        "name": "Reaction Kinetics",
        "description": "Study of reaction rates, rate laws, activation energy, and catalysis",
        "phenomena_examples": [
            "Oscillating reactions (Belousov-Zhabotinsky)",
            "Clock reactions (iodine clock)",
            "Enzyme kinetics (Michaelis-Menten)",
            "Temperature effects on reaction rates",
            "Catalytic decomposition of hydrogen peroxide",
        ],
        "typical_materials": [
            "spectrophotometer", "stopwatch", "thermometer", "pH meter", "reagents", "catalysts",
        ],
    },
    {
        "id": "electrochemistry",
        "name": "Electrochemistry",
        "description": "Study of electron transfer, redox reactions, batteries, and electrolysis",
        "phenomena_examples": [
            "Voltaic cell construction",
            "Electrolysis of water",
            "Electroplating metals",
            "Corrosion and rust formation",
            "Fuel cell efficiency",
        ],
        "typical_materials": [
            "electrodes", "multimeter", "salt bridge", "electrolyte solutions",
            "power supply", "beakers",
        ],
    },
    {
        "id": "thermochemistry",
        "name": "Thermochemistry",
        "description": (
            "Study of heat in chemical reactions, enthalpy, calorimetry, and energy changes"
        ),
        "phenomena_examples": [
            "Heat of combustion measurements",
            "Endothermic vs exothermic reactions",
            "Hess's law verification",
            "Bond energy calculations",
            "Phase transition enthalpies",
        ],
        "typical_materials": [
            "calorimeter", "thermometer", "insulation", "balance", "chemicals", "ignition source",
        ],
    },
    {
        "id": "acid_base",
        "name": "Acid-Base Chemistry",
        "description": "Study of acids, bases, pH, buffers, titrations, and neutralization",
        "phenomena_examples": [
            "pH indicator color changes",
            "Buffer capacity experiments",
            "Titration curves",
            "Weak vs strong acid behavior",
            "Antacid effectiveness",
        ],
        "typical_materials": [
            "pH meter", "indicators", "burette", "pipette", "acids", "bases", "buffer solutions",
        ],
    },
    {
        "id": "crystallography",
        "name": "Crystallography & Solid State",
        "description": (
            "Study of crystal growth, structure, polymorphism, and solid-state properties"
        ),
        "phenomena_examples": [
            "Crystal growing from supersaturated solutions",
            "Polymorphism in compounds",
            "Recrystallization purification",
            "Crystal habit modification",
            "Supersaturation and nucleation",
        ],
        "typical_materials": [
            "microscope", "crystallization dishes", "saturated solutions",
            "seed crystals", "temperature control",
        ],
    },
    {
        "id": "colloids_surfaces",
        "name": "Colloids & Surface Chemistry",
        "description": (
            "Study of colloids, emulsions, foams, surface tension, and adsorption"
        ),
        "phenomena_examples": [
            "Tyndall effect in colloids",
            "Emulsion stability",
            "Soap bubble lifetime",
            "Adsorption isotherms",
            "Micelle formation",
        ],
        "typical_materials": [
            "laser pointer", "surfactants", "oils", "water", "microscope", "tensiometer",
        ],
    },
    {
        "id": "polymer_chemistry",
        "name": "Polymer Chemistry",
        "description": (
            "Study of polymers, polymerization, plastics, and macromolecular properties"
        ),
        "phenomena_examples": [
            "Slime and gel formation",
            "Cross-linking effects",
            "Polymer degradation",
            "Superabsorbent polymers",
            "Biodegradable plastics",
        ],
        "typical_materials": [
            "monomers", "catalysts", "cross-linkers", "molds", "viscometer", "tensile tester",
        ],
    },
    {
        "id": "photochemistry",
        "name": "Photochemistry",
        "description": (
            "Study of light-induced reactions, fluorescence, photosynthesis, and luminescence"
        ),
        "phenomena_examples": [
            "Photodegradation of dyes",
            "Chemiluminescence (glow sticks)",
            "Fluorescence quenching",
            "Photochromic materials",
            "UV-induced reactions",
        ],
        "typical_materials": [
            "UV lamp", "spectrophotometer", "fluorescent materials",
            "light filters", "photosensitive compounds",
        ],
    },
    {
        "id": "analytical_chemistry",
        "name": "Analytical Chemistry",
        "description": (
            "Study of chemical analysis, separation, identification, and quantification"
        ),
        "phenomena_examples": [
            "Chromatography separations",
            "Flame tests for metals",
            "Spectroscopic identification",
            "Gravimetric analysis",
            "Colorimetric determination",
        ],
        "typical_materials": [
            "chromatography paper", "spectrophotometer", "balance",
            "flame source", "indicators", "standards",
        ],
    },
    {
        "id": "organic_synthesis",
        "name": "Organic Synthesis",
        "description": (
            "Study of organic reactions, synthesis routes, and functional group transformations"
        ),
        "phenomena_examples": [
            "Esterification reactions",
            "Saponification (soap making)",
            "Extraction and purification",
            "Fermentation processes",
            "Natural product isolation",
        ],
        "typical_materials": [
            "reflux apparatus", "separatory funnel", "distillation setup",
            "organic solvents", "reagents",
        ],
    },
    {
        "id": "environmental_chemistry",
        "name": "Environmental Chemistry",
        "description": (
            "Study of chemical processes in environment, pollution, and remediation"
        ),
        "phenomena_examples": [
            "Water quality testing",
            "Acid rain simulation",
            "Heavy metal detection",
            "Bioremediation effectiveness",
            "Greenhouse gas behavior",
        ],
        "typical_materials": [
            "water testing kits", "pH meter", "gas sensors", "soil samples",
            "indicator organisms",
        ],
    },
    {
        "id": "food_chemistry",
        "name": "Food Chemistry",
        "description": (
            "Study of chemical composition of food, reactions during cooking, and preservation"
        ),
        "phenomena_examples": [
            "Maillard reaction (browning)",
            "Vitamin C degradation",
            "Pectin gelation",
            "Enzyme activity in fruits",
            "Fermentation products",
        ],
        "typical_materials": [
            "food samples", "iodine solution", "Benedict's solution",
            "pH indicators", "heating apparatus",
        ],
    },
]

# =============================================================================
# BIOLOGY DOMAINS (12 domains)
# =============================================================================

BIOLOGY_DOMAINS: list[dict[str, Any]] = [
    {
        "id": "microbiology",
        "name": "Microbiology",
        "description": "Study of microorganisms including bacteria, fungi, and their behaviors",
        "phenomena_examples": [
            "Bacterial growth curves",
            "Antibiotic resistance zones",
            "Yeast fermentation rates",
            "Biofilm formation",
            "Microbial competition",
        ],
        "typical_materials": [
            "petri dishes", "agar", "incubator", "microscope",
            "sterile equipment", "bacterial cultures",
        ],
    },
    {
        "id": "plant_physiology",
        "name": "Plant Physiology",
        "description": "Study of plant functions including photosynthesis, growth, and responses",
        "phenomena_examples": [
            "Photosynthesis rate measurement",
            "Transpiration and stomatal behavior",
            "Tropism responses (photo, gravi, hydro)",
            "Germination conditions",
            "Allelopathy effects",
        ],
        "typical_materials": [
            "plants", "light source", "CO2 sensor", "potometer", "growth chambers", "soil",
        ],
    },
    {
        "id": "animal_behavior",
        "name": "Animal Behavior (Ethology)",
        "description": "Study of animal behaviors, learning, and responses to stimuli",
        "phenomena_examples": [
            "Invertebrate taxis and kinesis",
            "Learning in simple organisms",
            "Social behavior in insects",
            "Predator-prey interactions",
            "Circadian rhythms",
        ],
        "typical_materials": [
            "choice chambers", "invertebrates", "mazes", "timers",
            "environmental controls", "video recording",
        ],
    },
    {
        "id": "genetics",
        "name": "Genetics & Heredity",
        "description": "Study of inheritance, DNA, mutations, and genetic variation",
        "phenomena_examples": [
            "Mendelian inheritance in fast-breeding organisms",
            "DNA extraction and visualization",
            "Mutation rates in microbes",
            "Genetic variation in populations",
            "Epigenetic effects",
        ],
        "typical_materials": [
            "fast-breeding organisms", "DNA extraction kit", "gel electrophoresis",
            "microscope", "statistical tools",
        ],
    },
    {
        "id": "cell_biology",
        "name": "Cell Biology",
        "description": "Study of cell structure, function, division, and organelles",
        "phenomena_examples": [
            "Osmosis in plant cells",
            "Cell membrane permeability",
            "Mitosis observation",
            "Enzyme localization",
            "Cell respiration rates",
        ],
        "typical_materials": [
            "microscope", "stains", "onion cells", "sucrose solutions", "respirometer", "slides",
        ],
    },
    {
        "id": "ecology",
        "name": "Ecology",
        "description": (
            "Study of organisms and their interactions with environment and each other"
        ),
        "phenomena_examples": [
            "Population dynamics",
            "Species diversity indices",
            "Predator-prey cycles",
            "Ecological succession",
            "Carrying capacity limits",
        ],
        "typical_materials": [
            "quadrats", "sampling equipment", "identification guides",
            "environmental sensors", "data loggers",
        ],
    },
    {
        "id": "biochemistry",
        "name": "Biochemistry",
        "description": (
            "Study of chemical processes in living organisms, enzymes, and metabolism"
        ),
        "phenomena_examples": [
            "Enzyme kinetics (catalase, amylase)",
            "Protein denaturation",
            "Metabolic pathway analysis",
            "Vitamin content analysis",
            "Lipid and carbohydrate tests",
        ],
        "typical_materials": [
            "spectrophotometer", "enzyme sources", "substrates", "buffers",
            "water bath", "test reagents",
        ],
    },
    {
        "id": "human_physiology",
        "name": "Human Physiology",
        "description": (
            "Study of human body functions, reflexes, and physiological responses"
        ),
        "phenomena_examples": [
            "Heart rate variability",
            "Reaction time measurements",
            "Respiratory rate changes",
            "Skin sensitivity mapping",
            "Exercise effects on physiology",
        ],
        "typical_materials": [
            "heart rate monitor", "spirometer", "stopwatch",
            "blood pressure cuff", "exercise equipment",
        ],
    },
    {
        "id": "immunology",
        "name": "Immunology",
        "description": "Study of immune system, antibodies, and disease resistance",
        "phenomena_examples": [
            "Antibody-antigen reactions",
            "Blood typing",
            "Antimicrobial properties of natural substances",
            "Inflammation responses",
            "Vaccination principles",
        ],
        "typical_materials": [
            "blood typing kits", "ELISA plates", "antisera",
            "natural antimicrobials", "microscope",
        ],
    },
    {
        "id": "evolutionary_biology",
        "name": "Evolutionary Biology",
        "description": "Study of evolution, adaptation, natural selection, and speciation",
        "phenomena_examples": [
            "Adaptation in fast-reproducing organisms",
            "Artificial selection experiments",
            "Comparative anatomy studies",
            "Hardy-Weinberg equilibrium",
            "Evolutionary fitness measurements",
        ],
        "typical_materials": [
            "fast-breeding organisms", "selection pressures", "measurement tools",
            "statistical software", "fossil replicas",
        ],
    },
    {
        "id": "neurobiology",
        "name": "Neurobiology",
        "description": "Study of nervous system, neural responses, and sensory processing",
        "phenomena_examples": [
            "Sensory threshold determination",
            "Reflex arc timing",
            "Learning and memory tests",
            "Nerve impulse conduction",
            "Sensory adaptation",
        ],
        "typical_materials": [
            "stimuli generators", "timers", "EEG sensors", "response buttons", "maze equipment",
        ],
    },
    {
        "id": "environmental_biology",
        "name": "Environmental Biology",
        "description": "Study of environmental impacts on organisms and ecosystems",
        "phenomena_examples": [
            "Pollution effects on organisms",
            "Bioindicator species responses",
            "Water quality bioassays",
            "Climate effects on phenology",
            "Habitat fragmentation impacts",
        ],
        "typical_materials": [
            "bioindicator organisms", "water quality kits", "environmental chambers",
            "sampling equipment", "data loggers",
        ],
    },
]

# =============================================================================
# UNIFIED DOMAIN ACCESS
# =============================================================================

# Map field names to their domain lists
DOMAINS_BY_FIELD: dict[str, list[dict[str, Any]]] = {
    "physics": PHYSICS_DOMAINS,
    "chemistry": CHEMISTRY_DOMAINS,
    "biology": BIOLOGY_DOMAINS,
}


def get_domains_for_field(field: str) -> list[dict[str, Any]]:
    """Get all domains for a specific field.

    Args:
        field: Field name ('physics', 'chemistry', or 'biology')

    Returns:
        List of domain dicts for that field, empty list if field not found
    """
    return DOMAINS_BY_FIELD.get(field.lower(), [])


def get_domain_description(domain_id: str, field: str | None = None) -> dict[str, Any] | None:
    """Get detailed description for a specific domain.

    Args:
        domain_id: Domain identifier (e.g., 'fluid_dynamics', 'microbiology')
        field: Optional field to search in. If None, searches all fields.

    Returns:
        Domain dict with name, description, examples, materials
        None if domain_id not found
    """
    if field:
        # Search only in specified field
        domains = get_domains_for_field(field)
        for domain in domains:
            if domain["id"] == domain_id:
                return domain
        return None
    else:
        # Search all fields
        for field_domains in DOMAINS_BY_FIELD.values():
            for domain in field_domains:
                if domain["id"] == domain_id:
                    return domain
        return None


def get_all_domain_names(field: str | None = None) -> list[str]:
    """Get list of all domain names for display.

    Args:
        field: Optional field filter. If None, returns names from all fields.

    Returns:
        List of domain names (e.g., ['Fluid Dynamics', 'Classical Mechanics', ...])
    """
    if field:
        return [domain["name"] for domain in get_domains_for_field(field)]
    else:
        names: list[str] = []
        for field_domains in DOMAINS_BY_FIELD.values():
            names.extend(domain["name"] for domain in field_domains)
        return names


def get_all_domain_ids(field: str | None = None) -> list[str]:
    """Get list of all domain IDs for validation.

    Args:
        field: Optional field filter. If None, returns IDs from all fields.

    Returns:
        List of domain IDs (e.g., ['fluid_dynamics', 'mechanics', ...])
    """
    if field:
        return [domain["id"] for domain in get_domains_for_field(field)]
    else:
        ids: list[str] = []
        for field_domains in DOMAINS_BY_FIELD.values():
            ids.extend(domain["id"] for domain in field_domains)
        return ids
