"""Workshop type enum for the Question Workshop system."""

from enum import StrEnum


class WorkshopType(StrEnum):
    """Available question workshop pipeline types."""

    PHENOMENON = "phenomenon"
    CLAIMS = "claims"
    HYPOTHESIS = "hypothesis"
    GAPS = "gaps"
    QUESTIONED = "questioned"
    RETRACTIONS = "retractions"
    THEORY = "theory"
    FORGE = "forge"
    MODELING = "modeling"
