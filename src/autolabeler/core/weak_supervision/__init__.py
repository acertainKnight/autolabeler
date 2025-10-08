"""
Weak Supervision module for programmatic labeling at scale.

This module implements weak supervision with Snorkel/FlyingSquid for aggregating
noisy labeling functions into high-quality training labels, enabling 10-100× faster
annotation compared to manual labeling.
"""

from .labeling_functions import (
    ABSTAIN,
    NEGATIVE,
    POSITIVE,
    LabelingFunction,
    create_keyword_lf,
    create_regex_lf,
    create_length_lf,
)
from .snorkel_integrator import WeakSupervisionService

__all__ = [
    "ABSTAIN",
    "NEGATIVE",
    "POSITIVE",
    "LabelingFunction",
    "create_keyword_lf",
    "create_regex_lf",
    "create_length_lf",
    "WeakSupervisionService",
]
