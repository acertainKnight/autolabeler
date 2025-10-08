"""
Active Learning module for intelligent sample selection.

This module implements active learning strategies to reduce annotation costs
by 40-70% through intelligent sample selection.
"""

from .sampler import ActiveLearningSampler, ALState
from .strategies import (
    SamplingStrategy,
    UncertaintySampler,
    DiversitySampler,
    CommitteeSampler,
    HybridSampler,
)
from .stopping_criteria import StoppingCriteria

__all__ = [
    "ActiveLearningSampler",
    "ALState",
    "SamplingStrategy",
    "UncertaintySampler",
    "DiversitySampler",
    "CommitteeSampler",
    "HybridSampler",
    "StoppingCriteria",
]
