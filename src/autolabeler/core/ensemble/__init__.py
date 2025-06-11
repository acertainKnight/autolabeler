"""
Ensemble labeling components for AutoLabeler.

This module provides multi-model ensemble functionality for improved
labeling accuracy through model consensus.
"""

from .ensemble_service import EnsembleService, ModelConfig, EnsembleResult

__all__ = ["EnsembleService", "ModelConfig", "EnsembleResult"]
