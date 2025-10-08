"""Prompt optimization module for AutoLabeler."""

from .dspy_optimizer import (
    DSPyOptimizer,
    DSPyConfig,
    DSPyOptimizationResult,
    LabelingSignature,
    LabelingModule,
)

__all__ = [
    'DSPyOptimizer',
    'DSPyConfig',
    'DSPyOptimizationResult',
    'LabelingSignature',
    'LabelingModule',
]
