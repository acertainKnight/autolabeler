"""
Core components for AutoLabeler.

This module provides the foundational services and utilities for the
AutoLabeler system, organized into specialized submodules.
"""

# Base components
from .base import ConfigurableComponent, ProgressTracker, BatchProcessor

# Configuration classes
from .configs import (
    LabelingConfig,
    BatchConfig,
    GenerationConfig,
    EvaluationConfig,
    RuleGenerationConfig,
    EnsembleConfig,
    PromptConfig,
    KnowledgeBaseConfig,
    DataSplitConfig,
    ComponentConfig
)

# Services
from .labeling import LabelingService
from .data import DataSplitService
from .evaluation import EvaluationService
from .generation import SyntheticGenerationService
from .rules import RuleGenerationService
from .ensemble import EnsembleService, ModelConfig, EnsembleResult
from .knowledge import KnowledgeStore, PromptManager, PromptRecord

__all__ = [
    # Base
    "ConfigurableComponent",
    "ProgressTracker",
    "BatchProcessor",
    # Configs
    "LabelingConfig",
    "BatchConfig",
    "GenerationConfig",
    "EvaluationConfig",
    "RuleGenerationConfig",
    "EnsembleConfig",
    "PromptConfig",
    "KnowledgeBaseConfig",
    "DataSplitConfig",
    "ComponentConfig",
    # Services
    "LabelingService",
    "DataSplitService",
    "EvaluationService",
    "SyntheticGenerationService",
    "RuleGenerationService",
    "EnsembleService",
    "ModelConfig",
    "EnsembleResult",
    # Knowledge
    "KnowledgeStore",
    "PromptManager",
    "PromptRecord"
]
