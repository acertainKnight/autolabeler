"""
AutoLabeler: Advanced AI-Powered Data Labeling System

A sophisticated data labeling system that leverages Large Language Models (LLMs)
with Retrieval-Augmented Generation (RAG), ensemble learning, and synthetic data
generation to automatically label text datasets with high accuracy and explainability.
"""

__version__ = "2.0.0"

# Main interface
from .autolabeler_v2 import AutoLabelerV2 as AutoLabeler

# Core services and configs for advanced usage
from .core import (
    LabelingService,
    DataSplitService,
    EvaluationService,
    SyntheticGenerationService,
    RuleGenerationService,
    EnsembleService,
    KnowledgeStore,
    PromptManager,
    LabelingConfig,
    BatchConfig,
    GenerationConfig,
    EvaluationConfig,
    RuleGenerationConfig,
    EnsembleConfig,
    ModelConfig,
)

# Foundational settings
from .config import Settings

# Data models
from .models import LabelResponse

__all__ = [
    # Main interface
    "AutoLabeler",
    "Settings",
    # Core services
    "LabelingService",
    "DataSplitService",
    "EvaluationService",
    "SyntheticGenerationService",
    "RuleGenerationService",
    "EnsembleService",
    "KnowledgeStore",
    "PromptManager",
    # Configuration
    "LabelingConfig",
    "BatchConfig",
    "GenerationConfig",
    "EvaluationConfig",
    "RuleGenerationConfig",
    "EnsembleConfig",
    "ModelConfig",
    # Data models
    "LabelResponse",
]
