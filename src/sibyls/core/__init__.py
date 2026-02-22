"""
Core components for the Sibyls labeling pipeline.

This module provides the foundational services for evidence-based
labeling with heterogeneous jury voting and confidence calibration.
"""

# Dataset configuration
from .dataset_config import DatasetConfig, ModelConfig

# Prompt management
from .prompts import PromptRegistry

# LLM providers
from .llm_providers import (
    LLMProvider,
    LLMResponse,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    OpenRouterProvider,
    get_provider,
)

# Labeling pipeline
from .labeling import LabelingPipeline, LabelResult

# Quality & confidence
from .quality import ConfidenceScorer, QualityMonitor, ConfidenceCalibrator

# Evaluation
from .evaluation import EvaluationService

# Optimization
from .optimization import DSPyOptimizer, DSPyConfig, DSPyOptimizationResult

# Utilities
from .utils import evaluation_utils, data_utils

__all__ = [
    # Configuration
    "DatasetConfig",
    "ModelConfig",
    # Prompts
    "PromptRegistry",
    # LLM providers
    "LLMProvider",
    "LLMResponse",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "OpenRouterProvider",
    "get_provider",
    # Labeling
    "LabelingPipeline",
    "LabelResult",
    # Quality & confidence
    "ConfidenceScorer",
    "QualityMonitor",
    "ConfidenceCalibrator",
    # Evaluation
    "EvaluationService",
    # Optimization
    "DSPyOptimizer",
    "DSPyConfig",
    "DSPyOptimizationResult",
    # Utilities
    "evaluation_utils",
    "data_utils",
]
