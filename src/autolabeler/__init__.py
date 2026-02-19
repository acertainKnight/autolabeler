"""
AutoLabeler: Unified Data Labeling Automation Service

A modern data labeling service using evidence-based LLM jury pipelines with
heterogeneous models, confidence calibration, and structured markdown prompts.
"""

__version__ = "2.0.0"

# New unified pipeline
from .core.labeling import LabelingPipeline, LabelResult
from .core.dataset_config import DatasetConfig, ModelConfig
from .core.prompts import PromptRegistry
from .core.quality import ConfidenceScorer
from .core.llm_providers import (
    LLMProvider,
    LLMResponse,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    OpenRouterProvider,
    get_provider,
)

# Evaluation and quality monitoring
from .core.evaluation import EvaluationService
from .core.quality import QualityMonitor, ConfidenceCalibrator

# DSPy optimization
from .core.optimization import DSPyOptimizer, DSPyConfig, DSPyOptimizationResult

# Utilities
from .core.utils import evaluation_utils, data_utils

__all__ = [
    # Main pipeline
    "LabelingPipeline",
    "LabelResult",
    "DatasetConfig",
    "ModelConfig",
    "PromptRegistry",
    # LLM providers
    "LLMProvider",
    "LLMResponse",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "OpenRouterProvider",
    "get_provider",
    # Quality & confidence
    "ConfidenceScorer",
    "QualityMonitor",
    "ConfidenceCalibrator",
    # Evaluation
    "EvaluationService",
    # DSPy optimization
    "DSPyOptimizer",
    "DSPyConfig",
    "DSPyOptimizationResult",
    # Utilities
    "evaluation_utils",
    "data_utils",
]
