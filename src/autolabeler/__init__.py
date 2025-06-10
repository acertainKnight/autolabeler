"""LLM autolabeler package."""

from .config import Settings
from .labeler import AutoLabeler
from .models import (
    LabelResponse,
    SyntheticExample,
    SyntheticBatch,
    LabelingRule,
    RuleSet,
    RuleGenerationResult,
    RuleUpdateResult,
)
from .openrouter import OpenRouterClient
from .corporate import CorporateOpenAIClient
from .ensemble import EnsembleLabeler, EnsembleResult
from .model_config import ModelConfig, EnsembleMethod, ModelRun
from .knowledge_base import KnowledgeBase
from .prompt_store import PromptStore, PromptRecord
from .synthetic_generator import SyntheticDataGenerator
from .rule_generator import RuleGenerator
from .cli import cli

__all__ = [
    "AutoLabeler",
    "LabelResponse",
    "SyntheticExample",
    "SyntheticBatch",
    "LabelingRule",
    "RuleSet",
    "RuleGenerationResult",
    "RuleUpdateResult",
    "OpenRouterClient",
    "CorporateOpenAIClient",
    "Settings",
    "EnsembleLabeler",
    "EnsembleResult",
    "ModelConfig",
    "EnsembleMethod",
    "ModelRun",
    "KnowledgeBase",
    "PromptStore",
    "PromptRecord",
    "SyntheticDataGenerator",
    "RuleGenerator",
    "cli"
]
