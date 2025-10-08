"""Multi-agent system for specialized annotation tasks."""

from .agent_system import (
    AgentConfig,
    CoordinatorAgent,
    EntityRecognitionAgent,
    MultiAgentConfig,
    RelationExtractionAgent,
    SentimentAgent,
    SpecializedAgent,
    ValidatorAgent,
)

__all__ = [
    "SpecializedAgent",
    "AgentConfig",
    "EntityRecognitionAgent",
    "RelationExtractionAgent",
    "SentimentAgent",
    "ValidatorAgent",
    "CoordinatorAgent",
    "MultiAgentConfig",
]
