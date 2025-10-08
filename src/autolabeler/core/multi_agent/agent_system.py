"""
Multi-agent system for specialized annotation tasks.

This module implements a coordinator-based multi-agent architecture where
specialized agents handle different types of annotation tasks (NER, relations,
sentiment, etc.) with intelligent routing and performance tracking.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from ...config import Settings
from ..llm_providers import get_llm_client


class AgentConfig(BaseModel):
    """Configuration for a specialized agent."""

    agent_id: str = Field(description="Unique agent identifier")
    agent_type: str = Field(description="Type of agent (ner, relations, sentiment, etc.)")
    llm_config: dict[str, Any] = Field(
        default_factory=dict, description="LLM configuration"
    )
    settings: Settings | None = Field(None, description="Application settings")
    entity_types: list[str] | None = Field(
        None, description="Entity types for NER agents"
    )
    relation_types: list[str] | None = Field(
        None, description="Relation types for relation agents"
    )
    sentiment_labels: list[str] | None = Field(
        None, description="Sentiment labels for sentiment agents"
    )
    validation_criteria: dict[str, Any] | None = Field(
        None, description="Validation criteria for validator agents"
    )
    max_retries: int = Field(3, description="Maximum retry attempts")
    timeout_seconds: int = Field(30, description="Request timeout in seconds")


class MultiAgentConfig(BaseModel):
    """Configuration for multi-agent coordinator."""

    coordinator_id: str = Field(
        default="coordinator_main", description="Coordinator identifier"
    )
    agent_configs: list[AgentConfig] = Field(
        default_factory=list, description="Agent configurations"
    )
    routing_strategy: str = Field(
        default="performance_based", description="Routing strategy"
    )
    enable_parallel: bool = Field(True, description="Enable parallel execution")
    max_concurrent_agents: int = Field(5, description="Max concurrent agent tasks")
    performance_window: int = Field(
        100, description="Window size for performance tracking"
    )


class SpecializedAgent(ABC):
    """Base class for specialized annotation agents."""

    def __init__(self, config: AgentConfig):
        """
        Initialize specialized agent.

        Args:
            config: Agent configuration.
        """
        self.config = config
        self.performance_history: list[dict[str, Any]] = []
        self.total_tasks = 0
        self.successful_tasks = 0
        self.llm_client = None

        if config.settings and config.llm_config:
            self.llm_client = get_llm_client(config.settings, config.llm_config)

    @abstractmethod
    def annotate(self, text: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Perform specialized annotation.

        Args:
            text: Text to annotate.
            context: Additional context for annotation.

        Returns:
            Annotation result dictionary.
        """
        pass

    @abstractmethod
    async def aannotate(self, text: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Perform specialized annotation asynchronously.

        Args:
            text: Text to annotate.
            context: Additional context for annotation.

        Returns:
            Annotation result dictionary.
        """
        pass

    @abstractmethod
    def can_handle(self, task_type: str) -> bool:
        """
        Check if agent can handle task type.

        Args:
            task_type: Type of task.

        Returns:
            True if agent can handle task.
        """
        pass

    def get_performance_score(self) -> float:
        """
        Calculate agent performance score.

        Returns:
            Performance score between 0 and 1.
        """
        if self.total_tasks == 0:
            return 0.5  # Default score for new agents

        success_rate = self.successful_tasks / self.total_tasks

        # Consider recent performance more heavily
        if len(self.performance_history) > 10:
            recent_success = sum(
                1 for h in self.performance_history[-10:] if h.get("success", False)
            )
            recent_rate = recent_success / 10
            # Weight: 70% recent, 30% overall
            return 0.7 * recent_rate + 0.3 * success_rate

        return success_rate

    def record_performance(self, success: bool, confidence: float = 0.0) -> None:
        """
        Record task performance.

        Args:
            success: Whether task was successful.
            confidence: Confidence score of prediction.
        """
        self.total_tasks += 1
        if success:
            self.successful_tasks += 1

        self.performance_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "confidence": confidence,
            }
        )

        # Keep history bounded
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]


class EntityRecognitionAgent(SpecializedAgent):
    """Specialized agent for named entity recognition."""

    def __init__(self, config: AgentConfig):
        """Initialize NER agent."""
        super().__init__(config)
        self.entity_types = config.entity_types or [
            "PERSON",
            "ORGANIZATION",
            "LOCATION",
        ]

    def can_handle(self, task_type: str) -> bool:
        """Check if agent handles NER tasks."""
        return task_type.lower() in ["ner", "entity_extraction", "named_entities"]

    def annotate(self, text: str, context: dict[str, Any]) -> dict[str, Any]:
        """Extract named entities from text."""
        if not self.llm_client:
            raise ValueError("LLM client not initialized")

        prompt = self._create_ner_prompt(text, self.entity_types)

        try:
            response = self.llm_client.invoke(prompt)

            # Parse response (simplified - in practice would use structured output)
            entities = self._parse_entities(response)

            self.record_performance(success=True, confidence=0.9)

            return {
                "entities": entities,
                "confidence": 0.9,
                "agent_id": self.config.agent_id,
                "agent_type": "entity_recognition",
            }

        except Exception as e:
            logger.error(f"NER agent failed: {e}")
            self.record_performance(success=False)
            raise

    async def aannotate(self, text: str, context: dict[str, Any]) -> dict[str, Any]:
        """Extract named entities asynchronously."""
        if not self.llm_client:
            raise ValueError("LLM client not initialized")

        prompt = self._create_ner_prompt(text, self.entity_types)

        try:
            response = await self.llm_client.ainvoke(prompt)
            entities = self._parse_entities(response)

            self.record_performance(success=True, confidence=0.9)

            return {
                "entities": entities,
                "confidence": 0.9,
                "agent_id": self.config.agent_id,
                "agent_type": "entity_recognition",
            }

        except Exception as e:
            logger.error(f"NER agent failed: {e}")
            self.record_performance(success=False)
            raise

    def _create_ner_prompt(self, text: str, entity_types: list[str]) -> str:
        """Create NER prompt."""
        return f"""Extract named entities from the following text.

Entity types to identify: {', '.join(entity_types)}

Text: {text}

For each entity, provide:
- text: the entity mention
- type: entity type
- start: start position
- end: end position

Format as JSON list."""

    def _parse_entities(self, response: Any) -> list[dict[str, Any]]:
        """Parse entities from response."""
        # Simplified parser - in practice would use structured output validation
        if hasattr(response, "content"):
            response = response.content

        # Return empty list as fallback
        return []


class RelationExtractionAgent(SpecializedAgent):
    """Specialized agent for relation extraction."""

    def __init__(self, config: AgentConfig):
        """Initialize relation extraction agent."""
        super().__init__(config)
        self.relation_types = config.relation_types or ["is_a", "part_of", "located_in"]

    def can_handle(self, task_type: str) -> bool:
        """Check if agent handles relation extraction."""
        return task_type.lower() in ["relations", "relation_extraction", "relationships"]

    def annotate(self, text: str, context: dict[str, Any]) -> dict[str, Any]:
        """Extract relations from text."""
        if not self.llm_client:
            raise ValueError("LLM client not initialized")

        entities = context.get("entities", [])
        prompt = self._create_relation_prompt(text, entities, self.relation_types)

        try:
            response = self.llm_client.invoke(prompt)
            relations = self._parse_relations(response)

            self.record_performance(success=True, confidence=0.85)

            return {
                "relations": relations,
                "confidence": 0.85,
                "agent_id": self.config.agent_id,
                "agent_type": "relation_extraction",
            }

        except Exception as e:
            logger.error(f"Relation agent failed: {e}")
            self.record_performance(success=False)
            raise

    async def aannotate(self, text: str, context: dict[str, Any]) -> dict[str, Any]:
        """Extract relations asynchronously."""
        if not self.llm_client:
            raise ValueError("LLM client not initialized")

        entities = context.get("entities", [])
        prompt = self._create_relation_prompt(text, entities, self.relation_types)

        try:
            response = await self.llm_client.ainvoke(prompt)
            relations = self._parse_relations(response)

            self.record_performance(success=True, confidence=0.85)

            return {
                "relations": relations,
                "confidence": 0.85,
                "agent_id": self.config.agent_id,
                "agent_type": "relation_extraction",
            }

        except Exception as e:
            logger.error(f"Relation agent failed: {e}")
            self.record_performance(success=False)
            raise

    def _create_relation_prompt(
        self, text: str, entities: list[dict], relation_types: list[str]
    ) -> str:
        """Create relation extraction prompt."""
        return f"""Extract relationships between entities in the text.

Entities: {entities}
Relation types: {', '.join(relation_types)}

Text: {text}

For each relation, provide:
- subject: subject entity
- predicate: relation type
- object: object entity
- confidence: confidence score

Format as JSON list."""

    def _parse_relations(self, response: Any) -> list[dict[str, Any]]:
        """Parse relations from response."""
        if hasattr(response, "content"):
            response = response.content
        return []


class SentimentAgent(SpecializedAgent):
    """Specialized agent for sentiment analysis."""

    def __init__(self, config: AgentConfig):
        """Initialize sentiment agent."""
        super().__init__(config)
        self.sentiment_labels = config.sentiment_labels or [
            "positive",
            "negative",
            "neutral",
        ]

    def can_handle(self, task_type: str) -> bool:
        """Check if agent handles sentiment analysis."""
        return task_type.lower() in ["sentiment", "sentiment_analysis", "opinion"]

    def annotate(self, text: str, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze sentiment of text."""
        if not self.llm_client:
            raise ValueError("LLM client not initialized")

        prompt = self._create_sentiment_prompt(text, self.sentiment_labels)

        try:
            response = self.llm_client.invoke(prompt)
            sentiment = self._parse_sentiment(response)

            self.record_performance(success=True, confidence=sentiment["confidence"])

            return {
                "sentiment": sentiment["label"],
                "confidence": sentiment["confidence"],
                "agent_id": self.config.agent_id,
                "agent_type": "sentiment_analysis",
            }

        except Exception as e:
            logger.error(f"Sentiment agent failed: {e}")
            self.record_performance(success=False)
            raise

    async def aannotate(self, text: str, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze sentiment asynchronously."""
        if not self.llm_client:
            raise ValueError("LLM client not initialized")

        prompt = self._create_sentiment_prompt(text, self.sentiment_labels)

        try:
            response = await self.llm_client.ainvoke(prompt)
            sentiment = self._parse_sentiment(response)

            self.record_performance(success=True, confidence=sentiment["confidence"])

            return {
                "sentiment": sentiment["label"],
                "confidence": sentiment["confidence"],
                "agent_id": self.config.agent_id,
                "agent_type": "sentiment_analysis",
            }

        except Exception as e:
            logger.error(f"Sentiment agent failed: {e}")
            self.record_performance(success=False)
            raise

    def _create_sentiment_prompt(self, text: str, labels: list[str]) -> str:
        """Create sentiment analysis prompt."""
        return f"""Analyze the sentiment of the following text.

Available labels: {', '.join(labels)}

Text: {text}

Provide:
- label: sentiment label
- confidence: confidence score (0-1)

Format as JSON."""

    def _parse_sentiment(self, response: Any) -> dict[str, Any]:
        """Parse sentiment from response."""
        if hasattr(response, "content"):
            response = response.content
        # Simplified - return default
        return {"label": "neutral", "confidence": 0.8}


class ValidatorAgent(SpecializedAgent):
    """Specialized agent for quality validation."""

    def __init__(self, config: AgentConfig):
        """Initialize validator agent."""
        super().__init__(config)
        self.validation_criteria = config.validation_criteria or {}

    def can_handle(self, task_type: str) -> bool:
        """Check if agent handles validation."""
        return task_type.lower() in ["validation", "quality_check", "verify"]

    def annotate(self, text: str, context: dict[str, Any]) -> dict[str, Any]:
        """Validate annotation quality."""
        annotations = context.get("annotations", {})

        validation_results = {
            "is_valid": True,
            "issues": [],
            "confidence": 1.0,
            "agent_id": self.config.agent_id,
            "agent_type": "validation",
        }

        # Check for missing required fields
        for field in self.validation_criteria.get("required_fields", []):
            if field not in annotations:
                validation_results["is_valid"] = False
                validation_results["issues"].append(f"Missing required field: {field}")

        # Check confidence thresholds
        min_confidence = self.validation_criteria.get("min_confidence", 0.5)
        for key, value in annotations.items():
            if isinstance(value, dict) and "confidence" in value:
                if value["confidence"] < min_confidence:
                    validation_results["is_valid"] = False
                    validation_results["issues"].append(
                        f"Low confidence for {key}: {value['confidence']}"
                    )

        self.record_performance(success=True, confidence=1.0)
        return validation_results

    async def aannotate(self, text: str, context: dict[str, Any]) -> dict[str, Any]:
        """Validate annotation quality asynchronously."""
        # Validation is fast, so sync and async are the same
        return self.annotate(text, context)


class CoordinatorAgent:
    """Coordinates multiple specialized agents for complex annotation tasks."""

    def __init__(self, config: MultiAgentConfig):
        """
        Initialize coordinator agent.

        Args:
            config: Multi-agent configuration.
        """
        self.config = config
        self.agents: dict[str, SpecializedAgent] = {}
        self._register_agents()

    def _register_agents(self) -> None:
        """Register specialized agents from config."""
        for agent_config in self.config.agent_configs:
            agent_class = self._get_agent_class(agent_config.agent_type)
            if agent_class:
                agent = agent_class(agent_config)
                self.agents[agent_config.agent_id] = agent
                logger.info(f"Registered agent: {agent_config.agent_id}")

    def _get_agent_class(self, agent_type: str) -> type[SpecializedAgent] | None:
        """Get agent class by type."""
        agent_map = {
            "ner": EntityRecognitionAgent,
            "entity_extraction": EntityRecognitionAgent,
            "relations": RelationExtractionAgent,
            "relation_extraction": RelationExtractionAgent,
            "sentiment": SentimentAgent,
            "sentiment_analysis": SentimentAgent,
            "validation": ValidatorAgent,
        }
        return agent_map.get(agent_type.lower())

    def route_task(
        self, text: str, task_type: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Route task to appropriate agent.

        Args:
            text: Text to process.
            task_type: Type of task.
            context: Additional context.

        Returns:
            Task result.
        """
        context = context or {}

        # Find capable agents
        capable_agents = [
            agent for agent in self.agents.values() if agent.can_handle(task_type)
        ]

        if not capable_agents:
            raise ValueError(f"No agent can handle task type: {task_type}")

        # Select best agent based on performance
        best_agent = self._select_best_agent(capable_agents)

        # Execute task
        result = best_agent.annotate(text, context)

        return result

    async def aroute_task(
        self, text: str, task_type: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Route task to appropriate agent asynchronously."""
        context = context or {}

        capable_agents = [
            agent for agent in self.agents.values() if agent.can_handle(task_type)
        ]

        if not capable_agents:
            raise ValueError(f"No agent can handle task type: {task_type}")

        best_agent = self._select_best_agent(capable_agents)
        result = await best_agent.aannotate(text, context)

        return result

    def parallel_annotation(
        self, text: str, task_types: list[str], context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute multiple annotation tasks in parallel.

        Args:
            text: Text to process.
            task_types: List of task types.
            context: Additional context.

        Returns:
            Merged results from all tasks.
        """
        context = context or {}

        async def run_parallel():
            tasks = [self.aroute_task(text, task_type, context) for task_type in task_types]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            merged = {}
            for task_type, result in zip(task_types, results):
                if isinstance(result, Exception):
                    logger.error(f"Task {task_type} failed: {result}")
                    merged[task_type] = {"error": str(result)}
                else:
                    merged[task_type] = result

            return merged

        return asyncio.run(run_parallel())

    def _select_best_agent(self, agents: list[SpecializedAgent]) -> SpecializedAgent:
        """
        Select best agent based on performance.

        Args:
            agents: List of capable agents.

        Returns:
            Best performing agent.
        """
        if len(agents) == 1:
            return agents[0]

        # Select based on performance score
        best_agent = max(agents, key=lambda a: a.get_performance_score())
        return best_agent

    def get_performance_report(self) -> dict[str, Any]:
        """
        Get performance report for all agents.

        Returns:
            Performance report dictionary.
        """
        report = {
            "coordinator_id": self.config.coordinator_id,
            "total_agents": len(self.agents),
            "agents": {},
        }

        for agent_id, agent in self.agents.items():
            report["agents"][agent_id] = {
                "agent_type": agent.config.agent_type,
                "total_tasks": agent.total_tasks,
                "successful_tasks": agent.successful_tasks,
                "success_rate": agent.get_performance_score(),
                "can_handle": [
                    task_type
                    for task_type in ["ner", "relations", "sentiment", "validation"]
                    if agent.can_handle(task_type)
                ],
            }

        return report
