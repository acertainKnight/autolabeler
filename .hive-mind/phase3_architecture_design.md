# Phase 3 Architecture Design Document

**Date:** October 8, 2025
**Researcher Agent:** Phase 3 Investigation
**Status:** Complete ✅

---

## Executive Summary

This document defines the **detailed architecture** for Phase 3 advanced features integration into AutoLabeler. Phase 3 adds an **intelligence layer** on top of Phase 1's quality foundations and Phase 2's core capabilities, enabling:

1. **Multi-Agent Architecture** - Specialist agents with coordination
2. **Drift Detection** - Real-time monitoring and alerting
3. **STAPLE Algorithm** - Weighted multi-annotator consensus
4. **DPO/RLHF** - Task-specific model alignment
5. **Constitutional AI** - Principled annotation consistency

**Design Principles:**
- **Modularity:** Each feature is independently usable
- **Backward Compatibility:** All existing APIs continue to work
- **Performance:** Minimal latency overhead (<2×)
- **Extensibility:** Easy to add new agents, detectors, principles

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Module Architecture](#2-module-architecture)
3. [Multi-Agent System Design](#3-multi-agent-system-design)
4. [Drift Detection System](#4-drift-detection-system)
5. [STAPLE Integration](#5-staple-integration)
6. [DPO/RLHF Pipeline](#6-dporlhf-pipeline)
7. [Constitutional AI Framework](#7-constitutional-ai-framework)
8. [Data Flow and Integration](#8-data-flow-and-integration)
9. [API Design](#9-api-design)
10. [Testing Strategy](#10-testing-strategy)
11. [Deployment Architecture](#11-deployment-architecture)
12. [Performance Considerations](#12-performance-considerations)

---

## 1. System Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AutoLabeler Phase 3                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Phase 1    │  │   Phase 2    │  │   Phase 3    │        │
│  │   Quality    │  │    Core      │  │  Advanced    │        │
│  │  Foundations │  │ Capabilities │  │Intelligence  │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
│         │                 │                  │                 │
│         v                 v                  v                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │            AutoLabeler Core Interface                    │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                 Phase 3 Components                       │  │
│  │                                                           │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐        │  │
│  │  │Multi-Agent │  │   Drift    │  │  STAPLE    │        │  │
│  │  │   System   │  │ Detection  │  │ Aggregation│        │  │
│  │  └────────────┘  └────────────┘  └────────────┘        │  │
│  │                                                           │  │
│  │  ┌────────────┐  ┌────────────┐                         │  │
│  │  │   DPO/     │  │Constitutional                        │  │
│  │  │   RLHF     │  │     AI     │                         │  │
│  │  └────────────┘  └────────────┘                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │            Phase 1 & 2 Foundation                        │  │
│  │  • LabelingService  • EnsembleService                    │  │
│  │  • QualityMonitor   • KnowledgeStore                     │  │
│  │  • DSPy Optimizer   • GraphRAG/RAPTOR                    │  │
│  │  • Active Learning  • Data Versioning                    │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Interactions

```
User Request
     │
     v
┌─────────────────────────────────────┐
│   AutoLabelerPhase3                 │
│   • Route to appropriate component  │
│   • Coordinate multi-feature flows  │
└────┬──────────────┬─────────────────┘
     │              │
     v              v
┌──────────┐   ┌────────────────────┐
│Standard  │   │Advanced (Phase 3)  │
│Labeling  │   │Features            │
│(Phase 1) │   └──────┬─────────────┘
└──────────┘          │
                      │
        ┌─────────────┼─────────────┬────────────────┐
        │             │             │                │
        v             v             v                v
┌─────────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐
│Multi-Agent  │ │  Drift   │ │ STAPLE   │ │Constitutional  │
│Coordination │ │Detection │ │          │ │    AI          │
└─────────────┘ └──────────┘ └──────────┘ └────────────────┘
        │             │             │                │
        └─────────────┴─────────────┴────────────────┘
                      │
                      v
            ┌─────────────────────┐
            │  Quality Monitor    │
            │  (Phase 1)          │
            └─────────────────────┘
```

### 1.3 Design Decisions

**Decision 1: Modular Phase 3 Components**
- **Rationale:** Each feature is independently usable, can be enabled/disabled
- **Trade-off:** Slightly more complex API, but maximum flexibility
- **Alternative Considered:** Monolithic Phase 3 service (rejected: too rigid)

**Decision 2: Extend Existing Services**
- **Rationale:** Maintain backward compatibility, gradual migration
- **Trade-off:** Some code duplication, but smoother adoption
- **Alternative Considered:** Replace existing services (rejected: breaking changes)

**Decision 3: LangGraph for Multi-Agent**
- **Rationale:** Purpose-built for agent coordination, mature ecosystem
- **Trade-off:** Additional dependency, but worth the abstraction
- **Alternative Considered:** Custom implementation (rejected: reinventing wheel)

**Decision 4: STAPLE from Scratch**
- **Rationale:** No mature Python library, algorithm is straightforward
- **Trade-off:** Maintenance burden, but full control
- **Alternative Considered:** Wrapper around medical imaging library (rejected: heavy dependencies)

---

## 2. Module Architecture

### 2.1 Directory Structure

```
src/autolabeler/
├── core/
│   ├── agents/                     # NEW - Multi-agent system
│   │   ├── __init__.py
│   │   ├── base.py                # Base agent classes
│   │   ├── specialist_agents.py   # NER, sentiment, etc.
│   │   ├── coordinator.py         # Task routing and aggregation
│   │   ├── validator.py           # Quality validation agent
│   │   └── config.py              # Agent configurations
│   │
│   ├── constitutional/             # NEW - Constitutional AI
│   │   ├── __init__.py
│   │   ├── principles.py          # Principle definitions
│   │   ├── annotator.py           # Constitutional annotator
│   │   ├── evaluator.py           # Adherence evaluation
│   │   └── config.py              # Constitutional configs
│   │
│   ├── drift/                      # NEW - Drift detection
│   │   ├── __init__.py
│   │   ├── statistical_tests.py   # PSI, KS test, chi-square
│   │   ├── domain_classifier.py   # Domain classifier approach
│   │   ├── embedding_drift.py     # Embedding-based detection
│   │   ├── monitor.py             # Comprehensive monitor
│   │   └── config.py              # Drift detection configs
│   │
│   ├── alignment/                  # NEW - DPO/RLHF
│   │   ├── __init__.py
│   │   ├── preference_data.py     # Preference dataset generation
│   │   ├── dpo_trainer.py         # DPO training wrapper
│   │   ├── evaluator.py           # Alignment evaluation
│   │   └── config.py              # Training configurations
│   │
│   ├── ensemble/                   # ENHANCED - Add STAPLE
│   │   ├── __init__.py
│   │   ├── ensemble_service.py    # Existing ensemble service
│   │   ├── staple.py              # NEW - STAPLE algorithm
│   │   └── aggregation.py         # NEW - Aggregation methods
│   │
│   ├── quality/                    # ENHANCED - Integrate drift
│   │   ├── __init__.py
│   │   ├── monitor.py             # Existing quality monitor
│   │   ├── calibrator.py          # Existing calibrator
│   │   └── drift_integration.py   # NEW - Drift integration
│   │
│   ├── labeling/                   # ENHANCED - Add Phase 3 support
│   │   ├── __init__.py
│   │   ├── labeling_service.py    # Existing service
│   │   └── advanced_service.py    # NEW - Phase 3 features
│   │
│   └── configs.py                  # ENHANCED - Add Phase 3 configs
│
├── autolabeler_v3.py              # NEW - Phase 3 main interface
│
└── cli.py                          # ENHANCED - Add Phase 3 commands
```

### 2.2 Module Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                    External Dependencies                     │
│  langgraph, evidently, transformers, trl, peft             │
└─────────┬───────────────────────────────────────────────────┘
          │
          v
┌─────────────────────────────────────────────────────────────┐
│                    Phase 3 Modules                          │
│  • agents (depends on: langgraph, langchain)                │
│  • drift (depends on: evidently, scipy, sklearn)            │
│  • alignment (depends on: transformers, trl, peft)          │
│  • constitutional (depends on: langchain, openai)           │
│  • staple (depends on: numpy, scipy)                        │
└─────────┬───────────────────────────────────────────────────┘
          │
          v
┌─────────────────────────────────────────────────────────────┐
│                Phase 1 & 2 Foundation                       │
│  • LabelingService • EnsembleService • QualityMonitor       │
│  • KnowledgeStore  • DSPy Optimizer  • GraphRAG             │
└─────────────────────────────────────────────────────────────┘
```

**Key Principles:**
- **No Circular Dependencies:** Phase 3 depends on Phase 1/2, not vice versa
- **Optional Imports:** Phase 3 modules use lazy imports for optional features
- **Feature Flags:** Enable/disable Phase 3 features without breaking Phase 1/2

---

## 3. Multi-Agent System Design

### 3.1 Agent Hierarchy

```
                    ┌──────────────────┐
                    │   Coordinator    │
                    │     Agent        │
                    │  • Task routing  │
                    │  • Aggregation   │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            v                v                v
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │  Specialist │  │  Specialist │  │  Validator  │
    │   Agent 1   │  │   Agent 2   │  │    Agent    │
    │             │  │             │  │             │
    │  • NER      │  │  • Sentiment│  │  • Quality  │
    │  • Entities │  │  • Opinion  │  │  • Checks   │
    └─────────────┘  └─────────────┘  └─────────────┘
```

### 3.2 Class Design

```python
# agents/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel, Field

class AgentCapability(str, Enum):
    """Agent capability types."""
    ENTITY_RECOGNITION = "entity_recognition"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RELATION_EXTRACTION = "relation_extraction"
    TEXT_CLASSIFICATION = "text_classification"
    QUALITY_VALIDATION = "quality_validation"

class AgentResult(BaseModel):
    """Result from an agent's processing."""
    agent_id: str
    agent_type: str
    annotations: Dict[str, Any]
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[AgentCapability],
        model_config: ModelConfig
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.model_config = model_config
        self.performance_history: List[Dict] = []

    @abstractmethod
    async def process(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> AgentResult:
        """Process text and return annotations."""
        pass

    @abstractmethod
    def get_confidence(self, result: AgentResult) -> float:
        """Calculate confidence for result."""
        pass

    def log_performance(self, result: AgentResult, ground_truth: Any = None):
        """Log performance metrics."""
        pass


# agents/specialist_agents.py
class EntityRecognitionAgent(BaseAgent):
    """Agent specialized in named entity recognition."""

    def __init__(self, agent_id: str, model_config: ModelConfig):
        super().__init__(
            agent_id=agent_id,
            agent_type="entity_recognition",
            capabilities=[AgentCapability.ENTITY_RECOGNITION],
            model_config=model_config
        )

    async def process(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> AgentResult:
        """Extract named entities from text."""
        # Use LLM to extract entities
        prompt = self._create_ner_prompt(text, context)
        response = await self._call_llm(prompt)

        entities = self._parse_entities(response)

        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            annotations={"entities": entities},
            confidence=self._calculate_confidence(entities),
            reasoning=f"Extracted {len(entities)} entities"
        )


class SentimentAnalysisAgent(BaseAgent):
    """Agent specialized in sentiment analysis."""

    def __init__(self, agent_id: str, model_config: ModelConfig):
        super().__init__(
            agent_id=agent_id,
            agent_type="sentiment_analysis",
            capabilities=[AgentCapability.SENTIMENT_ANALYSIS],
            model_config=model_config
        )

    async def process(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> AgentResult:
        """Analyze sentiment of text."""
        prompt = self._create_sentiment_prompt(text, context)
        response = await self._call_llm(prompt)

        sentiment = self._parse_sentiment(response)

        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            annotations={"sentiment": sentiment},
            confidence=sentiment.get("confidence", 0.5),
            reasoning=sentiment.get("reasoning", "")
        )


class ValidatorAgent(BaseAgent):
    """Agent that validates outputs from other agents."""

    def __init__(self, agent_id: str, model_config: ModelConfig):
        super().__init__(
            agent_id=agent_id,
            agent_type="validator",
            capabilities=[AgentCapability.QUALITY_VALIDATION],
            model_config=model_config
        )

    async def process(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> AgentResult:
        """Validate annotations from specialist agents."""
        # Get results from other agents
        agent_results: List[AgentResult] = context.get("agent_results", [])

        # Check for consistency
        consistency_score = self._check_consistency(agent_results)

        # Check for completeness
        completeness_score = self._check_completeness(agent_results, context)

        # Check for quality
        quality_score = self._check_quality(agent_results, text)

        validation = {
            "passed": all([
                consistency_score > 0.7,
                completeness_score > 0.8,
                quality_score > 0.7
            ]),
            "consistency_score": consistency_score,
            "completeness_score": completeness_score,
            "quality_score": quality_score
        }

        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            annotations=validation,
            confidence=(consistency_score + completeness_score + quality_score) / 3,
            reasoning=f"Validation: {'PASS' if validation['passed'] else 'FAIL'}"
        )


# agents/coordinator.py
class CoordinatorAgent:
    """Coordinates multiple specialist agents."""

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        aggregation_strategy: str = "weighted_vote"
    ):
        self.agents = agents
        self.aggregation_strategy = aggregation_strategy
        self.task_router = TaskRouter()

    async def coordinate(
        self,
        text: str,
        task_config: MultiAgentTaskConfig
    ) -> MultiAgentResult:
        """
        Coordinate multi-agent annotation.

        Steps:
        1. Decompose task into subtasks
        2. Route subtasks to appropriate agents
        3. Execute agents in parallel
        4. Validate results
        5. Aggregate final result
        """
        # 1. Decompose task
        subtasks = self.task_router.decompose(text, task_config)

        # 2 & 3. Route and execute
        results = await self._execute_agents(text, subtasks)

        # 4. Validate
        if task_config.enable_validation:
            validation_result = await self._validate(text, results)
            if not validation_result.annotations["passed"]:
                # Retry with different agents or escalate
                results = await self._retry_failed(text, subtasks, validation_result)

        # 5. Aggregate
        final_result = self._aggregate(results, task_config)

        return final_result

    async def _execute_agents(
        self,
        text: str,
        subtasks: List[Subtask]
    ) -> List[AgentResult]:
        """Execute agents in parallel."""
        tasks = []

        for subtask in subtasks:
            agent = self.agents.get(subtask.agent_type)
            if agent:
                tasks.append(agent.process(text, subtask.context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failures
        return [r for r in results if isinstance(r, AgentResult)]
```

### 3.3 LangGraph Integration

```python
# agents/langgraph_coordinator.py
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    """State shared across agents."""
    text: str
    task_config: MultiAgentTaskConfig
    entity_results: Optional[AgentResult]
    sentiment_results: Optional[AgentResult]
    validation_results: Optional[AgentResult]
    final_annotations: Optional[Dict]

def create_multiagent_graph() -> StateGraph:
    """Create LangGraph for multi-agent workflow."""
    workflow = StateGraph(AgentState)

    # Add agent nodes
    workflow.add_node("entity_agent", entity_agent_node)
    workflow.add_node("sentiment_agent", sentiment_agent_node)
    workflow.add_node("validator_agent", validator_agent_node)
    workflow.add_node("aggregator", aggregator_node)

    # Define edges (workflow)
    workflow.add_edge("START", "entity_agent")
    workflow.add_edge("START", "sentiment_agent")  # Parallel execution
    workflow.add_edge("entity_agent", "validator_agent")
    workflow.add_edge("sentiment_agent", "validator_agent")
    workflow.add_edge("validator_agent", "aggregator")
    workflow.add_edge("aggregator", END)

    return workflow.compile()

async def entity_agent_node(state: AgentState) -> AgentState:
    """Entity recognition agent node."""
    agent = EntityRecognitionAgent("ner", model_config)
    result = await agent.process(state["text"], {})
    state["entity_results"] = result
    return state

async def sentiment_agent_node(state: AgentState) -> AgentState:
    """Sentiment analysis agent node."""
    agent = SentimentAnalysisAgent("sentiment", model_config)
    result = await agent.process(state["text"], {})
    state["sentiment_results"] = result
    return state
```

---

## 4. Drift Detection System

### 4.1 Architecture

```
┌────────────────────────────────────────────────────────────┐
│              Drift Detection System                        │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │         ComprehensiveDriftMonitor                    │ │
│  │  • Manages baseline and current data                 │ │
│  │  • Coordinates multiple detection methods            │ │
│  │  • Generates alerts and recommendations              │ │
│  └────────┬─────────────────────────────────────────────┘ │
│           │                                                │
│  ┌────────┴─────────┬───────────────┬────────────────┐   │
│  │                  │               │                │   │
│  v                  v               v                v   │
│ ┌────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐  │
│ │  PSI   │   │    KS    │   │ Domain   │   │Embedding│  │
│ │ Test   │   │   Test   │   │Classifier│   │  Drift  │  │
│ └────────┘   └──────────┘   └──────────┘   └─────────┘  │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │               Alert System                           │ │
│  │  • Email/Slack notifications                         │ │
│  │  • Dashboard updates                                 │ │
│  │  • Automated remediation triggers                    │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

### 4.2 Class Design

```python
# drift/monitor.py
from typing import Protocol, Dict, List, Optional
from enum import Enum

class DriftTest(str, Enum):
    """Available drift detection tests."""
    PSI = "psi"
    KS = "kolmogorov_smirnov"
    DOMAIN_CLASSIFIER = "domain_classifier"
    EMBEDDING = "embedding_based"
    CHI_SQUARE = "chi_square"

class DriftAlert(BaseModel):
    """Drift detection alert."""
    timestamp: datetime
    severity: Literal["low", "moderate", "high", "critical"]
    tests_detecting_drift: int
    total_tests: int
    recommendation: str
    detailed_results: Dict[str, Any]

class DriftDetector(Protocol):
    """Protocol for drift detectors."""

    def detect(
        self,
        baseline_data: np.ndarray,
        current_data: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Detect drift between baseline and current data."""
        ...

class ComprehensiveDriftMonitor:
    """
    Comprehensive drift monitoring system.

    Combines multiple detection methods for robust monitoring.
    """

    def __init__(
        self,
        dataset_name: str,
        baseline_window: int = 1000,
        detection_window: int = 100,
        tests: List[DriftTest] = None,
        alert_threshold: int = 2
    ):
        self.dataset_name = dataset_name
        self.baseline_window = baseline_window
        self.detection_window = detection_window
        self.tests = tests or [DriftTest.PSI, DriftTest.KS, DriftTest.EMBEDDING]
        self.alert_threshold = alert_threshold

        # Storage
        self.baseline_data: Optional[np.ndarray] = None
        self.baseline_embeddings: Optional[np.ndarray] = None
        self.drift_history: List[DriftAlert] = []

        # Initialize detectors
        self.detectors = self._initialize_detectors()

    def _initialize_detectors(self) -> Dict[DriftTest, DriftDetector]:
        """Initialize all drift detectors."""
        from .statistical_tests import PSIDetector, KSDetector
        from .domain_classifier import DomainClassifierDetector
        from .embedding_drift import EmbeddingDriftDetector

        detectors = {}

        if DriftTest.PSI in self.tests:
            detectors[DriftTest.PSI] = PSIDetector()
        if DriftTest.KS in self.tests:
            detectors[DriftTest.KS] = KSDetector()
        if DriftTest.DOMAIN_CLASSIFIER in self.tests:
            detectors[DriftTest.DOMAIN_CLASSIFIER] = DomainClassifierDetector()
        if DriftTest.EMBEDDING in self.tests:
            detectors[DriftTest.EMBEDDING] = EmbeddingDriftDetector()

        return detectors

    def set_baseline(
        self,
        data: pd.DataFrame,
        text_column: str
    ):
        """Establish baseline distributions."""
        # Store raw data for statistical tests
        self.baseline_data = data[text_column].values

        # Generate embeddings
        self.baseline_embeddings = self._generate_embeddings(data[text_column])

        logger.info(f"Baseline established: {len(data)} samples")

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        text_column: str
    ) -> DriftAlert:
        """
        Detect drift in current data vs. baseline.

        Returns:
            DriftAlert with comprehensive drift assessment
        """
        if self.baseline_data is None:
            raise ValueError("Must call set_baseline() first")

        # Generate current embeddings
        current_embeddings = self._generate_embeddings(current_data[text_column])

        # Run all tests
        results = {}
        drift_count = 0

        for test_name, detector in self.detectors.items():
            if test_name == DriftTest.EMBEDDING:
                result = detector.detect(
                    self.baseline_embeddings,
                    current_embeddings
                )
            else:
                result = detector.detect(
                    self.baseline_embeddings,
                    current_embeddings
                )

            results[test_name.value] = result

            if result.get("drift_detected", False):
                drift_count += 1

        # Create alert
        drift_alert = DriftAlert(
            timestamp=datetime.now(),
            severity=self._calculate_severity(drift_count, len(self.tests)),
            tests_detecting_drift=drift_count,
            total_tests=len(self.tests),
            recommendation=self._get_recommendation(drift_count, len(self.tests)),
            detailed_results=results
        )

        # Store history
        self.drift_history.append(drift_alert)

        # Send alert if needed
        if drift_count >= self.alert_threshold:
            self._send_alert(drift_alert)

        return drift_alert

    def _calculate_severity(
        self,
        drift_count: int,
        total_tests: int
    ) -> str:
        """Calculate alert severity."""
        ratio = drift_count / total_tests

        if ratio >= 0.75:
            return "critical"
        elif ratio >= 0.5:
            return "high"
        elif ratio >= 0.25:
            return "moderate"
        else:
            return "low"

    def _send_alert(self, alert: DriftAlert):
        """Send drift detection alert."""
        logger.warning(
            f"⚠️  DRIFT ALERT: {alert.severity.upper()} severity. "
            f"{alert.tests_detecting_drift}/{alert.total_tests} tests detected drift. "
            f"Recommendation: {alert.recommendation}"
        )

        # TODO: Integrate with email/Slack/PagerDuty
```

### 4.3 Integration with Quality Monitor

```python
# quality/drift_integration.py
class QualityMonitorWithDrift(QualityMonitor):
    """Quality monitor with integrated drift detection."""

    def __init__(
        self,
        dataset_name: str,
        metric_distance: str = "nominal",
        enable_drift_detection: bool = True
    ):
        super().__init__(dataset_name, metric_distance)

        if enable_drift_detection:
            self.drift_monitor = ComprehensiveDriftMonitor(dataset_name)

    def comprehensive_monitoring(
        self,
        df: pd.DataFrame,
        text_column: str,
        annotator_columns: List[str],
        is_baseline: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive monitoring with drift detection.

        Combines IAA monitoring with drift detection.
        """
        # Calculate IAA
        iaa_result = self.calculate_krippendorff_alpha(df, annotator_columns)

        # Drift detection
        if is_baseline:
            self.drift_monitor.set_baseline(df, text_column)
            drift_result = {"status": "baseline_established"}
        else:
            drift_alert = self.drift_monitor.detect_drift(df, text_column)
            drift_result = drift_alert.dict()

        # Combined assessment
        return {
            "timestamp": datetime.now().isoformat(),
            "quality": {
                "iaa": iaa_result,
                "status": self._assess_quality_status(iaa_result)
            },
            "drift": drift_result,
            "overall_health": self._calculate_overall_health(iaa_result, drift_result)
        }
```

---

## 5. STAPLE Integration

### 5.1 STAPLE Algorithm Module

```python
# ensemble/staple.py
class STAPLEAlgorithm:
    """
    STAPLE (Simultaneous Truth and Performance Level Estimation) algorithm.

    Expectation-Maximization algorithm for multi-annotator consensus.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.annotator_quality: Dict[int, Dict[str, np.ndarray]] = {}

    def estimate_ground_truth(
        self,
        annotations: np.ndarray,
        max_iterations: int = 50,
        convergence_threshold: float = 1e-5
    ) -> Tuple[np.ndarray, Dict[int, Dict[str, np.ndarray]]]:
        """
        Estimate ground truth and annotator quality via EM algorithm.

        Args:
            annotations: (n_items, n_annotators) array, -1 for missing
            max_iterations: Maximum EM iterations
            convergence_threshold: Convergence threshold

        Returns:
            (ground_truth, annotator_quality)
        """
        n_items, n_annotators = annotations.shape

        # Initialize
        ground_truth = self._initialize_ground_truth(annotations)
        self._initialize_annotator_quality(n_annotators)

        # EM algorithm
        for iteration in range(max_iterations):
            old_ground_truth = ground_truth.copy()

            # E-step: Update ground truth
            ground_truth = self._e_step(annotations)

            # M-step: Update annotator quality
            self._m_step(annotations, ground_truth)

            # Check convergence
            changes = np.sum(ground_truth != old_ground_truth)
            if changes < convergence_threshold * n_items:
                break

        return ground_truth, self.annotator_quality

    def _e_step(self, annotations: np.ndarray) -> np.ndarray:
        """E-step: Estimate ground truth given quality parameters."""
        # Implementation in research report
        pass

    def _m_step(
        self,
        annotations: np.ndarray,
        ground_truth: np.ndarray
    ):
        """M-step: Update quality parameters given ground truth."""
        # Implementation in research report
        pass
```

### 5.2 Integration with EnsembleService

```python
# ensemble/ensemble_service.py (extension)
class EnsembleService:
    # ... existing code ...

    def aggregate_with_staple(
        self,
        df: pd.DataFrame,
        label_columns: List[str],
        n_classes: int = 2
    ) -> pd.DataFrame:
        """
        Aggregate multiple annotations using STAPLE.

        Args:
            df: DataFrame with annotations
            label_columns: Columns containing annotations
            n_classes: Number of classes

        Returns:
            DataFrame with consensus labels and performance metrics
        """
        from .staple import STAPLEAlgorithm

        # Extract annotations matrix
        annotations = self._prepare_annotations(df, label_columns)

        # Run STAPLE
        staple = STAPLEAlgorithm(num_classes=n_classes)
        consensus, performance = staple.estimate_ground_truth(annotations)

        # Add to dataframe
        result_df = df.copy()
        result_df["staple_consensus"] = consensus
        result_df["staple_confidence"] = self._calculate_staple_confidence(
            consensus, annotations
        )

        # Add performance metrics
        result_df["annotator_performance"] = [performance] * len(df)

        logger.info(
            f"STAPLE aggregation complete: {len(df)} items, "
            f"{len(label_columns)} annotators"
        )

        return result_df
```

---

## 6. DPO/RLHF Pipeline

### 6.1 Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    DPO Training Pipeline                     │
│                                                              │
│  1. Preference Data Generation                               │
│     ├─> High vs. Low Confidence                             │
│     ├─> Expert vs. Model Disagreements                       │
│     └─> Multi-Annotator Consensus vs. Outliers              │
│                                                              │
│  2. Dataset Preparation                                      │
│     ├─> Format as (prompt, chosen, rejected) triples        │
│     ├─> Train/val split                                      │
│     └─> Tokenization                                         │
│                                                              │
│  3. DPO Training                                             │
│     ├─> Load base model                                      │
│     ├─> Configure LoRA                                       │
│     ├─> Train with TRL DPOTrainer                           │
│     └─> Save aligned model                                   │
│                                                              │
│  4. Evaluation                                               │
│     ├─> Preference accuracy                                  │
│     ├─> Annotation quality vs. base model                    │
│     └─> Production deployment decision                       │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Class Design

```python
# alignment/dpo_trainer.py
class DPOAlignmentPipeline:
    """
    Complete pipeline for DPO model alignment.

    Handles preference data generation, training, and evaluation.
    """

    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        dpo_config: DPOConfig = None
    ):
        self.base_model_name = base_model_name
        self.output_dir = Path(output_dir)
        self.dpo_config = dpo_config or DPOConfig()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_preference_data(
        self,
        annotations_df: pd.DataFrame,
        strategy: Literal["confidence", "expert", "consensus"] = "confidence"
    ) -> Dataset:
        """
        Generate preference dataset from annotations.

        Args:
            annotations_df: DataFrame with annotations
            strategy: Preference generation strategy

        Returns:
            HuggingFace Dataset with preferences
        """
        from .preference_data import (
            generate_confidence_based_preferences,
            generate_expert_disagreement_preferences,
            generate_consensus_based_preferences
        )

        if strategy == "confidence":
            preferences_df = generate_confidence_based_preferences(annotations_df)
        elif strategy == "expert":
            preferences_df = generate_expert_disagreement_preferences(annotations_df)
        elif strategy == "consensus":
            preferences_df = generate_consensus_based_preferences(annotations_df)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(preferences_df)

        logger.info(f"Generated {len(dataset)} preference pairs using {strategy} strategy")

        return dataset

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None
    ) -> str:
        """
        Train model with DPO.

        Args:
            train_dataset: Training preferences
            val_dataset: Validation preferences (optional)

        Returns:
            Path to saved model
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOTrainer

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        # Configure DPO trainer
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # Will create reference copy
            args=self.dpo_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer
        )

        # Train
        logger.info("Starting DPO training...")
        trainer.train()

        # Save
        model_path = self.output_dir / "dpo_aligned_model"
        trainer.save_model(str(model_path))

        logger.info(f"DPO training complete. Model saved to {model_path}")

        return str(model_path)

    def evaluate(
        self,
        aligned_model_path: str,
        test_dataset: Dataset,
        test_annotations: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate aligned model vs. base model.

        Args:
            aligned_model_path: Path to DPO-aligned model
            test_dataset: Test preference dataset
            test_annotations: Test annotations with ground truth

        Returns:
            Evaluation metrics
        """
        from .evaluator import DPOEvaluator

        evaluator = DPOEvaluator(aligned_model_path, self.base_model_name)

        # Preference accuracy
        pref_metrics = evaluator.evaluate_preference_accuracy(test_dataset)

        # Annotation quality
        quality_metrics = evaluator.compare_annotation_quality(
            test_annotations,
            text_column="text",
            gold_label_column="gold_label"
        )

        return {
            "preference_accuracy": pref_metrics,
            "annotation_quality": quality_metrics,
            "recommendation": self._get_deployment_recommendation(
                pref_metrics,
                quality_metrics
            )
        }

    def _get_deployment_recommendation(
        self,
        pref_metrics: Dict,
        quality_metrics: Dict
    ) -> str:
        """Get deployment recommendation based on metrics."""
        pref_acc = pref_metrics.get("preference_accuracy", 0)
        quality_improvement = quality_metrics.get("relative_improvement", 0)

        if pref_acc > 0.7 and quality_improvement > 0.1:
            return "DEPLOY: Significant improvement observed"
        elif pref_acc > 0.6 and quality_improvement > 0.05:
            return "CONSIDER: Moderate improvement, test further"
        else:
            return "DO NOT DEPLOY: Insufficient improvement"
```

---

## 7. Constitutional AI Framework

### 7.1 Principle-Based Architecture

```
┌──────────────────────────────────────────────────────────────┐
│           Constitutional Annotation Framework                 │
│                                                              │
│  1. Principle Definition                                     │
│     └─> Constitutional principles for annotation            │
│                                                              │
│  2. Generate Initial Annotation                              │
│     └─> LLM generates label + reasoning                      │
│                                                              │
│  3. Critique Against Principles                              │
│     ├─> For each principle:                                  │
│     │   ├─> Check adherence                                  │
│     │   └─> Generate critique if violated                    │
│     └─> Aggregate critiques                                  │
│                                                              │
│  4. Revise Based on Critique                                 │
│     └─> LLM revises annotation to address critiques         │
│                                                              │
│  5. Iterate (optional)                                       │
│     └─> Repeat critique-revise until convergence            │
│                                                              │
│  6. Final Validation                                         │
│     └─> Calculate adherence score                            │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 Class Design

```python
# constitutional/principles.py
class ConstitutionalPrinciple(BaseModel):
    """A single constitutional principle for annotation."""
    name: str
    principle: str
    critique_prompt: str
    revision_prompt: str
    weight: float = 1.0  # Importance weight

class CritiqueResult(BaseModel):
    """Result of critiquing an annotation against a principle."""
    principle_name: str
    violated: bool
    critique: str
    severity: float  # 0-1

class ConstitutionalAnnotator:
    """
    Annotator that uses Constitutional AI for principled consistency.
    """

    def __init__(
        self,
        model_name: str,
        principles: Dict[str, ConstitutionalPrinciple],
        max_iterations: int = 2
    ):
        self.model_name = model_name
        self.principles = principles
        self.max_iterations = max_iterations
        self.llm = self._initialize_llm()

    def annotate_constitutional(
        self,
        text: str,
        task_description: str,
        guidelines: str
    ) -> Dict[str, Any]:
        """
        Perform constitutional annotation.

        Args:
            text: Text to annotate
            task_description: Description of annotation task
            guidelines: Annotation guidelines

        Returns:
            Final annotation with critique history
        """
        # 1. Generate initial annotation
        current_annotation = self._generate_initial_annotation(
            text, task_description, guidelines
        )

        critique_history = []

        # 2-4. Iterative critique and revision
        for iteration in range(self.max_iterations):
            # Critique against all principles
            critiques = self._critique_annotation(
                text, current_annotation, guidelines
            )

            # Check for violations
            violations = [c for c in critiques if c.violated]

            if not violations:
                # No violations, converged
                break

            # Revise based on critiques
            current_annotation = self._revise_annotation(
                text, current_annotation, violations, guidelines
            )

            critique_history.append({
                "iteration": iteration + 1,
                "critiques": [c.dict() for c in critiques],
                "violations_found": len(violations)
            })

        # 5. Final validation
        final_critiques = self._critique_annotation(
            text, current_annotation, guidelines
        )
        adherence_score = self._calculate_adherence_score(final_critiques)

        return {
            "text": text,
            "label": current_annotation["label"],
            "confidence": current_annotation.get("confidence", 0.8),
            "reasoning": current_annotation.get("reasoning", ""),
            "constitutional": {
                "adherence_score": adherence_score,
                "iterations": len(critique_history),
                "critique_history": critique_history,
                "final_critiques": [c.dict() for c in final_critiques]
            }
        }
```

---

## 8. Data Flow and Integration

### 8.1 End-to-End Flow

```
User Request
     │
     v
┌──────────────────────────────────────┐
│  AutoLabelerPhase3.label_advanced()  │
│  • Determines which Phase 3 features │
│  • Coordinates execution              │
└───────┬──────────────────────────────┘
        │
        v
┌───────────────────────────────────────┐
│  Feature Selection                    │
│  • Multi-agent?                       │
│  • Constitutional AI?                 │
│  • Drift monitoring?                  │
│  • STAPLE aggregation?                │
└───────┬───────────────────────────────┘
        │
        └─────┬──────────────┬───────────────┬──────────────┐
              │              │               │              │
              v              v               v              v
     ┌────────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐
     │Multi-Agent │  │Constitu-  │  │  Drift   │  │  STAPLE   │
     │Coordination│  │tional AI  │  │Detection │  │Aggregation│
     └────────────┘  └───────────┘  └──────────┘  └───────────┘
              │              │               │              │
              └──────────────┴───────────────┴──────────────┘
                                     │
                                     v
                          ┌────────────────────┐
                          │  Quality Monitor   │
                          │  • Log metrics     │
                          │  • Update dashboard│
                          └────────────────────┘
                                     │
                                     v
                          ┌────────────────────┐
                          │  Return Results    │
                          │  to User           │
                          └────────────────────┘
```

### 8.2 Configuration Flow

```python
# Phase 3 Configuration System
class Phase3Config(BaseModel):
    """Unified Phase 3 configuration."""

    # Multi-Agent
    enable_multi_agent: bool = False
    multi_agent_config: Optional[MultiAgentConfig] = None

    # Drift Detection
    enable_drift_detection: bool = False
    drift_config: Optional[DriftDetectionConfig] = None

    # STAPLE
    enable_staple: bool = False
    staple_config: Optional[STAPLEConfig] = None

    # Constitutional AI
    enable_constitutional: bool = False
    constitutional_config: Optional[ConstitutionalConfig] = None

    # DPO/RLHF
    dpo_model_path: Optional[str] = None  # Use DPO-aligned model if provided

class AutoLabelerPhase3(AutoLabeler):
    """AutoLabeler with Phase 3 features."""

    def __init__(
        self,
        dataset_name: str,
        settings: Settings,
        phase3_config: Phase3Config = None
    ):
        super().__init__(dataset_name, settings)
        self.phase3_config = phase3_config or Phase3Config()

        # Initialize Phase 3 components (lazy loading)
        self._multi_agent_service = None
        self._drift_monitor = None
        self._constitutional_annotator = None

    def label_advanced(
        self,
        df: pd.DataFrame,
        text_column: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Label with Phase 3 advanced features.

        Automatically routes through enabled features.
        """
        results_df = df.copy()

        # 1. Drift detection (if enabled)
        if self.phase3_config.enable_drift_detection:
            drift_alert = self._check_drift(df, text_column)
            results_df["drift_detected"] = drift_alert.severity != "low"

        # 2. Constitutional or Multi-Agent annotation
        if self.phase3_config.enable_multi_agent:
            results_df = self._label_with_multi_agent(results_df, text_column)
        elif self.phase3_config.enable_constitutional:
            results_df = self._label_with_constitutional(results_df, text_column)
        else:
            # Standard labeling
            results_df = self.label(results_df, text_column)

        # 3. STAPLE aggregation (if multi-annotator)
        if self.phase3_config.enable_staple and self._has_multiple_annotators(results_df):
            results_df = self._aggregate_with_staple(results_df)

        return results_df
```

---

## 9. API Design

### 9.1 Main Interface

```python
# autolabeler_v3.py
class AutoLabelerPhase3(AutoLabeler):
    """
    AutoLabeler with Phase 3 advanced features.

    Provides backward-compatible API with additional Phase 3 methods.
    """

    # ========== Phase 3 Feature APIs ==========

    def label_with_multiagent(
        self,
        df: pd.DataFrame,
        text_column: str,
        agent_config: MultiAgentConfig
    ) -> pd.DataFrame:
        """Label using multi-agent architecture."""
        pass

    def label_with_constitutional(
        self,
        df: pd.DataFrame,
        text_column: str,
        principles: Dict[str, Dict[str, str]] = None
    ) -> pd.DataFrame:
        """Label with Constitutional AI."""
        pass

    def aggregate_with_staple(
        self,
        df: pd.DataFrame,
        label_columns: List[str],
        n_classes: int = 2
    ) -> pd.DataFrame:
        """Aggregate annotations using STAPLE."""
        pass

    def monitor_drift(
        self,
        df: pd.DataFrame,
        text_column: str,
        is_baseline: bool = False
    ) -> DriftAlert:
        """Monitor for distribution drift."""
        pass

    def align_with_dpo(
        self,
        preference_df: pd.DataFrame,
        output_dir: str,
        strategy: str = "confidence"
    ) -> str:
        """Align model using Direct Preference Optimization."""
        pass

    # ========== Convenience Methods ==========

    def label_advanced(
        self,
        df: pd.DataFrame,
        text_column: str,
        enable_all: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Label with automatic Phase 3 feature selection.

        Args:
            df: DataFrame to label
            text_column: Text column name
            enable_all: Enable all Phase 3 features
            **kwargs: Feature-specific configurations

        Returns:
            DataFrame with annotations and Phase 3 metadata
        """
        pass
```

### 9.2 CLI Interface

```bash
# Phase 3 CLI Commands

# Multi-agent annotation
autolabeler multiagent \
  --input data.csv \
  --text-column review \
  --agents entity,sentiment,validator \
  --output results.csv

# Drift detection
autolabeler drift \
  --baseline baseline.csv \
  --current current.csv \
  --text-column text \
  --tests psi,ks,embedding

# STAPLE aggregation
autolabeler staple \
  --input annotations.csv \
  --annotator-columns ann1,ann2,ann3 \
  --classes 3 \
  --output consensus.csv

# Constitutional annotation
autolabeler constitutional \
  --input data.csv \
  --text-column text \
  --principles principles.json \
  --output results.csv

# DPO alignment
autolabeler align-dpo \
  --preferences preferences.csv \
  --base-model llama-3.1-8b \
  --output-dir ./dpo_model \
  --strategy confidence
```

---

## 10. Testing Strategy

### 10.1 Test Coverage

| Component | Unit Tests | Integration Tests | Performance Tests |
|-----------|-----------|-------------------|-------------------|
| Multi-Agent | 25 tests | 10 tests | 5 tests |
| Drift Detection | 30 tests | 8 tests | 5 tests |
| STAPLE | 20 tests | 5 tests | 3 tests |
| Constitutional AI | 15 tests | 7 tests | 3 tests |
| DPO/RLHF | 20 tests | 10 tests | 5 tests |
| **Total** | **110 tests** | **40 tests** | **21 tests** |

### 10.2 Test Examples

```python
# Test Multi-Agent Coordination
def test_multiagent_parallel_execution():
    """Test agents execute in parallel."""
    service = MultiAgentService(...)

    start_time = time.time()
    result = await service.coordinate(text, config)
    elapsed = time.time() - start_time

    # Should be faster than sequential execution
    assert elapsed < sequential_time * 0.6

# Test Drift Detection
def test_drift_detection_sensitivity():
    """Test drift detection catches known shifts."""
    monitor = ComprehensiveDriftMonitor(...)

    # Baseline: Normal distribution
    baseline = np.random.normal(0, 1, (1000, 10))
    monitor.set_baseline_embeddings(baseline)

    # Current: Shifted distribution
    current = np.random.normal(2, 1, (100, 10))
    alert = monitor.detect_drift_embeddings(current)

    assert alert.tests_detecting_drift >= 2
    assert alert.severity in ["high", "critical"]

# Test STAPLE Convergence
def test_staple_convergence():
    """Test STAPLE converges within iterations."""
    annotations = generate_test_annotations(...)
    staple = STAPLEAlgorithm(num_classes=3)

    consensus, quality = staple.estimate_ground_truth(
        annotations,
        max_iterations=50
    )

    # Should converge
    assert len(consensus) == annotations.shape[0]
    assert all(0 <= c < 3 for c in consensus)
```

---

## 11. Deployment Architecture

### 11.1 Production Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                  Production Environment                      │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │          Load Balancer (NGINX)                        │  │
│  └───────┬───────────────────────────────────────────────┘  │
│          │                                                   │
│    ┌─────┴─────┬─────────────┬─────────────┬──────────┐    │
│    │           │             │             │          │    │
│    v           v             v             v          v    │
│  ┌──────┐  ┌──────┐     ┌──────┐     ┌──────┐  ┌──────┐  │
│  │ API  │  │ API  │ ... │ API  │     │Worker│  │Worker│  │
│  │ Node │  │ Node │     │ Node │     │ Node │  │ Node │  │
│  │  1   │  │  2   │     │  N   │     │  1   │  │  2   │  │
│  └──┬───┘  └──┬───┘     └──┬───┘     └───┬──┘  └───┬──┘  │
│     │         │            │              │         │      │
│     └─────────┴────────────┴──────────────┴─────────┘      │
│                            │                                │
│                            v                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │            Shared Services Layer                      │  │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │  │
│  │  │PostgreSQL│  │  Redis   │  │  Vector DB         │ │  │
│  │  │(metadata)│  │  (cache) │  │  (embeddings)      │ │  │
│  │  └──────────┘  └──────────┘  └────────────────────┘ │  │
│  │                                                        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │  │
│  │  │ Message  │  │Monitoring│  │  Model Registry    │ │  │
│  │  │  Queue   │  │(Grafana) │  │  (MLflow)          │ │  │
│  │  └──────────┘  └──────────┘  └────────────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │          GPU Cluster (for DPO training)               │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │  │
│  │  │  GPU     │  │  GPU     │  │  GPU     │           │  │
│  │  │  Node 1  │  │  Node 2  │  │  Node 3  │           │  │
│  │  └──────────┘  └──────────┘  └──────────┘           │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 11.2 Scalability Considerations

**API Nodes:**
- Stateless design for horizontal scaling
- Auto-scaling based on request volume
- Health checks and graceful shutdown

**Worker Nodes:**
- Background task processing (DPO training, batch labeling)
- Queue-based task distribution
- GPU scheduling for training tasks

**Data Layer:**
- PostgreSQL for metadata and audit logs
- Redis for caching and session state
- Vector DB (Pinecone/Weaviate) for embeddings

**Monitoring:**
- Prometheus + Grafana for metrics
- LangSmith for agent tracing
- Evidently for drift monitoring dashboards

---

## 12. Performance Considerations

### 12.1 Latency Budget

| Operation | Target Latency (p95) | Notes |
|-----------|---------------------|-------|
| Standard labeling | <2s | Phase 1/2 baseline |
| Multi-agent (3 agents) | <4s | 2× overhead acceptable |
| Constitutional AI (2 iterations) | <6s | Critique+revise cycles |
| Drift detection | <500ms | Real-time monitoring |
| STAPLE aggregation | <1s | For 1000 items |

### 12.2 Optimization Strategies

**1. Parallel Agent Execution**
```python
# Execute agents concurrently
tasks = [agent.process(text) for agent in agents]
results = await asyncio.gather(*tasks)
```

**2. Caching**
```python
# Cache embeddings for drift detection
@functools.lru_cache(maxsize=10000)
def get_embedding(text: str) -> np.ndarray:
    return embedding_model.encode(text)
```

**3. Batch Processing**
```python
# Process multiple texts in batches
results = labeler.label_batch(
    texts,
    batch_size=32,
    max_concurrency=4
)
```

**4. Model Quantization (DPO)**
```python
# Use 8-bit quantization for faster inference
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    device_map="auto"
)
```

---

## 13. Migration Path

### 13.1 Gradual Adoption

**Week 8-9: Foundation Features**
1. Deploy STAPLE algorithm (no API changes required)
2. Enable drift detection monitoring (passive observation)
3. Test performance impact

**Week 10-11: Advanced Features**
4. Roll out multi-agent architecture (opt-in feature flag)
5. Enable Constitutional AI (specific use cases)
6. Monitor latency and accuracy improvements

**Week 12: Model Alignment (Optional)**
7. Generate preference data from production
8. Train DPO model offline
9. A/B test aligned model vs. base model
10. Gradual rollout if successful

### 13.2 Backward Compatibility

**All existing APIs continue to work:**
```python
# Phase 1 API - still works
labeler = AutoLabeler("dataset", settings)
results = labeler.label(df, "text")

# Phase 3 API - new features
labeler3 = AutoLabelerPhase3("dataset", settings, phase3_config)
results = labeler3.label_advanced(df, "text")
```

---

## 14. Summary and Recommendations

### 14.1 Architecture Strengths

1. **Modularity:** Each Phase 3 feature is independently usable
2. **Extensibility:** Easy to add new agents, detectors, principles
3. **Performance:** Designed for <2× latency overhead
4. **Backward Compatibility:** All Phase 1/2 APIs preserved
5. **Production-Ready:** Scalable architecture with monitoring

### 14.2 Implementation Priority

**Must-Have (Weeks 8-9):**
- STAPLE algorithm (high value, low risk)
- Drift detection (critical for production monitoring)

**Should-Have (Weeks 10-11):**
- Multi-agent architecture (significant quality improvement)
- Constitutional AI (improves consistency)

**Nice-to-Have (Week 12, Optional):**
- DPO/RLHF (requires GPU, highest complexity)

### 14.3 Success Metrics

- **Multi-Agent:** +10-15% accuracy improvement
- **Drift Detection:** 95% sensitivity, <10% false positives
- **STAPLE:** +5-10% vs. majority voting
- **Constitutional AI:** >95% principle adherence
- **DPO:** +15-25% on aligned tasks (if implemented)

---

**Architecture Status:** ✅ Complete and Production-Ready
**Date:** October 8, 2025
**Next Action:** Begin implementation with STAPLE and Drift Detection (Week 8)
