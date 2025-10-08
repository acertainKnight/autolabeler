# AutoLabeler Implementation Roadmap & Testing Strategy
## TESTER/INTEGRATION AGENT - Strategic Implementation Plan

**Document Version:** 1.0
**Date:** 2025-10-07
**Project:** AutoLabeler v2 Enhancement Initiative
**Mission:** Detailed implementation roadmap, testing strategy, and integration plan for 2024-2025 state-of-the-art annotation capabilities

---

## Executive Summary

This document provides a comprehensive implementation roadmap for enhancing AutoLabeler from its current state to a production-grade, state-of-the-art annotation system incorporating the latest research and best practices from 2023-2025. The roadmap is organized into three implementation phases with specific timelines, dependencies, and risk mitigation strategies.

**Current State Analysis:**
- ✅ **Strengths**: Solid modular architecture, RAG-based labeling, ensemble methods, batch processing, CLI interface
- ⚠️ **Gaps**: No DSPy integration, limited quality monitoring, no active learning, basic confidence calibration, missing data versioning
- 🎯 **Target**: Production-grade system matching 2024-2025 state-of-the-art with 40-70% cost reduction and 10-100× speed improvements

---

## Table of Contents

1. [Feature Prioritization Matrix](#1-feature-prioritization-matrix)
2. [Technical Architecture](#2-technical-architecture)
3. [Phase 1: Quick Wins (1-2 Weeks)](#3-phase-1-quick-wins-1-2-weeks)
4. [Phase 2: Core Features (3-6 Weeks)](#4-phase-2-core-features-3-6-weeks)
5. [Phase 3: Advanced Features (7-12 Weeks)](#5-phase-3-advanced-features-7-12-weeks)
6. [Testing Strategy](#6-testing-strategy)
7. [Migration & Compatibility](#7-migration--compatibility)
8. [Risk Assessment](#8-risk-assessment)
9. [Success Metrics](#9-success-metrics)

---

## 1. Feature Prioritization Matrix

### Priority Classification Criteria

| Priority | Complexity | Impact | Dependencies | Timeline |
|----------|-----------|--------|--------------|----------|
| **P0 - Critical** | Low-Medium | High | Minimal | 1-2 weeks |
| **P1 - High** | Medium | High | P0 complete | 3-6 weeks |
| **P2 - Medium** | Medium-High | Medium | P0-P1 complete | 7-12 weeks |
| **P3 - Low** | High | Low-Medium | Optional | Future |

### Feature Breakdown by Priority

#### P0: Quick Wins (Immediate Impact, Low Risk)

| Feature | Impact | Complexity | Dependencies | Timeline |
|---------|--------|-----------|--------------|----------|
| **Structured Output Validation** | High | Low | None | 2-3 days |
| **Confidence Calibration** | High | Low | None | 3-4 days |
| **Quality Metrics Dashboard** | High | Medium | None | 5-7 days |
| **Krippendorff's Alpha** | High | Low | None | 2-3 days |
| **Cost Tracking** | Medium | Low | None | 2-3 days |

**Total Phase Duration:** 1-2 weeks

#### P1: Core Features (Foundation Building)

| Feature | Impact | Complexity | Dependencies | Timeline |
|---------|--------|-----------|--------------|----------|
| **DSPy Integration** | Very High | Medium | P0 complete | 7-10 days |
| **Advanced RAG (GraphRAG/RAPTOR)** | High | Medium | None | 7-10 days |
| **Active Learning Loop** | High | Medium | P0 metrics | 10-14 days |
| **Weak Supervision (Snorkel)** | High | Medium-High | None | 10-14 days |
| **Data Versioning (DVC)** | High | Low | None | 3-5 days |

**Total Phase Duration:** 3-6 weeks

#### P2: Advanced Features (Production Enhancement)

| Feature | Impact | Complexity | Dependencies | Timeline |
|---------|--------|-----------|--------------|----------|
| **Multi-Agent Architecture** | Medium | High | P1 DSPy | 14-21 days |
| **DPO/RLHF Integration** | Medium | High | Large dataset | 14-21 days |
| **Drift Detection** | Medium | Medium | P0 metrics | 7-10 days |
| **Advanced Ensemble (STAPLE)** | Medium | Medium | Current ensemble | 7-10 days |
| **Constitutional AI** | Low-Medium | High | P1 complete | 14-21 days |

**Total Phase Duration:** 7-12 weeks

---

## 2. Technical Architecture

### 2.1 Current Architecture Analysis

```
Current AutoLabeler Architecture (as-is):
┌─────────────────────────────────────────────────────────────┐
│                    AutoLabeler Main Interface                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Core Services                         │
├──────────────┬──────────────┬──────────────┬────────────────┤
│ Labeling     │ Ensemble     │ Synthetic    │ Rule           │
│ Service      │ Service      │ Service      │ Service        │
└──────────────┴──────────────┴──────────────┴────────────────┘
        │              │               │              │
        ▼              ▼               ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│              Knowledge Management Layer                      │
├──────────────────────┬──────────────────────────────────────┤
│ KnowledgeStore       │ PromptManager                        │
│ (FAISS + Metadata)   │ (Template + Tracking)                │
└──────────────────────┴──────────────────────────────────────┘
        │                          │
        ▼                          ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM Provider Layer                              │
├──────────────────────┬──────────────────────────────────────┤
│ OpenRouter          │ Corporate Proxy                       │
│ OpenAI              │ (via Factory Pattern)                 │
└──────────────────────┴──────────────────────────────────────┘
```

**Strengths:**
- ✅ Clean service-oriented architecture
- ✅ Modular design with single responsibility
- ✅ Configuration-driven components
- ✅ Batch processing with resume capability
- ✅ Knowledge base with RAG retrieval
- ✅ Multi-model ensemble support

**Gaps Identified:**
- ❌ No prompt optimization framework (DSPy)
- ❌ Limited quality monitoring (no real-time metrics)
- ❌ Basic RAG (no advanced variants like GraphRAG)
- ❌ No active learning implementation
- ❌ Missing weak supervision integration
- ❌ No data versioning system
- ❌ Limited confidence calibration
- ❌ No drift detection
- ❌ Missing annotation provenance tracking

### 2.2 Proposed Architecture (to-be)

```
Enhanced AutoLabeler Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                    AutoLabeler Main Interface                        │
│              (Configuration-driven orchestration)                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Enhanced Core Services Layer                       │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│ Labeling     │ Ensemble     │ Synthetic    │ Rule Gen    │ Active   │
│ Service      │ Service      │ Service      │ Service     │ Learning │
│ (+ DSPy)     │ (+ STAPLE)   │ (Enhanced)   │ (Enhanced)  │ Service  │
└──────────────┴──────────────┴──────────────┴─────────────┴──────────┘
        │              │               │              │           │
        ▼              ▼               ▼              ▼           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Enhanced Knowledge Management Layer                     │
├──────────────────────┬──────────────────────┬──────────────────────┤
│ KnowledgeStore       │ PromptManager        │ WeakSupervision      │
│ - FAISS/ChromaDB     │ - DSPy Integration   │ - Snorkel Integration│
│ - GraphRAG           │ - Version Control    │ - Labeling Functions │
│ - RAPTOR             │ - A/B Testing        │ - FlyingSquid        │
└──────────────────────┴──────────────────────┴──────────────────────┘
        │                          │                      │
        ▼                          ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Quality & Monitoring Layer (NEW)                    │
├──────────────────────┬──────────────────────┬──────────────────────┤
│ QualityMonitor       │ DriftDetector        │ CostTracker          │
│ - Krippendorff α     │ - PSI/KS Tests       │ - Per-annotation     │
│ - Confidence Cal.    │ - Embedding Drift    │ - Budget Alerts      │
│ - IAA Tracking       │ - Performance Decay  │ - ROI Metrics        │
└──────────────────────┴──────────────────────┴──────────────────────┘
        │                          │                      │
        ▼                          ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Data Management & Versioning Layer (NEW)                │
├──────────────────────┬──────────────────────┬──────────────────────┤
│ DataVersioning       │ ProvenanceTracker    │ ExperimentTracker    │
│ - DVC Integration    │ - Full Lineage       │ - MLflow/W&B         │
│ - Git-like Ops       │ - Audit Logs         │ - Metrics Compare    │
└──────────────────────┴──────────────────────┴──────────────────────┘
        │                          │                      │
        ▼                          ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Enhanced LLM Provider Layer                             │
├──────────────────────┬──────────────────────┬──────────────────────┤
│ OpenRouter          │ Corporate Proxy       │ Local Models         │
│ OpenAI              │ Anthropic             │ vLLM/Ollama          │
│ (Structured Output) │ (Claude + Structured) │ (On-premise)         │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

### 2.3 New Components Specification

#### 2.3.1 Quality Monitor Service

```python
# Location: src/autolabeler/core/quality/quality_monitor.py

class QualityMonitorService(ConfigurableComponent):
    """
    Comprehensive quality monitoring with real-time metrics.

    Features:
    - Krippendorff's alpha calculation on overlapping samples
    - Confidence calibration (temperature scaling, Platt scaling)
    - Inter-annotator agreement tracking
    - Automated anomaly detection
    - Quality score computation (CQAA metric)
    """

    def calculate_krippendorff_alpha(
        self,
        df: pd.DataFrame,
        annotator_cols: list[str]
    ) -> float:
        """Calculate Krippendorff's alpha for multi-annotator agreement."""
        pass

    def calibrate_confidence(
        self,
        predictions: pd.DataFrame,
        method: str = "temperature_scaling"
    ) -> pd.DataFrame:
        """Calibrate model confidence scores."""
        pass

    def detect_quality_issues(self) -> dict[str, Any]:
        """Detect systematic quality issues and return recommendations."""
        pass
```

#### 2.3.2 Active Learning Service

```python
# Location: src/autolabeler/core/active_learning/active_learning_service.py

class ActiveLearningService(ConfigurableComponent):
    """
    Active learning implementation with multiple sampling strategies.

    Features:
    - Uncertainty sampling (margin, entropy, least confidence)
    - Diversity sampling (TCM heuristic)
    - Hybrid strategies for cold start
    - Stopping criteria (performance plateau detection)
    - Integration with labeling pipeline
    """

    def select_samples(
        self,
        unlabeled_df: pd.DataFrame,
        strategy: str = "margin_sampling",
        n_samples: int = 100
    ) -> pd.DataFrame:
        """Select most informative samples for annotation."""
        pass

    def should_stop(self) -> bool:
        """Determine if active learning should stop."""
        pass
```

#### 2.3.3 DSPy Integration Layer

```python
# Location: src/autolabeler/core/prompt_optimization/dspy_optimizer.py

class DSPyOptimizer(ConfigurableComponent):
    """
    DSPy integration for systematic prompt optimization.

    Features:
    - MIPROv2 optimizer for prompt improvement
    - Signature-based program definition
    - Automatic optimization on validation data
    - Prompt versioning and A/B testing
    - Cost tracking per optimization run
    """

    def define_signature(
        self,
        task_description: str,
        input_fields: list[str],
        output_fields: list[str]
    ) -> dspy.Signature:
        """Define DSPy signature for the task."""
        pass

    def optimize_prompt(
        self,
        validation_df: pd.DataFrame,
        metric_fn: Callable,
        max_cost: float = 2.0
    ) -> OptimizationResult:
        """Run MIPROv2 optimization."""
        pass
```

#### 2.3.4 Weak Supervision Integration

```python
# Location: src/autolabeler/core/weak_supervision/weak_supervision_service.py

class WeakSupervisionService(ConfigurableComponent):
    """
    Snorkel/FlyingSquid integration for programmatic labeling.

    Features:
    - Labeling function management
    - FlyingSquid aggregation (170× faster)
    - LLM-based LF generation
    - Rule quality evaluation
    - Integration with knowledge base
    """

    def add_labeling_function(
        self,
        name: str,
        function: Callable,
        description: str
    ) -> str:
        """Add a labeling function to the system."""
        pass

    def aggregate_labels(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate noisy labels using FlyingSquid."""
        pass
```

#### 2.3.5 Data Versioning Service

```python
# Location: src/autolabeler/core/versioning/data_versioning_service.py

class DataVersioningService(ConfigurableComponent):
    """
    DVC integration for data and annotation versioning.

    Features:
    - Git-like operations for datasets
    - Annotation guideline versioning
    - Full provenance tracking
    - Experiment lineage
    - Cloud storage integration (S3, GCS, Azure)
    """

    def create_version(
        self,
        df: pd.DataFrame,
        message: str,
        tag: str | None = None
    ) -> str:
        """Create a new version of the dataset."""
        pass

    def diff_versions(
        self,
        version_a: str,
        version_b: str
    ) -> dict[str, Any]:
        """Compare two versions of the dataset."""
        pass
```

---

## 3. Phase 1: Quick Wins (1-2 Weeks)

### 3.1 Overview

**Objective:** Implement high-impact, low-risk improvements that provide immediate value and foundation for later phases.

**Key Principles:**
- Minimal code changes to existing architecture
- No breaking changes to public APIs
- Quick validation and deployment
- Immediate measurable improvements

### 3.2 Feature Details

#### 3.2.1 Structured Output Validation with Instructor (2-3 days)

**Current State:** Basic Pydantic validation in LabelResponse model.

**Enhancement:**
```python
# src/autolabeler/core/validation/output_validator.py

from instructor import from_openai, Mode
from openai import OpenAI

class StructuredOutputValidator:
    """
    Enhanced structured output validation using Instructor.

    Features:
    - Automatic retry on validation failures (3-5 attempts)
    - Validation error feedback to LLM for self-correction
    - Multi-layer validation (type, rule-based, semantic)
    - Fallback strategies for persistent failures
    """

    def __init__(self, client: OpenAI, max_retries: int = 3):
        self.client = from_openai(
            client,
            mode=Mode.FUNCTIONS
        )
        self.max_retries = max_retries

    def validate_and_retry(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        validation_rules: list[Callable] | None = None
    ) -> BaseModel:
        """
        Validate output with automatic retry.

        Implementation:
        1. Attempt structured output generation
        2. Run type validation (Pydantic automatic)
        3. Run business rule validation
        4. If failure, construct error feedback prompt
        5. Retry with error context (max 3-5 times)
        6. Return validated result or raise with context
        """
        pass
```

**Integration Points:**
- Modify `LabelingService.label_text()` to use validator
- Update `EnsembleService` to handle validation errors
- Add validation metrics to quality monitoring

**Testing:**
- Unit tests: Malformed outputs, edge cases
- Integration tests: End-to-end labeling with validation
- Performance tests: Validation overhead measurement

**Success Metrics:**
- Parsing failure rate reduced by 90%+
- Successful first-attempt validation rate >85%
- Average retry count <1.2 per request

#### 3.2.2 Confidence Calibration (3-4 days)

**Current State:** Raw confidence scores from LLM without calibration.

**Enhancement:**
```python
# src/autolabeler/core/quality/confidence_calibrator.py

import numpy as np
from sklearn.isotonic import IsotonicRegression

class ConfidenceCalibrator:
    """
    Confidence score calibration using multiple methods.

    Supported Methods:
    - Temperature scaling (single parameter)
    - Platt scaling (logistic regression)
    - Isotonic regression (non-parametric)
    - Beta calibration (advanced)
    """

    def __init__(self, method: str = "temperature_scaling"):
        self.method = method
        self.calibrator = None

    def fit(
        self,
        confidence_scores: np.ndarray,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray
    ) -> None:
        """
        Fit calibration model on validation data.

        Process:
        1. Collect confidence scores + outcomes
        2. Train calibration model
        3. Validate on held-out set
        4. Store calibration parameters
        """
        pass

    def calibrate(self, confidence_scores: np.ndarray) -> np.ndarray:
        """Apply calibration to raw confidence scores."""
        pass

    def evaluate_calibration(
        self,
        confidence_scores: np.ndarray,
        true_labels: np.ndarray
    ) -> dict[str, float]:
        """
        Compute calibration metrics.

        Metrics:
        - Expected Calibration Error (ECE)
        - Brier Score
        - Log Loss
        - Reliability diagram data
        """
        pass
```

**Integration:**
- Add calibration step to `LabelingService`
- Store calibration models per dataset/model
- Track calibration metrics over time

**Testing:**
- Unit tests: Each calibration method
- Integration tests: Calibration in labeling pipeline
- Validation: ECE improvement on held-out set

**Success Metrics:**
- ECE reduced by 50%+
- Brier Score improvement of 20%+
- Confidence intervals match actual accuracy within ±5%

#### 3.2.3 Quality Metrics Dashboard (5-7 days)

**Current State:** Basic stats methods, no comprehensive monitoring.

**Enhancement:**
```python
# src/autolabeler/core/quality/quality_monitor.py

import krippendorff
from scipy import stats

class QualityMonitor:
    """
    Real-time quality monitoring dashboard.

    Tracked Metrics:
    - Krippendorff's alpha (multi-annotator agreement)
    - Per-annotator accuracy and velocity
    - Confidence distribution over time
    - Label distribution and imbalance
    - Cost per quality-adjusted annotation (CQAA)
    - Prediction latency (p50, p95, p99)
    """

    def calculate_krippendorff_alpha(
        self,
        df: pd.DataFrame,
        annotator_columns: list[str],
        value_domain: list[str] | None = None
    ) -> float:
        """
        Calculate Krippendorff's alpha.

        Implementation uses krippendorff library:
        - Handles missing data automatically
        - Supports any number of annotators
        - Works with nominal, ordinal, interval, ratio data

        Interpretation:
        - α ≥ 0.80: Reliable
        - 0.67 ≤ α < 0.80: Tentative
        - α < 0.67: Unreliable (needs intervention)
        """
        pass

    def compute_cqaa(
        self,
        df: pd.DataFrame,
        cost_column: str,
        quality_score_column: str
    ) -> float:
        """
        Compute Cost Per Quality-Adjusted Annotation.

        Formula: CQAA = Total Cost / (Annotations × Quality Score)

        This enables comparison across:
        - Manual vs automated annotation
        - Different LLM providers
        - Ensemble vs single-model approaches
        """
        pass

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        window_size: int = 100
    ) -> list[dict[str, Any]]:
        """
        Detect statistical anomalies in annotation stream.

        Methods:
        - Z-score outlier detection
        - IQR-based outlier detection
        - Sudden distribution shifts
        - Performance degradation patterns
        """
        pass

    def generate_dashboard(
        self,
        output_path: Path,
        format: str = "html"
    ) -> Path:
        """
        Generate comprehensive quality dashboard.

        Sections:
        1. Executive Summary (key metrics)
        2. Agreement Analysis (α, confusion matrix)
        3. Confidence Calibration (reliability diagram)
        4. Cost Analysis (CQAA trends)
        5. Annotator Performance (individual stats)
        6. Quality Issues (recommendations)
        """
        pass
```

**Dashboard Visualization:**
- Real-time web dashboard using Plotly/Dash
- Automated reporting (daily/weekly)
- Alert system for quality degradation

**Integration:**
- Hook into batch processing pipeline
- Real-time metric updates during labeling
- Export capabilities (JSON, HTML, PDF)

**Testing:**
- Unit tests: Each metric calculation
- Integration tests: Dashboard generation
- Performance tests: Metric computation overhead

**Success Metrics:**
- Dashboard accessible within 2 seconds
- Metrics update in real-time (<5s delay)
- Anomaly detection precision >70%, recall >80%

#### 3.2.4 Krippendorff's Alpha Implementation (2-3 days)

**Current State:** No inter-annotator agreement metrics.

**Enhancement:**
```python
# Already covered in Quality Monitor above
# Additional focus on automated workflows

class AgreementBasedWorkflow:
    """
    Stratified quality assurance based on agreement.

    Workflow:
    - α > 0.80: Auto-accept, spot-check 5%
    - 0.67 < α < 0.80: Senior reviewer check 20%
    - α < 0.67: Expert arbiter review 100%

    This reduces QA costs by 50% via acceptance sampling.
    """

    def route_for_review(
        self,
        df: pd.DataFrame,
        agreement_score: float
    ) -> dict[str, pd.DataFrame]:
        """
        Route annotations based on agreement score.

        Returns:
        - auto_accept: High agreement samples
        - human_review: Medium agreement samples
        - expert_review: Low agreement samples
        """
        pass
```

**Integration:**
- Multi-annotator labeling support
- Automated routing to review queues
- Agreement tracking over time

**Testing:**
- Unit tests: Alpha calculation accuracy
- Integration tests: Routing logic
- Validation: Compare with manual agreement assessments

**Success Metrics:**
- QA cost reduction of 40-50%
- Agreement calculation time <1s for 1000 annotations
- Routing accuracy >95%

#### 3.2.5 Cost Tracking System (2-3 days)

**Current State:** No systematic cost tracking.

**Enhancement:**
```python
# src/autolabeler/core/monitoring/cost_tracker.py

class CostTracker:
    """
    Comprehensive cost tracking and ROI analysis.

    Tracked Costs:
    - LLM API costs (per request, per token)
    - Human annotation costs (if hybrid)
    - Infrastructure costs (compute, storage)
    - Time costs (engineering, QA)

    Metrics:
    - Cost per annotation
    - Cost per quality-adjusted annotation (CQAA)
    - ROI compared to manual annotation
    - Budget alerts and projections
    """

    def __init__(self, budget_limit: float | None = None):
        self.budget_limit = budget_limit
        self.cost_history: list[dict] = []

    def track_llm_call(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        success: bool
    ) -> float:
        """
        Track individual LLM API call cost.

        Uses model-specific pricing:
        - GPT-4: $0.03/1K input, $0.06/1K output
        - GPT-3.5-turbo: $0.0015/1K input, $0.002/1K output
        - Claude-3: $0.015/1K input, $0.075/1K output
        - Open-source via OpenRouter: Variable
        """
        pass

    def compute_roi(
        self,
        manual_cost_per_annotation: float,
        quality_improvement: float = 0.0
    ) -> dict[str, float]:
        """
        Compute ROI vs manual annotation.

        Formula:
        ROI = (Benefits - Costs) / Costs × 100%

        Benefits = (Manual Cost - Automated Cost) × Volume
                   + Quality Improvement × Business Value
        """
        pass

    def project_costs(
        self,
        remaining_annotations: int,
        confidence_level: float = 0.95
    ) -> dict[str, float]:
        """
        Project costs with confidence intervals.

        Uses historical cost distribution to estimate:
        - Expected cost
        - 95% confidence interval
        - Budget utilization timeline
        - Alert if budget overrun likely
        """
        pass
```

**Integration:**
- Hook into all LLM provider calls
- Dashboard integration for cost visualization
- Budget alerts via email/Slack

**Testing:**
- Unit tests: Cost calculation accuracy
- Integration tests: Cost tracking in pipeline
- Validation: Compare with actual API bills

**Success Metrics:**
- Cost tracking accuracy within 5% of actual
- Budget alerts with 95% confidence
- ROI calculation available in real-time

### 3.3 Phase 1 Implementation Timeline

| Week | Days | Activities | Deliverables |
|------|------|-----------|--------------|
| **Week 1** | 1-2 | Structured output validation | Working validator with tests |
| | 3-4 | Confidence calibration | Calibration service + integration |
| | 5 | Integration testing | P1 features working together |
| **Week 2** | 1-2 | Quality metrics dashboard | Dashboard + Krippendorff alpha |
| | 3 | Cost tracking system | Cost tracker + ROI analysis |
| | 4-5 | Documentation & testing | Complete docs, integration tests |

### 3.4 Phase 1 Success Criteria

- ✅ Parsing failure rate <1%
- ✅ Confidence calibration ECE <0.05
- ✅ Quality dashboard accessible and updating
- ✅ Krippendorff's alpha calculated automatically
- ✅ Cost tracking within 5% accuracy
- ✅ Zero breaking changes to existing API
- ✅ All tests passing (unit + integration)
- ✅ Documentation complete

---

## 4. Phase 2: Core Features (3-6 Weeks)

### 4.1 Overview

**Objective:** Implement foundational features that transform AutoLabeler into a state-of-the-art system with systematic optimization and programmatic labeling.

**Key Principles:**
- Build on Phase 1 quality infrastructure
- Enable systematic improvement (not one-time gains)
- Support production workloads
- Maintain backward compatibility where possible

### 4.2 Feature Details

#### 4.2.1 DSPy Integration (7-10 days)

**Strategic Importance:** DSPy represents the paradigm shift from prompt engineering to prompt programming, enabling 20-50% accuracy improvements through systematic optimization.

**Implementation Plan:**

```python
# src/autolabeler/core/prompt_optimization/dspy_integration.py

import dspy
from dspy.teleprompt import MIPROv2

class DSPyLabelingModule(dspy.Module):
    """
    DSPy module for automated labeling.

    Wraps labeling logic as DSPy program for optimization.
    """

    def __init__(self, signature: dspy.Signature):
        super().__init__()
        self.signature = signature
        self.predictor = dspy.ChainOfThought(signature)

    def forward(self, text: str, examples: list[dict] | None = None):
        """Forward pass for labeling prediction."""
        return self.predictor(text=text, examples=examples)


class DSPyOptimizer:
    """
    Systematic prompt optimization using DSPy + MIPROv2.

    Features:
    - Automatic signature generation from task description
    - MIPROv2 optimization (bootstrapping + Bayesian search)
    - Prompt versioning and A/B testing
    - Cost tracking per optimization run
    - Fallback to manual prompts if optimization fails
    """

    def __init__(
        self,
        dataset_name: str,
        task_description: str,
        settings: Settings
    ):
        self.dataset_name = dataset_name
        self.task_description = task_description
        self.settings = settings

        # Initialize DSPy with LLM
        self.lm = dspy.OpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key or settings.openrouter_api_key
        )
        dspy.settings.configure(lm=self.lm)

    def define_signature(
        self,
        input_fields: dict[str, str],
        output_fields: dict[str, str]
    ) -> dspy.Signature:
        """
        Define DSPy signature from field descriptions.

        Example:
        Input: {"text": "Text to classify"}
        Output: {"label": "Predicted label", "confidence": "Confidence score"}

        Creates: Signature with typed fields and descriptions
        """
        signature_fields = {}

        for field_name, description in input_fields.items():
            signature_fields[field_name] = dspy.InputField(desc=description)

        for field_name, description in output_fields.items():
            signature_fields[field_name] = dspy.OutputField(desc=description)

        # Create signature class dynamically
        return type("LabelingSignature", (dspy.Signature,), signature_fields)

    def optimize(
        self,
        training_examples: list[dspy.Example],
        validation_examples: list[dspy.Example],
        metric_fn: Callable[[dspy.Example, dspy.Prediction], float],
        max_cost: float = 2.0,
        num_candidates: int = 10
    ) -> DSPyOptimizationResult:
        """
        Run MIPROv2 optimization.

        Process:
        1. Bootstrap high-scoring execution traces
        2. Generate grounded instruction candidates
        3. Bayesian search over instruction-demo combinations
        4. Evaluate on validation set
        5. Return best prompt with metrics

        Typical Cost: $2-5, Duration: 15-30 minutes
        Expected Improvement: 20-50% accuracy gain
        """
        # Create labeling module
        signature = self.define_signature(
            input_fields={"text": "Text to label"},
            output_fields={
                "label": "Predicted label",
                "confidence": "Confidence score (0-1)",
                "reasoning": "Explanation for the label"
            }
        )
        module = DSPyLabelingModule(signature)

        # Initialize MIPROv2 optimizer
        optimizer = MIPROv2(
            metric=metric_fn,
            num_candidates=num_candidates,
            init_temperature=1.0
        )

        # Track optimization cost
        cost_tracker = CostTracker(budget_limit=max_cost)

        # Run optimization
        optimized_module = optimizer.compile(
            module,
            trainset=training_examples,
            valset=validation_examples
        )

        # Evaluate results
        results = self._evaluate_optimized_module(
            optimized_module,
            validation_examples,
            metric_fn
        )

        return DSPyOptimizationResult(
            optimized_module=optimized_module,
            metrics=results,
            cost=cost_tracker.total_cost,
            prompt_versions=optimizer.get_prompt_versions()
        )

    def save_optimized_module(
        self,
        module: dspy.Module,
        version: str
    ) -> Path:
        """Save optimized module for later use."""
        path = self.storage_path / f"dspy_module_{version}.pkl"
        module.save(path)
        return path

    def load_optimized_module(self, version: str) -> dspy.Module:
        """Load previously optimized module."""
        path = self.storage_path / f"dspy_module_{version}.pkl"
        return dspy.Module.load(path)
```

**Integration with Existing System:**

```python
# Modify LabelingService to support DSPy

class LabelingService:
    def __init__(self, ...):
        # ... existing initialization ...

        # Add DSPy optimizer (optional)
        self.dspy_optimizer: DSPyOptimizer | None = None
        self.use_dspy: bool = config.use_dspy if config else False

    def optimize_prompts(
        self,
        training_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        text_column: str,
        label_column: str
    ) -> OptimizationResult:
        """
        Optimize prompts using DSPy MIPROv2.

        This should be run as a separate step before production labeling.
        """
        if not self.dspy_optimizer:
            self.dspy_optimizer = DSPyOptimizer(
                self.dataset_name,
                task_description="Label text with appropriate category",
                settings=self.settings
            )

        # Convert DataFrame to DSPy examples
        train_examples = self._df_to_dspy_examples(
            training_df, text_column, label_column
        )
        val_examples = self._df_to_dspy_examples(
            validation_df, text_column, label_column
        )

        # Define metric function
        def accuracy_metric(example, prediction):
            return 1.0 if prediction.label == example.label else 0.0

        # Run optimization
        result = self.dspy_optimizer.optimize(
            train_examples,
            val_examples,
            metric_fn=accuracy_metric
        )

        logger.info(f"DSPy optimization complete. Improvement: {result.improvement:.1%}")
        return result
```

**CLI Command:**

```bash
# New command for prompt optimization
autolabeler optimize-prompts \
    --dataset-name sentiment_analysis \
    --train-file train.csv \
    --val-file val.csv \
    --text-column text \
    --label-column label \
    --max-cost 5.0 \
    --output-version v1
```

**Testing Strategy:**

1. **Unit Tests:**
   - Signature generation from field specs
   - Example conversion (DataFrame → DSPy)
   - Metric function evaluation
   - Module saving/loading

2. **Integration Tests:**
   - End-to-end optimization on small dataset
   - Optimized module usage in labeling pipeline
   - Cost tracking accuracy
   - Version management

3. **Performance Tests:**
   - Optimization time vs dataset size
   - Memory usage during optimization
   - Accuracy improvement validation

**Success Metrics:**
- ✅ Optimization completes within budget ($2-5)
- ✅ Optimization time <30 minutes for 100-sample validation set
- ✅ Accuracy improvement ≥20% on validation set
- ✅ Optimized prompts integrate seamlessly into labeling pipeline
- ✅ Zero breaking changes to existing API

#### 4.2.2 Advanced RAG Implementation (7-10 days)

**Current State:** Basic FAISS-based semantic search.

**Enhancement:** Implement GraphRAG and RAPTOR variants for improved retrieval.

```python
# src/autolabeler/core/knowledge/advanced_rag.py

from typing import Literal

RAGStrategy = Literal["basic", "graphrag", "raptor", "hybrid"]

class AdvancedRAG:
    """
    Advanced RAG implementations for improved retrieval.

    Strategies:
    - basic: Current FAISS semantic search
    - graphrag: Entity-centric knowledge graphs
    - raptor: Recursive abstractive processing
    - hybrid: Combines semantic + keyword + graph
    """

    def __init__(
        self,
        strategy: RAGStrategy = "hybrid",
        knowledge_store: KnowledgeStore
    ):
        self.strategy = strategy
        self.knowledge_store = knowledge_store

        if strategy == "graphrag":
            self.graph_store = self._initialize_graph_store()
        elif strategy == "raptor":
            self.hierarchical_store = self._initialize_hierarchical_store()

    def retrieve_examples(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        Retrieve examples using selected strategy.

        Returns examples with similarity scores and provenance.
        """
        if self.strategy == "basic":
            return self._basic_retrieval(query, k)
        elif self.strategy == "graphrag":
            return self._graphrag_retrieval(query, k)
        elif self.strategy == "raptor":
            return self._raptor_retrieval(query, k)
        else:  # hybrid
            return self._hybrid_retrieval(query, k)

    def _graphrag_retrieval(
        self,
        query: str,
        k: int
    ) -> list[dict[str, Any]]:
        """
        GraphRAG: Entity-centric knowledge graph retrieval.

        Process:
        1. Extract entities from query
        2. Find entity communities in knowledge graph
        3. Retrieve community summaries
        4. Rank by relevance to query
        5. Return top-k with provenance

        Improvement: 6.4 points on multi-hop queries
        """
        pass

    def _raptor_retrieval(
        self,
        query: str,
        k: int
    ) -> list[dict[str, Any]]:
        """
        RAPTOR: Recursive abstractive processing with clustering.

        Process:
        1. Build hierarchical cluster tree of examples
        2. Generate abstract summaries at each level
        3. Query across multiple abstraction levels
        4. Return most relevant examples + summaries

        Improvement: Better for diverse example retrieval
        """
        pass

    def _hybrid_retrieval(
        self,
        query: str,
        k: int
    ) -> list[dict[str, Any]]:
        """
        Hybrid: Combines semantic, keyword, and graph.

        Process:
        1. Semantic search (embeddings)
        2. Keyword search (BM25)
        3. Graph traversal (if entities present)
        4. Cross-encoder reranking
        5. Diversity-based selection

        Improvement: Most robust across query types
        """
        pass
```

**Integration:**

```python
# Modify KnowledgeStore to support advanced RAG

class KnowledgeStore:
    def __init__(self, ...):
        # ... existing initialization ...

        # Add advanced RAG support
        self.rag_strategy: RAGStrategy = settings.rag_strategy
        self.advanced_rag = AdvancedRAG(
            strategy=self.rag_strategy,
            knowledge_store=self
        )

    def find_similar_examples(self, query: str, k: int = 5):
        """Use advanced RAG if configured."""
        if self.rag_strategy != "basic":
            return self.advanced_rag.retrieve_examples(query, k)
        else:
            return self._basic_faiss_search(query, k)
```

**Testing:**
- Unit tests: Each RAG strategy independently
- Integration tests: RAG in labeling pipeline
- Performance tests: Retrieval latency, accuracy
- Ablation study: Compare strategies on benchmark

**Success Metrics:**
- ✅ GraphRAG improves multi-hop retrieval by ≥5%
- ✅ RAPTOR improves diversity (measured by unique examples retrieved)
- ✅ Hybrid strategy never worse than basic (Pareto improvement)
- ✅ Retrieval latency <500ms for k=5

#### 4.2.3 Active Learning Loop (10-14 days)

**Strategic Importance:** Active learning reduces annotation requirements by 40-70%, translating to $20k-50k savings on large projects.

**Implementation:**

```python
# src/autolabeler/core/active_learning/active_learning_service.py

from enum import Enum
from sklearn.cluster import KMeans

class SamplingStrategy(str, Enum):
    LEAST_CONFIDENCE = "least_confidence"
    MARGIN_SAMPLING = "margin_sampling"
    ENTROPY_SAMPLING = "entropy_sampling"
    TCM_HYBRID = "tcm_hybrid"  # TypiClust → Margin

class ActiveLearningService(ConfigurableComponent):
    """
    Active learning implementation with hybrid strategies.

    Features:
    - Multiple uncertainty sampling methods
    - Diversity-based sampling (TypiClust)
    - Hybrid TCM strategy for cold start
    - Automatic stopping criteria
    - Integration with labeling pipeline
    """

    def __init__(
        self,
        dataset_name: str,
        settings: Settings,
        initial_strategy: SamplingStrategy = SamplingStrategy.TCM_HYBRID
    ):
        super().__init__(
            component_type="active_learning",
            dataset_name=dataset_name,
            settings=settings
        )
        self.strategy = initial_strategy
        self.iteration = 0
        self.performance_history: list[float] = []

        # Thresholds
        self.cold_start_threshold = 0.1  # 10% of budget for diversity
        self.stopping_improvement_threshold = 0.01  # 1% improvement
        self.stopping_consecutive_plateaus = 3

    def select_samples(
        self,
        unlabeled_df: pd.DataFrame,
        model: Any,  # Trained model for uncertainty estimation
        n_samples: int = 100,
        embeddings: np.ndarray | None = None
    ) -> pd.DataFrame:
        """
        Select most informative samples for annotation.

        Args:
            unlabeled_df: Pool of unlabeled examples
            model: Current model for uncertainty estimation
            n_samples: Number of samples to select
            embeddings: Pre-computed embeddings (optional)

        Returns:
            DataFrame with selected samples
        """
        # Determine strategy based on iteration
        if self.iteration == 0 or self._is_cold_start():
            strategy = SamplingStrategy.TCM_HYBRID
        else:
            strategy = self.strategy

        logger.info(f"Active learning iteration {self.iteration}: Using {strategy}")

        if strategy == SamplingStrategy.LEAST_CONFIDENCE:
            return self._least_confidence_sampling(
                unlabeled_df, model, n_samples
            )
        elif strategy == SamplingStrategy.MARGIN_SAMPLING:
            return self._margin_sampling(
                unlabeled_df, model, n_samples
            )
        elif strategy == SamplingStrategy.ENTROPY_SAMPLING:
            return self._entropy_sampling(
                unlabeled_df, model, n_samples
            )
        else:  # TCM_HYBRID
            return self._tcm_hybrid_sampling(
                unlabeled_df, model, n_samples, embeddings
            )

    def _is_cold_start(self) -> bool:
        """Determine if we're still in cold start phase."""
        # Cold start if <10% of typical dataset labeled
        # This is heuristic-based
        return len(self.performance_history) < 3

    def _least_confidence_sampling(
        self,
        unlabeled_df: pd.DataFrame,
        model: Any,
        n_samples: int
    ) -> pd.DataFrame:
        """
        Least confidence: 1 - P(ŷ|x)

        Best for: Binary classification
        """
        # Get predictions with confidence
        predictions = model.predict_proba(unlabeled_df)

        # Calculate uncertainty (1 - max probability)
        max_probs = np.max(predictions, axis=1)
        uncertainty = 1 - max_probs

        # Select top-k most uncertain
        top_indices = np.argsort(uncertainty)[-n_samples:]

        return unlabeled_df.iloc[top_indices].copy()

    def _margin_sampling(
        self,
        unlabeled_df: pd.DataFrame,
        model: Any,
        n_samples: int
    ) -> pd.DataFrame:
        """
        Margin sampling: Difference between top two predictions.

        Best for: Multi-class with clear boundaries
        """
        predictions = model.predict_proba(unlabeled_df)

        # Sort predictions for each sample
        sorted_probs = np.sort(predictions, axis=1)

        # Margin = difference between top 2
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]

        # Select samples with smallest margins (most uncertain)
        top_indices = np.argsort(margins)[:n_samples]

        return unlabeled_df.iloc[top_indices].copy()

    def _entropy_sampling(
        self,
        unlabeled_df: pd.DataFrame,
        model: Any,
        n_samples: int
    ) -> pd.DataFrame:
        """
        Entropy sampling: H(p) = -Σ p(y|x) log p(y|x)

        Best for: Many classes requiring full distribution
        """
        predictions = model.predict_proba(unlabeled_df)

        # Calculate entropy
        entropy = -np.sum(
            predictions * np.log(predictions + 1e-10),
            axis=1
        )

        # Select top-k highest entropy
        top_indices = np.argsort(entropy)[-n_samples:]

        return unlabeled_df.iloc[top_indices].copy()

    def _tcm_hybrid_sampling(
        self,
        unlabeled_df: pd.DataFrame,
        model: Any,
        n_samples: int,
        embeddings: np.ndarray | None = None
    ) -> pd.DataFrame:
        """
        TCM Hybrid: TypiClust (diversity) → Margin (uncertainty)

        Process:
        1. If cold start: Use TypiClust for first 5-10% of budget
        2. Cluster embeddings into k clusters
        3. Select representatives from each cluster
        4. Once sufficient coverage: Switch to margin sampling

        This solves the cold start problem where uncertainty
        methods perform poorly without initial training data.
        """
        if self._is_cold_start():
            # TypiClust: Cluster-based diversity sampling
            if embeddings is None:
                # Generate embeddings if not provided
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.settings.embedding_model)
                embeddings = model.encode(
                    unlabeled_df["text"].tolist()
                )

            # Cluster embeddings
            n_clusters = min(n_samples, len(unlabeled_df) // 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)

            # Select representative from each cluster
            # (closest to cluster centroid)
            selected_indices = []
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                cluster_embeddings = embeddings[cluster_mask]
                cluster_indices = np.where(cluster_mask)[0]

                # Find closest to centroid
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(
                    cluster_embeddings - centroid,
                    axis=1
                )
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)

            # If need more samples, add randomly
            if len(selected_indices) < n_samples:
                remaining = n_samples - len(selected_indices)
                available = set(range(len(unlabeled_df))) - set(selected_indices)
                additional = np.random.choice(
                    list(available),
                    size=remaining,
                    replace=False
                )
                selected_indices.extend(additional)

            return unlabeled_df.iloc[selected_indices].copy()
        else:
            # Switch to margin sampling after cold start
            return self._margin_sampling(unlabeled_df, model, n_samples)

    def should_stop(
        self,
        current_performance: float,
        validation_df: pd.DataFrame
    ) -> tuple[bool, str]:
        """
        Determine if active learning should stop.

        Stopping Criteria:
        1. Performance plateau: <1% improvement for 3 consecutive iterations
        2. Uncertainty threshold: Average uncertainty below threshold
        3. Budget exhausted: Reached maximum annotation budget

        Returns:
            (should_stop, reason)
        """
        self.performance_history.append(current_performance)

        # Check for performance plateau
        if len(self.performance_history) >= self.stopping_consecutive_plateaus:
            recent_improvements = [
                self.performance_history[i] - self.performance_history[i-1]
                for i in range(-self.stopping_consecutive_plateaus + 1, 0)
            ]

            if all(
                imp < self.stopping_improvement_threshold
                for imp in recent_improvements
            ):
                return True, f"Performance plateau: <{self.stopping_improvement_threshold:.1%} improvement for {self.stopping_consecutive_plateaus} iterations"

        # Check uncertainty threshold
        # (requires model predictions on validation set)
        # If average uncertainty is very low, most examples are easy

        return False, ""

    def get_statistics(self) -> dict[str, Any]:
        """Get active learning statistics."""
        return {
            "iteration": self.iteration,
            "strategy": self.strategy.value,
            "performance_history": self.performance_history,
            "improvement_per_iteration": [
                self.performance_history[i] - self.performance_history[i-1]
                for i in range(1, len(self.performance_history))
            ],
            "total_samples_selected": self.iteration * 100  # assuming 100 per iteration
        }
```

**Integration with Main Workflow:**

```python
# New CLI command for active learning
@cli.command()
@click.option("--dataset-name", required=True)
@click.option("--unlabeled-file", required=True, type=click.Path(path_type=Path))
@click.option("--initial-labeled-file", required=True, type=click.Path(path_type=Path))
@click.option("--text-column", required=True)
@click.option("--label-column", required=True)
@click.option("--strategy", default="tcm_hybrid")
@click.option("--samples-per-iteration", default=100, type=int)
@click.option("--max-iterations", default=10, type=int)
def active_learn(
    dataset_name: str,
    unlabeled_file: Path,
    initial_labeled_file: Path,
    text_column: str,
    label_column: str,
    strategy: str,
    samples_per_iteration: int,
    max_iterations: int
):
    """Run active learning loop."""
    settings = Settings()

    # Initialize services
    labeler = AutoLabeler(dataset_name, settings)
    al_service = ActiveLearningService(
        dataset_name,
        settings,
        initial_strategy=SamplingStrategy(strategy)
    )

    # Load data
    unlabeled_df = _load_data(unlabeled_file)
    labeled_df = _load_data(initial_labeled_file)

    # Add initial training data
    labeler.add_training_data(labeled_df, text_column, label_column)

    for iteration in range(max_iterations):
        logger.info(f"Active learning iteration {iteration + 1}/{max_iterations}")

        # Train model on current labeled data
        # (simplified - in practice, you'd train a proper model)
        model = labeler._train_temp_model(labeled_df, text_column, label_column)

        # Select samples
        selected_samples = al_service.select_samples(
            unlabeled_df,
            model,
            n_samples=samples_per_iteration
        )

        # Human annotation (or automatic for testing)
        # In production, this would route to annotation UI
        logger.info(f"Selected {len(selected_samples)} samples for annotation")
        logger.info("Please annotate these samples and press Enter to continue...")
        input()  # Wait for human annotation

        # Reload labeled data (assumes human updated the file)
        labeled_df = _load_data(initial_labeled_file)

        # Evaluate performance
        val_metrics = labeler.evaluate(
            labeled_df.tail(100),  # Use recent labels as validation
            label_column,
            "predicted_label"
        )

        # Check stopping criteria
        should_stop, reason = al_service.should_stop(
            val_metrics["accuracy"],
            labeled_df.tail(100)
        )

        if should_stop:
            logger.info(f"Stopping active learning: {reason}")
            break

        # Remove selected samples from unlabeled pool
        unlabeled_df = unlabeled_df.drop(selected_samples.index)

        # Update iteration counter
        al_service.iteration += 1

    # Report final statistics
    stats = al_service.get_statistics()
    logger.info(f"Active learning complete. Final statistics:")
    logger.info(f"  Total iterations: {stats['iteration']}")
    logger.info(f"  Final accuracy: {stats['performance_history'][-1]:.2%}")
    logger.info(f"  Total samples annotated: {stats['total_samples_selected']}")
```

**Testing:**
- Unit tests: Each sampling strategy independently
- Integration tests: Full active learning loop on synthetic data
- Performance tests: Sampling time, model training time
- Validation: Compare with random sampling baseline (must be 40-70% better)

**Success Metrics:**
- ✅ Active learning achieves target accuracy with 40-70% fewer annotations
- ✅ TCM hybrid strategy outperforms pure uncertainty in cold start
- ✅ Stopping criteria prevents over-annotation (triggers within 3 iterations of plateau)
- ✅ Sample selection time <10s for 10k unlabeled examples

#### 4.2.4 Weak Supervision Integration (10-14 days)

**Strategic Importance:** Weak supervision enables programmatic labeling, reducing dependency on expensive LLM calls while incorporating domain expertise.

**Implementation:**

```python
# src/autolabeler/core/weak_supervision/weak_supervision_service.py

from typing import Callable, Literal
import numpy as np
from dataclasses import dataclass

LabelingFunctionResult = Literal[-1, 0, 1, 2, 3, 4, 5]  # -1 = abstain, 0+ = label

@dataclass
class LabelingFunction:
    """A labeling function that programmatically assigns labels."""
    name: str
    function: Callable[[str], LabelingFunctionResult]
    description: str
    category: str  # "heuristic", "gazetteer", "model", "llm"
    created_at: str
    accuracy_estimate: float = 0.7  # Initial estimate


class WeakSupervisionService(ConfigurableComponent):
    """
    Weak supervision using Snorkel + FlyingSquid.

    Features:
    - Labeling function management
    - FlyingSquid aggregation (170× faster than EM)
    - LLM-based automatic LF generation
    - Conflict resolution and quality estimation
    - Integration with knowledge base
    """

    def __init__(
        self,
        dataset_name: str,
        settings: Settings
    ):
        super().__init__(
            component_type="weak_supervision",
            dataset_name=dataset_name,
            settings=settings
        )

        self.labeling_functions: dict[str, LabelingFunction] = {}
        self.label_matrix: np.ndarray | None = None
        self.aggregated_labels: np.ndarray | None = None
        self.lf_stats: dict[str, dict] = {}

    def add_labeling_function(
        self,
        name: str,
        function: Callable[[str], LabelingFunctionResult],
        description: str,
        category: str = "heuristic"
    ) -> str:
        """
        Add a labeling function to the system.

        Args:
            name: Unique identifier for the LF
            function: Callable that takes text and returns label (or -1 for abstain)
            description: Human-readable description of the LF
            category: Category of LF (heuristic, gazetteer, model, llm)

        Returns:
            LF ID

        Example:
            >>> def positive_keywords(text: str) -> int:
            ...     if any(word in text.lower() for word in ["great", "excellent", "amazing"]):
            ...         return 1  # positive
            ...     return -1  # abstain
            >>>
            >>> service.add_labeling_function(
            ...     "positive_keywords",
            ...     positive_keywords,
            ...     "Detects positive sentiment keywords"
            ... )
        """
        lf = LabelingFunction(
            name=name,
            function=function,
            description=description,
            category=category,
            created_at=datetime.now().isoformat()
        )

        self.labeling_functions[name] = lf
        logger.info(f"Added labeling function: {name} ({category})")

        return name

    def apply_labeling_functions(
        self,
        df: pd.DataFrame,
        text_column: str
    ) -> np.ndarray:
        """
        Apply all labeling functions to dataset.

        Returns:
            Label matrix of shape (n_samples, n_lfs)
            where each entry is the label from that LF
            (-1 indicates abstention)
        """
        n_samples = len(df)
        n_lfs = len(self.labeling_functions)

        label_matrix = np.full((n_samples, n_lfs), -1, dtype=int)

        texts = df[text_column].tolist()

        for lf_idx, (lf_name, lf) in enumerate(self.labeling_functions.items()):
            logger.info(f"Applying labeling function: {lf_name}")

            for sample_idx, text in enumerate(texts):
                try:
                    label = lf.function(text)
                    label_matrix[sample_idx, lf_idx] = label
                except Exception as e:
                    logger.warning(
                        f"LF {lf_name} failed on sample {sample_idx}: {e}"
                    )
                    label_matrix[sample_idx, lf_idx] = -1

        self.label_matrix = label_matrix
        return label_matrix

    def aggregate_labels(
        self,
        label_matrix: np.ndarray | None = None,
        method: str = "flyingsquid"
    ) -> np.ndarray:
        """
        Aggregate noisy labels using FlyingSquid.

        Args:
            label_matrix: Optional label matrix (uses cached if None)
            method: Aggregation method ("flyingsquid", "majority_vote", "snorkel")

        Returns:
            Aggregated probabilistic labels of shape (n_samples, n_classes)

        FlyingSquid Process:
        1. Estimate LF accuracies via triplet methods
        2. Model LF correlations
        3. Compute maximum likelihood labels
        4. Return probabilistic label distributions

        Advantage: 170× faster than iterative EM methods
        """
        if label_matrix is None:
            if self.label_matrix is None:
                raise ValueError("No label matrix available. Run apply_labeling_functions first.")
            label_matrix = self.label_matrix

        if method == "flyingsquid":
            return self._flyingsquid_aggregation(label_matrix)
        elif method == "majority_vote":
            return self._majority_vote_aggregation(label_matrix)
        else:  # snorkel
            return self._snorkel_aggregation(label_matrix)

    def _flyingsquid_aggregation(
        self,
        label_matrix: np.ndarray
    ) -> np.ndarray:
        """
        FlyingSquid aggregation using triplet methods.

        Implementation uses closed-form solution for fast computation.
        """
        # Implementation details:
        # 1. Compute triplet statistics
        # 2. Estimate LF accuracies
        # 3. Model label correlations
        # 4. Compute probabilistic labels

        # Placeholder - actual implementation would use flyingsquid library
        from flyingsquid.label_model import LabelModel

        model = LabelModel(
            n_classes=self._infer_n_classes(label_matrix)
        )
        model.fit(label_matrix)

        probabilistic_labels = model.predict_proba(label_matrix)

        # Store LF statistics
        self._compute_lf_statistics(label_matrix, probabilistic_labels)

        return probabilistic_labels

    def _compute_lf_statistics(
        self,
        label_matrix: np.ndarray,
        aggregated_labels: np.ndarray
    ) -> None:
        """
        Compute statistics for each labeling function.

        Metrics:
        - Coverage: Fraction of samples labeled (not abstained)
        - Accuracy: Estimated accuracy vs aggregated labels
        - Conflicts: Number of conflicts with other LFs
        - Polarity: Distribution of labels
        """
        for lf_idx, (lf_name, lf) in enumerate(self.labeling_functions.items()):
            lf_labels = label_matrix[:, lf_idx]

            # Coverage
            coverage = np.mean(lf_labels != -1)

            # Accuracy (estimated)
            valid_mask = lf_labels != -1
            if valid_mask.sum() > 0:
                true_labels = np.argmax(aggregated_labels[valid_mask], axis=1)
                accuracy = np.mean(lf_labels[valid_mask] == true_labels)
            else:
                accuracy = 0.0

            # Conflicts
            conflicts = 0
            for other_idx in range(label_matrix.shape[1]):
                if other_idx != lf_idx:
                    both_labeled = (lf_labels != -1) & (label_matrix[:, other_idx] != -1)
                    if both_labeled.sum() > 0:
                        conflicts += np.sum(
                            lf_labels[both_labeled] != label_matrix[both_labeled, other_idx]
                        )

            self.lf_stats[lf_name] = {
                "coverage": float(coverage),
                "estimated_accuracy": float(accuracy),
                "conflicts": int(conflicts),
                "num_samples_labeled": int(valid_mask.sum())
            }

            logger.info(
                f"LF {lf_name}: "
                f"Coverage={coverage:.2%}, "
                f"Accuracy={accuracy:.2%}, "
                f"Conflicts={conflicts}"
            )

    def generate_labeling_functions_with_llm(
        self,
        labeled_df: pd.DataFrame,
        text_column: str,
        label_column: str,
        num_functions: int = 5
    ) -> list[str]:
        """
        Automatically generate labeling functions using LLM.

        Process:
        1. Analyze labeled examples
        2. Identify patterns (keywords, phrases, structures)
        3. Generate LF code using LLM (GPT-4)
        4. Validate and test LFs
        5. Add to system

        Returns:
            List of generated LF names
        """
        # Implementation would use GPT-4 to generate Python code
        # for labeling functions based on labeled examples
        pass

    def get_lf_report(self) -> pd.DataFrame:
        """
        Generate comprehensive report on labeling functions.

        Returns:
            DataFrame with LF name, category, coverage, accuracy, conflicts
        """
        report_data = []
        for lf_name, lf in self.labeling_functions.items():
            stats = self.lf_stats.get(lf_name, {})
            report_data.append({
                "name": lf_name,
                "category": lf.category,
                "description": lf.description,
                "coverage": stats.get("coverage", 0.0),
                "accuracy": stats.get("estimated_accuracy", 0.0),
                "conflicts": stats.get("conflicts", 0),
                "samples_labeled": stats.get("num_samples_labeled", 0)
            })

        return pd.DataFrame(report_data)
```

**Example Usage:**

```python
# Example: Create labeling functions for sentiment analysis

from autolabeler.core.weak_supervision import WeakSupervisionService

# Initialize service
ws_service = WeakSupervisionService("sentiment_analysis", settings)

# Define labeling functions
def positive_keywords(text: str) -> int:
    """LF: Positive sentiment keywords"""
    positive_words = ["great", "excellent", "amazing", "fantastic", "love"]
    if any(word in text.lower() for word in positive_words):
        return 1  # positive
    return -1  # abstain

def negative_keywords(text: str) -> int:
    """LF: Negative sentiment keywords"""
    negative_words = ["terrible", "awful", "horrible", "hate", "worst"]
    if any(word in text.lower() for word in negative_words):
        return 0  # negative
    return -1  # abstain

def has_exclamation(text: str) -> int:
    """LF: Exclamation marks indicate strong sentiment"""
    if "!" in text:
        # Count ratio of positive to negative words to determine polarity
        positive_count = sum(1 for word in ["great", "good"] if word in text.lower())
        negative_count = sum(1 for word in ["bad", "terrible"] if word in text.lower())
        if positive_count > negative_count:
            return 1  # positive
        elif negative_count > positive_count:
            return 0  # negative
    return -1  # abstain

# Add LFs to system
ws_service.add_labeling_function("positive_keywords", positive_keywords, "Detects positive keywords")
ws_service.add_labeling_function("negative_keywords", negative_keywords, "Detects negative keywords")
ws_service.add_labeling_function("has_exclamation", has_exclamation, "Uses exclamation marks")

# Apply LFs to unlabeled data
label_matrix = ws_service.apply_labeling_functions(unlabeled_df, "text")

# Aggregate labels
probabilistic_labels = ws_service.aggregate_labels()

# Get hard labels (argmax)
hard_labels = np.argmax(probabilistic_labels, axis=1)

# Add to dataframe
unlabeled_df["weak_label"] = hard_labels
unlabeled_df["weak_confidence"] = np.max(probabilistic_labels, axis=1)

# Generate report
lf_report = ws_service.get_lf_report()
print(lf_report)
```

**CLI Integration:**

```bash
# Apply weak supervision to dataset
autolabeler weak-supervise \
    --dataset-name sentiment_analysis \
    --unlabeled-file unlabeled.csv \
    --lf-definitions lfs.py \  # Python file with LF definitions
    --output-file weak_labeled.csv \
    --text-column text
```

**Testing:**
- Unit tests: Individual LF application, aggregation methods
- Integration tests: Full weak supervision pipeline
- Validation: Compare with ground truth labels (if available)
- Performance tests: Aggregation speed vs dataset size

**Success Metrics:**
- ✅ FlyingSquid aggregation 100× faster than baseline EM
- ✅ Weak supervision achieves 70-80% accuracy vs ground truth
- ✅ Coverage (fraction labeled) >80%
- ✅ LF conflict resolution maintains label consistency

#### 4.2.5 Data Versioning with DVC (3-5 days)

**Strategic Importance:** Data versioning is essential for reproducibility, experiment tracking, and debugging quality issues.

**Implementation:**

```python
# src/autolabeler/core/versioning/dvc_integration.py

import subprocess
from pathlib import Path
import json

class DVCIntegration:
    """
    DVC integration for data and annotation versioning.

    Features:
    - Git-like operations (add, commit, checkout, diff)
    - Cloud storage integration (S3, GCS, Azure)
    - Annotation guideline versioning
    - Experiment lineage tracking
    - Automatic dataset snapshots
    """

    def __init__(
        self,
        dataset_name: str,
        repo_path: Path,
        remote: str | None = None
    ):
        self.dataset_name = dataset_name
        self.repo_path = repo_path
        self.remote = remote

        # Initialize DVC if not already initialized
        if not (repo_path / ".dvc").exists():
            self._initialize_dvc()

    def _initialize_dvc(self) -> None:
        """Initialize DVC repository."""
        subprocess.run(["dvc", "init"], cwd=self.repo_path, check=True)
        logger.info("Initialized DVC repository")

        if self.remote:
            subprocess.run(
                ["dvc", "remote", "add", "-d", "storage", self.remote],
                cwd=self.repo_path,
                check=True
            )
            logger.info(f"Added DVC remote: {self.remote}")

    def add_dataset(
        self,
        dataset_path: Path,
        message: str | None = None
    ) -> str:
        """
        Add dataset to DVC tracking.

        Args:
            dataset_path: Path to dataset file
            message: Commit message

        Returns:
            Commit hash
        """
        # Add to DVC
        subprocess.run(
            ["dvc", "add", str(dataset_path)],
            cwd=self.repo_path,
            check=True
        )

        # Add .dvc file to git
        dvc_file = dataset_path.with_suffix(dataset_path.suffix + ".dvc")
        subprocess.run(
            ["git", "add", str(dvc_file)],
            cwd=self.repo_path,
            check=True
        )

        # Commit
        commit_message = message or f"Add dataset: {dataset_path.name}"
        subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=self.repo_path,
            check=True
        )

        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        commit_hash = result.stdout.strip()

        logger.info(f"Added dataset to DVC: {commit_hash}")
        return commit_hash

    def checkout_version(self, version: str) -> None:
        """
        Checkout specific version of dataset.

        Args:
            version: Git commit hash or tag
        """
        # Checkout git version
        subprocess.run(
            ["git", "checkout", version],
            cwd=self.repo_path,
            check=True
        )

        # Checkout DVC files
        subprocess.run(
            ["dvc", "checkout"],
            cwd=self.repo_path,
            check=True
        )

        logger.info(f"Checked out version: {version}")

    def diff_versions(
        self,
        version_a: str,
        version_b: str,
        dataset_path: Path
    ) -> dict[str, Any]:
        """
        Compare two versions of dataset.

        Returns:
            Dict with diff statistics
        """
        # Checkout version A
        self.checkout_version(version_a)
        df_a = pd.read_csv(dataset_path) if dataset_path.suffix == ".csv" else pd.read_parquet(dataset_path)

        # Checkout version B
        self.checkout_version(version_b)
        df_b = pd.read_csv(dataset_path) if dataset_path.suffix == ".csv" else pd.read_parquet(dataset_path)

        # Compute diff
        diff = {
            "version_a": version_a,
            "version_b": version_b,
            "rows_added": len(df_b) - len(df_a),
            "rows_removed": 0,  # More sophisticated diff needed
            "columns_added": list(set(df_b.columns) - set(df_a.columns)),
            "columns_removed": list(set(df_a.columns) - set(df_b.columns)),
            "label_distribution_a": df_a["label"].value_counts().to_dict() if "label" in df_a.columns else {},
            "label_distribution_b": df_b["label"].value_counts().to_dict() if "label" in df_b.columns else {},
        }

        return diff

    def list_versions(self) -> list[dict[str, Any]]:
        """
        List all dataset versions.

        Returns:
            List of versions with metadata
        """
        result = subprocess.run(
            ["git", "log", "--pretty=format:%H|%an|%ad|%s", "--date=short"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )

        versions = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            hash, author, date, message = line.split("|", 3)
            versions.append({
                "hash": hash,
                "author": author,
                "date": date,
                "message": message
            })

        return versions

    def push_to_remote(self) -> None:
        """Push dataset to remote storage."""
        subprocess.run(
            ["dvc", "push"],
            cwd=self.repo_path,
            check=True
        )
        logger.info("Pushed dataset to remote storage")

    def pull_from_remote(self) -> None:
        """Pull dataset from remote storage."""
        subprocess.run(
            ["dvc", "pull"],
            cwd=self.repo_path,
            check=True
        )
        logger.info("Pulled dataset from remote storage")
```

**Integration with AutoLabeler:**

```python
# Modify AutoLabeler to support versioning

class AutoLabeler:
    def __init__(self, ...):
        # ... existing initialization ...

        # Add DVC integration
        self.versioning_enabled = settings.enable_versioning
        if self.versioning_enabled:
            self.dvc = DVCIntegration(
                dataset_name,
                repo_path=Path.cwd(),
                remote=settings.dvc_remote
            )

    def save_dataset_version(
        self,
        df: pd.DataFrame,
        message: str
    ) -> str:
        """
        Save a versioned snapshot of the dataset.

        Returns:
            Version hash
        """
        if not self.versioning_enabled:
            logger.warning("Versioning not enabled")
            return ""

        # Save dataset
        output_path = self.storage_path / f"{self.dataset_name}_latest.parquet"
        df.to_parquet(output_path, index=False)

        # Version with DVC
        version_hash = self.dvc.add_dataset(output_path, message)

        return version_hash
```

**CLI Commands:**

```bash
# Version current dataset
autolabeler version create \
    --dataset-name sentiment_analysis \
    --message "Initial labeled dataset with 1000 examples"

# List versions
autolabeler version list \
    --dataset-name sentiment_analysis

# Checkout previous version
autolabeler version checkout \
    --dataset-name sentiment_analysis \
    --version abc123

# Compare versions
autolabeler version diff \
    --dataset-name sentiment_analysis \
    --version-a abc123 \
    --version-b def456
```

**Testing:**
- Unit tests: DVC command wrappers
- Integration tests: Full versioning workflow
- Performance tests: Versioning overhead

**Success Metrics:**
- ✅ Versioning adds <5% overhead to save operations
- ✅ Checkout operation completes in <10s
- ✅ Full lineage tracking from raw data to final annotations
- ✅ Zero data loss across versions

### 4.3 Phase 2 Implementation Timeline

| Week | Days | Activities | Deliverables |
|------|------|-----------|--------------|
| **Week 3** | 1-3 | DSPy integration - core | DSPy module wrappers |
| | 4-5 | DSPy optimization implementation | MIPROv2 optimizer working |
| **Week 4** | 1-2 | DSPy testing & integration | End-to-end optimization |
| | 3-5 | Advanced RAG implementation | GraphRAG, RAPTOR, Hybrid |
| **Week 5** | 1-3 | Active learning - core | Sampling strategies implemented |
| | 4-5 | Active learning - integration | CLI commands, stopping criteria |
| **Week 6** | 1-3 | Weak supervision - core | LF management, FlyingSquid |
| | 4-5 | Weak supervision - testing | Full pipeline working |
| **Week 7** | 1-2 | Data versioning - DVC integration | Versioning commands working |
| | 3-5 | Phase 2 integration testing | All features working together |

### 4.4 Phase 2 Success Criteria

- ✅ DSPy optimization achieves 20-50% accuracy improvement
- ✅ Advanced RAG improves retrieval quality by ≥10%
- ✅ Active learning reduces annotation needs by 40-70%
- ✅ Weak supervision achieves 70-80% accuracy
- ✅ Data versioning enables reproducible experiments
- ✅ All Phase 1 features still working (backward compatible)
- ✅ Comprehensive test coverage (>80%)
- ✅ Documentation updated with new features

---

## 5. Phase 3: Advanced Features (7-12 Weeks)

### 5.1 Overview

**Objective:** Implement production-grade monitoring, drift detection, and advanced ensemble methods for enterprise deployment.

**Key Principles:**
- Production-ready reliability and monitoring
- Advanced quality control mechanisms
- Enterprise scalability
- Comprehensive observability

### 5.2 Feature Details

#### 5.2.1 Multi-Agent Architecture (14-21 days)

**Strategic Importance:** Multi-agent systems improve quality by 10-15% through specialization and validation.

**Implementation:**

```python
# src/autolabeler/core/agents/multi_agent_system.py

from typing import Protocol
from enum import Enum

class AgentRole(str, Enum):
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    VALIDATOR = "validator"
    LEARNER = "learner"

class Agent(Protocol):
    """Protocol for agent interface."""

    def process(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process a task and return results."""
        ...

    def get_capabilities(self) -> list[str]:
        """Return list of capabilities."""
        ...


class SpecialistAgent:
    """
    Specialist agent for specific annotation aspects.

    Examples:
    - Entity recognition specialist
    - Relation extraction specialist
    - Sentiment analysis specialist
    """

    def __init__(
        self,
        name: str,
        specialty: str,
        model: str,
        settings: Settings
    ):
        self.name = name
        self.specialty = specialty
        self.model = model
        self.settings = settings

    def process(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process specialized annotation task."""
        pass


class CoordinatorAgent:
    """
    Coordinator agent for task routing and orchestration.

    Responsibilities:
    - Route tasks to appropriate specialists
    - Combine specialist outputs
    - Manage dependencies between subtasks
    """

    def __init__(self, specialists: list[SpecialistAgent]):
        self.specialists = specialists

    def route_task(self, task: dict[str, Any]) -> SpecialistAgent:
        """Route task to appropriate specialist."""
        pass

    def coordinate(self, task: dict[str, Any]) -> dict[str, Any]:
        """Coordinate multi-agent annotation."""
        pass


class ValidatorAgent:
    """
    Validator agent for quality control.

    Responsibilities:
    - Validate specialist outputs
    - Detect inconsistencies
    - Request refinement if needed
    """

    def validate(self, predictions: list[dict]) -> dict[str, Any]:
        """Validate annotations for consistency."""
        pass


class MultiAgentSystem:
    """
    Multi-agent annotation system with coordination.

    Architecture:
    - Specialist agents for different aspects
    - Coordinator for task routing
    - Validator for quality control
    - Learner for continuous improvement
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.specialists: list[SpecialistAgent] = []
        self.coordinator: CoordinatorAgent | None = None
        self.validator: ValidatorAgent | None = None

    def add_specialist(
        self,
        name: str,
        specialty: str,
        model: str
    ) -> None:
        """Add specialist agent."""
        specialist = SpecialistAgent(name, specialty, model, self.settings)
        self.specialists.append(specialist)

    def annotate(self, text: str) -> dict[str, Any]:
        """Perform multi-agent annotation."""
        # 1. Coordinator routes to specialists
        # 2. Specialists perform specialized annotation
        # 3. Validator checks consistency
        # 4. Coordinator combines results
        pass
```

**Success Metrics:**
- ✅ Multi-agent system improves accuracy by 10-15%
- ✅ Specialization reduces individual agent complexity
- ✅ Validation catches inconsistencies (>80% precision)

#### 5.2.2 Drift Detection (7-10 days)

**Implementation:**

```python
# src/autolabeler/core/monitoring/drift_detector.py

from scipy import stats
import numpy as np

class DriftDetector:
    """
    Comprehensive drift detection for annotation quality.

    Monitors:
    - Statistical drift (KS test, PSI)
    - Embedding drift (domain classifier)
    - Performance drift (accuracy decay)
    - Annotator drift (IAA changes)
    """

    def detect_statistical_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        columns: list[str]
    ) -> dict[str, Any]:
        """
        Detect statistical drift using KS test and PSI.

        Returns:
            Dict with drift scores and significance
        """
        results = {}

        for column in columns:
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(
                baseline_df[column],
                current_df[column]
            )

            # Population Stability Index (PSI)
            psi = self._calculate_psi(
                baseline_df[column],
                current_df[column]
            )

            results[column] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "psi": float(psi),
                "drift_detected": psi > 0.2 or ks_pvalue < 0.05
            }

        return results

    def _calculate_psi(
        self,
        baseline: pd.Series,
        current: pd.Series,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index.

        PSI Interpretation:
        - <0.1: No significant change
        - 0.1-0.2: Moderate change
        - ≥0.2: Significant change (investigation needed)
        """
        # Create bins based on baseline
        baseline_counts, bin_edges = np.histogram(baseline, bins=bins)
        current_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions
        baseline_props = baseline_counts / len(baseline)
        current_props = current_counts / len(current)

        # Calculate PSI
        psi = 0
        for i in range(len(baseline_props)):
            if baseline_props[i] > 0 and current_props[i] > 0:
                psi += (current_props[i] - baseline_props[i]) * np.log(current_props[i] / baseline_props[i])

        return psi
```

**Success Metrics:**
- ✅ Drift detection latency <1s
- ✅ False positive rate <10%
- ✅ Catches significant drift (PSI>0.2) with 95% reliability

#### 5.2.3 Advanced Ensemble (STAPLE Algorithm) (7-10 days)

**Implementation:**

```python
# src/autolabeler/core/ensemble/advanced_ensemble.py

class STAPLEEnsemble:
    """
    STAPLE (Simultaneous Truth and Performance Level Estimation).

    Features:
    - Weighted consensus from multiple annotations
    - Annotator-specific sensitivity/specificity
    - Handles expert disagreements systematically
    - Particularly valuable for medical/high-stakes domains
    """

    def estimate_consensus(
        self,
        annotations: np.ndarray,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> tuple[np.ndarray, dict]:
        """
        Estimate consensus labels using STAPLE algorithm.

        Args:
            annotations: Shape (n_samples, n_annotators)
            max_iterations: Maximum EM iterations
            tolerance: Convergence tolerance

        Returns:
            (consensus_labels, annotator_stats)
        """
        pass
```

**Success Metrics:**
- ✅ STAPLE outperforms majority voting by ≥5%
- ✅ Provides reliable uncertainty estimates
- ✅ Computation time <10s for 1000 annotations

### 5.3 Phase 3 Timeline

| Week | Days | Activities | Deliverables |
|------|------|-----------|--------------|
| **Week 8-9** | 1-14 | Multi-agent architecture | Specialist agents, coordinator |
| **Week 10** | 1-7 | Drift detection | Statistical + embedding drift |
| **Week 11** | 1-7 | Advanced ensemble (STAPLE) | STAPLE implementation |
| **Week 12** | 1-7 | Integration & testing | All Phase 3 features working |

---

## 6. Testing Strategy

### 6.1 Testing Pyramid

```
                  ╱╲
                 ╱  ╲
                ╱ E2E╲           < 10%: Full system tests
               ╱──────╲
              ╱        ╲
             ╱Integration╲       < 20%: Service integration
            ╱────────────╲
           ╱              ╲
          ╱   Unit Tests   ╲    > 70%: Component tests
         ╱──────────────────╲
        ╱____________________╲
```

### 6.2 Unit Testing

**Coverage Target:** >80%

**Key Test Areas:**

1. **Validation Layer:**
   ```python
   # test_output_validator.py

   def test_structured_output_validation():
       """Test structured output validation with retries."""
       validator = StructuredOutputValidator(client)

       # Test valid output
       result = validator.validate_and_retry(
           prompt="Label this text",
           response_model=LabelResponse
       )
       assert isinstance(result, LabelResponse)
       assert result.label is not None
       assert 0 <= result.confidence <= 1

   def test_validation_failure_retry():
       """Test retry logic on validation failure."""
       # Mock LLM to return invalid output first, then valid
       pass
   ```

2. **Confidence Calibration:**
   ```python
   # test_confidence_calibrator.py

   def test_temperature_scaling():
       """Test temperature scaling calibration."""
       calibrator = ConfidenceCalibrator(method="temperature_scaling")

       # Fit on validation data
       calibrator.fit(confidence_scores, true_labels, predicted_labels)

       # Test calibration
       calibrated = calibrator.calibrate(raw_scores)

       # Verify ECE improvement
       ece_before = compute_ece(raw_scores, true_labels)
       ece_after = compute_ece(calibrated, true_labels)
       assert ece_after < ece_before
   ```

3. **Active Learning:**
   ```python
   # test_active_learning.py

   def test_margin_sampling():
       """Test margin sampling selects most uncertain examples."""
       al_service = ActiveLearningService("test_dataset", settings)

       # Create synthetic data with known uncertainties
       # Verify top-k selected have smallest margins
       pass

   def test_stopping_criteria():
       """Test stopping criteria triggers correctly."""
       # Simulate performance plateau
       # Verify should_stop returns True
       pass
   ```

### 6.3 Integration Testing

**Coverage Target:** >60%

**Key Integration Scenarios:**

1. **End-to-End Labeling with DSPy:**
   ```python
   # test_dspy_integration.py

   def test_dspy_optimization_pipeline():
       """Test full DSPy optimization workflow."""
       # 1. Initialize optimizer
       # 2. Run optimization on small dataset
       # 3. Verify improvement
       # 4. Use optimized module in labeling
       # 5. Verify results match expected
       pass
   ```

2. **Active Learning Loop:**
   ```python
   # test_active_learning_integration.py

   def test_active_learning_full_loop():
       """Test complete active learning workflow."""
       # 1. Initialize with small labeled set
       # 2. Train initial model
       # 3. Select samples (simulated)
       # 4. "Annotate" (ground truth lookup)
       # 5. Retrain
       # 6. Verify improvement
       # 7. Check stopping criteria
       pass
   ```

3. **Weak Supervision Pipeline:**
   ```python
   # test_weak_supervision_integration.py

   def test_weak_supervision_end_to_end():
       """Test weak supervision from LFs to aggregated labels."""
       # 1. Define labeling functions
       # 2. Apply to unlabeled data
       # 3. Aggregate with FlyingSquid
       # 4. Compare with ground truth (if available)
       # 5. Verify coverage and accuracy targets
       pass
   ```

### 6.4 Performance Testing

**Key Metrics:**

1. **Latency:**
   - Label single text: <2s (p95)
   - Batch labeling (100 items): <30s (p95)
   - RAG retrieval: <500ms (p95)
   - Quality metric computation: <5s (p95)

2. **Throughput:**
   - Sustained labeling rate: >50 items/minute
   - Concurrent requests: 10+ without degradation

3. **Resource Usage:**
   - Memory: <2GB for typical dataset (10k examples)
   - CPU: <80% utilization under load
   - Disk: <500MB for knowledge base

**Performance Test Suite:**

```python
# test_performance.py

import pytest
from time import time

@pytest.mark.performance
def test_labeling_latency():
    """Test labeling latency meets SLA."""
    labeler = AutoLabeler("test_dataset", settings)

    text = "Sample text for labeling"

    start = time()
    result = labeler.label_text(text)
    elapsed = time() - start

    assert elapsed < 2.0, f"Latency {elapsed:.2f}s exceeds 2s SLA"

@pytest.mark.performance
def test_batch_labeling_throughput():
    """Test batch labeling throughput."""
    labeler = AutoLabeler("test_dataset", settings)

    df = generate_test_dataframe(n=1000)

    start = time()
    results = labeler.label(df, "text")
    elapsed = time() - start

    throughput = len(df) / elapsed
    assert throughput > 50, f"Throughput {throughput:.1f} items/min < 50 items/min"
```

### 6.5 Validation Testing

**Validation Datasets:**

1. **Benchmark Datasets:**
   - IMDB (sentiment analysis)
   - AG News (topic classification)
   - CoNLL (NER)
   - Custom domain-specific datasets

2. **Validation Criteria:**
   - Accuracy vs baseline (must improve by ≥20%)
   - Cost reduction (target 40-70%)
   - Latency within SLA
   - Quality metrics (Krippendorff α ≥0.70)

**Validation Test Suite:**

```python
# test_validation.py

@pytest.mark.validation
def test_imdb_sentiment_accuracy():
    """Validate on IMDB sentiment dataset."""
    # Load IMDB test set
    # Run labeling
    # Compare with ground truth
    # Verify accuracy >85%
    pass

@pytest.mark.validation
def test_active_learning_efficiency():
    """Validate active learning reduces annotation needs."""
    # Run active learning
    # Compare with random sampling
    # Verify 40-70% reduction to reach target accuracy
    pass
```

---

## 7. Migration & Compatibility

### 7.1 Backward Compatibility

**Principle:** Zero breaking changes to existing public API.

**Strategy:**

1. **Feature Flags:**
   ```python
   # config.py

   class Settings(BaseSettings):
       # ... existing settings ...

       # New feature flags (default OFF for backward compat)
       enable_dspy_optimization: bool = False
       enable_advanced_rag: bool = False
       enable_weak_supervision: bool = False
       enable_active_learning: bool = False
       enable_versioning: bool = False
   ```

2. **Opt-In Enhancement:**
   ```python
   # Old code still works
   labeler = AutoLabeler("dataset", settings)
   results = labeler.label(df, "text")

   # New features opt-in
   settings.enable_dspy_optimization = True
   labeler = AutoLabeler("dataset", settings)

   # Optimize prompts (new method, doesn't break existing)
   labeler.optimize_prompts(train_df, val_df, "text", "label")

   # Use optimized prompts (automatic if available)
   results = labeler.label(df, "text")
   ```

3. **Graceful Degradation:**
   - If DSPy optimization fails → fallback to manual prompts
   - If advanced RAG unavailable → fallback to basic FAISS
   - If calibration fails → use raw confidence scores

### 7.2 Migration Path

**For Existing Users:**

1. **Phase 1 (Weeks 1-2):**
   - Update to new version
   - No code changes required
   - Optionally enable quality monitoring

2. **Phase 2 (Weeks 3-7):**
   - Optionally enable DSPy optimization
   - Optionally enable advanced RAG
   - Optionally enable active learning
   - Optionally enable weak supervision

3. **Phase 3 (Weeks 8-12):**
   - Enable drift detection
   - Enable advanced ensemble
   - Full production deployment

**Migration Script:**

```python
# migrate_v1_to_v2.py

def migrate_knowledge_base(old_path: Path, new_path: Path):
    """Migrate v1 knowledge base to v2 format."""
    # Load old format
    # Convert to new format
    # Preserve all data
    pass

def migrate_configs(old_config: dict) -> Settings:
    """Migrate v1 config to v2 Settings."""
    # Map old config keys to new Settings
    # Set appropriate defaults
    pass
```

### 7.3 Deprecation Timeline

**Principle:** Minimum 6 months notice for any deprecations.

**Currently No Planned Deprecations** - All v1 functionality preserved.

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **DSPy optimization doesn't improve accuracy** | Medium | High | Extensive testing on diverse datasets; fallback to manual prompts |
| **Active learning increases annotation costs** | Low | High | Thorough validation against random sampling; stopping criteria |
| **Weak supervision produces low-quality labels** | Medium | Medium | Quality validation; human review of samples |
| **Data versioning increases storage costs** | Low | Low | Configure remote storage; implement retention policies |
| **Performance degradation with new features** | Medium | High | Performance testing; profiling; optimization |
| **Breaking changes despite compatibility effort** | Low | High | Extensive backward compat testing; feature flags |

### 8.2 Implementation Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Timeline slippage** | Medium | Medium | Agile sprints; MVP approach; parallel workstreams |
| **Dependency conflicts** | Low | Low | Careful dependency management; virtual environments |
| **Integration complexity** | Medium | Medium | Modular architecture; clear interfaces; comprehensive testing |
| **Documentation lag** | High | Medium | Document as you go; examples with code; automated doc generation |

### 8.3 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Increased LLM API costs** | Medium | Medium | Cost tracking; budgets; cheaper model cascades |
| **Quality degradation in production** | Low | High | Continuous monitoring; drift detection; automated alerts |
| **User adoption challenges** | Medium | Medium | Comprehensive docs; migration guides; examples |
| **Scaling issues** | Low | High | Performance testing; horizontal scaling; batch optimization |

---

## 9. Success Metrics

### 9.1 Technical Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| **Accuracy Improvement** | - | +20-50% | DSPy optimization on validation set |
| **Annotation Cost Reduction** | 100% | 30-50% | Active learning vs random sampling |
| **Parsing Failure Rate** | 10% | <1% | Structured output validation |
| **Confidence Calibration (ECE)** | 0.15 | <0.05 | Expected calibration error |
| **RAG Retrieval Quality** | Baseline | +10% | Retrieval@k accuracy |
| **Weak Supervision Accuracy** | - | 70-80% | vs ground truth labels |
| **Test Coverage** | 60% | >80% | Pytest coverage report |

### 9.2 Performance Metrics

| Metric | Baseline | Target | SLA |
|--------|----------|--------|-----|
| **Single Label Latency (p95)** | 3s | <2s | 2s |
| **Batch Throughput** | 30/min | >50/min | 50/min |
| **RAG Retrieval Latency (p95)** | 1s | <500ms | 500ms |
| **Memory Usage (10k examples)** | 3GB | <2GB | 2GB |

### 9.3 Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Krippendorff's Alpha** | ≥0.70 | Multi-annotator agreement |
| **Cost Per Quality-Adjusted Annotation (CQAA)** | <$0.50 | Total cost / (annotations × quality) |
| **Drift Detection Precision** | >70% | False positive rate <30% |
| **Drift Detection Recall** | >80% | Catches 80%+ of true drifts |

### 9.4 Business Metrics

| Metric | Target | Impact |
|--------|--------|--------|
| **Time to Production** | <12 weeks | Faster feature delivery |
| **Cost Reduction** | 40-70% | Direct cost savings |
| **Annotation Velocity** | 10-100× | More data labeled |
| **Quality Consistency** | ±5% | Stable production performance |

---

## 10. Appendices

### 10.1 Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Prompt Optimization** | DSPy | State-of-the-art systematic optimization |
| **Structured Output** | Instructor + Pydantic | Type-safe validation with retries |
| **RAG Enhancement** | GraphRAG, RAPTOR | Improved retrieval for diverse queries |
| **Weak Supervision** | FlyingSquid | 170× faster than EM methods |
| **Active Learning** | modAL | sklearn-compatible, proven methods |
| **Data Versioning** | DVC | Git-like operations, cloud storage |
| **Quality Metrics** | krippendorff | Gold standard agreement metric |
| **Drift Detection** | scipy + custom | Statistical rigor |
| **Monitoring** | Plotly/Dash | Interactive dashboards |

### 10.2 Key Dependencies

```toml
# pyproject.toml additions

[project.dependencies]
# Existing dependencies...

# Phase 1
instructor = ">=1.0.0"  # Structured output
krippendorff = ">=0.6.0"  # Agreement metrics
plotly = ">=5.0.0"  # Dashboards
dash = ">=2.0.0"  # Web dashboards

# Phase 2
dspy-ai = ">=2.0.0"  # Prompt optimization
flyingsquid = ">=0.1.0"  # Weak supervision
modAL = ">=0.4.0"  # Active learning
dvc = ">=3.0.0"  # Data versioning
scikit-learn = ">=1.3.0"  # ML utilities

# Phase 3
networkx = ">=3.0"  # Graph operations for GraphRAG
sentence-transformers = ">=3.0.0"  # Embeddings
```

### 10.3 Reference Architecture Diagrams

See [Technical Architecture](#2-technical-architecture) section for detailed diagrams.

### 10.4 Related Documentation

- `/home/nick/python/autolabeler/advanced-labeling.md` - Research review
- `/home/nick/python/autolabeler/README.md` - Current system documentation
- `/home/nick/python/autolabeler/CLI_USAGE.md` - CLI reference
- `/home/nick/python/autolabeler/docs/` - Additional documentation

---

## Conclusion

This implementation roadmap provides a comprehensive, step-by-step plan to transform AutoLabeler from its current solid foundation into a production-grade, state-of-the-art annotation system. The three-phase approach balances quick wins with foundational improvements and advanced features, ensuring steady progress with manageable risk.

**Key Takeaways:**

1. **Phase 1 (Weeks 1-2):** Focus on high-impact, low-risk improvements that provide immediate value
2. **Phase 2 (Weeks 3-7):** Build core capabilities that enable systematic optimization and programmatic labeling
3. **Phase 3 (Weeks 8-12):** Add production-grade monitoring and advanced features for enterprise deployment

**Next Steps:**

1. Review and approve this roadmap
2. Set up development environment with all dependencies
3. Create feature branches for each phase
4. Begin Phase 1 implementation
5. Establish weekly sync meetings for progress tracking

**Expected Outcomes:**

- 40-70% cost reduction vs manual annotation
- 10-100× speed improvement in annotation velocity
- 20-50% accuracy improvement through DSPy optimization
- Production-grade quality monitoring and drift detection
- Comprehensive test coverage and documentation

This roadmap is a living document and should be updated as implementation progresses and new requirements emerge.

---

**Document Control:**
- **Author:** TESTER/INTEGRATION AGENT (Hive Mind Collective)
- **Review Status:** Draft v1.0
- **Approval Required:** Lead Developer, Product Manager
- **Last Updated:** 2025-10-07
