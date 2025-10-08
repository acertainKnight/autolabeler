# AutoLabeler API Specifications & Architecture
## Comprehensive API Reference and System Design

**Document Version:** 1.0
**Date:** 2025-10-07
**Project:** AutoLabeler v2 Enhancement Initiative
**Purpose:** Define API contracts, interfaces, and architecture for all new components

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Core Service APIs](#2-core-service-apis)
3. [Quality Monitoring APIs](#3-quality-monitoring-apis)
4. [Active Learning APIs](#4-active-learning-apis)
5. [Weak Supervision APIs](#5-weak-supervision-apis)
6. [DSPy Integration APIs](#6-dspy-integration-apis)
7. [Data Versioning APIs](#7-data-versioning-apis)
8. [Configuration Schemas](#8-configuration-schemas)
9. [Data Models](#9-data-models)
10. [Error Handling](#10-error-handling)

---

## 1. Architecture Overview

### 1.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AutoLabeler System Architecture                        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                  Client Layer                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Python API          CLI Interface          Web UI (Future)                     │
│  autolabeler.py      cli.py                                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Main Orchestration Layer                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│  AutoLabeler (Main Interface)                                                   │
│  - Configuration Management                                                      │
│  - Service Orchestration                                                         │
│  - Workflow Coordination                                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
│   Labeling Services    │  │   Quality Services     │  │  Learning Services     │
├────────────────────────┤  ├────────────────────────┤  ├────────────────────────┤
│ • LabelingService      │  │ • QualityMonitor       │  │ • ActiveLearning       │
│ • EnsembleService      │  │ • ConfidenceCalibrator │  │ • WeakSupervision      │
│ • SyntheticService     │  │ • OutputValidator      │  │ • DSPyOptimizer        │
│ • RuleService          │  │ • DriftDetector        │  │                        │
└────────────────────────┘  └────────────────────────┘  └────────────────────────┘
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Knowledge & Storage Layer                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│  • KnowledgeStore (FAISS/ChromaDB)    • PromptManager (DSPy)                   │
│  • DVCIntegration (Versioning)        • MetricsStore (Time-series)             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            External Services Layer                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│  • LLM Providers (OpenAI, Anthropic, OpenRouter)                                │
│  • Vector Stores (FAISS, ChromaDB, Pinecone)                                   │
│  • Cloud Storage (S3, GCS, Azure)                                              │
│  • Monitoring (MLflow, W&B, Custom)                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Interaction Flow

```
User Request → AutoLabeler → LabelingService → LLM Provider
                    ↓              ↓
              QualityMonitor   KnowledgeStore
                    ↓              ↓
              MetricsStore    VectorStore (FAISS)
```

### 1.3 Data Flow Diagram

```
┌──────────────┐
│ Input Data   │
│ (unlabeled)  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Pre-processing & Validation         │
│  - Schema validation                 │
│  - Duplicate detection               │
│  - Text normalization                │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Knowledge Retrieval (RAG)           │
│  - Query embedding                   │
│  - Similar example retrieval         │
│  - Context preparation               │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Prompt Generation                   │
│  - Template rendering                │
│  - Example injection                 │
│  - DSPy optimization (if enabled)    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  LLM Inference                       │
│  - Structured output generation      │
│  - Confidence estimation             │
│  - Reasoning extraction              │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Post-processing & Validation        │
│  - Output validation (Instructor)    │
│  - Confidence calibration            │
│  - Quality checks                    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Knowledge Base Update               │
│  - High-confidence examples          │
│  - Embedding generation              │
│  - Metadata tracking                 │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────┐
│ Output Data  │
│ (labeled)    │
└──────────────┘
```

---

## 2. Core Service APIs

### 2.1 LabelingService API

#### 2.1.1 Class Definition

```python
class LabelingService(ConfigurableComponent, ProgressTracker, BatchProcessor):
    """
    Core service for all text labeling operations.

    This service orchestrates knowledge retrieval (RAG), prompt generation,
    LLM interaction, and result processing.
    """

    def __init__(
        self,
        dataset_name: str,
        settings: Settings,
        config: LabelingConfig | None = None,
    ) -> None:
        """
        Initialize the labeling service.

        Args:
            dataset_name: Unique identifier for the dataset
            settings: Global settings instance
            config: Labeling-specific configuration

        Raises:
            ValueError: If dataset_name is invalid
            FileNotFoundError: If template_path doesn't exist
        """
```

#### 2.1.2 Public Methods

```python
def label_text(
    self,
    text: str,
    config: LabelingConfig | None = None,
    template_path: Path | None = None,
    ruleset: dict[str, Any] | None = None,
) -> LabelResponse:
    """
    Label a single text with the configured settings.

    Args:
        text: Text to label
        config: Override default labeling config
        template_path: Override default template
        ruleset: Optional ruleset for rule-enhanced labeling

    Returns:
        LabelResponse with label, confidence, and reasoning

    Raises:
        ValidationError: If output validation fails
        LLMError: If LLM call fails after retries

    Example:
        >>> service = LabelingService("sentiment", settings)
        >>> result = service.label_text("This is great!")
        >>> print(result.label, result.confidence)
        'positive' 0.92
    """

async def alabel_text(
    self,
    text: str,
    config: LabelingConfig | None = None,
    template_path: Path | None = None,
    ruleset: dict[str, Any] | None = None,
) -> LabelResponse:
    """
    Asynchronously label a single text.

    Same interface as label_text but runs asynchronously for
    high-throughput applications.

    Args:
        text: Text to label
        config: Override default labeling config
        template_path: Override default template
        ruleset: Optional ruleset for rule-enhanced labeling

    Returns:
        LabelResponse with label, confidence, and reasoning

    Example:
        >>> import asyncio
        >>> result = await service.alabel_text("This is great!")
    """

def label_dataframe(
    self,
    df: pd.DataFrame,
    text_column: str,
    label_column: str = "predicted_label",
    config: LabelingConfig | None = None,
    batch_config: BatchConfig | None = None,
    template_path: Path | None = None,
    ruleset: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Label an entire dataset with batch processing support.

    Args:
        df: DataFrame containing texts to label
        text_column: Name of column containing text
        label_column: Name of column for predicted labels
        config: Labeling configuration
        batch_config: Batch processing configuration
        template_path: Override default template
        ruleset: Optional ruleset

    Returns:
        DataFrame with added label and confidence columns

    Raises:
        KeyError: If text_column doesn't exist
        ValueError: If dataframe is empty

    Example:
        >>> df = pd.DataFrame({"text": ["text1", "text2"]})
        >>> results = service.label_dataframe(df, "text")
        >>> print(results[["text", "predicted_label", "predicted_label_confidence"]])
    """

def get_stats(self) -> dict[str, Any]:
    """
    Get statistics about the labeling service.

    Returns:
        Dictionary containing:
        - dataset_name: Name of the dataset
        - knowledge_base_stats: KB statistics (size, sources, etc.)
        - prompt_analytics: Prompt usage and performance
        - progress_info: Current progress information

    Example:
        >>> stats = service.get_stats()
        >>> print(f"KB size: {stats['knowledge_base_stats']['total_examples']}")
    """

def analyze_rag_diversity(self) -> dict[str, Any]:
    """
    Analyze the diversity of RAG examples being retrieved.

    Returns:
        Dictionary containing:
        - total_queries: Number of queries analyzed
        - unique_examples_retrieved: Count of unique examples
        - diversity_ratio: Ratio of unique to total retrievals
        - repeated_example_percentage: % of repeated examples
        - most_common_examples: Top frequently retrieved examples

    Example:
        >>> analysis = service.analyze_rag_diversity()
        >>> if analysis['diversity_ratio'] < 0.5:
        ...     print("Warning: Low RAG diversity")
    """

def check_rag_issues(self) -> dict[str, Any]:
    """
    Check for common RAG issues that could cause problems.

    Returns:
        Dictionary containing:
        - issues_found: Count of issues detected
        - issues: List of issue descriptions
        - recommendations: List of recommended fixes
        - knowledge_base_stats: Current KB statistics

    Example:
        >>> issues = service.check_rag_issues()
        >>> for issue in issues['issues']:
        ...     print(f"Issue: {issue}")
        >>> for rec in issues['recommendations']:
        ...     print(f"Recommendation: {rec}")
    """
```

---

## 3. Quality Monitoring APIs

### 3.1 QualityMonitor API

#### 3.1.1 Class Definition

```python
class QualityMonitor(ConfigurableComponent):
    """
    Comprehensive quality monitoring with real-time metrics.

    Tracks:
    - Inter-annotator agreement (Krippendorff's alpha)
    - Confidence calibration metrics
    - Cost per quality-adjusted annotation
    - Performance trends over time
    - Anomaly detection
    """

    def __init__(
        self,
        dataset_name: str,
        settings: Settings,
        monitoring_config: MonitoringConfig | None = None
    ) -> None:
        """
        Initialize quality monitor.

        Args:
            dataset_name: Dataset identifier
            settings: Global settings
            monitoring_config: Monitoring-specific configuration
        """
```

#### 3.1.2 Public Methods

```python
def calculate_krippendorff_alpha(
    self,
    df: pd.DataFrame,
    annotator_columns: list[str],
    value_domain: list[str] | None = None,
    level_of_measurement: str = "nominal"
) -> float:
    """
    Calculate Krippendorff's alpha for multi-annotator agreement.

    Args:
        df: DataFrame with annotations
        annotator_columns: List of column names for each annotator
        value_domain: Optional domain of possible values
        level_of_measurement: Type of data ("nominal", "ordinal", "interval", "ratio")

    Returns:
        Alpha value between -1 and 1
        - α ≥ 0.80: Reliable
        - 0.67 ≤ α < 0.80: Tentative
        - α < 0.67: Unreliable

    Raises:
        ValueError: If fewer than 2 annotators provided
        ValueError: If level_of_measurement is invalid

    Example:
        >>> df = pd.DataFrame({
        ...     "annotator_1": ["pos", "neg", "pos"],
        ...     "annotator_2": ["pos", "neg", "neu"],
        ...     "annotator_3": ["pos", "neg", "pos"]
        ... })
        >>> alpha = monitor.calculate_krippendorff_alpha(
        ...     df,
        ...     ["annotator_1", "annotator_2", "annotator_3"]
        ... )
        >>> print(f"Agreement: {alpha:.3f}")
    """

def compute_cqaa(
    self,
    df: pd.DataFrame,
    cost_column: str,
    quality_score_column: str
) -> float:
    """
    Compute Cost Per Quality-Adjusted Annotation.

    Formula: CQAA = Total Cost / (Annotations × Average Quality Score)

    Args:
        df: DataFrame with cost and quality data
        cost_column: Column containing per-annotation cost
        quality_score_column: Column containing quality scores [0, 1]

    Returns:
        Cost per quality-adjusted annotation in currency units

    Example:
        >>> cqaa = monitor.compute_cqaa(
        ...     results_df,
        ...     cost_column="llm_cost",
        ...     quality_score_column="confidence"
        ... )
        >>> print(f"CQAA: ${cqaa:.4f}")
    """

def detect_anomalies(
    self,
    df: pd.DataFrame,
    metric_columns: list[str],
    window_size: int = 100,
    n_sigma: float = 3.0
) -> list[dict[str, Any]]:
    """
    Detect statistical anomalies in annotation stream.

    Uses z-score outlier detection within sliding windows.

    Args:
        df: DataFrame with metrics
        metric_columns: Columns to monitor for anomalies
        window_size: Size of sliding window
        n_sigma: Number of standard deviations for outlier threshold

    Returns:
        List of detected anomalies with details:
        - index: Row index of anomaly
        - metric: Metric name
        - value: Anomalous value
        - z_score: Z-score
        - recommendation: Suggested action

    Example:
        >>> anomalies = monitor.detect_anomalies(
        ...     results_df,
        ...     metric_columns=["confidence", "latency_ms"]
        ... )
        >>> for anomaly in anomalies:
        ...     print(f"Anomaly at {anomaly['index']}: {anomaly['metric']} = {anomaly['value']}")
    """

def generate_dashboard(
    self,
    df: pd.DataFrame,
    output_path: Path,
    format: str = "html",
    include_sections: list[str] | None = None
) -> Path:
    """
    Generate comprehensive quality dashboard.

    Args:
        df: DataFrame with annotation results and metrics
        output_path: Path to save dashboard
        format: Output format ("html", "pdf", "json")
        include_sections: Optional list of sections to include

    Returns:
        Path to generated dashboard file

    Available sections:
    - executive_summary: Key metrics overview
    - agreement_analysis: IAA metrics and confusion matrix
    - confidence_calibration: Reliability diagram and calibration metrics
    - cost_analysis: CQAA trends and cost breakdown
    - performance_trends: Accuracy, throughput over time
    - quality_issues: Detected issues and recommendations

    Example:
        >>> dashboard_path = monitor.generate_dashboard(
        ...     results_df,
        ...     output_path=Path("reports/quality_dashboard.html"),
        ...     include_sections=["executive_summary", "cost_analysis"]
        ... )
        >>> print(f"Dashboard saved to {dashboard_path}")
    """

def track_metric(
    self,
    metric_name: str,
    value: float,
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None
) -> None:
    """
    Track a metric value over time.

    Args:
        metric_name: Name of the metric (e.g., "accuracy", "latency")
        value: Metric value
        timestamp: Optional timestamp (defaults to now)
        metadata: Optional additional metadata

    Example:
        >>> monitor.track_metric("accuracy", 0.85)
        >>> monitor.track_metric("latency_ms", 245.3, metadata={"model": "gpt-4"})
    """

def get_metric_history(
    self,
    metric_name: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None
) -> pd.DataFrame:
    """
    Retrieve historical metric values.

    Args:
        metric_name: Name of the metric
        start_time: Optional start time filter
        end_time: Optional end time filter

    Returns:
        DataFrame with columns: timestamp, value, metadata

    Example:
        >>> history = monitor.get_metric_history("accuracy")
        >>> print(history.describe())
    """
```

### 3.2 ConfidenceCalibrator API

#### 3.2.1 Class Definition

```python
class ConfidenceCalibrator:
    """
    Confidence score calibration using multiple methods.

    Supported Methods:
    - temperature_scaling: Single-parameter calibration
    - platt_scaling: Logistic regression calibration
    - isotonic_regression: Non-parametric calibration
    - beta_calibration: Advanced probabilistic calibration
    """

    def __init__(
        self,
        method: CalibrationMethod = "temperature_scaling"
    ) -> None:
        """
        Initialize confidence calibrator.

        Args:
            method: Calibration method to use

        Raises:
            ValueError: If method is not supported
        """
```

#### 3.2.2 Public Methods

```python
def fit(
    self,
    confidence_scores: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray
) -> None:
    """
    Fit calibration model on validation data.

    Args:
        confidence_scores: Raw confidence scores from model
        true_labels: Ground truth labels
        predicted_labels: Predicted labels

    Raises:
        ValueError: If arrays have different lengths
        ValueError: If confidence scores not in [0, 1]

    Example:
        >>> calibrator = ConfidenceCalibrator("temperature_scaling")
        >>> calibrator.fit(val_confidences, val_true_labels, val_pred_labels)
    """

def calibrate(
    self,
    confidence_scores: np.ndarray
) -> np.ndarray:
    """
    Apply calibration to raw confidence scores.

    Args:
        confidence_scores: Raw confidence scores

    Returns:
        Calibrated confidence scores

    Raises:
        RuntimeError: If calibrator not fitted

    Example:
        >>> calibrated = calibrator.calibrate(test_confidences)
    """

def evaluate_calibration(
    self,
    confidence_scores: np.ndarray,
    true_labels: np.ndarray
) -> dict[str, float]:
    """
    Compute calibration metrics.

    Args:
        confidence_scores: Confidence scores (raw or calibrated)
        true_labels: Ground truth labels

    Returns:
        Dictionary with metrics:
        - ece: Expected Calibration Error
        - mce: Maximum Calibration Error
        - brier_score: Brier score
        - log_loss: Logarithmic loss
        - reliability_diagram_data: Data for plotting

    Example:
        >>> metrics = calibrator.evaluate_calibration(
        ...     calibrated_scores,
        ...     true_labels
        ... )
        >>> print(f"ECE: {metrics['ece']:.4f}")
    """

def save(self, path: Path) -> None:
    """
    Save fitted calibrator to disk.

    Args:
        path: Path to save calibrator

    Example:
        >>> calibrator.save(Path("models/calibrator_v1.pkl"))
    """

@classmethod
def load(cls, path: Path) -> "ConfidenceCalibrator":
    """
    Load fitted calibrator from disk.

    Args:
        path: Path to saved calibrator

    Returns:
        Loaded calibrator instance

    Example:
        >>> calibrator = ConfidenceCalibrator.load(Path("models/calibrator_v1.pkl"))
    """
```

---

## 4. Active Learning APIs

### 4.1 ActiveLearningService API

#### 4.1.1 Class Definition

```python
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
    ) -> None:
        """
        Initialize active learning service.

        Args:
            dataset_name: Dataset identifier
            settings: Global settings
            initial_strategy: Initial sampling strategy
        """
```

#### 4.1.2 Public Methods

```python
def select_samples(
    self,
    unlabeled_df: pd.DataFrame,
    model: Any,
    n_samples: int = 100,
    embeddings: np.ndarray | None = None
) -> pd.DataFrame:
    """
    Select most informative samples for annotation.

    Args:
        unlabeled_df: Pool of unlabeled examples
        model: Current model for uncertainty estimation (must have predict_proba)
        n_samples: Number of samples to select
        embeddings: Pre-computed embeddings (optional, for diversity sampling)

    Returns:
        DataFrame with selected samples

    Raises:
        ValueError: If n_samples > len(unlabeled_df)
        AttributeError: If model doesn't have predict_proba method

    Example:
        >>> selected = al_service.select_samples(
        ...     unlabeled_df,
        ...     trained_model,
        ...     n_samples=100
        ... )
        >>> print(f"Selected {len(selected)} samples for annotation")
    """

def should_stop(
    self,
    current_performance: float,
    validation_df: pd.DataFrame
) -> tuple[bool, str]:
    """
    Determine if active learning should stop.

    Stopping Criteria:
    1. Performance plateau: <1% improvement for N consecutive iterations
    2. Uncertainty threshold: Average uncertainty below threshold
    3. Budget exhausted: Reached maximum annotation budget

    Args:
        current_performance: Current model performance (e.g., accuracy)
        validation_df: Validation set for uncertainty estimation

    Returns:
        Tuple of (should_stop: bool, reason: str)

    Example:
        >>> should_stop, reason = al_service.should_stop(0.85, val_df)
        >>> if should_stop:
        ...     print(f"Stopping: {reason}")
    """

def get_statistics(self) -> dict[str, Any]:
    """
    Get active learning statistics.

    Returns:
        Dictionary with:
        - iteration: Current iteration number
        - strategy: Current sampling strategy
        - performance_history: List of performance values
        - improvement_per_iteration: List of improvements
        - total_samples_selected: Total samples annotated

    Example:
        >>> stats = al_service.get_statistics()
        >>> print(f"Total annotations: {stats['total_samples_selected']}")
    """

def reset(self) -> None:
    """
    Reset active learning state.

    Clears performance history and resets iteration counter.
    Useful for starting a new active learning cycle.

    Example:
        >>> al_service.reset()
    """
```

---

## 5. Weak Supervision APIs

### 5.1 WeakSupervisionService API

#### 5.1.1 Class Definition

```python
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
    ) -> None:
        """
        Initialize weak supervision service.

        Args:
            dataset_name: Dataset identifier
            settings: Global settings
        """
```

#### 5.1.2 Public Methods

```python
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
        description: Human-readable description
        category: Category ("heuristic", "gazetteer", "model", "llm")

    Returns:
        LF ID

    Raises:
        ValueError: If name already exists
        TypeError: If function doesn't have correct signature

    Example:
        >>> def positive_keywords(text: str) -> int:
        ...     if any(word in text.lower() for word in ["great", "excellent"]):
        ...         return 1  # positive
        ...     return -1  # abstain
        >>>
        >>> lf_id = ws_service.add_labeling_function(
        ...     "positive_keywords",
        ...     positive_keywords,
        ...     "Detects positive sentiment keywords"
        ... )
    """

def apply_labeling_functions(
    self,
    df: pd.DataFrame,
    text_column: str
) -> np.ndarray:
    """
    Apply all labeling functions to dataset.

    Args:
        df: DataFrame with texts
        text_column: Name of text column

    Returns:
        Label matrix of shape (n_samples, n_lfs)
        where each entry is the label from that LF
        (-1 indicates abstention)

    Raises:
        KeyError: If text_column doesn't exist

    Example:
        >>> label_matrix = ws_service.apply_labeling_functions(
        ...     unlabeled_df,
        ...     "text"
        ... )
        >>> print(f"Coverage: {(label_matrix != -1).mean():.2%}")
    """

def aggregate_labels(
    self,
    label_matrix: np.ndarray | None = None,
    method: str = "flyingsquid"
) -> np.ndarray:
    """
    Aggregate noisy labels using FlyingSquid or other methods.

    Args:
        label_matrix: Optional label matrix (uses cached if None)
        method: Aggregation method ("flyingsquid", "majority_vote", "snorkel")

    Returns:
        Aggregated probabilistic labels of shape (n_samples, n_classes)

    Raises:
        ValueError: If label_matrix not provided and not cached

    Example:
        >>> probabilistic_labels = ws_service.aggregate_labels()
        >>> hard_labels = np.argmax(probabilistic_labels, axis=1)
    """

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
    3. Generate LF code using LLM
    4. Validate and test LFs
    5. Add to system

    Args:
        labeled_df: DataFrame with labeled examples
        text_column: Text column name
        label_column: Label column name
        num_functions: Number of LFs to generate

    Returns:
        List of generated LF names

    Example:
        >>> lf_names = ws_service.generate_labeling_functions_with_llm(
        ...     labeled_df,
        ...     "text",
        ...     "label",
        ...     num_functions=5
        ... )
        >>> print(f"Generated {len(lf_names)} labeling functions")
    """

def get_lf_report(self) -> pd.DataFrame:
    """
    Generate comprehensive report on labeling functions.

    Returns:
        DataFrame with columns:
        - name: LF name
        - category: LF category
        - description: LF description
        - coverage: Fraction of samples labeled
        - accuracy: Estimated accuracy
        - conflicts: Number of conflicts with other LFs
        - samples_labeled: Number of samples labeled

    Example:
        >>> report = ws_service.get_lf_report()
        >>> print(report.sort_values("accuracy", ascending=False))
    """

def remove_labeling_function(self, name: str) -> None:
    """
    Remove a labeling function from the system.

    Args:
        name: LF name to remove

    Raises:
        KeyError: If LF doesn't exist

    Example:
        >>> ws_service.remove_labeling_function("low_coverage_lf")
    """
```

---

## 6. DSPy Integration APIs

### 6.1 DSPyOptimizer API

#### 6.1.1 Class Definition

```python
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
    ) -> None:
        """
        Initialize DSPy optimizer.

        Args:
            dataset_name: Dataset identifier
            task_description: Natural language task description
            settings: Global settings
        """
```

#### 6.1.2 Public Methods

```python
def define_signature(
    self,
    input_fields: dict[str, str],
    output_fields: dict[str, str]
) -> dspy.Signature:
    """
    Define DSPy signature from field descriptions.

    Args:
        input_fields: Dict of {field_name: description}
        output_fields: Dict of {field_name: description}

    Returns:
        DSPy Signature class

    Example:
        >>> signature = optimizer.define_signature(
        ...     input_fields={"text": "Text to classify"},
        ...     output_fields={
        ...         "label": "Predicted label",
        ...         "confidence": "Confidence score 0-1"
        ...     }
        ... )
    """

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

    Args:
        training_examples: Training examples for bootstrapping
        validation_examples: Validation examples for evaluation
        metric_fn: Metric function (example, prediction) -> score
        max_cost: Maximum cost in dollars
        num_candidates: Number of candidate prompts to evaluate

    Returns:
        DSPyOptimizationResult with:
        - optimized_module: Optimized DSPy module
        - metrics: Performance metrics
        - cost: Total optimization cost
        - prompt_versions: All generated prompt versions

    Raises:
        BudgetExceededError: If cost exceeds max_cost
        OptimizationFailedError: If optimization fails

    Example:
        >>> result = optimizer.optimize(
        ...     train_examples,
        ...     val_examples,
        ...     metric_fn=lambda ex, pred: 1.0 if pred.label == ex.label else 0.0,
        ...     max_cost=5.0
        ... )
        >>> print(f"Accuracy: {result.metrics['accuracy']:.2%}")
        >>> print(f"Cost: ${result.cost:.2f}")
    """

def save_optimized_module(
    self,
    module: dspy.Module,
    version: str
) -> Path:
    """
    Save optimized module for later use.

    Args:
        module: Optimized DSPy module
        version: Version identifier

    Returns:
        Path to saved module

    Example:
        >>> path = optimizer.save_optimized_module(result.optimized_module, "v1")
    """

def load_optimized_module(
    self,
    version: str
) -> dspy.Module:
    """
    Load previously optimized module.

    Args:
        version: Version identifier

    Returns:
        Loaded DSPy module

    Raises:
        FileNotFoundError: If version doesn't exist

    Example:
        >>> module = optimizer.load_optimized_module("v1")
    """
```

---

## 7. Data Versioning APIs

### 7.1 DVCIntegration API

#### 7.1.1 Class Definition

```python
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
    ) -> None:
        """
        Initialize DVC integration.

        Args:
            dataset_name: Dataset identifier
            repo_path: Path to DVC repository
            remote: Optional remote storage URL
        """
```

#### 7.1.2 Public Methods

```python
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

    Raises:
        FileNotFoundError: If dataset_path doesn't exist

    Example:
        >>> commit_hash = dvc.add_dataset(
        ...     Path("data/labeled_v1.parquet"),
        ...     message="Initial labeled dataset"
        ... )
    """

def checkout_version(
    self,
    version: str
) -> None:
    """
    Checkout specific version of dataset.

    Args:
        version: Git commit hash or tag

    Raises:
        ValueError: If version doesn't exist

    Example:
        >>> dvc.checkout_version("abc123")
    """

def diff_versions(
    self,
    version_a: str,
    version_b: str,
    dataset_path: Path
) -> dict[str, Any]:
    """
    Compare two versions of dataset.

    Args:
        version_a: First version (commit hash or tag)
        version_b: Second version
        dataset_path: Path to dataset file

    Returns:
        Dict with diff statistics:
        - rows_added: Number of rows added
        - rows_removed: Number of rows removed
        - columns_added: List of new columns
        - columns_removed: List of removed columns
        - label_distribution_a: Label distribution in version A
        - label_distribution_b: Label distribution in version B

    Example:
        >>> diff = dvc.diff_versions("v1", "v2", Path("data/labeled.parquet"))
        >>> print(f"Rows added: {diff['rows_added']}")
    """

def list_versions(self) -> list[dict[str, Any]]:
    """
    List all dataset versions.

    Returns:
        List of versions with metadata:
        - hash: Commit hash
        - author: Committer name
        - date: Commit date
        - message: Commit message

    Example:
        >>> versions = dvc.list_versions()
        >>> for v in versions:
        ...     print(f"{v['date']} - {v['message']}")
    """

def push_to_remote(self) -> None:
    """
    Push dataset to remote storage.

    Raises:
        RemoteNotConfiguredError: If no remote configured
        PushFailedError: If push fails

    Example:
        >>> dvc.push_to_remote()
    """

def pull_from_remote(self) -> None:
    """
    Pull dataset from remote storage.

    Raises:
        RemoteNotConfiguredError: If no remote configured
        PullFailedError: If pull fails

    Example:
        >>> dvc.pull_from_remote()
    """
```

---

## 8. Configuration Schemas

### 8.1 LabelingConfig

```python
class LabelingConfig(BaseModel):
    """Configuration for labeling operations."""

    use_rag: bool = Field(
        True,
        description="Whether to use RAG for example retrieval"
    )

    k_examples: int | None = Field(
        None,
        description="Number of examples to retrieve (None = use default)"
    )

    prefer_human_examples: bool = Field(
        True,
        description="Prefer human-labeled examples over model-generated"
    )

    confidence_threshold: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for knowledge base updates"
    )

    save_to_knowledge_base: bool = Field(
        True,
        description="Save high-confidence predictions to KB"
    )

    model_name: str | None = Field(
        None,
        description="Model name/identifier (None = use default)"
    )

    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )

    max_tokens: int | None = Field(
        None,
        gt=0,
        description="Maximum tokens in response"
    )

    max_examples: int = Field(
        5,
        gt=0,
        description="Maximum examples to retrieve"
    )

    metadata_columns: list[str] | None = Field(
        None,
        description="Additional metadata columns to include"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "use_rag": True,
                "k_examples": 5,
                "prefer_human_examples": True,
                "confidence_threshold": 0.8,
                "save_to_knowledge_base": True,
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.1
            }
        }
```

### 8.2 MonitoringConfig

```python
class MonitoringConfig(BaseModel):
    """Configuration for quality monitoring."""

    enable_krippendorff_alpha: bool = Field(
        True,
        description="Calculate Krippendorff's alpha for agreement"
    )

    enable_confidence_calibration: bool = Field(
        True,
        description="Enable confidence calibration"
    )

    enable_anomaly_detection: bool = Field(
        True,
        description="Enable anomaly detection"
    )

    anomaly_window_size: int = Field(
        100,
        gt=0,
        description="Window size for anomaly detection"
    )

    anomaly_n_sigma: float = Field(
        3.0,
        gt=0,
        description="Number of standard deviations for outlier threshold"
    )

    dashboard_update_interval: int = Field(
        300,
        gt=0,
        description="Dashboard update interval in seconds"
    )

    track_metrics: list[str] = Field(
        default_factory=lambda: ["accuracy", "latency", "cost"],
        description="Metrics to track over time"
    )
```

---

## 9. Data Models

### 9.1 LabelResponse

```python
class LabelResponse(BaseModel):
    """Response from labeling operation."""

    label: str = Field(
        description="Predicted label"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score [0, 1]"
    )

    reasoning: str | None = Field(
        None,
        description="Explanation for the label"
    )

    metadata: dict[str, Any] | None = Field(
        None,
        description="Additional metadata"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "label": "positive",
            "confidence": 0.92,
            "reasoning": "Contains positive sentiment keywords like 'great' and 'excellent'",
            "metadata": {"model": "gpt-3.5-turbo", "latency_ms": 245}
        }
    })
```

### 9.2 DSPyOptimizationResult

```python
class DSPyOptimizationResult(BaseModel):
    """Result from DSPy optimization."""

    optimized_module: Any = Field(
        description="Optimized DSPy module"
    )

    metrics: dict[str, float] = Field(
        description="Performance metrics on validation set"
    )

    cost: float = Field(
        description="Total optimization cost in dollars"
    )

    improvement: float = Field(
        description="Improvement over baseline (fractional)"
    )

    prompt_versions: list[dict[str, Any]] = Field(
        description="All generated prompt versions"
    )

    optimization_time: float = Field(
        description="Optimization time in seconds"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "metrics": {"accuracy": 0.85, "f1": 0.83},
            "cost": 2.45,
            "improvement": 0.23,
            "optimization_time": 1200.0
        }
    })
```

---

## 10. Error Handling

### 10.1 Exception Hierarchy

```python
class AutoLabelerError(Exception):
    """Base exception for AutoLabeler."""
    pass

class ValidationError(AutoLabelerError):
    """Raised when output validation fails."""
    pass

class LLMError(AutoLabelerError):
    """Raised when LLM call fails."""
    pass

class ConfigurationError(AutoLabelerError):
    """Raised when configuration is invalid."""
    pass

class StorageError(AutoLabelerError):
    """Raised when storage operations fail."""
    pass

class OptimizationError(AutoLabelerError):
    """Raised when DSPy optimization fails."""
    pass

class BudgetExceededError(OptimizationError):
    """Raised when optimization exceeds budget."""
    pass
```

### 10.2 Error Handling Patterns

```python
# Recommended error handling pattern

try:
    result = labeling_service.label_text(text)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    # Handle validation failure (e.g., retry with different config)
except LLMError as e:
    logger.error(f"LLM call failed: {e}")
    # Handle LLM failure (e.g., use fallback model)
except AutoLabelerError as e:
    logger.error(f"AutoLabeler error: {e}")
    # Handle other errors
```

---

## Conclusion

This API specification provides comprehensive documentation of all public interfaces for AutoLabeler enhancements. All APIs follow consistent patterns:

- **Type-safe interfaces** using Pydantic models
- **Clear documentation** with examples
- **Explicit error handling** with custom exceptions
- **Backward compatibility** through optional parameters
- **Configuration-driven** behavior

**Key Design Principles:**
- Fail fast with clear error messages
- Provide sensible defaults
- Enable progressive disclosure of complexity
- Support both synchronous and asynchronous usage
- Maintain clean separation of concerns

---

**Document Control:**
- **Author:** TESTER/INTEGRATION AGENT
- **Version:** 1.0
- **Last Updated:** 2025-10-07
