"""
Data schemas and models for Quality Control System.

This module defines all Pydantic models, database schemas, and data structures
used by the quality control framework.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# ENUMERATIONS
# ============================================================================

class AnnotatorType(str, Enum):
    """Type of annotator that produced the label."""
    HUMAN = "human"
    LLM = "llm"
    ENSEMBLE = "ensemble"
    RULE_BASED = "rule_based"
    SYNTHETIC = "synthetic"


class ReviewStatus(str, Enum):
    """Status of annotation in review workflow."""
    AUTO_ACCEPT = "auto_accept"
    PENDING_REVIEW = "pending_review"
    EXPERT_NEEDED = "expert_needed"
    HUMAN_VALIDATED = "human_validated"
    REJECTED = "rejected"


class DriftInterpretation(str, Enum):
    """Interpretation of drift severity."""
    STABLE = "stable"
    SLIGHTLY_UNSTABLE = "slightly_unstable"
    UNSTABLE = "unstable"
    CRITICAL = "critical"


class AlertSeverity(str, Enum):
    """Severity level for quality alerts."""
    INFO = "info"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Type of quality alert."""
    ACCURACY_DROP = "accuracy_drop"
    IAA_LOW = "iaa_low"
    CALIBRATION_POOR = "calibration_poor"
    DRIFT_DETECTED = "drift_detected"
    COST_SPIKE = "cost_spike"
    REVIEW_RATE_HIGH = "review_rate_high"


# ============================================================================
# CONFIGURATION MODELS
# ============================================================================

class IAAConfig(BaseModel):
    """Configuration for inter-annotator agreement calculation."""

    model_config = ConfigDict(extra="forbid")

    # Agreement metric
    metric: str = Field(
        default="nominal",
        description="Distance metric: 'nominal', 'ordinal', 'interval', 'ratio'"
    )

    # Routing thresholds
    auto_accept_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="IAA threshold for auto-accepting annotations"
    )
    review_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="IAA threshold requiring human review"
    )
    expert_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="IAA threshold requiring expert review"
    )

    # Statistical parameters
    bootstrap_samples: int = Field(
        default=1000,
        gt=0,
        description="Number of bootstrap samples for confidence interval"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Confidence level for interval estimation"
    )

    # Annotator requirements
    min_annotators: int = Field(
        default=2,
        ge=2,
        description="Minimum annotators required for IAA calculation"
    )
    max_annotators: int | None = Field(
        default=None,
        description="Maximum annotators to consider (None = no limit)"
    )

    # Tracking options
    save_provenance: bool = Field(
        default=True,
        description="Save detailed provenance information"
    )
    track_temporal: bool = Field(
        default=True,
        description="Track IAA trends over time"
    )
    window_size: int = Field(
        default=100,
        gt=0,
        description="Window size for temporal tracking"
    )


class CalibrationConfig(BaseModel):
    """Configuration for confidence calibration."""

    model_config = ConfigDict(extra="forbid")

    method: str = Field(
        default="temperature_scaling",
        description="Calibration method: 'temperature_scaling', 'platt_scaling', 'isotonic'"
    )

    n_bins: int = Field(
        default=10,
        gt=0,
        description="Number of bins for calibration curve"
    )

    recalibration_frequency_hours: int = Field(
        default=24,
        gt=0,
        description="Hours between automatic recalibration"
    )

    max_iter: int = Field(
        default=50,
        gt=0,
        description="Maximum optimization iterations"
    )

    save_calibration_curves: bool = Field(
        default=True,
        description="Save reliability diagrams"
    )


class DriftDetectionConfig(BaseModel):
    """Configuration for drift detection."""

    model_config = ConfigDict(extra="forbid")

    # PSI thresholds
    psi_warning_threshold: float = Field(
        default=0.10,
        ge=0.0,
        description="PSI threshold for warning"
    )
    psi_alert_threshold: float = Field(
        default=0.25,
        ge=0.0,
        description="PSI threshold for alert"
    )

    # Embedding drift
    embedding_drift_method: str = Field(
        default="domain_classifier",
        description="Method: 'domain_classifier', 'mmd', 'wasserstein'"
    )
    embedding_drift_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="AUC threshold for domain classifier"
    )

    # Statistical tests
    statistical_test_alpha: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Significance level for statistical tests"
    )

    # Monitoring parameters
    window_size: int = Field(
        default=1000,
        gt=0,
        description="Sliding window size for drift calculation"
    )
    check_interval: int = Field(
        default=100,
        gt=0,
        description="Check drift every N samples"
    )

    # Features to monitor
    monitor_label_distribution: bool = Field(default=True)
    monitor_confidence_distribution: bool = Field(default=True)
    monitor_embeddings: bool = Field(default=True)

    # Alert configuration
    alert_on_warning: bool = Field(default=False)
    alert_on_drift: bool = Field(default=True)


class HITLRoutingConfig(BaseModel):
    """Configuration for human-in-the-loop routing."""

    model_config = ConfigDict(extra="forbid")

    # Confidence thresholds
    auto_accept_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for auto-acceptance"
    )
    expert_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for expert review"
    )

    # Adaptive thresholding
    enable_adaptive_threshold: bool = Field(
        default=True,
        description="Dynamically adjust thresholds based on performance"
    )
    target_accuracy: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Target accuracy for adaptive thresholding"
    )
    cost_constraint_usd: float | None = Field(
        default=None,
        description="Maximum cost constraint (USD)"
    )

    # Complexity-based routing
    use_complexity_adjustment: bool = Field(
        default=True,
        description="Adjust routing based on task complexity"
    )
    complexity_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for complexity in routing decision"
    )

    # Cost tracking
    llm_cost_per_annotation: float = Field(
        default=0.001,
        description="Cost per LLM annotation (USD)"
    )
    human_review_cost: float = Field(
        default=0.10,
        description="Cost per human review (USD)"
    )
    expert_review_cost: float = Field(
        default=0.50,
        description="Cost per expert review (USD)"
    )

    # Quality targets
    min_human_review_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum % for quality assurance"
    )
    max_human_review_rate: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Maximum % to cap costs"
    )


class AlertConfig(BaseModel):
    """Configuration for alert system."""

    model_config = ConfigDict(extra="forbid")

    # Performance alerts
    accuracy_drop_threshold: float = Field(
        default=0.10,
        description="Accuracy drop threshold for alert"
    )
    f1_drop_threshold: float = Field(
        default=0.10,
        description="F1 drop threshold for alert"
    )

    # Agreement alerts
    iaa_alpha_threshold: float = Field(
        default=0.60,
        description="IAA alpha threshold for alert"
    )
    consecutive_low_agreement: int = Field(
        default=100,
        description="Consecutive low agreement samples to trigger alert"
    )

    # Calibration alerts
    ece_threshold: float = Field(
        default=0.15,
        description="ECE threshold for calibration alert"
    )

    # Drift alerts
    psi_threshold: float = Field(
        default=0.25,
        description="PSI threshold for drift alert"
    )
    embedding_drift_threshold: float = Field(
        default=0.30,
        description="Embedding drift threshold for alert"
    )

    # Cost alerts
    cost_spike_multiplier: float = Field(
        default=2.0,
        description="Cost spike multiplier for alert"
    )
    review_rate_threshold: float = Field(
        default=0.50,
        description="Review rate threshold for alert"
    )

    # Alert channels
    log_alerts: bool = Field(default=True)
    email_recipients: list[str] = Field(default_factory=list)
    slack_webhook: str | None = Field(default=None)


class QualityControlConfig(BaseModel):
    """Master configuration for quality control system."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(description="Dataset identifier")
    version: str = Field(default="1.0", description="Config version")

    # Component configurations
    iaa: IAAConfig = Field(default_factory=IAAConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    drift_detection: DriftDetectionConfig = Field(default_factory=DriftDetectionConfig)
    hitl_routing: HITLRoutingConfig = Field(default_factory=HITLRoutingConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)

    # Dashboard settings
    dashboard_enabled: bool = Field(default=True)
    dashboard_update_interval_seconds: int = Field(default=60, gt=0)
    dashboard_retention_days: int = Field(default=90, gt=0)


# ============================================================================
# DATA MODELS
# ============================================================================

class AnnotationProvenance(BaseModel):
    """Complete provenance tracking for a single annotation."""

    model_config = ConfigDict(extra="allow")

    # Identifiers
    annotation_id: str = Field(description="Unique annotation identifier")
    text_id: str = Field(description="Identifier for the text being annotated")
    text_hash: str = Field(description="SHA-256 hash of text content")
    dataset_name: str = Field(description="Dataset identifier")

    # Annotation content
    text: str = Field(description="The annotated text")
    label: str = Field(description="Assigned label")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score"
    )

    # Annotator information
    annotator_id: str = Field(description="Annotator identifier")
    annotator_type: AnnotatorType = Field(description="Type of annotator")

    # Model details (if LLM/ensemble)
    model_name: str | None = Field(default=None)
    model_version: str | None = Field(default=None)
    temperature: float | None = Field(default=None, ge=0.0)
    prompt_id: str | None = Field(default=None)

    # Agreement metrics
    iaa_alpha: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Krippendorff's alpha (if multi-annotator)"
    )
    disagreement_count: int = Field(
        default=0,
        ge=0,
        description="Number of disagreeing annotators"
    )
    alternative_labels: list[str] = Field(
        default_factory=list,
        description="Alternative labels from other annotators"
    )

    # Quality indicators
    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite quality score"
    )
    review_status: ReviewStatus = Field(
        default=ReviewStatus.PENDING_REVIEW,
        description="Current review status"
    )
    human_validated: bool = Field(
        default=False,
        description="Whether validated by human"
    )
    validation_notes: str | None = Field(default=None)

    # Temporal tracking
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Last update timestamp"
    )
    annotation_duration_ms: int | None = Field(
        default=None,
        ge=0,
        description="Time taken to annotate (ms)"
    )

    # Task metadata
    task_description: str | None = Field(default=None)
    guideline_version: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class IAAAgreementResult(BaseModel):
    """Results from IAA calculation."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(
        ge=-1.0,
        le=1.0,
        description="Krippendorff's alpha"
    )
    confidence_interval: tuple[float, float] = Field(
        description="Bootstrap confidence interval"
    )
    interpretation: str = Field(
        description="Human-readable interpretation"
    )

    per_item_agreement: dict[str, float] = Field(
        default_factory=dict,
        description="Agreement score per item"
    )
    annotator_reliability: dict[str, float] = Field(
        default_factory=dict,
        description="Reliability score per annotator"
    )

    # Disagreement analysis
    disagreement_matrix: list[list[float]] = Field(
        default_factory=list,
        description="Confusion matrix between annotators"
    )
    systematic_errors: list[tuple[str, str, int]] = Field(
        default_factory=list,
        description="Common label confusions (label1, label2, count)"
    )

    # Metadata
    num_items: int = Field(ge=0)
    num_annotators: int = Field(ge=0)
    metric: str = Field(description="Distance metric used")
    timestamp: datetime = Field(default_factory=datetime.now)


class PerformanceMetrics(BaseModel):
    """Performance metrics calculated on sliding window."""

    model_config = ConfigDict(extra="forbid")

    # Classification metrics
    accuracy: float = Field(ge=0.0, le=1.0)
    f1_weighted: float = Field(ge=0.0, le=1.0)
    precision_weighted: float = Field(ge=0.0, le=1.0)
    recall_weighted: float = Field(ge=0.0, le=1.0)
    cohen_kappa: float = Field(ge=-1.0, le=1.0)

    # Per-class metrics
    per_class_f1: dict[str, float] = Field(default_factory=dict)
    per_class_support: dict[str, int] = Field(default_factory=dict)

    # Confusion matrix
    confusion_matrix: list[list[int]] = Field(default_factory=list)
    label_names: list[str] = Field(default_factory=list)

    # Confidence metrics
    mean_confidence: float = Field(ge=0.0, le=1.0)
    confidence_std: float = Field(ge=0.0)
    confidence_accuracy_correlation: float = Field(ge=-1.0, le=1.0)

    # Agreement metrics
    mean_iaa_alpha: float | None = Field(default=None, ge=-1.0, le=1.0)

    # Temporal info
    window_start: datetime
    window_end: datetime
    num_samples: int = Field(ge=0)


class CalibrationMetrics(BaseModel):
    """Confidence calibration quality metrics."""

    model_config = ConfigDict(extra="forbid")

    # Calibration quality
    expected_calibration_error: float = Field(
        ge=0.0,
        le=1.0,
        description="ECE score"
    )
    maximum_calibration_error: float = Field(
        ge=0.0,
        le=1.0,
        description="MCE score"
    )
    brier_score: float = Field(
        ge=0.0,
        description="Brier score"
    )
    log_loss: float = Field(
        ge=0.0,
        description="Log loss"
    )

    # Calibration curve data
    bin_confidences: list[float] = Field(
        default_factory=list,
        description="Mean predicted confidence per bin"
    )
    bin_accuracies: list[float] = Field(
        default_factory=list,
        description="True accuracy per bin"
    )
    bin_counts: list[int] = Field(
        default_factory=list,
        description="Sample count per bin"
    )

    # Reliability analysis
    overconfident_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="% where confidence > accuracy"
    )
    underconfident_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="% where confidence < accuracy"
    )
    perfect_calibration_gap: float = Field(
        ge=0.0,
        description="Area between curve and diagonal"
    )

    # Metadata
    num_samples: int = Field(ge=0)
    num_bins: int = Field(ge=0)
    calibration_method: str | None = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.now)


class CostMetrics(BaseModel):
    """Annotation cost tracking."""

    model_config = ConfigDict(extra="forbid")

    # LLM costs
    total_llm_tokens: int = Field(ge=0)
    total_llm_cost_usd: float = Field(ge=0.0)
    cost_per_annotation: float = Field(ge=0.0)

    # Human costs
    human_review_count: int = Field(ge=0)
    expert_review_count: int = Field(ge=0)
    human_review_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="% requiring human review"
    )
    estimated_human_cost_usd: float = Field(ge=0.0)
    total_cost_usd: float = Field(ge=0.0)

    # Efficiency metrics
    auto_accept_rate: float = Field(ge=0.0, le=1.0)
    expert_review_rate: float = Field(ge=0.0, le=1.0)
    average_annotation_time_ms: float = Field(ge=0.0)

    # Cost savings
    baseline_human_cost: float = Field(ge=0.0)
    cost_savings_usd: float = Field(ge=0.0)
    cost_savings_percentage: float = Field(ge=0.0, le=1.0)

    # Temporal
    window_start: datetime
    window_end: datetime
    num_annotations: int = Field(ge=0)


class DriftReport(BaseModel):
    """Comprehensive drift detection report."""

    model_config = ConfigDict(extra="forbid")

    # Overall assessment
    drift_detected: bool
    overall_interpretation: DriftInterpretation

    # PSI analysis
    label_psi: float = Field(ge=0.0, description="Label distribution PSI")
    label_interpretation: DriftInterpretation
    confidence_psi: float = Field(ge=0.0, description="Confidence distribution PSI")
    confidence_interpretation: DriftInterpretation

    # Embedding drift
    embedding_drift_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Embedding drift AUC"
    )
    embedding_drift_detected: bool | None = Field(default=None)
    drifting_dimensions: list[int] = Field(
        default_factory=list,
        description="Top drifting embedding dimensions"
    )

    # Statistical tests
    statistical_test_name: str | None = Field(default=None)
    statistical_test_statistic: float | None = Field(default=None)
    statistical_test_pvalue: float | None = Field(default=None)
    statistical_drift_detected: bool | None = Field(default=None)

    # Recommendations
    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommended actions"
    )

    # Metadata
    baseline_date: datetime
    current_date: datetime
    num_baseline_samples: int = Field(ge=0)
    num_current_samples: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)


class RoutingDecision(BaseModel):
    """Routing decision for a single prediction."""

    model_config = ConfigDict(extra="forbid")

    annotation_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    complexity_score: float | None = Field(default=None, ge=0.0, le=1.0)

    # Routing outcome
    review_status: ReviewStatus
    routed_to: str = Field(
        description="'auto_accept', 'human_review', 'expert_review'"
    )

    # Cost estimate
    estimated_cost_usd: float = Field(ge=0.0)

    # Decision factors
    confidence_threshold_used: float = Field(ge=0.0, le=1.0)
    complexity_adjusted: bool = Field(default=False)
    decision_reasoning: str | None = Field(default=None)

    timestamp: datetime = Field(default_factory=datetime.now)


class RoutingResult(BaseModel):
    """Aggregate routing results for a batch."""

    model_config = ConfigDict(extra="forbid")

    total_annotations: int = Field(ge=0)
    auto_accept_count: int = Field(ge=0)
    human_review_count: int = Field(ge=0)
    expert_review_count: int = Field(ge=0)

    # Rates
    auto_accept_rate: float = Field(ge=0.0, le=1.0)
    human_review_rate: float = Field(ge=0.0, le=1.0)
    expert_review_rate: float = Field(ge=0.0, le=1.0)

    # Cost breakdown
    total_cost_usd: float = Field(ge=0.0)
    llm_cost_usd: float = Field(ge=0.0)
    human_cost_usd: float = Field(ge=0.0)
    expert_cost_usd: float = Field(ge=0.0)

    # Individual decisions
    decisions: list[RoutingDecision] = Field(default_factory=list)

    # Metadata
    threshold_used: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)


class HumanCorrection(BaseModel):
    """Record of a human correction."""

    model_config = ConfigDict(extra="forbid")

    correction_id: str
    annotation_id: str
    dataset_name: str

    # Correction details
    original_label: str
    corrected_label: str
    original_confidence: float = Field(ge=0.0, le=1.0)

    # Annotator info
    annotator_id: str
    annotator_expertise: str | None = Field(default=None)
    correction_notes: str | None = Field(default=None)

    # Analysis
    systematic_error: bool = Field(default=False)
    error_category: str | None = Field(default=None)
    added_to_knowledge_base: bool = Field(default=False)

    # Temporal
    correction_timestamp: datetime = Field(default_factory=datetime.now)


class CorrectionAnalysis(BaseModel):
    """Analysis of human correction patterns."""

    model_config = ConfigDict(extra="forbid")

    total_corrections: int = Field(ge=0)
    overall_error_rate: float = Field(ge=0.0, le=1.0)

    # Error patterns
    error_rate_by_confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Error rate per confidence bin"
    )
    systematic_errors: list[tuple[str, str, int]] = Field(
        default_factory=list,
        description="Frequent label confusions"
    )
    model_weaknesses: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Topics/patterns with high error rate"
    )

    # Routing improvements
    recommended_threshold_adjustments: dict[str, float] = Field(
        default_factory=dict
    )
    expected_improvement: float | None = Field(default=None)

    # Metadata
    window_size: int = Field(gt=0)
    analysis_date: datetime = Field(default_factory=datetime.now)


class QualityAlert(BaseModel):
    """Quality alert notification."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str
    dataset_name: str
    alert_type: AlertType
    severity: AlertSeverity

    # Alert details
    metric_name: str
    metric_value: float
    threshold_value: float
    alert_message: str

    # Context
    window_start: datetime | None = Field(default=None)
    window_end: datetime | None = Field(default=None)
    affected_samples: int = Field(default=0, ge=0)

    # Actions
    recommended_actions: list[str] = Field(default_factory=list)
    auto_remediation_attempted: bool = Field(default=False)
    remediation_notes: str | None = Field(default=None)

    # Temporal
    triggered_at: datetime = Field(default_factory=datetime.now)
    resolved_at: datetime | None = Field(default=None)
    acknowledged_by: str | None = Field(default=None)


class CalibrationResult(BaseModel):
    """Results from calibration fitting."""

    model_config = ConfigDict(extra="forbid")

    method: str
    success: bool

    # Calibration parameters
    temperature: float | None = Field(default=None)
    platt_a: float | None = Field(default=None)
    platt_b: float | None = Field(default=None)

    # Performance
    ece_before: float = Field(ge=0.0, le=1.0)
    ece_after: float = Field(ge=0.0, le=1.0)
    improvement: float

    # Metadata
    num_samples: int = Field(ge=0)
    calibration_date: datetime = Field(default_factory=datetime.now)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_annotation_provenance(
    text: str,
    label: str,
    confidence: float,
    annotator_id: str,
    annotator_type: AnnotatorType,
    dataset_name: str,
    **kwargs
) -> AnnotationProvenance:
    """
    Factory function to create AnnotationProvenance with sensible defaults.

    Args:
        text: The annotated text
        label: Assigned label
        confidence: Confidence score
        annotator_id: Annotator identifier
        annotator_type: Type of annotator
        dataset_name: Dataset identifier
        **kwargs: Additional fields

    Returns:
        AnnotationProvenance instance
    """
    import hashlib
    import uuid

    text_hash = hashlib.sha256(text.encode()).hexdigest()
    annotation_id = str(uuid.uuid4())
    text_id = kwargs.pop('text_id', str(uuid.uuid4()))

    return AnnotationProvenance(
        annotation_id=annotation_id,
        text_id=text_id,
        text_hash=text_hash,
        text=text,
        label=label,
        confidence=confidence,
        annotator_id=annotator_id,
        annotator_type=annotator_type,
        dataset_name=dataset_name,
        **kwargs
    )


def serialize_for_parquet(obj: BaseModel) -> dict[str, Any]:
    """
    Serialize Pydantic model for Parquet storage.

    Handles nested structures and datetime serialization.
    """
    data = obj.model_dump()

    # Convert datetime to ISO string
    for key, value in data.items():
        if isinstance(value, datetime):
            data[key] = value.isoformat()
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
            # Serialize list of dicts as JSON string
            import json
            data[key] = json.dumps(value)

    return data


def deserialize_from_parquet(data: dict[str, Any], model_class: type[BaseModel]) -> BaseModel:
    """
    Deserialize Parquet row to Pydantic model.

    Args:
        data: Dictionary from Parquet row
        model_class: Target Pydantic model class

    Returns:
        Model instance
    """
    import json

    # Convert ISO strings back to datetime
    for key, value in data.items():
        if isinstance(value, str) and 'timestamp' in key.lower():
            try:
                data[key] = datetime.fromisoformat(value)
            except ValueError:
                pass
        elif isinstance(value, str) and value.startswith('['):
            # Deserialize JSON lists
            try:
                data[key] = json.loads(value)
            except json.JSONDecodeError:
                pass

    return model_class(**data)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create annotation provenance
    provenance = create_annotation_provenance(
        text="This movie was fantastic!",
        label="positive",
        confidence=0.95,
        annotator_id="gpt-4",
        annotator_type=AnnotatorType.LLM,
        dataset_name="movie_reviews",
        model_name="gpt-4-turbo",
        temperature=0.1
    )
    print(f"Created provenance: {provenance.annotation_id}")

    # Example: Configuration
    config = QualityControlConfig(
        dataset_name="movie_reviews",
        iaa=IAAConfig(
            auto_accept_threshold=0.95,
            review_threshold=0.70
        ),
        hitl_routing=HITLRoutingConfig(
            enable_adaptive_threshold=True,
            target_accuracy=0.95
        )
    )
    print(f"\nConfig: {config.model_dump_json(indent=2)}")

    # Example: Serialize for Parquet
    serialized = serialize_for_parquet(provenance)
    print(f"\nSerialized keys: {list(serialized.keys())}")
