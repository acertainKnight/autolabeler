# Quality Monitoring and Confidence Calibration

This module provides comprehensive quality monitoring and confidence calibration capabilities for the AutoLabeler system.

## Components

### 1. ConfidenceCalibrator

Calibrates model confidence scores to better reflect true prediction accuracy.

**Features:**
- **Temperature Scaling**: Applies a single temperature parameter to logits
- **Platt Scaling**: Uses logistic regression for calibration
- **Evaluation Metrics**: ECE, MCE, Brier Score, Log Loss
- **State Persistence**: Save and load calibration parameters

**Example:**
```python
from sibyls.core.quality import ConfidenceCalibrator

# Initialize calibrator
calibrator = ConfidenceCalibrator(method="temperature", n_bins=10)

# Fit on training data
calibrator.fit(
    confidence_scores=train_confidences,
    true_labels=train_true_labels,
    predicted_labels=train_pred_labels
)

# Calibrate new predictions
calibrated_scores = calibrator.calibrate(test_confidences)

# Evaluate calibration quality
metrics = calibrator.evaluate_calibration(
    test_confidences,
    test_true_labels,
    test_pred_labels,
    apply_calibration=True
)

print(f"ECE: {metrics['expected_calibration_error']:.4f}")
print(f"Brier Score: {metrics['brier_score']:.4f}")
```

### 2. QualityMonitor

Monitors annotation quality with inter-annotator agreement and performance tracking.

**Features:**
- **Krippendorff's Alpha**: Measures inter-annotator agreement
- **Per-Annotator Metrics**: Accuracy, Cohen's kappa, confidence statistics
- **CQAA**: Cost Per Quality-Adjusted Annotation
- **Anomaly Detection**: Identifies quality issues automatically

**Example:**
```python
from sibyls.core.quality import QualityMonitor
import pandas as pd

# Initialize monitor
monitor = QualityMonitor(dataset_name="sentiment_analysis")

# Calculate inter-annotator agreement
df = pd.read_csv("annotations.csv")
alpha_result = monitor.calculate_krippendorff_alpha(
    df,
    annotator_columns=["annotator1", "annotator2", "annotator3"]
)

print(f"Krippendorff's Alpha: {alpha_result['alpha']:.4f}")
if alpha_result['alpha'] > 0.8:
    print("Excellent agreement!")
elif alpha_result['alpha'] > 0.67:
    print("Good agreement")
else:
    print("Poor agreement - review annotation guidelines")

# Track per-annotator performance
metrics = monitor.track_annotator_metrics(
    df,
    annotator_id_column="annotator_id",
    label_column="label",
    gold_label_column="gold_standard",
    confidence_column="confidence"
)

# Calculate cost efficiency
cqaa = monitor.calculate_cqaa(
    annotations=1000,
    accuracy=0.85,
    cost_per_annotation=0.50
)
print(f"CQAA: ${cqaa['cqaa']:.4f} per quality-adjusted annotation")

# Detect anomalies
anomalies = monitor.detect_anomalies(
    metrics,
    accuracy_threshold=0.7,
    confidence_std_threshold=0.3
)

for anomaly in anomalies:
    print(f"⚠️  {anomaly['annotator_id']}: {anomaly['issue']}")
```

### 3. Quality Dashboard

Interactive Streamlit dashboard for real-time quality monitoring.

**Features:**
- Confidence calibration analysis and visualization
- Inter-annotator agreement tracking
- Per-annotator performance metrics
- Anomaly detection and alerts
- Calibration curve plotting
- Confidence distribution analysis

**Launch Dashboard:**
```bash
# Install dashboard dependencies
pip install -e ".[dashboard]"

# Run dashboard
streamlit run src/sibyls/dashboard/quality_dashboard.py
```

**Dashboard Tabs:**

1. **Confidence Calibration**
   - Before/after calibration metrics
   - Calibration curves
   - Confidence distribution plots
   - ECE, Brier Score, Log Loss

2. **Inter-Annotator Agreement**
   - Krippendorff's Alpha calculation
   - Pairwise agreement matrix
   - Per-item agreement statistics
   - Quality interpretation

3. **Annotator Performance**
   - Per-annotator accuracy
   - Cohen's Kappa scores
   - Annotation counts
   - Confidence statistics

4. **Quality Anomalies**
   - Low accuracy detection
   - Unusual annotation rates
   - Confidence pattern issues
   - Severity levels and recommendations

## Metrics Explained

### Expected Calibration Error (ECE)

Measures the difference between predicted confidence and actual accuracy across bins.

- **Range**: 0 to 1 (lower is better)
- **Interpretation**:
  - < 0.05: Well calibrated
  - 0.05-0.15: Moderately calibrated
  - \> 0.15: Poorly calibrated

### Krippendorff's Alpha

Reliability coefficient for inter-annotator agreement, accounting for chance.

- **Range**: -1 to 1 (higher is better)
- **Interpretation**:
  - \> 0.80: Excellent agreement
  - 0.67-0.80: Good agreement (tentatively reliable)
  - 0.60-0.67: Moderate agreement
  - < 0.60: Poor agreement (unreliable)

### Cost Per Quality-Adjusted Annotation (CQAA)

Economic efficiency metric that accounts for annotation quality.

```
CQAA = Total Cost / (Annotations × Accuracy^quality_weight)
```

- Lower CQAA = Better cost-efficiency
- Useful for comparing different annotation strategies
- Quality weight can be adjusted based on task criticality

## Integration with AutoLabeler

### Evaluating Model Confidence

```python
from sibyls.core.labeling import LabelingService
from sibyls.core.quality import ConfidenceCalibrator

# Label data
service = LabelingService("sentiment", settings)
results_df = service.label_dataframe(df, "text")

# Calibrate confidence scores
calibrator = ConfidenceCalibrator()
calibrator.fit(
    train_df["confidence"],
    train_df["true_label"],
    train_df["predicted_label"]
)

# Apply to new predictions
results_df["calibrated_confidence"] = calibrator.calibrate(
    results_df["confidence"]
)
```

### Monitoring Ensemble Quality

```python
from sibyls.core.ensemble import EnsembleService
from sibyls.core.quality import QualityMonitor

# Create ensemble
ensemble = EnsembleService("sentiment", settings)
ensemble.add_model(ModelConfig(model_name="gpt-4"))
ensemble.add_model(ModelConfig(model_name="claude-3"))

# Label and track agreement
results_df = ensemble.label_dataframe_ensemble(df, "text")

# Monitor inter-model agreement (treat models as annotators)
monitor = QualityMonitor("sentiment")
agreement = monitor.calculate_krippendorff_alpha(
    results_df,
    ["model_1_label", "model_2_label"]
)

print(f"Inter-model agreement: {agreement['alpha']:.4f}")
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run quality module tests
pytest tests/test_unit/quality/ -v

# Run with coverage
pytest tests/test_unit/quality/ --cov=sibyls.core.quality --cov-report=html
```

## Best Practices

### Confidence Calibration

1. **Split your data**: Use separate train/test sets for calibration
2. **Choose method wisely**: Temperature scaling works well for neural networks, Platt scaling for traditional models
3. **Monitor ECE**: Track ECE over time to detect calibration drift
4. **Recalibrate periodically**: As model or data distribution changes

### Quality Monitoring

1. **Set appropriate thresholds**: Adjust based on task difficulty and cost
2. **Track trends**: Monitor Krippendorff's alpha over time
3. **Investigate anomalies**: Use anomaly detection to identify training needs
4. **Calculate CQAA**: Optimize for cost-efficiency without sacrificing quality

### Dashboard Usage

1. **Upload labeled datasets**: Include confidence scores and multiple annotators
2. **Experiment with parameters**: Try different bin counts and thresholds
3. **Export insights**: Save calibration parameters and quality reports
4. **Regular monitoring**: Check dashboard weekly during active labeling

## References

- Krippendorff, K. (2004). "Reliability in Content Analysis"
- Guo et al. (2017). "On Calibration of Modern Neural Networks"
- Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines"
