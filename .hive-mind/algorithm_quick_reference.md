# Quality Control Algorithms - Quick Reference

**Purpose:** Quick lookup for algorithm implementations during development
**Audience:** Implementation team
**Date:** 2025-10-07

---

## Table of Contents

1. [Inter-Annotator Agreement (IAA)](#1-inter-annotator-agreement)
2. [Confidence Calibration](#2-confidence-calibration)
3. [Drift Detection](#3-drift-detection)
4. [Routing Optimization](#4-routing-optimization)

---

## 1. Inter-Annotator Agreement

### Krippendorff's Alpha

**Python Implementation:**

```python
import krippendorff
import numpy as np

def calculate_krippendorff_alpha(
    reliability_matrix: np.ndarray,
    metric: str = "nominal"
) -> float:
    """
    Calculate Krippendorff's alpha for IAA.

    Args:
        reliability_matrix: [items x annotators], NaN for missing
        metric: "nominal", "ordinal", "interval", "ratio"

    Returns:
        Alpha value [-1, 1] (>0.8 is good)
    """
    alpha = krippendorff.alpha(
        reliability_data=reliability_matrix.T,  # Transpose to [annotators x items]
        level_of_measurement=metric
    )
    return alpha

# Example usage
reliability_matrix = np.array([
    [0, 0, 1],  # Item 1: Annotators 1,2,3 → labels 0,0,1
    [1, 1, 1],  # Item 2: All agree on label 1
    [0, np.nan, 0],  # Item 3: Annotator 2 missing
])
alpha = calculate_krippendorff_alpha(reliability_matrix, metric="nominal")
print(f"Alpha: {alpha:.3f}")
```

**Bootstrap Confidence Interval:**

```python
from scipy.stats import bootstrap

def krippendorff_alpha_with_ci(
    reliability_matrix: np.ndarray,
    metric: str = "nominal",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> tuple[float, tuple[float, float]]:
    """
    Calculate alpha with bootstrap confidence interval.

    Returns:
        (alpha, (lower_ci, upper_ci))
    """
    alpha = calculate_krippendorff_alpha(reliability_matrix, metric)

    # Bootstrap
    def alpha_statistic(data, axis):
        return calculate_krippendorff_alpha(data, metric)

    rng = np.random.default_rng(42)
    res = bootstrap(
        (reliability_matrix,),
        alpha_statistic,
        n_resamples=n_bootstrap,
        confidence_level=confidence_level,
        random_state=rng
    )

    return alpha, (res.confidence_interval.low, res.confidence_interval.high)
```

**Interpretation:**

| Alpha Range | Interpretation | Action |
|-------------|----------------|--------|
| α ≥ 0.95 | Excellent agreement | Auto-accept |
| 0.80 ≤ α < 0.95 | Good agreement | Monitor |
| 0.70 ≤ α < 0.80 | Moderate agreement | Human review |
| 0.60 ≤ α < 0.70 | Low agreement | Expert review |
| α < 0.60 | Poor agreement | Guideline clarification needed |

---

## 2. Confidence Calibration

### Temperature Scaling

**Implementation:**

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

class TemperatureScaling:
    """Calibrate multi-class predictions."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """
        Learn optimal temperature.

        Args:
            logits: [N x C] pre-softmax logits
            labels: [N] true labels (integer class indices)
        """
        def nll_loss(T):
            """Negative log-likelihood."""
            scaled_probs = softmax(logits / T, axis=1)
            nll = -np.log(scaled_probs[np.arange(len(labels)), labels] + 1e-10).mean()
            return nll

        result = minimize(
            nll_loss,
            x0=np.array([1.0]),
            method='L-BFGS-B',
            bounds=[(0.01, 10.0)],
            options={'maxiter': 50}
        )

        self.temperature = result.x[0]

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        return softmax(logits / self.temperature, axis=1)

# Example usage
logits = np.random.randn(100, 3)  # 100 samples, 3 classes
labels = np.random.randint(0, 3, 100)

calibrator = TemperatureScaling()
calibrator.fit(logits, labels)
calibrated_probs = calibrator.calibrate(logits)

print(f"Optimal temperature: {calibrator.temperature:.3f}")
```

### Platt Scaling (Binary Classification)

**Implementation:**

```python
from sklearn.linear_model import LogisticRegression

class PlattScaling:
    """Calibrate binary classification."""

    def __init__(self):
        self.A = 1.0
        self.B = 0.0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """
        Learn Platt parameters.

        Args:
            scores: [N] model scores (logits or probabilities)
            labels: [N] binary labels (0/1)
        """
        clf = LogisticRegression()
        clf.fit(scores.reshape(-1, 1), labels)

        self.A = clf.coef_[0][0]
        self.B = clf.intercept_[0]

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        return 1.0 / (1.0 + np.exp(self.A * scores + self.B))

# Example usage
scores = np.random.randn(100)
labels = (scores > 0).astype(int)

calibrator = PlattScaling()
calibrator.fit(scores, labels)
calibrated_probs = calibrator.calibrate(scores)
```

### Expected Calibration Error (ECE)

**Implementation:**

```python
def calculate_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error.

    Args:
        confidences: [N] predicted confidence scores [0,1]
        predictions: [N] predicted class labels
        ground_truth: [N] true class labels
        n_bins: Number of confidence bins

    Returns:
        ECE score [0,1] (lower is better)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # Find samples in this bin
        bin_mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])

        if bin_mask.sum() > 0:
            # Average confidence in bin
            bin_confidence = confidences[bin_mask].mean()

            # Accuracy in bin
            bin_accuracy = (predictions[bin_mask] == ground_truth[bin_mask]).mean()

            # Weighted contribution to ECE
            weight = bin_mask.sum() / len(confidences)
            ece += weight * abs(bin_confidence - bin_accuracy)

    return ece

# Example usage
confidences = np.random.uniform(0, 1, 1000)
predictions = (confidences > 0.5).astype(int)
ground_truth = np.random.randint(0, 2, 1000)

ece = calculate_ece(confidences, predictions, ground_truth)
print(f"ECE: {ece:.3f}")
```

**Interpretation:**

| ECE Range | Calibration Quality | Action |
|-----------|---------------------|--------|
| ECE < 0.05 | Excellent | No action |
| 0.05 ≤ ECE < 0.15 | Good | Monitor |
| ECE ≥ 0.15 | Poor | Recalibrate |

---

## 3. Drift Detection

### Population Stability Index (PSI)

**Implementation:**

```python
def calculate_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    bins: int = 10
) -> dict:
    """
    Calculate Population Stability Index.

    Args:
        baseline: Baseline distribution (training data)
        current: Current distribution (production data)
        bins: Number of bins

    Returns:
        {'psi': float, 'interpretation': str, 'per_bin_psi': list}
    """
    # Create bins from baseline
    bin_edges = np.percentile(baseline, np.linspace(0, 100, bins + 1))

    # Histogram for both distributions
    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    # Convert to percentages (add small epsilon)
    baseline_pct = (baseline_counts + 1e-10) / baseline_counts.sum()
    current_pct = (current_counts + 1e-10) / current_counts.sum()

    # Calculate PSI
    psi_values = (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
    psi = psi_values.sum()

    # Interpret
    if psi < 0.10:
        interpretation = "stable"
    elif psi < 0.25:
        interpretation = "slightly_unstable"
    else:
        interpretation = "unstable"

    return {
        'psi': float(psi),
        'interpretation': interpretation,
        'per_bin_psi': psi_values.tolist(),
        'bin_edges': bin_edges.tolist()
    }

# Example usage
baseline = np.random.normal(0, 1, 1000)
current = np.random.normal(0.2, 1.1, 1000)  # Shifted distribution

result = calculate_psi(baseline, current)
print(f"PSI: {result['psi']:.3f} ({result['interpretation']})")
```

**Interpretation:**

| PSI Range | Interpretation | Action |
|-----------|----------------|--------|
| PSI < 0.10 | No significant change | Continue monitoring |
| 0.10 ≤ PSI < 0.25 | Slight change | Investigate causes |
| PSI ≥ 0.25 | Major shift | Alert, consider retraining |

### Embedding Drift (Domain Classifier)

**Implementation:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def detect_embedding_drift(
    baseline_embeddings: np.ndarray,
    current_embeddings: np.ndarray,
    threshold: float = 0.75
) -> dict:
    """
    Detect drift using domain classifier.

    Args:
        baseline_embeddings: [N x D] training embeddings
        current_embeddings: [M x D] production embeddings
        threshold: AUC threshold for drift detection

    Returns:
        {
            'drift_detected': bool,
            'drift_score': float,  # AUC
            'drifting_dimensions': list[int]
        }
    """
    # Prepare binary classification dataset
    X_baseline = baseline_embeddings
    y_baseline = np.zeros(len(X_baseline))

    X_current = current_embeddings
    y_current = np.ones(len(X_current))

    X = np.vstack([X_baseline, X_current])
    y = np.concatenate([y_baseline, y_current])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train classifier
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    # Detect drift
    drift_detected = auc > threshold

    # Identify drifting dimensions
    feature_importance = np.abs(clf.coef_[0])
    top_k = 10
    drifting_dims = np.argsort(feature_importance)[-top_k:].tolist()

    return {
        'drift_detected': drift_detected,
        'drift_score': float(auc),
        'interpretation': 'Significant drift' if drift_detected else 'No drift',
        'drifting_dimensions': drifting_dims
    }

# Example usage
baseline_embeddings = np.random.randn(500, 128)
current_embeddings = np.random.randn(500, 128) + 0.3  # Shifted embeddings

result = detect_embedding_drift(baseline_embeddings, current_embeddings)
print(f"Drift AUC: {result['drift_score']:.3f}")
print(f"Drift detected: {result['drift_detected']}")
```

### Kolmogorov-Smirnov Test (Continuous Features)

**Implementation:**

```python
from scipy.stats import ks_2samp

def ks_drift_test(
    baseline: np.ndarray,
    current: np.ndarray,
    alpha: float = 0.05
) -> dict:
    """
    Kolmogorov-Smirnov test for distribution shift.

    Args:
        baseline: Baseline samples
        current: Current samples
        alpha: Significance level

    Returns:
        {'drift_detected': bool, 'statistic': float, 'p_value': float}
    """
    statistic, p_value = ks_2samp(baseline, current)

    drift_detected = p_value < alpha

    return {
        'test_name': 'Kolmogorov-Smirnov',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'drift_detected': drift_detected,
        'interpretation': f"{'Significant' if drift_detected else 'No significant'} drift (p={p_value:.4f})"
    }

# Example usage
baseline = np.random.normal(0, 1, 1000)
current = np.random.normal(0.5, 1, 1000)  # Shifted mean

result = ks_drift_test(baseline, current)
print(f"KS statistic: {result['statistic']:.3f}, p={result['p_value']:.4f}")
print(f"Drift detected: {result['drift_detected']}")
```

---

## 4. Routing Optimization

### Adaptive Threshold Optimization

**Implementation:**

```python
def optimize_routing_threshold(
    confidences: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    target_accuracy: float = 0.95,
    llm_cost: float = 0.001,
    human_cost: float = 0.10
) -> dict:
    """
    Find optimal auto-accept threshold.

    Args:
        confidences: Model confidence scores
        predictions: Model predictions
        ground_truth: True labels
        target_accuracy: Minimum acceptable accuracy
        llm_cost: Cost per LLM annotation (USD)
        human_cost: Cost per human review (USD)

    Returns:
        {
            'optimal_threshold': float,
            'expected_accuracy': float,
            'expected_human_review_rate': float,
            'expected_cost': float
        }
    """
    thresholds = np.linspace(0.5, 0.99, 100)
    results = []

    for t in thresholds:
        # Auto-accept mask
        auto_accept = confidences >= t

        # Accuracy on auto-accepted samples
        if auto_accept.sum() > 0:
            accuracy = (predictions[auto_accept] == ground_truth[auto_accept]).mean()
        else:
            accuracy = 0.0

        # Cost calculation
        auto_accept_rate = auto_accept.mean()
        human_review_rate = 1 - auto_accept_rate
        cost = auto_accept_rate * llm_cost + human_review_rate * human_cost

        results.append({
            'threshold': t,
            'accuracy': accuracy,
            'human_review_rate': human_review_rate,
            'cost': cost
        })

    results_df = pd.DataFrame(results)

    # Find minimum threshold meeting accuracy target
    feasible = results_df[results_df['accuracy'] >= target_accuracy]

    if len(feasible) > 0:
        # Choose threshold minimizing cost
        optimal = feasible.loc[feasible['cost'].idxmin()]
    else:
        # No threshold meets accuracy target, choose best accuracy
        optimal = results_df.loc[results_df['accuracy'].idxmax()]

    return {
        'optimal_threshold': float(optimal['threshold']),
        'expected_accuracy': float(optimal['accuracy']),
        'expected_human_review_rate': float(optimal['human_review_rate']),
        'expected_cost': float(optimal['cost'])
    }

# Example usage
confidences = np.random.uniform(0, 1, 1000)
predictions = (confidences > 0.5).astype(int)
ground_truth = np.random.randint(0, 2, 1000)

result = optimize_routing_threshold(
    confidences, predictions, ground_truth,
    target_accuracy=0.95
)
print(f"Optimal threshold: {result['optimal_threshold']:.3f}")
print(f"Expected accuracy: {result['expected_accuracy']:.2%}")
print(f"Human review rate: {result['expected_human_review_rate']:.2%}")
print(f"Cost per annotation: ${result['expected_cost']:.4f}")
```

### Complexity-Adjusted Routing

**Implementation:**

```python
def complexity_adjusted_routing(
    confidence: float,
    complexity_score: float,
    base_threshold: float = 0.95,
    complexity_weight: float = 0.3
) -> str:
    """
    Route prediction based on confidence and complexity.

    Args:
        confidence: Model confidence [0,1]
        complexity_score: Task complexity [0,1]
        base_threshold: Base auto-accept threshold
        complexity_weight: Weight for complexity adjustment

    Returns:
        'auto_accept', 'human_review', or 'expert_review'
    """
    # Adjust threshold based on complexity
    adjusted_threshold = base_threshold + complexity_weight * complexity_score

    if confidence >= adjusted_threshold:
        return 'auto_accept'
    elif confidence >= 0.70:
        return 'human_review'
    else:
        return 'expert_review'

# Example usage
confidence = 0.92
complexity_score = 0.6  # Moderately complex

decision = complexity_adjusted_routing(confidence, complexity_score)
print(f"Routing decision: {decision}")
```

---

## Performance Optimization Tips

### 1. Vectorize Operations
```python
# BAD: Loop over samples
for i in range(len(confidences)):
    if confidences[i] > threshold:
        accept[i] = True

# GOOD: Vectorized
accept = confidences > threshold
```

### 2. Use NumPy Broadcasting
```python
# Calculate pairwise distances efficiently
distances = np.linalg.norm(embeddings[:, None] - embeddings[None, :], axis=2)
```

### 3. Batch Database Operations
```python
# BAD: Insert one by one
for annotation in annotations:
    cursor.execute("INSERT INTO ...", annotation)

# GOOD: Batch insert
cursor.executemany("INSERT INTO ...", annotations)
```

### 4. Cache Expensive Computations
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def compute_embedding(text: str) -> np.ndarray:
    # Expensive embedding computation
    return embeddings_model.encode(text)
```

---

## Testing Utilities

### Generate Synthetic Data

```python
def generate_test_annotations(
    n_samples: int = 1000,
    n_annotators: int = 3,
    agreement_level: float = 0.8
) -> np.ndarray:
    """
    Generate synthetic annotations for testing.

    Args:
        n_samples: Number of items
        n_annotators: Number of annotators
        agreement_level: Proportion of agreeing annotations

    Returns:
        [n_samples x n_annotators] reliability matrix
    """
    # True labels
    true_labels = np.random.randint(0, 3, n_samples)

    # Generate annotations
    annotations = np.zeros((n_samples, n_annotators), dtype=int)

    for i in range(n_samples):
        for j in range(n_annotators):
            if np.random.rand() < agreement_level:
                # Agree with true label
                annotations[i, j] = true_labels[i]
            else:
                # Random disagreement
                annotations[i, j] = np.random.randint(0, 3)

    return annotations
```

### Mock Drift Scenario

```python
def generate_drift_scenario() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate baseline and drifted distributions.

    Returns:
        (baseline, drifted)
    """
    # Baseline: Normal(0, 1)
    baseline = np.random.normal(0, 1, 1000)

    # Drifted: Normal(0.5, 1.2) - shifted mean and variance
    drifted = np.random.normal(0.5, 1.2, 1000)

    return baseline, drifted
```

---

## Common Pitfalls

### 1. IAA Calculation
❌ **Wrong:** Using Cohen's kappa with >2 annotators
✅ **Correct:** Use Krippendorff's alpha for multi-annotator scenarios

### 2. Calibration
❌ **Wrong:** Calibrating on training set
✅ **Correct:** Always calibrate on separate validation set

### 3. Drift Detection
❌ **Wrong:** Checking drift after every single sample
✅ **Correct:** Use sliding windows (e.g., every 100 samples)

### 4. Routing
❌ **Wrong:** Fixed thresholds across all tasks
✅ **Correct:** Adjust thresholds based on task complexity and cost constraints

---

## References

1. **Krippendorff's Alpha:** Krippendorff (2011) "Computing Krippendorff's Alpha-Reliability"
2. **Temperature Scaling:** Guo et al. (ICML 2017) "On Calibration of Modern Neural Networks"
3. **PSI:** Credit scoring industry standard (SAS, 2000s)
4. **Drift Detection:** Rabanser et al. (2019) "Failing Loudly: Methods for Detecting Dataset Shift"

---

**Last Updated:** 2025-10-07
**Maintainer:** Analyst Agent - Hive Mind Collective
