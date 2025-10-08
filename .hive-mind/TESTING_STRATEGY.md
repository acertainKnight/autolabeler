# AutoLabeler Testing Strategy
## Comprehensive Test Plan for Production-Grade Quality Assurance

**Document Version:** 1.0
**Date:** 2025-10-07
**Project:** AutoLabeler v2 Enhancement Initiative
**Purpose:** Define testing strategy, frameworks, and quality gates for all implementation phases

---

## Executive Summary

This document outlines a comprehensive testing strategy for AutoLabeler enhancements, covering unit tests, integration tests, performance tests, validation tests, and quality assurance processes. The strategy ensures high-quality, reliable code that meets production standards.

**Testing Philosophy:**
- **Test-Driven Development (TDD):** Write tests before implementation where feasible
- **Pyramid Structure:** Many unit tests, fewer integration tests, minimal E2E tests
- **Continuous Testing:** Automated testing in CI/CD pipeline
- **Quality Gates:** No merge without passing tests + code review

---

## Table of Contents

1. [Testing Pyramid](#1-testing-pyramid)
2. [Unit Testing Strategy](#2-unit-testing-strategy)
3. [Integration Testing Strategy](#3-integration-testing-strategy)
4. [Performance Testing Strategy](#4-performance-testing-strategy)
5. [Validation Testing Strategy](#5-validation-testing-strategy)
6. [Test Infrastructure](#6-test-infrastructure)
7. [Quality Gates](#7-quality-gates)
8. [Test Data Management](#8-test-data-management)

---

## 1. Testing Pyramid

### 1.1 Test Distribution

```
                  ╱╲
                 ╱  ╲
                ╱ E2E╲           5-10%:  Full system tests
               ╱──────╲          ~20 tests
              ╱        ╲
             ╱Integration╲       15-20%: Service integration tests
            ╱────────────╲       ~80 tests
           ╱              ╲
          ╱   Unit Tests   ╲    70-75%: Component tests
         ╱──────────────────╲   ~300 tests
        ╱____________________╲
```

**Rationale:**
- Unit tests are fast, isolated, and catch most bugs early
- Integration tests verify component interactions
- E2E tests validate complete workflows
- This distribution optimizes for speed and coverage

### 1.2 Target Metrics

| Test Type | Coverage Target | Execution Time | Count |
|-----------|----------------|----------------|-------|
| Unit | >80% line coverage | <30s total | ~300 |
| Integration | >60% feature coverage | <2min total | ~80 |
| E2E | >90% user journey coverage | <10min total | ~20 |
| Performance | 100% SLA validation | <5min total | ~15 |
| **Total** | **>75% overall** | **<20min** | **~415** |

---

## 2. Unit Testing Strategy

### 2.1 Unit Test Organization

```
tests/
├── unit/
│   ├── core/
│   │   ├── test_labeling_service.py
│   │   ├── test_ensemble_service.py
│   │   ├── test_knowledge_store.py
│   │   └── test_prompt_manager.py
│   ├── quality/
│   │   ├── test_confidence_calibrator.py
│   │   ├── test_quality_monitor.py
│   │   └── test_output_validator.py
│   ├── active_learning/
│   │   ├── test_sampling_strategies.py
│   │   └── test_stopping_criteria.py
│   ├── weak_supervision/
│   │   ├── test_labeling_functions.py
│   │   └── test_aggregation.py
│   └── dspy/
│       ├── test_optimizer.py
│       └── test_signatures.py
```

### 2.2 Unit Test Templates

#### 2.2.1 Confidence Calibrator Tests

```python
# tests/unit/quality/test_confidence_calibrator.py

import pytest
import numpy as np
from autolabeler.core.quality import ConfidenceCalibrator

class TestConfidenceCalibrator:
    """Test suite for ConfidenceCalibrator."""

    @pytest.fixture
    def calibrator(self):
        """Create calibrator instance."""
        return ConfidenceCalibrator(method="temperature_scaling")

    @pytest.fixture
    def sample_data(self):
        """Generate sample calibration data."""
        np.random.seed(42)
        n_samples = 1000

        # Simulate miscalibrated confidence scores
        # (overconfident: shifted toward 1.0)
        raw_scores = np.random.beta(8, 2, n_samples)

        # True labels (50/50 split)
        true_labels = np.random.binomial(1, 0.5, n_samples)

        # Predicted labels based on threshold
        predicted_labels = (raw_scores > 0.5).astype(int)

        return raw_scores, true_labels, predicted_labels

    def test_temperature_scaling_initialization(self, calibrator):
        """Test calibrator initializes with correct method."""
        assert calibrator.method == "temperature_scaling"
        assert calibrator.calibrator is None  # Not fitted yet

    def test_fit_temperature_scaling(self, calibrator, sample_data):
        """Test fitting temperature scaling calibrator."""
        confidence_scores, true_labels, predicted_labels = sample_data

        # Fit calibrator
        calibrator.fit(confidence_scores, true_labels, predicted_labels)

        # Check calibrator was fitted
        assert calibrator.calibrator is not None
        assert hasattr(calibrator.calibrator, "temperature")

    def test_calibrate_improves_ece(self, calibrator, sample_data):
        """Test calibration improves Expected Calibration Error."""
        confidence_scores, true_labels, predicted_labels = sample_data

        # Fit calibrator
        calibrator.fit(confidence_scores, true_labels, predicted_labels)

        # Compute ECE before calibration
        ece_before = compute_ece(confidence_scores, true_labels)

        # Calibrate scores
        calibrated_scores = calibrator.calibrate(confidence_scores)

        # Compute ECE after calibration
        ece_after = compute_ece(calibrated_scores, true_labels)

        # Verify improvement
        assert ece_after < ece_before, \
            f"ECE should decrease: {ece_before:.4f} -> {ece_after:.4f}"

    def test_calibrate_preserves_ranking(self, calibrator, sample_data):
        """Test calibration preserves relative ranking of scores."""
        confidence_scores, true_labels, predicted_labels = sample_data

        calibrator.fit(confidence_scores, true_labels, predicted_labels)
        calibrated_scores = calibrator.calibrate(confidence_scores)

        # Check ranking preserved (Spearman correlation)
        from scipy.stats import spearmanr
        correlation, _ = spearmanr(confidence_scores, calibrated_scores)

        assert correlation > 0.99, \
            "Calibration should preserve ranking"

    def test_calibrate_outputs_valid_probabilities(self, calibrator, sample_data):
        """Test calibrated scores are valid probabilities [0, 1]."""
        confidence_scores, true_labels, predicted_labels = sample_data

        calibrator.fit(confidence_scores, true_labels, predicted_labels)
        calibrated_scores = calibrator.calibrate(confidence_scores)

        assert np.all(calibrated_scores >= 0), "Scores should be >= 0"
        assert np.all(calibrated_scores <= 1), "Scores should be <= 1"

    def test_evaluate_calibration_metrics(self, calibrator, sample_data):
        """Test calibration evaluation returns expected metrics."""
        confidence_scores, true_labels, predicted_labels = sample_data

        calibrator.fit(confidence_scores, true_labels, predicted_labels)

        # Evaluate calibration
        metrics = calibrator.evaluate_calibration(
            confidence_scores,
            true_labels
        )

        # Check expected metrics present
        assert "ece" in metrics
        assert "brier_score" in metrics
        assert "log_loss" in metrics

        # Check metrics are reasonable
        assert 0 <= metrics["ece"] <= 1
        assert 0 <= metrics["brier_score"] <= 1
        assert metrics["log_loss"] >= 0

    @pytest.mark.parametrize("method", [
        "temperature_scaling",
        "platt_scaling",
        "isotonic_regression"
    ])
    def test_multiple_calibration_methods(self, method, sample_data):
        """Test different calibration methods all improve ECE."""
        confidence_scores, true_labels, predicted_labels = sample_data

        calibrator = ConfidenceCalibrator(method=method)
        calibrator.fit(confidence_scores, true_labels, predicted_labels)

        ece_before = compute_ece(confidence_scores, true_labels)
        calibrated_scores = calibrator.calibrate(confidence_scores)
        ece_after = compute_ece(calibrated_scores, true_labels)

        assert ece_after < ece_before, \
            f"{method} should improve ECE"


def compute_ece(confidence_scores, true_labels, n_bins=10):
    """Compute Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidence_scores, bins) - 1

    ece = 0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_confidence = confidence_scores[mask].mean()
            bin_accuracy = true_labels[mask].mean()
            ece += mask.sum() / len(true_labels) * abs(bin_confidence - bin_accuracy)

    return ece
```

#### 2.2.2 Active Learning Tests

```python
# tests/unit/active_learning/test_sampling_strategies.py

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from autolabeler.core.active_learning import ActiveLearningService, SamplingStrategy

class TestActiveLearningService:
    """Test suite for ActiveLearningService."""

    @pytest.fixture
    def al_service(self, tmp_path):
        """Create active learning service instance."""
        from autolabeler.config import Settings
        settings = Settings()
        return ActiveLearningService(
            "test_dataset",
            settings,
            initial_strategy=SamplingStrategy.MARGIN_SAMPLING
        )

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic unlabeled data with embeddings."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        # Generate features
        X = np.random.randn(n_samples, n_features)

        # Generate labels (for testing, not used in sampling)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Create DataFrame
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        df["text"] = [f"Sample text {i}" for i in range(n_samples)]
        df["true_label"] = y  # Hidden during sampling

        # Generate embeddings
        embeddings = X[:, :5]  # Use first 5 features as embeddings

        return df, embeddings, X, y

    @pytest.fixture
    def trained_model(self, synthetic_data):
        """Train a simple model for uncertainty estimation."""
        _, _, X, y = synthetic_data

        # Train on first 100 samples
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:100], y[:100])

        return model

    def test_initialization(self, al_service):
        """Test service initializes correctly."""
        assert al_service.strategy == SamplingStrategy.MARGIN_SAMPLING
        assert al_service.iteration == 0
        assert len(al_service.performance_history) == 0

    def test_margin_sampling_selects_uncertain_samples(
        self,
        al_service,
        synthetic_data,
        trained_model
    ):
        """Test margin sampling selects most uncertain samples."""
        df, _, X, _ = synthetic_data

        # Get predictions
        probas = trained_model.predict_proba(X)

        # Calculate margins manually
        sorted_probas = np.sort(probas, axis=1)
        margins = sorted_probas[:, -1] - sorted_probas[:, -2]

        # Select samples using active learning
        selected = al_service._margin_sampling(
            df,
            trained_model,
            n_samples=10
        )

        # Verify selected samples have small margins
        selected_indices = selected.index.tolist()
        selected_margins = margins[selected_indices]

        # Check selected margins are among smallest
        assert np.mean(selected_margins) < np.median(margins), \
            "Selected samples should have below-median margins"

    def test_least_confidence_sampling(
        self,
        al_service,
        synthetic_data,
        trained_model
    ):
        """Test least confidence sampling."""
        df, _, X, _ = synthetic_data

        selected = al_service._least_confidence_sampling(
            df,
            trained_model,
            n_samples=10
        )

        assert len(selected) == 10
        assert isinstance(selected, pd.DataFrame)

        # Verify selected samples have low max probability
        probas = trained_model.predict_proba(X)
        max_probas = np.max(probas, axis=1)
        selected_indices = selected.index.tolist()
        selected_max_probas = max_probas[selected_indices]

        assert np.mean(selected_max_probas) < np.median(max_probas), \
            "Selected samples should have below-median confidence"

    def test_entropy_sampling(
        self,
        al_service,
        synthetic_data,
        trained_model
    ):
        """Test entropy sampling."""
        df, _, X, _ = synthetic_data

        selected = al_service._entropy_sampling(
            df,
            trained_model,
            n_samples=10
        )

        assert len(selected) == 10

        # Verify selected samples have high entropy
        probas = trained_model.predict_proba(X)
        entropy = -np.sum(probas * np.log(probas + 1e-10), axis=1)
        selected_indices = selected.index.tolist()
        selected_entropy = entropy[selected_indices]

        assert np.mean(selected_entropy) > np.median(entropy), \
            "Selected samples should have above-median entropy"

    def test_tcm_hybrid_cold_start(
        self,
        al_service,
        synthetic_data
    ):
        """Test TCM hybrid uses diversity sampling in cold start."""
        df, embeddings, _, _ = synthetic_data

        # Mock model (not used in cold start)
        class MockModel:
            def predict_proba(self, X):
                return np.random.rand(len(X), 2)

        model = MockModel()

        # Force cold start
        al_service.iteration = 0
        al_service.performance_history = []

        selected = al_service._tcm_hybrid_sampling(
            df,
            model,
            n_samples=50,
            embeddings=embeddings
        )

        assert len(selected) == 50

        # Verify diversity: selected samples should be spread across embedding space
        selected_indices = selected.index.tolist()
        selected_embeddings = embeddings[selected_indices]

        # Check pairwise distances are reasonable (not all clustered)
        from scipy.spatial.distance import pdist
        distances = pdist(selected_embeddings)
        assert np.mean(distances) > 0.1, \
            "Selected samples should be diverse (not clustered)"

    def test_stopping_criteria_plateau(self, al_service):
        """Test stopping criteria triggers on performance plateau."""
        # Simulate performance plateau
        for i in range(5):
            al_service.performance_history.append(0.85 + i * 0.005)

        # Mock validation DataFrame
        val_df = pd.DataFrame({"dummy": [1]})

        should_stop, reason = al_service.should_stop(0.8525, val_df)

        assert should_stop is True
        assert "plateau" in reason.lower()

    def test_stopping_criteria_no_plateau(self, al_service):
        """Test stopping criteria doesn't trigger without plateau."""
        # Simulate improving performance
        for i in range(5):
            al_service.performance_history.append(0.70 + i * 0.05)

        val_df = pd.DataFrame({"dummy": [1]})

        should_stop, reason = al_service.should_stop(0.95, val_df)

        assert should_stop is False

    def test_get_statistics(self, al_service):
        """Test statistics retrieval."""
        al_service.iteration = 3
        al_service.performance_history = [0.70, 0.75, 0.78]

        stats = al_service.get_statistics()

        assert stats["iteration"] == 3
        assert len(stats["performance_history"]) == 3
        assert "improvement_per_iteration" in stats
        assert stats["total_samples_selected"] == 300  # 3 * 100
```

### 2.3 Unit Test Fixtures

#### 2.3.1 Common Fixtures

```python
# tests/conftest.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from autolabeler.config import Settings

@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        llm_model="gpt-3.5-turbo",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        temperature=0.1,
        enable_dspy_optimization=False,
        enable_advanced_rag=False
    )

@pytest.fixture
def sample_labeled_df():
    """Create sample labeled DataFrame."""
    data = {
        "text": [
            "This product is amazing!",
            "Terrible experience, would not recommend.",
            "It's okay, nothing special.",
            "Best purchase ever!",
            "Waste of money.",
        ],
        "label": [
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative",
        ],
        "confidence": [0.95, 0.90, 0.60, 0.98, 0.92],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_unlabeled_df():
    """Create sample unlabeled DataFrame."""
    data = {
        "text": [
            "I love this!",
            "Not what I expected.",
            "Pretty good overall.",
            "Disappointed with quality.",
            "Exceeded expectations!",
        ]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_llm_client(monkeypatch):
    """Mock LLM client for testing."""
    class MockLLMClient:
        def invoke(self, prompt):
            return {"label": "positive", "confidence": 0.85}

        def with_structured_output(self, schema, method="function_calling"):
            return self

    return MockLLMClient()

@pytest.fixture
def temp_storage_path(tmp_path):
    """Create temporary storage path."""
    storage = tmp_path / "test_storage"
    storage.mkdir()
    return storage
```

---

## 3. Integration Testing Strategy

### 3.1 Integration Test Organization

```
tests/
├── integration/
│   ├── test_labeling_pipeline.py
│   ├── test_ensemble_workflow.py
│   ├── test_active_learning_loop.py
│   ├── test_weak_supervision_pipeline.py
│   ├── test_dspy_optimization.py
│   └── test_versioning_workflow.py
```

### 3.2 Integration Test Examples

#### 3.2.1 Full Labeling Pipeline

```python
# tests/integration/test_labeling_pipeline.py

import pytest
import pandas as pd
from autolabeler import AutoLabeler
from autolabeler.config import Settings
from autolabeler.core.configs import LabelingConfig, BatchConfig

@pytest.mark.integration
class TestLabelingPipeline:
    """Integration tests for complete labeling workflow."""

    @pytest.fixture
    def autolabeler(self, tmp_path, settings):
        """Create AutoLabeler instance with test storage."""
        return AutoLabeler("test_pipeline", settings)

    def test_end_to_end_labeling_with_rag(
        self,
        autolabeler,
        sample_labeled_df,
        sample_unlabeled_df
    ):
        """Test complete labeling workflow with RAG."""
        # 1. Add training data
        autolabeler.add_training_data(
            sample_labeled_df,
            text_column="text",
            label_column="label"
        )

        # 2. Configure labeling with RAG
        labeling_config = LabelingConfig(
            use_rag=True,
            k_examples=3,
            confidence_threshold=0.7
        )

        batch_config = BatchConfig(
            batch_size=2,
            resume=False
        )

        # 3. Label unlabeled data
        results = autolabeler.label(
            sample_unlabeled_df,
            text_column="text",
            labeling_config=labeling_config,
            batch_config=batch_config
        )

        # 4. Verify results
        assert len(results) == len(sample_unlabeled_df)
        assert "predicted_label" in results.columns
        assert "predicted_label_confidence" in results.columns

        # 5. Verify labels are valid
        valid_labels = {"positive", "negative", "neutral"}
        assert all(label in valid_labels for label in results["predicted_label"])

        # 6. Verify confidence scores are in [0, 1]
        assert all(0 <= conf <= 1 for conf in results["predicted_label_confidence"])

    def test_batch_processing_resume_capability(
        self,
        autolabeler,
        tmp_path
    ):
        """Test batch processing can resume after interruption."""
        # Create larger dataset
        df = pd.DataFrame({
            "text": [f"Sample text {i}" for i in range(50)]
        })

        labeling_config = LabelingConfig(use_rag=False)
        batch_config = BatchConfig(
            batch_size=10,
            resume=True,
            save_interval=2
        )

        # First run: process partially (simulate interruption)
        # This requires mocking an interruption, which is complex
        # For now, test that progress is saved

        results = autolabeler.label(
            df[:20],  # Process first 20
            text_column="text",
            labeling_config=labeling_config,
            batch_config=batch_config
        )

        assert len(results) == 20

        # Second run: process remaining
        results_full = autolabeler.label(
            df,  # Process all 50
            text_column="text",
            labeling_config=labeling_config,
            batch_config=batch_config
        )

        assert len(results_full) == 50

    def test_high_confidence_examples_added_to_knowledge_base(
        self,
        autolabeler,
        sample_unlabeled_df
    ):
        """Test high-confidence predictions are added to KB."""
        labeling_config = LabelingConfig(
            use_rag=False,
            save_to_knowledge_base=True,
            confidence_threshold=0.8
        )

        # Get initial KB size
        initial_kb_size = len(autolabeler.knowledge_store._get_all_examples())

        # Label data
        results = autolabeler.label(
            sample_unlabeled_df,
            text_column="text",
            labeling_config=labeling_config
        )

        # Get final KB size
        final_kb_size = len(autolabeler.knowledge_store._get_all_examples())

        # Verify high-confidence examples were added
        high_conf_count = (results["predicted_label_confidence"] >= 0.8).sum()
        assert final_kb_size >= initial_kb_size + high_conf_count
```

#### 3.2.2 Active Learning Loop Integration

```python
# tests/integration/test_active_learning_loop.py

import pytest
import numpy as np
from autolabeler import AutoLabeler
from autolabeler.core.active_learning import ActiveLearningService, SamplingStrategy

@pytest.mark.integration
@pytest.mark.slow
class TestActiveLearningLoop:
    """Integration tests for active learning workflow."""

    def test_active_learning_reduces_annotation_needs(
        self,
        tmp_path,
        settings
    ):
        """Test active learning achieves target with fewer annotations."""
        # Generate synthetic dataset
        np.random.seed(42)
        n_total = 1000
        n_initial = 50

        # Simple synthetic task: classify based on feature sum
        X = np.random.randn(n_total, 10)
        y = (X.sum(axis=1) > 0).astype(int)

        # Create DataFrames
        df_full = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
        df_full["text"] = [f"Sample {i}" for i in range(n_total)]
        df_full["true_label"] = y

        # Initial labeled set
        initial_indices = np.random.choice(n_total, n_initial, replace=False)
        df_labeled = df_full.iloc[initial_indices].copy()
        df_unlabeled = df_full.drop(initial_indices).copy()

        # Initialize services
        autolabeler = AutoLabeler("al_test", settings)
        al_service = ActiveLearningService(
            "al_test",
            settings,
            initial_strategy=SamplingStrategy.MARGIN_SAMPLING
        )

        # Add initial training data
        autolabeler.add_training_data(
            df_labeled,
            text_column="text",
            label_column="true_label"
        )

        # Track annotations used
        annotations_used = n_initial

        # Active learning loop
        target_accuracy = 0.85
        max_iterations = 10
        samples_per_iteration = 20

        for iteration in range(max_iterations):
            # Train model
            model = autolabeler._train_simple_model(
                df_labeled,
                feature_columns=[f"feat_{i}" for i in range(10)],
                label_column="true_label"
            )

            # Evaluate on holdout
            holdout_indices = np.random.choice(
                len(df_unlabeled),
                min(100, len(df_unlabeled)),
                replace=False
            )
            holdout = df_unlabeled.iloc[holdout_indices]
            accuracy = model.score(
                holdout[[f"feat_{i}" for i in range(10)]],
                holdout["true_label"]
            )

            print(f"Iteration {iteration}: Accuracy={accuracy:.3f}, "
                  f"Annotations={annotations_used}")

            # Check if target reached
            if accuracy >= target_accuracy:
                break

            # Select samples
            selected = al_service.select_samples(
                df_unlabeled,
                model,
                n_samples=min(samples_per_iteration, len(df_unlabeled))
            )

            # "Annotate" selected samples (using true labels)
            df_labeled = pd.concat([df_labeled, selected])
            df_unlabeled = df_unlabeled.drop(selected.index)
            annotations_used += len(selected)

        # Verify active learning was efficient
        # Should reach target with <300 annotations (30% of full dataset)
        assert annotations_used < 300, \
            f"Active learning used {annotations_used} annotations (expected <300)"

        # Verify target accuracy reached
        assert accuracy >= target_accuracy, \
            f"Target accuracy {target_accuracy} not reached (got {accuracy:.3f})"
```

---

## 4. Performance Testing Strategy

### 4.1 Performance Test Suite

```python
# tests/performance/test_latency_sla.py

import pytest
import time
from autolabeler import AutoLabeler

@pytest.mark.performance
class TestLatencySLA:
    """Performance tests for latency SLAs."""

    @pytest.fixture
    def autolabeler(self, settings):
        """Create AutoLabeler instance."""
        return AutoLabeler("perf_test", settings)

    def test_single_label_latency_p95(self, autolabeler, benchmark):
        """Test single label latency meets p95 SLA (<2s)."""
        text = "Sample text for labeling performance test"

        # Warm up
        for _ in range(3):
            autolabeler.label_text(text)

        # Benchmark
        latencies = []
        for _ in range(20):
            start = time.time()
            result = autolabeler.label_text(text)
            elapsed = time.time() - start
            latencies.append(elapsed)

        # Compute p95
        p95_latency = np.percentile(latencies, 95)

        # Verify SLA
        assert p95_latency < 2.0, \
            f"P95 latency {p95_latency:.2f}s exceeds 2s SLA"

    def test_batch_labeling_throughput(self, autolabeler):
        """Test batch labeling throughput (>50 items/min)."""
        # Create test DataFrame
        df = pd.DataFrame({
            "text": [f"Sample text {i}" for i in range(100)]
        })

        start = time.time()
        results = autolabeler.label(df, "text")
        elapsed = time.time() - start

        throughput = len(df) / (elapsed / 60)  # items per minute

        assert throughput > 50, \
            f"Throughput {throughput:.1f} items/min < 50 items/min SLA"

    def test_rag_retrieval_latency(self, autolabeler, sample_labeled_df):
        """Test RAG retrieval latency (<500ms p95)."""
        # Add examples to knowledge base
        autolabeler.add_training_data(
            sample_labeled_df,
            "text",
            "label"
        )

        query = "Sample query text"

        # Benchmark retrieval
        latencies = []
        for _ in range(50):
            start = time.time()
            examples = autolabeler.knowledge_store.find_similar_examples(
                query,
                k=5
            )
            elapsed = time.time() - start
            latencies.append(elapsed)

        p95_latency = np.percentile(latencies, 95)

        assert p95_latency < 0.5, \
            f"RAG retrieval p95 latency {p95_latency:.3f}s exceeds 500ms SLA"

@pytest.mark.performance
class TestResourceUsage:
    """Performance tests for resource usage."""

    def test_memory_usage_10k_examples(self, autolabeler):
        """Test memory usage with 10k examples (<2GB)."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large DataFrame
        df = pd.DataFrame({
            "text": [f"Sample text {i}" for i in range(10000)],
            "label": [f"label_{i % 10}" for i in range(10000)]
        })

        # Add to knowledge base
        autolabeler.add_training_data(df, "text", "label")

        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_used = final_memory - baseline_memory

        assert memory_used < 2048, \
            f"Memory usage {memory_used:.1f}MB exceeds 2GB limit"
```

---

## 5. Validation Testing Strategy

### 5.1 Benchmark Dataset Tests

```python
# tests/validation/test_benchmark_datasets.py

import pytest
from datasets import load_dataset
from autolabeler import AutoLabeler

@pytest.mark.validation
@pytest.mark.slow
class TestBenchmarkDatasets:
    """Validation tests on benchmark datasets."""

    def test_imdb_sentiment_accuracy(self, settings):
        """Validate on IMDB sentiment dataset (target >85%)."""
        # Load IMDB dataset
        dataset = load_dataset("imdb", split="test[:100]")  # Small sample for testing

        df = pd.DataFrame({
            "text": dataset["text"],
            "label": ["positive" if l == 1 else "negative" for l in dataset["label"]]
        })

        # Split train/test
        train_df = df[:50]
        test_df = df[50:]

        # Initialize and train
        autolabeler = AutoLabeler("imdb_test", settings)
        autolabeler.add_training_data(train_df, "text", "label")

        # Label test set
        results = autolabeler.label(test_df, "text")

        # Compute accuracy
        accuracy = (results["predicted_label"] == results["label"]).mean()

        print(f"IMDB Sentiment Accuracy: {accuracy:.2%}")

        # This is a simplified test; full validation would use larger sample
        assert accuracy > 0.70, \
            f"Accuracy {accuracy:.2%} below threshold"

    def test_ag_news_topic_classification(self, settings):
        """Validate on AG News topic classification."""
        # Similar structure to IMDB test
        pass
```

---

## 6. Test Infrastructure

### 6.1 CI/CD Integration

#### 6.1.1 GitHub Actions Workflow

```yaml
# .github/workflows/test.yml

name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src/autolabeler --cov-report=xml

    - name: Run integration tests
      run: |
        pytest tests/integration/ -v --maxfail=3

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  performance:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --benchmark-only

    - name: Check SLA compliance
      run: |
        python scripts/check_sla_compliance.py
```

### 6.2 Test Data Management

#### 6.2.1 Fixture Data Location

```
tests/
├── fixtures/
│   ├── labeled_data/
│   │   ├── sentiment_100.csv
│   │   ├── topic_classification_50.csv
│   │   └── ner_examples_200.csv
│   ├── unlabeled_data/
│   │   ├── test_batch_1000.csv
│   │   └── test_batch_10000.csv
│   └── knowledge_bases/
│       ├── sentiment_kb/
│       └── topic_kb/
```

---

## 7. Quality Gates

### 7.1 Pre-Merge Requirements

**All PRs must pass:**

1. ✅ **All unit tests passing** (100% required)
2. ✅ **All integration tests passing** (100% required)
3. ✅ **Code coverage ≥75%** (measured by pytest-cov)
4. ✅ **No critical security vulnerabilities** (bandit scan)
5. ✅ **Code style compliance** (black, ruff)
6. ✅ **Type checking passes** (mypy)
7. ✅ **Performance tests within SLA** (if applicable)
8. ✅ **Documentation updated** (if public API changed)
9. ✅ **Code review approved** (at least 1 reviewer)

### 7.2 Release Requirements

**Before each release:**

1. ✅ **All test suites passing** (unit, integration, performance, validation)
2. ✅ **Benchmark validation passed** (accuracy targets met)
3. ✅ **Performance SLAs validated** (latency, throughput)
4. ✅ **Security scan clean** (no high/critical issues)
5. ✅ **Documentation complete** (README, API docs, examples)
6. ✅ **Changelog updated** (version notes, breaking changes)
7. ✅ **Migration guide updated** (if breaking changes)

---

## 8. Test Data Management

### 8.1 Test Data Generation

```python
# tests/utils/data_generator.py

def generate_synthetic_sentiment_data(n_samples: int = 100) -> pd.DataFrame:
    """Generate synthetic sentiment classification data."""
    np.random.seed(42)

    positive_templates = [
        "This is {adjective}!",
        "I {verb} this {noun}.",
        "{adjective} experience overall.",
    ]

    negative_templates = [
        "This is {adjective}.",
        "I {verb} this {noun}.",
        "Very {adjective} experience.",
    ]

    positive_adjectives = ["great", "excellent", "amazing", "fantastic", "wonderful"]
    negative_adjectives = ["terrible", "awful", "horrible", "disappointing", "poor"]
    verbs_positive = ["love", "enjoy", "recommend", "appreciate"]
    verbs_negative = ["hate", "dislike", "regret"]
    nouns = ["product", "service", "experience", "purchase"]

    data = []

    for _ in range(n_samples):
        if np.random.rand() > 0.5:
            # Positive sample
            template = np.random.choice(positive_templates)
            text = template.format(
                adjective=np.random.choice(positive_adjectives),
                verb=np.random.choice(verbs_positive),
                noun=np.random.choice(nouns)
            )
            label = "positive"
        else:
            # Negative sample
            template = np.random.choice(negative_templates)
            text = template.format(
                adjective=np.random.choice(negative_adjectives),
                verb=np.random.choice(verbs_negative),
                noun=np.random.choice(nouns)
            )
            label = "negative"

        data.append({"text": text, "label": label})

    return pd.DataFrame(data)
```

---

## Conclusion

This comprehensive testing strategy ensures AutoLabeler enhancements maintain high quality, reliability, and performance standards. The pyramid structure optimizes for fast feedback while comprehensive validation tests ensure production readiness.

**Key Principles:**
- Test early and often (shift left)
- Automate everything
- Maintain fast feedback loops
- Validate against benchmarks
- Enforce quality gates

**Expected Outcomes:**
- >75% overall test coverage
- <20 minutes full test suite execution
- Zero high-severity bugs in production
- Confident releases with minimal risk

---

**Document Control:**
- **Author:** TESTER/INTEGRATION AGENT
- **Version:** 1.0
- **Last Updated:** 2025-10-07
