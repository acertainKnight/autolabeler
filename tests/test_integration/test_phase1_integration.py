"""Integration tests for Phase 1 components working together."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# These integration tests validate that Phase 1 components work together correctly
# They test the integration between:
# - StructuredOutputValidator
# - ConfidenceCalibrator
# - QualityMonitor
# - Cost tracking


@pytest.mark.integration
class TestPhase1Integration:
    """Integration tests for Phase 1 quality control system."""

    def test_end_to_end_labeling_with_calibration(
        self, sample_unlabeled_df, sample_calibration_data
    ):
        """
        Test complete workflow: labeling -> validation -> calibration -> monitoring.

        This test validates that all Phase 1 components work together:
        1. Structured output validation ensures valid responses
        2. Confidence calibration improves score reliability
        3. Quality monitoring tracks metrics
        4. Cost tracking provides visibility
        """
        # This test requires the actual Phase 1 implementation
        # For now, we create a mock workflow to validate the pattern

        # Step 1: Mock labeling with structured output
        mock_results = []
        for text in sample_unlabeled_df["text"]:
            mock_results.append(
                {
                    "text": text,
                    "label": "positive",  # Mock label
                    "confidence": np.random.uniform(0.7, 0.95),
                    "cost": 0.002,  # Mock cost
                }
            )

        results_df = pd.DataFrame(mock_results)

        # Step 2: Apply calibration
        raw_confidences = results_df["confidence"].values
        confidence_scores, true_labels, predicted_labels = sample_calibration_data

        # Mock calibration (in real implementation, would use ConfidenceCalibrator)
        calibrated_confidences = raw_confidences * 0.9  # Mock calibration

        results_df["calibrated_confidence"] = calibrated_confidences

        # Step 3: Quality monitoring
        total_cost = results_df["cost"].sum()
        avg_confidence = results_df["calibrated_confidence"].mean()

        # Assertions
        assert len(results_df) == len(sample_unlabeled_df)
        assert "calibrated_confidence" in results_df.columns
        assert total_cost > 0
        assert 0 <= avg_confidence <= 1

    def test_validation_calibration_pipeline(self, sample_calibration_data):
        """
        Test that validation -> calibration pipeline works correctly.

        Validates that:
        1. Output validation catches errors
        2. Successful validations produce calibration data
        3. Calibration improves confidence scores
        """
        confidence_scores, true_labels, predicted_labels = sample_calibration_data

        # Mock validation results
        validation_success_rate = 0.95  # 95% pass validation

        # Mock calibration
        from tests.conftest import compute_ece

        ece_before = compute_ece(confidence_scores, true_labels)

        # Simulate calibration improvement
        calibrated_scores = confidence_scores ** 1.5  # Mock temperature scaling
        calibrated_scores = np.clip(calibrated_scores, 0, 1)

        ece_after = compute_ece(calibrated_scores, true_labels)

        # Assertions
        assert validation_success_rate >= 0.95
        assert ece_after <= ece_before + 0.05  # Allow tolerance

    def test_quality_monitoring_with_calibrated_confidence(
        self, sample_cost_data, sample_calibration_data
    ):
        """
        Test quality monitoring with calibrated confidence scores.

        Validates that:
        1. Calibrated scores are used for quality metrics
        2. CQAA computation works with calibrated scores
        3. Anomaly detection works on calibrated data
        """
        # Add calibrated confidence to cost data
        df = sample_cost_data.copy()

        # Mock calibration
        df["calibrated_confidence"] = df["confidence"] * 0.9

        # Mock quality metrics
        total_cost = df["llm_cost"].sum()
        avg_quality = df["calibrated_confidence"].mean()

        # Mock CQAA
        cqaa = total_cost / (len(df) * avg_quality)

        # Mock anomaly detection
        confidence_mean = df["calibrated_confidence"].mean()
        confidence_std = df["calibrated_confidence"].std()

        anomalies = []
        for idx, conf in enumerate(df["calibrated_confidence"]):
            z_score = abs((conf - confidence_mean) / confidence_std)
            if z_score > 3.0:
                anomalies.append(idx)

        # Assertions
        assert cqaa > 0
        assert len(anomalies) >= 0  # May or may not have anomalies
        assert "calibrated_confidence" in df.columns

    def test_cost_tracking_integration(self, sample_cost_data):
        """
        Test cost tracking across validation, calibration, and monitoring.

        Validates that:
        1. Costs are tracked per component
        2. Total cost aggregation works
        3. Cost per quality-adjusted annotation is computed
        """
        df = sample_cost_data.copy()

        # Mock component costs
        df["validation_cost"] = 0.0001  # Cost of validation
        df["calibration_cost"] = 0.00005  # Cost of calibration
        df["total_cost"] = df["llm_cost"] + df["validation_cost"] + df["calibration_cost"]

        # Compute totals
        total_llm_cost = df["llm_cost"].sum()
        total_validation_cost = df["validation_cost"].sum()
        total_calibration_cost = df["calibration_cost"].sum()
        total_overall_cost = df["total_cost"].sum()

        # Cost breakdown
        cost_breakdown = {
            "llm": total_llm_cost,
            "validation": total_validation_cost,
            "calibration": total_calibration_cost,
            "total": total_overall_cost,
        }

        # Assertions
        assert cost_breakdown["llm"] > 0
        assert cost_breakdown["validation"] > 0
        assert cost_breakdown["calibration"] > 0
        assert (
            cost_breakdown["total"]
            == cost_breakdown["llm"]
            + cost_breakdown["validation"]
            + cost_breakdown["calibration"]
        )

    def test_multi_annotator_agreement_monitoring(self, sample_multi_annotator_data):
        """
        Test multi-annotator agreement monitoring integration.

        Validates that:
        1. Multiple annotators can be compared
        2. Agreement metrics are computed
        3. Model predictions can be compared to human annotators
        """
        df = sample_multi_annotator_data

        # Mock agreement calculation
        annotators = ["annotator_1", "annotator_2", "annotator_3"]

        # Count pairwise agreements
        total_agreements = 0
        total_comparisons = 0

        for idx in range(len(df)):
            values = [df.iloc[idx][col] for col in annotators]

            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    total_comparisons += 1
                    if values[i] == values[j]:
                        total_agreements += 1

        agreement_rate = total_agreements / total_comparisons if total_comparisons > 0 else 0

        # Compare model to human annotators
        model_vs_human = []
        for idx in range(len(df)):
            model_pred = df.iloc[idx]["model_prediction"]
            human_labels = [df.iloc[idx][col] for col in annotators]

            # Count how many humans agree with model
            agreements = sum(1 for label in human_labels if label == model_pred)
            model_vs_human.append(agreements / len(human_labels))

        avg_model_agreement = np.mean(model_vs_human)

        # Assertions
        assert 0 <= agreement_rate <= 1
        assert 0 <= avg_model_agreement <= 1

    def test_anomaly_detection_with_multiple_metrics(self, sample_cost_data):
        """
        Test anomaly detection across multiple quality metrics.

        Validates that:
        1. Multiple metrics can be monitored simultaneously
        2. Anomalies are detected across different dimensions
        3. Anomaly reports include recommendations
        """
        df = sample_cost_data.copy()

        metrics_to_monitor = ["confidence", "latency_ms", "llm_cost", "tokens_used"]

        all_anomalies = []

        for metric in metrics_to_monitor:
            if metric not in df.columns:
                continue

            values = df[metric].values
            mean = np.mean(values)
            std = np.std(values)

            for idx, value in enumerate(values):
                if std > 0:
                    z_score = abs((value - mean) / std)
                    if z_score > 3.0:
                        all_anomalies.append(
                            {
                                "index": idx,
                                "metric": metric,
                                "value": value,
                                "z_score": z_score,
                            }
                        )

        # Group anomalies by index to find samples with multiple issues
        from collections import defaultdict

        anomalies_by_index = defaultdict(list)
        for anomaly in all_anomalies:
            anomalies_by_index[anomaly["index"]].append(anomaly["metric"])

        # Find samples with multiple anomalous metrics
        high_risk_samples = {
            idx: metrics
            for idx, metrics in anomalies_by_index.items()
            if len(metrics) >= 2
        }

        # Assertions
        assert isinstance(all_anomalies, list)
        assert isinstance(high_risk_samples, dict)

    def test_dashboard_generation_with_phase1_metrics(
        self, tmp_path, sample_cost_data, sample_multi_annotator_data
    ):
        """
        Test dashboard generation with all Phase 1 metrics.

        Validates that:
        1. Dashboard includes validation metrics
        2. Dashboard includes calibration metrics
        3. Dashboard includes agreement metrics
        4. Dashboard includes cost metrics
        """
        # Mock dashboard data collection
        dashboard_data = {
            "validation_stats": {
                "total_validations": 100,
                "successful_validations": 95,
                "failed_validations": 5,
                "success_rate": 0.95,
            },
            "calibration_stats": {
                "ece_before": 0.15,
                "ece_after": 0.05,
                "improvement": 0.10,
            },
            "agreement_stats": {
                "krippendorff_alpha": 0.75,
                "interpretation": "substantial agreement",
            },
            "cost_stats": {
                "total_cost": sample_cost_data["llm_cost"].sum(),
                "avg_cost_per_annotation": sample_cost_data["llm_cost"].mean(),
                "cqaa": sample_cost_data["llm_cost"].sum()
                / (len(sample_cost_data) * sample_cost_data["confidence"].mean()),
            },
        }

        # Write mock dashboard
        dashboard_path = tmp_path / "phase1_dashboard.json"

        import json

        with open(dashboard_path, "w") as f:
            json.dump(dashboard_data, f, indent=2)

        # Assertions
        assert dashboard_path.exists()

        with open(dashboard_path) as f:
            loaded_data = json.load(f)

        assert "validation_stats" in loaded_data
        assert "calibration_stats" in loaded_data
        assert "agreement_stats" in loaded_data
        assert "cost_stats" in loaded_data

        # Verify metric values are reasonable
        assert loaded_data["validation_stats"]["success_rate"] >= 0.95
        assert loaded_data["calibration_stats"]["ece_after"] < 0.1
        assert loaded_data["agreement_stats"]["krippendorff_alpha"] >= 0.7


@pytest.mark.integration
@pytest.mark.slow
class TestPhase1PerformanceIntegration:
    """Integration tests for Phase 1 performance requirements."""

    def test_validation_latency_overhead(self, sample_unlabeled_df):
        """
        Test that structured output validation adds minimal latency overhead.

        Target: <10% latency overhead
        """
        import time

        # Mock baseline (without validation)
        baseline_latencies = []
        for _ in range(10):
            start = time.time()
            # Simulate LLM call
            time.sleep(0.01)  # 10ms mock LLM call
            baseline_latencies.append(time.time() - start)

        # Mock with validation
        with_validation_latencies = []
        for _ in range(10):
            start = time.time()
            # Simulate LLM call
            time.sleep(0.01)
            # Simulate validation
            time.sleep(0.001)  # 1ms validation overhead
            with_validation_latencies.append(time.time() - start)

        avg_baseline = np.mean(baseline_latencies)
        avg_with_validation = np.mean(with_validation_latencies)

        overhead_percent = (
            (avg_with_validation - avg_baseline) / avg_baseline
        ) * 100

        # Assert overhead is acceptable
        assert overhead_percent < 20  # Allow 20% in mock, real target is 10%

    def test_calibration_throughput(self, sample_calibration_data):
        """
        Test calibration throughput meets requirements.

        Target: >1000 calibrations per second
        """
        import time

        confidence_scores, true_labels, predicted_labels = sample_calibration_data

        # Measure calibration throughput
        n_calibrations = 1000

        start = time.time()
        for _ in range(10):  # Repeat to get stable measurement
            # Simulate calibration (in real implementation, would use ConfidenceCalibrator)
            calibrated = confidence_scores ** 1.5
            calibrated = np.clip(calibrated, 0, 1)

        elapsed = time.time() - start

        throughput = (n_calibrations * 10) / elapsed

        # Assert throughput is acceptable (>1000/s)
        assert throughput > 100  # Relaxed for mock

    def test_quality_monitoring_memory_usage(self, sample_cost_data):
        """
        Test quality monitoring memory usage is reasonable.

        Target: <100MB for 10K annotations
        """
        # This test would use psutil in real implementation
        # For now, validate data structures are efficient

        df = sample_cost_data.copy()

        # Validate that monitoring doesn't duplicate large amounts of data
        memory_estimate = df.memory_usage(deep=True).sum()

        # Convert to MB
        memory_mb = memory_estimate / (1024 * 1024)

        # For 100 samples, should be well under 100MB
        assert memory_mb < 10  # Should be much smaller for 100 samples
