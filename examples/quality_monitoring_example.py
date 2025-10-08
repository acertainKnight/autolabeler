"""
Example: Quality Monitoring and Confidence Calibration

This script demonstrates how to use the quality monitoring features
including confidence calibration and inter-annotator agreement analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from autolabeler.core.quality import ConfidenceCalibrator, QualityMonitor


def generate_sample_data(n_samples: int = 500) -> pd.DataFrame:
    """Generate sample annotation data for demonstration."""
    np.random.seed(42)

    # Generate true labels
    true_labels = np.random.choice(["positive", "negative", "neutral"], n_samples)

    # Simulate model predictions with some errors
    predicted_labels = true_labels.copy()
    error_mask = np.random.random(n_samples) > 0.85  # 15% error rate
    predicted_labels[error_mask] = np.random.choice(
        ["positive", "negative", "neutral"], np.sum(error_mask)
    )

    # Generate overconfident confidence scores
    confidence_scores = np.random.beta(8, 2, n_samples)

    # Simulate 3 annotators with different agreement levels
    annotator1 = true_labels.copy()
    annotator2 = true_labels.copy()
    annotator3 = true_labels.copy()

    # Add disagreements
    for annotator, error_rate in [(annotator1, 0.05), (annotator2, 0.10), (annotator3, 0.15)]:
        errors = np.random.choice(n_samples, size=int(n_samples * error_rate), replace=False)
        for idx in errors:
            annotator[idx] = np.random.choice(["positive", "negative", "neutral"])

    df = pd.DataFrame({
        "text": [f"Sample text {i}" for i in range(n_samples)],
        "true_label": true_labels,
        "predicted_label": predicted_labels,
        "confidence": confidence_scores,
        "annotator1": annotator1,
        "annotator2": annotator2,
        "annotator3": annotator3,
        "annotator_id": np.random.choice(["ann_A", "ann_B", "ann_C"], n_samples),
    })

    return df


def demo_confidence_calibration():
    """Demonstrate confidence calibration."""
    logger.info("=" * 60)
    logger.info("CONFIDENCE CALIBRATION DEMO")
    logger.info("=" * 60)

    # Generate data
    df = generate_sample_data(1000)

    # Split into train/test
    train_df = df.iloc[:700]
    test_df = df.iloc[700:]

    # Initialize calibrator
    calibrator = ConfidenceCalibrator(method="temperature", n_bins=10)

    logger.info("\n1. Fitting Temperature Scaling Calibrator...")
    calibrator.fit(
        train_df["confidence"].values,
        train_df["true_label"].values,
        train_df["predicted_label"].values,
    )

    logger.info(f"   Optimal temperature: {calibrator.temperature:.4f}")

    # Evaluate before calibration
    logger.info("\n2. Evaluating BEFORE Calibration...")
    metrics_before = calibrator.evaluate_calibration(
        test_df["confidence"].values,
        test_df["true_label"].values,
        test_df["predicted_label"].values,
        apply_calibration=False,
    )

    logger.info(f"   ECE: {metrics_before['expected_calibration_error']:.4f}")
    logger.info(f"   Brier Score: {metrics_before['brier_score']:.4f}")
    logger.info(f"   Calibration Gap: {metrics_before['calibration_gap']:.4f}")

    # Evaluate after calibration
    logger.info("\n3. Evaluating AFTER Calibration...")
    metrics_after = calibrator.evaluate_calibration(
        test_df["confidence"].values,
        test_df["true_label"].values,
        test_df["predicted_label"].values,
        apply_calibration=True,
    )

    logger.info(f"   ECE: {metrics_after['expected_calibration_error']:.4f}")
    logger.info(f"   Brier Score: {metrics_after['brier_score']:.4f}")
    logger.info(f"   Calibration Gap: {metrics_after['calibration_gap']:.4f}")

    # Show improvement
    ece_improvement = metrics_before['expected_calibration_error'] - metrics_after['expected_calibration_error']
    logger.info(f"\n4. Improvement in ECE: {ece_improvement:.4f} (lower is better)")

    # Apply calibration to predictions
    logger.info("\n5. Calibrating Test Predictions...")
    calibrated_conf = calibrator.calibrate(test_df["confidence"].values)
    logger.info(f"   Mean confidence (before): {test_df['confidence'].mean():.4f}")
    logger.info(f"   Mean confidence (after): {calibrated_conf.mean():.4f}")

    return calibrator


def demo_krippendorff_alpha():
    """Demonstrate inter-annotator agreement calculation."""
    logger.info("\n" + "=" * 60)
    logger.info("KRIPPENDORFF'S ALPHA DEMO")
    logger.info("=" * 60)

    # Generate data
    df = generate_sample_data(200)

    # Initialize monitor
    monitor = QualityMonitor(dataset_name="demo_sentiment")

    logger.info("\n1. Calculating Krippendorff's Alpha...")
    result = monitor.calculate_krippendorff_alpha(
        df,
        ["annotator1", "annotator2", "annotator3"],
    )

    logger.info(f"   Krippendorff's Alpha: {result['alpha']:.4f}")
    logger.info(f"   Number of Annotators: {result['n_annotators']}")
    logger.info(f"   Number of Items: {result['n_items']}")
    logger.info(f"   Mean Pairwise Agreement: {result['mean_pairwise_agreement']:.4f}")

    # Interpretation
    logger.info("\n2. Interpretation:")
    if result['alpha'] > 0.8:
        logger.info("   ✅ EXCELLENT - Data is highly reliable for decision-making")
    elif result['alpha'] > 0.67:
        logger.info("   ⚠️  GOOD - Data is tentatively reliable, consider review")
    elif result['alpha'] > 0.60:
        logger.info("   ⚠️  MODERATE - Some concerns about data quality")
    else:
        logger.info("   ❌ POOR - Data quality issues, review annotation guidelines")

    # Per-item agreement
    logger.info("\n3. Item-level Agreement:")
    logger.info(f"   Items with full agreement: {result['items_with_full_agreement']}")
    logger.info(f"   Items with disagreement: {result['items_with_disagreement']}")
    logger.info(f"   Mean per-item agreement: {result['per_item_agreement_mean']:.4f}")

    # Pairwise agreement
    logger.info("\n4. Pairwise Agreement:")
    for pair, agreement in result['pairwise_agreement'].items():
        logger.info(f"   {pair}: {agreement:.4f}")

    return monitor


def demo_annotator_tracking():
    """Demonstrate per-annotator performance tracking."""
    logger.info("\n" + "=" * 60)
    logger.info("ANNOTATOR PERFORMANCE TRACKING DEMO")
    logger.info("=" * 60)

    # Generate data
    df = generate_sample_data(300)

    # Initialize monitor
    monitor = QualityMonitor(dataset_name="demo_sentiment")

    logger.info("\n1. Tracking Annotator Metrics...")
    metrics = monitor.track_annotator_metrics(
        df,
        annotator_id_column="annotator_id",
        label_column="predicted_label",
        gold_label_column="true_label",
        confidence_column="confidence",
    )

    logger.info(f"   Tracked {len(metrics)} annotators\n")

    # Display metrics for each annotator
    logger.info("2. Per-Annotator Statistics:")
    for annotator_id, m in metrics.items():
        logger.info(f"\n   Annotator: {annotator_id}")
        logger.info(f"   - Annotations: {m['n_annotations']}")
        if 'accuracy' in m:
            logger.info(f"   - Accuracy: {m['accuracy']:.4f}")
        if 'cohen_kappa' in m:
            logger.info(f"   - Cohen's Kappa: {m['cohen_kappa']:.4f}")
        if 'confidence_stats' in m:
            conf = m['confidence_stats']
            logger.info(f"   - Mean Confidence: {conf['mean']:.4f}")
            logger.info(f"   - Std Confidence: {conf['std']:.4f}")

    return metrics, monitor


def demo_anomaly_detection(metrics: dict, monitor: QualityMonitor):
    """Demonstrate anomaly detection."""
    logger.info("\n" + "=" * 60)
    logger.info("ANOMALY DETECTION DEMO")
    logger.info("=" * 60)

    logger.info("\n1. Detecting Quality Anomalies...")
    anomalies = monitor.detect_anomalies(
        metrics,
        accuracy_threshold=0.8,
        confidence_std_threshold=0.3,
        annotation_rate_zscore_threshold=2.5,
    )

    if anomalies:
        logger.info(f"   ⚠️  Detected {len(anomalies)} anomalies\n")

        logger.info("2. Anomaly Details:")
        for anomaly in anomalies:
            severity_emoji = "🔴" if anomaly['severity'] == 'high' else "🟡" if anomaly['severity'] == 'medium' else "🟢"
            logger.info(f"\n   {severity_emoji} Annotator: {anomaly['annotator_id']}")
            logger.info(f"   - Issue: {anomaly['issue']}")
            logger.info(f"   - Value: {anomaly['value']}")
            logger.info(f"   - Severity: {anomaly['severity']}")
            if 'note' in anomaly:
                logger.info(f"   - Note: {anomaly['note']}")
    else:
        logger.info("   ✅ No anomalies detected!")


def demo_cqaa():
    """Demonstrate Cost Per Quality-Adjusted Annotation (CQAA) calculation."""
    logger.info("\n" + "=" * 60)
    logger.info("COST PER QUALITY-ADJUSTED ANNOTATION (CQAA) DEMO")
    logger.info("=" * 60)

    monitor = QualityMonitor(dataset_name="demo")

    # Compare different scenarios
    scenarios = [
        {"name": "High Quality, High Cost", "annotations": 1000, "accuracy": 0.95, "cost": 1.0},
        {"name": "Medium Quality, Medium Cost", "annotations": 1000, "accuracy": 0.85, "cost": 0.5},
        {"name": "Low Quality, Low Cost", "annotations": 1000, "accuracy": 0.70, "cost": 0.25},
    ]

    logger.info("\n1. Comparing Different Annotation Strategies:\n")

    for scenario in scenarios:
        cqaa = monitor.calculate_cqaa(
            annotations=scenario["annotations"],
            accuracy=scenario["accuracy"],
            cost_per_annotation=scenario["cost"],
        )

        logger.info(f"   {scenario['name']}:")
        logger.info(f"   - Annotations: {scenario['annotations']}")
        logger.info(f"   - Accuracy: {scenario['accuracy']:.2f}")
        logger.info(f"   - Cost per annotation: ${scenario['cost']:.2f}")
        logger.info(f"   - CQAA: ${cqaa['cqaa']:.4f}")
        logger.info(f"   - Total Cost: ${cqaa['total_cost']:.2f}")
        logger.info(f"   - Quality-Adjusted Annotations: {cqaa['quality_adjusted_annotations']:.2f}\n")

    logger.info("\n2. Interpretation:")
    logger.info("   Lower CQAA = Better cost-efficiency for quality")
    logger.info("   The optimal strategy depends on task criticality and budget constraints")


def demo_quality_summary():
    """Demonstrate comprehensive quality summary."""
    logger.info("\n" + "=" * 60)
    logger.info("QUALITY SUMMARY DEMO")
    logger.info("=" * 60)

    # Generate data
    df = generate_sample_data(300)

    # Initialize monitor
    monitor = QualityMonitor(dataset_name="demo_sentiment")

    # Run various analyses
    monitor.calculate_krippendorff_alpha(df, ["annotator1", "annotator2", "annotator3"])

    metrics = monitor.track_annotator_metrics(
        df,
        "annotator_id",
        "predicted_label",
        "true_label",
        "confidence",
    )

    monitor.detect_anomalies(metrics)

    # Get summary
    logger.info("\n1. Quality Summary:")
    summary = monitor.get_quality_summary()

    logger.info(f"   Dataset: {summary['dataset_name']}")
    logger.info(f"   Quality Snapshots: {summary['n_snapshots']}")
    logger.info(f"   Tracked Annotators: {summary['n_tracked_annotators']}")
    logger.info(f"   Detected Anomalies: {summary['n_anomalies']}")

    if 'latest_krippendorff_alpha' in summary:
        logger.info(f"   Latest Krippendorff's Alpha: {summary['latest_krippendorff_alpha']:.4f}")

    if 'mean_annotator_accuracy' in summary:
        logger.info(f"   Mean Annotator Accuracy: {summary['mean_annotator_accuracy']:.4f}")
        logger.info(f"   Min Annotator Accuracy: {summary['min_annotator_accuracy']:.4f}")
        logger.info(f"   Max Annotator Accuracy: {summary['max_annotator_accuracy']:.4f}")

    if 'recent_anomalies' in summary:
        logger.info(f"\n2. Recent Anomalies ({len(summary['recent_anomalies'])}):")
        for anomaly in summary['recent_anomalies'][:3]:
            logger.info(f"   - {anomaly['annotator_id']}: {anomaly['issue']}")


def main():
    """Run all quality monitoring demos."""
    logger.info("🚀 AutoLabeler Quality Monitoring Demo\n")

    # 1. Confidence Calibration
    calibrator = demo_confidence_calibration()

    # 2. Inter-Annotator Agreement
    monitor = demo_krippendorff_alpha()

    # 3. Annotator Performance Tracking
    metrics, monitor = demo_annotator_tracking()

    # 4. Anomaly Detection
    demo_anomaly_detection(metrics, monitor)

    # 5. CQAA Calculation
    demo_cqaa()

    # 6. Quality Summary
    demo_quality_summary()

    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETE!")
    logger.info("=" * 60)
    logger.info("\nNext Steps:")
    logger.info("1. Run the Streamlit dashboard: streamlit run src/autolabeler/dashboard/quality_dashboard.py")
    logger.info("2. Try with your own data")
    logger.info("3. Integrate with AutoLabeler workflows")


if __name__ == "__main__":
    main()
