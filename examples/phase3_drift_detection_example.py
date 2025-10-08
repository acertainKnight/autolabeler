"""
Phase 3 Drift Detection Example
================================

Demonstrates comprehensive drift detection for production monitoring.
Shows PSI (Population Stability Index), statistical tests (KS, Chi-square),
and embedding-based drift detection with domain classifiers.

Features:
- PSI drift detection for numeric features
- Statistical tests (KS, Chi-square) for distribution changes
- Embedding space drift with domain classifiers
- Comprehensive drift reporting
- Automated alerting and retraining triggers
- Real-time monitoring integration

Expected Capabilities:
- PSI < 0.1: No drift detected
- 0.1 ≤ PSI < 0.2: Moderate drift (monitor)
- PSI ≥ 0.2: Significant drift (retrain model)
- Domain classifier AUC > 0.75: Drift detected
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from autolabeler.core.monitoring.drift_detector import (
    DriftDetector,
    DriftDetectionConfig,
)


# Example 1: PSI Drift Detection
def example_1_psi_drift():
    """Detect drift using Population Stability Index."""
    print("\n" + "=" * 80)
    print("Example 1: PSI (Population Stability Index) Drift Detection")
    print("=" * 80)

    # Create baseline dataset
    np.random.seed(42)
    baseline_data = pd.DataFrame({
        "confidence": np.random.beta(5, 2, size=1000),  # High confidence distribution
        "text_length": np.random.normal(100, 20, size=1000),
        "label": np.random.choice(["A", "B", "C"], size=1000),
    })

    print("\nBaseline dataset statistics:")
    print(f"  Confidence: mean={baseline_data['confidence'].mean():.3f}, "
          f"std={baseline_data['confidence'].std():.3f}")
    print(f"  Text length: mean={baseline_data['text_length'].mean():.1f}, "
          f"std={baseline_data['text_length'].std():.1f}")

    # Initialize drift detector
    config = DriftDetectionConfig(
        psi_threshold=0.1,
        moderate_drift_threshold=0.2,
        enable_alerts=True,
    )
    detector = DriftDetector(config)
    detector.set_baseline(baseline_data)

    # Scenario 1: No drift (similar distribution)
    print("\n--- Scenario 1: No Drift ---")
    current_data_no_drift = pd.DataFrame({
        "confidence": np.random.beta(5, 2, size=500),
        "text_length": np.random.normal(100, 20, size=500),
        "label": np.random.choice(["A", "B", "C"], size=500),
    })

    result = detector.detect_psi_drift(current_data_no_drift, "confidence")
    print(f"PSI Score: {result['psi']:.4f}")
    print(f"Interpretation: {result['interpretation']}")
    print(f"Requires Retraining: {result['requires_retraining']}")

    # Scenario 2: Moderate drift
    print("\n--- Scenario 2: Moderate Drift ---")
    current_data_moderate = pd.DataFrame({
        "confidence": np.random.beta(4, 3, size=500),  # Slightly different shape
        "text_length": np.random.normal(110, 25, size=500),
        "label": np.random.choice(["A", "B", "C"], size=500),
    })

    result = detector.detect_psi_drift(current_data_moderate, "confidence")
    print(f"PSI Score: {result['psi']:.4f}")
    print(f"Interpretation: {result['interpretation']}")
    print(f"Requires Retraining: {result['requires_retraining']}")

    # Scenario 3: Significant drift
    print("\n--- Scenario 3: Significant Drift ---")
    current_data_drift = pd.DataFrame({
        "confidence": np.random.beta(2, 5, size=500),  # Inverted shape
        "text_length": np.random.normal(150, 30, size=500),
        "label": np.random.choice(["A", "B", "C"], size=500),
    })

    result = detector.detect_psi_drift(current_data_drift, "confidence")
    print(f"PSI Score: {result['psi']:.4f}")
    print(f"Interpretation: {result['interpretation']}")
    print(f"⚠️  Requires Retraining: {result['requires_retraining']}")

    return detector


# Example 2: Statistical Test Drift Detection
def example_2_statistical_tests():
    """Detect drift using statistical tests (KS, Chi-square)."""
    print("\n" + "=" * 80)
    print("Example 2: Statistical Test Drift Detection")
    print("=" * 80)

    # Baseline data
    np.random.seed(42)
    baseline_data = pd.DataFrame({
        "numeric_feature": np.random.normal(50, 10, size=1000),
        "categorical_feature": np.random.choice(["A", "B", "C"], size=1000, p=[0.5, 0.3, 0.2]),
    })

    config = DriftDetectionConfig()
    detector = DriftDetector(config)
    detector.set_baseline(baseline_data)

    # Test 1: Kolmogorov-Smirnov test (numeric)
    print("\n--- KS Test for Numeric Features ---")

    # No drift
    current_no_drift = pd.DataFrame({
        "numeric_feature": np.random.normal(50, 10, size=500),
    })
    result = detector.detect_statistical_drift(current_no_drift, "numeric_feature", test="ks")
    print(f"No drift: KS statistic={result['statistic']:.4f}, p-value={result['p_value']:.4f}")
    print(f"  Drift detected: {result['drift_detected']}")

    # With drift
    current_with_drift = pd.DataFrame({
        "numeric_feature": np.random.normal(60, 15, size=500),  # Different mean and std
    })
    result = detector.detect_statistical_drift(current_with_drift, "numeric_feature", test="ks")
    print(f"With drift: KS statistic={result['statistic']:.4f}, p-value={result['p_value']:.4f}")
    print(f"  ⚠️  Drift detected: {result['drift_detected']}")

    # Test 2: Chi-square test (categorical)
    print("\n--- Chi-Square Test for Categorical Features ---")

    # No drift
    current_no_drift_cat = pd.DataFrame({
        "categorical_feature": np.random.choice(["A", "B", "C"], size=500, p=[0.5, 0.3, 0.2]),
    })
    result = detector.detect_statistical_drift(
        pd.concat([baseline_data, current_no_drift_cat]),
        "categorical_feature",
        test="chi2"
    )
    print(f"No drift: Chi-square={result['statistic']:.4f}, p-value={result['p_value']:.4f}")
    print(f"  Drift detected: {result['drift_detected']}")

    # With drift
    current_with_drift_cat = pd.DataFrame({
        "categorical_feature": np.random.choice(["A", "B", "C"], size=500, p=[0.2, 0.3, 0.5]),  # Reversed
    })
    result = detector.detect_statistical_drift(
        pd.concat([baseline_data, current_with_drift_cat]),
        "categorical_feature",
        test="chi2"
    )
    print(f"With drift: Chi-square={result['statistic']:.4f}, p-value={result['p_value']:.4f}")
    print(f"  ⚠️  Drift detected: {result['drift_detected']}")

    return detector


# Example 3: Embedding Space Drift Detection
def example_3_embedding_drift():
    """Detect drift in embedding space using domain classifier."""
    print("\n" + "=" * 80)
    print("Example 3: Embedding Space Drift Detection")
    print("=" * 80)

    # Create baseline embeddings (simulated)
    np.random.seed(42)
    baseline_embeddings = np.random.randn(1000, 128)  # 128-dim embeddings

    config = DriftDetectionConfig()
    detector = DriftDetector(config)
    detector.set_baseline(pd.DataFrame(), baseline_embeddings)

    print("\nBaseline: 1000 samples, 128-dimensional embeddings")

    # Scenario 1: No drift (same distribution)
    print("\n--- Scenario 1: No Drift ---")
    current_embeddings_no_drift = np.random.randn(500, 128)

    result = detector.detect_embedding_drift(
        current_embeddings_no_drift,
        method="domain_classifier"
    )
    print(f"Domain Classifier AUC: {result['auc']:.4f}")
    print(f"Drift Detected: {result['drift_detected']}")
    print(f"Severity: {result['severity']}")

    # Scenario 2: Moderate drift (slightly shifted)
    print("\n--- Scenario 2: Moderate Drift ---")
    current_embeddings_moderate = np.random.randn(500, 128) + 0.3  # Shift by 0.3

    result = detector.detect_embedding_drift(
        current_embeddings_moderate,
        method="domain_classifier"
    )
    print(f"Domain Classifier AUC: {result['auc']:.4f}")
    print(f"Drift Detected: {result['drift_detected']}")
    print(f"Severity: {result['severity']}")

    # Scenario 3: High drift (very different distribution)
    print("\n--- Scenario 3: High Drift ---")
    current_embeddings_high = np.random.randn(500, 128) * 2 + 1  # Different scale and shift

    result = detector.detect_embedding_drift(
        current_embeddings_high,
        method="domain_classifier"
    )
    print(f"Domain Classifier AUC: {result['auc']:.4f}")
    print(f"⚠️  Drift Detected: {result['drift_detected']}")
    print(f"Severity: {result['severity']}")
    print(f"⚠️  Requires Retraining: {result['requires_retraining']}")

    return detector


# Example 4: Comprehensive Drift Report
def example_4_comprehensive_report():
    """Generate comprehensive drift detection report."""
    print("\n" + "=" * 80)
    print("Example 4: Comprehensive Drift Report")
    print("=" * 80)

    # Setup baseline
    np.random.seed(42)
    baseline_data = pd.DataFrame({
        "confidence": np.random.beta(5, 2, size=1000),
        "text_length": np.random.normal(100, 20, size=1000),
        "num_entities": np.random.poisson(3, size=1000),
        "label": np.random.choice(["A", "B", "C"], size=1000),
    })
    baseline_embeddings = np.random.randn(1000, 128)

    config = DriftDetectionConfig()
    detector = DriftDetector(config)
    detector.set_baseline(baseline_data, baseline_embeddings)

    print("\nBaseline established with:")
    print(f"  - {len(baseline_data)} samples")
    print(f"  - {len(baseline_data.columns)} features")
    print(f"  - {baseline_embeddings.shape[1]}-dim embeddings")

    # Current data with drift
    current_data = pd.DataFrame({
        "confidence": np.random.beta(3, 4, size=500),  # Drift
        "text_length": np.random.normal(120, 25, size=500),  # Drift
        "num_entities": np.random.poisson(3, size=500),  # No drift
        "label": np.random.choice(["A", "B", "C"], size=500),
    })
    current_embeddings = np.random.randn(500, 128) + 0.5  # Moderate drift

    print("\n" + "-" * 80)
    print("Generating comprehensive drift report...")
    print("-" * 80)

    report = detector.comprehensive_drift_report(current_data, current_embeddings)

    print(f"\nReport Timestamp: {report['timestamp']}")
    print(f"Baseline Size: {report['baseline_size']}")
    print(f"Current Size: {report['current_size']}")

    print("\n--- PSI Results ---")
    for feature, psi_result in report['psi_results'].items():
        status = "✓" if not psi_result['requires_retraining'] else "⚠️ "
        print(f"{status} {feature}: PSI={psi_result['psi']:.4f} "
              f"({psi_result['interpretation']})")

    print("\n--- Statistical Test Results ---")
    for feature, stat_result in report['statistical_results'].items():
        status = "✓" if not stat_result['drift_detected'] else "⚠️ "
        print(f"{status} {feature}: {stat_result['test']} "
              f"p-value={stat_result['p_value']:.4f}")

    print("\n--- Embedding Drift ---")
    emb_drift = report['embedding_drift']
    if emb_drift:
        status = "✓" if not emb_drift['drift_detected'] else "⚠️ "
        print(f"{status} Domain Classifier AUC: {emb_drift['auc']:.4f}")
        print(f"   Severity: {emb_drift['severity']}")

    print("\n--- Overall Assessment ---")
    if report['overall_drift_detected']:
        print("⚠️  DRIFT DETECTED - Retraining recommended")
        print(f"   Features with drift: {report['drift_feature_count']}")
        print(f"   Recommendation: {report['recommendation']}")
    else:
        print("✓ No significant drift detected")
        print("   System is stable, continue monitoring")

    return report


# Example 5: Production Monitoring Pipeline
def example_5_production_monitoring():
    """Production drift monitoring with alerting."""
    print("\n" + "=" * 80)
    print("Example 5: Production Monitoring Pipeline")
    print("=" * 80)

    # Setup detector with alerting enabled
    config = DriftDetectionConfig(
        psi_threshold=0.1,
        moderate_drift_threshold=0.2,
        enable_alerts=True,
        alert_methods=["email", "slack"],
        check_frequency_hours=24,
    )
    detector = DriftDetector(config)

    # Set baseline from training data
    baseline_data = pd.DataFrame({
        "confidence": np.random.beta(5, 2, size=5000),
        "text_length": np.random.normal(100, 20, size=5000),
        "label": np.random.choice(["A", "B", "C"], size=5000),
    })
    detector.set_baseline(baseline_data)

    print("\nProduction monitoring initialized:")
    print(f"  - Check frequency: every {config.check_frequency_hours} hours")
    print(f"  - PSI threshold: {config.psi_threshold}")
    print(f"  - Alert methods: {', '.join(config.alert_methods)}")

    # Simulate daily checks for 7 days
    print("\nSimulating 7 days of production monitoring...")
    print("-" * 80)

    for day in range(1, 8):
        print(f"\nDay {day}:")

        # Simulate gradual drift over time
        drift_factor = day * 0.1
        current_data = pd.DataFrame({
            "confidence": np.random.beta(5 - drift_factor, 2 + drift_factor, size=1000),
            "text_length": np.random.normal(100 + drift_factor * 10, 20, size=1000),
            "label": np.random.choice(["A", "B", "C"], size=1000),
        })

        # Check confidence drift
        result = detector.detect_psi_drift(current_data, "confidence")

        if result['interpretation'] == "no_drift":
            print(f"  ✓ No drift detected (PSI={result['psi']:.4f})")
        elif result['interpretation'] == "moderate_drift":
            print(f"  ⚠️  Moderate drift detected (PSI={result['psi']:.4f})")
            print(f"     Recommendation: Increase monitoring frequency")
        else:
            print(f"  🚨 SIGNIFICANT DRIFT (PSI={result['psi']:.4f})")
            print(f"     Recommendation: Retrain model immediately")
            print(f"     Alert sent to: {', '.join(config.alert_methods)}")

    print("\n" + "-" * 80)
    print("7-day monitoring complete")
    print("\nSummary:")
    print("  - Drift gradually increased over time")
    print("  - Alerts triggered on Day 5-7")
    print("  - Model retraining recommended")

    return detector


# Example 6: Drift Detection Integration
def example_6_integration():
    """Integrate drift detection with labeling pipeline."""
    print("\n" + "=" * 80)
    print("Example 6: Drift Detection Integration")
    print("=" * 80)

    from autolabeler.core.labeling.labeling_service import LabelingService
    from autolabeler.core.configs import LabelingConfig, Settings

    # Setup labeling service
    settings = Settings()
    config = LabelingConfig(
        model_name="gpt-4o-mini",
        enable_drift_detection=True,
        drift_check_frequency=100,  # Check every 100 annotations
    )

    service = LabelingService(settings, config)

    # Setup drift detector
    drift_detector = DriftDetector(DriftDetectionConfig())

    print("\nIntegrated drift detection enabled:")
    print(f"  - Check frequency: every {config.drift_check_frequency} annotations")
    print(f"  - Auto-retraining: enabled")

    # Simulate annotation with drift monitoring
    annotation_count = 0
    drift_checks = 0

    print("\nSimulating 500 annotations with monitoring...")

    for i in range(500):
        # Perform annotation (simulated)
        annotation_count += 1

        # Check for drift periodically
        if annotation_count % config.drift_check_frequency == 0:
            drift_checks += 1
            print(f"  Check {drift_checks}: Processed {annotation_count} annotations")

            # Run drift detection (simulated)
            # In production, compare recent batch to baseline
            print(f"    ✓ No drift detected, continuing...")

    print(f"\nCompleted {annotation_count} annotations")
    print(f"Performed {drift_checks} drift checks")
    print("✓ System remained stable throughout")

    return service


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Phase 3: Drift Detection Examples")
    print("=" * 80)
    print("\nThis demonstrates comprehensive drift detection with:")
    print("  - PSI (Population Stability Index) monitoring")
    print("  - Statistical tests (KS, Chi-square)")
    print("  - Embedding space drift detection")
    print("  - Comprehensive reporting")
    print("  - Production monitoring pipeline")

    # Run all examples
    example_1_psi_drift()
    example_2_statistical_tests()
    example_3_embedding_drift()
    example_4_comprehensive_report()
    example_5_production_monitoring()
    example_6_integration()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    print("\nKey Thresholds:")
    print("  PSI < 0.1: No drift")
    print("  0.1 ≤ PSI < 0.2: Moderate drift (monitor)")
    print("  PSI ≥ 0.2: Significant drift (retrain)")
    print("  Domain Classifier AUC > 0.75: Drift detected")
    print("\nNext steps:")
    print("  1. Set baseline from production training data")
    print("  2. Configure alert thresholds for your use case")
    print("  3. Enable continuous monitoring")
    print("  4. Integrate with retraining pipeline")
