"""
Drift detection system for monitoring annotation quality and distribution shifts.

This module implements multiple drift detection methods including Population Stability
Index (PSI), statistical tests (KS, Chi-square), and embedding-based drift detection.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from scipy.stats import chisquare, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class DriftDetectionConfig(BaseModel):
    """Configuration for drift detection."""

    psi_threshold: float = Field(
        0.2, ge=0.0, description="PSI threshold for significant drift"
    )
    statistical_alpha: float = Field(
        0.05, ge=0.0, le=1.0, description="Significance level for statistical tests"
    )
    domain_classifier_threshold: float = Field(
        0.75, ge=0.5, le=1.0, description="AUC threshold for domain classifier drift"
    )
    min_samples: int = Field(
        100, gt=0, description="Minimum samples for drift detection"
    )
    num_bins: int = Field(10, gt=1, description="Number of bins for PSI calculation")
    test_size: float = Field(
        0.3, ge=0.1, le=0.5, description="Test size for domain classifier"
    )


class DriftDetector:
    """
    Detect distribution drift in annotation data.

    Implements multiple drift detection methods:
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov test
    - Chi-square test
    - Domain classifier method for embeddings

    Example:
        >>> detector = DriftDetector(config)
        >>> detector.set_baseline(baseline_df, baseline_embeddings)
        >>> report = detector.comprehensive_drift_report(current_df, current_embeddings)
        >>> if report["overall_drift_detected"]:
        ...     print("Drift detected! Consider retraining.")
    """

    def __init__(self, config: DriftDetectionConfig | None = None):
        """
        Initialize drift detector.

        Args:
            config: Drift detection configuration.
        """
        self.config = config or DriftDetectionConfig()
        self.baseline_data: pd.DataFrame | None = None
        self.baseline_embeddings: np.ndarray | None = None

    def set_baseline(
        self,
        data: pd.DataFrame,
        embeddings: np.ndarray | None = None
    ) -> None:
        """
        Set baseline distribution for comparison.

        Args:
            data: Baseline DataFrame.
            embeddings: Optional baseline embeddings.
        """
        if len(data) < self.config.min_samples:
            logger.warning(
                f"Baseline has only {len(data)} samples, "
                f"minimum {self.config.min_samples} recommended"
            )

        self.baseline_data = data.copy()
        self.baseline_embeddings = embeddings.copy() if embeddings is not None else None

        logger.info(f"Set baseline with {len(data)} samples")

    def detect_psi_drift(
        self,
        current_data: pd.DataFrame,
        feature_column: str,
        num_bins: int | None = None,
    ) -> dict[str, Any]:
        """
        Detect drift using Population Stability Index (PSI).

        PSI measures the shift in distribution between baseline and current data.
        - PSI < 0.1: No significant drift
        - 0.1 <= PSI < 0.2: Moderate drift
        - PSI >= 0.2: Significant drift (retraining recommended)

        Args:
            current_data: Current DataFrame.
            feature_column: Column name to check for drift.
            num_bins: Number of bins for histogram (defaults to config).

        Returns:
            Dictionary with PSI results.
        """
        if self.baseline_data is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")

        if feature_column not in self.baseline_data.columns:
            raise ValueError(f"Feature {feature_column} not in baseline data")

        if feature_column not in current_data.columns:
            raise ValueError(f"Feature {feature_column} not in current data")

        num_bins = num_bins or self.config.num_bins

        baseline_feature = self.baseline_data[feature_column].dropna()
        current_feature = current_data[feature_column].dropna()

        # Create bins based on baseline
        bins = np.histogram_bin_edges(baseline_feature, bins=num_bins)

        # Calculate distributions
        baseline_dist, _ = np.histogram(baseline_feature, bins=bins)
        current_dist, _ = np.histogram(current_feature, bins=bins)

        # Normalize to probabilities
        baseline_dist = baseline_dist / baseline_dist.sum()
        current_dist = current_dist / current_dist.sum()

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        baseline_dist = baseline_dist + epsilon
        current_dist = current_dist + epsilon

        # Calculate PSI
        psi = np.sum(
            (current_dist - baseline_dist)
            * np.log(current_dist / baseline_dist)
        )

        # Interpret
        if psi < 0.1:
            interpretation = "no_drift"
        elif psi < self.config.psi_threshold:
            interpretation = "moderate_drift"
        else:
            interpretation = "significant_drift"

        return {
            "method": "psi",
            "feature": feature_column,
            "psi": float(psi),
            "interpretation": interpretation,
            "requires_retraining": psi >= self.config.psi_threshold,
            "baseline_samples": len(baseline_feature),
            "current_samples": len(current_feature),
        }

    def detect_embedding_drift(
        self,
        current_embeddings: np.ndarray,
        method: str = "domain_classifier",
    ) -> dict[str, Any]:
        """
        Detect drift in embedding space using domain classifier.

        Trains a classifier to distinguish baseline vs. current embeddings.
        High AUC indicates that distributions are separable (drift detected).

        Args:
            current_embeddings: Current embedding matrix.
            method: Detection method (currently only "domain_classifier").

        Returns:
            Dictionary with drift detection results.
        """
        if self.baseline_embeddings is None:
            raise ValueError("Baseline embeddings not set")

        if method == "domain_classifier":
            return self._domain_classifier_drift(current_embeddings)
        else:
            raise ValueError(f"Unknown embedding drift method: {method}")

    def _domain_classifier_drift(
        self, current_embeddings: np.ndarray
    ) -> dict[str, Any]:
        """
        Detect drift using domain classifier method.

        Args:
            current_embeddings: Current embedding matrix.

        Returns:
            Drift detection results.
        """
        # Combine baseline (label 0) and current (label 1)
        X = np.vstack([self.baseline_embeddings, current_embeddings])
        y = np.array(
            [0] * len(self.baseline_embeddings) + [1] * len(current_embeddings)
        )

        # Shuffle and split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=42, stratify=y
        )

        # Train domain classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        # Interpret: AUC close to 0.5 = no drift, close to 1.0 = significant drift
        drift_detected = auc > self.config.domain_classifier_threshold

        if auc > 0.85:
            severity = "high"
        elif auc > self.config.domain_classifier_threshold:
            severity = "medium"
        else:
            severity = "low"

        return {
            "method": "domain_classifier",
            "auc": float(auc),
            "drift_detected": drift_detected,
            "severity": severity,
            "requires_retraining": auc > 0.80,
            "baseline_samples": len(self.baseline_embeddings),
            "current_samples": len(current_embeddings),
        }

    def detect_statistical_drift(
        self,
        current_data: pd.DataFrame,
        feature_column: str,
        test: str = "ks",
    ) -> dict[str, Any]:
        """
        Detect drift using statistical tests.

        Supports:
        - "ks": Kolmogorov-Smirnov test (for continuous features)
        - "chi2": Chi-square test (for categorical features)

        Args:
            current_data: Current DataFrame.
            feature_column: Column to test.
            test: Statistical test to use.

        Returns:
            Dictionary with test results.
        """
        if self.baseline_data is None:
            raise ValueError("Baseline not set")

        if feature_column not in self.baseline_data.columns:
            raise ValueError(f"Feature {feature_column} not in baseline data")

        if feature_column not in current_data.columns:
            raise ValueError(f"Feature {feature_column} not in current data")

        baseline_feature = self.baseline_data[feature_column].dropna()
        current_feature = current_data[feature_column].dropna()

        if test == "ks":
            # Kolmogorov-Smirnov test for continuous distributions
            statistic, p_value = ks_2samp(baseline_feature, current_feature)
            test_name = "Kolmogorov-Smirnov"

        elif test == "chi2":
            # Chi-square test for categorical distributions
            baseline_counts = baseline_feature.value_counts()
            current_counts = current_feature.value_counts()

            # Align categories
            all_categories = set(baseline_counts.index) | set(current_counts.index)
            baseline_aligned = [
                baseline_counts.get(cat, 0) for cat in all_categories
            ]
            current_aligned = [current_counts.get(cat, 0) for cat in all_categories]

            # Add small constant to avoid division by zero
            baseline_aligned = [max(x, 1) for x in baseline_aligned]

            statistic, p_value = chisquare(current_aligned, baseline_aligned)
            test_name = "Chi-Square"

        else:
            raise ValueError(f"Unknown statistical test: {test}")

        drift_detected = p_value < self.config.statistical_alpha

        return {
            "method": "statistical",
            "test": test_name,
            "feature": feature_column,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": drift_detected,
            "significance_level": self.config.statistical_alpha,
            "baseline_samples": len(baseline_feature),
            "current_samples": len(current_feature),
        }

    def comprehensive_drift_report(
        self,
        current_data: pd.DataFrame,
        current_embeddings: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive drift detection report.

        Applies all drift detection methods and aggregates results.

        Args:
            current_data: Current DataFrame.
            current_embeddings: Optional current embeddings.

        Returns:
            Comprehensive drift report.
        """
        if self.baseline_data is None:
            raise ValueError("Baseline not set")

        report = {
            "timestamp": datetime.now().isoformat(),
            "baseline_size": len(self.baseline_data),
            "current_size": len(current_data),
            "psi_results": {},
            "statistical_results": {},
            "embedding_drift": None,
            "overall_drift_detected": False,
            "drift_score": 0.0,
        }

        # PSI for numeric features
        numeric_features = current_data.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            if feature in self.baseline_data.columns:
                try:
                    psi_result = self.detect_psi_drift(current_data, feature)
                    report["psi_results"][feature] = psi_result
                except Exception as e:
                    logger.warning(f"PSI failed for {feature}: {e}")

        # Statistical tests for categorical features
        categorical_features = current_data.select_dtypes(
            include=["object", "category"]
        ).columns
        for feature in categorical_features:
            if feature in self.baseline_data.columns:
                try:
                    stat_result = self.detect_statistical_drift(
                        current_data, feature, test="chi2"
                    )
                    report["statistical_results"][feature] = stat_result
                except Exception as e:
                    logger.warning(f"Statistical test failed for {feature}: {e}")

        # Embedding drift
        if current_embeddings is not None and self.baseline_embeddings is not None:
            try:
                report["embedding_drift"] = self.detect_embedding_drift(
                    current_embeddings
                )
            except Exception as e:
                logger.warning(f"Embedding drift detection failed: {e}")

        # Overall determination
        drift_signals = []

        # Collect PSI signals
        for psi_result in report["psi_results"].values():
            drift_signals.append(1.0 if psi_result["requires_retraining"] else 0.0)

        # Collect statistical test signals
        for stat_result in report["statistical_results"].values():
            drift_signals.append(1.0 if stat_result["drift_detected"] else 0.0)

        # Collect embedding signal
        if report["embedding_drift"]:
            drift_signals.append(
                1.0 if report["embedding_drift"]["drift_detected"] else 0.0
            )

        if drift_signals:
            drift_score = sum(drift_signals) / len(drift_signals)
            report["drift_score"] = drift_score
            # Overall drift if >30% of signals indicate drift
            report["overall_drift_detected"] = drift_score > 0.3
        else:
            report["drift_score"] = 0.0
            report["overall_drift_detected"] = False

        # Recommendations
        report["recommendations"] = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: dict[str, Any]) -> list[str]:
        """
        Generate recommendations based on drift report.

        Args:
            report: Drift detection report.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        if report["overall_drift_detected"]:
            recommendations.append(
                "Significant drift detected. Consider retraining the model."
            )

            # Identify most drifted features
            high_drift_features = []

            for feature, result in report["psi_results"].items():
                if result["requires_retraining"]:
                    high_drift_features.append(f"{feature} (PSI={result['psi']:.3f})")

            for feature, result in report["statistical_results"].items():
                if result["drift_detected"]:
                    high_drift_features.append(
                        f"{feature} (p={result['p_value']:.3f})"
                    )

            if high_drift_features:
                recommendations.append(
                    f"Features with high drift: {', '.join(high_drift_features)}"
                )

            if report.get("embedding_drift", {}).get("severity") == "high":
                recommendations.append(
                    "High embedding drift detected. "
                    "Distribution has shifted significantly."
                )

        else:
            recommendations.append("No significant drift detected. Model is stable.")

        return recommendations

    def reset_baseline(self) -> None:
        """Reset baseline data and embeddings."""
        self.baseline_data = None
        self.baseline_embeddings = None
        logger.info("Baseline reset")
