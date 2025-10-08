"""
Quality monitoring for annotation and labeling tasks.

This module provides metrics for inter-annotator agreement, quality tracking,
and anomaly detection in labeling workflows.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import cohen_kappa_score, confusion_matrix


class QualityMonitor:
    """
    Monitors quality metrics for annotation and labeling tasks.

    Tracks inter-annotator agreement using Krippendorff's alpha, per-annotator
    performance metrics, confidence distributions, and CQAA (Cost Per Quality-Adjusted
    Annotation). Also provides anomaly detection for quality issues.

    Example:
        >>> monitor = QualityMonitor(dataset_name="sentiment")
        >>> alpha = monitor.calculate_krippendorff_alpha(df, ["annotator1", "annotator2"])
        >>> metrics = monitor.track_annotator_metrics(df, "annotator_id", "label", "gold_label")
        >>> cqaa = monitor.calculate_cqaa(annotations=100, accuracy=0.85, cost_per_annotation=0.5)
    """

    def __init__(
        self,
        dataset_name: str,
        metric_distance: Literal["nominal", "ordinal", "interval", "ratio"] = "nominal",
    ) -> None:
        """
        Initialize the quality monitor.

        Args:
            dataset_name: Name of the dataset being monitored.
            metric_distance: Distance metric for Krippendorff's alpha calculation.
        """
        self.dataset_name = dataset_name
        self.metric_distance = metric_distance

        # Tracking storage
        self.annotator_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.quality_snapshots: list[dict[str, Any]] = []
        self.anomalies: list[dict[str, Any]] = []

    def calculate_krippendorff_alpha(
        self,
        df: pd.DataFrame,
        annotator_columns: list[str],
        value_domain: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate Krippendorff's alpha for inter-annotator agreement.

        Krippendorff's alpha is a reliability coefficient that measures agreement
        between annotators, accounting for chance agreement.

        Args:
            df: DataFrame with annotations.
            annotator_columns: Column names for different annotators.
            value_domain: Complete set of possible values (optional).

        Returns:
            Dictionary containing alpha value and agreement statistics.

        Example:
            >>> result = monitor.calculate_krippendorff_alpha(
            ...     df, ["annotator1", "annotator2", "annotator3"]
            ... )
            >>> print(f"Krippendorff's alpha: {result['alpha']:.3f}")
        """
        if not annotator_columns:
            raise ValueError("Must provide at least 2 annotator columns")

        # Create reliability data matrix (annotators x items)
        reliability_data = df[annotator_columns].values.T

        # Calculate alpha
        alpha = self._compute_krippendorff_alpha(
            reliability_data,
            metric=self.metric_distance,
            value_domain=value_domain,
        )

        # Calculate pairwise agreement
        pairwise_agreement = self._calculate_pairwise_agreement(df, annotator_columns)

        # Calculate per-item agreement
        per_item_agreement = self._calculate_per_item_agreement(df, annotator_columns)

        result = {
            "alpha": float(alpha),
            "n_annotators": len(annotator_columns),
            "n_items": len(df),
            "metric": self.metric_distance,
            "pairwise_agreement": pairwise_agreement,
            "mean_pairwise_agreement": float(np.mean(list(pairwise_agreement.values()))),
            "per_item_agreement_mean": float(np.mean(per_item_agreement)),
            "per_item_agreement_std": float(np.std(per_item_agreement)),
            "items_with_full_agreement": int(np.sum(per_item_agreement == 1.0)),
            "items_with_disagreement": int(np.sum(per_item_agreement < 1.0)),
            "timestamp": datetime.now().isoformat(),
        }

        # Store snapshot
        self.quality_snapshots.append(result)

        logger.info(
            f"Krippendorff's alpha: {alpha:.4f} "
            f"({len(annotator_columns)} annotators, {len(df)} items)"
        )

        return result

    def track_annotator_metrics(
        self,
        df: pd.DataFrame,
        annotator_id_column: str,
        label_column: str,
        gold_label_column: str | None = None,
        confidence_column: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Track per-annotator performance metrics.

        Args:
            df: DataFrame with annotations.
            annotator_id_column: Column containing annotator identifiers.
            label_column: Column containing annotations/labels.
            gold_label_column: Column with gold standard labels (if available).
            confidence_column: Column with confidence scores (if available).

        Returns:
            Dictionary mapping annotator IDs to their metrics.

        Example:
            >>> metrics = monitor.track_annotator_metrics(
            ...     df, "annotator", "predicted_label", "true_label"
            ... )
            >>> for ann_id, m in metrics.items():
            ...     print(f"{ann_id}: accuracy={m['accuracy']:.3f}")
        """
        annotators = df[annotator_id_column].unique()
        metrics = {}

        for annotator in annotators:
            annotator_df = df[df[annotator_id_column] == annotator]

            annotator_metrics = {
                "annotator_id": str(annotator),
                "n_annotations": len(annotator_df),
                "timestamp": datetime.now().isoformat(),
            }

            # Label distribution
            label_dist = annotator_df[label_column].value_counts().to_dict()
            annotator_metrics["label_distribution"] = {str(k): int(v) for k, v in label_dist.items()}

            # Accuracy metrics (if gold labels available)
            if gold_label_column and gold_label_column in annotator_df.columns:
                gold_labels = annotator_df[gold_label_column]
                pred_labels = annotator_df[label_column]

                # Remove missing values
                valid_mask = (~gold_labels.isna()) & (~pred_labels.isna())
                gold_valid = gold_labels[valid_mask]
                pred_valid = pred_labels[valid_mask]

                if len(gold_valid) > 0:
                    accuracy = float((gold_valid == pred_valid).mean())
                    annotator_metrics["accuracy"] = accuracy
                    annotator_metrics["n_correct"] = int((gold_valid == pred_valid).sum())
                    annotator_metrics["n_incorrect"] = int((gold_valid != pred_valid).sum())

                    # Cohen's kappa (for multi-class)
                    if len(gold_valid.unique()) > 1:
                        try:
                            kappa = cohen_kappa_score(gold_valid, pred_valid)
                            annotator_metrics["cohen_kappa"] = float(kappa)
                        except Exception as e:
                            logger.warning(f"Could not calculate kappa for {annotator}: {e}")

            # Confidence metrics (if available)
            if confidence_column and confidence_column in annotator_df.columns:
                confidences = annotator_df[confidence_column].dropna()
                if len(confidences) > 0:
                    annotator_metrics["confidence_stats"] = {
                        "mean": float(confidences.mean()),
                        "std": float(confidences.std()),
                        "median": float(confidences.median()),
                        "min": float(confidences.min()),
                        "max": float(confidences.max()),
                    }

            # Annotation speed (if timestamps available)
            if "timestamp" in annotator_df.columns:
                try:
                    timestamps = pd.to_datetime(annotator_df["timestamp"])
                    if len(timestamps) > 1:
                        time_diffs = timestamps.diff().dt.total_seconds().dropna()
                        annotator_metrics["avg_annotation_time_seconds"] = float(time_diffs.mean())
                except Exception as e:
                    logger.debug(f"Could not calculate annotation speed: {e}")

            metrics[str(annotator)] = annotator_metrics

            # Store in history
            self.annotator_history[str(annotator)].append(annotator_metrics)

        return metrics

    def calculate_cqaa(
        self,
        annotations: int,
        accuracy: float,
        cost_per_annotation: float,
        quality_weight: float = 1.0,
    ) -> dict[str, float]:
        """
        Calculate Cost Per Quality-Adjusted Annotation (CQAA).

        CQAA = Total Cost / (Number of Annotations × Accuracy^quality_weight)

        Lower CQAA indicates better cost-efficiency for quality.

        Args:
            annotations: Number of annotations completed.
            accuracy: Overall accuracy (0-1).
            cost_per_annotation: Cost per single annotation.
            quality_weight: Weight for quality adjustment (higher = more penalty for low accuracy).

        Returns:
            Dictionary with CQAA and component metrics.

        Example:
            >>> cqaa = monitor.calculate_cqaa(
            ...     annotations=1000,
            ...     accuracy=0.85,
            ...     cost_per_annotation=0.25
            ... )
            >>> print(f"CQAA: ${cqaa['cqaa']:.4f}")
        """
        if annotations <= 0 or accuracy <= 0 or cost_per_annotation < 0:
            raise ValueError("Invalid input values for CQAA calculation")

        total_cost = annotations * cost_per_annotation
        quality_adjusted_annotations = annotations * (accuracy ** quality_weight)
        cqaa = total_cost / quality_adjusted_annotations

        return {
            "cqaa": float(cqaa),
            "total_cost": float(total_cost),
            "quality_adjusted_annotations": float(quality_adjusted_annotations),
            "annotations": annotations,
            "accuracy": accuracy,
            "cost_per_annotation": cost_per_annotation,
            "quality_weight": quality_weight,
        }

    def detect_anomalies(
        self,
        annotator_metrics: dict[str, dict[str, Any]],
        accuracy_threshold: float = 0.7,
        confidence_std_threshold: float = 0.3,
        annotation_rate_zscore_threshold: float = 2.5,
    ) -> list[dict[str, Any]]:
        """
        Detect anomalies in annotator performance.

        Flags annotators with:
        - Low accuracy
        - Unusual confidence patterns
        - Abnormal annotation rates

        Args:
            annotator_metrics: Metrics from track_annotator_metrics.
            accuracy_threshold: Minimum acceptable accuracy.
            confidence_std_threshold: Maximum acceptable confidence std dev.
            annotation_rate_zscore_threshold: Z-score threshold for annotation rate.

        Returns:
            List of detected anomalies.

        Example:
            >>> metrics = monitor.track_annotator_metrics(df, "annotator", "label", "gold")
            >>> anomalies = monitor.detect_anomalies(metrics)
            >>> for anomaly in anomalies:
            ...     print(f"{anomaly['annotator_id']}: {anomaly['issue']}")
        """
        detected_anomalies = []

        # Extract annotation counts for z-score calculation
        annotation_counts = []
        for metrics in annotator_metrics.values():
            if "n_annotations" in metrics:
                annotation_counts.append(metrics["n_annotations"])

        if annotation_counts:
            mean_annotations = np.mean(annotation_counts)
            std_annotations = np.std(annotation_counts)
        else:
            mean_annotations = 0
            std_annotations = 1

        for annotator_id, metrics in annotator_metrics.items():
            # Check accuracy
            if "accuracy" in metrics and metrics["accuracy"] < accuracy_threshold:
                anomaly = {
                    "annotator_id": annotator_id,
                    "issue": "low_accuracy",
                    "value": metrics["accuracy"],
                    "threshold": accuracy_threshold,
                    "severity": "high" if metrics["accuracy"] < 0.5 else "medium",
                    "timestamp": datetime.now().isoformat(),
                }
                detected_anomalies.append(anomaly)

            # Check confidence patterns
            if "confidence_stats" in metrics:
                conf_std = metrics["confidence_stats"]["std"]
                if conf_std < confidence_std_threshold:
                    anomaly = {
                        "annotator_id": annotator_id,
                        "issue": "low_confidence_variance",
                        "value": conf_std,
                        "threshold": confidence_std_threshold,
                        "severity": "low",
                        "timestamp": datetime.now().isoformat(),
                        "note": "Annotator may be providing uniform confidence scores",
                    }
                    detected_anomalies.append(anomaly)

            # Check annotation rate
            if "n_annotations" in metrics and std_annotations > 0:
                n_annotations = metrics["n_annotations"]
                z_score = abs((n_annotations - mean_annotations) / std_annotations)

                if z_score > annotation_rate_zscore_threshold:
                    anomaly = {
                        "annotator_id": annotator_id,
                        "issue": "unusual_annotation_rate",
                        "value": n_annotations,
                        "z_score": float(z_score),
                        "threshold": annotation_rate_zscore_threshold,
                        "severity": "medium",
                        "timestamp": datetime.now().isoformat(),
                        "note": "Significantly different from mean annotation count",
                    }
                    detected_anomalies.append(anomaly)

        # Store anomalies
        self.anomalies.extend(detected_anomalies)

        if detected_anomalies:
            logger.warning(f"Detected {len(detected_anomalies)} quality anomalies")

        return detected_anomalies

    def get_quality_summary(self) -> dict[str, Any]:
        """
        Get comprehensive quality summary.

        Returns:
            Dictionary with overall quality metrics and trends.
        """
        summary = {
            "dataset_name": self.dataset_name,
            "n_snapshots": len(self.quality_snapshots),
            "n_anomalies": len(self.anomalies),
            "n_tracked_annotators": len(self.annotator_history),
        }

        # Latest Krippendorff's alpha
        if self.quality_snapshots:
            latest_snapshot = self.quality_snapshots[-1]
            summary["latest_krippendorff_alpha"] = latest_snapshot.get("alpha")
            summary["latest_mean_pairwise_agreement"] = latest_snapshot.get("mean_pairwise_agreement")

        # Annotator summary
        if self.annotator_history:
            accuracies = []
            for history in self.annotator_history.values():
                if history and "accuracy" in history[-1]:
                    accuracies.append(history[-1]["accuracy"])

            if accuracies:
                summary["mean_annotator_accuracy"] = float(np.mean(accuracies))
                summary["std_annotator_accuracy"] = float(np.std(accuracies))
                summary["min_annotator_accuracy"] = float(np.min(accuracies))
                summary["max_annotator_accuracy"] = float(np.max(accuracies))

        # Recent anomalies
        if self.anomalies:
            recent_anomalies = sorted(
                self.anomalies,
                key=lambda x: x["timestamp"],
                reverse=True,
            )[:5]
            summary["recent_anomalies"] = recent_anomalies

        return summary

    def _compute_krippendorff_alpha(
        self,
        reliability_data: np.ndarray,
        metric: str = "nominal",
        value_domain: list | None = None,
    ) -> float:
        """
        Compute Krippendorff's alpha coefficient.

        Implementation based on Krippendorff (2004).

        Args:
            reliability_data: 2D array (annotators x items).
            metric: Distance metric (nominal, ordinal, interval, ratio).
            value_domain: Complete set of possible values.

        Returns:
            Alpha coefficient (-1 to 1, higher is better).
        """
        # For nominal data, keep as object type to handle strings
        # For numeric data (ordinal, interval, ratio), convert to float
        if metric in ["interval", "ratio"]:
            reliability_data = np.array(reliability_data, dtype=float)
        else:
            reliability_data = np.array(reliability_data, dtype=object)

        # Get value domain
        if value_domain is None:
            # Extract unique non-NaN values
            if metric in ["interval", "ratio"]:
                mask = ~np.isnan(reliability_data.astype(float))
                value_domain = np.unique(reliability_data[mask])
            else:
                # For nominal/ordinal, filter None and NaN
                flat_data = reliability_data.flatten()
                value_domain = np.unique([v for v in flat_data if v is not None and not (isinstance(v, float) and np.isnan(v))])

        n_values = len(value_domain)
        if n_values < 2:
            logger.warning("Not enough unique values for alpha calculation")
            return np.nan

        # Create coincidence matrix
        coincidence_matrix = np.zeros((n_values, n_values))

        # Value to index mapping
        value_to_idx = {v: i for i, v in enumerate(value_domain)}

        # Compute pairwise coincidences
        n_items = reliability_data.shape[1]

        for item_idx in range(n_items):
            item_values = reliability_data[:, item_idx]

            # Remove NaN/None values
            if metric in ["interval", "ratio"]:
                valid_mask = ~np.isnan(item_values.astype(float))
                item_values = item_values[valid_mask]
            else:
                item_values = np.array([v for v in item_values if v is not None and not (isinstance(v, float) and np.isnan(v))])

            n_annotators = len(item_values)
            if n_annotators < 2:
                continue

            # All pairwise combinations
            for i in range(n_annotators):
                for j in range(n_annotators):
                    if i != j:
                        val_i = item_values[i]
                        val_j = item_values[j]

                        idx_i = value_to_idx.get(val_i)
                        idx_j = value_to_idx.get(val_j)

                        if idx_i is not None and idx_j is not None:
                            # Weight by 1 / (n_annotators - 1)
                            coincidence_matrix[idx_i, idx_j] += 1.0 / (n_annotators - 1)

        # Calculate observed disagreement
        n_total = np.sum(coincidence_matrix)
        if n_total == 0:
            return np.nan

        # Marginal totals
        n_c = np.sum(coincidence_matrix, axis=1)
        n_k = np.sum(coincidence_matrix, axis=0)

        # Distance function
        distance_matrix = self._get_distance_matrix(value_domain, metric)

        # Observed disagreement
        D_o = 0.0
        for c in range(n_values):
            for k in range(n_values):
                D_o += distance_matrix[c, k] * coincidence_matrix[c, k]
        D_o = D_o / n_total

        # Expected disagreement
        D_e = 0.0
        for c in range(n_values):
            for k in range(n_values):
                D_e += distance_matrix[c, k] * n_c[c] * n_k[k]
        D_e = D_e / (n_total * (n_total - 1))

        # Alpha coefficient
        if D_e == 0:
            return 1.0 if D_o == 0 else 0.0

        alpha = 1 - (D_o / D_e)
        return alpha

    def _get_distance_matrix(
        self,
        value_domain: np.ndarray,
        metric: str,
    ) -> np.ndarray:
        """Get distance matrix for given metric."""
        n = len(value_domain)
        distance_matrix = np.zeros((n, n))

        if metric == "nominal":
            # Nominal: 0 if same, 1 if different
            for i in range(n):
                for j in range(n):
                    distance_matrix[i, j] = 0.0 if i == j else 1.0

        elif metric == "ordinal":
            # Ordinal: squared rank difference
            for i in range(n):
                for j in range(n):
                    if i == j:
                        distance_matrix[i, j] = 0.0
                    else:
                        # Sum of ranks between i and j
                        g_i = np.sum(np.arange(i + 1))
                        g_j = np.sum(np.arange(j + 1))
                        distance_matrix[i, j] = (g_i - g_j) ** 2

        elif metric in ["interval", "ratio"]:
            # Interval/Ratio: squared difference
            for i in range(n):
                for j in range(n):
                    distance_matrix[i, j] = (value_domain[i] - value_domain[j]) ** 2

        return distance_matrix

    def _calculate_pairwise_agreement(
        self,
        df: pd.DataFrame,
        annotator_columns: list[str],
    ) -> dict[str, float]:
        """Calculate agreement between each pair of annotators."""
        pairwise = {}

        for i in range(len(annotator_columns)):
            for j in range(i + 1, len(annotator_columns)):
                col1 = annotator_columns[i]
                col2 = annotator_columns[j]

                # Get valid pairs (both annotators have labels)
                valid_mask = (~df[col1].isna()) & (~df[col2].isna())
                valid_df = df[valid_mask]

                if len(valid_df) > 0:
                    agreement = (valid_df[col1] == valid_df[col2]).mean()
                    pair_key = f"{col1}_{col2}"
                    pairwise[pair_key] = float(agreement)

        return pairwise

    def _calculate_per_item_agreement(
        self,
        df: pd.DataFrame,
        annotator_columns: list[str],
    ) -> np.ndarray:
        """Calculate agreement for each item across all annotators."""
        per_item = []

        for _, row in df.iterrows():
            values = [row[col] for col in annotator_columns if not pd.isna(row[col])]

            if len(values) > 1:
                # Most common value count / total values
                from collections import Counter
                counts = Counter(values)
                most_common_count = counts.most_common(1)[0][1]
                agreement = most_common_count / len(values)
                per_item.append(agreement)
            else:
                per_item.append(1.0)  # Single annotator = perfect agreement

        return np.array(per_item)
