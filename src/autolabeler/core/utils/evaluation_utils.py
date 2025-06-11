from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(y_true: list, y_pred: list) -> dict[str, Any]:
    """Calculate a comprehensive set of classification metrics."""
    # Ensure all labels are strings to prevent sorting errors with mixed types
    y_true_str = [str(label) for label in y_true]
    y_pred_str = [str(label) for label in y_pred]
    labels = sorted(list(set(y_true_str) | set(y_pred_str)))

    report = classification_report(
        y_true_str, y_pred_str, output_dict=True, labels=labels, zero_division=0
    )

    # Basic metrics
    metrics = {
        "accuracy": accuracy_score(y_true_str, y_pred_str),
        "f1_macro": f1_score(y_true_str, y_pred_str, average="macro", zero_division=0),
        "precision_macro": precision_score(
            y_true_str, y_pred_str, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true_str, y_pred_str, average="macro", zero_division=0),
        "num_samples": len(y_true_str),
    }

    # Per-class metrics
    per_class_metrics = {}
    for label, data in report.items():
        if label in labels:
            per_class_metrics[f"f1_{label}"] = data["f1-score"]
            per_class_metrics[f"precision_{label}"] = data["precision"]
            per_class_metrics[f"recall_{label}"] = data["support"]
            per_class_metrics[f"support_{label}"] = data["support"]
    metrics["per_class"] = per_class_metrics

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true_str, y_pred_str, labels=labels)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["labels"] = labels
    except Exception as e:
        logger.warning(f"Could not generate confusion matrix: {e}")
        metrics["confusion_matrix"] = []
        metrics["labels"] = []

    return metrics


def analyze_confidence(
    y_true: pd.Series, y_pred: pd.Series, confidences: pd.Series
) -> dict[str, Any]:
    """
    Analyze prediction confidence and its relationship to accuracy.

    Args:
        y_true: Series of ground truth labels.
        y_pred: Series of predicted labels.
        confidences: Series of prediction confidence scores.

    Returns:
        A dictionary containing confidence analysis.
    """
    correct = (y_true.astype(str) == y_pred.astype(str))

    conf_stats = {
        "mean_confidence": float(confidences.mean()),
        "std_confidence": float(confidences.std()),
        "min_confidence": float(confidences.min()),
        "max_confidence": float(confidences.max()),
        "mean_confidence_correct": float(confidences[correct].mean()) if correct.any() else 0.0,
        "mean_confidence_incorrect": float(confidences[~correct].mean()) if (~correct).any() else 0.0,
    }

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    threshold_analysis = {}
    for threshold in thresholds:
        high_conf_mask = confidences >= threshold
        if high_conf_mask.sum() > 0:
            threshold_analysis[f"threshold_{threshold}"] = {
                "accuracy": float(correct[high_conf_mask].mean()),
                "coverage": float(high_conf_mask.mean()),
                "count": int(high_conf_mask.sum()),
            }

    calibration = _compute_calibration(confidences, correct)

    return {
        "confidence_stats": conf_stats,
        "threshold_analysis": threshold_analysis,
        "calibration": calibration,
    }


def _compute_calibration(
    confidences: pd.Series, correct: pd.Series, n_bins: int = 10
) -> dict[str, Any]:
    """Compute confidence calibration metrics."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]

    calibration_data = []
    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            bin_confidence = confidences[in_bin].mean()
            bin_accuracy = correct[in_bin].mean()
            ece += prop_in_bin * abs(bin_confidence - bin_accuracy)
            calibration_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "bin_confidence": float(bin_confidence),
                "bin_accuracy": float(bin_accuracy),
                "proportion": float(prop_in_bin),
                "count": int(in_bin.sum()),
            })

    return {
        "bins": calibration_data,
        "expected_calibration_error": float(ece),
        "n_bins": n_bins,
    }
