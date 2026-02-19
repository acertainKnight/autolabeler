from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
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
    """Calculate a comprehensive set of classification metrics.

    Filters out -99 (not applicable) and NaN values before calculating metrics.
    -99 indicates "not labeled" or "not applicable" and should not be counted.
    """
    # Filter out -99 and NaN values before evaluation
    filtered_true = []
    filtered_pred = []

    for yt, yp in zip(y_true, y_pred):
        yt_str = str(yt)
        yp_str = str(yp)

        # Skip -99 (not applicable) and NaN values
        if yt_str == "-99" or yp_str == "-99":
            continue
        if yt_str == "nan" or yp_str == "nan" or yt_str == "None" or yp_str == "None":
            continue

        filtered_true.append(yt_str)
        filtered_pred.append(yp_str)

    if len(filtered_true) == 0:
        logger.warning("No valid labels found after filtering -99 and NaN")
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "num_samples": 0,
            "num_filtered": len(y_true),
            "per_class": {},
            "confusion_matrix": [],
            "labels": [],
        }

    y_true_str = filtered_true
    y_pred_str = filtered_pred
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
        "num_filtered": len(y_true) - len(y_true_str),  # How many were filtered out
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

    Filters out -99 (not applicable) values before analysis.

    Args:
        y_true: Series of ground truth labels.
        y_pred: Series of predicted labels.
        confidences: Series of prediction confidence scores.

    Returns:
        A dictionary containing confidence analysis.
    """
    # Filter out -99 and NaN values
    y_true_str = y_true.astype(str)
    y_pred_str = y_pred.astype(str)

    valid_mask = (y_true_str != "-99") & (y_pred_str != "-99")
    valid_mask &= (y_true_str != "nan") & (y_pred_str != "nan")

    y_true_filtered = y_true_str[valid_mask]
    y_pred_filtered = y_pred_str[valid_mask]
    confidences_filtered = confidences[valid_mask]

    if len(y_true_filtered) == 0:
        logger.warning("No valid labels for confidence analysis after filtering -99")
        return {
            "mean_confidence": 0.0,
            "std_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "num_filtered": len(y_true),
        }

    correct = (y_true_filtered == y_pred_filtered)

    conf_stats = {
        "mean_confidence": float(confidences_filtered.mean()),
        "std_confidence": float(confidences_filtered.std()),
        "min_confidence": float(confidences_filtered.min()),
        "max_confidence": float(confidences_filtered.max()),
        "mean_confidence_correct": float(confidences_filtered[correct].mean()) if correct.any() else 0.0,
        "mean_confidence_incorrect": float(confidences_filtered[~correct].mean()) if (~correct).any() else 0.0,
        "num_samples": len(confidences_filtered),
        "num_filtered": len(y_true) - len(y_true_filtered),
    }

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    threshold_analysis = {}
    for threshold in thresholds:
        high_conf_mask = confidences_filtered >= threshold
        if high_conf_mask.sum() > 0:
            threshold_analysis[f"threshold_{threshold}"] = {
                "accuracy": float(correct[high_conf_mask].mean()),
                "coverage": float(high_conf_mask.mean()),
                "count": int(high_conf_mask.sum()),
            }

    calibration = _compute_calibration(confidences_filtered, correct)

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


def calculate_ordinal_metrics(y_true: list, y_pred: list) -> dict[str, float]:
    """
    Calculate ordinal classification metrics (MAE, off-by-one accuracy).
    
    Useful for tasks with ordered labels like hawk_dove (-2, -1, 0, 1, 2).
    Excludes -99 (not applicable) from ordinal calculations.
    
    Args:
        y_true: List of ground truth labels (as strings)
        y_pred: List of predicted labels (as strings)
        
    Returns:
        Dictionary with MAE and off-by-one accuracy metrics
    """
    # Convert to lists if needed
    y_true_list = list(y_true) if not isinstance(y_true, list) else y_true
    y_pred_list = list(y_pred) if not isinstance(y_pred, list) else y_pred
    
    # Convert to string and filter valid numeric pairs
    valid_pairs = []
    for yt, yp in zip(y_true_list, y_pred_list):
        yt_str = str(yt)
        yp_str = str(yp)
        
        # Skip -99 (not applicable) and NaN values
        if yt_str == "-99" or yp_str == "-99":
            continue
        if yt_str == "nan" or yp_str == "nan":
            continue
            
        # Try to convert to float to validate numeric values
        try:
            yt_num = float(yt_str)
            yp_num = float(yp_str)
            valid_pairs.append((yt_num, yp_num))
        except (ValueError, TypeError):
            # Skip invalid numeric values (e.g., "+", "]}1", etc.)
            continue
    
    if len(valid_pairs) == 0:
        return {
            "mae": 0.0,
            "off_by_one_accuracy": 0.0,
            "valid_samples": 0,
        }
    
    # Separate into arrays
    y_true_numeric = np.array([pair[0] for pair in valid_pairs])
    y_pred_numeric = np.array([pair[1] for pair in valid_pairs])
    
    # Calculate MAE
    mae = float(np.mean(np.abs(y_true_numeric - y_pred_numeric)))
    
    # Calculate off-by-one accuracy (exact match or within 1)
    off_by_one = np.abs(y_true_numeric - y_pred_numeric) <= 1
    off_by_one_accuracy = float(np.mean(off_by_one))
    
    return {
        "mae": mae,
        "off_by_one_accuracy": off_by_one_accuracy,
        "valid_samples": len(valid_pairs),
    }


def calculate_metrics_exclude_zero(y_true: list, y_pred: list) -> dict[str, Any]:
    """
    Calculate metrics excluding zero class (neutral/no position).
    
    Filters out both -99 (not applicable), NaN, and 0 (neutral) values.
    Useful for evaluating performance on only strong directional labels.
    
    Args:
        y_true: List of ground truth labels
        y_pred: List of predicted labels
        
    Returns:
        Dictionary with classification metrics excluding zero class
    """
    # Filter out -99, NaN, and 0 values
    filtered_true = []
    filtered_pred = []

    for yt, yp in zip(y_true, y_pred):
        yt_str = str(yt)
        yp_str = str(yp)

        # Skip -99 (not applicable), NaN, and 0 (neutral) values
        if yt_str in ["-99", "nan", "None", "0", "0.0"]:
            continue
        if yp_str in ["-99", "nan", "None", "0", "0.0"]:
            continue

        filtered_true.append(yt_str)
        filtered_pred.append(yp_str)

    if len(filtered_true) == 0:
        logger.warning("No valid labels found after filtering -99, NaN, and 0")
        return {
            "accuracy_exclude_zero": 0.0,
            "f1_macro_exclude_zero": 0.0,
            "precision_macro_exclude_zero": 0.0,
            "recall_macro_exclude_zero": 0.0,
            "num_samples_exclude_zero": 0,
            "num_filtered_exclude_zero": len(y_true),
        }

    y_true_str = filtered_true
    y_pred_str = filtered_pred
    labels = sorted(list(set(y_true_str) | set(y_pred_str)))

    # Basic metrics excluding zero
    metrics = {
        "accuracy_exclude_zero": accuracy_score(y_true_str, y_pred_str),
        "f1_macro_exclude_zero": f1_score(y_true_str, y_pred_str, average="macro", zero_division=0),
        "precision_macro_exclude_zero": precision_score(
            y_true_str, y_pred_str, average="macro", zero_division=0
        ),
        "recall_macro_exclude_zero": recall_score(y_true_str, y_pred_str, average="macro", zero_division=0),
        "num_samples_exclude_zero": len(y_true_str),
        "num_filtered_exclude_zero": len(y_true) - len(y_true_str),
        "labels_exclude_zero": labels,
    }

    return metrics


def bucket_to_3_class(labels: list) -> list:
    """
    Convert 5-class ordinal labels to 3-class by bucketing extreme values.
    
    Converts: -2,-1 -> -1, 0 -> 0, 1,2 -> 1
    This creates a simplified dovish/neutral/hawkish classification.
    
    Args:
        labels: List of labels (strings or numeric)
        
    Returns:
        List of bucketed labels as strings
    """
    bucketed = []
    for label in labels:
        label_str = str(label)
        
        # Handle special cases first
        if label_str in ["-99", "nan", "None"]:
            bucketed.append(label_str)
            continue
            
        try:
            label_num = float(label_str)
            if label_num <= -1:  # -2, -1 -> -1 (dovish)
                bucketed.append("-1")
            elif label_num == 0:   # 0 -> 0 (neutral)
                bucketed.append("0")
            elif label_num >= 1:   # 1, 2 -> 1 (hawkish)
                bucketed.append("1")
            else:
                # Fallback for unexpected values
                bucketed.append(label_str)
        except (ValueError, TypeError):
            # Keep non-numeric labels as-is
            bucketed.append(label_str)
            
    return bucketed


def calculate_3_class_metrics(y_true: list, y_pred: list) -> dict[str, Any]:
    """
    Calculate metrics for 3-class bucketed version (-2,-1 -> -1, 0 -> 0, 1,2 -> 1).
    
    Args:
        y_true: List of ground truth labels
        y_pred: List of predicted labels
        
    Returns:
        Dictionary with 3-class classification metrics
    """
    # Bucket to 3 classes
    y_true_3class = bucket_to_3_class(y_true)
    y_pred_3class = bucket_to_3_class(y_pred)
    
    # Calculate standard metrics on bucketed data
    metrics_3class = calculate_metrics(y_true_3class, y_pred_3class)
    
    # Rename keys to indicate 3-class version
    renamed_metrics = {}
    for key, value in metrics_3class.items():
        if key.startswith("per_class"):
            renamed_metrics[f"3class_{key}"] = value
        else:
            renamed_metrics[f"3class_{key}"] = value
    
    return renamed_metrics


def calculate_3_class_exclude_zero_metrics(y_true: list, y_pred: list) -> dict[str, Any]:
    """
    Calculate 3-class metrics excluding zero/neutral class.
    
    First buckets to 3 classes, then excludes 0 (neutral).
    Results in binary dovish (-1) vs hawkish (1) classification.
    
    Args:
        y_true: List of ground truth labels
        y_pred: List of predicted labels
        
    Returns:
        Dictionary with 3-class metrics excluding neutral
    """
    # Bucket to 3 classes first
    y_true_3class = bucket_to_3_class(y_true)
    y_pred_3class = bucket_to_3_class(y_pred)
    
    # Then exclude zero
    metrics_3class_no_zero = calculate_metrics_exclude_zero(y_true_3class, y_pred_3class)
    
    # Rename keys to indicate 3-class exclude zero version
    renamed_metrics = {}
    for key, value in metrics_3class_no_zero.items():
        renamed_metrics[f"3class_{key}"] = value
    
    return renamed_metrics
