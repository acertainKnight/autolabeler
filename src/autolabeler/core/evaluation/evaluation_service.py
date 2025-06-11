"""Service for evaluating labeling performance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from ...config import Settings
from ..base import ConfigurableComponent
from ..configs import EvaluationConfig
from ..utils import evaluation_utils


class EvaluationService(ConfigurableComponent):
    """
    Service for evaluating labeling model performance.

    Provides comprehensive evaluation metrics, confidence analysis,
    and report generation capabilities.

    Example:
        >>> config = EvaluationConfig(save_results=True, create_report=True)
        >>> service = EvaluationService("sentiment", settings)
        >>> results = service.evaluate(test_df, "true_label", "predicted_label", config)
    """

    def __init__(
        self,
        dataset_name: str,
        settings: Settings,
        config: EvaluationConfig | None = None,
        results_dir: Path | None = None,
    ) -> None:
        """Initialize the evaluation service."""
        super().__init__(
            component_type="evaluation_service",
            dataset_name=dataset_name,
            settings=settings,
        )
        self.config = config or EvaluationConfig()
        self.results_dir = results_dir or Path("results") / dataset_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._evaluation_history = []

    def evaluate(
        self,
        df: pd.DataFrame,
        true_label_column: str,
        pred_label_column: str,
        confidence_column: str | None = None,
        output_report_path: Path | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate predictions and optionally generate a report.

        Args:
            df: DataFrame with true and predicted labels.
            true_label_column: Column with ground truth labels.
            pred_label_column: Column with predicted labels.
            confidence_column: Column with confidence scores.
            output_report_path: Path to save a human-readable report.

        Returns:
            A dictionary with comprehensive evaluation metrics.
        """
        valid_df = df.dropna(subset=[true_label_column, pred_label_column]).copy()
        if valid_df.empty:
            logger.error("No valid predictions to evaluate.")
            return {}

        y_true = valid_df[true_label_column]
        y_pred = valid_df[pred_label_column]

        metrics = evaluation_utils.calculate_metrics(y_true, y_pred)
        results = {"metrics": metrics}

        if confidence_column and confidence_column in valid_df.columns:
            results["confidence_analysis"] = evaluation_utils.analyze_confidence(
                y_true, y_pred, valid_df[confidence_column]
            )

        if self.config.save_results:
            results_path = self.results_dir / "evaluation_metrics.json"
            self._save_results(results, results_path)

        if output_report_path or self.config.create_report:
            report_path = output_report_path or self.results_dir / "evaluation_report.md"
            self._create_report(results, report_path)

        # Track in history
        self._evaluation_history.append(results)

        return results

    def evaluate_with_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        true_column: str,
        pred_column: str,
        confidence_column: str | None = None,
        config: EvaluationConfig | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate with explicit train/test split information.

        Provides additional analysis comparing train vs test performance.
        """
        config = config or EvaluationConfig()

        # Evaluate test set
        test_results = self.evaluate(
            test_df, true_column, pred_column, confidence_column, config
        )

        # Evaluate train set if predictions available
        train_results = None
        if pred_column in train_df.columns:
            # Temporarily disable saving/reporting for train evaluation
            train_config = config.model_copy()
            train_config.save_results = False
            train_config.create_report = False

            train_results = self.evaluate(
                train_df, true_column, pred_column, confidence_column, train_config
            )

        # Combine results
        combined_results = {
            "test_results": test_results,
            "train_results": train_results,
            "split_info": {
                "train_size": len(train_df),
                "test_size": len(test_df),
                "train_ratio": len(train_df) / (len(train_df) + len(test_df)),
            },
        }

        # Add overfitting analysis if train results available
        if train_results:
            combined_results["overfitting_analysis"] = self._analyze_overfitting(
                train_results, test_results
            )

        return combined_results

    def _calculate_detailed_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> dict[str, Any]:
        """Calculate detailed classification metrics."""
        # Get per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0
        )

        # Create per-class dictionary
        labels = sorted(set(y_true) | set(y_pred))
        per_class_metrics = {}

        for i, label in enumerate(labels):
            if i < len(precision):  # Handle case where label might not be in predictions
                per_class_metrics[str(label)] = {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]) if i < len(support) else 0,
                }

        # Get confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        return {
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "labels": [str(l) for l in labels],
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            ),
        }

    def _analyze_confidence(
        self,
        df: pd.DataFrame,
        true_column: str,
        pred_column: str,
        confidence_column: str,
    ) -> dict[str, Any]:
        """Analyze confidence scores vs accuracy."""
        # Create confidence bins
        df["confidence_bin"] = pd.cut(
            df[confidence_column],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        )

        # Calculate accuracy per bin
        confidence_analysis = {}

        for bin_label in df["confidence_bin"].cat.categories:
            bin_mask = df["confidence_bin"] == bin_label
            if bin_mask.any():
                bin_df = df[bin_mask]
                bin_accuracy = (
                    bin_df[true_column] == bin_df[pred_column]
                ).mean()

                confidence_analysis[bin_label] = {
                    "count": len(bin_df),
                    "accuracy": float(bin_accuracy),
                    "avg_confidence": float(bin_df[confidence_column].mean()),
                }

        # Overall confidence statistics
        confidence_analysis["overall"] = {
            "mean_confidence": float(df[confidence_column].mean()),
            "std_confidence": float(df[confidence_column].std()),
            "correlation_with_accuracy": float(
                df[confidence_column].corr(
                    (df[true_column] == df[pred_column]).astype(int)
                )
            ),
        }

        return confidence_analysis

    def _analyze_errors(
        self,
        df: pd.DataFrame,
        true_column: str,
        pred_column: str,
    ) -> dict[str, Any]:
        """Analyze prediction errors."""
        # Get error mask
        error_mask = df[true_column] != df[pred_column]
        error_df = df[error_mask]

        if len(error_df) == 0:
            return {"error_rate": 0.0, "num_errors": 0}

        # Count error types
        error_counts = {}
        for _, row in error_df.iterrows():
            error_type = f"{row[true_column]} -> {row[pred_column]}"
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        # Sort by frequency
        sorted_errors = sorted(
            error_counts.items(), key=lambda x: x[1], reverse=True
        )

        return {
            "error_rate": float(len(error_df) / len(df)),
            "num_errors": len(error_df),
            "top_error_types": sorted_errors[:10],  # Top 10 error types
            "error_distribution": error_counts,
        }

    def _analyze_overfitting(
        self,
        train_results: dict[str, Any],
        test_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze potential overfitting."""
        train_acc = train_results.get("accuracy", 0)
        test_acc = test_results.get("accuracy", 0)

        train_f1 = train_results.get("macro_f1", 0)
        test_f1 = test_results.get("macro_f1", 0)

        return {
            "accuracy_gap": float(train_acc - test_acc),
            "f1_gap": float(train_f1 - test_f1),
            "likely_overfitting": (train_acc - test_acc) > 0.1,
            "severity": "high" if (train_acc - test_acc) > 0.2 else
                       "medium" if (train_acc - test_acc) > 0.1 else "low",
        }

    def _save_results(self, results: dict[str, Any], output_path: Path) -> None:
        """Save evaluation results to a JSON file."""
        try:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Evaluation results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")

    def _create_report(self, results: dict[str, Any], output_path: Path) -> None:
        """Create a human-readable evaluation report."""
        try:
            with open(output_path, 'w') as f:
                f.write(f"# Evaluation Report: {self.dataset_name}\n\n")

                metrics = results.get("metrics", {})
                f.write("## Performance Metrics\n\n")
                f.write(f"- **Accuracy:** {metrics.get('accuracy', 0):.4f}\n")
                f.write(f"- **F1 (Weighted):** {metrics.get('f1_weighted', 0):.4f}\n")
                f.write(f"- **Precision (Weighted):** {metrics.get('precision_weighted', 0):.4f}\n")
                f.write(f"- **Recall (Weighted):** {metrics.get('recall_weighted', 0):.4f}\n")
                f.write(f"- **Cohen's Kappa:** {metrics.get('cohen_kappa', 0):.4f}\n\n")

                if "confidence_analysis" in results:
                    conf = results["confidence_analysis"]
                    f.write("## Confidence Analysis\n\n")
                    stats = conf.get("confidence_stats", {})
                    calib = conf.get("calibration", {})
                    f.write(f"- **Mean Confidence:** {stats.get('mean_confidence', 0):.4f}\n")
                    f.write(f"- **Expected Calibration Error (ECE):** {calib.get('expected_calibration_error', 0):.4f}\n\n")

                if "classification_report" in metrics:
                    f.write("## Classification Report\n\n")
                    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
                    f.write(report_df.to_markdown(index=True))

            logger.info(f"Evaluation report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to create evaluation report: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about evaluations."""
        return {
            "dataset_name": self.dataset_name,
            "num_evaluations": len(self._evaluation_history),
            "latest_accuracy": (
                self._evaluation_history[-1].get("accuracy", 0)
                if self._evaluation_history else 0
            ),
        }
