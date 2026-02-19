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

from ..utils import evaluation_utils


class EvaluationService:
    """
    Service for evaluating labeling model performance.

    Provides comprehensive evaluation metrics, confidence analysis,
    and report generation capabilities.

    Example:
        >>> service = EvaluationService("sentiment")
        >>> results = service.evaluate(
        ...     test_df, "true_label", "predicted_label",
        ...     save_results=True, create_report=True
        ... )
    """

    def __init__(
        self,
        dataset_name: str,
        results_dir: Path | None = None,
    ) -> None:
        """Initialize the evaluation service.
        
        Parameters:
            dataset_name: Name of dataset (for organizing results)
            results_dir: Directory to save results (default: "results/{dataset_name}")
        """
        self.dataset_name = dataset_name
        self.results_dir = results_dir or Path("results") / dataset_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._evaluation_history = []
        
        logger.info(f"Initialized EvaluationService for dataset: {dataset_name}")

    def evaluate(
        self,
        df: pd.DataFrame,
        true_label_column: str,
        pred_label_column: str,
        confidence_column: str | None = None,
        output_report_path: Path | None = None,
        task_name: str | None = None,
        use_comprehensive_metrics: bool = False,
        save_results: bool = False,
        create_report: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate predictions and optionally generate a report.

        Args:
            df: DataFrame with true and predicted labels.
            true_label_column: Column with ground truth labels.
            pred_label_column: Column with predicted labels.
            confidence_column: Column with confidence scores.
            output_report_path: Path to save a human-readable report.
            task_name: Optional task name for ordinal metrics.
            use_comprehensive_metrics: If True, includes exclude-zero and 3-class metrics.
            save_results: Whether to save results to JSON.
            create_report: Whether to create markdown report.

        Returns:
            A dictionary with comprehensive evaluation metrics.
        """
        valid_df = df.dropna(subset=[true_label_column, pred_label_column]).copy()
        if valid_df.empty:
            logger.error("No valid predictions to evaluate.")
            return {}

        y_true = valid_df[true_label_column]
        y_pred = valid_df[pred_label_column]

        if use_comprehensive_metrics and task_name:
            # Use comprehensive metrics including all variants
            metrics = self._calculate_comprehensive_metrics(
                y_true.tolist(), y_pred.tolist(), task_name
            )
        else:
            # Use standard metrics only
            metrics = evaluation_utils.calculate_metrics(y_true, y_pred)
            
            # Add ordinal metrics for hawk_dove task if task name provided
            if task_name == "hawk_dove":
                ordinal_metrics = evaluation_utils.calculate_ordinal_metrics(
                    y_true.tolist(), y_pred.tolist()
                )
                metrics.update(ordinal_metrics)

        results = {"metrics": metrics}

        if confidence_column and confidence_column in valid_df.columns:
            results["confidence_analysis"] = evaluation_utils.analyze_confidence(
                y_true, y_pred, valid_df[confidence_column]
            )

        if save_results:
            results_path = self.results_dir / "evaluation_metrics.json"
            self._save_results(results, results_path)

        if output_report_path or create_report:
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
        save_results: bool = False,
        create_report: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate with explicit train/test split information.

        Provides additional analysis comparing train vs test performance.
        """
        # Evaluate test set
        test_results = self.evaluate(
            test_df, true_column, pred_column, confidence_column,
            save_results=save_results, create_report=create_report
        )

        # Evaluate train set if predictions available
        train_results = None
        if pred_column in train_df.columns:
            # Don't save/report for train evaluation
            train_results = self.evaluate(
                train_df, true_column, pred_column, confidence_column,
                save_results=False, create_report=False
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

    def calculate_inter_rater_agreement(
        self,
        df: pd.DataFrame,
        task_name: str,
        rater_columns: list[str],
    ) -> dict[str, Any]:
        """Calculate inter-rater agreement metrics between multiple annotators.

        This is useful for evaluating agreement between:
        - Multiple human annotators
        - Multiple LLM annotators
        - Human annotators vs LLM annotators

        Args:
            df: DataFrame with predictions from multiple raters
            task_name: Task name for reporting
            rater_columns: List of column names containing rater predictions
                          (e.g., ["label_task_llm1", "label_task_llm2", "label_task_llm3"])

        Returns:
            Dictionary with inter-rater agreement metrics:
                - pairwise_kappa: Cohen's Kappa for each pair of raters
                - mean_kappa: Average Cohen's Kappa across all pairs
                - fleiss_kappa: Fleiss' Kappa (agreement across all raters)
                - krippendorff_alpha: Krippendorff's Alpha (optional, if available)
                - percent_agreement: Simple percentage agreement
                - unanimous_rate: Rate of unanimous agreement

        Example:
            >>> service = EvaluationService("fed_headlines", settings)
            >>> agreement = service.calculate_inter_rater_agreement(
            ...     voting_df,
            ...     "relevancy",
            ...     ["label_relevancy_gpt4", "label_relevancy_claude", "label_relevancy_gemini"]
            ... )
            >>> print(f"Mean Kappa: {agreement['mean_kappa']:.3f}")
        """
        from sklearn.metrics import cohen_kappa_score

        # Validate columns exist
        missing_cols = [col for col in rater_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Drop rows with any NaN values in rater columns
        # Failed API calls now return None (which becomes NaN), so dropna() handles them
        valid_df = df[rater_columns].copy()
        valid_df = valid_df.dropna()
        
        if len(valid_df) == 0:
            logger.warning("No valid rows after removing NaN and invalid values - no agreement metrics can be calculated")
            logger.warning("This typically happens when annotators stopped at different points or all failed to label")
            return {}

        rows_excluded = len(df) - len(valid_df)
        logger.info(f"Calculating inter-rater agreement for {task_name}")
        logger.info(f"  Raters: {len(rater_columns)}")
        logger.info(f"  Total examples: {len(df)}")
        logger.info(f"  Valid examples (all raters labeled): {len(valid_df)}")
        if rows_excluded > 0:
            logger.warning(f"  ⚠ Excluded {rows_excluded} rows with missing/failed labels")

        # Convert all columns to string type to ensure consistency
        # sklearn's cohen_kappa_score requires consistent data types
        for col in rater_columns:
            valid_df[col] = valid_df[col].astype(str)
        
        # Log unique values for debugging
        unique_values = set()
        for col in rater_columns:
            unique_values.update(valid_df[col].unique())
        logger.info(f"  Unique label values across all raters: {sorted(unique_values)}")

        # Calculate pairwise Cohen's Kappa
        pairwise_kappa = {}
        kappa_values = []

        for i in range(len(rater_columns)):
            for j in range(i + 1, len(rater_columns)):
                rater1, rater2 = rater_columns[i], rater_columns[j]
                kappa = cohen_kappa_score(valid_df[rater1], valid_df[rater2])
                pair_name = f"{rater1}_vs_{rater2}"
                pairwise_kappa[pair_name] = float(kappa)
                kappa_values.append(kappa)

                logger.info(f"  {rater1} vs {rater2}: κ = {kappa:.3f}")

        # Calculate Fleiss' Kappa (multi-rater agreement)
        try:
            from statsmodels.stats.inter_rater import fleiss_kappa as fleiss_kappa_calc
            from statsmodels.stats.inter_rater import aggregate_raters

            # Aggregate rater data
            rater_matrix = valid_df[rater_columns].values
            agg_data, categories = aggregate_raters(rater_matrix)
            fleiss = float(fleiss_kappa_calc(agg_data))
            logger.info(f"  Fleiss' Kappa: κ = {fleiss:.3f}")
        except ImportError:
            logger.warning("statsmodels not installed, skipping Fleiss' Kappa")
            fleiss = None
        except Exception as e:
            logger.warning(f"Could not calculate Fleiss' Kappa: {e}")
            fleiss = None

        # Calculate simple percent agreement
        # Count rows where all raters agree
        unanimous_mask = valid_df.apply(lambda row: len(set(row)) == 1, axis=1)
        unanimous_count = unanimous_mask.sum()
        unanimous_rate = unanimous_count / len(valid_df)

        logger.info(f"  Unanimous agreement rate: {unanimous_rate:.3f}")

        return {
            "task_name": task_name,
            "num_raters": len(rater_columns),
            "num_examples": len(valid_df),
            "pairwise_kappa": pairwise_kappa,
            "mean_kappa": float(sum(kappa_values) / len(kappa_values)) if kappa_values else 0.0,
            "min_kappa": float(min(kappa_values)) if kappa_values else 0.0,
            "max_kappa": float(max(kappa_values)) if kappa_values else 0.0,
            "fleiss_kappa": fleiss,
            "unanimous_rate": float(unanimous_rate),
            "unanimous_count": int(unanimous_count),
        }

    def compare_human_vs_llm(
        self,
        df: pd.DataFrame,
        task_name: str,
        human_label_column: str,
        llm_label_column: str,
        llm_confidence_column: str | None = None,
    ) -> dict[str, Any]:
        """Compare LLM predictions against human labels.

        Args:
            df: DataFrame with both human and LLM labels
            task_name: Task name for reporting
            human_label_column: Column with human labels (ground truth)
            llm_label_column: Column with LLM predictions
            llm_confidence_column: Optional column with LLM confidence scores

        Returns:
            Dictionary with comparison metrics:
                - accuracy: Overall accuracy
                - cohens_kappa: Cohen's Kappa agreement
                - f1_weighted: Weighted F1 score
                - confusion_matrix: Confusion matrix
                - per_class_metrics: Precision/recall/F1 per class
                - confidence_analysis: Confidence vs accuracy (if confidence provided)

        Example:
            >>> comparison = service.compare_human_vs_llm(
            ...     df, "relevancy", "label_relevancy", "label_relevancy_consensus"
            ... )
        """
        # Use existing evaluate method
        results = self.evaluate(
            df=df,
            true_label_column=human_label_column,
            pred_label_column=llm_label_column,
            confidence_column=llm_confidence_column,
        )

        # Add task name
        results["task_name"] = task_name
        results["comparison_type"] = "human_vs_llm"

        logger.info(f"Human vs LLM comparison for {task_name}:")
        if "metrics" in results:
            logger.info(f"  Accuracy: {results['metrics'].get('accuracy', 0):.3f}")
            logger.info(f"  Cohen's Kappa: {results['metrics'].get('cohen_kappa', 0):.3f}")
            logger.info(f"  F1 (weighted): {results['metrics'].get('f1_weighted', 0):.3f}")

        return results

    def evaluate_individual_models(
        self,
        df: pd.DataFrame,
        task_name: str,
        human_label_column: str,
        model_columns: list[str],
        model_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate individual model performance on a task.

        Args:
            df: DataFrame with human labels and individual model predictions
            task_name: Name of the task being evaluated
            human_label_column: Column with ground truth human labels
            model_columns: List of column names with individual model predictions
            model_names: Optional list of model names (defaults to column names)

        Returns:
            Dictionary with individual model evaluation results
        """
        if model_names is None:
            model_names = model_columns

        results = {
            "task_name": task_name,
            "individual_models": {},
            "model_comparison": {},
        }

        valid_df = df.dropna(subset=[human_label_column]).copy()
        if valid_df.empty:
            logger.error(f"No valid human labels found for {task_name}")
            return results

        y_true = valid_df[human_label_column].tolist()

        # Evaluate each model individually
        model_metrics = {}
        for model_col, model_name in zip(model_columns, model_names):
            if model_col not in valid_df.columns:
                logger.warning(f"Model column {model_col} not found, skipping")
                continue

            model_df = valid_df.dropna(subset=[model_col]).copy()
            if model_df.empty:
                logger.warning(f"No valid predictions for {model_name}, skipping")
                continue

            y_pred = model_df[model_col].tolist()
            y_true_model = model_df[human_label_column].tolist()

            # Calculate comprehensive metrics for this model
            metrics = self._calculate_comprehensive_metrics(y_true_model, y_pred, task_name)
            
            model_metrics[model_name] = {
                "metrics": metrics,
                "num_predictions": len(y_pred),
                "coverage": len(y_pred) / len(valid_df),  # What fraction of examples this model labeled
            }

            logger.info(f"Individual model evaluation - {model_name}:")
            logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
            logger.info(f"  F1 (macro): {metrics.get('f1_macro', 0):.3f}")
            logger.info(f"  Coverage: {model_metrics[model_name]['coverage']:.1%}")

        results["individual_models"] = model_metrics

        # Create comparison summary
        if model_metrics:
            comparison = {}
            for metric_name in ["accuracy", "f1_macro", "precision_macro", "recall_macro"]:
                metric_values = {}
                for model_name, model_data in model_metrics.items():
                    metric_values[model_name] = model_data["metrics"].get(metric_name, 0.0)
                
                comparison[metric_name] = {
                    "values": metric_values,
                    "best_model": max(metric_values.items(), key=lambda x: x[1])[0] if metric_values else None,
                    "worst_model": min(metric_values.items(), key=lambda x: x[1])[0] if metric_values else None,
                    "range": max(metric_values.values()) - min(metric_values.values()) if metric_values else 0.0,
                }

            results["model_comparison"] = comparison

        return results

    def _calculate_comprehensive_metrics(
        self, 
        y_true: list, 
        y_pred: list, 
        task_name: str
    ) -> dict[str, Any]:
        """
        Calculate comprehensive metrics including standard, exclude-zero, and 3-class variants.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            task_name: Name of the task (used to determine if ordinal metrics apply)

        Returns:
            Dictionary with all metric variants
        """
        # Standard metrics (includes all classes, excludes -99)
        standard_metrics = evaluation_utils.calculate_metrics(y_true, y_pred)
        
        # Exclude-zero metrics (excludes -99 and 0)
        exclude_zero_metrics = evaluation_utils.calculate_metrics_exclude_zero(y_true, y_pred)
        
        # 3-class metrics (buckets -2,-1 -> -1, 0 -> 0, 1,2 -> 1)
        three_class_metrics = evaluation_utils.calculate_3_class_metrics(y_true, y_pred)
        
        # 3-class exclude zero metrics (binary: -1 vs 1)
        three_class_exclude_zero_metrics = evaluation_utils.calculate_3_class_exclude_zero_metrics(y_true, y_pred)
        
        # Combine all metrics
        comprehensive_metrics = {**standard_metrics}
        comprehensive_metrics.update(exclude_zero_metrics)
        comprehensive_metrics.update(three_class_metrics)
        comprehensive_metrics.update(three_class_exclude_zero_metrics)
        
        # Add ordinal metrics for hawk_dove task
        if task_name == "hawk_dove":
            ordinal_metrics = evaluation_utils.calculate_ordinal_metrics(y_true, y_pred)
            comprehensive_metrics.update(ordinal_metrics)
            
            # Also calculate ordinal metrics for 3-class version
            y_true_3class = evaluation_utils.bucket_to_3_class(y_true)
            y_pred_3class = evaluation_utils.bucket_to_3_class(y_pred)
            ordinal_3class_metrics = evaluation_utils.calculate_ordinal_metrics(y_true_3class, y_pred_3class)
            
            # Rename 3-class ordinal metrics
            for key, value in ordinal_3class_metrics.items():
                comprehensive_metrics[f"3class_{key}"] = value
        
        return comprehensive_metrics
