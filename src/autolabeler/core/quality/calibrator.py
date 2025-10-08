"""
Confidence calibration for model predictions.

This module provides methods for calibrating confidence scores to better reflect
true prediction accuracy using Temperature Scaling and Platt Scaling.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


class ConfidenceCalibrator:
    """
    Calibrates model confidence scores to better align with true accuracy.

    Supports Temperature Scaling and Platt Scaling for confidence calibration,
    and provides metrics like Expected Calibration Error (ECE) and Brier Score.

    Example:
        >>> calibrator = ConfidenceCalibrator(method="temperature")
        >>> calibrator.fit(confidence_scores, true_labels, predicted_labels)
        >>> calibrated = calibrator.calibrate(new_confidence_scores)
        >>> metrics = calibrator.evaluate_calibration(test_confidence, test_true, test_pred)
    """

    def __init__(
        self,
        method: Literal["temperature", "platt"] = "temperature",
        n_bins: int = 10,
    ) -> None:
        """
        Initialize the confidence calibrator.

        Args:
            method: Calibration method ("temperature" or "platt").
            n_bins: Number of bins for ECE calculation.
        """
        self.method = method
        self.n_bins = n_bins
        self.is_fitted = False

        # Calibration parameters
        self.temperature: float = 1.0  # For temperature scaling
        self.platt_model: LogisticRegression | None = None  # For Platt scaling

        # Calibration history
        self.calibration_history: list[dict[str, Any]] = []

    def fit(
        self,
        confidence_scores: np.ndarray | pd.Series | list[float],
        true_labels: np.ndarray | pd.Series | list[str],
        predicted_labels: np.ndarray | pd.Series | list[str],
    ) -> ConfidenceCalibrator:
        """
        Fit the calibration model to training data.

        Args:
            confidence_scores: Confidence scores from the model (0-1).
            true_labels: Ground truth labels.
            predicted_labels: Predicted labels from the model.

        Returns:
            Self for method chaining.

        Example:
            >>> calibrator = ConfidenceCalibrator()
            >>> calibrator.fit(train_conf, train_true, train_pred)
        """
        # Convert to numpy arrays
        confidence_scores = np.asarray(confidence_scores, dtype=float)
        true_labels = np.asarray(true_labels)
        predicted_labels = np.asarray(predicted_labels)

        # Validate inputs
        if len(confidence_scores) != len(true_labels) or len(confidence_scores) != len(predicted_labels):
            raise ValueError("All input arrays must have the same length")

        if not np.all((confidence_scores >= 0) & (confidence_scores <= 1)):
            raise ValueError("Confidence scores must be between 0 and 1")

        # Create binary correctness indicator
        correctness = (true_labels == predicted_labels).astype(int)

        if self.method == "temperature":
            self._fit_temperature_scaling(confidence_scores, correctness)
        elif self.method == "platt":
            self._fit_platt_scaling(confidence_scores, correctness)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True

        # Store calibration info
        self.calibration_history.append({
            "method": self.method,
            "n_samples": len(confidence_scores),
            "temperature": self.temperature if self.method == "temperature" else None,
            "mean_confidence_before": float(np.mean(confidence_scores)),
            "accuracy": float(np.mean(correctness)),
        })

        logger.info(f"Calibrator fitted using {self.method} scaling on {len(confidence_scores)} samples")
        return self

    def calibrate(
        self,
        confidence_scores: np.ndarray | pd.Series | list[float],
    ) -> np.ndarray:
        """
        Calibrate confidence scores using the fitted model.

        Args:
            confidence_scores: Raw confidence scores to calibrate.

        Returns:
            Calibrated confidence scores.

        Example:
            >>> calibrated_scores = calibrator.calibrate([0.9, 0.8, 0.7])
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibrating scores")

        confidence_scores = np.asarray(confidence_scores, dtype=float)

        if self.method == "temperature":
            return self._apply_temperature_scaling(confidence_scores)
        elif self.method == "platt":
            return self._apply_platt_scaling(confidence_scores)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

    def evaluate_calibration(
        self,
        confidence_scores: np.ndarray | pd.Series | list[float],
        true_labels: np.ndarray | pd.Series | list[str],
        predicted_labels: np.ndarray | pd.Series | list[str],
        apply_calibration: bool = True,
    ) -> dict[str, Any]:
        """
        Evaluate calibration quality using multiple metrics.

        Args:
            confidence_scores: Confidence scores to evaluate.
            true_labels: Ground truth labels.
            predicted_labels: Predicted labels.
            apply_calibration: Whether to apply calibration before evaluation.

        Returns:
            Dictionary containing ECE, Brier score, log loss, and calibration bins.

        Example:
            >>> metrics = calibrator.evaluate_calibration(test_conf, test_true, test_pred)
            >>> print(f"ECE: {metrics['expected_calibration_error']:.4f}")
        """
        # Convert to numpy arrays
        confidence_scores = np.asarray(confidence_scores, dtype=float)
        true_labels = np.asarray(true_labels)
        predicted_labels = np.asarray(predicted_labels)

        # Apply calibration if requested
        if apply_calibration and self.is_fitted:
            confidence_scores = self.calibrate(confidence_scores)

        # Create binary correctness indicator
        correctness = (true_labels == predicted_labels).astype(int)

        # Calculate ECE
        ece, bins_data = self._compute_ece(confidence_scores, correctness)

        # Calculate Brier score (handle degenerate case with only one class)
        try:
            brier = brier_score_loss(correctness, confidence_scores)
        except ValueError as e:
            if "only one label" in str(e):
                # Degenerate case: all predictions are same class
                # Brier score = mean squared error when all labels are 0 or 1
                brier = float(np.mean((correctness - confidence_scores) ** 2))
            else:
                raise

        # Calculate log loss (with small epsilon to avoid log(0))
        epsilon = 1e-10
        conf_clipped = np.clip(confidence_scores, epsilon, 1 - epsilon)
        try:
            logloss = log_loss(correctness, conf_clipped)
        except ValueError as e:
            if "only one label" in str(e):
                # Degenerate case: compute log loss manually
                if correctness[0] == 1:
                    logloss = float(-np.mean(np.log(conf_clipped)))
                else:
                    logloss = float(-np.mean(np.log(1 - conf_clipped)))
            else:
                raise

        # Maximum Calibration Error (MCE)
        mce = max(bin_data["calibration_error"] for bin_data in bins_data) if bins_data else 0.0

        # Average confidence and accuracy
        mean_confidence = float(np.mean(confidence_scores))
        accuracy = float(np.mean(correctness))

        return {
            "expected_calibration_error": float(ece),
            "maximum_calibration_error": float(mce),
            "brier_score": float(brier),
            "log_loss": float(logloss),
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
            "calibration_gap": abs(mean_confidence - accuracy),
            "bins": bins_data,
            "n_samples": len(confidence_scores),
            "is_calibrated": apply_calibration and self.is_fitted,
        }

    def _fit_temperature_scaling(
        self,
        confidence_scores: np.ndarray,
        correctness: np.ndarray,
    ) -> None:
        """
        Fit temperature scaling parameter.

        Uses grid search to find optimal temperature that minimizes ECE.
        """
        best_temperature = 1.0
        best_ece = float('inf')

        # Convert confidences to logits (inverse sigmoid)
        epsilon = 1e-10
        conf_clipped = np.clip(confidence_scores, epsilon, 1 - epsilon)
        logits = np.log(conf_clipped / (1 - conf_clipped))

        # Grid search over temperature values
        temperatures = np.concatenate([
            np.linspace(0.1, 1.0, 10),
            np.linspace(1.0, 5.0, 20),
        ])

        for temp in temperatures:
            # Apply temperature scaling
            scaled_logits = logits / temp
            scaled_conf = 1 / (1 + np.exp(-scaled_logits))

            # Calculate ECE
            ece, _ = self._compute_ece(scaled_conf, correctness)

            if ece < best_ece:
                best_ece = ece
                best_temperature = temp

        self.temperature = best_temperature
        logger.info(f"Optimal temperature found: {best_temperature:.4f} (ECE: {best_ece:.4f})")

    def _fit_platt_scaling(
        self,
        confidence_scores: np.ndarray,
        correctness: np.ndarray,
    ) -> None:
        """
        Fit Platt scaling using logistic regression.

        Trains a logistic regression model to map confidence scores to calibrated probabilities.
        """
        # Reshape for sklearn
        X = confidence_scores.reshape(-1, 1)
        y = correctness

        # Train logistic regression
        self.platt_model = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
        )
        self.platt_model.fit(X, y)

        logger.info("Platt scaling model fitted successfully")

    def _apply_temperature_scaling(self, confidence_scores: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to confidence scores."""
        epsilon = 1e-10
        conf_clipped = np.clip(confidence_scores, epsilon, 1 - epsilon)

        # Convert to logits
        logits = np.log(conf_clipped / (1 - conf_clipped))

        # Apply temperature
        scaled_logits = logits / self.temperature

        # Convert back to probabilities
        calibrated = 1 / (1 + np.exp(-scaled_logits))

        return np.clip(calibrated, 0.0, 1.0)

    def _apply_platt_scaling(self, confidence_scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to confidence scores."""
        if self.platt_model is None:
            raise ValueError("Platt model not fitted")

        X = confidence_scores.reshape(-1, 1)
        calibrated = self.platt_model.predict_proba(X)[:, 1]

        return np.clip(calibrated, 0.0, 1.0)

    def _compute_ece(
        self,
        confidence_scores: np.ndarray,
        correctness: np.ndarray,
    ) -> tuple[float, list[dict[str, Any]]]:
        """
        Compute Expected Calibration Error (ECE).

        ECE is the weighted average of the absolute difference between
        confidence and accuracy across bins.
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        bins_data = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            n_in_bin = np.sum(in_bin)

            if n_in_bin > 0:
                # Calculate bin statistics
                bin_confidence = np.mean(confidence_scores[in_bin])
                bin_accuracy = np.mean(correctness[in_bin])
                bin_weight = n_in_bin / len(confidence_scores)

                # Calibration error for this bin
                calibration_error = abs(bin_confidence - bin_accuracy)
                ece += bin_weight * calibration_error

                bins_data.append({
                    "bin_lower": float(bin_lower),
                    "bin_upper": float(bin_upper),
                    "bin_confidence": float(bin_confidence),
                    "bin_accuracy": float(bin_accuracy),
                    "calibration_error": float(calibration_error),
                    "n_samples": int(n_in_bin),
                    "weight": float(bin_weight),
                })

        return ece, bins_data

    def get_calibration_summary(self) -> dict[str, Any]:
        """
        Get summary of calibration state.

        Returns:
            Dictionary with calibration parameters and history.
        """
        return {
            "method": self.method,
            "is_fitted": self.is_fitted,
            "n_bins": self.n_bins,
            "temperature": self.temperature if self.method == "temperature" else None,
            "has_platt_model": self.platt_model is not None,
            "calibration_history": self.calibration_history,
        }

    def save_state(self) -> dict[str, Any]:
        """
        Save calibrator state for serialization.

        Returns:
            Dictionary containing all calibration parameters.
        """
        state = {
            "method": self.method,
            "n_bins": self.n_bins,
            "is_fitted": self.is_fitted,
            "temperature": self.temperature,
            "calibration_history": self.calibration_history,
        }

        if self.platt_model is not None:
            state["platt_coefficients"] = {
                "coef": self.platt_model.coef_.tolist(),
                "intercept": self.platt_model.intercept_.tolist(),
            }

        return state

    def load_state(self, state: dict[str, Any]) -> None:
        """
        Load calibrator state from dictionary.

        Args:
            state: Dictionary containing calibration parameters.
        """
        self.method = state["method"]
        self.n_bins = state["n_bins"]
        self.is_fitted = state["is_fitted"]
        self.temperature = state["temperature"]
        self.calibration_history = state.get("calibration_history", [])

        if "platt_coefficients" in state and self.method == "platt":
            self.platt_model = LogisticRegression()
            self.platt_model.coef_ = np.array(state["platt_coefficients"]["coef"])
            self.platt_model.intercept_ = np.array(state["platt_coefficients"]["intercept"])
            self.platt_model.classes_ = np.array([0, 1])

        logger.info(f"Calibrator state loaded: method={self.method}, fitted={self.is_fitted}")
