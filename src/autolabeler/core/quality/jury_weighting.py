"""Dynamic jury weighting for adaptive model aggregation.

This module implements per-model, per-class reliability weighting that significantly
outperforms static majority voting (LLM Jury-on-Demand, ICLR 2026, +18.5% improvement).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class JuryWeightLearner:
    """Learns per-model, per-label-class reliability weights from calibration data.
    
    Two modes:
    1. Static weights (simple): Per-model accuracy on each label class from a
       calibration set. Stored as a JSON dict. Requires no additional dependencies.
    2. Instance-adaptive weights (advanced): Lightweight logistic regression that
       takes text embedding features + model identity and predicts P(model_correct | text).
       Uses sentence-transformers for embeddings. (Future enhancement)
    
    The simple mode gives most of the benefit and is implemented here.
    
    Attributes:
        weights: Dict mapping (model_name, label) -> weight (probability of correctness)
        default_weight: Fallback weight for unseen (model, label) pairs
        
    Example:
        >>> learner = JuryWeightLearner()
        >>> learner.fit(calibration_df, model_columns, true_label_column)
        >>> learner.save("outputs/jury_weights.json")
        >>> 
        >>> # Later, in the pipeline
        >>> weights_path = "outputs/jury_weights.json"
        >>> learner = JuryWeightLearner.load(weights_path)
        >>> weight = learner.get_weight("claude-sonnet", "0")
    """
    
    def __init__(self):
        """Initialize empty weight learner."""
        self.weights: dict[tuple[str, str], float] = {}
        self.default_weight: float = 0.5
        self.metadata: dict[str, Any] = {}
        
    def fit(
        self,
        df: pd.DataFrame,
        model_columns: list[str],
        true_label_column: str,
        model_names: list[str] | None = None,
    ) -> JuryWeightLearner:
        """Learn reliability weights from calibration data.
        
        For each (model, label) pair, computes the accuracy: what fraction of
        times when this model predicted this label, was it correct?
        
        Parameters:
            df: Calibration dataframe with human labels and model predictions
            model_columns: List of column names containing model predictions
            true_label_column: Column name containing ground truth labels
            model_names: Optional list of human-readable model names (defaults to column names)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> df = pd.read_csv("human_labels.csv")
            >>> learner = JuryWeightLearner()
            >>> learner.fit(
            ...     df,
            ...     model_columns=["gpt4_label", "claude_label", "gemini_label"],
            ...     true_label_column="human_label",
            ...     model_names=["gpt-4o", "claude-sonnet", "gemini-pro"]
            ... )
        """
        if model_names is None:
            model_names = model_columns
            
        if len(model_columns) != len(model_names):
            raise ValueError("model_columns and model_names must have same length")
        
        # Get all unique labels from the data
        all_labels = set(df[true_label_column].unique())
        for col in model_columns:
            all_labels.update(df[col].dropna().unique())
        
        # Compute per-model, per-class accuracy
        weights = {}
        stats = []
        
        for col, name in zip(model_columns, model_names):
            # Skip if column doesn't exist
            if col not in df.columns:
                logger.warning(f"Column {col} not found in dataframe")
                continue
            
            for label in all_labels:
                # Get all cases where this model predicted this label
                mask = df[col] == label
                n_predicted = mask.sum()
                
                if n_predicted == 0:
                    # Model never predicted this label - use default
                    continue
                
                # Of those predictions, how many were correct?
                correct = (df.loc[mask, true_label_column] == label).sum()
                accuracy = correct / n_predicted
                
                weights[(name, str(label))] = accuracy
                stats.append({
                    "model": name,
                    "label": str(label),
                    "n_predicted": int(n_predicted),
                    "n_correct": int(correct),
                    "accuracy": float(accuracy),
                })
        
        self.weights = weights
        
        # Compute overall default weight (mean across all weights)
        if weights:
            self.default_weight = float(np.mean(list(weights.values())))
        
        # Store metadata
        self.metadata = {
            "n_models": len(model_names),
            "n_labels": len(all_labels),
            "n_calibration_samples": len(df),
            "model_names": model_names,
            "labels": sorted(str(l) for l in all_labels),
            "default_weight": self.default_weight,
            "per_class_stats": stats,
        }
        
        logger.info(
            f"Learned jury weights from {len(df)} calibration samples. "
            f"Default weight: {self.default_weight:.3f}"
        )
        logger.info(f"Weight table: {len(weights)} (model, label) pairs")
        
        return self
    
    def get_weight(self, model_name: str, label: str) -> float:
        """Get reliability weight for a (model, label) pair.
        
        Parameters:
            model_name: Model identifier (e.g., "claude-sonnet")
            label: Predicted label (e.g., "0", "-1")
            
        Returns:
            Weight between 0 and 1 (probability this model is correct when it predicts this label)
        """
        return self.weights.get((model_name, str(label)), self.default_weight)
    
    def get_weights_vector(
        self,
        model_names: list[str],
        labels: list[str],
    ) -> list[float]:
        """Get weights for multiple (model, label) pairs.
        
        Useful for batch processing in the pipeline.
        
        Parameters:
            model_names: List of model names (same length as labels)
            labels: List of predicted labels (same length as model_names)
            
        Returns:
            List of weights
        """
        return [self.get_weight(m, l) for m, l in zip(model_names, labels)]
    
    def save(self, path: str | Path) -> None:
        """Save learned weights to JSON file.
        
        Parameters:
            path: Output JSON file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert weights dict to JSON-serializable format
        weights_serializable = {
            f"{model}|||{label}": weight
            for (model, label), weight in self.weights.items()
        }
        
        data = {
            "weights": weights_serializable,
            "default_weight": self.default_weight,
            "metadata": self.metadata,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved jury weights to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> JuryWeightLearner:
        """Load learned weights from JSON file.
        
        Parameters:
            path: Input JSON file path
            
        Returns:
            Loaded JuryWeightLearner instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        learner = cls()
        
        # Deserialize weights
        learner.weights = {
            tuple(key.split("|||")): weight
            for key, weight in data["weights"].items()
        }
        learner.default_weight = data["default_weight"]
        learner.metadata = data.get("metadata", {})
        
        logger.info(f"Loaded jury weights from {path}")
        logger.info(f"  {len(learner.weights)} (model, label) pairs")
        logger.info(f"  Default weight: {learner.default_weight:.3f}")
        
        return learner
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of learned weights.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.weights:
            return {"message": "No weights learned yet"}
        
        weights_array = np.array(list(self.weights.values()))
        
        return {
            "n_weight_pairs": len(self.weights),
            "mean_weight": float(np.mean(weights_array)),
            "std_weight": float(np.std(weights_array)),
            "min_weight": float(np.min(weights_array)),
            "max_weight": float(np.max(weights_array)),
            "default_weight": self.default_weight,
            **self.metadata,
        }
