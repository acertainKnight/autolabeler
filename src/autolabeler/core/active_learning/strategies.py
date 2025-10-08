"""
Sampling strategies for active learning.

Implements various strategies for selecting the most informative samples:
- Uncertainty sampling (least confident, margin, entropy)
- Diversity sampling (clustering-based, core-set selection)
- Committee disagreement (ensemble-based)
- Hybrid (uncertainty + diversity)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

if TYPE_CHECKING:
    from ..configs import ActiveLearningConfig


class SamplingStrategy(ABC):
    """Base class for active learning sampling strategies."""

    def __init__(self, config: ActiveLearningConfig):
        """
        Initialize sampling strategy.

        Args:
            config: Active learning configuration
        """
        self.config = config

    @abstractmethod
    def select(
        self,
        unlabeled_df: pd.DataFrame,
        labeled_df: pd.DataFrame,
        predictions: list,
        batch_size: int,
    ) -> pd.Index:
        """
        Select samples from unlabeled pool.

        Args:
            unlabeled_df: Unlabeled data pool
            labeled_df: Currently labeled data
            predictions: Predictions for unlabeled pool
            batch_size: Number of samples to select

        Returns:
            Indices of selected samples
        """
        pass


class UncertaintySampler(SamplingStrategy):
    """
    Uncertainty-based sampling strategy.

    Selects samples where the model is least confident, based on various
    uncertainty metrics.
    """

    def select(
        self,
        unlabeled_df: pd.DataFrame,
        labeled_df: pd.DataFrame,
        predictions: list,
        batch_size: int,
    ) -> pd.Index:
        """Select samples with highest uncertainty."""
        logger.debug(
            f"Uncertainty sampling: Selecting {batch_size} from {len(unlabeled_df)} samples"
        )

        # Calculate uncertainty scores
        uncertainty_scores = np.array(
            [self._calculate_uncertainty(pred) for pred in predictions]
        )

        # Handle edge case: not enough samples
        actual_batch_size = min(batch_size, len(unlabeled_df))

        # Select top-k most uncertain
        top_indices = np.argsort(uncertainty_scores)[-actual_batch_size:]

        return unlabeled_df.index[top_indices]

    def _calculate_uncertainty(self, prediction) -> float:
        """
        Calculate uncertainty score for a prediction.

        Args:
            prediction: Prediction object with confidence score

        Returns:
            Uncertainty score (higher = more uncertain)
        """
        method = self.config.uncertainty_method

        # Extract confidence score
        if hasattr(prediction, "confidence"):
            confidence = prediction.confidence
        else:
            confidence = 0.5  # Default to maximum uncertainty

        if method == "least_confident":
            # 1 - max(P(y|x))
            return 1 - confidence

        elif method == "margin":
            # For margin, we need top-2 probabilities
            # Since we only have top confidence, approximate as:
            # margin = confidence - (1 - confidence) / (n_classes - 1)
            # For binary: margin ≈ 2 * confidence - 1
            # Uncertainty = 1 - |margin|
            margin = abs(2 * confidence - 1)
            return 1 - margin

        elif method == "entropy":
            # For binary classification with confidence p:
            # H = -p*log(p) - (1-p)*log(1-p)
            # Normalize to [0, 1]
            if confidence == 0 or confidence == 1:
                return 0  # No uncertainty

            entropy = -confidence * np.log2(confidence) - (1 - confidence) * np.log2(
                1 - confidence
            )
            max_entropy = 1.0  # Binary classification
            return entropy / max_entropy

        else:
            # Default to least confident
            return 1 - confidence


class DiversitySampler(SamplingStrategy):
    """
    Diversity-based sampling strategy.

    Selects diverse samples using clustering to ensure broad coverage
    of the feature space.
    """

    def __init__(self, config: ActiveLearningConfig):
        """Initialize diversity sampler with embedding model."""
        super().__init__(config)

        # Initialize embedding model
        try:
            self.embedder = SentenceTransformer(config.embedding_model)
            logger.info(f"Loaded embedding model: {config.embedding_model}")
        except Exception as e:
            logger.warning(
                f"Failed to load embedding model {config.embedding_model}: {e}. "
                f"Using default model."
            )
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def select(
        self,
        unlabeled_df: pd.DataFrame,
        labeled_df: pd.DataFrame,
        predictions: list,
        batch_size: int,
    ) -> pd.Index:
        """Select diverse samples using clustering."""
        logger.debug(
            f"Diversity sampling: Selecting {batch_size} from {len(unlabeled_df)} samples"
        )

        # Get text column
        text_column = self.config.text_column
        if text_column not in unlabeled_df.columns:
            logger.warning(
                f"Text column '{text_column}' not found. Using first column."
            )
            text_column = unlabeled_df.columns[0]

        # Extract texts
        texts = unlabeled_df[text_column].tolist()

        # Generate embeddings
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.embedder.encode(texts, show_progress_bar=False)

        # Handle edge case: fewer samples than batch size
        actual_batch_size = min(batch_size, len(unlabeled_df))

        # Cluster and select representatives
        kmeans = KMeans(n_clusters=actual_batch_size, random_state=42, n_init=10)
        kmeans.fit(embeddings)

        # Select sample nearest to each cluster center
        selected_indices = []
        for i in range(actual_batch_size):
            cluster_mask = kmeans.labels_ == i
            cluster_embeddings = embeddings[cluster_mask]

            if len(cluster_embeddings) == 0:
                continue

            center = kmeans.cluster_centers_[i]

            # Find nearest to center
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            nearest_idx = np.argmin(distances)

            # Map back to original index
            cluster_indices = np.where(cluster_mask)[0]
            selected_indices.append(cluster_indices[nearest_idx])

        return unlabeled_df.index[selected_indices]


class CommitteeSampler(SamplingStrategy):
    """
    Committee disagreement sampling strategy.

    Selects samples where ensemble members disagree most, indicating
    high uncertainty and information gain.
    """

    def select(
        self,
        unlabeled_df: pd.DataFrame,
        labeled_df: pd.DataFrame,
        predictions: list,
        batch_size: int,
    ) -> pd.Index:
        """Select samples with highest committee disagreement."""
        logger.debug(
            f"Committee sampling: Selecting {batch_size} from {len(unlabeled_df)} samples"
        )

        # Calculate disagreement scores
        disagreement_scores = np.array(
            [self._calculate_disagreement(pred) for pred in predictions]
        )

        # Handle edge case: not enough samples
        actual_batch_size = min(batch_size, len(unlabeled_df))

        # Select top-k with highest disagreement
        top_indices = np.argsort(disagreement_scores)[-actual_batch_size:]

        return unlabeled_df.index[top_indices]

    def _calculate_disagreement(self, prediction) -> float:
        """
        Calculate committee disagreement score.

        Args:
            prediction: Prediction object (possibly ensemble result)

        Returns:
            Disagreement score (higher = more disagreement)
        """
        # Check if prediction has individual predictions (ensemble)
        if not hasattr(prediction, "individual_predictions"):
            # No ensemble, fall back to confidence-based uncertainty
            confidence = getattr(prediction, "confidence", 0.5)
            return 1 - confidence

        individual_preds = prediction.individual_predictions
        if not individual_preds:
            return 0.0

        # Extract labels from ensemble members
        labels = [p.get("label") for p in individual_preds if "label" in p]

        if not labels:
            return 0.0

        # Calculate vote entropy
        vote_counts = Counter(labels)
        total = len(labels)

        entropy = -sum(
            (count / total) * np.log(count / total) for count in vote_counts.values()
        )

        # Normalize by maximum possible entropy
        num_unique = len(vote_counts)
        if num_unique <= 1:
            return 0.0  # All agree

        max_entropy = np.log(num_unique)
        return entropy / max_entropy


class HybridSampler(SamplingStrategy):
    """
    Hybrid sampling combining uncertainty and diversity.

    Uses a two-step approach:
    1. Select top-k*3 most uncertain samples
    2. Among those, select k most diverse samples

    This ensures selected samples are both informative (uncertain)
    and representative (diverse).
    """

    def __init__(self, config: ActiveLearningConfig):
        """Initialize hybrid sampler with sub-strategies."""
        super().__init__(config)
        self.uncertainty_sampler = UncertaintySampler(config)
        self.diversity_sampler = DiversitySampler(config)

    def select(
        self,
        unlabeled_df: pd.DataFrame,
        labeled_df: pd.DataFrame,
        predictions: list,
        batch_size: int,
    ) -> pd.Index:
        """Select samples combining uncertainty and diversity."""
        logger.debug(
            f"Hybrid sampling (alpha={self.config.hybrid_alpha}): "
            f"Selecting {batch_size} from {len(unlabeled_df)} samples"
        )

        # Handle edge case: not enough samples
        if len(unlabeled_df) <= batch_size:
            return unlabeled_df.index

        # Step 1: Select top-k*3 most uncertain samples
        oversample_factor = 3
        num_uncertain = min(batch_size * oversample_factor, len(unlabeled_df))

        uncertainty_scores = np.array(
            [self.uncertainty_sampler._calculate_uncertainty(pred) for pred in predictions]
        )
        top_uncertain_idx = np.argsort(uncertainty_scores)[-num_uncertain:]

        # Step 2: Among uncertain samples, select diverse ones
        uncertain_df = unlabeled_df.iloc[top_uncertain_idx]
        uncertain_predictions = [predictions[i] for i in top_uncertain_idx]

        # Use diversity sampler on filtered dataset
        diverse_indices = self.diversity_sampler.select(
            unlabeled_df=uncertain_df,
            labeled_df=labeled_df,
            predictions=uncertain_predictions,
            batch_size=batch_size,
        )

        return diverse_indices
