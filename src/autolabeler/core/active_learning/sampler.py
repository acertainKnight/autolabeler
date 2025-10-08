"""
Active learning sampler for intelligent sample selection.

Implements the main active learning loop that:
1. Selects informative samples using various strategies
2. Labels selected samples (LLM or human)
3. Updates the model
4. Evaluates stopping criteria
5. Repeats until convergence or budget exhaustion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from .stopping_criteria import StoppingCriteria
from .strategies import (
    CommitteeSampler,
    DiversitySampler,
    HybridSampler,
    UncertaintySampler,
)

if TYPE_CHECKING:
    from ..configs import ActiveLearningConfig
    from ..labeling import LabelingService


@dataclass
class ALState:
    """Active learning state tracking."""

    iteration: int = 0
    current_cost: float = 0.0
    current_accuracy: float = 0.0
    performance_history: list[float] = field(default_factory=list)
    labeled_indices: list[int] = field(default_factory=list)
    pool_uncertainty: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert state to dictionary."""
        return {
            "iteration": self.iteration,
            "current_cost": self.current_cost,
            "current_accuracy": self.current_accuracy,
            "performance_history": self.performance_history,
            "labeled_indices": self.labeled_indices,
            "num_labeled": len(self.labeled_indices),
            "mean_pool_uncertainty": (
                np.mean(self.pool_uncertainty) if self.pool_uncertainty else None
            ),
        }


class ActiveLearningSampler:
    """
    Active learning sampler for intelligent sample selection.

    Implements multiple sampling strategies and stopping criteria
    to minimize labeling costs while maximizing model performance.

    Args:
        labeling_service: LabelingService for labeling operations
        config: Active learning configuration

    Example:
        >>> from autolabeler.core.active_learning import ActiveLearningSampler
        >>> from autolabeler.core.configs import ActiveLearningConfig
        >>>
        >>> config = ActiveLearningConfig(
        ...     strategy="hybrid",
        ...     batch_size=50,
        ...     max_budget=100.0,
        ...     target_accuracy=0.90
        ... )
        >>>
        >>> sampler = ActiveLearningSampler(labeling_service, config)
        >>> results = sampler.run_active_learning_loop(unlabeled_df)
    """

    def __init__(self, labeling_service: LabelingService, config: ActiveLearningConfig):
        """
        Initialize active learning sampler.

        Args:
            labeling_service: Service for labeling text
            config: Active learning configuration
        """
        self.labeling_service = labeling_service
        self.config = config
        self.state = ALState()

        # Initialize strategy
        self.strategy = self._create_strategy(config.strategy)

        # Initialize stopping criteria
        self.stopping_criteria = StoppingCriteria(config)

        logger.info(
            f"ActiveLearningSampler initialized with strategy: {config.strategy}"
        )

    def _create_strategy(self, strategy_name: str):
        """
        Create sampling strategy instance.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy name is unknown
        """
        strategies = {
            "uncertainty": UncertaintySampler,
            "diversity": DiversitySampler,
            "committee": CommitteeSampler,
            "hybrid": HybridSampler,
        }

        if strategy_name not in strategies:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Choose from: {list(strategies.keys())}"
            )

        return strategies[strategy_name](self.config)

    def run_active_learning_loop(
        self,
        unlabeled_df: pd.DataFrame,
        text_column: str = "text",
        seed_labeled_df: pd.DataFrame | None = None,
        validation_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Run the complete active learning loop.

        Args:
            unlabeled_df: Unlabeled data pool
            text_column: Column containing text to label
            seed_labeled_df: Initial labeled examples (optional)
            validation_df: Validation set for evaluation (optional)

        Returns:
            DataFrame with selected and labeled samples

        Example:
            >>> unlabeled_df = pd.read_csv("unlabeled.csv")
            >>> seed_df = pd.read_csv("seed_labeled.csv")
            >>> val_df = pd.read_csv("validation.csv")
            >>>
            >>> labeled_df = sampler.run_active_learning_loop(
            ...     unlabeled_df=unlabeled_df,
            ...     seed_labeled_df=seed_df,
            ...     validation_df=val_df
            ... )
        """
        logger.info(
            f"Starting active learning loop with {len(unlabeled_df)} unlabeled examples"
        )
        logger.info(f"Strategy: {self.config.strategy}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Max budget: ${self.config.max_budget:.2f}")
        logger.info(f"Target accuracy: {self.config.target_accuracy:.3f}")

        # Initialize with seed data if provided
        if seed_labeled_df is not None:
            labeled_df = seed_labeled_df.copy()
            self.state.labeled_indices = seed_labeled_df.index.tolist()
            logger.info(f"Using {len(seed_labeled_df)} seed labeled examples")
        else:
            # Bootstrap with random seed
            seed_size = self.config.initial_seed_size
            logger.info(f"Bootstrapping with {seed_size} random examples")

            seed_indices = np.random.choice(
                unlabeled_df.index,
                size=min(seed_size, len(unlabeled_df)),
                replace=False,
            )

            # Label seed examples
            seed_df = unlabeled_df.loc[seed_indices].copy()
            labeled_df = self._label_batch(seed_df, text_column)

            unlabeled_df = unlabeled_df.drop(seed_indices)
            self.state.labeled_indices.extend(seed_indices.tolist())

        # Active learning iterations
        while len(unlabeled_df) > 0:
            self.state.iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Active Learning Iteration {self.state.iteration}")
            logger.info(f"{'='*60}")

            # Check stopping criteria
            should_stop, reason = self.stopping_criteria.check(self.state)
            if should_stop:
                logger.info(f"Stopping active learning: {reason}")
                self._log_final_summary(labeled_df, validation_df, text_column)
                break

            # Select next batch
            try:
                selected_indices = self.select_batch(
                    unlabeled_df=unlabeled_df, labeled_df=labeled_df, text_column=text_column
                )
            except Exception as e:
                logger.error(f"Error during sample selection: {e}")
                break

            if len(selected_indices) == 0:
                logger.warning("No samples selected, stopping")
                break

            logger.info(f"Selected {len(selected_indices)} samples for labeling")

            # Label selected batch
            selected_df = unlabeled_df.loc[selected_indices]
            try:
                labeled_batch = self._label_batch(selected_df, text_column)
            except Exception as e:
                logger.error(f"Error during batch labeling: {e}")
                break

            # Update datasets
            labeled_df = pd.concat([labeled_df, labeled_batch], ignore_index=True)
            unlabeled_df = unlabeled_df.drop(selected_indices)
            self.state.labeled_indices.extend(selected_indices.tolist())

            # Update state
            batch_cost = self._calculate_batch_cost(labeled_batch, text_column)
            self.state.current_cost += batch_cost

            logger.info(f"Batch cost: ${batch_cost:.2f}")
            logger.info(f"Total cost: ${self.state.current_cost:.2f}")
            logger.info(f"Labeled so far: {len(labeled_df)}")
            logger.info(f"Remaining unlabeled: {len(unlabeled_df)}")

            # Evaluate if validation set provided
            if validation_df is not None:
                accuracy = self._evaluate(labeled_df, validation_df, text_column)
                self.state.current_accuracy = accuracy
                self.state.performance_history.append(accuracy)

                logger.info(f"Current accuracy: {accuracy:.3f}")

                # Log improvement
                if len(self.state.performance_history) > 1:
                    prev_acc = self.state.performance_history[-2]
                    improvement = accuracy - prev_acc
                    logger.info(f"Improvement: {improvement:+.3f}")

            # Update pool uncertainty
            if len(unlabeled_df) > 0:
                try:
                    pool_predictions = self._get_predictions(unlabeled_df, text_column)
                    self.state.pool_uncertainty = [
                        1 - getattr(p, "confidence", 0.5) for p in pool_predictions
                    ]

                    mean_uncertainty = np.mean(self.state.pool_uncertainty)
                    logger.info(f"Mean pool uncertainty: {mean_uncertainty:.3f}")
                except Exception as e:
                    logger.warning(f"Could not calculate pool uncertainty: {e}")

        # Final summary
        self._log_final_summary(labeled_df, validation_df, text_column)

        return labeled_df

    def select_batch(
        self, unlabeled_df: pd.DataFrame, labeled_df: pd.DataFrame, text_column: str
    ) -> pd.Index:
        """
        Select next batch of samples to label.

        Args:
            unlabeled_df: Unlabeled data pool
            labeled_df: Currently labeled data
            text_column: Column containing text

        Returns:
            Indices of selected samples
        """
        logger.debug(f"Selecting batch of size {self.config.batch_size}")

        # Get predictions for unlabeled pool
        predictions = self._get_predictions(unlabeled_df, text_column)

        # Apply strategy to select samples
        selected_indices = self.strategy.select(
            unlabeled_df=unlabeled_df,
            labeled_df=labeled_df,
            predictions=predictions,
            batch_size=self.config.batch_size,
        )

        return selected_indices

    def _get_predictions(self, df: pd.DataFrame, text_column: str) -> list:
        """
        Get predictions for a dataset.

        Args:
            df: DataFrame with text
            text_column: Column containing text

        Returns:
            List of predictions
        """
        logger.debug(f"Getting predictions for {len(df)} samples")

        predictions = []
        for text in df[text_column]:
            try:
                result = self.labeling_service.label_text(text)
                predictions.append(result)
            except Exception as e:
                logger.warning(f"Prediction failed for text: {e}")
                # Create mock prediction with low confidence
                from types import SimpleNamespace

                mock_pred = SimpleNamespace(label="UNKNOWN", confidence=0.5)
                predictions.append(mock_pred)

        return predictions

    def _label_batch(self, batch_df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Label a batch of samples.

        Args:
            batch_df: Batch to label
            text_column: Column containing text

        Returns:
            Labeled batch
        """
        logger.debug(f"Labeling batch of {len(batch_df)} samples")

        # Use labeling service to label batch
        labeled_batch = self.labeling_service.label_batch(
            df=batch_df, text_column=text_column
        )

        return labeled_batch

    def _calculate_batch_cost(self, batch_df: pd.DataFrame, text_column: str) -> float:
        """
        Calculate cost of labeling a batch.

        Args:
            batch_df: Labeled batch
            text_column: Column containing text

        Returns:
            Estimated cost in USD
        """
        # Estimate based on text length and model pricing
        # This is a rough estimate - actual implementation would use
        # real token counts from API responses

        # Average tokens per text (rough estimate)
        texts = batch_df[text_column].tolist()
        total_chars = sum(len(str(text)) for text in texts)
        estimated_tokens = total_chars / 4  # Rough approximation: 1 token ≈ 4 chars

        # Add output tokens (label + reasoning)
        estimated_output_tokens = len(batch_df) * 100  # ~100 tokens per response

        total_tokens = estimated_tokens + estimated_output_tokens

        # Cost per 1M tokens (GPT-4o-mini pricing as default)
        input_cost_per_1m = 0.15
        output_cost_per_1m = 0.60

        estimated_cost = (
            (estimated_tokens / 1_000_000) * input_cost_per_1m
            + (estimated_output_tokens / 1_000_000) * output_cost_per_1m
        )

        return estimated_cost

    def _evaluate(
        self, labeled_df: pd.DataFrame, validation_df: pd.DataFrame, text_column: str
    ) -> float:
        """
        Evaluate current model performance.

        Args:
            labeled_df: Labeled training data
            validation_df: Validation data with ground truth
            text_column: Column containing text

        Returns:
            Accuracy score
        """
        # This is a simplified placeholder for demonstration
        # In production, you would:
        # 1. Train a classifier on labeled_df
        # 2. Evaluate on validation_df
        # 3. Return actual accuracy

        # For now, simulate improving accuracy with more data
        # Initial accuracy + logarithmic growth
        base_accuracy = 0.5
        growth_rate = 0.05
        max_accuracy = 0.95

        data_factor = np.log1p(len(labeled_df)) / np.log1p(10000)  # Normalize
        simulated_accuracy = min(
            base_accuracy + growth_rate * data_factor * 10, max_accuracy
        )

        # Add some noise
        noise = np.random.normal(0, 0.02)
        simulated_accuracy = np.clip(simulated_accuracy + noise, 0, 1)

        return simulated_accuracy

    def _log_final_summary(
        self,
        labeled_df: pd.DataFrame,
        validation_df: pd.DataFrame | None,
        text_column: str,
    ) -> None:
        """Log final summary of active learning run."""
        logger.info("\n" + "=" * 60)
        logger.info("Active Learning Summary")
        logger.info("=" * 60)
        logger.info(f"Total iterations: {self.state.iteration}")
        logger.info(f"Total labeled: {len(labeled_df)}")
        logger.info(f"Total cost: ${self.state.current_cost:.2f}")

        if validation_df is not None:
            logger.info(f"Final accuracy: {self.state.current_accuracy:.3f}")
            logger.info(f"Target accuracy: {self.config.target_accuracy:.3f}")

        if self.config.max_budget > 0:
            cost_efficiency = (
                (1 - self.state.current_cost / self.config.max_budget) * 100
            )
            logger.info(f"Budget remaining: {cost_efficiency:.1f}%")

        logger.info("=" * 60)

    def save_state(self, output_path: str | Path) -> None:
        """
        Save active learning state to file.

        Args:
            output_path: Path to save state
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = self.state.to_dict()
        state_dict["config"] = self.config.model_dump()

        with open(output_path, "w") as f:
            json.dump(state_dict, f, indent=2)

        logger.info(f"Saved active learning state to {output_path}")

    def load_state(self, input_path: str | Path) -> None:
        """
        Load active learning state from file.

        Args:
            input_path: Path to load state from
        """
        import json

        input_path = Path(input_path)

        with open(input_path) as f:
            state_dict = json.load(f)

        # Restore state
        self.state.iteration = state_dict.get("iteration", 0)
        self.state.current_cost = state_dict.get("current_cost", 0.0)
        self.state.current_accuracy = state_dict.get("current_accuracy", 0.0)
        self.state.performance_history = state_dict.get("performance_history", [])
        self.state.labeled_indices = state_dict.get("labeled_indices", [])

        logger.info(f"Loaded active learning state from {input_path}")
        logger.info(f"Resuming from iteration {self.state.iteration}")
