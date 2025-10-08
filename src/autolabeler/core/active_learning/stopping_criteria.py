"""
Stopping criteria for active learning loops.

Implements various criteria to determine when to stop active learning:
- Performance plateau detection
- Budget exhaustion
- Target performance reached
- Low uncertainty in remaining pool
- Maximum iterations reached
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from ..configs import ActiveLearningConfig
    from .sampler import ALState


class StoppingCriteria:
    """
    Stopping criteria manager for active learning.

    Evaluates multiple stopping conditions and provides reasoning for
    termination decisions.
    """

    def __init__(self, config: ActiveLearningConfig):
        """
        Initialize stopping criteria.

        Args:
            config: Active learning configuration
        """
        self.config = config

    def check(self, state: ALState) -> tuple[bool, str]:
        """
        Check all stopping criteria.

        Args:
            state: Current active learning state

        Returns:
            Tuple of (should_stop, reason)
        """
        # Check each criterion in order of priority

        # 1. Check budget (critical)
        if self._check_budget(state):
            reason = (
                f"Budget exhausted: ${state.current_cost:.2f} / ${self.config.max_budget:.2f}"
            )
            logger.info(f"Stopping criterion met: {reason}")
            return True, "budget_exhausted"

        # 2. Check target performance (success condition)
        if self._check_target_reached(state):
            reason = (
                f"Target accuracy reached: {state.current_accuracy:.3f} / "
                f"{self.config.target_accuracy:.3f}"
            )
            logger.info(f"Stopping criterion met: {reason}")
            return True, "target_reached"

        # 3. Check performance plateau
        if self._check_plateau(state):
            reason = (
                f"Performance plateau detected: No improvement over "
                f"{self.config.patience} iterations"
            )
            logger.info(f"Stopping criterion met: {reason}")
            return True, "performance_plateau"

        # 4. Check low uncertainty
        if self._check_low_uncertainty(state):
            reason = (
                f"Low pool uncertainty: {np.mean(state.pool_uncertainty):.3f} < "
                f"{self.config.uncertainty_threshold:.3f}"
            )
            logger.info(f"Stopping criterion met: {reason}")
            return True, "low_uncertainty"

        # 5. Check max iterations (safety limit)
        if self._check_max_iterations(state):
            reason = f"Maximum iterations reached: {state.iteration} / {self.config.max_iterations}"
            logger.info(f"Stopping criterion met: {reason}")
            return True, "max_iterations"

        # No stopping criterion met
        return False, "continue"

    def _check_budget(self, state: ALState) -> bool:
        """
        Check if budget is exhausted.

        Args:
            state: Current state

        Returns:
            True if should stop due to budget
        """
        # Reserve 10% buffer for final evaluation
        buffer = 0.1
        threshold = self.config.max_budget * (1 - buffer)

        return state.current_cost >= threshold

    def _check_target_reached(self, state: ALState) -> bool:
        """
        Check if target performance is reached.

        Args:
            state: Current state

        Returns:
            True if target reached
        """
        return state.current_accuracy >= self.config.target_accuracy

    def _check_plateau(self, state: ALState) -> bool:
        """
        Check if performance has plateaued.

        A plateau is detected when recent improvements are all below
        the threshold for `patience` iterations.

        Args:
            state: Current state

        Returns:
            True if plateau detected
        """
        # Need at least patience+1 measurements
        if len(state.performance_history) < self.config.patience + 1:
            return False

        # Calculate recent improvements
        recent_improvements = [
            state.performance_history[i] - state.performance_history[i - 1]
            for i in range(-self.config.patience, 0)
        ]

        # Check if all improvements below threshold
        all_small = all(imp < self.config.improvement_threshold for imp in recent_improvements)

        if all_small:
            logger.debug(
                f"Plateau check: Recent improvements {recent_improvements} "
                f"all below {self.config.improvement_threshold}"
            )

        return all_small

    def _check_low_uncertainty(self, state: ALState) -> bool:
        """
        Check if remaining pool has low uncertainty.

        Low uncertainty indicates the model is confident on all
        remaining examples, suggesting diminishing returns.

        Args:
            state: Current state

        Returns:
            True if pool uncertainty is low
        """
        if not state.pool_uncertainty:
            return False

        mean_uncertainty = np.mean(state.pool_uncertainty)
        is_low = mean_uncertainty < self.config.uncertainty_threshold

        if is_low:
            logger.debug(
                f"Low uncertainty check: Mean uncertainty {mean_uncertainty:.3f} "
                f"below threshold {self.config.uncertainty_threshold:.3f}"
            )

        return is_low

    def _check_max_iterations(self, state: ALState) -> bool:
        """
        Check if maximum iterations reached.

        Args:
            state: Current state

        Returns:
            True if max iterations reached
        """
        return state.iteration >= self.config.max_iterations

    def get_status_summary(self, state: ALState) -> dict:
        """
        Get summary of all stopping criteria status.

        Args:
            state: Current state

        Returns:
            Dictionary with status of each criterion
        """
        return {
            "budget_exhausted": self._check_budget(state),
            "target_reached": self._check_target_reached(state),
            "plateau_detected": self._check_plateau(state),
            "low_uncertainty": self._check_low_uncertainty(state),
            "max_iterations": self._check_max_iterations(state),
            "current_iteration": state.iteration,
            "current_cost": state.current_cost,
            "current_accuracy": state.current_accuracy,
            "mean_pool_uncertainty": (
                np.mean(state.pool_uncertainty) if state.pool_uncertainty else None
            ),
        }
