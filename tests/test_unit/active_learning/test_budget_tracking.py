"""
Unit tests for budget tracking in active learning.

Tests the cost calculation, cumulative tracking, and budget threshold detection
in the active learning sampler and stopping criteria.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from autolabeler.core.active_learning.sampler import ActiveLearningSampler, ALState
from autolabeler.core.active_learning.stopping_criteria import StoppingCriteria
from autolabeler.core.configs import ActiveLearningConfig


@pytest.fixture
def mock_labeling_service():
    """Create a mock labeling service."""
    service = MagicMock()
    service.label_text = MagicMock()
    service.label_batch = MagicMock()
    return service


@pytest.fixture
def default_config():
    """Create default active learning config."""
    return ActiveLearningConfig(
        strategy="uncertainty",
        batch_size=10,
        max_budget=100.0,
        target_accuracy=0.90,
        max_iterations=20,
        patience=3,
        improvement_threshold=0.01,
        uncertainty_threshold=0.1,
    )


@pytest.fixture
def sampler(mock_labeling_service, default_config):
    """Create an active learning sampler with mocked service."""
    return ActiveLearningSampler(mock_labeling_service, default_config)


class TestBudgetCostCalculation:
    """Test cost calculation for batches."""

    def test_calculate_batch_cost_basic(self, sampler):
        """Test basic cost calculation for a batch."""
        # Create a small batch with known text lengths
        batch_df = pd.DataFrame({
            "text": ["Hello world", "Testing batch cost calculation", "Short text"],
        })

        cost = sampler._calculate_batch_cost(batch_df, "text")

        # Cost should be positive
        assert cost > 0
        # Cost should be reasonable (not astronomical)
        assert cost < 1.0  # Should be < $1 for 3 short texts

    def test_calculate_batch_cost_scales_with_text_length(self, sampler):
        """Test that cost scales with text length."""
        # Short text batch
        short_batch = pd.DataFrame({
            "text": ["a" * 50, "b" * 50, "c" * 50],
        })

        # Long text batch
        long_batch = pd.DataFrame({
            "text": ["a" * 500, "b" * 500, "c" * 500],
        })

        short_cost = sampler._calculate_batch_cost(short_batch, "text")
        long_cost = sampler._calculate_batch_cost(long_batch, "text")

        # Longer text should cost more
        assert long_cost > short_cost
        # Cost should scale roughly linearly
        assert long_cost > short_cost * 5  # 10x text = more than 5x cost

    def test_calculate_batch_cost_scales_with_batch_size(self, sampler):
        """Test that cost scales with batch size."""
        # Small batch
        small_batch = pd.DataFrame({
            "text": ["sample text"] * 5,
        })

        # Large batch
        large_batch = pd.DataFrame({
            "text": ["sample text"] * 50,
        })

        small_cost = sampler._calculate_batch_cost(small_batch, "text")
        large_cost = sampler._calculate_batch_cost(large_batch, "text")

        # Larger batch should cost more
        assert large_cost > small_cost
        # Cost should scale roughly linearly with batch size
        assert large_cost > small_cost * 5

    def test_calculate_batch_cost_empty_batch(self, sampler):
        """Test cost calculation for empty batch."""
        empty_batch = pd.DataFrame({"text": []})

        cost = sampler._calculate_batch_cost(empty_batch, "text")

        # Empty batch should have zero or near-zero cost
        assert cost >= 0
        assert cost < 0.01

    def test_calculate_batch_cost_with_null_text(self, sampler):
        """Test cost calculation handles null text gracefully."""
        batch_df = pd.DataFrame({
            "text": ["valid text", None, "more text", ""],
        })

        # Should not raise an error
        cost = sampler._calculate_batch_cost(batch_df, "text")
        assert cost >= 0

    def test_calculate_batch_cost_realistic_values(self, sampler):
        """Test that cost values are realistic for GPT-4o-mini pricing."""
        # Create batch with realistic text lengths
        batch_df = pd.DataFrame({
            "text": [
                "This is a typical sentence for classification." * 3,
                "Another example with moderate length." * 2,
                "Short text",
            ],
        })

        cost = sampler._calculate_batch_cost(batch_df, "text")

        # For 3 texts with ~100-200 tokens each, cost should be in cents
        assert 0.0001 < cost < 0.05  # Between 0.01 cents and 5 cents


class TestBudgetThresholdDetection:
    """Test budget threshold detection in stopping criteria."""

    def test_budget_not_exceeded_at_start(self, default_config):
        """Test that budget check passes when no cost incurred."""
        criteria = StoppingCriteria(default_config)
        state = ALState(current_cost=0.0)

        should_stop, reason = criteria.check(state)

        assert not should_stop
        assert reason == "continue"

    def test_budget_exceeded_exact_match(self, default_config):
        """Test budget check when cost exactly matches budget."""
        criteria = StoppingCriteria(default_config)
        state = ALState(current_cost=100.0)

        should_stop, reason = criteria.check(state)

        # Should stop due to budget (with 10% buffer, 90.0 is threshold)
        assert should_stop
        assert reason == "budget_exhausted"

    def test_budget_threshold_with_buffer(self, default_config):
        """Test that budget check includes 10% buffer."""
        criteria = StoppingCriteria(default_config)

        # At 89% of budget - should not stop
        state_below = ALState(current_cost=89.0)
        should_stop, reason = criteria.check(state_below)
        assert not should_stop

        # At 91% of budget - should stop (90% is threshold)
        state_above = ALState(current_cost=91.0)
        should_stop, reason = criteria.check(state_above)
        assert should_stop
        assert reason == "budget_exhausted"

    def test_budget_exceeded_by_large_amount(self, default_config):
        """Test budget check when cost far exceeds budget."""
        criteria = StoppingCriteria(default_config)
        state = ALState(current_cost=500.0)

        should_stop, reason = criteria.check(state)

        assert should_stop
        assert reason == "budget_exhausted"

    def test_zero_budget_configuration(self):
        """Test that zero budget is handled."""
        config = ActiveLearningConfig(
            strategy="uncertainty",
            batch_size=10,
            max_budget=0.0,  # Zero budget
        )
        criteria = StoppingCriteria(config)
        state = ALState(current_cost=0.01)

        should_stop, reason = criteria.check(state)

        # Should stop immediately with any cost
        assert should_stop
        assert reason == "budget_exhausted"

    def test_very_small_budget(self):
        """Test with very small budget (pennies)."""
        config = ActiveLearningConfig(
            strategy="uncertainty",
            batch_size=10,
            max_budget=0.10,  # 10 cents
        )
        criteria = StoppingCriteria(config)
        state = ALState(current_cost=0.09)

        should_stop, reason = criteria.check(state)

        # At 90% of 0.10 = 0.09, should stop
        assert should_stop
        assert reason == "budget_exhausted"

    def test_budget_priority_over_other_criteria(self, default_config):
        """Test that budget check has priority over other stopping criteria."""
        criteria = StoppingCriteria(default_config)

        # State that would pass other criteria but exceeds budget
        state = ALState(
            current_cost=95.0,  # Exceeds budget threshold
            current_accuracy=0.95,  # Exceeds target
            iteration=1,  # Below max iterations
        )

        should_stop, reason = criteria.check(state)

        # Should stop due to budget, not accuracy
        assert should_stop
        assert reason == "budget_exhausted"


class TestCumulativeCostTracking:
    """Test cumulative cost tracking across iterations."""

    def test_cost_accumulates_across_iterations(self, sampler):
        """Test that costs accumulate properly across iterations."""
        # Start with zero cost
        assert sampler.state.current_cost == 0.0

        # Simulate multiple batch costs
        batch1 = pd.DataFrame({"text": ["text 1", "text 2"]})
        batch2 = pd.DataFrame({"text": ["text 3", "text 4"]})
        batch3 = pd.DataFrame({"text": ["text 5", "text 6"]})

        cost1 = sampler._calculate_batch_cost(batch1, "text")
        cost2 = sampler._calculate_batch_cost(batch2, "text")
        cost3 = sampler._calculate_batch_cost(batch3, "text")

        # Manually update state to simulate loop
        sampler.state.current_cost += cost1
        sampler.state.current_cost += cost2
        sampler.state.current_cost += cost3

        expected_total = cost1 + cost2 + cost3

        assert sampler.state.current_cost == pytest.approx(expected_total)

    def test_state_tracking_after_multiple_updates(self, sampler):
        """Test that state tracks all cost updates correctly."""
        # Perform multiple cost updates
        for i in range(10):
            batch = pd.DataFrame({"text": [f"iteration {i}"]})
            cost = sampler._calculate_batch_cost(batch, "text")
            sampler.state.current_cost += cost

        # Cost should have accumulated
        assert sampler.state.current_cost > 0
        # Should be measurable but not huge
        assert sampler.state.current_cost < 1.0

    def test_state_persists_across_iterations(self, sampler):
        """Test that state doesn't reset between operations."""
        initial_cost = 5.0
        sampler.state.current_cost = initial_cost

        # Add more cost
        batch = pd.DataFrame({"text": ["new batch"]})
        additional_cost = sampler._calculate_batch_cost(batch, "text")
        sampler.state.current_cost += additional_cost

        # Should have both initial and additional cost
        assert sampler.state.current_cost > initial_cost
        assert sampler.state.current_cost == pytest.approx(initial_cost + additional_cost)


class TestGracefulShutdown:
    """Test graceful shutdown behavior when budget is exhausted."""

    @patch.object(ActiveLearningSampler, "_label_batch")
    @patch.object(ActiveLearningSampler, "_get_predictions")
    def test_loop_stops_on_budget_exhaustion(
        self, mock_predictions, mock_label_batch, sampler
    ):
        """Test that loop stops gracefully when budget is exhausted."""
        # Set a very small budget
        sampler.config.max_budget = 0.10

        # Create minimal dataset
        unlabeled_df = pd.DataFrame({
            "text": [f"text {i}" for i in range(20)],
        })

        # Mock predictions and labeling
        mock_result = MagicMock()
        mock_result.label = "TEST_LABEL"
        mock_result.confidence = 0.8
        mock_predictions.return_value = [mock_result] * len(unlabeled_df)
        mock_label_batch.return_value = pd.DataFrame({
            "text": ["labeled"],
            "label": ["TEST"],
        })

        # Run loop - should stop before processing all data
        result_df = sampler.run_active_learning_loop(
            unlabeled_df=unlabeled_df.copy(),
            text_column="text",
        )

        # Should have stopped due to budget
        assert len(result_df) < len(unlabeled_df)
        # Should have incurred some cost
        assert sampler.state.current_cost > 0

    def test_stopping_criteria_status_summary(self, default_config):
        """Test that status summary includes budget information."""
        criteria = StoppingCriteria(default_config)
        state = ALState(
            current_cost=50.0,
            current_accuracy=0.85,
            iteration=5,
        )

        summary = criteria.get_status_summary(state)

        # Should include all key metrics
        assert "current_cost" in summary
        assert "budget_exhausted" in summary
        assert summary["current_cost"] == 50.0
        assert summary["current_iteration"] == 5


class TestEdgeCases:
    """Test edge cases for budget tracking."""

    def test_negative_budget_configuration(self):
        """Test that negative budget is rejected by config validation."""
        with pytest.raises(Exception):  # Pydantic will raise validation error
            ActiveLearningConfig(
                strategy="uncertainty",
                batch_size=10,
                max_budget=-10.0,  # Negative budget should fail validation
            )

    def test_extremely_large_budget(self, mock_labeling_service):
        """Test with extremely large budget."""
        config = ActiveLearningConfig(
            strategy="uncertainty",
            batch_size=10,
            max_budget=1_000_000.0,  # $1M budget
            max_iterations=5,  # But limit iterations
        )
        sampler = ActiveLearningSampler(mock_labeling_service, config)

        # Should not stop due to budget in reasonable number of iterations
        state = ALState(current_cost=1000.0, iteration=1)
        criteria = StoppingCriteria(config)

        should_stop, reason = criteria.check(state)
        # Should not stop on budget with $1000 spent out of $1M
        assert reason != "budget_exhausted"

    def test_budget_exhausted_exact_threshold(self):
        """Test budget check at exact threshold (90% of budget)."""
        config = ActiveLearningConfig(
            strategy="uncertainty",
            batch_size=10,
            max_budget=100.0,
        )
        criteria = StoppingCriteria(config)

        # At exactly 90.0 (the threshold)
        state = ALState(current_cost=90.0)
        should_stop, reason = criteria.check(state)

        assert should_stop
        assert reason == "budget_exhausted"

    def test_floating_point_precision_budget(self):
        """Test that floating point precision doesn't cause issues."""
        config = ActiveLearningConfig(
            strategy="uncertainty",
            batch_size=10,
            max_budget=100.0,
        )
        criteria = StoppingCriteria(config)

        # Test with floating point edge case
        state = ALState(current_cost=89.99999999)
        should_stop, reason = criteria.check(state)

        # Should handle precision correctly
        assert not should_stop

    def test_cost_tracking_with_zero_cost_batches(self, sampler):
        """Test that zero-cost batches don't break tracking."""
        sampler.state.current_cost = 10.0

        # Add zero cost (edge case)
        sampler.state.current_cost += 0.0

        assert sampler.state.current_cost == 10.0

    def test_unicode_text_cost_calculation(self, sampler):
        """Test cost calculation with unicode characters."""
        batch_df = pd.DataFrame({
            "text": [
                "Hello 世界",  # English + Chinese
                "مرحبا",  # Arabic
                "Привет",  # Russian
                "🎉🎊",  # Emojis
            ],
        })

        # Should handle unicode without errors
        cost = sampler._calculate_batch_cost(batch_df, "text")
        assert cost > 0


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_state_to_dict_includes_cost(self, sampler):
        """Test that state serialization includes cost."""
        sampler.state.current_cost = 42.50
        sampler.state.iteration = 10

        state_dict = sampler.state.to_dict()

        assert "current_cost" in state_dict
        assert state_dict["current_cost"] == 42.50
        assert "iteration" in state_dict

    def test_config_serialization_includes_budget(self, default_config):
        """Test that config serialization includes budget."""
        config_dict = default_config.model_dump()

        assert "max_budget" in config_dict
        assert config_dict["max_budget"] == 100.0

    def test_existing_state_fields_still_work(self, sampler):
        """Test that existing state fields aren't broken."""
        # All existing fields should still be accessible
        assert hasattr(sampler.state, "iteration")
        assert hasattr(sampler.state, "current_accuracy")
        assert hasattr(sampler.state, "performance_history")
        assert hasattr(sampler.state, "labeled_indices")
        assert hasattr(sampler.state, "pool_uncertainty")

        # And the new field
        assert hasattr(sampler.state, "current_cost")


class TestCostReporting:
    """Test cost reporting and logging."""

    def test_state_dict_includes_cost_info(self, sampler):
        """Test that state dictionary includes cost information."""
        sampler.state.current_cost = 25.75
        sampler.state.iteration = 5
        sampler.state.current_accuracy = 0.88

        state_dict = sampler.state.to_dict()

        assert "current_cost" in state_dict
        assert state_dict["current_cost"] == 25.75

    def test_stopping_criteria_summary_includes_cost(self, default_config):
        """Test that stopping criteria summary includes cost."""
        criteria = StoppingCriteria(default_config)
        state = ALState(
            current_cost=75.5,
            current_accuracy=0.92,
            iteration=8,
        )

        summary = criteria.get_status_summary(state)

        assert "current_cost" in summary
        assert "budget_exhausted" in summary
        assert summary["current_cost"] == 75.5


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    @patch.object(ActiveLearningSampler, "_label_batch")
    @patch.object(ActiveLearningSampler, "_get_predictions")
    def test_realistic_budget_constrained_workflow(
        self, mock_predictions, mock_label_batch, sampler
    ):
        """Test a realistic budget-constrained active learning workflow."""
        # Configure for budget-constrained scenario
        sampler.config.max_budget = 5.0  # $5 budget
        sampler.config.batch_size = 5
        sampler.config.max_iterations = 20

        # Create test dataset
        unlabeled_df = pd.DataFrame({
            "text": [f"Sample text number {i} for testing" for i in range(100)],
        })

        # Mock the labeling
        mock_result = MagicMock()
        mock_result.label = "TEST"
        mock_result.confidence = 0.75
        mock_predictions.return_value = [mock_result] * 100

        def mock_batch_label(df, text_column):
            return pd.DataFrame({
                "text": df[text_column].tolist(),
                "label": ["TEST"] * len(df),
            })

        mock_label_batch.side_effect = mock_batch_label

        # Run active learning
        result_df = sampler.run_active_learning_loop(
            unlabeled_df=unlabeled_df,
            text_column="text",
        )

        # Verify budget tracking worked
        assert sampler.state.current_cost > 0
        # Should have stopped before processing all data due to budget
        assert len(result_df) < len(unlabeled_df)
        # Cost should be close to budget limit
        assert sampler.state.current_cost <= sampler.config.max_budget
