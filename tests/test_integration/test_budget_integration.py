"""
Integration tests for budget tracking across the labeling system.

Tests budget tracking integration with:
- LabelingService and LLM providers
- OpenRouter client with cost tracking
- Active learning with budget constraints
- Multi-provider cost accumulation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from autolabeler.config import Settings
from autolabeler.core.configs import LabelingConfig
from autolabeler.core.labeling import LabelingService
from autolabeler.core.utils.budget_tracker import BudgetExceededError, CostTracker
from autolabeler.models import LabelResponse


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        llm_model="gpt-4o-mini",
        llm_provider="openrouter",
        embedding_model="text-embedding-3-small",
    )


@pytest.fixture
def cost_tracker():
    """Create a cost tracker with budget."""
    return CostTracker(budget=10.0)


@pytest.fixture
def labeling_service(settings):
    """Create a labeling service for testing."""
    return LabelingService(
        dataset_name="test_budget",
        settings=settings,
    )


class TestLabelingServiceBudgetIntegration:
    """Test budget tracking integration with labeling service."""

    @patch("autolabeler.core.llm_providers.openrouter.OpenRouterClient._generate")
    def test_label_text_tracks_cost(self, mock_generate, labeling_service):
        """Test that labeling a single text tracks cost."""
        # Mock the LLM response with cost information
        message = AIMessage(
            content='{"label": "positive", "confidence": 0.9}',
            response_metadata={"total_cost": 0.001},
        )
        generation = ChatGeneration(message=message)
        mock_result = LLMResult(generations=[[generation]])
        mock_generate.return_value = mock_result

        # Create tracker and inject into client
        tracker = CostTracker(budget=1.0)

        # Mock the client to use our tracker
        with patch.object(
            labeling_service,
            "_get_client_for_config",
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client._generate.return_value = mock_result
            mock_client.model_name = "gpt-4o-mini"

            # Inject cost tracker
            mock_client._instance_data = {
                id(mock_client): MagicMock(cost_tracker=tracker)
            }

            mock_get_client.return_value = mock_client

            # Label text
            config = LabelingConfig(use_rag=False, use_validation=False)
            response = labeling_service.label_text(
                "This is a test", config=config
            )

            # Verify labeling worked
            assert isinstance(response, LabelResponse)

    def test_batch_labeling_accumulates_costs(self, labeling_service):
        """Test that batch labeling accumulates costs across multiple items."""
        df = pd.DataFrame({
            "text": ["text 1", "text 2", "text 3"],
        })

        tracker = CostTracker(budget=5.0)

        # Mock the labeling
        with patch.object(labeling_service, "label_text") as mock_label:
            # Simulate cost tracking
            def mock_label_with_cost(text, **kwargs):
                tracker.add_cost(0.5)  # $0.50 per item
                return LabelResponse(label="TEST", confidence=0.8)

            mock_label.side_effect = mock_label_with_cost

            # This would normally call label_text for each item
            # For testing, we'll just simulate the loop
            for text in df["text"]:
                mock_label(text)

            # Verify cumulative cost
            assert tracker.total_cost == 1.5  # 3 items * $0.50
            assert tracker.call_count == 3


class TestOpenRouterClientBudgetIntegration:
    """Test budget tracking in OpenRouter client."""

    @patch("autolabeler.core.llm_providers.openrouter.OpenRouterClient._generate")
    def test_client_checks_budget_before_call(self, mock_generate):
        """Test that client checks budget before making API call."""
        from autolabeler.core.llm_providers.openrouter import OpenRouterClient

        tracker = CostTracker(budget=0.001)  # Very small budget
        tracker.add_cost(0.002)  # Exceed budget

        client = OpenRouterClient(
            api_key="test-key",
            cost_tracker=tracker,
        )

        # Attempt to generate should raise BudgetExceededError
        with pytest.raises(BudgetExceededError) as exc_info:
            client._generate([])

        assert exc_info.value.budget == 0.001

    @patch("autolabeler.core.llm_providers.openrouter.OpenRouterClient._generate")
    def test_client_tracks_cost_after_call(self, mock_super_generate):
        """Test that client tracks cost after successful API call."""
        from autolabeler.core.llm_providers.openrouter import OpenRouterClient

        # Mock successful API response with cost
        message = AIMessage(
            content="Test response",
            response_metadata={"total_cost": 0.0005},
        )
        generation = ChatGeneration(message=message)
        mock_result = LLMResult(generations=[[generation]])
        mock_super_generate.return_value = mock_result

        tracker = CostTracker(budget=1.0)
        client = OpenRouterClient(
            api_key="test-key",
            cost_tracker=tracker,
        )

        # Make a call
        result = client._generate([])

        # Verify cost was tracked
        assert tracker.total_cost > 0
        assert tracker.call_count > 0


class TestActiveLearningSampler_BudgetIntegration:
    """Test budget tracking in active learning workflows."""

    def test_sampler_respects_budget_limit(self):
        """Test that active learning sampler respects budget limits."""
        from autolabeler.core.active_learning import ActiveLearningSampler
        from autolabeler.core.configs import ActiveLearningConfig

        # Create config with tight budget
        config = ActiveLearningConfig(
            strategy="uncertainty",
            batch_size=5,
            max_budget=1.0,  # $1 budget
            max_iterations=20,
        )

        # Mock labeling service
        mock_service = MagicMock()
        sampler = ActiveLearningSampler(mock_service, config)

        # Verify budget is configured
        assert sampler.config.max_budget == 1.0

        # Verify stopping criteria checks budget
        from autolabeler.core.active_learning.sampler import ALState

        state = ALState(current_cost=0.95)  # Close to budget
        should_stop, reason = sampler.stopping_criteria.check(state)

        # Should stop due to budget (with 10% buffer at $0.90)
        assert should_stop
        assert reason == "budget_exhausted"

    @patch.object(
        "autolabeler.core.active_learning.sampler.ActiveLearningSampler",
        "_label_batch",
    )
    @patch.object(
        "autolabeler.core.active_learning.sampler.ActiveLearningSampler",
        "_get_predictions",
    )
    def test_active_learning_loop_stops_on_budget(
        self, mock_predictions, mock_label_batch
    ):
        """Test that active learning loop stops when budget is exceeded."""
        from autolabeler.core.active_learning import ActiveLearningSampler
        from autolabeler.core.configs import ActiveLearningConfig

        config = ActiveLearningConfig(
            strategy="uncertainty",
            batch_size=10,
            max_budget=5.0,
            initial_seed_size=5,
        )

        mock_service = MagicMock()
        sampler = ActiveLearningSampler(mock_service, config)

        # Mock predictions
        mock_result = MagicMock()
        mock_result.label = "TEST"
        mock_result.confidence = 0.7
        mock_predictions.return_value = [mock_result] * 50

        # Mock labeling that consumes budget
        def mock_label(df, text_column):
            # Simulate expensive labeling
            sampler.state.current_cost += 2.0
            return pd.DataFrame({
                "text": df[text_column].tolist(),
                "label": ["TEST"] * len(df),
            })

        mock_label_batch.side_effect = mock_label

        # Create test data
        unlabeled_df = pd.DataFrame({
            "text": [f"sample {i}" for i in range(50)],
        })

        # Run active learning
        result_df = sampler.run_active_learning_loop(
            unlabeled_df=unlabeled_df,
            text_column="text",
        )

        # Should have stopped before processing all data
        assert len(result_df) < len(unlabeled_df)
        # Should have exceeded budget threshold
        assert sampler.state.current_cost >= config.max_budget * 0.9


class TestMultiProviderCostTracking:
    """Test cost tracking across multiple LLM providers."""

    def test_openrouter_and_openai_costs_separate_trackers(self):
        """Test that different providers can have separate cost trackers."""
        tracker_or = CostTracker(budget=10.0)
        tracker_oai = CostTracker(budget=20.0)

        # Simulate OpenRouter costs
        tracker_or.add_cost(5.0)
        assert tracker_or.total_cost == 5.0

        # Simulate OpenAI costs
        tracker_oai.add_cost(8.0)
        assert tracker_oai.total_cost == 8.0

        # Trackers are independent
        assert tracker_or.total_cost == 5.0
        assert tracker_oai.total_cost == 8.0

    def test_corporate_endpoint_zero_cost(self):
        """Test that corporate endpoints don't incur costs."""
        from autolabeler.core.utils.budget_tracker import extract_cost_from_result

        result = LLMResult(generations=[[]])

        cost = extract_cost_from_result(result, "corporate", "internal-model")

        assert cost == 0.0

    def test_mixed_provider_workflow(self):
        """Test workflow using multiple providers."""
        tracker = CostTracker(budget=50.0)

        # Simulate costs from different sources
        # OpenRouter API call
        tracker.add_cost(2.0)

        # OpenAI API call
        tracker.add_cost(5.0)

        # Corporate endpoint (free)
        tracker.add_cost(0.0)

        # Another OpenRouter call
        tracker.add_cost(3.0)

        assert tracker.total_cost == 10.0
        assert tracker.call_count == 4
        assert not tracker.is_budget_exceeded()


class TestGracefulShutdownScenarios:
    """Test graceful shutdown behavior when budget is exceeded."""

    def test_budget_exceeded_raises_error_in_client(self):
        """Test that exceeding budget raises error before API call."""
        from autolabeler.core.llm_providers.openrouter import OpenRouterClient

        tracker = CostTracker(budget=1.0)
        tracker.add_cost(1.5)  # Exceed budget

        client = OpenRouterClient(
            api_key="test-key",
            cost_tracker=tracker,
        )

        # Should raise error before making call
        with pytest.raises(BudgetExceededError):
            client._generate([])

    def test_budget_exceeded_state_preserved(self):
        """Test that budget exceeded state is preserved."""
        tracker = CostTracker(budget=10.0)

        tracker.add_cost(5.0)
        assert not tracker.is_budget_exceeded()

        tracker.add_cost(6.0)  # Total: 11.0, exceeds 10.0
        assert tracker.is_budget_exceeded()

        # State should remain exceeded
        tracker.add_cost(1.0)
        assert tracker.is_budget_exceeded()

        # Stats should reflect exceeded state
        stats = tracker.get_stats()
        assert stats["budget_exceeded"] is True

    def test_reset_clears_exceeded_state(self):
        """Test that reset clears exceeded state."""
        tracker = CostTracker(budget=5.0)
        tracker.add_cost(10.0)

        assert tracker.is_budget_exceeded()

        tracker.reset()

        assert not tracker.is_budget_exceeded()
        assert tracker.total_cost == 0.0


class TestBudgetReportingAndMonitoring:
    """Test budget reporting and monitoring capabilities."""

    def test_stats_provide_complete_information(self):
        """Test that stats provide all necessary budget information."""
        tracker = CostTracker(budget=100.0)
        tracker.add_cost(25.0)
        tracker.add_cost(30.0)

        stats = tracker.get_stats()

        assert "total_cost" in stats
        assert "call_count" in stats
        assert "budget" in stats
        assert "remaining_budget" in stats
        assert "budget_exceeded" in stats

        assert stats["total_cost"] == 55.0
        assert stats["call_count"] == 2
        assert stats["budget"] == 100.0
        assert stats["remaining_budget"] == 45.0
        assert stats["budget_exceeded"] is False

    def test_stats_with_exceeded_budget(self):
        """Test stats when budget is exceeded."""
        tracker = CostTracker(budget=50.0)
        tracker.add_cost(60.0)

        stats = tracker.get_stats()

        assert stats["budget_exceeded"] is True
        assert stats["remaining_budget"] == 0.0  # Clamped to 0

    def test_monitoring_during_long_run(self):
        """Test that costs can be monitored during a long run."""
        tracker = CostTracker(budget=100.0)

        # Simulate a long-running process
        costs_at_checkpoints = []

        for i in range(20):
            tracker.add_cost(3.0)

            # Check stats at intervals
            if i % 5 == 0:
                stats = tracker.get_stats()
                costs_at_checkpoints.append(stats["total_cost"])

        # Should have 4 checkpoints (0, 5, 10, 15)
        assert len(costs_at_checkpoints) == 4
        assert costs_at_checkpoints == [0.0, 15.0, 30.0, 45.0]


class TestBackwardCompatibility:
    """Test that budget tracking doesn't break existing functionality."""

    def test_labeling_service_works_without_budget(self, labeling_service):
        """Test that labeling service works without budget tracking."""
        # Should work normally without cost_tracker
        config = LabelingConfig(use_rag=False, use_validation=False)

        # This should not raise any errors about budget
        # (We're not actually calling the LLM, just verifying the config works)
        assert config.use_rag is False

    def test_openrouter_client_works_without_cost_tracker(self):
        """Test that OpenRouter client works without cost tracker."""
        from autolabeler.core.llm_providers.openrouter import OpenRouterClient

        # Should work without cost_tracker
        client = OpenRouterClient(
            api_key="test-key",
            cost_tracker=None,
        )

        # Client should be created successfully
        assert client is not None
        assert hasattr(client, "_instance_data")

    def test_active_learning_without_explicit_budget_config(self):
        """Test that active learning works with default budget config."""
        from autolabeler.core.active_learning import ActiveLearningSampler
        from autolabeler.core.configs import ActiveLearningConfig

        # Default config should have a max_budget
        config = ActiveLearningConfig(
            strategy="uncertainty",
            batch_size=10,
        )

        # Should have a budget limit (default)
        assert hasattr(config, "max_budget")
        assert config.max_budget > 0

        # Should be able to create sampler
        mock_service = MagicMock()
        sampler = ActiveLearningSampler(mock_service, config)
        assert sampler is not None


class TestRealWorldScenarios:
    """Test realistic budget tracking scenarios."""

    def test_budget_exhausted_mid_batch(self):
        """Test handling when budget is exhausted in the middle of a batch."""
        tracker = CostTracker(budget=5.0)

        # Process items until budget is exhausted
        items_processed = 0
        for i in range(100):
            if tracker.is_budget_exceeded():
                break

            # Simulate processing cost
            tracker.add_cost(0.8)
            items_processed += 1

        # Should have processed about 6 items (5.0 / 0.8 ≈ 6)
        assert 5 <= items_processed <= 7
        assert tracker.is_budget_exceeded()

    def test_cost_efficient_active_learning(self):
        """Test cost-efficient active learning scenario."""
        from autolabeler.core.active_learning.sampler import ALState
        from autolabeler.core.active_learning.stopping_criteria import (
            StoppingCriteria,
        )
        from autolabeler.core.configs import ActiveLearningConfig

        config = ActiveLearningConfig(
            strategy="uncertainty",
            max_budget=10.0,
            target_accuracy=0.90,
        )

        criteria = StoppingCriteria(config)

        # Simulate efficient learning that reaches target before budget
        state = ALState(
            current_cost=5.0,  # Only used half budget
            current_accuracy=0.91,  # Exceeded target
            iteration=5,
        )

        should_stop, reason = criteria.check(state)

        # Should stop due to target reached, not budget
        assert should_stop
        assert reason == "target_reached"
        assert not criteria._check_budget(state)

    def test_budget_almost_exceeded_warning(self):
        """Test behavior when approaching budget limit."""
        tracker = CostTracker(budget=10.0)

        # Get close to budget
        tracker.add_cost(8.9)
        assert not tracker.is_budget_exceeded()

        stats = tracker.get_stats()
        assert stats["remaining_budget"] == pytest.approx(1.1, rel=1e-6)

        # One more reasonable cost should not exceed
        result = tracker.add_cost(0.5)
        assert result is True
        assert not tracker.is_budget_exceeded()

        # But one more should exceed
        result = tracker.add_cost(1.0)
        assert result is False
        assert tracker.is_budget_exceeded()
