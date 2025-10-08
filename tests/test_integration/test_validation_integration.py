"""
Integration tests for StructuredOutputValidator with LabelingService.

These tests verify that the validator works correctly when integrated
with the actual LabelingService and real LLM clients.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

from autolabeler import Settings
from autolabeler.core.labeling.labeling_service import LabelingService
from autolabeler.core.configs import LabelingConfig
from autolabeler.models import LabelResponse


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        openrouter_api_key="test-key",
        llm_model="test-model",
        results_dir=tempfile.mkdtemp(),
    )


@pytest.fixture
def labeling_service(settings):
    """Create a LabelingService instance."""
    return LabelingService(
        dataset_name="test_dataset",
        settings=settings,
    )


class TestValidationIntegration:
    """Integration tests for validation with LabelingService."""

    @patch("autolabeler.core.llm_providers.openrouter.OpenRouterClient")
    def test_label_text_with_validation_enabled(
        self, mock_openrouter, labeling_service, settings
    ):
        """Test labeling with validation enabled."""
        # Mock the LLM client
        mock_client = Mock()
        mock_openrouter.return_value = mock_client

        # Mock successful response
        mock_response = LabelResponse(
            label="positive",
            confidence=0.9,
            reasoning="Strong positive sentiment",
        )

        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_response
        mock_client.with_structured_output.return_value = mock_structured_llm

        # Configure with validation enabled
        config = LabelingConfig(
            use_validation=True,
            validation_max_retries=3,
            allowed_labels=["positive", "negative", "neutral"],
            use_rag=False,
        )

        result = labeling_service.label_text(
            text="This is a great product!",
            config=config,
        )

        assert result.label == "positive"
        assert result.confidence == 0.9

    @patch("autolabeler.core.llm_providers.openrouter.OpenRouterClient")
    def test_label_text_with_validation_disabled(
        self, mock_openrouter, labeling_service, settings
    ):
        """Test labeling with validation disabled (legacy mode)."""
        mock_client = Mock()
        mock_openrouter.return_value = mock_client

        mock_response = LabelResponse(
            label="negative",
            confidence=0.85,
        )

        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_response
        mock_client.with_structured_output.return_value = mock_structured_llm

        config = LabelingConfig(
            use_validation=False,
            use_rag=False,
        )

        result = labeling_service.label_text(
            text="This product is terrible",
            config=config,
        )

        assert result.label == "negative"
        assert result.confidence == 0.85

    @patch("autolabeler.core.llm_providers.openrouter.OpenRouterClient")
    def test_validation_retry_on_invalid_label(
        self, mock_openrouter, labeling_service, settings
    ):
        """Test that validation retries when label is invalid."""
        mock_client = Mock()
        mock_openrouter.return_value = mock_client

        # First attempt: invalid label
        invalid_response = LabelResponse(
            label="very_positive",  # Not in allowed_labels
            confidence=0.9,
        )

        # Second attempt: valid label
        valid_response = LabelResponse(
            label="positive",
            confidence=0.9,
        )

        mock_structured_llm = Mock()
        mock_structured_llm.invoke.side_effect = [invalid_response, valid_response]
        mock_client.with_structured_output.return_value = mock_structured_llm

        config = LabelingConfig(
            use_validation=True,
            validation_max_retries=3,
            allowed_labels=["positive", "negative", "neutral"],
            use_rag=False,
        )

        result = labeling_service.label_text(
            text="Great product!",
            config=config,
        )

        assert result.label == "positive"
        # Should have made 2 calls (1 failed, 1 succeeded)
        assert mock_structured_llm.invoke.call_count == 2

    @patch("autolabeler.core.llm_providers.openrouter.OpenRouterClient")
    def test_get_validation_statistics(
        self, mock_openrouter, labeling_service, settings
    ):
        """Test that validation statistics are tracked correctly."""
        mock_client = Mock()
        mock_openrouter.return_value = mock_client

        mock_response = LabelResponse(label="positive", confidence=0.9)
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_response
        mock_client.with_structured_output.return_value = mock_structured_llm

        config = LabelingConfig(
            use_validation=True,
            validation_max_retries=3,
            use_rag=False,
        )

        # Make a few labeling calls
        for _ in range(3):
            labeling_service.label_text(
                text="Test text",
                config=config,
            )

        # Get statistics
        stats = labeling_service.get_validation_stats()

        assert stats["total_validation_attempts"] == 3
        assert stats["total_successful_validations"] == 3
        assert stats["overall_success_rate"] == 100.0

    @patch("autolabeler.core.llm_providers.openrouter.OpenRouterClient")
    def test_validation_without_allowed_labels(
        self, mock_openrouter, labeling_service, settings
    ):
        """Test validation without specifying allowed_labels."""
        mock_client = Mock()
        mock_openrouter.return_value = mock_client

        mock_response = LabelResponse(
            label="any_label",
            confidence=0.9,
        )

        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_response
        mock_client.with_structured_output.return_value = mock_structured_llm

        config = LabelingConfig(
            use_validation=True,
            validation_max_retries=3,
            allowed_labels=None,  # No label restrictions
            use_rag=False,
        )

        result = labeling_service.label_text(
            text="Test text",
            config=config,
        )

        # Should accept any label when allowed_labels is None
        assert result.label == "any_label"

    def test_validator_caching(self, labeling_service):
        """Test that validators are cached per configuration."""
        config1 = LabelingConfig(
            use_validation=True,
            validation_max_retries=3,
            use_rag=False,
        )

        config2 = LabelingConfig(
            use_validation=True,
            validation_max_retries=5,  # Different retry count
            use_rag=False,
        )

        # Get validators for different configs
        validator1 = labeling_service._get_validator_for_config(config1)
        validator2 = labeling_service._get_validator_for_config(config2)

        # Should be different validators
        assert validator1 is not validator2
        assert validator1.max_retries == 3
        assert validator2.max_retries == 5

        # Getting same config again should return cached validator
        validator1_again = labeling_service._get_validator_for_config(config1)
        assert validator1 is validator1_again


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
