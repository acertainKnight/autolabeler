"""
Comprehensive unit tests for StructuredOutputValidator.

Tests cover:
- Basic validation success scenarios
- Type validation errors and retry
- Business rule validation
- Error feedback construction
- Statistics tracking
- Edge cases and error handling
"""

from unittest.mock import Mock, MagicMock, patch
import pytest
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from autolabeler.core.validation.output_validator import (
    StructuredOutputValidator,
    ValidationError,
    create_field_value_validator,
    create_confidence_validator,
    create_non_empty_validator,
)


# Test models
class SimpleResponse(BaseModel):
    """Simple test response model."""

    label: str = Field(description="The predicted label")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


class ComplexResponse(BaseModel):
    """Complex test response model with optional fields."""

    label: str
    confidence: float
    reasoning: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


# Test fixtures
@pytest.fixture
def mock_client():
    """Create a mock LangChain BaseChatModel."""
    client = Mock()
    return client


@pytest.fixture
def validator(mock_client):
    """Create a StructuredOutputValidator with mock client."""
    return StructuredOutputValidator(mock_client, max_retries=3)


@pytest.fixture
def sample_prompt():
    """Sample prompt for testing."""
    return "Label this text: 'This is a great product!'"


# Test StructuredOutputValidator initialization
class TestInitialization:
    """Test validator initialization."""

    def test_init_with_defaults(self, mock_client):
        """Test initialization with default parameters."""
        validator = StructuredOutputValidator(mock_client)

        assert validator.client == mock_client
        assert validator.max_retries == 3
        assert validator.enable_fallback is True
        assert validator.total_attempts == 0
        assert validator.successful_validations == 0
        assert validator.failed_validations == 0

    def test_init_with_custom_retries(self, mock_client):
        """Test initialization with custom max_retries."""
        validator = StructuredOutputValidator(mock_client, max_retries=5)

        assert validator.max_retries == 5

    def test_init_with_fallback_disabled(self, mock_client):
        """Test initialization with fallback disabled."""
        validator = StructuredOutputValidator(
            mock_client, max_retries=2, enable_fallback=False
        )

        assert validator.enable_fallback is False


# Test successful validation
class TestSuccessfulValidation:
    """Test cases where validation succeeds."""

    def test_first_attempt_success(self, validator, mock_client, sample_prompt):
        """Test successful validation on first attempt."""
        # Mock successful response
        mock_response = SimpleResponse(label="positive", confidence=0.9)

        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_response
        mock_client.with_structured_output.return_value = mock_structured_llm

        result = validator.validate_and_retry(
            prompt=sample_prompt,
            response_model=SimpleResponse,
        )

        assert result == mock_response
        assert validator.successful_validations == 1
        assert validator.failed_validations == 0
        assert validator.retry_count_histogram[0] == 1

    def test_success_with_validation_rules(self, validator, mock_client, sample_prompt):
        """Test successful validation with custom business rules."""
        mock_response = SimpleResponse(label="positive", confidence=0.9)

        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_response
        mock_client.with_structured_output.return_value = mock_structured_llm

        # Define a validation rule that passes
        def validate_label(response: SimpleResponse) -> tuple[bool, str]:
            if response.label in {"positive", "negative", "neutral"}:
                return True, ""
            return False, "Invalid label"

        result = validator.validate_and_retry(
            prompt=sample_prompt,
            response_model=SimpleResponse,
            validation_rules=[validate_label],
        )

        assert result == mock_response
        assert validator.successful_validations == 1

    def test_success_after_retry(self, validator, mock_client, sample_prompt):
        """Test successful validation after one retry."""
        # First attempt: invalid response
        invalid_response = SimpleResponse(label="invalid_label", confidence=0.9)
        # Second attempt: valid response
        valid_response = SimpleResponse(label="positive", confidence=0.9)

        mock_structured_llm = Mock()
        mock_structured_llm.invoke.side_effect = [invalid_response, valid_response]
        mock_client.with_structured_output.return_value = mock_structured_llm

        # Define validation rule
        def validate_label(response: SimpleResponse) -> tuple[bool, str]:
            if response.label in {"positive", "negative", "neutral"}:
                return True, ""
            return False, f"Invalid label: {response.label}"

        result = validator.validate_and_retry(
            prompt=sample_prompt,
            response_model=SimpleResponse,
            validation_rules=[validate_label],
        )

        assert result == valid_response
        assert validator.successful_validations == 1
        assert mock_structured_llm.invoke.call_count == 2
        assert validator.retry_count_histogram[1] == 1  # Success on second attempt (index 1)


# Test validation failures
class TestValidationFailures:
    """Test cases where validation fails."""

    def test_type_validation_failure_all_retries(
        self, validator, mock_client, sample_prompt
    ):
        """Test failure when type validation fails on all attempts."""
        # Mock Pydantic validation error
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.side_effect = PydanticValidationError.from_exception_data(
            "SimpleResponse",
            [
                {
                    "type": "missing",
                    "loc": ("label",),
                    "msg": "Field required",
                    "input": {},
                }
            ],
        )
        mock_client.with_structured_output.return_value = mock_structured_llm

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_and_retry(
                prompt=sample_prompt,
                response_model=SimpleResponse,
            )

        error = exc_info.value
        assert error.attempts == 4  # Initial + 3 retries
        assert validator.failed_validations == 1
        assert len(error.validation_errors) == 4
        assert "Type validation errors" in error.validation_errors[0]

    def test_business_rule_failure_all_retries(
        self, validator, mock_client, sample_prompt
    ):
        """Test failure when business rule fails on all attempts."""
        # Always return invalid label
        invalid_response = SimpleResponse(label="invalid_label", confidence=0.9)

        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = invalid_response
        mock_client.with_structured_output.return_value = mock_structured_llm

        # Define strict validation rule
        def validate_label(response: SimpleResponse) -> tuple[bool, str]:
            if response.label in {"positive", "negative"}:
                return True, ""
            return False, f"Label must be 'positive' or 'negative', got '{response.label}'"

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_and_retry(
                prompt=sample_prompt,
                response_model=SimpleResponse,
                validation_rules=[validate_label],
            )

        error = exc_info.value
        assert error.attempts == 4
        assert validator.failed_validations == 1
        assert "Label must be 'positive' or 'negative'" in error.validation_errors[0]

    def test_multiple_validation_rules_failure(
        self, validator, mock_client, sample_prompt
    ):
        """Test with multiple validation rules where one fails."""
        response = SimpleResponse(label="positive", confidence=1.5)  # Invalid confidence

        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = response
        mock_client.with_structured_output.return_value = mock_structured_llm

        def validate_label(resp: SimpleResponse) -> tuple[bool, str]:
            return True, ""

        def validate_confidence(resp: SimpleResponse) -> tuple[bool, str]:
            if 0.0 <= resp.confidence <= 1.0:
                return True, ""
            return False, f"Confidence {resp.confidence} out of range [0.0, 1.0]"

        with pytest.raises(ValidationError):
            validator.validate_and_retry(
                prompt=sample_prompt,
                response_model=SimpleResponse,
                validation_rules=[validate_label, validate_confidence],
            )


# Test error feedback construction
class TestErrorFeedback:
    """Test error feedback prompt construction."""

    def test_construct_error_feedback_with_response(self, validator):
        """Test error feedback includes previous response."""
        original_prompt = "Label this text"
        previous_response = SimpleResponse(label="invalid", confidence=0.5)
        error_message = "Invalid label value"
        attempt_number = 1

        feedback = validator._construct_error_feedback(
            original_prompt=original_prompt,
            previous_response=previous_response,
            error_message=error_message,
            attempt_number=attempt_number,
        )

        assert original_prompt in feedback
        assert error_message in feedback
        assert "attempt 1" in feedback.lower()
        assert '"label": "invalid"' in feedback
        assert '"confidence": 0.5' in feedback

    def test_construct_error_feedback_without_response(self, validator):
        """Test error feedback when previous response is None."""
        original_prompt = "Label this text"
        error_message = "Parsing failed"
        attempt_number = 2

        feedback = validator._construct_error_feedback(
            original_prompt=original_prompt,
            previous_response=None,
            error_message=error_message,
            attempt_number=attempt_number,
        )

        assert original_prompt in feedback
        assert error_message in feedback
        assert "attempt 2" in feedback.lower()
        assert "Previous Response" not in feedback

    def test_format_pydantic_error(self, validator):
        """Test formatting of Pydantic validation errors."""
        pydantic_error = PydanticValidationError.from_exception_data(
            "SimpleResponse",
            [
                {
                    "type": "missing",
                    "loc": ("label",),
                    "msg": "Field required",
                    "input": {},
                },
                {
                    "type": "float_parsing",
                    "loc": ("confidence",),
                    "msg": "Input should be a valid number",
                    "input": "invalid",
                },
            ],
        )

        formatted = validator._format_pydantic_error(pydantic_error)

        assert "Type validation errors" in formatted
        assert "label" in formatted
        assert "Field required" in formatted
        assert "confidence" in formatted
        assert "Input should be a valid number" in formatted


# Test statistics tracking
class TestStatistics:
    """Test statistics tracking and reporting."""

    def test_statistics_after_successful_validation(
        self, validator, mock_client, sample_prompt
    ):
        """Test statistics are updated after successful validation."""
        mock_response = SimpleResponse(label="positive", confidence=0.9)
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_response
        mock_client.with_structured_output.return_value = mock_structured_llm

        validator.validate_and_retry(sample_prompt, SimpleResponse)

        stats = validator.get_statistics()

        assert stats["total_attempts"] == 1
        assert stats["successful_validations"] == 1
        assert stats["failed_validations"] == 0
        assert stats["success_rate"] == 100.0
        assert stats["average_retries"] == 0.0
        assert stats["first_attempt_success_rate"] == 100.0

    def test_statistics_after_multiple_validations(self, validator, mock_client):
        """Test statistics after multiple validation attempts."""
        mock_structured_llm = Mock()
        mock_client.with_structured_output.return_value = mock_structured_llm

        # First validation: success on first try
        mock_structured_llm.invoke.return_value = SimpleResponse(
            label="positive", confidence=0.9
        )
        validator.validate_and_retry("prompt1", SimpleResponse)

        # Second validation: success on second try
        mock_structured_llm.invoke.side_effect = [
            SimpleResponse(label="invalid", confidence=0.9),
            SimpleResponse(label="positive", confidence=0.9),
        ]

        def validate_label(resp: SimpleResponse) -> tuple[bool, str]:
            if resp.label in {"positive", "negative"}:
                return True, ""
            return False, "Invalid label"

        validator.validate_and_retry(
            "prompt2", SimpleResponse, validation_rules=[validate_label]
        )

        stats = validator.get_statistics()

        assert stats["total_attempts"] == 2
        assert stats["successful_validations"] == 2
        assert stats["average_retries"] == 0.5  # (0 + 1) / 2

    def test_reset_statistics(self, validator, mock_client, sample_prompt):
        """Test resetting statistics."""
        mock_response = SimpleResponse(label="positive", confidence=0.9)
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_response
        mock_client.with_structured_output.return_value = mock_structured_llm

        validator.validate_and_retry(sample_prompt, SimpleResponse)

        # Verify stats exist
        assert validator.total_attempts == 1

        # Reset
        validator.reset_statistics()

        # Verify reset
        assert validator.total_attempts == 0
        assert validator.successful_validations == 0
        assert validator.failed_validations == 0
        assert len(validator.retry_count_histogram) == 0


# Test validation rule builders
class TestValidationRuleBuilders:
    """Test the helper functions for creating validation rules."""

    def test_create_field_value_validator_success(self):
        """Test field value validator with valid value."""
        validator = create_field_value_validator("label", {"positive", "negative"})

        response = SimpleResponse(label="positive", confidence=0.9)
        is_valid, error = validator(response)

        assert is_valid is True
        assert error == ""

    def test_create_field_value_validator_failure(self):
        """Test field value validator with invalid value."""
        validator = create_field_value_validator("label", {"positive", "negative"})

        response = SimpleResponse(label="neutral", confidence=0.9)
        is_valid, error = validator(response)

        assert is_valid is False
        assert "neutral" in error
        assert "positive" in error or "negative" in error

    def test_create_field_value_validator_missing_field(self):
        """Test field value validator with missing field."""
        validator = create_field_value_validator("missing_field", {"value"})

        response = SimpleResponse(label="positive", confidence=0.9)
        is_valid, error = validator(response)

        assert is_valid is False
        assert "missing_field" in error

    def test_create_confidence_validator_success(self):
        """Test confidence validator with valid confidence."""
        validator = create_confidence_validator(0.0, 1.0)

        response = SimpleResponse(label="positive", confidence=0.5)
        is_valid, error = validator(response)

        assert is_valid is True
        assert error == ""

    def test_create_confidence_validator_out_of_range(self):
        """Test confidence validator with out-of-range confidence."""
        validator = create_confidence_validator(0.0, 1.0)

        response = SimpleResponse(label="positive", confidence=1.5)
        is_valid, error = validator(response)

        assert is_valid is False
        assert "1.5" in error
        assert "range" in error.lower()

    def test_create_confidence_validator_no_field(self):
        """Test confidence validator when field doesn't exist."""

        class NoConfidenceResponse(BaseModel):
            label: str

        validator = create_confidence_validator()
        response = NoConfidenceResponse(label="positive")
        is_valid, error = validator(response)

        # Should pass when field is optional
        assert is_valid is True

    def test_create_non_empty_validator_success(self):
        """Test non-empty validator with non-empty value."""
        validator = create_non_empty_validator("label")

        response = SimpleResponse(label="positive", confidence=0.9)
        is_valid, error = validator(response)

        assert is_valid is True
        assert error == ""

    def test_create_non_empty_validator_empty_string(self):
        """Test non-empty validator with empty string."""
        validator = create_non_empty_validator("label")

        response = SimpleResponse(label="", confidence=0.9)
        is_valid, error = validator(response)

        assert is_valid is False
        assert "cannot be empty" in error

    def test_create_non_empty_validator_whitespace(self):
        """Test non-empty validator with whitespace-only string."""
        validator = create_non_empty_validator("label")

        response = SimpleResponse(label="   ", confidence=0.9)
        is_valid, error = validator(response)

        assert is_valid is False
        assert "cannot be empty" in error


# Test edge cases
class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_validation_rules_list(self, validator, mock_client, sample_prompt):
        """Test with empty validation rules list."""
        mock_response = SimpleResponse(label="positive", confidence=0.9)
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_response
        mock_client.with_structured_output.return_value = mock_structured_llm

        result = validator.validate_and_retry(
            prompt=sample_prompt,
            response_model=SimpleResponse,
            validation_rules=[],
        )

        assert result == mock_response

    def test_validation_rule_raises_exception(
        self, validator, mock_client, sample_prompt
    ):
        """Test when validation rule itself raises an exception."""
        mock_response = SimpleResponse(label="positive", confidence=0.9)
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_response
        mock_client.with_structured_output.return_value = mock_structured_llm

        def buggy_validator(response: SimpleResponse) -> tuple[bool, str]:
            raise ValueError("Validator bug!")

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_and_retry(
                prompt=sample_prompt,
                response_model=SimpleResponse,
                validation_rules=[buggy_validator],
            )

        error = exc_info.value
        assert "Validator bug!" in str(error.validation_errors)

    def test_max_retries_zero(self, mock_client, sample_prompt):
        """Test with max_retries=0 (no retries allowed)."""
        validator = StructuredOutputValidator(mock_client, max_retries=0)

        invalid_response = SimpleResponse(label="invalid", confidence=0.9)
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = invalid_response
        mock_client.with_structured_output.return_value = mock_structured_llm

        def validate_label(resp: SimpleResponse) -> tuple[bool, str]:
            if resp.label in {"positive", "negative"}:
                return True, ""
            return False, "Invalid label"

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_and_retry(
                prompt=sample_prompt,
                response_model=SimpleResponse,
                validation_rules=[validate_label],
            )

        error = exc_info.value
        assert error.attempts == 1  # Only initial attempt, no retries

    def test_complex_model_with_optional_fields(self, validator, mock_client):
        """Test validation with complex model containing optional fields."""
        mock_response = ComplexResponse(
            label="positive",
            confidence=0.9,
            reasoning="Good sentiment indicators",
            metadata={"source": "test"},
        )

        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_response
        mock_client.with_structured_output.return_value = mock_structured_llm

        result = validator.validate_and_retry(
            prompt="Test prompt",
            response_model=ComplexResponse,
        )

        assert result == mock_response
        assert result.reasoning == "Good sentiment indicators"
        assert result.metadata == {"source": "test"}


# Integration tests
class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_retry_cycle_with_eventual_success(
        self, validator, mock_client, sample_prompt
    ):
        """Test a full retry cycle that eventually succeeds."""
        # Sequence of responses: fail, fail, succeed
        responses = [
            SimpleResponse(label="invalid1", confidence=0.5),
            SimpleResponse(label="invalid2", confidence=0.6),
            SimpleResponse(label="positive", confidence=0.9),
        ]

        mock_structured_llm = Mock()
        mock_structured_llm.invoke.side_effect = responses
        mock_client.with_structured_output.return_value = mock_structured_llm

        def validate_label(resp: SimpleResponse) -> tuple[bool, str]:
            if resp.label in {"positive", "negative", "neutral"}:
                return True, ""
            return False, f"Invalid label: {resp.label}"

        result = validator.validate_and_retry(
            prompt=sample_prompt,
            response_model=SimpleResponse,
            validation_rules=[validate_label],
        )

        assert result.label == "positive"
        assert mock_structured_llm.invoke.call_count == 3
        assert validator.successful_validations == 1
        assert validator.retry_count_histogram[2] == 1  # Success on 3rd attempt (index 2)

    def test_statistics_across_multiple_calls(self, validator, mock_client):
        """Test statistics accumulate correctly across multiple calls."""
        mock_structured_llm = Mock()
        mock_client.with_structured_output.return_value = mock_structured_llm

        # Call 1: Success first try
        mock_structured_llm.invoke.return_value = SimpleResponse(
            label="positive", confidence=0.9
        )
        validator.validate_and_retry("prompt1", SimpleResponse)

        # Call 2: Fail all retries
        mock_structured_llm.invoke.side_effect = PydanticValidationError.from_exception_data(
            "SimpleResponse",
            [{"type": "missing", "loc": ("label",), "msg": "Field required", "input": {}}],
        )
        try:
            validator.validate_and_retry("prompt2", SimpleResponse)
        except ValidationError:
            pass

        # Call 3: Success after one retry
        mock_structured_llm.invoke.side_effect = [
            SimpleResponse(label="invalid", confidence=0.9),
            SimpleResponse(label="negative", confidence=0.8),
        ]

        def validate_label(resp: SimpleResponse) -> tuple[bool, str]:
            if resp.label in {"positive", "negative"}:
                return True, ""
            return False, "Invalid"

        validator.validate_and_retry(
            "prompt3", SimpleResponse, validation_rules=[validate_label]
        )

        stats = validator.get_statistics()
        assert stats["total_attempts"] == 3
        assert stats["successful_validations"] == 2
        assert stats["failed_validations"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
