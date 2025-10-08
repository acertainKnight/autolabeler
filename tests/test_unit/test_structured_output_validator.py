"""Unit tests for StructuredOutputValidator."""

import pytest
from pydantic import BaseModel, Field, ValidationError
from typing import Optional


# Mock models for testing
class LabelResponse(BaseModel):
    """Response model for labeling."""

    label: str = Field(description="Predicted label")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: Optional[str] = Field(None, description="Explanation for label")


class MultiLabelResponse(BaseModel):
    """Response model for multi-label classification."""

    labels: list[str] = Field(description="Predicted labels")
    confidence_scores: list[float] = Field(description="Confidence for each label")
    reasoning: Optional[str] = None


# Mock StructuredOutputValidator
class StructuredOutputValidator:
    """
    Validates and ensures structured outputs from LLMs.

    This is a mock implementation for testing purposes.
    The actual implementation will use Instructor in Phase 1.
    """

    def __init__(self, max_retries: int = 3):
        """
        Initialize validator.

        Args:
            max_retries: Maximum number of retry attempts
        """
        self.max_retries = max_retries
        self.validation_attempts = 0
        self.validation_errors = []

    def validate(self, output: dict, response_model: type[BaseModel]) -> BaseModel:
        """
        Validate output against Pydantic model.

        Args:
            output: Raw output dictionary
            response_model: Pydantic model to validate against

        Returns:
            Validated model instance

        Raises:
            ValidationError: If validation fails after retries
        """
        self.validation_attempts += 1

        try:
            return response_model(**output)
        except ValidationError as e:
            self.validation_errors.append(e)
            raise

    def validate_with_retry(
        self, output: dict, response_model: type[BaseModel], retry_fn=None
    ) -> BaseModel:
        """
        Validate with automatic retry on failure.

        Args:
            output: Raw output dictionary
            response_model: Pydantic model to validate against
            retry_fn: Optional function to call for retry

        Returns:
            Validated model instance

        Raises:
            ValidationError: If validation fails after all retries
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return self.validate(output, response_model)
            except ValidationError as e:
                last_error = e
                if attempt < self.max_retries - 1 and retry_fn:
                    # Call retry function to get new output
                    output = retry_fn(output, e)

        # All retries exhausted
        raise last_error

    def get_validation_stats(self) -> dict:
        """Get validation statistics."""
        return {
            "total_attempts": self.validation_attempts,
            "total_errors": len(self.validation_errors),
            "success_rate": (
                (self.validation_attempts - len(self.validation_errors))
                / max(1, self.validation_attempts)
            ),
        }


@pytest.mark.unit
class TestStructuredOutputValidator:
    """Test suite for StructuredOutputValidator."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = StructuredOutputValidator(max_retries=3)
        assert validator.max_retries == 3
        assert validator.validation_attempts == 0
        assert len(validator.validation_errors) == 0

    def test_validate_valid_output(self):
        """Test validation with valid output."""
        validator = StructuredOutputValidator()

        output = {
            "label": "positive",
            "confidence": 0.95,
            "reasoning": "Contains positive sentiment",
        }

        result = validator.validate(output, LabelResponse)

        assert isinstance(result, LabelResponse)
        assert result.label == "positive"
        assert result.confidence == 0.95
        assert result.reasoning == "Contains positive sentiment"
        assert validator.validation_attempts == 1

    def test_validate_missing_required_field(self):
        """Test validation fails with missing required field."""
        validator = StructuredOutputValidator()

        output = {
            "label": "positive",
            # Missing confidence field
        }

        with pytest.raises(ValidationError):
            validator.validate(output, LabelResponse)

        assert validator.validation_attempts == 1
        assert len(validator.validation_errors) == 1

    def test_validate_invalid_type(self):
        """Test validation fails with invalid field type."""
        validator = StructuredOutputValidator()

        output = {
            "label": "positive",
            "confidence": "high",  # Should be float
        }

        with pytest.raises(ValidationError):
            validator.validate(output, LabelResponse)

    def test_validate_out_of_range_value(self):
        """Test validation fails with out-of-range value."""
        validator = StructuredOutputValidator()

        output = {
            "label": "positive",
            "confidence": 1.5,  # Should be <= 1.0
        }

        with pytest.raises(ValidationError):
            validator.validate(output, LabelResponse)

    def test_validate_optional_field_missing(self):
        """Test validation succeeds with missing optional field."""
        validator = StructuredOutputValidator()

        output = {
            "label": "positive",
            "confidence": 0.85,
            # reasoning is optional, so it's okay to omit
        }

        result = validator.validate(output, LabelResponse)

        assert result.reasoning is None

    def test_validate_extra_fields_ignored(self):
        """Test validation ignores extra fields."""
        validator = StructuredOutputValidator()

        output = {
            "label": "positive",
            "confidence": 0.85,
            "extra_field": "should be ignored",
        }

        result = validator.validate(output, LabelResponse)

        assert not hasattr(result, "extra_field")

    def test_validate_multi_label_output(self):
        """Test validation with multi-label response."""
        validator = StructuredOutputValidator()

        output = {
            "labels": ["positive", "enthusiastic"],
            "confidence_scores": [0.9, 0.7],
        }

        result = validator.validate(output, MultiLabelResponse)

        assert isinstance(result, MultiLabelResponse)
        assert result.labels == ["positive", "enthusiastic"]
        assert result.confidence_scores == [0.9, 0.7]

    def test_validate_with_retry_success_first_attempt(self):
        """Test validate_with_retry succeeds on first attempt."""
        validator = StructuredOutputValidator(max_retries=3)

        output = {
            "label": "positive",
            "confidence": 0.85,
        }

        result = validator.validate_with_retry(output, LabelResponse)

        assert isinstance(result, LabelResponse)
        assert validator.validation_attempts == 1

    def test_validate_with_retry_success_after_retry(self):
        """Test validate_with_retry succeeds after retry."""
        validator = StructuredOutputValidator(max_retries=3)

        # Start with invalid output
        output = {
            "label": "positive",
            "confidence": "invalid",
        }

        # Retry function fixes the output
        def retry_fn(output, error):
            return {
                "label": "positive",
                "confidence": 0.85,
            }

        result = validator.validate_with_retry(output, LabelResponse, retry_fn=retry_fn)

        assert isinstance(result, LabelResponse)
        assert validator.validation_attempts == 2  # First attempt + 1 retry

    def test_validate_with_retry_exhausts_retries(self):
        """Test validate_with_retry fails after exhausting retries."""
        validator = StructuredOutputValidator(max_retries=3)

        output = {
            "label": "positive",
            "confidence": "always_invalid",
        }

        # Retry function doesn't fix the problem
        def retry_fn(output, error):
            return output  # Return same invalid output

        with pytest.raises(ValidationError):
            validator.validate_with_retry(output, LabelResponse, retry_fn=retry_fn)

        assert validator.validation_attempts == 3  # All retries used
        assert len(validator.validation_errors) == 3

    def test_get_validation_stats_no_errors(self):
        """Test validation stats with no errors."""
        validator = StructuredOutputValidator()

        # Perform 5 successful validations
        for i in range(5):
            output = {"label": "positive", "confidence": 0.8}
            validator.validate(output, LabelResponse)

        stats = validator.get_validation_stats()

        assert stats["total_attempts"] == 5
        assert stats["total_errors"] == 0
        assert stats["success_rate"] == 1.0

    def test_get_validation_stats_with_errors(self):
        """Test validation stats with some errors."""
        validator = StructuredOutputValidator()

        # 3 successful validations
        for i in range(3):
            output = {"label": "positive", "confidence": 0.8}
            validator.validate(output, LabelResponse)

        # 2 failed validations
        for i in range(2):
            try:
                output = {"label": "positive"}  # Missing confidence
                validator.validate(output, LabelResponse)
            except ValidationError:
                pass

        stats = validator.get_validation_stats()

        assert stats["total_attempts"] == 5
        assert stats["total_errors"] == 2
        assert stats["success_rate"] == 0.6

    def test_validation_error_tracking(self):
        """Test that validation errors are tracked."""
        validator = StructuredOutputValidator()

        # Cause 3 validation errors
        invalid_outputs = [
            {"label": "positive"},  # Missing confidence
            {"label": "positive", "confidence": "invalid"},  # Wrong type
            {"label": "positive", "confidence": 2.0},  # Out of range
        ]

        for output in invalid_outputs:
            try:
                validator.validate(output, LabelResponse)
            except ValidationError:
                pass

        assert len(validator.validation_errors) == 3
        assert validator.validation_attempts == 3

    @pytest.mark.parametrize(
        "output,should_pass",
        [
            ({"label": "pos", "confidence": 0.5}, True),
            ({"label": "pos", "confidence": 0.0}, True),  # Edge case
            ({"label": "pos", "confidence": 1.0}, True),  # Edge case
            ({"label": "pos", "confidence": -0.1}, False),  # Below range
            ({"label": "pos", "confidence": 1.1}, False),  # Above range
            ({"confidence": 0.5}, False),  # Missing label
        ],
    )
    def test_validation_edge_cases(self, output, should_pass):
        """Test various edge cases for validation."""
        validator = StructuredOutputValidator()

        if should_pass:
            result = validator.validate(output, LabelResponse)
            assert isinstance(result, LabelResponse)
        else:
            with pytest.raises(ValidationError):
                validator.validate(output, LabelResponse)

    def test_validation_with_complex_nested_model(self):
        """Test validation with nested Pydantic models."""

        class NestedResponse(BaseModel):
            label_response: LabelResponse
            metadata: dict

        validator = StructuredOutputValidator()

        output = {
            "label_response": {
                "label": "positive",
                "confidence": 0.85,
            },
            "metadata": {"source": "test"},
        }

        result = validator.validate(output, NestedResponse)

        assert isinstance(result, NestedResponse)
        assert isinstance(result.label_response, LabelResponse)
        assert result.label_response.label == "positive"

    def test_validation_preserves_data_types(self):
        """Test that validation preserves correct data types."""
        validator = StructuredOutputValidator()

        output = {
            "label": "positive",
            "confidence": 0.85,
        }

        result = validator.validate(output, LabelResponse)

        assert isinstance(result.label, str)
        assert isinstance(result.confidence, float)

    def test_multiple_validators_independent(self):
        """Test that multiple validators maintain independent state."""
        validator1 = StructuredOutputValidator()
        validator2 = StructuredOutputValidator()

        output = {"label": "positive", "confidence": 0.85}

        validator1.validate(output, LabelResponse)
        validator1.validate(output, LabelResponse)

        assert validator1.validation_attempts == 2
        assert validator2.validation_attempts == 0
