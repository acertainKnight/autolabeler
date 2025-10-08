"""
Structured Output Validator with automatic retry and error feedback.

This module implements multi-layer validation with automatic retry mechanisms
for LLM-based structured outputs using Instructor-style patterns.
"""

from __future__ import annotations

from typing import Any, Callable, Type, TypeVar

from langchain_core.language_models import BaseChatModel
from loguru import logger
from pydantic import BaseModel, ValidationError as PydanticValidationError

T = TypeVar("T", bound=BaseModel)


class ValidationError(Exception):
    """Custom exception for validation failures after all retries."""

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
        validation_errors: list[str] | None = None,
        attempts: int = 0,
    ):
        """
        Initialize ValidationError.

        Args:
            message: Human-readable error message
            original_error: The underlying exception that caused the failure
            validation_errors: List of validation error messages from attempts
            attempts: Number of attempts made before failure
        """
        super().__init__(message)
        self.original_error = original_error
        self.validation_errors = validation_errors or []
        self.attempts = attempts


class StructuredOutputValidator:
    """
    Enhanced structured output validation using automatic retry with error feedback.

    This validator implements a multi-layer validation approach:
    1. Type validation (automatic via Pydantic)
    2. Business rule validation (custom validation functions)
    3. Semantic validation (optional domain-specific checks)

    When validation fails, the LLM receives structured error feedback and
    is given another chance to correct the output.

    Features:
    - Automatic retry on validation failures (configurable max retries)
    - Structured error feedback to LLM for self-correction
    - Multi-layer validation (type, business rules, semantic)
    - Fallback strategies for persistent failures
    - Detailed logging and error tracking
    """

    def __init__(
        self,
        client: BaseChatModel,
        max_retries: int = 3,
        enable_fallback: bool = True,
    ):
        """
        Initialize the validator.

        Args:
            client: LangChain BaseChatModel instance for LLM calls
            max_retries: Maximum number of retry attempts (default: 3)
            enable_fallback: Whether to enable fallback strategies on final failure
        """
        self.client = client
        self.max_retries = max_retries
        self.enable_fallback = enable_fallback

        # Statistics tracking
        self.total_attempts = 0
        self.successful_validations = 0
        self.failed_validations = 0
        self.retry_count_histogram: dict[int, int] = {}

        logger.info(
            f"StructuredOutputValidator initialized with max_retries={max_retries}"
        )

    def validate_and_retry(
        self,
        prompt: str,
        response_model: Type[T],
        validation_rules: list[Callable[[T], tuple[bool, str]]] | None = None,
        method: str = "function_calling",
    ) -> T:
        """
        Validate LLM output with automatic retry on validation failures.

        This is the main entry point for validation. It attempts to get a valid
        structured output from the LLM, retrying with error feedback if validation
        fails.

        Args:
            prompt: The prompt to send to the LLM
            response_model: Pydantic model class for the expected response
            validation_rules: Optional list of validation functions. Each function
                takes the parsed model and returns (is_valid, error_message)
            method: LangChain structured output method (default: "function_calling")

        Returns:
            Validated instance of response_model

        Raises:
            ValidationError: If validation fails after all retry attempts

        Example:
            >>> validator = StructuredOutputValidator(client, max_retries=3)
            >>> result = validator.validate_and_retry(
            ...     prompt="Label this text: 'Great product!'",
            ...     response_model=LabelResponse,
            ...     validation_rules=[validate_label_in_schema]
            ... )
        """
        validation_rules = validation_rules or []
        validation_errors: list[str] = []
        current_prompt = prompt

        self.total_attempts += 1

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    f"Validation attempt {attempt + 1}/{self.max_retries + 1}"
                )

                # Get structured output from LLM
                structured_llm = self.client.with_structured_output(
                    response_model, method=method
                )
                response = structured_llm.invoke(current_prompt)

                # Run type validation (automatically done by Pydantic in invoke)
                # This is implicit - if we get here, Pydantic validation passed

                # Run business rule validation
                is_valid, error_message = self._run_business_rules(
                    response, validation_rules
                )

                if is_valid:
                    self.successful_validations += 1
                    self._update_retry_histogram(attempt)
                    logger.debug(f"Validation successful on attempt {attempt + 1}")
                    return response

                # Validation failed - construct error feedback for retry
                validation_errors.append(error_message)
                logger.warning(
                    f"Attempt {attempt + 1} failed validation: {error_message}"
                )

                if attempt < self.max_retries:
                    current_prompt = self._construct_error_feedback(
                        original_prompt=prompt,
                        previous_response=response,
                        error_message=error_message,
                        attempt_number=attempt + 1,
                    )
                else:
                    # Final attempt failed
                    self.failed_validations += 1
                    error = ValidationError(
                        f"Validation failed after {self.max_retries + 1} attempts",
                        validation_errors=validation_errors,
                        attempts=self.max_retries + 1,
                    )
                    raise error

            except ValidationError:
                # Re-raise our own ValidationError without catching it
                raise

            except PydanticValidationError as e:
                # Type validation failed
                error_msg = self._format_pydantic_error(e)
                validation_errors.append(error_msg)
                logger.warning(f"Attempt {attempt + 1} - Pydantic validation error: {error_msg}")

                if attempt < self.max_retries:
                    current_prompt = self._construct_error_feedback(
                        original_prompt=prompt,
                        previous_response=None,
                        error_message=error_msg,
                        attempt_number=attempt + 1,
                    )
                else:
                    self.failed_validations += 1
                    raise ValidationError(
                        f"Type validation failed after {self.max_retries + 1} attempts",
                        original_error=e,
                        validation_errors=validation_errors,
                        attempts=self.max_retries + 1,
                    )

            except Exception as e:
                # Unexpected error
                error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
                validation_errors.append(error_msg)
                logger.error(f"Attempt {attempt + 1} - {error_msg}")

                if attempt < self.max_retries:
                    current_prompt = self._construct_error_feedback(
                        original_prompt=prompt,
                        previous_response=None,
                        error_message=error_msg,
                        attempt_number=attempt + 1,
                    )
                else:
                    self.failed_validations += 1
                    raise ValidationError(
                        f"Validation failed with unexpected error after {self.max_retries + 1} attempts",
                        original_error=e,
                        validation_errors=validation_errors,
                        attempts=self.max_retries + 1,
                    )

        # Should never reach here, but adding for completeness
        self.failed_validations += 1
        raise ValidationError(
            "Validation loop completed without success or proper error handling",
            validation_errors=validation_errors,
            attempts=self.max_retries + 1,
        )

    def _run_business_rules(
        self,
        response: T,
        validation_rules: list[Callable[[T], tuple[bool, str]]],
    ) -> tuple[bool, str]:
        """
        Run custom business rule validation on the response.

        Args:
            response: The parsed Pydantic model instance
            validation_rules: List of validation functions

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not validation_rules:
            return True, ""

        for i, rule in enumerate(validation_rules):
            try:
                is_valid, error_message = rule(response)
                if not is_valid:
                    logger.debug(f"Business rule {i + 1} failed: {error_message}")
                    return False, error_message
            except Exception as e:
                error_msg = f"Business rule {i + 1} raised exception: {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                return False, error_msg

        return True, ""

    def _construct_error_feedback(
        self,
        original_prompt: str,
        previous_response: BaseModel | None,
        error_message: str,
        attempt_number: int,
    ) -> str:
        """
        Construct a new prompt with error feedback for retry.

        This creates a prompt that includes:
        - The original task
        - What went wrong in the previous attempt
        - Specific guidance on how to fix the error

        Args:
            original_prompt: The original prompt
            previous_response: The previous (invalid) response, if available
            error_message: Description of what went wrong
            attempt_number: Current attempt number

        Returns:
            Enhanced prompt with error feedback
        """
        feedback_prompt = f"""{original_prompt}

---
IMPORTANT: Your previous attempt (attempt {attempt_number}) was invalid.

Error: {error_message}
"""

        if previous_response is not None:
            try:
                previous_json = previous_response.model_dump_json(indent=2)
                feedback_prompt += f"""
Previous Response:
{previous_json}
"""
            except Exception:
                feedback_prompt += "\nPrevious response could not be serialized.\n"

        feedback_prompt += """
Please provide a corrected response that addresses the error above.
Ensure all required fields are present and valid according to the schema.
"""

        return feedback_prompt

    def _format_pydantic_error(self, error: PydanticValidationError) -> str:
        """
        Format Pydantic validation error into a clear message for LLM feedback.

        Args:
            error: Pydantic ValidationError

        Returns:
            Formatted error message
        """
        error_messages = []
        for err in error.errors():
            field_path = " -> ".join(str(loc) for loc in err["loc"])
            error_type = err["type"]
            message = err["msg"]
            error_messages.append(f"Field '{field_path}': {message} (type: {error_type})")

        return "Type validation errors:\n" + "\n".join(error_messages)

    def _update_retry_histogram(self, attempts: int) -> None:
        """
        Update statistics on how many retries were needed.

        Args:
            attempts: Number of attempts needed (0-indexed, so 0 = first try)
        """
        self.retry_count_histogram[attempts] = (
            self.retry_count_histogram.get(attempts, 0) + 1
        )

    def get_statistics(self) -> dict[str, Any]:
        """
        Get validation statistics.

        Returns:
            Dictionary with validation metrics including:
            - total_attempts: Total validation attempts
            - successful_validations: Number of successful validations
            - failed_validations: Number of failed validations
            - success_rate: Percentage of successful validations
            - avg_retries: Average number of retries needed
            - retry_histogram: Distribution of retry counts
        """
        total = self.successful_validations + self.failed_validations
        success_rate = (
            (self.successful_validations / total * 100) if total > 0 else 0.0
        )

        # Calculate average retries
        total_retries = sum(
            attempts * count for attempts, count in self.retry_count_histogram.items()
        )
        total_successes = sum(self.retry_count_histogram.values())
        avg_retries = total_retries / total_successes if total_successes > 0 else 0.0

        return {
            "total_attempts": self.total_attempts,
            "successful_validations": self.successful_validations,
            "failed_validations": self.failed_validations,
            "success_rate": success_rate,
            "average_retries": avg_retries,
            "retry_histogram": dict(self.retry_count_histogram),
            "first_attempt_success_rate": (
                (
                    self.retry_count_histogram.get(0, 0)
                    / total_successes
                    * 100
                )
                if total_successes > 0
                else 0.0
            ),
        }

    def reset_statistics(self) -> None:
        """Reset all validation statistics."""
        self.total_attempts = 0
        self.successful_validations = 0
        self.failed_validations = 0
        self.retry_count_histogram.clear()
        logger.info("Validation statistics reset")


# Common validation rule builders
def create_field_value_validator(
    field_name: str, allowed_values: set[str]
) -> Callable[[BaseModel], tuple[bool, str]]:
    """
    Create a validation rule that checks if a field's value is in allowed set.

    Args:
        field_name: Name of the field to validate
        allowed_values: Set of allowed values

    Returns:
        Validation function

    Example:
        >>> validate_label = create_field_value_validator(
        ...     "label", {"positive", "negative", "neutral"}
        ... )
        >>> validator = StructuredOutputValidator(client)
        >>> result = validator.validate_and_retry(
        ...     prompt, LabelResponse, validation_rules=[validate_label]
        ... )
    """

    def validator(response: BaseModel) -> tuple[bool, str]:
        if not hasattr(response, field_name):
            return False, f"Response missing required field: {field_name}"

        value = getattr(response, field_name)
        if value not in allowed_values:
            return (
                False,
                f"Field '{field_name}' has invalid value '{value}'. "
                f"Must be one of: {', '.join(sorted(allowed_values))}",
            )
        return True, ""

    return validator


def create_confidence_validator(
    min_confidence: float = 0.0, max_confidence: float = 1.0
) -> Callable[[BaseModel], tuple[bool, str]]:
    """
    Create a validation rule that checks if confidence is in valid range.

    Args:
        min_confidence: Minimum allowed confidence (default: 0.0)
        max_confidence: Maximum allowed confidence (default: 1.0)

    Returns:
        Validation function
    """

    def validator(response: BaseModel) -> tuple[bool, str]:
        if not hasattr(response, "confidence"):
            return True, ""  # Optional field

        confidence = getattr(response, "confidence")
        if not isinstance(confidence, (int, float)):
            return False, f"Confidence must be numeric, got {type(confidence).__name__}"

        if not (min_confidence <= confidence <= max_confidence):
            return (
                False,
                f"Confidence {confidence} out of valid range [{min_confidence}, {max_confidence}]",
            )
        return True, ""

    return validator


def create_non_empty_validator(
    field_name: str,
) -> Callable[[BaseModel], tuple[bool, str]]:
    """
    Create a validation rule that checks if a string field is non-empty.

    Args:
        field_name: Name of the field to validate

    Returns:
        Validation function
    """

    def validator(response: BaseModel) -> tuple[bool, str]:
        if not hasattr(response, field_name):
            return False, f"Response missing required field: {field_name}"

        value = getattr(response, field_name)
        if not value or (isinstance(value, str) and not value.strip()):
            return False, f"Field '{field_name}' cannot be empty"
        return True, ""

    return validator
