# Structured Output Validation Guide

## Overview

The Structured Output Validator provides automatic validation and retry mechanisms for LLM-based structured outputs. It implements multi-layer validation with error feedback to improve the reliability of LLM responses.

## Features

- **Automatic Retry**: Retries LLM calls when validation fails (configurable 3-5 attempts)
- **Error Feedback**: Provides structured error feedback to the LLM for self-correction
- **Multi-Layer Validation**:
  - Type validation (automatic via Pydantic)
  - Business rule validation (custom validation functions)
  - Semantic validation (optional domain-specific checks)
- **Statistics Tracking**: Detailed metrics on validation performance
- **Fallback Strategies**: Configurable fallback behavior for persistent failures

## Quick Start

### Basic Usage

```python
from autolabeler import Settings, LabelingService
from autolabeler.core.configs import LabelingConfig

# Initialize service
settings = Settings(openrouter_api_key="your-key")
labeler = LabelingService("my_dataset", settings)

# Configure with validation
config = LabelingConfig(
    use_validation=True,
    validation_max_retries=3,
    allowed_labels=["positive", "negative", "neutral"]
)

# Label with automatic validation
result = labeler.label_text(
    text="This is a great product!",
    config=config
)

print(f"Label: {result.label}, Confidence: {result.confidence}")
```

### Validation Statistics

```python
# Get validation statistics after processing
stats = labeler.get_validation_stats()

print(f"Success Rate: {stats['overall_success_rate']:.1f}%")
print(f"Total Validations: {stats['total_successful_validations']}")
print(f"Average Retries: {stats['per_validator_stats'][...]['average_retries']:.2f}")
```

## Configuration Options

### LabelingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_validation` | bool | `True` | Enable/disable validation with retry |
| `validation_max_retries` | int | `3` | Maximum retry attempts on validation failure |
| `allowed_labels` | list[str] \| None | `None` | Whitelist of valid labels (optional) |

### Example Configurations

#### Strict Validation

```python
config = LabelingConfig(
    use_validation=True,
    validation_max_retries=5,  # More retries for strict validation
    allowed_labels=["spam", "ham"],  # Only these labels allowed
    confidence_threshold=0.8,
)
```

#### Lenient Validation

```python
config = LabelingConfig(
    use_validation=True,
    validation_max_retries=1,  # Minimal retries
    allowed_labels=None,  # Accept any label
)
```

#### Disabled Validation (Legacy Mode)

```python
config = LabelingConfig(
    use_validation=False,  # No validation, direct LLM output
)
```

## Advanced Usage

### Custom Validation Rules

You can define custom business rules beyond label validation:

```python
from autolabeler.core.validation import StructuredOutputValidator
from autolabeler.models import LabelResponse

# Define custom validation rule
def validate_high_confidence(response: LabelResponse) -> tuple[bool, str]:
    """Ensure confidence is above threshold."""
    if response.confidence >= 0.7:
        return True, ""
    return False, f"Confidence {response.confidence} is below 0.7 threshold"

# Create validator with custom rules
validator = StructuredOutputValidator(
    client=llm_client,
    max_retries=3
)

result = validator.validate_and_retry(
    prompt="Label this: 'Great product!'",
    response_model=LabelResponse,
    validation_rules=[validate_high_confidence]
)
```

### Built-in Validation Rule Builders

The module provides helper functions to create common validation rules:

```python
from autolabeler.core.validation import (
    create_field_value_validator,
    create_confidence_validator,
    create_non_empty_validator,
)

# Validate label is in allowed set
label_validator = create_field_value_validator(
    "label",
    {"positive", "negative", "neutral"}
)

# Validate confidence is in range
confidence_validator = create_confidence_validator(
    min_confidence=0.5,
    max_confidence=1.0
)

# Validate reasoning is not empty
reasoning_validator = create_non_empty_validator("reasoning")

# Use multiple validators
result = validator.validate_and_retry(
    prompt=prompt,
    response_model=LabelResponse,
    validation_rules=[
        label_validator,
        confidence_validator,
        reasoning_validator,
    ]
)
```

## Validation Workflow

### How It Works

1. **Initial Attempt**: LLM generates structured output
2. **Type Validation**: Pydantic validates field types automatically
3. **Business Rule Validation**: Custom rules are applied
4. **On Failure**:
   - Error feedback is constructed with specifics
   - Prompt is augmented with error context
   - LLM is given another chance (up to max_retries)
5. **On Success**: Validated response is returned

### Error Feedback Example

When validation fails, the LLM receives context like:

```
Original prompt...

---
IMPORTANT: Your previous attempt (attempt 1) was invalid.

Error: Field 'label' has invalid value 'very_positive'. Must be one of: negative, neutral, positive

Previous Response:
{
  "label": "very_positive",
  "confidence": 0.9
}

Please provide a corrected response that addresses the error above.
Ensure all required fields are present and valid according to the schema.
```

## Performance Metrics

### Success Metrics

After implementation, you should see:

- **Parsing failure rate reduced by 90%+**: Type validation catches malformed outputs
- **First-attempt success rate >85%**: Most outputs are valid on first try
- **Average retry count <1.2**: Minimal overhead from retries

### Monitoring Validation Performance

```python
# Get detailed statistics
stats = labeler.get_validation_stats()

# Key metrics to monitor
print(f"First Attempt Success: {stats['first_attempt_success_rate']:.1f}%")
print(f"Overall Success Rate: {stats['overall_success_rate']:.1f}%")
print(f"Average Retries: {stats['average_retries']:.2f}")

# Retry distribution
print("\nRetry Distribution:")
for attempts, count in stats['retry_histogram'].items():
    print(f"  {attempts} retries: {count} times")
```

## Best Practices

### 1. Use Validation by Default

Enable validation for all production workloads:

```python
config = LabelingConfig(use_validation=True)  # Default setting
```

### 2. Set Appropriate Retry Limits

- **3 retries** (default): Good balance for most use cases
- **5 retries**: For critical applications where failures are costly
- **1 retry**: For fast feedback during development

### 3. Define Allowed Labels

When labels are known in advance, always specify them:

```python
config = LabelingConfig(
    allowed_labels=["bug", "feature", "question", "documentation"]
)
```

This prevents:
- Typos in labels ("bgu" instead of "bug")
- Unexpected label variations ("bug_report" when only "bug" expected)
- Hallucinated labels not in your schema

### 4. Monitor Validation Statistics

Regularly check validation stats to identify issues:

```python
stats = labeler.get_validation_stats()

# Alert if success rate drops below threshold
if stats['overall_success_rate'] < 80:
    print("WARNING: Validation success rate is low!")
    print("Consider reviewing prompts or model selection")
```

### 5. Balance Retries vs. Cost

Each retry adds latency and cost. Monitor the trade-off:

```python
# Check average retries
avg_retries = stats['average_retries']

if avg_retries > 1.5:
    print("High retry rate detected!")
    print("Consider improving prompts or validation rules")
```

## Troubleshooting

### High Retry Rate

**Symptoms**: Average retries > 1.5, low first-attempt success rate

**Solutions**:
1. Review validation rules - are they too strict?
2. Improve prompt clarity
3. Check if model is appropriate for task
4. Reduce temperature for more consistent outputs

### Validation Always Fails

**Symptoms**: Failed validations even after max retries

**Solutions**:
1. Check if allowed_labels matches your actual label space
2. Verify custom validation rules are correct
3. Test with simpler prompts to isolate issue
4. Review error messages in logs

### Caching Issues

**Symptoms**: Configuration changes not taking effect

**Solution**: Validators are cached per configuration. Changing config creates new validator automatically.

## Examples

### Example 1: Sentiment Analysis with Strict Validation

```python
from autolabeler import Settings, LabelingService
from autolabeler.core.configs import LabelingConfig
import pandas as pd

settings = Settings(openrouter_api_key="your-key")
labeler = LabelingService("sentiment", settings)

config = LabelingConfig(
    use_validation=True,
    validation_max_retries=3,
    allowed_labels=["positive", "negative", "neutral"],
    temperature=0.3,  # Lower temperature for consistency
)

# Label dataset
df = pd.DataFrame({
    "text": [
        "This product is amazing!",
        "Terrible quality, don't buy",
        "It's okay, nothing special"
    ]
})

results = labeler.label_dataframe(
    df=df,
    text_column="text",
    config=config
)

print(results[["text", "predicted_label", "predicted_label_confidence"]])
```

### Example 2: Custom Multi-Field Validation

```python
from autolabeler.core.validation import StructuredOutputValidator
from autolabeler.models import LabelResponse

def validate_complete_response(response: LabelResponse) -> tuple[bool, str]:
    """Ensure response has reasoning when confidence is low."""
    if response.confidence < 0.8 and not response.reasoning:
        return False, "Low confidence predictions must include reasoning"
    return True, ""

validator = StructuredOutputValidator(client, max_retries=3)

result = validator.validate_and_retry(
    prompt=prompt,
    response_model=LabelResponse,
    validation_rules=[
        create_field_value_validator("label", {"spam", "ham"}),
        validate_complete_response,
    ]
)
```

### Example 3: Batch Processing with Validation Stats

```python
from autolabeler import Settings, LabelingService
from autolabeler.core.configs import LabelingConfig, BatchConfig

settings = Settings(openrouter_api_key="your-key")
labeler = LabelingService("large_dataset", settings)

labeling_config = LabelingConfig(
    use_validation=True,
    validation_max_retries=3,
    allowed_labels=["topic_a", "topic_b", "topic_c"],
)

batch_config = BatchConfig(
    batch_size=50,
    resume=True,
)

# Process large dataset
results = labeler.label_dataframe(
    df=large_df,
    text_column="text",
    config=labeling_config,
    batch_config=batch_config,
)

# Review validation performance
stats = labeler.get_validation_stats()
print(f"\nValidation Performance:")
print(f"  Success Rate: {stats['overall_success_rate']:.1f}%")
print(f"  Total Validated: {stats['total_successful_validations']}")
print(f"  Avg Retries: {stats['per_validator_stats'][next(iter(stats['per_validator_stats']))]['average_retries']:.2f}")
```

## API Reference

### StructuredOutputValidator

```python
class StructuredOutputValidator:
    def __init__(
        self,
        client: BaseChatModel,
        max_retries: int = 3,
        enable_fallback: bool = True,
    ):
        """Initialize the validator."""

    def validate_and_retry(
        self,
        prompt: str,
        response_model: Type[T],
        validation_rules: list[Callable[[T], tuple[bool, str]]] | None = None,
        method: str = "function_calling",
    ) -> T:
        """Validate LLM output with automatic retry."""

    def get_statistics(self) -> dict[str, Any]:
        """Get validation statistics."""

    def reset_statistics(self) -> None:
        """Reset all validation statistics."""
```

### Validation Rule Builders

```python
def create_field_value_validator(
    field_name: str,
    allowed_values: set[str]
) -> Callable[[BaseModel], tuple[bool, str]]:
    """Create validator for field with allowed values."""

def create_confidence_validator(
    min_confidence: float = 0.0,
    max_confidence: float = 1.0
) -> Callable[[BaseModel], tuple[bool, str]]:
    """Create validator for confidence range."""

def create_non_empty_validator(
    field_name: str,
) -> Callable[[BaseModel], tuple[bool, str]]:
    """Create validator for non-empty string fields."""
```

## Migration Guide

### From Legacy (No Validation)

```python
# OLD: No validation
config = LabelingConfig()
result = labeler.label_text(text, config=config)

# NEW: With validation (default)
config = LabelingConfig(
    use_validation=True,  # Now default
    allowed_labels=["positive", "negative", "neutral"]
)
result = labeler.label_text(text, config=config)
```

### Maintaining Backward Compatibility

```python
# Explicitly disable validation to maintain old behavior
config = LabelingConfig(use_validation=False)
result = labeler.label_text(text, config=config)
```

## Contributing

To extend the validation system:

1. Add new validation rule builders in `output_validator.py`
2. Add tests in `tests/test_unit/core/validation/test_output_validator.py`
3. Update this documentation with examples

## Support

For issues or questions:
- Review logs for detailed error messages
- Check validation statistics for patterns
- Refer to examples above for common use cases
