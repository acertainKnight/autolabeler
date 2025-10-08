"""
Example demonstrating the Structured Output Validation feature.

This script shows how to use validation with automatic retry to improve
the reliability of LLM-based labeling.
"""

from autolabeler import Settings, LabelingService
from autolabeler.core.configs import LabelingConfig
import pandas as pd


def example_basic_validation():
    """Basic example with validation enabled."""
    print("=" * 60)
    print("Example 1: Basic Validation")
    print("=" * 60)

    # Initialize
    settings = Settings(
        openrouter_api_key="your-api-key-here",
        llm_model="openai/gpt-3.5-turbo"
    )
    labeler = LabelingService("validation_demo", settings)

    # Configure with validation
    config = LabelingConfig(
        use_validation=True,
        validation_max_retries=3,
        allowed_labels=["positive", "negative", "neutral"],
        use_rag=False,
    )

    # Label text
    texts = [
        "This is an amazing product! Highly recommended.",
        "Terrible experience, would not buy again.",
        "It's okay, nothing special.",
    ]

    print("\nLabeling texts with validation enabled:")
    for text in texts:
        result = labeler.label_text(text, config=config)
        print(f"\nText: {text[:50]}...")
        print(f"  Label: {result.label}")
        print(f"  Confidence: {result.confidence:.2f}")
        if result.reasoning:
            print(f"  Reasoning: {result.reasoning[:100]}...")

    # Show statistics
    stats = labeler.get_validation_stats()
    print(f"\n\nValidation Statistics:")
    print(f"  Success Rate: {stats['overall_success_rate']:.1f}%")
    print(f"  Total Validations: {stats['total_successful_validations']}")


def example_batch_with_validation():
    """Example processing a dataframe with validation."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing with Validation")
    print("=" * 60)

    settings = Settings(
        openrouter_api_key="your-api-key-here",
        llm_model="openai/gpt-3.5-turbo"
    )
    labeler = LabelingService("batch_demo", settings)

    # Create sample dataset
    df = pd.DataFrame({
        "review": [
            "Outstanding quality! Will buy again.",
            "Poor packaging, product arrived damaged.",
            "Average product for the price.",
            "Exceeded my expectations!",
            "Not worth the money.",
        ]
    })

    # Configure with strict validation
    config = LabelingConfig(
        use_validation=True,
        validation_max_retries=5,
        allowed_labels=["positive", "negative", "neutral"],
        confidence_threshold=0.8,
        use_rag=False,
    )

    print("\nProcessing batch with validation:")
    results = labeler.label_dataframe(
        df=df,
        text_column="review",
        config=config,
    )

    print("\nResults:")
    print(results[["review", "predicted_label", "predicted_label_confidence"]])

    # Validation stats
    stats = labeler.get_validation_stats()
    print(f"\n\nValidation Performance:")
    print(f"  Success Rate: {stats['overall_success_rate']:.1f}%")
    print(f"  Total Successful: {stats['total_successful_validations']}")
    print(f"  Total Failed: {stats['total_failed_validations']}")


def example_custom_validation():
    """Example with custom validation rules."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Validation Rules")
    print("=" * 60)

    from autolabeler.core.validation import (
        StructuredOutputValidator,
        create_field_value_validator,
        create_confidence_validator,
    )
    from autolabeler.models import LabelResponse

    settings = Settings(
        openrouter_api_key="your-api-key-here",
        llm_model="openai/gpt-3.5-turbo"
    )
    labeler = LabelingService("custom_demo", settings)

    # Get the LLM client
    config = LabelingConfig(use_rag=False)
    client = labeler._get_client_for_config(config)

    # Create validator with custom rules
    validator = StructuredOutputValidator(
        client=client,
        max_retries=3,
    )

    # Define custom validation rules
    label_validator = create_field_value_validator(
        "label",
        {"spam", "ham"}
    )

    confidence_validator = create_confidence_validator(
        min_confidence=0.6,
        max_confidence=1.0
    )

    def require_reasoning_for_low_confidence(
        response: LabelResponse
    ) -> tuple[bool, str]:
        """Custom rule: require reasoning when confidence is low."""
        if response.confidence < 0.8 and not response.reasoning:
            return False, "Reasoning required for confidence below 0.8"
        return True, ""

    # Use validator directly
    prompt = """
    Classify the following email as spam or ham (not spam).

    Email: "Congratulations! You've won $1,000,000! Click here to claim."

    Provide:
    - label: either "spam" or "ham"
    - confidence: your confidence in the prediction (0.0 to 1.0)
    - reasoning: why you made this decision
    """

    print("\nValidating with custom rules:")
    result = validator.validate_and_retry(
        prompt=prompt,
        response_model=LabelResponse,
        validation_rules=[
            label_validator,
            confidence_validator,
            require_reasoning_for_low_confidence,
        ],
        method="function_calling",
    )

    print(f"\n  Label: {result.label}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Reasoning: {result.reasoning}")

    # Show validator statistics
    stats = validator.get_statistics()
    print(f"\n\nValidator Statistics:")
    print(f"  Success Rate: {stats['success_rate']:.1f}%")
    print(f"  Average Retries: {stats['average_retries']:.2f}")
    print(f"  First Attempt Success: {stats['first_attempt_success_rate']:.1f}%")


def example_without_validation():
    """Example showing legacy mode without validation."""
    print("\n" + "=" * 60)
    print("Example 4: Without Validation (Legacy Mode)")
    print("=" * 60)

    settings = Settings(
        openrouter_api_key="your-api-key-here",
        llm_model="openai/gpt-3.5-turbo"
    )
    labeler = LabelingService("no_validation_demo", settings)

    # Disable validation
    config = LabelingConfig(
        use_validation=False,
        use_rag=False,
    )

    print("\nLabeling without validation:")
    result = labeler.label_text(
        text="This is a test message.",
        config=config,
    )

    print(f"  Label: {result.label}")
    print(f"  Confidence: {result.confidence:.2f}")
    print("\nNote: No validation retries occurred")


def example_monitoring_validation():
    """Example showing how to monitor validation performance."""
    print("\n" + "=" * 60)
    print("Example 5: Monitoring Validation Performance")
    print("=" * 60)

    settings = Settings(
        openrouter_api_key="your-api-key-here",
        llm_model="openai/gpt-3.5-turbo"
    )
    labeler = LabelingService("monitoring_demo", settings)

    config = LabelingConfig(
        use_validation=True,
        validation_max_retries=3,
        allowed_labels=["positive", "negative", "neutral"],
        use_rag=False,
    )

    # Process some texts
    texts = [
        "Excellent service!",
        "Disappointing quality",
        "Average experience",
        "Highly recommended",
        "Not satisfied",
    ]

    print("\nProcessing texts and monitoring performance:")
    for i, text in enumerate(texts, 1):
        result = labeler.label_text(text, config=config)

        # Check stats after each call
        stats = labeler.get_validation_stats()
        success_rate = stats['overall_success_rate']

        print(f"\nText {i}: {text[:40]}...")
        print(f"  Label: {result.label}")
        print(f"  Current Success Rate: {success_rate:.1f}%")

        # Alert if success rate drops
        if success_rate < 80 and stats['total_successful_validations'] >= 3:
            print("  ⚠️  WARNING: Success rate below 80%!")

    # Final statistics
    final_stats = labeler.get_validation_stats()
    print("\n\nFinal Validation Statistics:")
    print(f"  Total Attempts: {final_stats['total_validation_attempts']}")
    print(f"  Successful: {final_stats['total_successful_validations']}")
    print(f"  Failed: {final_stats['total_failed_validations']}")
    print(f"  Overall Success Rate: {final_stats['overall_success_rate']:.1f}%")

    # Per-validator breakdown
    if final_stats['per_validator_stats']:
        validator_key = next(iter(final_stats['per_validator_stats']))
        validator_stats = final_stats['per_validator_stats'][validator_key]
        print(f"\n  Average Retries: {validator_stats['average_retries']:.2f}")
        print(f"  First Attempt Success: {validator_stats['first_attempt_success_rate']:.1f}%")
        print(f"\n  Retry Distribution:")
        for attempts, count in validator_stats['retry_histogram'].items():
            print(f"    {attempts} retries: {count} times")


if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("AutoLabeler Structured Output Validation Examples")
    print("=" * 60)
    print("\nThese examples demonstrate the validation feature.")
    print("Make sure to set your API key before running.")
    print("\n")

    # Run examples
    try:
        example_basic_validation()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        example_batch_with_validation()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        example_custom_validation()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        example_without_validation()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    try:
        example_monitoring_validation()
    except Exception as e:
        print(f"Example 5 failed: {e}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
