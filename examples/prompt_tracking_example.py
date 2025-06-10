#!/usr/bin/env python3
"""
Prompt Tracking and Analytics Example

This example demonstrates the comprehensive prompt tracking system with:
- Automatic storage of all prompts used in labeling
- Detailed analytics on prompt usage and performance
- Export capabilities for prompt history analysis
- Consensus prompt analysis across multiple models
- Prompt diversity analysis for ensemble systems
"""

from pathlib import Path
import pandas as pd
from loguru import logger

from autolabeler import (
    Settings, AutoLabeler, EnsembleLabeler,
    ModelConfig, EnsembleMethod, PromptStore
)


def demonstrate_single_labeler_prompt_tracking():
    """Demonstrate prompt tracking with a single AutoLabeler."""

    logger.info("=== Single Labeler Prompt Tracking ===")

    # Initialize settings and labeler
    settings = Settings()
    labeler = AutoLabeler("prompt_demo", settings)

    # Add some initial training data
    training_data = pd.DataFrame({
        "text": [
            "This product is amazing!",
            "Terrible quality, very disappointed",
            "It's okay, nothing special",
            "Outstanding service and quality",
            "Would not recommend this item"
        ],
        "sentiment": ["positive", "negative", "neutral", "positive", "negative"]
    })

    labeler.add_training_data(training_data, "text", "sentiment")

    # Label some new texts (this will create and track prompts)
    test_texts = [
        "I love this product so much!",
        "This is the worst purchase I've made",
        "The product is fine but nothing extraordinary",
        "Exceptional quality and fast delivery",
        "Complete waste of money",
        "Pretty good overall experience",
        "Not worth the price at all"
    ]

    logger.info("Labeling test texts and tracking prompts...")
    results = []
    for text in test_texts:
        result = labeler.label_text(text, use_rag=True)
        results.append({
            "text": text,
            "label": result.label,
            "confidence": result.confidence
        })
        logger.info(f"'{text[:30]}...' -> {result.label} ({result.confidence:.3f})")

    # Get prompt analytics
    logger.info("\n--- Prompt Analytics ---")
    analytics = labeler.get_prompt_analytics()

    logger.info(f"Total prompts used: {analytics['total_prompts']}")
    logger.info(f"Total prompt usage: {analytics['total_usage']}")
    logger.info(f"Average usage per prompt: {analytics['avg_usage_per_prompt']:.2f}")
    logger.info(f"Overall success rate: {analytics['success_rate']:.2%}")
    logger.info(f"Average confidence: {analytics['avg_confidence']:.3f}")

    # Show template usage
    if analytics['template_usage']:
        logger.info("\nTemplate usage:")
        for template, count in analytics['template_usage'].items():
            logger.info(f"  {template}: {count} uses")

    # Find most successful prompts
    logger.info("\n--- Most Successful Prompts ---")
    successful_prompts = labeler.get_most_successful_prompts(3)

    for i, prompt in enumerate(successful_prompts, 1):
        logger.info(f"{i}. Prompt ID: {prompt.prompt_id}")
        logger.info(f"   Usage: {prompt.usage_count} times")
        logger.info(f"   Success: {prompt.successful_predictions}/{prompt.successful_predictions + prompt.failed_predictions}")
        logger.info(f"   Avg confidence: {prompt.avg_confidence:.3f}")
        logger.info(f"   Examples used: {prompt.num_examples}")

    # Export prompt history
    logger.info("\n--- Exporting Prompt History ---")
    export_path = Path("single_labeler_prompt_history.csv")
    labeler.export_prompt_history(export_path)
    logger.info(f"Prompt history exported to {export_path}")

    # Show sample of exported data
    if export_path.exists():
        df = pd.read_csv(export_path)
        logger.info(f"Exported {len(df)} prompt records")
        logger.info("Sample columns: " + ", ".join(df.columns[:8]))

    return labeler


def demonstrate_ensemble_prompt_tracking():
    """Demonstrate prompt tracking across multiple models in an ensemble."""

    logger.info("\n=== Ensemble Prompt Tracking ===")

    # Initialize ensemble with multiple model configurations
    settings = Settings()
    ensemble = EnsembleLabeler("prompt_ensemble_demo", settings)

    # Create multiple model configurations
    configs = [
        ModelConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            description="Conservative model",
            tags=["conservative", "low_temp"]
        ),
        ModelConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            description="Balanced model",
            tags=["balanced", "medium_temp"]
        ),
        ModelConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.9,
            description="Creative model",
            tags=["creative", "high_temp"]
        )
    ]

    model_ids = []
    for config in configs:
        model_id = ensemble.add_model_config(config)
        model_ids.append(model_id)
        logger.info(f"Added model {model_id}: {config.description}")

    # Prepare test data
    test_data = pd.DataFrame({
        "review": [
            "This movie was absolutely fantastic!",
            "Worst film I've ever seen",
            "It was okay, nothing special",
            "Amazing cinematography and acting",
            "Boring and predictable plot",
            "Incredible story and direction"
        ]
    })

    # Run ensemble labeling (this creates prompts for each model)
    logger.info("\nRunning ensemble labeling...")
    ensemble_results = ensemble.label_dataframe_ensemble(
        test_data,
        text_column="review",
        model_ids=model_ids,
        ensemble_method=EnsembleMethod.confidence_weighted()
    )

    logger.info("Ensemble labeling completed")

    # Analyze prompt usage across the ensemble
    logger.info("\n--- Ensemble Prompt Analytics ---")
    ensemble_analytics = ensemble.get_ensemble_prompt_analytics()

    summary = ensemble_analytics['summary']
    logger.info(f"Total models: {summary['total_models']}")
    logger.info(f"Total prompts across all models: {summary['total_prompts_across_models']}")
    logger.info(f"Total usage across all models: {summary['total_usage_across_models']}")
    logger.info(f"Average prompts per model: {summary['avg_prompts_per_model']:.1f}")

    # Show individual model analytics
    logger.info("\n--- Individual Model Analytics ---")
    for model_id, model_data in ensemble_analytics['individual_models'].items():
        analytics = model_data['analytics']
        logger.info(f"\nModel {model_id} ({model_data['model_name']} T={model_data['temperature']}):")
        logger.info(f"  Prompts used: {analytics.get('total_prompts', 0)}")
        logger.info(f"  Total usage: {analytics.get('total_usage', 0)}")
        logger.info(f"  Success rate: {analytics.get('success_rate', 0):.2%}")

    # Analyze prompt diversity
    logger.info("\n--- Prompt Diversity Analysis ---")
    diversity = ensemble.analyze_prompt_diversity()

    if "error" not in diversity:
        logger.info(f"Total unique prompts: {diversity['total_unique_prompts']}")
        logger.info(f"Average Jaccard similarity: {diversity['avg_jaccard_similarity']:.3f}")

        logger.info("\nPrompts per model:")
        for model_id, count in diversity['prompts_per_model'].items():
            config = ensemble.model_configs[model_id]
            logger.info(f"  {model_id} (T={config.temperature}): {count} prompts")

        logger.info("\nPairwise overlaps:")
        for pair, overlap_data in diversity['pairwise_overlaps'].items():
            logger.info(f"  {pair}: {overlap_data['overlap_count']} overlapping prompts "
                       f"(Jaccard: {overlap_data['jaccard_similarity']:.3f})")

    # Find consensus prompts (used by multiple models)
    logger.info("\n--- Consensus Prompts ---")
    consensus_prompts = ensemble.find_consensus_prompts(min_models=2)

    logger.info(f"Found {len(consensus_prompts)} consensus prompts")
    for i, prompt_data in enumerate(consensus_prompts[:3], 1):  # Show top 3
        logger.info(f"\n{i}. Consensus Prompt (used by {prompt_data['num_models_used']} models):")
        logger.info(f"   Hash: {prompt_data['prompt_hash']}")
        logger.info(f"   Total usage: {prompt_data['total_usage']}")
        logger.info(f"   Avg success rate: {prompt_data['avg_success_rate']:.2%}")
        logger.info(f"   Models that used it:")
        for usage in prompt_data['model_usages']:
            logger.info(f"     - {usage['model_name']} T={usage['temperature']}: "
                       f"{usage['usage_count']} times, {usage['success_rate']:.2%} success")

    # Export all prompt histories
    logger.info("\n--- Exporting All Prompt Histories ---")
    export_dir = Path("ensemble_prompt_histories")
    ensemble.export_all_prompt_histories(export_dir)

    # List exported files
    if export_dir.exists():
        files = list(export_dir.glob("*.csv"))
        logger.info(f"Exported {len(files)} prompt history files:")
        for file in files:
            df = pd.read_csv(file)
            logger.info(f"  {file.name}: {len(df)} prompt records")

    return ensemble


def demonstrate_prompt_store_direct_usage():
    """Demonstrate direct usage of PromptStore for advanced analytics."""

    logger.info("\n=== Direct PromptStore Usage ===")

    # Create a prompt store directly
    store = PromptStore("direct_demo")

    # Manually store some example prompts
    example_prompts = [
        {
            "text": "You are a sentiment classifier. Classify this text: 'Great product!'",
            "template": "sentiment_v1.j2",
            "variables": {"text": "Great product!"},
            "model": "gpt-3.5-turbo",
            "tags": ["sentiment", "v1"]
        },
        {
            "text": "Analyze the sentiment of: 'Terrible service'",
            "template": "sentiment_v2.j2",
            "variables": {"text": "Terrible service"},
            "model": "gpt-3.5-turbo",
            "tags": ["sentiment", "v2"]
        },
        {
            "text": "You are a sentiment classifier. Classify this text: 'Amazing quality!'",
            "template": "sentiment_v1.j2",
            "variables": {"text": "Amazing quality!"},
            "model": "gpt-4",
            "tags": ["sentiment", "v1"]
        }
    ]

    logger.info("Storing example prompts...")
    for prompt_data in example_prompts:
        prompt_id = store.store_prompt(
            prompt_text=prompt_data["text"],
            template_source=prompt_data["template"],
            variables=prompt_data["variables"],
            model_name=prompt_data["model"],
            tags=prompt_data["tags"]
        )

        # Simulate some usage with random success/failure
        import random
        for _ in range(random.randint(1, 5)):
            success = random.choice([True, True, True, False])  # 75% success rate
            confidence = random.uniform(0.6, 0.95) if success else random.uniform(0.3, 0.7)
            store.update_prompt_result(prompt_id, success, confidence)

        logger.info(f"Stored prompt {prompt_id}: {prompt_data['text'][:50]}...")

    # Get analytics
    logger.info("\n--- Direct Store Analytics ---")
    analytics = store.get_prompt_analytics()

    logger.info(f"Total prompts: {analytics['total_prompts']}")
    logger.info(f"Total usage: {analytics['total_usage']}")
    logger.info(f"Success rate: {analytics['success_rate']:.2%}")

    logger.info("\nTemplate usage:")
    for template, count in analytics['template_usage'].items():
        logger.info(f"  {template}: {count} uses")

    logger.info("\nModel usage:")
    for model, count in analytics['model_usage'].items():
        logger.info(f"  {model}: {count} uses")

    # Find prompts by criteria
    logger.info("\n--- Finding Prompts by Criteria ---")

    v1_prompts = store.find_similar_prompts(template_source="sentiment_v1.j2")
    logger.info(f"Found {len(v1_prompts)} prompts using sentiment_v1.j2 template")

    gpt4_prompts = store.find_similar_prompts(model_name="gpt-4")
    logger.info(f"Found {len(gpt4_prompts)} prompts using gpt-4 model")

    # Export for analysis
    export_path = Path("direct_store_prompts.csv")
    store.export_prompts(export_path)
    logger.info(f"Exported prompt store to {export_path}")

    return store


def main():
    """Run all prompt tracking demonstrations."""

    logger.info("Starting comprehensive prompt tracking demonstration")

    # Single labeler demonstration
    single_labeler = demonstrate_single_labeler_prompt_tracking()

    # Ensemble demonstration
    ensemble = demonstrate_ensemble_prompt_tracking()

    # Direct PromptStore usage
    direct_store = demonstrate_prompt_store_direct_usage()

    # Summary
    logger.info("\n=== Summary ===")
    logger.info("Prompt tracking provides:")
    logger.info("✓ Complete audit trail of all prompts used")
    logger.info("✓ Performance analytics per prompt")
    logger.info("✓ Template and model usage statistics")
    logger.info("✓ Consensus prompt identification")
    logger.info("✓ Prompt diversity analysis")
    logger.info("✓ Export capabilities for further analysis")

    logger.info("\nPrompt tracking files created:")
    files_created = [
        "single_labeler_prompt_history.csv",
        "ensemble_prompt_histories/",
        "direct_store_prompts.csv"
    ]

    for file_path in files_created:
        path = Path(file_path)
        if path.exists():
            if path.is_file():
                logger.info(f"✓ {file_path}")
            else:
                files = list(path.glob("*.csv"))
                logger.info(f"✓ {file_path} ({len(files)} files)")

    logger.info("\nPrompt tracking demonstration completed!")


if __name__ == "__main__":
    main()
