#!/usr/bin/env python3
"""
Multi-Model Ensemble Labeling Example

This example demonstrates the advanced ensemble labeling system with:
- Multiple model configurations with different parameters
- Systematic experimentation with temperature and seed variations
- Multiple ensemble consolidation methods (majority vote, confidence weighted, etc.)
- Performance comparison across models
- Complete tracking of model runs and results
"""

from pathlib import Path
import pandas as pd
from loguru import logger

from autolabeler.config import Settings
from autolabeler.ensemble import EnsembleLabeler
from autolabeler.model_config import ModelConfig, EnsembleMethod


def main():
    """Demonstrate comprehensive ensemble labeling workflow."""

    # Initialize settings
    settings = Settings()
    logger.info("Starting multi-model ensemble labeling example")

    # Create EnsembleLabeler for sentiment analysis
    ensemble = EnsembleLabeler("sentiment_ensemble_demo", settings)

    # Step 1: Create multiple model configurations
    logger.info("Step 1: Creating multiple model configurations")

    # Conservative model (low temperature)
    conservative_config = ModelConfig(
        model_name="gpt-3.5-turbo",
        provider="openrouter",
        temperature=0.1,
        seed=42,
        description="Conservative model with low temperature",
        tags=["conservative", "low_temp"]
    )
    conservative_id = ensemble.add_model_config(conservative_config)

    # Creative model (high temperature)
    creative_config = ModelConfig(
        model_name="gpt-3.5-turbo",
        provider="openrouter",
        temperature=0.7,
        seed=42,
        description="Creative model with high temperature",
        tags=["creative", "high_temp"]
    )
    creative_id = ensemble.add_model_config(creative_config)

    # Balanced model (medium temperature)
    balanced_config = ModelConfig(
        model_name="gpt-3.5-turbo",
        provider="openrouter",
        temperature=0.3,
        seed=123,
        description="Balanced model with medium temperature",
        tags=["balanced", "medium_temp"]
    )
    balanced_id = ensemble.add_model_config(balanced_config)

    # Step 2: Create systematic variants
    logger.info("Step 2: Creating systematic model variants")
    variant_ids = ensemble.create_model_config_variants(
        base_model="gpt-3.5-turbo",
        provider="openrouter",
        temperature_range=[0.1, 0.5, 0.9],
        seed_range=[42, 123, 456]
    )

    logger.info(f"Created {len(variant_ids)} model variants")

    # Step 3: Prepare test data
    logger.info("Step 3: Preparing test data")
    test_data = pd.DataFrame({
        "review": [
            "This movie was absolutely phenomenal!",
            "Worst film I've ever seen, complete garbage",
            "It was okay, nothing special but not terrible",
            "Outstanding performance by all actors",
            "Boring and predictable storyline",
            "Amazing cinematography and direction",
            "Could not finish it, fell asleep",
            "Decent entertainment for a weekend",
            "Revolutionary filmmaking techniques",
            "Terrible acting and poor script"
        ],
        "true_sentiment": [
            "positive", "negative", "neutral", "positive", "negative",
            "positive", "negative", "neutral", "positive", "negative"
        ]
    })

    # Step 4: Single text ensemble prediction
    logger.info("Step 4: Single text ensemble prediction")
    sample_text = "This movie exceeded all my expectations!"

    # Try different ensemble methods
    ensemble_methods = [
        EnsembleMethod.majority_vote(),
        EnsembleMethod.confidence_weighted(),
        EnsembleMethod.high_agreement()
    ]

    for method in ensemble_methods:
        logger.info(f"\nUsing ensemble method: {method.method_name}")
        result = ensemble.label_text_ensemble(
            sample_text,
            model_ids=[conservative_id, creative_id, balanced_id],
            ensemble_method=method
        )

        logger.info(f"Ensemble prediction: {result.label}")
        logger.info(f"Confidence: {result.confidence:.3f}")
        logger.info(f"Model agreement: {result.model_agreement:.3f}")
        logger.info(f"Models used: {result.num_models_used}")

        if result.individual_predictions:
            logger.info("Individual model predictions:")
            for pred in result.individual_predictions:
                logger.info(f"  {pred['model_name']} (T={ensemble.model_configs[pred['model_id']].temperature}): "
                           f"{pred['label']} (conf: {pred['confidence']:.3f})")

    # Step 5: Full DataFrame ensemble labeling
    logger.info("\nStep 5: Full DataFrame ensemble labeling")

    # Use a subset of models for faster demonstration
    selected_models = [conservative_id, creative_id, balanced_id]

    # Run ensemble with confidence weighting
    ensemble_results = ensemble.label_dataframe_ensemble(
        test_data,
        text_column="review",
        model_ids=selected_models,
        ensemble_method=EnsembleMethod.confidence_weighted(),
        save_individual_results=True
    )

    logger.info("\nEnsemble labeling results:")
    for _, row in ensemble_results.iterrows():
        logger.info(
            f"'{row['review'][:50]}...' -> "
            f"Ensemble: {row['ensemble_label']} (conf: {row['ensemble_confidence']:.3f}, "
            f"agreement: {row['model_agreement']:.3f}) | "
            f"True: {row['true_sentiment']}"
        )

    # Step 6: Performance comparison
    logger.info("\nStep 6: Model performance comparison")
    performance_df = ensemble.compare_model_performance()

    if not performance_df.empty:
        logger.info("\nModel Performance Summary:")
        logger.info(performance_df[['model_name', 'temperature', 'avg_confidence', 'success_rate', 'description']].to_string(index=False))

        # Save performance comparison
        perf_output_path = Path("model_performance_comparison.csv")
        performance_df.to_csv(perf_output_path, index=False)
        logger.info(f"Performance comparison saved to {perf_output_path}")

    # Step 7: Ensemble summary and statistics
    logger.info("\nStep 7: Ensemble summary")
    summary = ensemble.get_ensemble_summary()

    logger.info(f"Dataset: {summary['dataset_name']}")
    logger.info(f"Total model configurations: {summary['num_model_configs']}")
    logger.info(f"Completed runs: {summary['num_completed_runs']}")

    logger.info("\nModel configurations:")
    for model_id, config in summary['model_configs'].items():
        logger.info(f"  {model_id}: {config['model_name']} (T={config['temperature']}) - {config['description']}")

    # Step 8: Compare ensemble methods
    logger.info("\nStep 8: Comparing ensemble methods")

    ensemble_methods_comparison = []

    for method in ensemble_methods:
        method_results = []

        for _, row in test_data.iterrows():
            try:
                result = ensemble.label_text_ensemble(
                    row["review"],
                    model_ids=selected_models,
                    ensemble_method=method
                )

                method_results.append({
                    "method": method.method_name,
                    "text": row["review"],
                    "predicted": result.label,
                    "true": row["true_sentiment"],
                    "confidence": result.confidence,
                    "agreement": result.model_agreement,
                    "correct": result.label == row["true_sentiment"]
                })

            except Exception as e:
                logger.error(f"Error with method {method.method_name}: {e}")

        ensemble_methods_comparison.extend(method_results)

    comparison_df = pd.DataFrame(ensemble_methods_comparison)

    if not comparison_df.empty:
        # Calculate accuracy by method
        method_accuracy = comparison_df.groupby('method').agg({
            'correct': 'mean',
            'confidence': 'mean',
            'agreement': 'mean'
        }).round(3)

        logger.info("\nEnsemble Method Comparison:")
        logger.info(method_accuracy.to_string())

        # Save comparison results
        comparison_output_path = Path("ensemble_methods_comparison.csv")
        comparison_df.to_csv(comparison_output_path, index=False)
        logger.info(f"Ensemble methods comparison saved to {comparison_output_path}")

    # Step 9: Export final ensemble results
    logger.info("\nStep 9: Exporting final results")

    # Create a comprehensive results summary
    final_results = ensemble_results.copy()
    final_results["accuracy"] = (
        final_results["ensemble_label"] == final_results["true_sentiment"]
    ).astype(int)

    final_output_path = Path("final_ensemble_results.csv")
    final_results.to_csv(final_output_path, index=False)
    logger.info(f"Final ensemble results saved to {final_output_path}")

    # Calculate overall accuracy
    overall_accuracy = final_results["accuracy"].mean()
    logger.info(f"Overall ensemble accuracy: {overall_accuracy:.3f}")

    logger.info("\nMulti-model ensemble labeling example completed!")
    logger.info(f"Check the ensemble_results/{ensemble.dataset_name}/ directory for detailed results")


if __name__ == "__main__":
    main()
