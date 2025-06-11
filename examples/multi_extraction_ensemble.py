"""
Multi-Field Extraction with Ensemble Modeling

This example demonstrates how to extract multiple fields (speaker, relevance, sentiment)
from headlines using ensemble modeling with multiple LLMs, seeds, and temperatures.

Pipeline: 3 LLMs × 3 seeds × 3 temperatures = 27 model predictions per headline
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Any
import json
from itertools import product

from autolabeler import AutoLabeler
from autolabeler.config import Settings
from autolabeler.ensemble import EnsembleLabeler
from autolabeler.models import MultiFieldLabelResponse


class HeadlineMultiExtractor:
    """
    Complete pipeline for multi-field extraction from headlines with ensemble modeling.

    Extracts speaker, relevance, and sentiment using multiple model configurations
    for robust predictions with uncertainty quantification.
    """

    def __init__(
        self,
        dataset_name: str,
        target_question: str,
        base_settings: Settings
    ) -> None:
        """
        Initialize the multi-extraction pipeline.

        Args:
            dataset_name (str): Name for the dataset/knowledge base.
            target_question (str): The question/guidelines for relevance assessment.
            base_settings (Settings): Base configuration settings.
        """
        self.dataset_name = dataset_name
        self.target_question = target_question
        self.base_settings = base_settings

        # Model configurations: 3 LLMs × 3 seeds × 3 temperatures = 27 models
        self.llm_models = [
            "anthropic/claude-3-sonnet",
            "openai/gpt-4o-mini",
            "google/gemini-pro"
        ]
        self.seeds = [42, 123, 789]
        self.temperatures = [0.1, 0.5, 0.9]

        # Generate all model configurations
        self.model_configs = self._generate_model_configs()

        # Initialize individual labelers for each config
        self.labelers = self._initialize_labelers()

    def _generate_model_configs(self) -> list[dict[str, Any]]:
        """Generate all combinations of LLM models, seeds, and temperatures."""
        configs = []

        for model, seed, temperature in product(self.llm_models, self.seeds, self.temperatures):
            config = {
                "name": f"{model.split('/')[-1]}_s{seed}_t{temperature}",
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "response_model": "MultiFieldLabelResponse",
                "template_path": "src/autolabeler/templates/multi_extraction_prompt.j2",
            }
            configs.append(config)

        return configs

    def _initialize_labelers(self) -> list[AutoLabeler]:
        """Initialize a labeler for each model configuration."""
        labelers = []

        for config in self.model_configs:
            # Create settings for this specific model
            model_settings = Settings(
                openrouter_api_key=self.base_settings.openrouter_api_key,
                llm_model=config["model"],
                temperature=config["temperature"],
                seed=config["seed"],
                max_examples_per_query=self.base_settings.max_examples_per_query,
                similarity_threshold=self.base_settings.similarity_threshold
            )

            # Create labeler with custom template
            labeler = AutoLabeler(
                dataset_name=f"{self.dataset_name}_{config['name']}",
                settings=model_settings,
                template_path=Path(config["template_path"])
            )

            # Override the response model using function_calling method
            labeler.llm = labeler.llm.with_structured_output(schema=MultiFieldLabelResponse, method="function_calling")

            labelers.append(labeler)

        return labelers

    def prepare_training_data(
        self,
        relevance_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        text_column: str = "headline",
        relevance_column: str = "relevance",
        sentiment_column: str = "sentiment"
    ) -> None:
        """
        Add labeled training data for relevance and sentiment to all labelers.

        Args:
            relevance_df (pd.DataFrame): DataFrame with relevance labels.
            sentiment_df (pd.DataFrame): DataFrame with sentiment labels.
            text_column (str): Column name containing headline text.
            relevance_column (str): Column name with relevance labels.
            sentiment_column (str): Column name with sentiment labels.
        """
        # Add relevance training data
        relevance_metadata = {
            "label_type": "relevance",
            "target_question": self.target_question,
            "task": "relevance_assessment"
        }

        for labeler in self.labelers:
            labeler.knowledge_base.add_labeled_data(
                relevance_df,
                text_column,
                relevance_column,
                source="human",
                additional_metadata=relevance_metadata
            )

        # Add sentiment training data
        sentiment_metadata = {
            "label_type": "sentiment",
            "target_question": self.target_question,
            "task": "sentiment_analysis"
        }

        for labeler in self.labelers:
            labeler.knowledge_base.add_labeled_data(
                sentiment_df,
                text_column,
                sentiment_column,
                source="human",
                additional_metadata=sentiment_metadata
            )

        print(f"Added training data to {len(self.labelers)} labelers:")
        print(f"  - Relevance examples: {len(relevance_df)}")
        print(f"  - Sentiment examples: {len(sentiment_df)}")

    def extract_from_headlines(
        self,
        headlines_df: pd.DataFrame,
        headline_column: str = "headline",
        context_columns: list[str] | None = None,
        past_headlines: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Extract speaker, relevance, and sentiment from headlines using all model configurations.

        Args:
            headlines_df (pd.DataFrame): DataFrame containing headlines to analyze.
            headline_column (str): Column name with headline text.
            context_columns (list[str] | None): Additional context columns to include.
            past_headlines (list[str] | None): Past headlines from same speech for context.

        Returns:
            pd.DataFrame: Results with predictions from all models plus ensemble.
        """
        all_results = []

        for idx, row in headlines_df.iterrows():
            headline = row[headline_column]

            # Prepare context for this headline
            context = {
                "target_question": self.target_question,
                "past_headlines": past_headlines or [],
                "headline_index": idx
            }

            # Add any additional context columns
            if context_columns:
                for col in context_columns:
                    if col in row and pd.notna(row[col]):
                        context[f"data_{col}"] = row[col]

            print(f"Processing headline {idx + 1}/{len(headlines_df)}: {headline[:50]}...")

            # Get predictions from all 27 models
            model_predictions = []

            for i, labeler in enumerate(self.labelers):
                config = self.model_configs[i]

                try:
                    result = labeler.label_text_with_context(headline, context)

                    prediction = {
                        "headline_index": idx,
                        "headline": headline,
                        "model_config": config["name"],
                        "model": config["model"],
                        "temperature": config["temperature"],
                        "seed": config["seed"],
                        "speaker": result.speaker,
                        "speaker_confidence": result.speaker_confidence,
                        "relevance": result.relevance,
                        "relevance_confidence": result.relevance_confidence,
                        "sentiment": result.sentiment,
                        "sentiment_confidence": result.sentiment_confidence,
                        "overall_confidence": result.overall_confidence,
                        "reasoning": result.reasoning,
                        "context_influence": result.context_influence,
                    }

                    model_predictions.append(prediction)

                except Exception as e:
                    print(f"  Error with model {config['name']}: {e}")
                    # Add failed prediction
                    prediction = {
                        "headline_index": idx,
                        "headline": headline,
                        "model_config": config["name"],
                        "model": config["model"],
                        "temperature": config["temperature"],
                        "seed": config["seed"],
                        "speaker": None,
                        "speaker_confidence": 0.0,
                        "relevance": None,
                        "relevance_confidence": 0.0,
                        "sentiment": None,
                        "sentiment_confidence": 0.0,
                        "overall_confidence": 0.0,
                        "reasoning": f"Error: {str(e)}",
                        "context_influence": None,
                    }
                    model_predictions.append(prediction)

            # Calculate ensemble predictions
            ensemble_result = self._calculate_ensemble(model_predictions)

            # Add ensemble result
            ensemble_result.update({
                "headline_index": idx,
                "headline": headline,
                "target_question": self.target_question,
                "context_past_headlines": len(past_headlines or []),
                "num_models_succeeded": len([p for p in model_predictions if p["speaker"] is not None]),
                "individual_predictions": json.dumps([
                    {k: v for k, v in pred.items()
                     if k not in ["headline_index", "headline"]}
                    for pred in model_predictions
                ])
            })

            all_results.append(ensemble_result)

        return pd.DataFrame(all_results)

    def _calculate_ensemble(self, predictions: list[dict]) -> dict[str, Any]:
        """Calculate ensemble predictions from individual model results."""
        valid_predictions = [p for p in predictions if p["speaker"] is not None]

        if not valid_predictions:
            return {
                "ensemble_speaker": None,
                "ensemble_speaker_confidence": 0.0,
                "ensemble_relevance": None,
                "ensemble_relevance_confidence": 0.0,
                "ensemble_sentiment": None,
                "ensemble_sentiment_confidence": 0.0,
                "ensemble_overall_confidence": 0.0,
                "ensemble_method": "no_valid_predictions",
                "speaker_uncertainty": 1.0,
                "relevance_uncertainty": 1.0,
                "sentiment_uncertainty": 1.0,
            }

        # Calculate majority vote and confidence for each field
        ensemble_result = {}

        for field in ["speaker", "relevance", "sentiment"]:
            values = [p[field] for p in valid_predictions if p[field] is not None]
            confidences = [p[f"{field}_confidence"] for p in valid_predictions if p[field] is not None]

            if values:
                # Majority vote
                from collections import Counter
                vote_counts = Counter(values)
                majority_value = vote_counts.most_common(1)[0][0]

                # Weighted average confidence for majority vote
                majority_confidences = [
                    confidences[i] for i, val in enumerate(values)
                    if val == majority_value
                ]
                avg_confidence = sum(majority_confidences) / len(majority_confidences)

                # Calculate uncertainty (disagreement)
                total_votes = len(values)
                majority_votes = vote_counts[majority_value]
                uncertainty = 1.0 - (majority_votes / total_votes)

                ensemble_result.update({
                    f"ensemble_{field}": majority_value,
                    f"ensemble_{field}_confidence": avg_confidence,
                    f"{field}_uncertainty": uncertainty,
                })
            else:
                ensemble_result.update({
                    f"ensemble_{field}": None,
                    f"ensemble_{field}_confidence": 0.0,
                    f"{field}_uncertainty": 1.0,
                })

        # Overall ensemble confidence
        field_confidences = [
            ensemble_result.get("ensemble_speaker_confidence", 0.0),
            ensemble_result.get("ensemble_relevance_confidence", 0.0),
            ensemble_result.get("ensemble_sentiment_confidence", 0.0),
        ]
        ensemble_result["ensemble_overall_confidence"] = sum(field_confidences) / len(field_confidences)
        ensemble_result["ensemble_method"] = "majority_vote_with_confidence_weighting"

        return ensemble_result

    def analyze_results(self, results_df: pd.DataFrame) -> dict[str, Any]:
        """Analyze the extraction results and provide comprehensive statistics."""
        analysis = {
            "total_headlines": len(results_df),
            "model_configurations": len(self.model_configs),
            "total_predictions": len(results_df) * len(self.model_configs),
        }

        # Success rates
        analysis["avg_models_succeeded"] = results_df["num_models_succeeded"].mean()
        analysis["success_rate"] = analysis["avg_models_succeeded"] / len(self.model_configs)

        # Confidence statistics
        confidence_fields = [
            "ensemble_speaker_confidence",
            "ensemble_relevance_confidence",
            "ensemble_sentiment_confidence",
            "ensemble_overall_confidence"
        ]

        for field in confidence_fields:
            if field in results_df.columns:
                analysis[f"{field}_mean"] = results_df[field].mean()
                analysis[f"{field}_std"] = results_df[field].std()

        # Uncertainty statistics
        uncertainty_fields = [
            "speaker_uncertainty",
            "relevance_uncertainty",
            "sentiment_uncertainty"
        ]

        for field in uncertainty_fields:
            if field in results_df.columns:
                analysis[f"{field}_mean"] = results_df[field].mean()
                analysis[f"{field}_high_count"] = (results_df[field] > 0.5).sum()

        # Label distributions
        for field in ["ensemble_speaker", "ensemble_relevance", "ensemble_sentiment"]:
            if field in results_df.columns:
                analysis[f"{field}_distribution"] = results_df[field].value_counts().to_dict()

        return analysis


def main_example():
    """Complete example demonstrating the multi-extraction ensemble pipeline."""
    print("=== Multi-Field Extraction with Ensemble Modeling ===")

    # Configuration
    base_settings = Settings(
        openrouter_api_key="your-openrouter-api-key",  # Replace with actual key
        max_examples_per_query=5,
        similarity_threshold=0.8
    )

    target_question = "How does this headline relate to climate change policy and environmental regulations?"

    # Initialize extractor
    extractor = HeadlineMultiExtractor(
        dataset_name="climate_headlines",
        target_question=target_question,
        base_settings=base_settings
    )

    print(f"Initialized extractor with {len(extractor.model_configs)} model configurations")

    # Step 1: Prepare training data
    print("\n1. Adding training data...")

    # Relevance training data
    relevance_data = pd.DataFrame({
        "headline": [
            "Biden announces new climate regulations for power plants",
            "Stock market reaches new highs amid tech rally",
            "EPA proposes stricter emissions standards for vehicles",
            "Celebrity couple announces engagement",
            "Scientists warn of accelerating ice sheet melting"
        ],
        "relevance": [
            "highly_relevant",
            "not_relevant",
            "highly_relevant",
            "not_relevant",
            "moderately_relevant"
        ]
    })

    # Sentiment training data
    sentiment_data = pd.DataFrame({
        "headline": [
            "Environmental groups praise new green energy initiatives",
            "Industry leaders slam costly climate regulations",
            "Bipartisan support grows for carbon pricing legislation",
            "Activists protest slow progress on climate action",
            "Economists predict growth from clean energy transition"
        ],
        "sentiment": [
            "positive",
            "negative",
            "positive",
            "negative",
            "positive"
        ]
    })

    extractor.prepare_training_data(relevance_data, sentiment_data)

    # Step 2: Prepare test headlines with context
    print("\n2. Preparing test headlines...")

    # Past headlines from same speech for context
    past_headlines = [
        "President addresses nation on environmental priorities",
        "Administration unveils comprehensive climate strategy",
        "New federal funding announced for renewable energy projects"
    ]

    # Headlines to analyze
    test_headlines = pd.DataFrame({
        "headline": [
            "President Biden calls for immediate action on carbon emissions",
            "Senator Johnson criticizes green energy subsidies as wasteful",
            "Tech companies pledge net-zero emissions by 2030"
        ],
        "speech_context": ["state_of_union"] * 3,
        "date": ["2024-01-15"] * 3
    })

    # Step 3: Run ensemble extraction
    print(f"\n3. Running ensemble extraction with {len(extractor.model_configs)} models...")
    print(f"This will generate {len(test_headlines) * len(extractor.model_configs)} total predictions...")

    results = extractor.extract_from_headlines(
        test_headlines,
        headline_column="headline",
        context_columns=["speech_context", "date"],
        past_headlines=past_headlines
    )

    # Step 4: Analyze results
    print("\n4. Analyzing results...")

    analysis = extractor.analyze_results(results)
    print(f"Total predictions generated: {analysis['total_predictions']}")
    print(f"Average success rate: {analysis['success_rate']:.1%}")
    print(f"Average ensemble confidence: {analysis.get('ensemble_overall_confidence_mean', 0):.3f}")

    # Step 5: Display results
    print("\n5. Sample Results:")
    display_cols = [
        "headline", "ensemble_speaker", "ensemble_relevance", "ensemble_sentiment",
        "ensemble_overall_confidence", "speaker_uncertainty", "num_models_succeeded"
    ]

    if all(col in results.columns for col in display_cols):
        print(results[display_cols].to_string(index=False))

    # Export results
    output_dir = Path("results/multi_extraction_ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_dir / "ensemble_results.csv", index=False)

    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\nResults exported to: {output_dir}")
    print("\nKey Features Demonstrated:")
    print("✓ Multi-field extraction (speaker, relevance, sentiment)")
    print("✓ 27 model configurations (3 LLMs × 3 seeds × 3 temperatures)")
    print("✓ Context-aware retrieval using past headlines")
    print("✓ Ensemble predictions with uncertainty quantification")
    print("✓ Individual model tracking and analysis")


if __name__ == "__main__":
    main_example()
