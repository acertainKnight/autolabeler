from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from .config import Settings
from .labeler import AutoLabeler
from .model_config import EnsembleMethod, ModelConfig, ModelRun
from .models import LabelResponse
from .prompt_store import PromptStore


class EnsembleResult(LabelResponse):
    """
    Enhanced result from ensemble prediction with additional metadata.

    Extends LabelResponse to include information about the ensemble
    process, model agreement, and individual model predictions.
    """

    ensemble_method: str | None = None
    model_agreement: float | None = None
    num_models_used: int | None = None
    individual_predictions: list[dict[str, Any]] | None = None
    disagreement_labels: list[str] | None = None


class EnsembleLabeler:
    """
    Multi-model ensemble labeling system with consolidation methods.

    Manages multiple AutoLabeler instances with different model configurations,
    runs parallel labeling experiments, and consolidates results using various
    ensemble methods like majority voting and confidence weighting.

    Args:
        dataset_name (str): Name of the dataset for ensemble labeling.
        settings (Settings): Base settings for the ensemble system.
        ensemble_dir (Path | None): Directory to store ensemble results.

    Example:
        >>> ensemble = EnsembleLabeler("sentiment_analysis", settings)
        >>> # Add multiple model configurations
        >>> ensemble.add_model_config(ModelConfig(
        ...     model_name="gpt-3.5-turbo", temperature=0.1, description="Conservative"
        ... ))
        >>> ensemble.add_model_config(ModelConfig(
        ...     model_name="gpt-3.5-turbo", temperature=0.7, description="Creative"
        ... ))
        >>> # Run ensemble labeling
        >>> results = ensemble.label_dataframe_ensemble(df, "text")
    """

    def __init__(
        self,
        dataset_name: str,
        settings: Settings,
        ensemble_dir: Path | None = None
    ) -> None:
        self.dataset_name = dataset_name
        self.settings = settings
        self.ensemble_dir = ensemble_dir or Path("ensemble_results") / dataset_name
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Storage for model configurations and runs
        self.model_configs: dict[str, ModelConfig] = {}
        self.model_runs: dict[str, ModelRun] = {}
        self.labelers: dict[str, AutoLabeler] = {}

        # Ensemble-level prompt store for tracking ensemble consolidation prompts
        self.ensemble_prompt_store = PromptStore(f"{dataset_name}_ensemble")

        # Load existing configurations if available
        self._load_ensemble_state()

        logger.info(f"Initialized EnsembleLabeler for dataset: {dataset_name}")

    def add_model_config(
        self,
        config: ModelConfig,
        create_labeler: bool = True
    ) -> str:
        """
        Add a model configuration to the ensemble.

        Args:
            config (ModelConfig): Model configuration to add.
            create_labeler (bool): Whether to create AutoLabeler instance immediately.

        Returns:
            str: Model configuration ID.

        Example:
            >>> config = ModelConfig(
            ...     model_name="gpt-4", temperature=0.2, description="High accuracy"
            ... )
            >>> model_id = ensemble.add_model_config(config)
        """
        if not config.model_id:
            config.model_id = config.generate_id()

        self.model_configs[config.model_id] = config

        if create_labeler:
            # Create AutoLabeler with this configuration
            labeler = AutoLabeler(self.dataset_name, self.settings)
            # Store the model config ID for prompt tracking
            labeler.current_model_config_id = config.model_id
            self.labelers[config.model_id] = labeler

        # Save configuration
        self._save_ensemble_state()

        logger.info(f"Added model config: {config}")
        return config.model_id

    def create_model_config_variants(
        self,
        base_model: str,
        provider: str = "openrouter",
        temperature_range: list[float] | None = None,
        seed_range: list[int] | None = None,
    ) -> list[str]:
        """
        Create multiple model configuration variants for systematic experimentation.

        Args:
            base_model (str): Base model name to create variants of.
            provider (str): Model provider.
            temperature_range (list[float] | None): List of temperature values to test.
            seed_range (list[int] | None): List of seed values to test.

        Returns:
            list[str]: List of created model configuration IDs.

        Example:
            >>> model_ids = ensemble.create_model_config_variants(
            ...     "gpt-3.5-turbo",
            ...     temperature_range=[0.1, 0.3, 0.7],
            ...     seed_range=[42, 123, 456]
            ... )
        """
        if temperature_range is None:
            temperature_range = [0.1, 0.3, 0.7]
        if seed_range is None:
            seed_range = [42]

        config_ids = []

        for temp in temperature_range:
            for seed in seed_range:
                config = ModelConfig(
                    model_name=base_model,
                    provider=provider,
                    temperature=temp,
                    seed=seed,
                    description=f"{base_model} T={temp} seed={seed}",
                    tags=[f"temp_{temp}", f"seed_{seed}", "variant"]
                )
                config_id = self.add_model_config(config)
                config_ids.append(config_id)

        logger.info(f"Created {len(config_ids)} model configuration variants")
        return config_ids

    def label_text_ensemble(
        self,
        text: str,
        model_ids: list[str] | None = None,
        ensemble_method: EnsembleMethod | None = None
    ) -> EnsembleResult:
        """
        Label a single text using multiple models and ensemble consolidation.

        Args:
            text (str): Text to label.
            model_ids (list[str] | None): Specific model IDs to use. Uses all if None.
            ensemble_method (EnsembleMethod | None): Ensemble method. Uses majority vote if None.

        Returns:
            EnsembleResult: Consolidated prediction with ensemble metadata.

        Example:
            >>> result = ensemble.label_text_ensemble("This movie was great!")
            >>> print(f"Label: {result.label}, Agreement: {result.model_agreement}")
        """
        if ensemble_method is None:
            ensemble_method = EnsembleMethod.majority_vote()

        model_ids = model_ids or list(self.model_configs.keys())

        if not model_ids:
            raise ValueError("No model configurations available")

        # Get predictions from each model
        individual_predictions = []
        for model_id in model_ids:
            try:
                labeler = self.labelers[model_id]
                config = self.model_configs[model_id]

                result = labeler.label_text(
                    text,
                    use_rag=config.use_rag,
                    k=config.max_examples,
                    prefer_human_examples=config.prefer_human_examples
                )

                individual_predictions.append({
                    "model_id": model_id,
                    "model_name": config.model_name,
                    "label": result.label,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "metadata": result.metadata
                })

            except Exception as e:
                logger.error(f"Failed to get prediction from model {model_id}: {e}")
                continue

        if not individual_predictions:
            raise ValueError("No successful predictions from any model")

        # Consolidate predictions
        ensemble_result = self._consolidate_predictions(
            individual_predictions, ensemble_method
        )

        # Add ensemble metadata
        ensemble_result.ensemble_method = ensemble_method.method_name
        ensemble_result.num_models_used = len(individual_predictions)
        ensemble_result.individual_predictions = individual_predictions

        return ensemble_result

    def label_dataframe_ensemble(
        self,
        df: pd.DataFrame,
        text_column: str,
        model_ids: list[str] | None = None,
        ensemble_method: EnsembleMethod | None = None,
        save_individual_results: bool = True
    ) -> pd.DataFrame:
        """
        Label a DataFrame using ensemble of multiple models.

        Args:
            df (pd.DataFrame): DataFrame containing text to label.
            text_column (str): Name of column containing text.
            model_ids (list[str] | None): Specific model IDs to use.
            ensemble_method (EnsembleMethod | None): Ensemble consolidation method.
            save_individual_results (bool): Whether to save individual model results.

        Returns:
            pd.DataFrame: DataFrame with ensemble predictions and metadata.

        Example:
            >>> results_df = ensemble.label_dataframe_ensemble(df, "review_text")
        """
        if ensemble_method is None:
            ensemble_method = EnsembleMethod.confidence_weighted()

        model_ids = model_ids or list(self.model_configs.keys())

        # First, get predictions from each model individually
        model_results = {}

        for model_id in model_ids:
            logger.info(f"Running predictions with model {model_id}")

            # Create model run record
            run = ModelRun(
                run_id=str(uuid.uuid4()),
                model_config_id=model_id,
                dataset_name=self.dataset_name,
                total_texts=len(df)
            )

            try:
                labeler = self.labelers[model_id]
                config = self.model_configs[model_id]

                # Label the DataFrame
                labeled_df = labeler.label_dataframe(
                    df,
                    text_column,
                    label_column=f"pred_{model_id}",
                    use_rag=config.use_rag,
                    save_to_knowledge_base=False,  # Don't auto-save during ensemble
                    confidence_threshold=config.confidence_threshold
                )

                model_results[model_id] = labeled_df

                # Update run statistics
                successful_preds = labeled_df[f"pred_{model_id}"].notna().sum()
                avg_conf = labeled_df[f"pred_{model_id}_confidence"].mean()

                run.successful_predictions = successful_preds
                run.failed_predictions = len(df) - successful_preds
                run.avg_confidence = avg_conf
                run.mark_completed()

                # Save individual results if requested
                if save_individual_results:
                    output_path = self.ensemble_dir / f"individual_{model_id}.csv"
                    labeled_df.to_csv(output_path, index=False)

            except Exception as e:
                logger.error(f"Failed to run model {model_id}: {e}")
                run.status = "failed"
                run.add_error(str(e))
                run.mark_completed()

            self.model_runs[run.run_id] = run

        # Now consolidate the results
        logger.info("Consolidating ensemble predictions")
        ensemble_results = []

        for idx, row in df.iterrows():
            text = str(row[text_column])

            # Collect predictions for this text from all models
            individual_preds = []
            for model_id in model_ids:
                if model_id in model_results:
                    model_df = model_results[model_id]
                    model_row = model_df.iloc[idx]

                    pred_col = f"pred_{model_id}"
                    conf_col = f"pred_{model_id}_confidence"

                    if pd.notna(model_row[pred_col]):
                        individual_preds.append({
                            "model_id": model_id,
                            "model_name": self.model_configs[model_id].model_name,
                            "label": model_row[pred_col],
                            "confidence": model_row[conf_col],
                            "reasoning": model_row.get(f"pred_{model_id}_reasoning"),
                            "metadata": {}
                        })

            if individual_preds:
                # Consolidate predictions for this text
                ensemble_result = self._consolidate_predictions(
                    individual_preds, ensemble_method
                )

                # Create result row
                result_row = row.to_dict()
                result_row["ensemble_label"] = ensemble_result.label
                result_row["ensemble_confidence"] = ensemble_result.confidence
                result_row["ensemble_method"] = ensemble_method.method_name
                result_row["model_agreement"] = ensemble_result.model_agreement
                result_row["num_models"] = len(individual_preds)
                result_row["individual_predictions"] = json.dumps(individual_preds)

                ensemble_results.append(result_row)
            else:
                # No successful predictions
                result_row = row.to_dict()
                result_row["ensemble_label"] = None
                result_row["ensemble_confidence"] = 0.0
                ensemble_results.append(result_row)

        ensemble_df = pd.DataFrame(ensemble_results)

        # Save ensemble results
        ensemble_output_path = self.ensemble_dir / f"ensemble_{ensemble_method.method_name}.csv"
        ensemble_df.to_csv(ensemble_output_path, index=False)

        # Save run summaries
        self._save_ensemble_state()

        logger.info(f"Ensemble labeling completed. Results saved to {ensemble_output_path}")
        return ensemble_df

    def _consolidate_predictions(
        self,
        predictions: list[dict[str, Any]],
        method: EnsembleMethod
    ) -> EnsembleResult:
        """
        Consolidate multiple predictions using the specified ensemble method.

        Args:
            predictions (list[dict]): List of individual model predictions.
            method (EnsembleMethod): Method to use for consolidation.

        Returns:
            EnsembleResult: Consolidated prediction result.
        """
        if not predictions:
            return EnsembleResult(label="unknown", confidence=0.0)

        # Filter by confidence threshold
        valid_predictions = [
            p for p in predictions
            if p["confidence"] >= method.min_confidence_threshold
        ]

        if not valid_predictions:
            # Fall back to all predictions if none meet threshold
            valid_predictions = predictions

        # Limit number of models if specified
        if method.max_models_to_consider:
            valid_predictions = sorted(
                valid_predictions,
                key=lambda x: x["confidence"],
                reverse=True
            )[:method.max_models_to_consider]

        if method.method_name == "majority_vote":
            return self._majority_vote_consolidation(valid_predictions, method)
        elif method.method_name == "confidence_weighted":
            return self._confidence_weighted_consolidation(valid_predictions, method)
        elif method.method_name == "high_agreement":
            return self._high_agreement_consolidation(valid_predictions, method)
        elif method.method_name == "human_validated":
            return self._human_validated_consolidation(valid_predictions, method)
        else:
            # Default to majority vote
            return self._majority_vote_consolidation(valid_predictions, method)

    def _majority_vote_consolidation(
        self,
        predictions: list[dict[str, Any]],
        method: EnsembleMethod
    ) -> EnsembleResult:
        """Simple majority voting consolidation."""
        labels = [p["label"] for p in predictions]
        label_counts = Counter(labels)

        # Get the most common label
        most_common_label, vote_count = label_counts.most_common(1)[0]

        # Calculate agreement (fraction of models that agree with majority)
        agreement = vote_count / len(predictions)

        # Average confidence of models that voted for the majority label
        majority_confidences = [
            p["confidence"] for p in predictions
            if p["label"] == most_common_label
        ]
        avg_confidence = sum(majority_confidences) / len(majority_confidences)

        return EnsembleResult(
            label=most_common_label,
            confidence=avg_confidence,
            model_agreement=agreement,
            disagreement_labels=list(set(labels) - {most_common_label})
        )

    def _confidence_weighted_consolidation(
        self,
        predictions: list[dict[str, Any]],
        method: EnsembleMethod
    ) -> EnsembleResult:
        """Confidence-weighted consolidation."""
        label_weights = defaultdict(float)
        total_weight = 0.0

        # Weight each label by model confidence
        for pred in predictions:
            weight = pred["confidence"]
            label_weights[pred["label"]] += weight
            total_weight += weight

        if total_weight == 0:
            return EnsembleResult(label="unknown", confidence=0.0)

        # Normalize weights and find best label
        label_probs = {
            label: weight / total_weight
            for label, weight in label_weights.items()
        }

        best_label = max(label_probs, key=label_probs.get)
        best_confidence = label_probs[best_label]

        # Calculate agreement as the probability mass of the winning label
        agreement = best_confidence

        return EnsembleResult(
            label=best_label,
            confidence=best_confidence,
            model_agreement=agreement,
            disagreement_labels=list(set(label_probs.keys()) - {best_label})
        )

    def _high_agreement_consolidation(
        self,
        predictions: list[dict[str, Any]],
        method: EnsembleMethod
    ) -> EnsembleResult:
        """High agreement consolidation - only return prediction if models agree strongly."""
        result = self._confidence_weighted_consolidation(predictions, method)

        # Only return prediction if agreement meets threshold
        if result.model_agreement >= method.min_agreement:
            return result
        else:
            return EnsembleResult(
                label="disagreement",
                confidence=0.0,
                model_agreement=result.model_agreement,
                disagreement_labels=result.disagreement_labels
            )

    def _human_validated_consolidation(
        self,
        predictions: list[dict[str, Any]],
        method: EnsembleMethod
    ) -> EnsembleResult:
        """Human-validated consolidation - placeholder for future human-in-the-loop."""
        # For now, use confidence weighting with higher thresholds
        return self._confidence_weighted_consolidation(predictions, method)

    def get_ensemble_summary(self) -> dict[str, Any]:
        """
        Get summary statistics for the ensemble system.

        Returns:
            dict: Summary including model configs, runs, and performance metrics.
        """
        summary = {
            "dataset_name": self.dataset_name,
            "num_model_configs": len(self.model_configs),
            "num_completed_runs": len([r for r in self.model_runs.values() if r.status == "completed"]),
            "model_configs": {
                mid: {
                    "model_name": config.model_name,
                    "temperature": config.temperature,
                    "description": config.description
                }
                for mid, config in self.model_configs.items()
            },
            "recent_runs": [
                {
                    "run_id": run.run_id,
                    "model_config_id": run.model_config_id,
                    "status": run.status,
                    "successful_predictions": run.successful_predictions,
                    "avg_confidence": run.avg_confidence,
                    "processing_time": run.processing_time_seconds
                }
                for run in sorted(
                    self.model_runs.values(),
                    key=lambda x: x.started_at,
                    reverse=True
                )[:10]  # Most recent 10 runs
            ]
        }
        return summary

    def _save_ensemble_state(self) -> None:
        """Save ensemble state to disk."""
        state = {
            "model_configs": {
                mid: config.to_dict()
                for mid, config in self.model_configs.items()
            },
            "model_runs": {
                rid: run.model_dump()
                for rid, run in self.model_runs.items()
            }
        }

        state_path = self.ensemble_dir / "ensemble_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def _load_ensemble_state(self) -> None:
        """Load ensemble state from disk."""
        state_path = self.ensemble_dir / "ensemble_state.json"

        if not state_path.exists():
            return

        try:
            with open(state_path, 'r') as f:
                state = json.load(f)

            # Load model configurations
            for mid, config_data in state.get("model_configs", {}).items():
                self.model_configs[mid] = ModelConfig.from_dict(config_data)

            # Load model runs
            for rid, run_data in state.get("model_runs", {}).items():
                self.model_runs[rid] = ModelRun(**run_data)

            logger.info(f"Loaded ensemble state: {len(self.model_configs)} configs, {len(self.model_runs)} runs")

        except Exception as e:
            logger.warning(f"Could not load ensemble state: {e}")

    def compare_model_performance(self) -> pd.DataFrame:
        """
        Compare performance across different model configurations.

        Returns:
            pd.DataFrame: Performance comparison table.
        """
        comparison_data = []

        for run in self.model_runs.values():
            if run.status == "completed":
                config = self.model_configs.get(run.model_config_id)
                if config:
                    comparison_data.append({
                        "model_config_id": run.model_config_id,
                        "model_name": config.model_name,
                        "temperature": config.temperature,
                        "seed": config.seed,
                        "avg_confidence": run.avg_confidence,
                        "success_rate": run.successful_predictions / run.total_texts if run.total_texts > 0 else 0,
                        "predictions_per_second": run.predictions_per_second,
                        "total_texts": run.total_texts,
                        "description": config.description
                    })

        return pd.DataFrame(comparison_data)

    def get_ensemble_prompt_analytics(self) -> dict[str, Any]:
        """
        Get analytics about prompts used across all models in the ensemble.

        Returns:
            dict: Combined prompt analytics from all models and ensemble operations.
        """
        # Get analytics from ensemble-level prompt store
        ensemble_analytics = self.ensemble_prompt_store.get_prompt_analytics()

        # Aggregate analytics from individual model labelers
        individual_analytics = {}
        total_prompts = 0
        total_usage = 0

        for model_id, labeler in self.labelers.items():
            model_analytics = labeler.get_prompt_analytics()
            individual_analytics[model_id] = {
                "model_name": self.model_configs[model_id].model_name,
                "temperature": self.model_configs[model_id].temperature,
                "analytics": model_analytics
            }
            total_prompts += model_analytics.get("total_prompts", 0)
            total_usage += model_analytics.get("total_usage", 0)

        combined_analytics = {
            "ensemble_prompts": ensemble_analytics,
            "individual_models": individual_analytics,
            "summary": {
                "total_models": len(self.labelers),
                "total_prompts_across_models": total_prompts,
                "total_usage_across_models": total_usage,
                "avg_prompts_per_model": total_prompts / max(len(self.labelers), 1)
            }
        }

        return combined_analytics

    def export_all_prompt_histories(self, output_dir: Path) -> None:
        """
        Export prompt histories from all models and ensemble operations.

        Args:
            output_dir (Path): Directory to save all prompt exports.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export ensemble-level prompts
        ensemble_file = output_dir / "ensemble_prompts.csv"
        self.ensemble_prompt_store.export_prompts(ensemble_file)

        # Export individual model prompts
        for model_id, labeler in self.labelers.items():
            model_config = self.model_configs[model_id]
            model_file = output_dir / f"model_{model_id}_{model_config.model_name}_T{model_config.temperature}_prompts.csv"
            labeler.export_prompt_history(model_file)

        logger.info(f"Exported prompt histories for {len(self.labelers)} models to {output_dir}")

    def analyze_prompt_diversity(self) -> dict[str, Any]:
        """
        Analyze diversity of prompts across different model configurations.

        Returns:
            dict: Analysis of prompt diversity and overlap between models.
        """
        model_prompt_sets = {}

        # Collect unique prompts from each model
        for model_id, labeler in self.labelers.items():
            prompt_ids = set(labeler.prompt_store.prompts.keys())
            model_prompt_sets[model_id] = prompt_ids

        if len(model_prompt_sets) < 2:
            return {"error": "Need at least 2 models for diversity analysis"}

        # Calculate overlaps
        all_prompts = set()
        for prompt_set in model_prompt_sets.values():
            all_prompts.update(prompt_set)

        overlaps = {}
        for model1 in model_prompt_sets:
            for model2 in model_prompt_sets:
                if model1 < model2:  # Avoid duplicates
                    overlap = len(model_prompt_sets[model1] & model_prompt_sets[model2])
                    union = len(model_prompt_sets[model1] | model_prompt_sets[model2])
                    jaccard = overlap / union if union > 0 else 0
                    overlaps[f"{model1}_{model2}"] = {
                        "overlap_count": overlap,
                        "jaccard_similarity": jaccard
                    }

        diversity_analysis = {
            "total_unique_prompts": len(all_prompts),
            "prompts_per_model": {
                model_id: len(prompt_set)
                for model_id, prompt_set in model_prompt_sets.items()
            },
            "pairwise_overlaps": overlaps,
            "avg_jaccard_similarity": (
                sum(overlap["jaccard_similarity"] for overlap in overlaps.values()) /
                max(len(overlaps), 1)
            )
        }

        return diversity_analysis

    def find_consensus_prompts(self, min_models: int = 2) -> list[dict[str, Any]]:
        """
        Find prompts that were used by multiple models (consensus prompts).

        Args:
            min_models (int): Minimum number of models that must have used the prompt.

        Returns:
            list: List of consensus prompts with usage statistics.
        """
        prompt_usage = {}  # prompt_text -> list of model_ids that used it

        # Collect all prompts and which models used them
        for model_id, labeler in self.labelers.items():
            for prompt_record in labeler.prompt_store.prompts.values():
                prompt_text = prompt_record.prompt_text
                if prompt_text not in prompt_usage:
                    prompt_usage[prompt_text] = []
                prompt_usage[prompt_text].append({
                    "model_id": model_id,
                    "model_name": self.model_configs[model_id].model_name,
                    "temperature": self.model_configs[model_id].temperature,
                    "usage_count": prompt_record.usage_count,
                    "success_rate": (
                        prompt_record.successful_predictions /
                        max(prompt_record.successful_predictions + prompt_record.failed_predictions, 1)
                    )
                })

        # Filter for consensus prompts
        consensus_prompts = []
        for prompt_text, model_usages in prompt_usage.items():
            if len(model_usages) >= min_models:
                consensus_prompts.append({
                    "prompt_text": prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text,
                    "prompt_hash": PromptRecord.generate_prompt_id(prompt_text),
                    "num_models_used": len(model_usages),
                    "model_usages": model_usages,
                    "total_usage": sum(usage["usage_count"] for usage in model_usages),
                    "avg_success_rate": sum(usage["success_rate"] for usage in model_usages) / len(model_usages)
                })

        # Sort by number of models and total usage
        consensus_prompts.sort(key=lambda x: (x["num_models_used"], x["total_usage"]), reverse=True)

        return consensus_prompts
