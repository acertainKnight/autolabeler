"""
Ensemble labeling service for multi-model predictions.

This module provides functionality for running multiple models in parallel
and consolidating their predictions using various ensemble methods.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from ...config import Settings
from ..base import BatchProcessor, ConfigurableComponent
from ..configs import BatchConfig, EnsembleConfig, LabelingConfig


class ModelConfig(BaseModel):
    """Configuration for a single model in the ensemble."""

    model_id: str | None = Field(default=None, description="Unique model identifier")
    model_name: str = Field(description="Model name/identifier")
    provider: str = Field(default="openrouter", description="Model provider")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Sampling temperature")
    max_tokens: int | None = Field(default=None, description="Maximum tokens in response")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")

    # RAG settings
    use_rag: bool = Field(default=True, description="Whether to use RAG examples")
    max_examples: int = Field(default=5, description="Maximum RAG examples")
    prefer_human_examples: bool = Field(default=True, description="Prefer human-labeled examples")

    # Metadata
    description: str | None = Field(default=None, description="Model configuration description")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")

    def generate_id(self) -> str:
        """Generate unique ID based on configuration."""
        key_parts = [
            self.model_name,
            f"T{self.temperature}",
            f"seed{self.seed}" if self.seed else "noseed"
        ]
        return "_".join(key_parts).replace("/", "-").replace(".", "_")[:64]


class EnsembleResult(BaseModel):
    """Result from ensemble prediction with metadata."""

    label: str = Field(description="Consolidated label")
    confidence: float = Field(description="Consolidated confidence score")
    reasoning: str | None = Field(default=None, description="Consolidated reasoning")

    # Ensemble metadata
    ensemble_method: str | None = Field(default=None, description="Method used for consolidation")
    model_agreement: float | None = Field(default=None, description="Agreement between models")
    num_models_used: int | None = Field(default=None, description="Number of models that contributed")
    individual_predictions: list[dict[str, Any]] | None = Field(default=None, description="Individual model predictions")
    disagreement_labels: list[str] | None = Field(default=None, description="Labels that models disagreed on")


class EnsembleService(ConfigurableComponent, BatchProcessor):
    """
    Service for ensemble labeling with multiple models.

    Manages multiple model configurations, runs parallel predictions,
    and consolidates results using various ensemble methods.

    Args:
        dataset_name (str): Name of the dataset.
        settings (Settings): Application settings.
        ensemble_dir (Path | None): Directory for storing ensemble results.

    Example:
        >>> service = EnsembleService("sentiment_analysis", settings)
        >>> service.add_model(ModelConfig(model_name="gpt-3.5-turbo", temperature=0.1))
        >>> result = service.label_text_ensemble("Great product!", method="majority_vote")
    """

    def __init__(
        self,
        dataset_name: str,
        settings: Settings,
        ensemble_dir: Path | None = None,
    ) -> None:
        """Initialize ensemble service with dataset configuration."""
        super().__init__(
            component_type="ensemble_service",
            dataset_name=dataset_name,
            settings=settings,
        )
        self.settings = settings

        self.dataset_name = dataset_name
        self.ensemble_dir = (
            ensemble_dir
            or self.storage_path
        )
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Model configurations
        self.model_configs: dict[str, ModelConfig] = {}
        self.model_services: dict[str, Any] = {}  # Will hold labeling services

        # Results storage
        self.results_file = self.ensemble_dir / "ensemble_results.json"
        self.model_performance: dict[str, dict[str, Any]] = {}

        # Load existing state
        self._load_state()

    def add_model(self, config: ModelConfig) -> str:
        """
        Add a model configuration to the ensemble.

        Args:
            config (ModelConfig): Model configuration.

        Returns:
            str: Model ID.

        Example:
            >>> model_id = service.add_model(
            ...     ModelConfig(model_name="gpt-4", temperature=0.2)
            ... )
        """
        if not config.model_id:
            config.model_id = config.generate_id()

        self.model_configs[config.model_id] = config

        # Initialize performance tracking
        self.model_performance[config.model_id] = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "avg_confidence": 0.0,
            "created_at": datetime.now().isoformat()
        }

        self._save_state()
        logger.info(f"Added model to ensemble: {config.model_id}")

        return config.model_id

    def create_model_variants(
        self,
        base_model: str,
        temperatures: list[float] | None = None,
        seeds: list[int] | None = None
    ) -> list[str]:
        """
        Create multiple model configuration variants.

        Args:
            base_model (str): Base model name.
            temperatures (list[float] | None): Temperature values to test.
            seeds (list[int] | None): Seed values to test.

        Returns:
            list[str]: Created model IDs.

        Example:
            >>> model_ids = service.create_model_variants(
            ...     "gpt-3.5-turbo",
            ...     temperatures=[0.1, 0.5, 0.9]
            ... )
        """
        temperatures = temperatures or [0.1, 0.3, 0.7]
        seeds = seeds or [42]

        model_ids = []
        for temp in temperatures:
            for seed in seeds:
                config = ModelConfig(
                    model_name=base_model,
                    temperature=temp,
                    seed=seed,
                    description=f"{base_model} T={temp} seed={seed}",
                )
                model_id = self.add_model(config)
                model_ids.append(model_id)

        return model_ids

    def label_text_ensemble(
        self,
        text: str,
        config: EnsembleConfig | None = None,
        model_ids: list[str] | None = None,
        ruleset: dict[str, Any] | None = None,
    ) -> EnsembleResult:
        """
        Label text using ensemble of models.

        Args:
            text (str): Text to label.
            config (EnsembleConfig | None): Ensemble configuration.
            model_ids (list[str] | None): Specific models to use.
            ruleset (dict[str, Any] | None): Ruleset for labeling.

        Returns:
            EnsembleResult: Consolidated prediction.

        Example:
            >>> result = service.label_text_ensemble(
            ...     "This movie was amazing!",
            ...     config=EnsembleConfig(method="confidence_weighted"),
            ...     ruleset={"genre": "action"}
            ... )
        """
        config = config or EnsembleConfig()
        model_ids = model_ids or list(self.model_configs.keys())

        if not model_ids:
            raise ValueError("No models configured for ensemble")

        # Get predictions from each model
        individual_predictions = self._get_individual_predictions(
            text, model_ids, ruleset=ruleset
        )

        if not individual_predictions:
            raise ValueError("No successful predictions from any model")

        # Consolidate predictions
        result = self._consolidate_predictions(individual_predictions, config)

        # Add ensemble metadata
        result.ensemble_method = config.method
        result.num_models_used = len(individual_predictions)
        result.individual_predictions = individual_predictions

        # Update model performance
        self._update_performance(individual_predictions, result)

        return result

    async def alabel_text_ensemble(
        self,
        text: str,
        config: EnsembleConfig | None = None,
        model_ids: list[str] | None = None,
        ruleset: dict[str, Any] | None = None,
    ) -> EnsembleResult:
        """Asynchronously label text using ensemble of models."""
        config = config or EnsembleConfig()
        model_ids = model_ids or list(self.model_configs.keys())

        if not model_ids:
            raise ValueError("No models configured for ensemble")

        # Get predictions from each model
        individual_predictions = await self._aget_individual_predictions(
            text, model_ids, ruleset=ruleset
        )

        if not individual_predictions:
            raise ValueError("No successful predictions from any model")

        # Consolidate predictions
        result = self._consolidate_predictions(individual_predictions, config)

        # Add ensemble metadata
        result.ensemble_method = config.method
        result.num_models_used = len(individual_predictions)
        result.individual_predictions = individual_predictions

        # Update model performance
        self._update_performance(individual_predictions, result)

        return result

    def label_dataframe_ensemble(
        self,
        df: pd.DataFrame,
        text_column: str,
        config: EnsembleConfig | None = None,
        batch_config: BatchConfig | None = None,
        model_ids: list[str] | None = None,
        save_individual: bool = True,
        ruleset: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Label DataFrame using ensemble of models.

        Args:
            df (pd.DataFrame): DataFrame to label.
            text_column (str): Text column name.
            config (EnsembleConfig | None): Ensemble configuration.
            batch_config (BatchConfig | None): Batch processing configuration.
            model_ids (list[str] | None): Specific models to use.
            save_individual (bool): Save individual model results.
            ruleset (dict[str, Any] | None): Ruleset for labeling.

        Returns:
            pd.DataFrame: DataFrame with ensemble predictions.

        Example:
            >>> results_df = service.label_dataframe_ensemble(
            ...     df, "review_text",
            ...     config=EnsembleConfig(method="high_agreement"),
            ...     ruleset={"genre": "action"}
            ... )
        """
        config = config or EnsembleConfig()
        batch_config = batch_config or BatchConfig()
        model_ids = model_ids or list(self.model_configs.keys())

        items = df.to_dict("records")

        def process_func(batch: list[dict]) -> list[dict]:
            async def process_batch_async() -> list[dict]:
                tasks = [
                    self.alabel_text_ensemble(
                        str(item[text_column]), config, model_ids, ruleset
                    )
                    for item in batch
                ]
                ensemble_results = await asyncio.gather(*tasks, return_exceptions=True)

                processed_batch = []
                for item, result in zip(batch, ensemble_results):
                    result_row = item.copy()
                    if isinstance(result, EnsembleResult):
                        result_row["ensemble_label"] = result.label
                        result_row["ensemble_confidence"] = result.confidence
                        result_row["ensemble_method"] = result.ensemble_method
                        result_row["model_agreement"] = result.model_agreement
                        result_row["num_models"] = result.num_models_used
                        if save_individual:
                            result_row["individual_predictions"] = json.dumps(
                                result.individual_predictions
                            )
                    else:
                        logger.error(f"Failed to get ensemble prediction: {result}")
                        result_row["ensemble_label"] = None
                        result_row["ensemble_confidence"] = 0.0
                        result_row["ensemble_error"] = str(result)
                    processed_batch.append(result_row)
                return processed_batch

            return asyncio.run(process_batch_async())

        # Process in batches
        all_results = self.process_in_batches(
            items=items,
            batch_size=batch_config.batch_size,
            process_func=process_func,
            desc="Ensemble labeling",
            resume_key=f"{self.dataset_name}_ensemble"
            if batch_config.resume
            else None,
        )

        # Create results DataFrame
        results_df = pd.DataFrame(all_results)

        # Save results
        output_path = (
            self.ensemble_dir
            / f"ensemble_{config.method}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        )
        results_df.to_csv(output_path, index=False)

        logger.info(f"Ensemble labeling completed. Results saved to {output_path}")
        return results_df

    def get_model_performance(self) -> pd.DataFrame:
        """
        Get performance comparison of all models.

        Returns:
            pd.DataFrame: Model performance metrics.

        Example:
            >>> perf = service.get_model_performance()
            >>> print(perf[["model_id", "success_rate", "avg_confidence"]])
        """
        data = []

        for model_id, perf in self.model_performance.items():
            config = self.model_configs.get(model_id)

            row = {
                "model_id": model_id,
                "model_name": config.model_name if config else "unknown",
                "temperature": config.temperature if config else None,
                "total_predictions": perf["total_predictions"],
                "successful_predictions": perf["successful_predictions"],
                "success_rate": (
                    perf["successful_predictions"] / max(perf["total_predictions"], 1)
                ),
                "avg_confidence": perf["avg_confidence"]
            }

            data.append(row)

        return pd.DataFrame(data).sort_values("success_rate", ascending=False)

    def _get_individual_predictions(
        self,
        text: str,
        model_ids: list[str],
        ruleset: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Get predictions from individual models."""
        predictions = []

        # Import here to avoid circular dependency
        from ..labeling.labeling_service import LabelingService

        for model_id in model_ids:
            config = self.model_configs[model_id]

            # Get or create labeling service for this model
            if model_id not in self.model_services:
                # Create labeling config from model config
                labeling_config = LabelingConfig(
                    model_name=config.model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    use_rag=config.use_rag,
                    max_examples=config.max_examples,
                    prefer_human_examples=config.prefer_human_examples
                )

                self.model_services[model_id] = LabelingService(
                    self.dataset_name,
                    labeling_config
                )

            try:
                # Get prediction
                service = self.model_services[model_id]
                result = service.label_text(text, ruleset=ruleset)

                predictions.append({
                    "model_id": model_id,
                    "model_name": config.model_name,
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "reasoning": result.get("reasoning")
                })

            except Exception as e:
                logger.error(f"Model {model_id} failed: {e}")
                continue

        return predictions

    async def _aget_individual_predictions(
        self,
        text: str,
        model_ids: list[str],
        ruleset: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Get predictions from individual models asynchronously."""
        from ..labeling.labeling_service import LabelingService

        async def get_prediction(model_id: str) -> dict[str, Any] | None:
            config = self.model_configs.get(model_id)
            if not config:
                logger.warning(f"No config found for model_id: {model_id}")
                return None

            if model_id not in self.model_services:
                labeling_config = LabelingConfig(
                    model_name=config.model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    use_rag=config.use_rag,
                    max_examples=config.max_examples,
                    prefer_human_examples=config.prefer_human_examples,
                )
                self.model_services[model_id] = LabelingService(
                    self.dataset_name, self.settings, labeling_config
                )

            service = self.model_services[model_id]
            try:
                result = await service.alabel_text(text, ruleset=ruleset)
                return {
                    "model_id": model_id,
                    "model_name": config.model_name,
                    "label": result.label,
                    "confidence": result.confidence,
                    "reasoning": getattr(result, "reasoning", None),
                }
            except Exception as e:
                logger.error(f"Model {model_id} failed during async prediction: {e}")
                return None

        tasks = [get_prediction(model_id) for model_id in model_ids]
        results = await asyncio.gather(*tasks)
        return [res for res in results if res is not None]

    def _consolidate_predictions(
        self,
        predictions: list[dict[str, Any]],
        config: EnsembleConfig
    ) -> EnsembleResult:
        """Consolidate predictions using specified method."""
        # Filter by confidence threshold
        valid_preds = [
            p for p in predictions
            if p["confidence"] >= config.min_confidence
        ]

        if not valid_preds:
            valid_preds = predictions  # Fall back to all

        # Apply method
        if config.method == "majority_vote":
            return self._majority_vote(valid_preds)
        elif config.method == "confidence_weighted":
            return self._confidence_weighted(valid_preds)
        elif config.method == "high_agreement":
            return self._high_agreement(valid_preds, config.agreement_threshold)
        else:
            # Default to majority vote
            return self._majority_vote(valid_preds)

    def _majority_vote(self, predictions: list[dict[str, Any]]) -> EnsembleResult:
        """Simple majority voting."""
        labels = [p["label"] for p in predictions]
        label_counts = Counter(labels)

        most_common_label, vote_count = label_counts.most_common(1)[0]
        agreement = vote_count / len(predictions)

        # Average confidence of majority voters
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

    def _confidence_weighted(self, predictions: list[dict[str, Any]]) -> EnsembleResult:
        """Confidence-weighted voting."""
        label_weights = defaultdict(float)
        total_weight = 0.0

        for pred in predictions:
            weight = pred["confidence"]
            label_weights[pred["label"]] += weight
            total_weight += weight

        if total_weight == 0:
            return EnsembleResult(label="unknown", confidence=0.0)

        # Normalize weights
        label_probs = {
            label: weight / total_weight
            for label, weight in label_weights.items()
        }

        best_label = max(label_probs, key=label_probs.get)
        best_confidence = label_probs[best_label]

        return EnsembleResult(
            label=best_label,
            confidence=best_confidence,
            model_agreement=best_confidence,
            disagreement_labels=list(set(label_probs.keys()) - {best_label})
        )

    def _high_agreement(
        self,
        predictions: list[dict[str, Any]],
        threshold: float
    ) -> EnsembleResult:
        """High agreement consolidation."""
        result = self._majority_vote(predictions)

        # Only accept if agreement meets threshold
        if result.model_agreement and result.model_agreement < threshold:
            result.label = "uncertain"
            result.confidence *= result.model_agreement

        return result

    def _update_performance(
        self,
        predictions: list[dict[str, Any]],
        result: EnsembleResult
    ) -> None:
        """Update model performance metrics."""
        for pred in predictions:
            model_id = pred["model_id"]
            perf = self.model_performance[model_id]

            # Update counts
            perf["total_predictions"] += 1
            if pred["label"] == result.label:
                perf["successful_predictions"] += 1

            # Update average confidence
            n = perf["total_predictions"]
            perf["avg_confidence"] = (
                (perf["avg_confidence"] * (n - 1) + pred["confidence"]) / n
            )

        self._save_state()

    def _save_state(self) -> None:
        """Save ensemble state to disk."""
        state = {
            "model_configs": {
                mid: config.model_dump() for mid, config in self.model_configs.items()
            },
            "model_performance": self.model_performance,
            "last_updated": datetime.now().isoformat()
        }

        with open(self.results_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load ensemble state from disk."""
        if not self.results_file.exists():
            return

        try:
            with open(self.results_file, 'r') as f:
                state = json.load(f)

            # Load model configs
            for mid, config_data in state.get("model_configs", {}).items():
                self.model_configs[mid] = ModelConfig(**config_data)

            # Load performance data
            self.model_performance = state.get("model_performance", {})

            logger.info(f"Loaded ensemble state with {len(self.model_configs)} models")

        except Exception as e:
            logger.warning(f"Could not load ensemble state: {e}")
