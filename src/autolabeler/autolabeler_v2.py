"""
Streamlined AutoLabeler v2 with modular architecture.

This is a refactored version that demonstrates better separation of concerns
and reduced cognitive complexity through composition rather than inheritance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from .config import Settings
from .core import (
    BatchConfig,
    DataSplitConfig,
    EnsembleConfig,
    EvaluationConfig,
    GenerationConfig,
    LabelingConfig,
    RuleGenerationConfig,
    DataSplitService,
    EnsembleService,
    EvaluationService,
    LabelingService,
    RuleGenerationService,
    SyntheticGenerationService,
)
from .core.ensemble.ensemble_service import ModelConfig
from .core.knowledge import KnowledgeStore


class AutoLabelerV2:
    """
    Streamlined AutoLabeler with modular service architecture.

    This version uses composition over inheritance, with separate services
    for each major functionality. This reduces complexity and makes the
    codebase easier to understand and maintain.

    Example:
        >>> settings = Settings(openrouter_api_key="your-key")
        >>> labeler = AutoLabelerV2("sentiment_analysis", settings)
        >>>
        >>> # Simple labeling
        >>> results = labeler.label(
        ...     df, "text",
        ...     labeling_config=LabelingConfig(use_rag=True),
        ...     batch_config=BatchConfig(batch_size=100)
        ... )
        >>>
        >>> # Advanced workflow with train/test split
        >>> workflow_results = labeler.run_workflow(
        ...     df, "text", "sentiment",
        ...     include_synthetic=True,
        ...     generate_rules=True
        ... )
    """

    def __init__(
        self,
        dataset_name: str,
        settings: Settings,
        labeling_config: LabelingConfig | None = None,
    ):
        """
        Initialize the AutoLabeler.

        Args:
            dataset_name: A unique identifier for the project or dataset.
            settings: An object containing configuration like API keys.
            labeling_config: Configuration for labeling tasks.
        """
        self.dataset_name = dataset_name
        self.settings = settings
        self.labeling_config = labeling_config or LabelingConfig()
        self._services: dict[str, Any] = {}
        logger.info(f"Initialized AutoLabelerV2 for dataset: {dataset_name}")

    def _get_service(self, service_name: str, service_class: Any) -> Any:
        """Lazy-load and cache a service."""
        if service_name not in self._services:
            self._services[service_name] = service_class(
                self.dataset_name, self.settings
            )
        return self._services[service_name]

    @property
    def labeling_service(self) -> LabelingService:
        """Lazy-load and cache the labeling service."""
        service_name = "labeling"
        if service_name not in self._services:
            self._services[service_name] = LabelingService(
                self.dataset_name, self.settings, self.labeling_config
            )
        return self._services[service_name]

    @property
    def split_service(self) -> DataSplitService:
        return self._get_service("split", DataSplitService)

    @property
    def eval_service(self) -> EvaluationService:
        return self._get_service("evaluation", EvaluationService)

    @property
    def synthetic_service(self) -> SyntheticGenerationService:
        return self._get_service("synthetic", SyntheticGenerationService)

    @property
    def rule_service(self) -> RuleGenerationService:
        return self._get_service("rules", RuleGenerationService)

    @property
    def ensemble_service(self) -> EnsembleService:
        return self._get_service("ensemble", EnsembleService)

    @property
    def knowledge_store(self) -> KnowledgeStore:
        """Direct access to the knowledge store for advanced use cases."""
        # KnowledgeStore is a dependency of other services, so we can get it from one.
        return self.labeling_service.knowledge_store

    def add_training_data(
        self, df: pd.DataFrame, text_column: str, label_column: str
    ) -> None:
        """Add labeled training examples to the knowledge store."""
        self.knowledge_store.add_examples(df, text_column, label_column, source="human")

    def label(
        self,
        df: pd.DataFrame,
        text_column: str,
        labeling_config: LabelingConfig | None = None,
        batch_config: BatchConfig | None = None,
        ruleset: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Label a dataset."""
        return self.labeling_service.label_dataframe(
            df,
            text_column,
            config=labeling_config,
            batch_config=batch_config,
            ruleset=ruleset,
        )

    def label_ensemble(
        self,
        df: pd.DataFrame,
        text_column: str,
        model_configs: list[dict | ModelConfig],
        config: EnsembleConfig | None = None,
        ruleset: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Label a dataset using a multi-model ensemble."""
        service = self.ensemble_service
        for mc in model_configs:
            if isinstance(mc, dict):
                service.add_model(ModelConfig(**mc))
            else:
                service.add_model(mc)
        return service.label_dataframe_ensemble(
            df, text_column, config=config, ruleset=ruleset
        )

    def evaluate(
        self,
        df: pd.DataFrame,
        true_label_column: str,
        pred_label_column: str,
        confidence_column: str | None = None,
        output_report_path: Path | None = None,
    ) -> dict[str, Any]:
        """Evaluate predictions."""
        return self.eval_service.evaluate(
            df,
            true_label_column,
            pred_label_column,
            confidence_column,
            output_report_path,
        )

    def split_data(
        self, df: pd.DataFrame, test_size: float = 0.2, stratify_column: str | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets."""
        config = DataSplitConfig(test_size=test_size, stratify_column=stratify_column)
        train_df, test_df, _ = self.split_service.create_split(df, config)
        return train_df, test_df

    def generate_synthetic_data(
        self,
        target_distribution: dict[str, int],
        config: GenerationConfig | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic data to balance label distribution."""
        return self.synthetic_service.balance_dataset(
            target_distribution, config=config
        )

    def generate_rules(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        config: RuleGenerationConfig | None = None,
        output_file: Path | None = None,
    ) -> dict[str, Any]:
        """Generate labeling rules from data."""
        ruleset_model = self.rule_service.generate_rules(
            df, text_column, label_column, config=config
        )
        ruleset_dict = ruleset_model.model_dump()
        if output_file:
            self.rule_service.export_rules(ruleset_dict, output_file)
        return ruleset_dict

    def export_results(self, df: pd.DataFrame, output_path: Path):
        """A utility to save a DataFrame to a file."""
        if output_path.suffix == '.parquet':
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        logger.info(f"Results exported to {output_path}")

    # ==================== Advanced Workflow Methods ====================

    def label_with_split(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        split_config: DataSplitConfig | None = None,
        labeling_config: LabelingConfig | None = None,
        batch_config: BatchConfig | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Label data with automatic train/test split and evaluation.

        Args:
            df: Input DataFrame with ground truth labels
            text_column: Text column name
            label_column: Ground truth label column
            split_config: Data split configuration
            labeling_config: Labeling configuration
            batch_config: Batch processing configuration

        Returns:
            Tuple of (labeled_test_df, evaluation_results)
        """
        # Create train/test split
        train_df, test_df, _ = self.split_service.create_split(df, split_config)

        # Ensure no data leakage
        train_df, test_df = self.split_service.ensure_no_leakage(
            train_df, test_df, text_column
        )

        # Add training data to knowledge base
        self.add_training_data(train_df, text_column, label_column)

        # Label test set
        pred_column = f"{label_column}_predicted"
        test_df_labeled = self.label(
            test_df, text_column, labeling_config, batch_config
        )

        # Evaluate results
        eval_results = self.evaluate(
            test_df_labeled, label_column, pred_column,
            f"{pred_column}_confidence"
        )

        return test_df_labeled, eval_results

    def balance_dataset_with_synthetic(
        self,
        target_distribution: dict[str, int] | str = "equal",
        generation_config: GenerationConfig | None = None,
        batch_config: BatchConfig | None = None,
    ) -> dict[str, list[Any]]:
        """
        Balance the dataset by generating synthetic examples.

        Args:
            target_distribution: Target label distribution or "equal"
            generation_config: Generation configuration
            batch_config: Batch processing configuration

        Returns:
            Dictionary mapping labels to generated examples
        """
        return self.synthetic_service.balance_dataset(
            target_distribution, generation_config, batch_config
        )

    def run_workflow(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        test_size: float = 0.2,
        include_synthetic: bool = False,
        generate_rules: bool = False,
        use_ensemble: bool = False,
    ) -> dict[str, Any]:
        """
        Run a complete labeling workflow with all bells and whistles.

        Args:
            df: Input DataFrame with ground truth labels
            text_column: Text column name
            label_column: Ground truth label column
            test_size: Fraction for test set
            include_synthetic: Whether to generate synthetic data for balance
            generate_rules: Whether to generate labeling rules
            use_ensemble: Whether to use ensemble labeling

        Returns:
            Dictionary containing all workflow results
        """
        results = {
            "dataset_name": self.dataset_name,
            "workflow_steps": [],
        }

        # Step 1: Create train/test split
        split_config = DataSplitConfig(test_size=test_size, stratify_column=label_column)
        train_df, test_df, _ = self.split_service.create_split(df, split_config)
        results["split_info"] = {
            "train_size": len(train_df),
            "test_size": len(test_df),
        }
        results["workflow_steps"].append("data_split")

        # Step 2: Add training data
        self.add_training_data(train_df, text_column, label_column)
        results["workflow_steps"].append("add_training_data")

        # Step 3: Generate synthetic data if requested
        if include_synthetic:
            synthetic_examples = self.balance_dataset_with_synthetic()
            results["synthetic_generation"] = {
                label: len(examples)
                for label, examples in synthetic_examples.items()
            }
            results["workflow_steps"].append("synthetic_generation")

        # Step 4: Generate rules if requested
        if generate_rules:
            rule_result = self.generate_rules(train_df, text_column, label_column)
            results["rule_generation"] = {
                "num_rules": len(rule_result.ruleset.rules),
                "covered_labels": len(rule_result.coverage_analysis["covered_labels"]),
            }
            results["workflow_steps"].append("rule_generation")

        # Step 5: Label test set
        pred_column = f"{label_column}_predicted"

        if use_ensemble:
            # Use ensemble labeling
            test_df_labeled = self.label_ensemble(
                test_df, text_column,
                model_configs=[
                    {'model_name': 'openai/gpt-4o-mini'},
                    {'model_name': 'google/gemini-2.5-flash-preview-05-20'}
                ]
            )
            results["workflow_steps"].append("ensemble_labeling")
        else:
            # Regular labeling
            test_df_labeled = self.label(test_df, text_column, labeling_config=LabelingConfig())
            results["workflow_steps"].append("labeling")

        # Step 6: Evaluate results
        eval_results = self.evaluate(
            test_df_labeled, label_column, pred_column,
            f"{pred_column}_confidence"
        )
        results["evaluation"] = eval_results
        results["workflow_steps"].append("evaluation")

        # Store final results
        results["test_df_labeled"] = test_df_labeled

        logger.info(
            f"Workflow complete. Accuracy: {eval_results.get('accuracy', 0):.3f}"
        )

        return results

    # ==================== Utility Methods ====================

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics about all services."""
        stats = {
            "dataset_name": self.dataset_name,
            "knowledge_base": self.knowledge_store.get_stats(),
            "services": {},
        }

        # Add stats from initialized services
        for service_name, service in self._services.items():
            stats["services"][service_name] = service.get_stats()

        return stats

    def export_knowledge_base(self, output_path: Path) -> None:
        """Export the knowledge base."""
        self.knowledge_store.export_knowledge_base(output_path)

    def clear_knowledge_base(self) -> None:
        """Clear all data from the knowledge base."""
        self.knowledge_store.clear_knowledge_base()
