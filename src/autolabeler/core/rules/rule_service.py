"""Rule generation service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ...models import LabelingRule, RuleSet
from ..base import ConfigurableComponent, ProgressTracker
from ..configs import BatchConfig, RuleGenerationConfig
from ..utils import ruleset_utils


class RuleGenerationService(ConfigurableComponent, ProgressTracker):
    """Service for generating labeling rules from data."""

    def __init__(self, dataset_name: str, settings: Any, config: RuleGenerationConfig | None = None):
        """Initialize the rule generation service."""
        ConfigurableComponent.__init__(
            self,
            component_type="rule_generation",
            dataset_name=dataset_name,
            settings=settings,
            config=config or RuleGenerationConfig(),
        )
        ProgressTracker.__init__(self, f"{dataset_name}_rule_progress.json")
        logger.info(f"Initialized RuleGenerationService for dataset: {dataset_name}")

    def generate_rules(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        config: RuleGenerationConfig | None = None,
        batch_config: BatchConfig | None = None,
    ) -> RuleSet:
        """Generate labeling rules from labeled data."""
        config = config or RuleGenerationConfig()
        # Placeholder for actual rule generation logic
        logger.info("Rule generation logic would be implemented here.")
        return RuleSet(
            dataset_name=self.dataset_name,
            task_description=config.task_description or "",
            label_categories=list(df[label_column].unique()),
            rules=[],
            general_guidelines=[],
            version="1.0",
            creation_timestamp="",
            last_updated="",
        )

    def export_rules(self, ruleset: RuleSet, output_path: Path) -> None:
        """Export the generated ruleset to a file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            # A simple JSON dump for now. Could be markdown, etc.
            import json
            json.dump(ruleset.model_dump(), f, indent=2)
        logger.info(f"Exported ruleset to {output_path}")

    def load_rules_for_prompt(self, ruleset_file: Path) -> dict[str, Any] | None:
        """Load and format a ruleset for use in a prompt."""
        return ruleset_utils.load_ruleset_for_prompt(ruleset_file)
