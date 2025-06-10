from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field


class PromptRecord(BaseModel):
    """
    Record of a specific prompt used in labeling.

    Stores the complete prompt text, metadata about its generation,
    and usage statistics for tracking and analysis.
    """

    prompt_id: str = Field(description="Unique hash-based identifier for this prompt")
    prompt_text: str = Field(description="Complete prompt text sent to the model")
    template_source: str | None = Field(default=None, description="Original template used")

    # Generation context
    dataset_name: str = Field(description="Dataset this prompt was used for")
    model_config_id: str | None = Field(default=None, description="Model configuration used")
    model_name: str | None = Field(default=None, description="Model that processed this prompt")

    # Template variables used
    variables: dict[str, Any] = Field(default_factory=dict, description="Variables substituted in template")

    # RAG context
    num_examples: int = Field(default=0, description="Number of RAG examples included")
    example_sources: list[str] = Field(default_factory=list, description="Sources of examples (human/model)")

    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = Field(default=1, description="Number of times this exact prompt was used")
    last_used: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Performance tracking
    successful_predictions: int = Field(default=0, description="Number of successful predictions")
    failed_predictions: int = Field(default=0, description="Number of failed predictions")
    avg_confidence: float | None = Field(default=None, description="Average confidence of predictions")

    # Tags and categorization
    tags: list[str] = Field(default_factory=list, description="Custom tags for organization")

    @classmethod
    def create_from_prompt(
        cls,
        prompt_text: str,
        dataset_name: str,
        template_source: str | None = None,
        variables: dict[str, Any] | None = None,
        model_config_id: str | None = None,
        model_name: str | None = None,
        num_examples: int = 0,
        example_sources: list[str] | None = None,
        tags: list[str] | None = None
    ) -> PromptRecord:
        """
        Create a new prompt record from prompt text and context.

        Args:
            prompt_text (str): The complete prompt text.
            dataset_name (str): Dataset this prompt is for.
            template_source (str | None): Original template used.
            variables (dict | None): Template variables used.
            model_config_id (str | None): Model configuration ID.
            model_name (str | None): Model name.
            num_examples (int): Number of RAG examples.
            example_sources (list[str] | None): Sources of examples.
            tags (list[str] | None): Custom tags.

        Returns:
            PromptRecord: New prompt record.
        """
        prompt_id = cls.generate_prompt_id(prompt_text)

        return cls(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            template_source=template_source,
            dataset_name=dataset_name,
            model_config_id=model_config_id,
            model_name=model_name,
            variables=variables or {},
            num_examples=num_examples,
            example_sources=example_sources or [],
            tags=tags or []
        )

    @staticmethod
    def generate_prompt_id(prompt_text: str) -> str:
        """
        Generate a unique ID for a prompt based on its content.

        Args:
            prompt_text (str): The prompt text to hash.

        Returns:
            str: Unique prompt identifier.
        """
        return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()[:16]

    def update_usage(self, successful: bool = True, confidence: float | None = None) -> None:
        """
        Update usage statistics for this prompt.

        Args:
            successful (bool): Whether the prediction was successful.
            confidence (float | None): Confidence score of the prediction.
        """
        self.usage_count += 1
        self.last_used = datetime.now().isoformat()

        if successful:
            self.successful_predictions += 1
        else:
            self.failed_predictions += 1

        if confidence is not None:
            if self.avg_confidence is None:
                self.avg_confidence = confidence
            else:
                # Running average
                total_predictions = self.successful_predictions + self.failed_predictions
                self.avg_confidence = (
                    (self.avg_confidence * (total_predictions - 1) + confidence) / total_predictions
                )


class PromptStore:
    """
    Storage and management system for prompt tracking.

    Maintains a persistent store of all prompts used in labeling,
    with deduplication, usage statistics, and analysis capabilities.

    Args:
        dataset_name (str): Name of the dataset for this prompt store.
        store_dir (Path | None): Directory to store prompt data.

    Example:
        >>> store = PromptStore("sentiment_analysis")
        >>> prompt_id = store.store_prompt("You are a sentiment classifier...", {...})
        >>> record = store.get_prompt(prompt_id)
    """

    def __init__(
        self,
        dataset_name: str,
        store_dir: Path | None = None
    ) -> None:
        self.dataset_name = dataset_name
        self.store_dir = store_dir or Path("prompt_store") / dataset_name
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Storage files
        self.prompts_file = self.store_dir / "prompts.json"
        self.usage_log_file = self.store_dir / "usage_log.csv"

        # In-memory cache
        self.prompts: dict[str, PromptRecord] = {}

        # Load existing prompts
        self._load_prompts()

        logger.info(f"Initialized PromptStore for {dataset_name} with {len(self.prompts)} prompts")

    def store_prompt(
        self,
        prompt_text: str,
        template_source: str | None = None,
        variables: dict[str, Any] | None = None,
        model_config_id: str | None = None,
        model_name: str | None = None,
        examples_used: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None
    ) -> str:
        """
        Store a prompt and return its unique ID.

        Args:
            prompt_text (str): Complete prompt text sent to model.
            template_source (str | None): Original template used.
            variables (dict | None): Template variables used.
            model_config_id (str | None): Model configuration ID.
            model_name (str | None): Model name.
            examples_used (list[dict] | None): RAG examples included.
            tags (list[str] | None): Custom tags.

        Returns:
            str: Unique prompt ID.

        Example:
            >>> prompt_id = store.store_prompt(
            ...     "Classify sentiment: 'Great movie!'",
            ...     template_source="sentiment_template.j2",
            ...     variables={"text": "Great movie!"},
            ...     model_name="gpt-3.5-turbo"
            ... )
        """
        prompt_id = PromptRecord.generate_prompt_id(prompt_text)

        if prompt_id in self.prompts:
            # Prompt already exists, update usage
            self.prompts[prompt_id].update_usage()
        else:
            # Create new prompt record
            num_examples = len(examples_used) if examples_used else 0
            example_sources = []

            if examples_used:
                example_sources = [
                    ex.get("source", "unknown") for ex in examples_used
                    if isinstance(ex, dict)
                ]

            record = PromptRecord.create_from_prompt(
                prompt_text=prompt_text,
                dataset_name=self.dataset_name,
                template_source=template_source,
                variables=variables,
                model_config_id=model_config_id,
                model_name=model_name,
                num_examples=num_examples,
                example_sources=example_sources,
                tags=tags
            )

            self.prompts[prompt_id] = record

        # Log usage
        self._log_usage(prompt_id, model_config_id, model_name)

        # Save to disk
        self._save_prompts()

        return prompt_id

    def get_prompt(self, prompt_id: str) -> PromptRecord | None:
        """
        Retrieve a prompt record by ID.

        Args:
            prompt_id (str): Unique prompt identifier.

        Returns:
            PromptRecord | None: Prompt record if found.
        """
        return self.prompts.get(prompt_id)

    def update_prompt_result(
        self,
        prompt_id: str,
        successful: bool = True,
        confidence: float | None = None
    ) -> None:
        """
        Update the result statistics for a prompt.

        Args:
            prompt_id (str): Prompt identifier.
            successful (bool): Whether prediction was successful.
            confidence (float | None): Confidence score.
        """
        if prompt_id in self.prompts:
            self.prompts[prompt_id].update_usage(successful, confidence)
            self._save_prompts()

    def find_similar_prompts(
        self,
        template_source: str | None = None,
        model_name: str | None = None,
        min_usage: int = 1,
        tags: list[str] | None = None
    ) -> list[PromptRecord]:
        """
        Find prompts matching specified criteria.

        Args:
            template_source (str | None): Filter by template source.
            model_name (str | None): Filter by model name.
            min_usage (int): Minimum usage count.
            tags (list[str] | None): Required tags.

        Returns:
            list[PromptRecord]: Matching prompt records.
        """
        matching_prompts = []

        for record in self.prompts.values():
            if template_source and record.template_source != template_source:
                continue
            if model_name and record.model_name != model_name:
                continue
            if record.usage_count < min_usage:
                continue
            if tags and not all(tag in record.tags for tag in tags):
                continue

            matching_prompts.append(record)

        return sorted(matching_prompts, key=lambda x: x.usage_count, reverse=True)

    def get_prompt_analytics(self) -> dict[str, Any]:
        """
        Get analytics about prompt usage.

        Returns:
            dict: Analytics data including usage patterns and performance.
        """
        if not self.prompts:
            return {"total_prompts": 0}

        records = list(self.prompts.values())

        analytics = {
            "total_prompts": len(records),
            "total_usage": sum(r.usage_count for r in records),
            "avg_usage_per_prompt": sum(r.usage_count for r in records) / len(records),
            "most_used_prompt": max(records, key=lambda x: x.usage_count).prompt_id,
            "success_rate": (
                sum(r.successful_predictions for r in records) /
                max(sum(r.successful_predictions + r.failed_predictions for r in records), 1)
            ),
            "avg_confidence": sum(
                r.avg_confidence for r in records if r.avg_confidence is not None
            ) / max(sum(1 for r in records if r.avg_confidence is not None), 1),
            "template_usage": {},
            "model_usage": {},
            "tag_usage": {}
        }

        # Template usage
        for record in records:
            template = record.template_source or "unknown"
            analytics["template_usage"][template] = analytics["template_usage"].get(template, 0) + record.usage_count

        # Model usage
        for record in records:
            model = record.model_name or "unknown"
            analytics["model_usage"][model] = analytics["model_usage"].get(model, 0) + record.usage_count

        # Tag usage
        for record in records:
            for tag in record.tags:
                analytics["tag_usage"][tag] = analytics["tag_usage"].get(tag, 0) + 1

        return analytics

    def export_prompts(self, output_path: Path, include_full_text: bool = True) -> None:
        """
        Export prompts to CSV for analysis.

        Args:
            output_path (Path): Path to save the export.
            include_full_text (bool): Whether to include full prompt text.
        """
        export_data = []

        for record in self.prompts.values():
            row = {
                "prompt_id": record.prompt_id,
                "dataset_name": record.dataset_name,
                "template_source": record.template_source,
                "model_name": record.model_name,
                "model_config_id": record.model_config_id,
                "usage_count": record.usage_count,
                "successful_predictions": record.successful_predictions,
                "failed_predictions": record.failed_predictions,
                "success_rate": (
                    record.successful_predictions /
                    max(record.successful_predictions + record.failed_predictions, 1)
                ),
                "avg_confidence": record.avg_confidence,
                "num_examples": record.num_examples,
                "example_sources": ",".join(record.example_sources),
                "tags": ",".join(record.tags),
                "created_at": record.created_at,
                "last_used": record.last_used
            }

            if include_full_text:
                row["prompt_text"] = record.prompt_text
                row["variables"] = json.dumps(record.variables)

            export_data.append(row)

        df = pd.DataFrame(export_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(export_data)} prompts to {output_path}")

    def cleanup_old_prompts(self, days_old: int = 30, min_usage: int = 1) -> int:
        """
        Clean up old, rarely used prompts.

        Args:
            days_old (int): Remove prompts older than this many days.
            min_usage (int): Only remove prompts with usage below this threshold.

        Returns:
            int: Number of prompts removed.
        """
        from datetime import datetime, timedelta

        cutoff_date = datetime.now() - timedelta(days=days_old)
        removed_count = 0

        prompts_to_remove = []
        for prompt_id, record in self.prompts.items():
            last_used_date = datetime.fromisoformat(record.last_used)

            if (last_used_date < cutoff_date and
                record.usage_count < min_usage):
                prompts_to_remove.append(prompt_id)

        for prompt_id in prompts_to_remove:
            del self.prompts[prompt_id]
            removed_count += 1

        if removed_count > 0:
            self._save_prompts()
            logger.info(f"Cleaned up {removed_count} old prompts")

        return removed_count

    def _load_prompts(self) -> None:
        """Load prompts from disk."""
        if not self.prompts_file.exists():
            return

        try:
            with open(self.prompts_file, 'r') as f:
                data = json.load(f)

            for prompt_id, prompt_data in data.items():
                self.prompts[prompt_id] = PromptRecord(**prompt_data)

            logger.info(f"Loaded {len(self.prompts)} prompts from disk")

        except Exception as e:
            logger.warning(f"Could not load prompts from disk: {e}")

    def _save_prompts(self) -> None:
        """Save prompts to disk."""
        try:
            data = {
                prompt_id: record.model_dump()
                for prompt_id, record in self.prompts.items()
            }

            with open(self.prompts_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save prompts to disk: {e}")

    def _log_usage(
        self,
        prompt_id: str,
        model_config_id: str | None,
        model_name: str | None
    ) -> None:
        """Log prompt usage to CSV for time-series analysis."""
        try:
            usage_entry = {
                "timestamp": datetime.now().isoformat(),
                "prompt_id": prompt_id,
                "model_config_id": model_config_id,
                "model_name": model_name,
                "dataset_name": self.dataset_name
            }

            # Append to usage log
            if self.usage_log_file.exists():
                df = pd.read_csv(self.usage_log_file)
                new_df = pd.concat([df, pd.DataFrame([usage_entry])], ignore_index=True)
            else:
                new_df = pd.DataFrame([usage_entry])

            new_df.to_csv(self.usage_log_file, index=False)

        except Exception as e:
            logger.warning(f"Could not log prompt usage: {e}")
