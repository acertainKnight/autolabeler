"""
Prompt management and analytics for tracking labeling prompts.

This module provides functionality for storing, tracking, and analyzing
prompts used in the labeling process.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field


class PromptRecord(BaseModel):
    """Record of a specific prompt used in labeling."""

    prompt_id: str = Field(description="Unique hash-based identifier")
    prompt_text: str = Field(description="Complete prompt text")
    template_source: str | None = Field(default=None, description="Original template")

    # Context
    dataset_name: str = Field(description="Dataset this prompt was used for")
    model_name: str | None = Field(default=None, description="Model that processed this prompt")

    # Variables and examples
    variables: dict[str, Any] = Field(default_factory=dict)
    num_examples: int = Field(default=0)
    example_sources: list[str] = Field(default_factory=list)

    # Usage tracking
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = Field(default=1)
    last_used: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Performance
    successful_predictions: int = Field(default=0)
    failed_predictions: int = Field(default=0)
    avg_confidence: float | None = Field(default=None)

    # Organization
    tags: list[str] = Field(default_factory=list)

    @staticmethod
    def generate_id(prompt_text: str) -> str:
        """Generate unique ID from prompt text."""
        return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()[:16]

    def update_usage(self, successful: bool = True, confidence: float | None = None) -> None:
        """Update usage statistics."""
        self.usage_count += 1
        self.last_used = datetime.now().isoformat()

        if successful:
            self.successful_predictions += 1
        else:
            self.failed_predictions += 1

        if confidence is not None:
            total_predictions = self.successful_predictions + self.failed_predictions
            if self.avg_confidence is None:
                self.avg_confidence = confidence
            else:
                # Running average
                self.avg_confidence = (
                    (self.avg_confidence * (total_predictions - 1) + confidence) / total_predictions
                )


class PromptManager:
    """
    Storage and management system for prompt tracking and analytics.

    Maintains a persistent store of all prompts used in labeling with
    deduplication, usage statistics, and analysis capabilities.

    Args:
        dataset_name (str): Name of the dataset.
        store_dir (Path | None): Directory to store prompt data.

    Example:
        >>> manager = PromptManager("sentiment_analysis")
        >>> prompt_id = manager.store_prompt("Classify sentiment: positive or negative", {...})
        >>> analytics = manager.get_analytics()
    """

    def __init__(self, dataset_name: str, store_dir: Path | None = None) -> None:
        """Initialize prompt manager with dataset-specific configuration."""
        self.dataset_name = dataset_name
        self.store_dir = store_dir or Path("prompt_store") / dataset_name
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Storage files
        self.prompts_file = self.store_dir / "prompts.json"
        self.usage_log_file = self.store_dir / "usage_log.parquet"

        # In-memory cache
        self.prompts: dict[str, PromptRecord] = {}

        # Load existing prompts
        self._load_prompts()

    def store_prompt(
        self,
        prompt_text: str,
        template_source: str | None = None,
        variables: dict[str, Any] | None = None,
        model_name: str | None = None,
        examples_used: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None
    ) -> str:
        """
        Store a prompt and return its unique ID.

        Args:
            prompt_text (str): Complete prompt text.
            template_source (str | None): Original template used.
            variables (dict | None): Template variables.
            model_name (str | None): Model name.
            examples_used (list[dict] | None): RAG examples included.
            tags (list[str] | None): Custom tags.

        Returns:
            str: Unique prompt ID.

        Example:
            >>> prompt_id = manager.store_prompt(
            ...     "Classify sentiment: 'Great movie!'",
            ...     template_source="sentiment.j2",
            ...     variables={"text": "Great movie!"}
            ... )
        """
        prompt_id = PromptRecord.generate_id(prompt_text)

        if prompt_id in self.prompts:
            # Update existing prompt
            self.prompts[prompt_id].update_usage()
        else:
            # Create new prompt record
            num_examples = len(examples_used) if examples_used else 0
            example_sources = [
                ex.get("source", "unknown") for ex in examples_used or []
                if isinstance(ex, dict)
            ]

            record = PromptRecord(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                template_source=template_source,
                dataset_name=self.dataset_name,
                model_name=model_name,
                variables=variables or {},
                num_examples=num_examples,
                example_sources=example_sources,
                tags=tags or []
            )

            self.prompts[prompt_id] = record

        # Log usage
        self._log_usage(prompt_id, model_name)

        # Save to disk
        self._save_prompts()

        return prompt_id

    def update_result(
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

    def find_prompts(
        self,
        template_source: str | None = None,
        model_name: str | None = None,
        min_usage: int = 1,
        tags: list[str] | None = None
    ) -> list[PromptRecord]:
        """
        Find prompts matching specified criteria.

        Args:
            template_source (str | None): Filter by template.
            model_name (str | None): Filter by model.
            min_usage (int): Minimum usage count.
            tags (list[str] | None): Required tags.

        Returns:
            list[PromptRecord]: Matching prompts.
        """
        matching = []

        for record in self.prompts.values():
            # Apply filters
            if template_source and record.template_source != template_source:
                continue
            if model_name and record.model_name != model_name:
                continue
            if record.usage_count < min_usage:
                continue
            if tags and not all(tag in record.tags for tag in tags):
                continue

            matching.append(record)

        return sorted(matching, key=lambda x: x.usage_count, reverse=True)

    def get_analytics(self) -> dict[str, Any]:
        """
        Get comprehensive analytics about prompt usage.

        Returns:
            dict: Analytics including usage patterns and performance.

        Example:
            >>> analytics = manager.get_analytics()
            >>> print(f"Total prompts: {analytics['total_prompts']}")
        """
        if not self.prompts:
            return {"total_prompts": 0}

        records = list(self.prompts.values())
        total_predictions = sum(r.successful_predictions + r.failed_predictions for r in records)

        analytics = {
            "total_prompts": len(records),
            "total_usage": sum(r.usage_count for r in records),
            "avg_usage_per_prompt": sum(r.usage_count for r in records) / len(records),
            "most_used_prompt": max(records, key=lambda x: x.usage_count).prompt_id,
            "success_rate": (
                sum(r.successful_predictions for r in records) / max(total_predictions, 1)
            ),
            "avg_confidence": self._calculate_avg_confidence(records),
            "template_usage": self._aggregate_by_field(records, "template_source"),
            "model_usage": self._aggregate_by_field(records, "model_name"),
            "tag_distribution": self._aggregate_tags(records)
        }

        return analytics

    def export_to_csv(self, output_path: Path, include_full_text: bool = False) -> None:
        """
        Export prompts to CSV for analysis.

        Args:
            output_path (Path): Output file path.
            include_full_text (bool): Include full prompt text.
        """
        data = []

        for record in self.prompts.values():
            row = {
                "prompt_id": record.prompt_id,
                "dataset_name": record.dataset_name,
                "template_source": record.template_source,
                "model_name": record.model_name,
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

            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(data)} prompts to {output_path}")

    def cleanup_old_prompts(self, days_old: int = 30, min_usage: int = 1) -> int:
        """
        Remove old, rarely used prompts.

        Args:
            days_old (int): Age threshold in days.
            min_usage (int): Usage threshold.

        Returns:
            int: Number of prompts removed.
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        removed_count = 0

        prompts_to_remove = []
        for prompt_id, record in self.prompts.items():
            last_used_date = datetime.fromisoformat(record.last_used)

            if last_used_date < cutoff_date and record.usage_count < min_usage:
                prompts_to_remove.append(prompt_id)

        for prompt_id in prompts_to_remove:
            del self.prompts[prompt_id]
            removed_count += 1

        if removed_count > 0:
            self._save_prompts()
            logger.info(f"Removed {removed_count} old prompts")

        return removed_count

    def _calculate_avg_confidence(self, records: list[PromptRecord]) -> float:
        """Calculate average confidence across all records."""
        confidences = [r.avg_confidence for r in records if r.avg_confidence is not None]
        return sum(confidences) / max(len(confidences), 1)

    def _aggregate_by_field(self, records: list[PromptRecord], field: str) -> dict[str, int]:
        """Aggregate usage counts by field value."""
        aggregated = {}
        for record in records:
            value = getattr(record, field, None) or "unknown"
            aggregated[value] = aggregated.get(value, 0) + record.usage_count
        return aggregated

    def _aggregate_tags(self, records: list[PromptRecord]) -> dict[str, int]:
        """Aggregate tag distribution."""
        tag_counts = {}
        for record in records:
            for tag in record.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts

    def _log_usage(self, prompt_id: str, model_name: str | None) -> None:
        """Log prompt usage for time-series analysis."""
        try:
            usage_entry = {
                "timestamp": datetime.now().isoformat(),
                "prompt_id": prompt_id,
                "model_name": model_name,
                "dataset_name": self.dataset_name
            }

            # Create or append to log
            if self.usage_log_file.exists():
                df = pd.read_parquet(self.usage_log_file)
                new_df = pd.concat([df, pd.DataFrame([usage_entry])], ignore_index=True)
            else:
                new_df = pd.DataFrame([usage_entry])

            new_df.to_parquet(self.usage_log_file, index=False)

        except Exception as e:
            logger.warning(f"Could not log prompt usage: {e}")

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
            logger.error(f"Failed to save prompts: {e}")

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
            logger.warning(f"Could not load prompts: {e}")
