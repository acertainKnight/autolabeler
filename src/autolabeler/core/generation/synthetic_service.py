"""Simplified synthetic data generation service."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import deque
import json

import pandas as pd
from langchain.schema import Document
from loguru import logger

from ...config import Settings
from ...models import SyntheticBatch, SyntheticExample
from ..base import BatchProcessor, ConfigurableComponent, ProgressTracker
from ..configs import BatchConfig, GenerationConfig
from ..knowledge import KnowledgeStore
from ..llm_providers import get_llm_client

from jinja2 import Template


class SyntheticGenerationService(ConfigurableComponent, ProgressTracker, BatchProcessor):
    """
    Service for generating synthetic labeled examples.

    Provides multiple strategies for synthetic data generation including
    paraphrasing, interpolation, and style variation.

    Example:
        >>> config = GenerationConfig(strategy="diverse_paraphrase", num_examples=10)
        >>> service = SyntheticGenerationService("sentiment", settings)
        >>> examples = service.generate_for_label("positive", config)
    """

    def __init__(
        self,
        dataset_name: str,
        settings: Settings,
        config: GenerationConfig | None = None,
    ):
        """Initialize the synthetic generation service."""
        ConfigurableComponent.__init__(
            self, "synthetic_service", dataset_name, settings
        )
        ProgressTracker.__init__(self, f"{dataset_name}_synthetic_progress.json")
        self.config = config or GenerationConfig()
        self.settings = settings

        self.knowledge_store = KnowledgeStore(
            dataset_name, settings=settings
        )
        self.llm_client = get_llm_client(settings, self.config)

        template_path = (
            Path(__file__).parent.parent.parent / "templates" / "synthetic_single.j2"
        )
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        self.template = Template(template_path.read_text())

        # Initialize structured LLMs with function_calling for OpenRouter compatibility
        self.single_llm = self.llm_client.with_structured_output(SyntheticExample, method="function_calling")
        self.batch_llm = self.llm_client.with_structured_output(SyntheticBatch, method="function_calling")

        # Load templates
        self._load_templates()

        # Debug prompt storage - stores last 10 rendered prompts
        self._debug_prompts: deque[dict[str, Any]] = deque(maxlen=10)
        self._debug_prompts_file = self.storage_path / "debug_prompts.json"

        logger.info(f"Initialized SyntheticGenerationService for dataset: {dataset_name}")

    def _get_progress_path(self) -> Path:
        """Get the path to the progress file."""
        return self.storage_path / self._progress_file

    def _load_templates(self) -> None:
        """Load generation templates."""
        templates_dir = Path(__file__).parent.parent.parent / "templates"
        self.single_template = Template(
            (templates_dir / "synthetic_single.j2").read_text()
        )
        self.batch_template = Template(
            (templates_dir / "synthetic_batch.j2").read_text()
        )

    def get_available_labels(self) -> list[str]:
        """Get all unique labels available in the knowledge base."""
        stats = self.knowledge_store.get_stats()
        return list(stats.get("label_distribution", {}).keys())

    def analyze_class_balance(self) -> dict[str, Any]:
        """Analyze class imbalance and suggest generation targets."""
        stats = self.knowledge_store.get_stats()
        distribution = stats.get("label_distribution", {})

        if not distribution:
            return {"error": "No labeled data available"}

        total = sum(distribution.values())
        expected_per_class = total / len(distribution)

        analysis = {
            "total_examples": total,
            "num_classes": len(distribution),
            "expected_per_class": expected_per_class,
            "distribution": distribution,
            "imbalance_ratio": max(distribution.values()) / min(distribution.values()) if distribution else 0,
            "suggestions": [],
        }

        # Generate suggestions
        for label, count in distribution.items():
            ratio = count / expected_per_class
            if ratio < 0.8:
                suggested = int(expected_per_class - count)
                analysis["suggestions"].append({
                    "label": label,
                    "current": count,
                    "suggested_additions": suggested,
                    "priority": "high" if ratio < 0.5 else "medium",
                })

        return analysis

    def generate_for_label(
        self,
        target_label: str,
        num_examples: int,
        config: GenerationConfig | None = None,
    ) -> list[SyntheticExample]:
        """Generate synthetic examples for a specific label."""
        config = config or self.config
        # Use function_calling method for OpenRouter compatibility
        structured_llm = self.llm_client.with_structured_output(SyntheticBatch, method="function_calling")

        # Get examples of the target label to use as inspiration
        inspiration_examples = self.knowledge_store.get_examples_by_label(
            target_label=target_label, max_examples=5, source_filter="human"
        )

        rendered_prompt = self.batch_template.render(
            target_label=target_label,
            num_examples=num_examples,
            inspiration_examples=[ex.page_content for ex in inspiration_examples],
            strategy=config.strategy,
            diversity_focus=config.diversity_focus,
            batch_constraints=config.batch_constraints,
        )

        # Store debug prompt
        self._store_debug_prompt({
            "method": "generate_for_label",
            "target_label": target_label,
            "num_examples": num_examples,
            "rendered_prompt": rendered_prompt,
            "template": "batch_template",
            "inspiration_examples_count": len(inspiration_examples),
            "strategy": config.strategy,
            "timestamp": pd.Timestamp.now().isoformat(),
        })

        try:
            response = structured_llm.invoke(rendered_prompt)
            examples = response.examples if response else []

            # Add to knowledge base if configured
            if config.add_to_knowledge_base and examples:
                self._add_to_knowledge_base(examples)

            return examples
        except Exception as e:
            logger.error(f"Failed to generate examples for label '{target_label}': {e}")
            return []

    def balance_dataset(
        self,
        target_distribution: dict[str, int] | str = "equal",
        config: GenerationConfig | None = None,
        batch_config: BatchConfig | None = None,
    ) -> dict[str, list[SyntheticExample]]:
        """
        Generate synthetic examples to balance the dataset.

        Args:
            target_distribution: Target distribution or "equal" for balanced
            config: Generation configuration
            batch_config: Batch processing configuration

        Returns:
            Dictionary mapping labels to generated examples
        """
        config = config or GenerationConfig()
        batch_config = batch_config or BatchConfig()

        # Analyze current distribution
        analysis = self.analyze_class_balance()
        current_dist = analysis["distribution"]

        # Determine target counts
        if target_distribution == "equal":
            max_count = max(current_dist.values())
            target_dist = {label: max_count for label in current_dist}
        else:
            target_dist = target_distribution

        # Generate examples for underrepresented classes
        all_generated = {}

        generation_tasks = []
        for label, target_count in target_dist.items():
            current_count = current_dist.get(label, 0)
            needed = target_count - current_count

            if needed > 0:
                generation_tasks.append({"label": label, "needed": needed})

        def process_func(batch: list[dict]) -> list[dict]:
            results = []
            for task in batch:
                label = task["label"]
                needed = task["needed"]
                logger.info(f"Generating {needed} examples for label '{label}'")
                generated_examples = self.generate_for_label(
                    label, needed, config
                )
                all_generated[label] = generated_examples
                results.append({"label": label, "generated_count": len(generated_examples)})
            return results

        self.process_in_batches(
            items=generation_tasks,
            batch_size=batch_config.batch_size,
            process_func=process_func,
            desc="Balancing dataset",
            resume_key=f"{self.dataset_name}_balancing" if batch_config.resume else None,
        )

        return all_generated

    def _get_source_examples(
        self,
        target_label: str,
        max_examples: int = 10,
    ) -> list[Document]:
        """Get source examples for a specific label."""
        # Try human examples first
        examples = self.knowledge_store.get_examples_by_label(
            target_label, max_examples, filter_source="human"
        )

        # Supplement with model examples if needed
        if len(examples) < max_examples:
            remaining = max_examples - len(examples)
            model_examples = self.knowledge_store.get_examples_by_label(
                target_label, remaining, filter_source="model"
            )
            examples.extend(model_examples)

        return examples

    def _generate_single(
        self,
        target_label: str,
        source_examples: list[Document],
        config: GenerationConfig,
    ) -> SyntheticExample:
        """Generate a single synthetic example."""
        prompt = self.single_template.render(
            source_examples=source_examples,
            target_label=target_label,
            strategy=config.strategy,
        )

        # Store debug prompt
        self._store_debug_prompt({
            "method": "_generate_single",
            "target_label": target_label,
            "rendered_prompt": prompt,
            "template": "single_template",
            "source_examples_count": len(source_examples),
            "strategy": config.strategy,
            "timestamp": pd.Timestamp.now().isoformat(),
        })

        try:
            result = self.single_llm.invoke(prompt)

            # Add metadata
            if not result.generation_metadata:
                result.generation_metadata = {}

            result.generation_metadata.update({
                "strategy": config.strategy,
                "target_label": target_label,
                "source_count": len(source_examples),
                "model_info": self.model_info,
            })

            return result

        except Exception as e:
            logger.error(f"Failed to generate single example: {e}")
            raise

    def _generate_batch(
        self,
        target_label: str,
        source_examples: list[Document],
        config: GenerationConfig,
        batch_config: BatchConfig,
    ) -> list[SyntheticExample]:
        """Generate a batch of synthetic examples."""
        prompt = self.batch_template.render(
            source_examples=source_examples,
            target_label=target_label,
            num_examples=config.num_examples,
            strategy=config.strategy,
            diversity_focus=config.diversity_focus,
            batch_constraints=config.batch_constraints,
        )

        # Store debug prompt
        self._store_debug_prompt({
            "method": "_generate_batch",
            "target_label": target_label,
            "rendered_prompt": prompt,
            "template": "batch_template",
            "source_examples_count": len(source_examples),
            "num_examples": config.num_examples,
            "strategy": config.strategy,
            "timestamp": pd.Timestamp.now().isoformat(),
        })

        try:
            result = self.batch_llm.invoke(prompt)

            # Add metadata to each example
            for example in result.examples:
                if not example.generation_metadata:
                    example.generation_metadata = {}

                example.generation_metadata.update({
                    "strategy": config.strategy,
                    "target_label": target_label,
                    "batch_generation": True,
                    "model_info": self.model_info,
                })

            # Filter by confidence if specified
            if config.confidence_threshold > 0:
                filtered = [
                    ex for ex in result.examples
                    if ex.confidence >= config.confidence_threshold
                ]
                logger.info(
                    f"Filtered {len(result.examples) - len(filtered)} "
                    f"examples below confidence threshold {config.confidence_threshold}"
                )
                return filtered

            return result.examples

        except Exception as e:
            logger.error(f"Failed to generate batch: {e}")
            raise

    def _add_to_knowledge_base(self, examples: list[SyntheticExample]) -> None:
        """Add generated examples to the knowledge base."""
        if not examples:
            return

        # Convert to DataFrame
        data = []
        for ex in examples:
            data.append({
                "text": ex.text,
                "label": ex.label,
                "confidence": ex.confidence,
                "source": "synthetic",
            })

        df = pd.DataFrame(data)

        # Add to knowledge base
        self.knowledge_store.add_examples(
            df, "text", "label", source="synthetic"
        )

        logger.info(f"Added {len(examples)} synthetic examples to knowledge base")

    def export_generated_examples(
        self,
        output_path: Path,
        filter_confidence: float | None = None,
    ) -> None:
        """Export all generated examples to a file."""
        # Load progress data
        progress_data = self.load_progress()

        if not progress_data:
            logger.warning("No generated examples found")
            return

        # Collect all examples
        all_examples = []
        for key, entry in progress_data.items():
            if "examples" in entry.get("data", {}):
                examples = entry["data"]["examples"]

                # Filter by confidence if specified
                if filter_confidence:
                    examples = [
                        ex for ex in examples
                        if ex.get("confidence", 0) >= filter_confidence
                    ]

                all_examples.extend(examples)

        # Save to file
        df = pd.DataFrame(all_examples)

        if output_path.suffix == ".csv":
            df.to_csv(output_path, index=False)
        elif output_path.suffix == ".json":
            df.to_json(output_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported export format: {output_path.suffix}")

        logger.info(f"Exported {len(all_examples)} examples to {output_path}")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the synthetic generation service."""
        return {
            "dataset_name": self.dataset_name,
            "available_labels": self.get_available_labels(),
            "balance_analysis": self.analyze_class_balance(),
            "progress_info": self.get_progress_info(),
        }

    def _store_debug_prompt(self, debug_info: dict[str, Any]) -> None:
        """Store a rendered prompt for debugging purposes."""
        self._debug_prompts.append(debug_info)

        # Optionally save to file for persistent debugging
        try:
            with open(self._debug_prompts_file, "w") as f:
                json.dump(list(self._debug_prompts), f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save debug prompts to file: {e}")

    def get_debug_prompts(self) -> list[dict[str, Any]]:
        """Get the last 10 rendered prompts for debugging."""
        return list(self._debug_prompts)

    def save_debug_prompts(self, output_path: Path | None = None) -> Path:
        """Save debug prompts to a specific file.

        Args:
            output_path: Path to save the debug prompts. If None, uses default location.

        Returns:
            Path: The path where the debug prompts were saved.
        """
        output_path = output_path or self._debug_prompts_file

        debug_data = {
            "dataset_name": self.dataset_name,
            "service": "synthetic_generation",
            "timestamp": pd.Timestamp.now().isoformat(),
            "prompts_count": len(self._debug_prompts),
            "prompts": list(self._debug_prompts),
        }

        with open(output_path, "w") as f:
            json.dump(debug_data, f, indent=2)

        logger.info(f"Saved {len(self._debug_prompts)} debug prompts to {output_path}")
        return output_path
