from __future__ import annotations

import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template
from langchain.schema import Document
from loguru import logger

from .config import Settings
from .corporate import CorporateOpenAIClient
from .knowledge_base import KnowledgeBase
from .models import SyntheticBatch, SyntheticExample
from .openrouter import OpenRouterClient
from .prompt_store import PromptStore


class SyntheticDataGenerator:
    """
    Framework for generating synthetic labeled examples based on existing data patterns.

    This class provides multiple strategies for synthetic data generation including
    paraphrasing, interpolation, extrapolation, and style variation. It integrates
    with the AutoLabeler ecosystem for seamless knowledge base updates.

    Args:
        dataset_name (str): Name of the dataset (used for knowledge base and prompt tracking).
        settings (Settings): Application settings containing LLM and embedding configs.
        knowledge_base (KnowledgeBase | None): Existing knowledge base to source patterns from.

    Example:
        >>> settings = Settings(openrouter_api_key="your-key")
        >>> generator = SyntheticDataGenerator("sentiment_analysis", settings)
        >>> synthetic_examples = generator.generate_examples_for_label(
        ...     "positive", num_examples=5, strategy="diverse_paraphrase"
        ... )
    """

    def __init__(
        self,
        dataset_name: str,
        settings: Settings,
        knowledge_base: KnowledgeBase | None = None
    ) -> None:
        self.dataset_name = dataset_name
        self.settings = settings

        # Use provided knowledge base or create new one
        self.knowledge_base = knowledge_base or KnowledgeBase(dataset_name, settings)

        # Initialize prompt store for tracking synthetic generation prompts
        self.prompt_store = PromptStore(f"{dataset_name}_synthetic")

        # Initialize LLM client based on configuration
        if settings.corporate_base_url:
            base_llm = CorporateOpenAIClient(
                api_key=settings.corporate_api_key,
                base_url=settings.corporate_base_url,
                model=settings.corporate_model,
            )
        else:
            base_llm = OpenRouterClient(
                api_key=settings.openrouter_api_key,
                model=settings.llm_model,
            )

        # Set up structured output for single and batch generation
        self.single_llm = base_llm.with_structured_output(schema=SyntheticExample)
        self.batch_llm = base_llm.with_structured_output(schema=SyntheticBatch)

        # Load generation templates
        templates_dir = Path(__file__).parent / "templates"
        self.single_template = Template(
            (templates_dir / "synthetic_single.j2").read_text()
        )
        self.batch_template = Template(
            (templates_dir / "synthetic_batch.j2").read_text()
        )

        # Track model configuration for provenance
        self.model_info = {
            "model": settings.corporate_model if settings.corporate_base_url else settings.llm_model,
            "provider": "corporate" if settings.corporate_base_url else "openrouter",
            "timestamp": datetime.now().isoformat(),
            "generation_type": "synthetic",
        }

        logger.info(f"Initialized SyntheticDataGenerator for dataset: {dataset_name}")

    def get_available_labels(self) -> list[str]:
        """
        Get all unique labels available in the knowledge base.

        Returns:
            list[str]: List of unique labels that can be used for synthetic generation.

        Example:
            >>> labels = generator.get_available_labels()
            >>> print(f"Available labels: {labels}")
        """
        stats = self.knowledge_base.get_stats()
        return list(stats.get("label_distribution", {}).keys())

    def get_label_distribution(self) -> dict[str, int]:
        """
        Get the current distribution of labels in the knowledge base.

        Returns:
            dict[str, int]: Mapping of labels to their counts.

        Example:
            >>> distribution = generator.get_label_distribution()
            >>> print(f"Positive examples: {distribution.get('positive', 0)}")
        """
        stats = self.knowledge_base.get_stats()
        return stats.get("label_distribution", {})

    def analyze_class_imbalance(self) -> dict[str, Any]:
        """
        Analyze class imbalance and suggest synthetic generation targets.

        Returns:
            dict: Analysis including imbalance metrics and generation recommendations.

        Example:
            >>> analysis = generator.analyze_class_imbalance()
            >>> for label in analysis["underrepresented_labels"]:
            ...     print(f"Consider generating more examples for: {label}")
        """
        distribution = self.get_label_distribution()
        if not distribution:
            return {"error": "No labeled data available"}

        total_examples = sum(distribution.values())
        expected_per_class = total_examples / len(distribution)

        analysis = {
            "total_examples": total_examples,
            "num_classes": len(distribution),
            "expected_per_class": expected_per_class,
            "distribution": distribution,
            "underrepresented_labels": [],
            "overrepresented_labels": [],
            "imbalance_ratio": max(distribution.values()) / min(distribution.values()),
        }

        for label, count in distribution.items():
            ratio = count / expected_per_class
            if ratio < 0.8:  # Less than 80% of expected
                analysis["underrepresented_labels"].append({
                    "label": label,
                    "current_count": count,
                    "suggested_additions": int(expected_per_class - count)
                })
            elif ratio > 1.2:  # More than 120% of expected
                analysis["overrepresented_labels"].append({
                    "label": label,
                    "current_count": count,
                    "excess_count": int(count - expected_per_class)
                })

        return analysis

    def get_source_examples_for_label(
        self,
        target_label: str,
        max_examples: int = 10,
        prefer_human_labeled: bool = True
    ) -> list[Document]:
        """
        Get source examples for a specific label to guide synthetic generation.

        Args:
            target_label (str): The label to get examples for.
            max_examples (int): Maximum number of source examples to retrieve.
            prefer_human_labeled (bool): Whether to prefer human-labeled examples.

        Returns:
            list[Document]: List of example documents with the target label.

        Example:
            >>> examples = generator.get_source_examples_for_label("positive", max_examples=5)
            >>> print(f"Found {len(examples)} source examples")
        """
        # Get examples with the target label
        if prefer_human_labeled:
            # Try human examples first
            examples = self.knowledge_base.get_examples_by_label(
                target_label, max_examples, filter_source="human"
            )
            # Supplement with model examples if needed
            if len(examples) < max_examples:
                remaining = max_examples - len(examples)
                model_examples = self.knowledge_base.get_examples_by_label(
                    target_label, remaining, filter_source="model"
                )
                examples.extend(model_examples)
        else:
            examples = self.knowledge_base.get_examples_by_label(
                target_label, max_examples
            )

        if not examples:
            logger.warning(f"No source examples found for label: {target_label}")

        return examples

    def generate_single_example(
        self,
        target_label: str,
        strategy: str = "paraphrase",
        source_examples: list[Document] | None = None,
        max_source_examples: int = 5
    ) -> SyntheticExample:
        """
        Generate a single synthetic example for the specified label.

        Args:
            target_label (str): The target label for the synthetic example.
            strategy (str): Generation strategy ("paraphrase", "interpolate", "extrapolate", "transform").
            source_examples (list[Document] | None): Source examples to base generation on.
            max_source_examples (int): Maximum source examples to use if not provided.

        Returns:
            SyntheticExample: Generated synthetic example with metadata.

        Example:
            >>> example = generator.generate_single_example(
            ...     "positive", strategy="paraphrase"
            ... )
            >>> print(f"Generated: {example.text}")
            >>> print(f"Confidence: {example.confidence}")
        """
        if source_examples is None:
            source_examples = self.get_source_examples_for_label(
                target_label, max_source_examples
            )

        if not source_examples:
            raise ValueError(f"No source examples available for label: {target_label}")

        # Render the prompt
        rendered_prompt = self.single_template.render(
            source_examples=source_examples,
            target_label=target_label,
            strategy=strategy
        )

        # Track the prompt
        template_source = "synthetic_single.j2"
        variables = {
            "target_label": target_label,
            "strategy": strategy,
            "num_source_examples": len(source_examples)
        }
        examples_data = [
            {
                "text": ex.page_content,
                "label": ex.metadata.get("label"),
                "source": ex.metadata.get("source", "unknown")
            }
            for ex in source_examples
        ]

        prompt_id = self.prompt_store.store_prompt(
            prompt_text=rendered_prompt,
            template_source=template_source,
            variables=variables,
            model_name=self.model_info.get("model"),
            examples_used=examples_data,
            tags=["synthetic_generation", "single_example", strategy]
        )

        try:
            result = self.single_llm.invoke(rendered_prompt)

            # Add generation metadata
            if result.generation_metadata is None:
                result.generation_metadata = {}

            result.generation_metadata.update({
                "generation_strategy": strategy,
                "source_examples_count": len(source_examples),
                "target_label": target_label,
                "prompt_id": prompt_id,
                "model_info": self.model_info,
                "generation_timestamp": datetime.now().isoformat(),
            })

            # Update prompt result statistics
            self.prompt_store.update_prompt_result(
                prompt_id=prompt_id,
                successful=True,
                confidence=result.confidence
            )

            logger.debug(f"Generated synthetic example for {target_label} with confidence {result.confidence}")
            return result

        except Exception as e:
            # Update prompt result statistics for failure
            self.prompt_store.update_prompt_result(
                prompt_id=prompt_id,
                successful=False
            )
            logger.error(f"Failed to generate synthetic example: {e}")
            raise

    def generate_batch_examples(
        self,
        target_label: str,
        num_examples: int = 5,
        strategy: str = "diverse_paraphrase",
        diversity_focus: str = "high",
        source_examples: list[Document] | None = None,
        max_source_examples: int = 10,
        batch_constraints: list[str] | None = None
    ) -> SyntheticBatch:
        """
        Generate a batch of diverse synthetic examples for the specified label.

        Args:
            target_label (str): The target label for synthetic examples.
            num_examples (int): Number of examples to generate in the batch.
            strategy (str): Generation strategy for the batch.
            diversity_focus (str): Level of diversity emphasis ("low", "medium", "high").
            source_examples (list[Document] | None): Source examples to base generation on.
            max_source_examples (int): Maximum source examples to use if not provided.
            batch_constraints (list[str] | None): Additional constraints for the batch.

        Returns:
            SyntheticBatch: Batch of generated synthetic examples with metadata.

        Example:
            >>> batch = generator.generate_batch_examples(
            ...     "negative", num_examples=3, strategy="pattern_expansion"
            ... )
            >>> for example in batch.examples:
            ...     print(f"Text: {example.text}")
        """
        if source_examples is None:
            source_examples = self.get_source_examples_for_label(
                target_label, max_source_examples
            )

        if not source_examples:
            raise ValueError(f"No source examples available for label: {target_label}")

        # Render the batch prompt
        rendered_prompt = self.batch_template.render(
            source_examples=source_examples,
            target_label=target_label,
            num_examples=num_examples,
            strategy=strategy,
            diversity_focus=diversity_focus,
            batch_constraints=batch_constraints
        )

        # Track the prompt
        template_source = "synthetic_batch.j2"
        variables = {
            "target_label": target_label,
            "num_examples": num_examples,
            "strategy": strategy,
            "diversity_focus": diversity_focus,
            "num_source_examples": len(source_examples)
        }
        examples_data = [
            {
                "text": ex.page_content,
                "label": ex.metadata.get("label"),
                "source": ex.metadata.get("source", "unknown")
            }
            for ex in source_examples
        ]

        prompt_id = self.prompt_store.store_prompt(
            prompt_text=rendered_prompt,
            template_source=template_source,
            variables=variables,
            model_name=self.model_info.get("model"),
            examples_used=examples_data,
            tags=["synthetic_generation", "batch_generation", strategy]
        )

        try:
            result = self.batch_llm.invoke(rendered_prompt)

            # Add batch metadata
            if result.batch_metadata is None:
                result.batch_metadata = {}

            result.batch_metadata.update({
                "generation_strategy": strategy,
                "diversity_focus": diversity_focus,
                "source_examples_count": len(source_examples),
                "target_label": target_label,
                "prompt_id": prompt_id,
                "model_info": self.model_info,
                "generation_timestamp": datetime.now().isoformat(),
                "requested_examples": num_examples,
                "actual_examples": len(result.examples),
            })

            # Add metadata to individual examples
            for i, example in enumerate(result.examples):
                if example.generation_metadata is None:
                    example.generation_metadata = {}
                example.generation_metadata.update({
                    "batch_position": i,
                    "batch_strategy": strategy,
                    "prompt_id": prompt_id,
                })

            # Calculate average confidence for prompt tracking
            avg_confidence = (
                sum(ex.confidence for ex in result.examples) / len(result.examples)
                if result.examples else 0.0
            )

            # Update prompt result statistics
            self.prompt_store.update_prompt_result(
                prompt_id=prompt_id,
                successful=True,
                confidence=avg_confidence
            )

            logger.info(f"Generated batch of {len(result.examples)} synthetic examples for {target_label}")
            return result

        except Exception as e:
            # Update prompt result statistics for failure
            self.prompt_store.update_prompt_result(
                prompt_id=prompt_id,
                successful=False
            )
            logger.error(f"Failed to generate synthetic batch: {e}")
            raise

    def generate_examples_for_label(
        self,
        target_label: str,
        num_examples: int = 5,
        strategy: str = "mixed",
        use_batch_generation: bool = True,
        add_to_knowledge_base: bool = True,
        confidence_threshold: float = 0.7
    ) -> list[SyntheticExample]:
        """
        Generate multiple synthetic examples for a label using optimal strategies.

        Args:
            target_label (str): The target label for synthetic examples.
            num_examples (int): Number of examples to generate.
            strategy (str): Generation strategy or "mixed" for automatic selection.
            use_batch_generation (bool): Whether to use batch generation for efficiency.
            add_to_knowledge_base (bool): Whether to add generated examples to knowledge base.
            confidence_threshold (float): Minimum confidence to add to knowledge base.

        Returns:
            list[SyntheticExample]: List of generated synthetic examples.

        Example:
            >>> examples = generator.generate_examples_for_label(
            ...     "neutral", num_examples=10, strategy="mixed"
            ... )
            >>> high_quality = [ex for ex in examples if ex.confidence > 0.8]
        """
        # Get source examples
        source_examples = self.get_source_examples_for_label(target_label)

        if not source_examples:
            raise ValueError(f"No source examples available for label: {target_label}")

        # Select strategy if mixed
        if strategy == "mixed":
            available_strategies = [
                "diverse_paraphrase", "pattern_expansion", "style_variation"
            ]
            strategy = random.choice(available_strategies)

        generated_examples = []

        if use_batch_generation and num_examples > 1:
            # Use batch generation for efficiency
            batch_result = self.generate_batch_examples(
                target_label=target_label,
                num_examples=num_examples,
                strategy=strategy,
                source_examples=source_examples
            )
            generated_examples = batch_result.examples
        else:
            # Generate examples individually
            for i in range(num_examples):
                try:
                    example = self.generate_single_example(
                        target_label=target_label,
                        strategy=strategy,
                        source_examples=source_examples
                    )
                    generated_examples.append(example)
                except Exception as e:
                    logger.warning(f"Failed to generate example {i+1}: {e}")

        # Add high-confidence examples to knowledge base
        if add_to_knowledge_base:
            high_confidence_examples = [
                ex for ex in generated_examples
                if ex.confidence >= confidence_threshold
            ]

            if high_confidence_examples:
                # Convert to DataFrame for knowledge base
                df_data = []
                for example in high_confidence_examples:
                    df_data.append({
                        "text": example.text,
                        "label": example.label,
                        "confidence": example.confidence,
                        "reasoning": example.reasoning,
                        "generation_metadata": example.generation_metadata,
                    })

                df = pd.DataFrame(df_data)
                self.knowledge_base.add_labeled_data(
                    df, "text", "label", source="synthetic"
                )

                logger.info(f"Added {len(high_confidence_examples)} high-confidence synthetic examples to knowledge base")

        return generated_examples

    def balance_dataset(
        self,
        target_balance: dict[str, int] | str = "equal",
        max_synthetic_per_label: int = 50,
        confidence_threshold: float = 0.7
    ) -> dict[str, list[SyntheticExample]]:
        """
        Generate synthetic examples to balance the dataset according to target distribution.

        Args:
            target_balance (dict[str, int] | str): Target counts per label or "equal" for uniform.
            max_synthetic_per_label (int): Maximum synthetic examples to generate per label.
            confidence_threshold (float): Minimum confidence to include in balancing.

        Returns:
            dict[str, list[SyntheticExample]]: Generated examples organized by label.

        Example:
            >>> # Balance to equal distribution
            >>> balanced = generator.balance_dataset("equal")
            >>> # Or specify exact targets
            >>> balanced = generator.balance_dataset({"positive": 100, "negative": 100})
        """
        current_distribution = self.get_label_distribution()
        if not current_distribution:
            raise ValueError("No labeled data available for balancing")

        # Calculate target distribution
        if target_balance == "equal":
            max_count = max(current_distribution.values())
            target_counts = {label: max_count for label in current_distribution}
        else:
            target_counts = target_balance

        generated_by_label = {}

        for label, target_count in target_counts.items():
            current_count = current_distribution.get(label, 0)
            needed = target_count - current_count

            if needed <= 0:
                logger.info(f"Label '{label}' already has sufficient examples ({current_count} >= {target_count})")
                generated_by_label[label] = []
                continue

            # Limit generation to maximum
            to_generate = min(needed, max_synthetic_per_label)

            logger.info(f"Generating {to_generate} synthetic examples for label '{label}' (current: {current_count}, target: {target_count})")

            try:
                examples = self.generate_examples_for_label(
                    target_label=label,
                    num_examples=to_generate,
                    strategy="mixed",
                    add_to_knowledge_base=True,
                    confidence_threshold=confidence_threshold
                )
                generated_by_label[label] = examples

            except Exception as e:
                logger.error(f"Failed to generate examples for label '{label}': {e}")
                generated_by_label[label] = []

        # Log final statistics
        total_generated = sum(len(examples) for examples in generated_by_label.values())
        logger.info(f"Dataset balancing complete. Generated {total_generated} total synthetic examples.")

        return generated_by_label

    def export_synthetic_examples(
        self,
        output_path: Path,
        include_metadata: bool = True,
        filter_confidence: float | None = None
    ) -> None:
        """
        Export all synthetic examples from the knowledge base to CSV.

        Args:
            output_path (Path): Path to save the exported synthetic examples.
            include_metadata (bool): Whether to include generation metadata columns.
            filter_confidence (float | None): Minimum confidence to include.

        Example:
            >>> generator.export_synthetic_examples(
            ...     Path("synthetic_data.csv"), filter_confidence=0.8
            ... )
        """
        # Get all synthetic examples from knowledge base
        stats = self.knowledge_base.get_stats()
        if "synthetic" not in stats.get("sources", {}):
            logger.warning("No synthetic examples found in knowledge base")
            return

        # Export with filtering
        self.knowledge_base.export_synthetic_data(
            output_path, include_metadata, filter_confidence
        )
        logger.info(f"Exported synthetic examples to {output_path}")

    def get_generation_analytics(self) -> dict[str, Any]:
        """
        Get analytics about synthetic data generation performance.

        Returns:
            dict: Analytics including generation success rates and quality metrics.

        Example:
            >>> analytics = generator.get_generation_analytics()
            >>> print(f"Success rate: {analytics['success_rate']:.2%}")
        """
        prompt_analytics = self.prompt_store.get_prompt_analytics()

        # Filter to synthetic generation prompts
        synthetic_prompts = [
            p for p in self.prompt_store.prompts.values()
            if "synthetic_generation" in p.tags
        ]

        if not synthetic_prompts:
            return {"message": "No synthetic generation prompts found"}

        total_prompts = len(synthetic_prompts)
        successful_prompts = sum(1 for p in synthetic_prompts if p.successful_predictions > 0)
        total_confidence = sum(p.avg_confidence or 0 for p in synthetic_prompts if p.avg_confidence)
        avg_confidence = total_confidence / len([p for p in synthetic_prompts if p.avg_confidence])

        # Strategy breakdown
        strategy_counts = Counter()
        for prompt in synthetic_prompts:
            for tag in prompt.tags:
                if tag not in ["synthetic_generation", "single_example", "batch_generation"]:
                    strategy_counts[tag] += 1

        return {
            "total_generation_prompts": total_prompts,
            "successful_prompts": successful_prompts,
            "success_rate": successful_prompts / total_prompts if total_prompts > 0 else 0,
            "average_confidence": avg_confidence,
            "strategy_usage": dict(strategy_counts),
            "knowledge_base_stats": self.knowledge_base.get_stats(),
        }
