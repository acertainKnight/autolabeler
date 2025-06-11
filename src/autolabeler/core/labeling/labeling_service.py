"""Labeling service with clear separation of concerns."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import deque
import json

import pandas as pd
from jinja2 import Template
from langchain_core.language_models import BaseChatModel
from loguru import logger

from ...config import Settings
from ...models import LabelResponse
from ..base import BatchProcessor, ConfigurableComponent, ProgressTracker
from ..configs import BatchConfig, LabelingConfig
from ..knowledge import KnowledgeStore, PromptManager
from ..llm_providers import get_llm_client


class LabelingService(ConfigurableComponent, ProgressTracker, BatchProcessor):
    """
    Core service for all text labeling operations.

    This service orchestrates knowledge retrieval (RAG), prompt generation,
    LLM interaction, and result processing.
    """

    def __init__(
        self,
        dataset_name: str,
        settings: Settings,
        config: LabelingConfig | None = None,
    ):
        """Initialize the labeling service."""
        ConfigurableComponent.__init__(
            self, "labeling_service", dataset_name, settings
        )
        ProgressTracker.__init__(self, f"{dataset_name}_labeling_progress.json")

        self.config = config or LabelingConfig()
        self.settings = settings
        self.client_cache: dict[str, BaseChatModel] = {}

        # Initialize core components
        self.knowledge_store = KnowledgeStore(
            dataset_name, settings=self.settings
        )
        self.prompt_manager = PromptManager(dataset_name)

        # Load default prompt template
        self.default_template_path = (
            Path(__file__).parent.parent.parent / "templates" / "label_prompt.j2"
        )
        self.template = self._load_template(self.settings.template_path)

        # Debug prompt storage - stores last 10 rendered prompts
        self._debug_prompts: deque[dict[str, Any]] = deque(maxlen=10)
        self._debug_prompts_file = self.storage_path / "debug_prompts.json"

        logger.info(f"LabelingService for '{dataset_name}' initialized.")

    def _get_client_for_config(self, config: LabelingConfig) -> BaseChatModel:
        """Get or create a cached LLM client for a specific configuration."""
        config_key = config.model_dump_json()
        if config_key not in self.client_cache:
            logger.info(f"Creating new LLM client for config: {config_key}")
            self.client_cache[config_key] = get_llm_client(self.settings, config)
        return self.client_cache[config_key]

    def _load_template(self, template_path: Path | None) -> Template:
        """Load a Jinja2 template from a given path or use the default."""
        path = template_path if template_path and template_path.exists() else self.default_template_path
        if not path.exists():
            raise FileNotFoundError(f"Template file not found at {path}")
        return Template(path.read_text())

    def _get_progress_path(self) -> Path:
        """Get the path to the progress file."""
        return self.storage_path / self._progress_file

    def _prepare_prompt(
        self,
        text: str,
        config: LabelingConfig,
        template: Template,
        ruleset: dict[str, Any] | None,
    ) -> tuple[str, str]:
        """Prepares a prompt for labeling and returns the rendered prompt and a prompt_id."""
        examples = []
        if config.use_rag:
            k = config.k_examples or self.settings.max_examples_per_query
            source_filter = "human" if config.prefer_human_examples else None
            examples = self.knowledge_store.find_similar_examples(
                text, k=k, source_filter=source_filter
            )

            # Debug logging for RAG retrieval
            if examples:
                logger.debug(f"RAG retrieval for '{text[:50]}...': Found {len(examples)} examples")
                for i, ex in enumerate(examples):
                    similarity = ex.get("similarity_score", 0.0)
                    example_text = ex.get("text", "")[:30]
                    logger.debug(f"  Example {i+1}: '{example_text}...' (similarity: {similarity:.3f})")
            else:
                logger.warning(f"No RAG examples found for text: '{text[:50]}...'")

        # Convert examples to the format expected by the template
        template_examples = []
        for ex in examples:
            template_ex = {
                "page_content": ex.get("text", ""),
                "metadata": {
                    "label": ex.get("label", ""),
                    "source": ex.get("source", ""),
                    "confidence": ex.get("confidence", 1.0),
                    **ex.get("metadata", {}),
                },
            }
            template_examples.append(template_ex)

        # Render prompt, now including the ruleset if provided
        rendered_prompt = template.render(
            text=text,
            examples=template_examples,
            ruleset=ruleset,
        )

        # Track prompt
        prompt_id = self.prompt_manager.store_prompt(
            prompt_text=rendered_prompt,
            template_source=str(template.filename),
            model_name=config.model_name or self.settings.llm_model,
            examples_used=examples,
        )

        # Store debug prompt with RAG examples
        self._store_debug_prompt({
            "prompt_id": prompt_id,
            "text": text,
            "rendered_prompt": rendered_prompt,
            "template_source": str(template.filename),
            "examples_count": len(examples),
            "examples_retrieved": [
                {
                    "text": ex.get("text", "")[:100],  # First 100 chars
                    "label": ex.get("label", ""),
                    "similarity_score": ex.get("similarity_score", 0.0)
                }
                for ex in examples
            ],
            "ruleset": ruleset,
            "timestamp": pd.Timestamp.now().isoformat(),
        })

        return rendered_prompt, prompt_id

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
            "timestamp": pd.Timestamp.now().isoformat(),
            "prompts_count": len(self._debug_prompts),
            "prompts": list(self._debug_prompts),
        }

        with open(output_path, "w") as f:
            json.dump(debug_data, f, indent=2)

        logger.info(f"Saved {len(self._debug_prompts)} debug prompts to {output_path}")
        return output_path

    def analyze_rag_diversity(self) -> dict[str, Any]:
        """
        Analyze the diversity of RAG examples being retrieved across different queries.

        Returns:
            dict[str, Any]: Analysis results including duplicate statistics and diversity metrics.

        Example:
            >>> analysis = labeler.analyze_rag_diversity()
            >>> print(f"Same examples returned {analysis['repeated_example_percentage']:.1f}% of the time")
        """
        if not self._debug_prompts:
            return {
                "error": "No debug prompts available for analysis",
                "total_queries": 0
            }

        all_examples = []
        query_examples = []

        for prompt_info in self._debug_prompts:
            if "examples_retrieved" in prompt_info:
                examples = prompt_info["examples_retrieved"]
                query_text = prompt_info.get("text", "")[:50]

                query_examples.append({
                    "query": query_text,
                    "example_count": len(examples),
                    "examples": [ex.get("text", "") for ex in examples]
                })

                for ex in examples:
                    all_examples.append(ex.get("text", ""))

        if not all_examples:
            return {
                "error": "No RAG examples found in debug prompts",
                "total_queries": len(self._debug_prompts)
            }

        # Calculate diversity metrics
        unique_examples = set(all_examples)
        total_retrievals = len(all_examples)
        unique_count = len(unique_examples)

        # Find most frequently retrieved examples
        from collections import Counter
        example_counts = Counter(all_examples)
        most_common = example_counts.most_common(5)

        # Check if the same examples are being returned for different queries
        repeated_examples = 0
        for query_info in query_examples:
            for other_query in query_examples:
                if query_info["query"] != other_query["query"]:
                    overlap = set(query_info["examples"]) & set(other_query["examples"])
                    repeated_examples += len(overlap)

        # Calculate percentage of queries that have identical example sets
        identical_sets = 0
        total_comparisons = 0
        for i, query1 in enumerate(query_examples):
            for j, query2 in enumerate(query_examples[i+1:], i+1):
                total_comparisons += 1
                if set(query1["examples"]) == set(query2["examples"]):
                    identical_sets += 1

        return {
            "total_queries": len(query_examples),
            "total_example_retrievals": total_retrievals,
            "unique_examples_retrieved": unique_count,
            "diversity_ratio": unique_count / total_retrievals if total_retrievals > 0 else 0,
            "repeated_example_count": repeated_examples,
            "repeated_example_percentage": (repeated_examples / total_retrievals * 100) if total_retrievals > 0 else 0,
            "identical_example_sets": identical_sets,
            "identical_sets_percentage": (identical_sets / total_comparisons * 100) if total_comparisons > 0 else 0,
            "most_common_examples": [
                {
                    "text": text[:100],
                    "retrieval_count": count,
                    "percentage": (count / total_retrievals * 100) if total_retrievals > 0 else 0
                }
                for text, count in most_common
            ],
            "knowledge_base_stats": self.knowledge_store.get_statistics()
        }

    def check_rag_issues(self) -> dict[str, Any]:
        """
        Check for common RAG issues that could cause repetitive examples.

        Returns:
            dict[str, Any]: Potential issues and recommendations.
        """
        issues = []
        recommendations = []

        # Check knowledge base size
        kb_stats = self.knowledge_store.get_statistics()
        total_examples = kb_stats.get("total_examples", 0)

        if total_examples == 0:
            issues.append("Knowledge base is empty - no examples to retrieve")
            recommendations.append("Add labeled training data to the knowledge base")
        elif total_examples < 10:
            issues.append(f"Very small knowledge base ({total_examples} examples)")
            recommendations.append("Add more diverse training examples to improve RAG diversity")

        # Check for duplicates in knowledge base
        duplicate_stats = self.knowledge_store.get_duplicate_statistics()
        if duplicate_stats.get("internal_duplicate_count", 0) > 0:
            dup_pct = duplicate_stats.get("duplicate_percentage", 0)
            issues.append(f"Knowledge base contains {duplicate_stats['internal_duplicate_count']} duplicates ({dup_pct:.1f}%)")
            recommendations.append("Remove duplicates from knowledge base using store.remove_duplicates()")

        # Check retrieval settings
        max_examples = self.settings.max_examples_per_query
        if max_examples >= total_examples and total_examples > 0:
            issues.append(f"Retrieving {max_examples} examples from only {total_examples} available")
            recommendations.append("Reduce max_examples_per_query setting or add more training data")

        # Analyze RAG diversity if debug data is available
        diversity_analysis = self.analyze_rag_diversity()
        if "error" not in diversity_analysis:
            diversity_ratio = diversity_analysis.get("diversity_ratio", 0)
            if diversity_ratio < 0.5:
                issues.append(f"Low RAG diversity ratio: {diversity_ratio:.2f}")
                recommendations.append("Check embedding model quality and add more diverse examples")

            identical_pct = diversity_analysis.get("identical_sets_percentage", 0)
            if identical_pct > 50:
                issues.append(f"Same examples returned for {identical_pct:.1f}% of query pairs")
                recommendations.append("Review similarity threshold and embedding model configuration")

        return {
            "issues_found": len(issues),
            "issues": issues,
            "recommendations": recommendations,
            "knowledge_base_stats": kb_stats,
            "duplicate_stats": duplicate_stats,
            "retrieval_settings": {
                "max_examples_per_query": max_examples,
                "use_rag": self.config.use_rag,
                "prefer_human_examples": self.config.prefer_human_examples,
                "embedding_model": self.settings.embedding_model
            }
        }

    def label_text(
        self,
        text: str,
        config: LabelingConfig | None = None,
        template_path: Path | None = None,
        ruleset: dict[str, Any] | None = None,
    ) -> LabelResponse:
        """Label a single text with the configured settings."""
        config = config or self.config
        template = self._load_template(template_path) if template_path else self.template
        llm_client = self._get_client_for_config(config)

        rendered_prompt, prompt_id = self._prepare_prompt(
            text, config, template, ruleset
        )

        try:
            # Get structured prediction with function_calling for OpenRouter compatibility
            structured_llm = llm_client.with_structured_output(LabelResponse, method="function_calling")
            response = structured_llm.invoke(rendered_prompt)

            self.prompt_manager.update_result(
                prompt_id, successful=True, confidence=response.confidence
            )
            return response

        except Exception as e:
            self.prompt_manager.update_result(prompt_id, successful=False)
            logger.error(f"Failed to label text: {e}")
            raise

    async def _aprepare_prompt(
        self,
        text: str,
        config: LabelingConfig,
        template: Template,
        ruleset: dict[str, Any] | None,
    ) -> tuple[str, str]:
        """Prepares a prompt for labeling and returns the rendered prompt and a prompt_id."""
        examples = []
        if config.use_rag:
            k = config.k_examples or self.settings.max_examples_per_query
            source_filter = "human" if config.prefer_human_examples else None
            # Assuming find_similar_examples has an async version or is thread-safe
            examples = await self.knowledge_store.afind_similar_examples(
                text, k=k, source_filter=source_filter
            )

            # Debug logging for RAG retrieval (async version)
            if examples:
                logger.debug(f"Async RAG retrieval for '{text[:50]}...': Found {len(examples)} examples")
                for i, ex in enumerate(examples):
                    similarity = ex.get("similarity_score", 0.0)
                    example_text = ex.get("text", "")[:30]
                    logger.debug(f"  Example {i+1}: '{example_text}...' (similarity: {similarity:.3f})")
            else:
                logger.warning(f"No RAG examples found for text (async): '{text[:50]}...'")

        template_examples = [
            {
                "page_content": ex.get("text", ""),
                "metadata": {
                    "label": ex.get("label", ""),
                    "source": ex.get("source", ""),
                    "confidence": ex.get("confidence", 1.0),
                    **ex.get("metadata", {}),
                },
            }
            for ex in examples
        ]

        rendered_prompt = template.render(
            text=text,
            examples=template_examples,
            ruleset=ruleset,
        )

        prompt_id = self.prompt_manager.store_prompt(
            prompt_text=rendered_prompt,
            template_source=str(template.filename),
            model_name=config.model_name or self.settings.llm_model,
            examples_used=examples,
        )

        # Store debug prompt with RAG examples (async version)
        self._store_debug_prompt({
            "prompt_id": prompt_id,
            "text": text,
            "rendered_prompt": rendered_prompt,
            "template_source": str(template.filename),
            "examples_count": len(examples),
            "examples_retrieved": [
                {
                    "text": ex.get("text", "")[:100],  # First 100 chars
                    "label": ex.get("label", ""),
                    "similarity_score": ex.get("similarity_score", 0.0)
                }
                for ex in examples
            ],
            "ruleset": ruleset,
            "timestamp": pd.Timestamp.now().isoformat(),
            "async": True,
        })

        return rendered_prompt, prompt_id

    async def alabel_text(
        self,
        text: str,
        config: LabelingConfig | None = None,
        template_path: Path | None = None,
        ruleset: dict[str, Any] | None = None,
    ) -> LabelResponse:
        """Asynchronously label a single text with the configured settings."""
        config = config or self.config
        template = self._load_template(template_path) if template_path else self.template
        llm_client = self._get_client_for_config(config)

        rendered_prompt, prompt_id = await self._aprepare_prompt(
            text, config, template, ruleset
        )

        try:
            # Get structured prediction with function_calling for OpenRouter compatibility
            structured_llm = llm_client.with_structured_output(LabelResponse, method="function_calling")
            response = await structured_llm.ainvoke(rendered_prompt)

            self.prompt_manager.update_result(
                prompt_id, successful=True, confidence=response.confidence
            )
            return response

        except Exception as e:
            self.prompt_manager.update_result(prompt_id, successful=False)
            logger.error(f"Failed to label text asynchronously: {e}")
            raise

    def label_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str = "predicted_label",
        config: LabelingConfig | None = None,
        batch_config: BatchConfig | None = None,
        template_path: Path | None = None,
        ruleset: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Label an entire dataset with batch processing support."""
        config = config or self.config
        batch_config = batch_config or BatchConfig()

        items = df.to_dict("records")

        def process_func(batch: list[dict]) -> list[dict]:
            template = self._load_template(template_path) if template_path else self.template
            llm_client = self._get_client_for_config(config)
            # Use function_calling method for OpenRouter compatibility
            structured_llm = llm_client.with_structured_output(LabelResponse, method="function_calling")

            # Step 1: Prepare prompts for all items in batch
            prompts_with_metadata = []
            for item in batch:
                try:
                    rendered_prompt, prompt_id = self._prepare_prompt(
                        item[text_column], config, template, ruleset
                    )
                    prompts_with_metadata.append(
                        {"prompt": rendered_prompt, "id": prompt_id, "item": item, "error": None}
                    )
                except Exception as e:
                    logger.error(f"Failed to prepare prompt for item: {item}. Error: {e}")
                    prompts_with_metadata.append({"prompt": None, "id": None, "item": item, "error": e})

            # Step 2: Batch call LLM for valid prompts
            prompts_to_run = [p["prompt"] for p in prompts_with_metadata if p["error"] is None]

            llm_responses = []
            if prompts_to_run:
                llm_responses = structured_llm.batch(prompts_to_run, config={"return_exceptions": True})

            # Step 3: Combine results
            results = []
            llm_response_iter = iter(llm_responses)
            for p_meta in prompts_with_metadata:
                item = p_meta["item"]
                if p_meta["error"] is not None:
                    results.append({**item, label_column: None, f"{label_column}_confidence": 0.0})
                    continue

                prompt_id = p_meta["id"]
                response = next(llm_response_iter)

                if isinstance(response, LabelResponse):
                    self.prompt_manager.update_result(
                        prompt_id, successful=True, confidence=response.confidence
                    )
                    results.append(
                        {
                            **item,
                            label_column: response.label,
                            f"{label_column}_confidence": response.confidence,
                        }
                    )
                else:  # Exception from LLM
                    self.prompt_manager.update_result(prompt_id, successful=False)
                    logger.error(f"Failed to label text '{item[text_column][:50]}...': {response}")
                    results.append({**item, label_column: None, f"{label_column}_confidence": 0.0})

            return results

        all_results = self.process_in_batches(
            items=items,
            batch_size=batch_config.batch_size,
            process_func=process_func,
            desc="Labeling dataset",
            resume_key=f"{self.dataset_name}_labeling" if batch_config.resume else None,
        )

        results_df = pd.DataFrame([res for res in all_results if res is not None])

        if config.save_to_knowledge_base and config.confidence_threshold > 0:
            confidence_col = f"{label_column}_confidence"
            if confidence_col in results_df.columns:
                high_conf_df = results_df[results_df[confidence_col] >= config.confidence_threshold]
                if not high_conf_df.empty:
                    self.knowledge_store.add_examples(
                        high_conf_df, text_column, label_column, source="model"
                    )

        return results_df

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the labeling service."""
        stats = {
            "dataset_name": self.dataset_name,
            "knowledge_base_stats": self.knowledge_store.get_stats(),
            "prompt_analytics": self.prompt_manager.get_analytics(),
            "progress_info": self.get_progress_info(),
        }

        return stats
