from __future__ import annotations

import csv
import json
from collections.abc import Iterable
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
from .models import LabelResponse
from .openrouter import OpenRouterClient
from .prompt_store import PromptStore
from .synthetic_generator import SyntheticDataGenerator


class AutoLabeler:
    """
    Advanced autolabeling pipeline with persistent knowledge bases and incremental learning.

    This class provides a complete pipeline for automatically labeling text data using LLMs
    with structured output, persistent knowledge bases with RAG retrieval, and full provenance
    tracking for model-generated labels.

    Args:
        dataset_name (str): Unique identifier for this dataset's knowledge base.
        settings (Settings): Application settings containing LLM and embedding configs.
        template_path (Path | None): Path to Jinja2 template file. Uses default if None.

    Example:
        >>> settings = Settings(openrouter_api_key="your-key")
        >>> labeler = AutoLabeler("sentiment_analysis", settings)
        >>> # Add initial labeled data
        >>> labeler.add_training_data(labeled_df, "text", "sentiment")
        >>> # Label new data with RAG examples
        >>> result = labeler.label_text("This movie was great!")
    """

    def __init__(
        self,
        dataset_name: str,
        settings: Settings,
        template_path: Path | None = None
    ) -> None:
        self.dataset_name = dataset_name
        self.settings = settings

        # Initialize knowledge base for this dataset
        self.knowledge_base = KnowledgeBase(dataset_name, settings)

        # Initialize prompt store for tracking all prompts
        self.prompt_store = PromptStore(dataset_name)

        # Initialize synthetic data generator
        self._synthetic_generator: SyntheticDataGenerator | None = None

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

        # Use modern structured output instead of PydanticOutputParser
        self.llm = base_llm.with_structured_output(schema=LabelResponse)

        # Load prompt template
        if template_path is None:
            template_path = Path(__file__).parent / "templates" / "label_prompt.j2"
        self.template = Template(template_path.read_text())

        # Track model configuration for provenance
        self.model_info = {
            "model": settings.corporate_model if settings.corporate_base_url else settings.llm_model,
            "provider": "corporate" if settings.corporate_base_url else "openrouter",
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Initialized AutoLabeler for dataset: {dataset_name}")

    def add_training_data(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        source: str = "human"
    ) -> None:
        """
        Add labeled training data to the knowledge base.

        Args:
            df (pd.DataFrame): DataFrame containing text and labels.
            text_column (str): Name of column containing text.
            label_column (str): Name of column containing labels.
            source (str): Source of labels ("human" or "model").

        Example:
            >>> labeler.add_training_data(labeled_df, "review", "sentiment")
        """
        self.knowledge_base.add_labeled_data(df, text_column, label_column, source)
        stats = self.knowledge_base.get_stats()
        logger.info(f"Knowledge base now contains {stats.get('total_examples', 0)} examples")

    def label_text(
        self,
        text: str,
        use_rag: bool = True,
        k: int | None = None,
        prefer_human_examples: bool = True,
        context_metadata: dict[str, Any] | None = None
    ) -> LabelResponse:
        """
        Label a single text using the configured LLM with optional RAG examples.

        Args:
            text (str): Text to label.
            use_rag (bool): Whether to use RAG examples from knowledge base.
            k (int | None): Number of examples to retrieve. Uses settings default if None.
            prefer_human_examples (bool): Prefer human-labeled examples over model-generated.
            context_metadata (dict[str, Any] | None): Additional context for retrieval filtering/boosting.

        Returns:
            LabelResponse: Structured response with label, confidence, and metadata.

        Example:
            >>> context = {"domain": "product_reviews", "category": "electronics"}
            >>> result = labeler.label_text("This product is amazing!", context_metadata=context)
            >>> print(f"Label: {result.label}, Confidence: {result.confidence}")
        """
        examples = []
        if use_rag:
            if prefer_human_examples:
                # Try to get human examples first with context-aware retrieval
                if context_metadata and hasattr(self.knowledge_base, 'get_similar_examples_with_metadata_filter'):
                    examples = self.knowledge_base.get_similar_examples_with_metadata_filter(
                        text,
                        k=k,
                        filter_source="human",
                        metadata_filters={f"data_{key}": value for key, value in context_metadata.items() if key not in ['boost', 'priority']},
                        boost_metadata=context_metadata.get('boost', {})
                    )
                else:
                    examples = self.knowledge_base.get_similar_examples(
                        text, k=k, filter_source="human"
                    )

                # If not enough human examples, supplement with model examples
                if len(examples) < (k or self.settings.max_examples_per_query):
                    remaining = (k or self.settings.max_examples_per_query) - len(examples)
                    if context_metadata and hasattr(self.knowledge_base, 'get_similar_examples_with_metadata_filter'):
                        model_examples = self.knowledge_base.get_similar_examples_with_metadata_filter(
                            text,
                            k=remaining,
                            filter_source="model",
                            metadata_filters={f"data_{key}": value for key, value in context_metadata.items() if key not in ['boost', 'priority']},
                            boost_metadata=context_metadata.get('boost', {})
                        )
                    else:
                        model_examples = self.knowledge_base.get_similar_examples(
                            text, k=remaining, filter_source="model"
                        )
                    examples.extend(model_examples)
            else:
                # Get any examples regardless of source with context-aware retrieval
                if context_metadata and hasattr(self.knowledge_base, 'get_similar_examples_with_metadata_filter'):
                    examples = self.knowledge_base.get_similar_examples_with_metadata_filter(
                        text,
                        k=k,
                        metadata_filters={f"data_{key}": value for key, value in context_metadata.items() if key not in ['boost', 'priority']},
                        boost_metadata=context_metadata.get('boost', {})
                    )
                else:
                    examples = self.knowledge_base.get_similar_examples(text, k=k)

        rendered_prompt = self.template.render(
            text=text,
            examples=examples,
            context_metadata=context_metadata,
        )

        # Store the prompt for tracking
        template_source = getattr(self.template, 'name', None) or "label_prompt.j2"
        variables = {"text": text, "context_metadata": context_metadata}
        examples_data = []

        if examples:
            examples_data = [
                {
                    "text": ex.page_content,
                    "source": ex.metadata.get("source", "unknown"),
                    "label": ex.metadata.get("label"),
                    "metadata": {k: v for k, v in ex.metadata.items() if k not in ["text", "source", "label"]}
                }
                for ex in examples
            ]

        prompt_id = self.prompt_store.store_prompt(
            prompt_text=rendered_prompt,
            template_source=template_source,
            variables=variables,
            model_config_id=getattr(self, 'current_model_config_id', None),
            model_name=self.model_info.get("model"),
            examples_used=examples_data,
            tags=["single_text"] + (["context_aware"] if context_metadata else [])
        )

        try:
            result = self.llm.invoke(rendered_prompt)

            # Update prompt result statistics
            self.prompt_store.update_prompt_result(
                prompt_id=prompt_id,
                successful=True,
                confidence=result.confidence
            )

            logger.debug(f"Labeled text with confidence {result.confidence}: {result.label} (prompt: {prompt_id})")
            return result
        except Exception as e:
            # Update prompt result with error
            self.prompt_store.update_prompt_result(
                prompt_id=prompt_id,
                successful=False,
                error_message=str(e)
            )
            logger.error(f"Error labeling text (prompt: {prompt_id}): {e}")
            raise

    def label_text_with_context(
        self,
        text: str,
        context: dict[str, Any],
        use_rag: bool = True,
        k: int | None = None,
        prefer_human_examples: bool = True
    ) -> LabelResponse:
        """
        Label text with rich contextual information for improved accuracy.

        Args:
            text (str): Text to label.
            context (dict[str, Any]): Rich context including domain, category, metadata filters, etc.
            use_rag (bool): Whether to use RAG examples from knowledge base.
            k (int | None): Number of examples to retrieve.
            prefer_human_examples (bool): Prefer human-labeled examples over model-generated.

        Returns:
            LabelResponse: Structured response with label, confidence, and metadata.

        Example:
            >>> context = {
            ...     "domain": "e-commerce",
            ...     "category": "electronics",
            ...     "user_intent": "purchase_decision",
            ...     "boost": {"data_quality_score": 1.5, "data_recency": 1.2},
            ...     "background": "Customer reviewing a smartphone after 30 days of use"
            ... }
            >>> result = labeler.label_text_with_context("Amazing battery life!", context)
        """
        return self.label_text(
            text=text,
            use_rag=use_rag,
            k=k,
            prefer_human_examples=prefer_human_examples,
            context_metadata=context
        )

    def label_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str = "predicted_label",
        use_rag: bool = True,
        save_to_knowledge_base: bool = True,
        confidence_threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        Label all texts in a DataFrame column with automatic knowledge base updates.

        Args:
            df (pd.DataFrame): DataFrame containing text data to label.
            text_column (str): Name of column containing text to label.
            label_column (str): Name of column to store predicted labels.
            use_rag (bool): Whether to use RAG examples from knowledge base.
            save_to_knowledge_base (bool): Whether to add high-confidence predictions back to KB.
            confidence_threshold (float): Minimum confidence to add predictions to KB.

        Returns:
            pd.DataFrame: DataFrame with added prediction columns and metadata.

        Example:
            >>> df = pd.DataFrame({"review": ["Great!", "Terrible", "Okay"]})
            >>> labeled_df = labeler.label_dataframe(df, "review")
        """
        results = []
        high_confidence_predictions = []

        for idx, row in df.iterrows():
            query = str(row[text_column])

            try:
                result = self.label_text(query, use_rag=use_rag)

                # Create result dictionary with all original data
                result_dict = row.to_dict()
                result_dict[label_column] = result.label
                result_dict[f"{label_column}_confidence"] = result.confidence

                if result.reasoning:
                    result_dict[f"{label_column}_reasoning"] = result.reasoning

                if result.metadata:
                    for k, v in result.metadata.items():
                        result_dict[f"{label_column}_meta_{k}"] = v

                results.append(result_dict)

                # Track high-confidence predictions for knowledge base update
                if (save_to_knowledge_base and
                    result.confidence >= confidence_threshold):
                    high_confidence_predictions.append({
                        text_column: query,
                        label_column: result.label,
                        "confidence": result.confidence,
                        "reasoning": result.reasoning,
                        "row_index": idx,
                    })

                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(df)} texts")

            except Exception as e:
                logger.error(f"Failed to label row {idx}: {e}")
                # Add row with null predictions
                result_dict = row.to_dict()
                result_dict[label_column] = None
                result_dict[f"{label_column}_confidence"] = 0.0
                results.append(result_dict)

        labeled_df = pd.DataFrame(results)

        # Add high-confidence predictions to knowledge base
        if high_confidence_predictions:
            predictions_df = pd.DataFrame(high_confidence_predictions)
            self._add_predictions_to_knowledge_base(predictions_df, text_column, label_column)

        logger.info(f"Labeled {len(labeled_df)} texts. Added {len(high_confidence_predictions)} high-confidence predictions to knowledge base.")

        return labeled_df

    def _add_predictions_to_knowledge_base(
        self,
        predictions_df: pd.DataFrame,
        text_column: str,
        label_column: str
    ) -> None:
        """Add model predictions to knowledge base with full provenance."""
        if predictions_df.empty:
            return

        # Enhanced model info with current parameters
        enhanced_model_info = self.model_info.copy()
        enhanced_model_info.update({
            "prediction_session": datetime.now().isoformat(),
            "num_predictions": len(predictions_df),
        })

        # Get the current template content for provenance
        template_content = self.template.environment.get_template(
            self.template.name or "label_prompt.j2"
        ).source if hasattr(self.template, 'environment') else str(self.template)

        self.knowledge_base.add_model_labels(
            predictions_df,
            text_column,
            label_column,
            enhanced_model_info,
            template_content[:500]  # Truncate for storage
        )

    def label_csv(
        self,
        input_file: Path,
        output_file: Path,
        text_column: str,
        label_column: str = "predicted_label",
        use_rag: bool = True,
        save_to_knowledge_base: bool = True,
    ) -> None:
        """
        Label texts in a CSV file and save results with knowledge base updates.

        Args:
            input_file (Path): Path to input CSV file.
            output_file (Path): Path to output CSV file.
            text_column (str): Name of column containing text to label.
            label_column (str): Name of column to store predicted labels.
            use_rag (bool): Whether to use RAG examples from knowledge base.
            save_to_knowledge_base (bool): Whether to add predictions back to KB.

        Example:
            >>> labeler.label_csv(
            ...     Path("reviews.csv"),
            ...     Path("labeled_reviews.csv"),
            ...     "review_text"
            ... )
        """
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)

        labeled_df = self.label_dataframe(
            df, text_column, label_column, use_rag, save_to_knowledge_base
        )

        logger.info(f"Saving results to {output_file}")
        labeled_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

    def get_knowledge_base_stats(self) -> dict[str, Any]:
        """
        Get statistics about the current knowledge base.

        Returns:
            dict: Statistics including total examples, sources, and metadata.

        Example:
            >>> stats = labeler.get_knowledge_base_stats()
            >>> print(f"Human examples: {stats['sources']['human']}")
        """
        return self.knowledge_base.get_stats()

    def export_knowledge_base(self, output_path: Path) -> None:
        """
        Export knowledge base data to CSV for inspection.

        Args:
            output_path (Path): Path to save the exported data.

        Example:
            >>> labeler.export_knowledge_base(Path("exported_kb.csv"))
        """
        import shutil
        shutil.copy2(self.knowledge_base.data_path, output_path)
        logger.info(f"Exported knowledge base to {output_path}")

    def clear_knowledge_base(self) -> None:
        """
        Clear all data from the knowledge base.

        Example:
            >>> labeler.clear_knowledge_base()  # Removes all stored data
        """
        self.knowledge_base.clear_knowledge_base()

    def get_prompt_analytics(self) -> dict[str, Any]:
        """
        Get analytics about prompts used by this labeler.

        Returns:
            dict: Prompt usage analytics including most used prompts and performance.

        Example:
            >>> analytics = labeler.get_prompt_analytics()
            >>> print(f"Total prompts used: {analytics['total_prompts']}")
            >>> print(f"Success rate: {analytics['success_rate']:.2%}")
        """
        return self.prompt_store.get_prompt_analytics()

    def export_prompt_history(self, output_path: Path, include_full_text: bool = True) -> None:
        """
        Export complete prompt history to CSV for analysis.

        Args:
            output_path (Path): Path to save the prompt export.
            include_full_text (bool): Whether to include full prompt text.

        Example:
            >>> labeler.export_prompt_history(Path("prompt_history.csv"))
        """
        self.prompt_store.export_prompts(output_path, include_full_text)

    def find_prompts_by_template(self, template_name: str) -> list:
        """
        Find all prompts that used a specific template.

        Args:
            template_name (str): Name of the template to search for.

        Returns:
            list: List of prompt records using that template.

        Example:
            >>> prompts = labeler.find_prompts_by_template("label_prompt.j2")
            >>> print(f"Found {len(prompts)} prompts using this template")
        """
        return self.prompt_store.find_similar_prompts(template_source=template_name)

    def get_most_successful_prompts(self, limit: int = 10) -> list:
        """
        Get the most successful prompts based on success rate and confidence.

        Args:
            limit (int): Maximum number of prompts to return.

        Returns:
            list: List of most successful prompt records.

        Example:
            >>> top_prompts = labeler.get_most_successful_prompts(5)
            >>> for prompt in top_prompts:
            ...     print(f"ID: {prompt.prompt_id}, Success: {prompt.successful_predictions}")
        """
        all_prompts = list(self.prompt_store.prompts.values())

        # Sort by success rate and then by confidence
        def success_score(prompt):
            total_preds = prompt.successful_predictions + prompt.failed_predictions
            if total_preds == 0:
                return 0
            success_rate = prompt.successful_predictions / total_preds
            confidence_boost = (prompt.avg_confidence or 0) * 0.1  # Small confidence bonus
            return success_rate + confidence_boost

        successful_prompts = sorted(all_prompts, key=success_score, reverse=True)
        return successful_prompts[:limit]


# Legacy functions for backward compatibility
def build_vector_store(df: pd.DataFrame, text_column: str, settings: Settings):
    """Legacy function - use AutoLabeler with KnowledgeBase instead."""
    logger.warning("build_vector_store is deprecated. Use AutoLabeler with KnowledgeBase.")
    kb = KnowledgeBase("temp_dataset", settings)
    kb.add_labeled_data(df, text_column, "label", source="legacy")
    return kb.vector_store


def get_similar_examples(store, text: str, k: int = 3) -> list[Document]:
    """Legacy function - use KnowledgeBase.get_similar_examples instead."""
    logger.warning("get_similar_examples is deprecated. Use KnowledgeBase.get_similar_examples.")
    if hasattr(store, 'similarity_search'):
        return store.similarity_search(text, k=k)
    return []
