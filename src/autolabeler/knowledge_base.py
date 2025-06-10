from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from loguru import logger

from .config import Settings


class KnowledgeBase:
    """
    Manages dataset-specific knowledge bases with persistent vector stores.

    Tracks both human-labeled and model-generated labels with full provenance
    information including model parameters, prompts, and generation metadata.

    Args:
        dataset_name (str): Unique identifier for this dataset's knowledge base.
        settings (Settings): Application settings for embedding and storage config.

    Example:
        >>> kb = KnowledgeBase("sentiment_reviews", settings)
        >>> kb.add_labeled_data(df, "text", "label")
        >>> examples = kb.get_similar_examples("This movie was great!")
    """

    def __init__(self, dataset_name: str, settings: Settings) -> None:
        self.dataset_name = dataset_name
        self.settings = settings
        self.kb_dir = settings.knowledge_base_dir / dataset_name
        self.kb_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.vector_store_path = self.kb_dir / "vector_store"
        self.metadata_path = self.kb_dir / "metadata.json"
        self.data_path = self.kb_dir / "knowledge_data.csv"

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model
        )

        # Load existing knowledge base if available
        self.vector_store: FAISS | None = None
        self.metadata: dict[str, Any] = {}
        self._load_knowledge_base()

    def _load_knowledge_base(self) -> None:
        """Load existing vector store and metadata from disk."""
        try:
            if self.vector_store_path.exists():
                self.vector_store = FAISS.load_local(
                    str(self.vector_store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded vector store for {self.dataset_name}")

            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata for {self.dataset_name}")

        except Exception as e:
            logger.warning(f"Could not load existing knowledge base: {e}")
            self.vector_store = None
            self.metadata = {}

    def _save_knowledge_base(self) -> None:
        """Save vector store and metadata to disk."""
        try:
            if self.vector_store is not None:
                self.vector_store.save_local(str(self.vector_store_path))

            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)

            logger.info(f"Saved knowledge base for {self.dataset_name}")

        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")
            raise

    def add_labeled_data(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        source: str = "human"
    ) -> None:
        """
        Add labeled data to the knowledge base.

        Args:
            df (pd.DataFrame): DataFrame containing text and labels.
            text_column (str): Name of column containing text.
            label_column (str): Name of column containing labels.
            source (str): Source of labels ("human" or "model").

        Example:
            >>> kb.add_labeled_data(labeled_df, "review", "sentiment")
        """
        # Filter only labeled rows
        labeled_df = df[df[label_column].notna()].copy()

        if labeled_df.empty:
            logger.warning("No labeled data found to add to knowledge base")
            return

        # Create documents with enhanced metadata
        documents = []
        for _, row in labeled_df.iterrows():
            metadata = {
                "text": str(row[text_column]),
                "label": str(row[label_column]),
                "source": source,
                "added_at": datetime.now().isoformat(),
                "dataset": self.dataset_name,
            }
            # Add any additional metadata from the row
            for col in df.columns:
                if col not in [text_column, label_column]:
                    metadata[f"data_{col}"] = row[col]

            documents.append(
                Document(
                    page_content=str(row[text_column]),
                    metadata=metadata
                )
            )

        # Add to vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            # Add new documents to existing store
            new_store = FAISS.from_documents(documents, self.embeddings)
            self.vector_store.merge_from(new_store)

        # Update metadata
        self.metadata.setdefault("total_examples", 0)
        self.metadata["total_examples"] += len(documents)
        self.metadata.setdefault("sources", {})
        self.metadata["sources"].setdefault(source, 0)
        self.metadata["sources"][source] += len(documents)
        self.metadata["last_updated"] = datetime.now().isoformat()

        # Save to disk
        self._save_knowledge_base()
        self._save_data_to_csv(labeled_df, text_column, label_column, source)

        logger.info(
            f"Added {len(documents)} {source} examples to {self.dataset_name} "
            f"knowledge base"
        )

    def add_model_labels(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        model_info: dict[str, Any],
        prompt_template: str,
    ) -> None:
        """
        Add model-generated labels with full provenance tracking.

        Args:
            df (pd.DataFrame): DataFrame with model-generated labels.
            text_column (str): Name of column containing text.
            label_column (str): Name of column containing model labels.
            model_info (dict): Model configuration and parameters used.
            prompt_template (str): Prompt template used for generation.

        Example:
            >>> model_info = {
            ...     "model": "gpt-3.5-turbo",
            ...     "temperature": 0.1,
            ...     "seed": 42
            ... }
            >>> kb.add_model_labels(df, "text", "pred_label", model_info, template)
        """
        # Create enhanced metadata for model labels
        enhanced_df = df.copy()
        enhanced_df["generation_model"] = model_info.get("model", "unknown")
        enhanced_df["generation_params"] = json.dumps(model_info)
        enhanced_df["prompt_template"] = prompt_template
        enhanced_df["generation_timestamp"] = datetime.now().isoformat()

        self.add_labeled_data(enhanced_df, text_column, label_column, source="model")

    def get_similar_examples(
        self,
        query_text: str,
        k: int | None = None,
        filter_source: str | None = None
    ) -> list[Document]:
        """
        Retrieve similar examples from the knowledge base.

        Args:
            query_text (str): Text to find similar examples for.
            k (int | None): Number of examples to retrieve. Uses settings default if None.
            filter_source (str | None): Filter by source ("human" or "model").

        Returns:
            list[Document]: Similar examples with metadata.

        Example:
            >>> examples = kb.get_similar_examples("Great movie!", k=3)
            >>> human_examples = kb.get_similar_examples("Good", filter_source="human")
        """
        if self.vector_store is None:
            return []

        k = k or self.settings.max_examples_per_query

        # Get more examples than needed to allow for filtering
        search_k = k * 3 if filter_source else k

        similar_docs = self.vector_store.similarity_search_with_score(
            query_text, k=search_k
        )

        # Filter by similarity threshold
        filtered_docs = [
            doc for doc, score in similar_docs
            if score <= (1 - self.settings.similarity_threshold)  # FAISS uses distance
        ]

        # Filter by source if specified
        if filter_source:
            filtered_docs = [
                doc for doc in filtered_docs
                if doc.metadata.get("source") == filter_source
            ]

        return filtered_docs[:k]

    def _save_data_to_csv(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        source: str
    ) -> None:
        """Save data to CSV for inspection and backup."""
        try:
            # Add source info to dataframe
            df_with_source = df.copy()
            df_with_source["source"] = source
            df_with_source["added_at"] = datetime.now().isoformat()

            # Append to existing CSV or create new one
            if self.data_path.exists():
                existing_df = pd.read_csv(self.data_path)
                combined_df = pd.concat([existing_df, df_with_source], ignore_index=True)
            else:
                combined_df = df_with_source

            combined_df.to_csv(self.data_path, index=False)

        except Exception as e:
            logger.warning(f"Could not save data to CSV: {e}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the knowledge base.

        Returns:
            dict: Statistics including total examples, sources, and label distribution.

        Example:
            >>> stats = kb.get_stats()
            >>> print(f"Total examples: {stats['total_examples']}")
        """
        stats = self.metadata.copy()
        if self.vector_store:
            stats["vector_store_size"] = self.vector_store.index.ntotal

        # Add label distribution
        stats["label_distribution"] = self.get_label_distribution()

        return stats

    def clear_knowledge_base(self) -> None:
        """
        Clear all data from the knowledge base.

        Example:
            >>> kb.clear_knowledge_base()  # Removes all stored data
        """
        import shutil

        if self.kb_dir.exists():
            shutil.rmtree(self.kb_dir)

        self.vector_store = None
        self.metadata = {}

        # Recreate directory
        self.kb_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Cleared knowledge base for {self.dataset_name}")

    def get_examples_by_label(
        self,
        target_label: str,
        max_examples: int = 10,
        filter_source: str | None = None
    ) -> list[Document]:
        """
        Get examples with a specific label from the knowledge base.

        Args:
            target_label (str): The label to filter examples by.
            max_examples (int): Maximum number of examples to return.
            filter_source (str | None): Filter by source ("human", "model", or "synthetic").

        Returns:
            list[Document]: Examples with the specified label.

        Example:
            >>> positive_examples = kb.get_examples_by_label("positive", max_examples=5)
            >>> human_examples = kb.get_examples_by_label("negative", filter_source="human")
        """
        if self.vector_store is None:
            return []

        # Get all documents from the vector store
        all_docs = []
        try:
            # Use a broad search to get all documents
            temp_docs = self.vector_store.similarity_search("", k=self.vector_store.index.ntotal)
            all_docs = temp_docs
        except Exception:
            # Fallback: use a more conservative approach
            try:
                temp_docs = self.vector_store.similarity_search(" ", k=1000)
                all_docs = temp_docs
            except Exception as e:
                logger.warning(f"Could not retrieve all documents: {e}")
                return []

        # Filter by label and source
        filtered_docs = []
        for doc in all_docs:
            if doc.metadata.get("label") == target_label:
                if filter_source is None or doc.metadata.get("source") == filter_source:
                    filtered_docs.append(doc)

            if len(filtered_docs) >= max_examples:
                break

        return filtered_docs[:max_examples]

    def get_label_distribution(self) -> dict[str, int]:
        """
        Get the distribution of labels in the knowledge base.

        Returns:
            dict[str, int]: Mapping of labels to their counts.

        Example:
            >>> distribution = kb.get_label_distribution()
            >>> print(f"Positive examples: {distribution.get('positive', 0)}")
        """
        if not self.data_path.exists():
            return {}

        try:
            df = pd.read_csv(self.data_path)
            if "label" in df.columns:
                return df["label"].value_counts().to_dict()
            # Fallback: look for any column that might contain labels
            for col in df.columns:
                if "label" in col.lower():
                    return df[col].value_counts().to_dict()
        except Exception as e:
            logger.warning(f"Could not read label distribution from CSV: {e}")

        return {}

    def export_synthetic_data(
        self,
        output_path: Path,
        include_metadata: bool = True,
        filter_confidence: float | None = None
    ) -> None:
        """
        Export synthetic examples from the knowledge base to CSV.

        Args:
            output_path (Path): Path to save the exported synthetic data.
            include_metadata (bool): Whether to include generation metadata columns.
            filter_confidence (float | None): Minimum confidence to include.

        Example:
            >>> kb.export_synthetic_data(Path("synthetic.csv"), filter_confidence=0.8)
        """
        if not self.data_path.exists():
            logger.warning("No data file found to export from")
            return

        try:
            df = pd.read_csv(self.data_path)

            # Filter for synthetic examples only
            synthetic_df = df[df["source"] == "synthetic"].copy()

            if synthetic_df.empty:
                logger.warning("No synthetic examples found in knowledge base")
                return

            # Filter by confidence if specified
            if filter_confidence is not None and "confidence" in synthetic_df.columns:
                synthetic_df = synthetic_df[synthetic_df["confidence"] >= filter_confidence]

            # Select columns to export
            if include_metadata:
                # Include all columns
                export_df = synthetic_df
            else:
                # Include only essential columns
                essential_cols = ["text", "label", "confidence", "source", "added_at"]
                available_cols = [col for col in essential_cols if col in synthetic_df.columns]
                export_df = synthetic_df[available_cols]

            export_df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(export_df)} synthetic examples to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export synthetic data: {e}")
            raise
