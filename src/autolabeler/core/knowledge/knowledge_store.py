"""
Knowledge store for managing labeled examples and embeddings.

This module provides the core knowledge management functionality for the AutoLabeler,
including vector storage, example retrieval, and metadata tracking.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from loguru import logger

from ...config import Settings
from ..base import ConfigurableComponent


class KnowledgeStore(ConfigurableComponent):
    """
    Manages dataset-specific knowledge bases with persistent vector stores.

    Tracks both human-labeled and model-generated labels with full provenance
    information including model parameters, prompts, and generation metadata.

    Args:
        dataset_name (str): Unique identifier for this dataset's knowledge base.
        settings (Settings): Configuration settings for the knowledge store.
        store_dir (Path | None): Directory for storing knowledge data.

    Example:
        >>> store = KnowledgeStore("sentiment_reviews", settings, store_dir)
        >>> store.add_examples(df, "text", "label", source="human")
        >>> similar = store.find_similar_examples("Great product!")
    """

    def __init__(
        self,
        dataset_name: str,
        settings: Settings,
        store_dir: Path | None = None,
    ) -> None:
        """Initialize knowledge store with dataset-specific configuration."""
        super().__init__(
            component_type="knowledge_store",
            dataset_name=dataset_name,
            settings=settings,
        )

        self.dataset_name = dataset_name
        self.store_dir = store_dir or Path("knowledge_bases") / dataset_name
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Storage paths
        self.vector_store_path = self.store_dir / "vector_store"
        self.metadata_path = self.store_dir / "metadata.json"
        self.examples_path = self.store_dir / "examples.parquet"

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

        # Storage components
        self.vector_store: FAISS | None = None
        self.metadata: dict[str, Any] = {
            "total_examples": 0,
            "sources": {},
            "label_distribution": {},
            "created_at": datetime.now().isoformat()
        }

        # Load existing data
        self._load_store()

    def _create_text_hash(self, text: str) -> str:
        """
        Create a hash for text content to detect duplicates.

        Args:
            text (str): Text content to hash.

        Returns:
            str: SHA-256 hash of the normalized text.
        """
        # Normalize text: strip whitespace, convert to lowercase
        normalized_text = text.strip().lower()
        return hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()

    def _get_existing_text_hashes(self) -> set[str]:
        """
        Get all existing text hashes from the knowledge store.

        Returns:
            set[str]: Set of existing text hashes.
        """
        if not self.examples_path.exists():
            return set()

        try:
            existing_df = pd.read_parquet(self.examples_path)
            if 'text_hash' in existing_df.columns:
                return set(existing_df['text_hash'].dropna())
            else:
                # Create hashes for existing data if they don't exist
                text_cols = [col for col in existing_df.columns if 'text' in col.lower()]
                if text_cols:
                    text_col = text_cols[0]  # Use first text column found
                    hashes = existing_df[text_col].apply(self._create_text_hash)
                    return set(hashes)
        except Exception as e:
            logger.warning(f"Could not load existing text hashes: {e}")

        return set()

    def _filter_duplicates(
        self,
        df: pd.DataFrame,
        text_column: str,
        existing_hashes: set[str] | None = None
    ) -> tuple[pd.DataFrame, int]:
        """
        Filter out duplicate examples based on text content.

        Args:
            df (pd.DataFrame): DataFrame containing examples.
            text_column (str): Column containing text.
            existing_hashes (set[str] | None): Set of existing text hashes.

        Returns:
            tuple[pd.DataFrame, int]: Filtered DataFrame and number of duplicates found.
        """
        if existing_hashes is None:
            existing_hashes = self._get_existing_text_hashes()

        # Create text hashes for new data
        df = df.copy()
        df['text_hash'] = df[text_column].apply(self._create_text_hash)

        # Filter out duplicates
        initial_count = len(df)
        df_filtered = df[~df['text_hash'].isin(existing_hashes)]
        duplicates_found = initial_count - len(df_filtered)

        return df_filtered, duplicates_found

    def add_examples(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        source: str = "human",
        metadata_columns: list[str] | None = None,
        confidence_column: str | None = None,
        allow_duplicates: bool = False
    ) -> int:
        """
        Add labeled examples to the knowledge store with duplicate detection.

        Args:
            df (pd.DataFrame): DataFrame containing examples.
            text_column (str): Column containing text.
            label_column (str): Column containing labels.
            source (str): Source of labels ("human", "model", "synthetic").
            metadata_columns (list[str] | None): Additional columns to include as metadata.
            confidence_column (str | None): Column containing confidence scores.
            allow_duplicates (bool): Whether to allow duplicate text entries.

        Returns:
            int: Number of examples added (excluding duplicates).

        Example:
            >>> added = store.add_examples(
            ...     labeled_df, "review", "sentiment",
            ...     metadata_columns=["product_id", "rating"]
            ... )
        """
        # Filter valid examples
        valid_df = df[df[label_column].notna()].copy()
        if valid_df.empty:
            logger.warning("No valid labeled examples to add")
            return 0

        # Filter duplicates unless explicitly allowed
        duplicates_found = 0
        if not allow_duplicates:
            valid_df, duplicates_found = self._filter_duplicates(valid_df, text_column)
            if duplicates_found > 0:
                logger.info(f"Filtered out {duplicates_found} duplicate examples")

            if valid_df.empty:
                logger.warning("No new examples to add after filtering duplicates")
                return 0

        # Add timestamp and source
        valid_df["kb_source"] = source
        valid_df["kb_added_at"] = datetime.now().isoformat()

        # Create documents for vector store
        documents = self._create_documents(
            valid_df, text_column, label_column,
            metadata_columns, confidence_column
        )

        # Update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            new_store = FAISS.from_documents(documents, self.embeddings)
            self.vector_store.merge_from(new_store)

        # Update metadata
        self._update_metadata(valid_df, label_column, source)

        # Save to persistent storage
        self._save_examples(valid_df)
        self._save_store()

        added_count = len(valid_df)
        logger.info(f"Added {added_count} {source} examples to knowledge store")
        if duplicates_found > 0:
            logger.info(f"Skipped {duplicates_found} duplicate examples")

        return added_count

    def find_similar_examples(
        self,
        text: str,
        k: int = 5,
        source_filter: str | None = None,
        confidence_threshold: float | None = None
    ) -> list[dict[str, Any]]:
        """
        Find similar examples from the knowledge store.

        Args:
            text (str): Query text.
            k (int): Number of examples to retrieve.
            source_filter (str | None): Filter by source type.
            confidence_threshold (float | None): Minimum confidence threshold.

        Returns:
            list[dict]: Similar examples with metadata (excluding exact matches).

        Example:
            >>> examples = store.find_similar_examples(
            ...     "Amazing product!", k=3, source_filter="human"
            ... )
        """
        if self.vector_store is None:
            return []

        # Create hash of query text for exact match filtering
        query_hash = self._create_text_hash(text)

        # Search with larger k to allow filtering
        search_k = k * 3 if source_filter or confidence_threshold else k
        results = self.vector_store.similarity_search_with_score(text, k=search_k)

        # Filter and format results
        filtered_examples = []
        for doc, score in results:
            metadata = doc.metadata
            example_text = metadata.get("text", "")

            # Exclude exact matches to avoid returning the same text we're trying to label
            if self._create_text_hash(example_text) == query_hash:
                continue

            # Apply filters
            if source_filter and metadata.get("source") != source_filter:
                continue
            if confidence_threshold and metadata.get("confidence", 1.0) < confidence_threshold:
                continue

            example = {
                "text": example_text,
                "label": metadata.get("label", ""),
                "source": metadata.get("source", ""),
                "confidence": metadata.get("confidence", 1.0),
                "similarity_score": 1 - score,  # Convert distance to similarity
                "metadata": {k: v for k, v in metadata.items()
                           if k not in ["text", "label", "source", "confidence"]}
            }
            filtered_examples.append(example)

            if len(filtered_examples) >= k:
                break

        return filtered_examples

    async def afind_similar_examples(
        self,
        text: str,
        k: int = 5,
        source_filter: str | None = None,
        confidence_threshold: float | None = None
    ) -> list[dict[str, Any]]:
        """
        Asynchronously find similar examples from the knowledge store.

        Args:
            text (str): Query text.
            k (int): Number of examples to retrieve.
            source_filter (str | None): Filter by source type.
            confidence_threshold (float | None): Minimum confidence threshold.

        Returns:
            list[dict]: Similar examples with metadata (excluding exact matches).
        """
        if self.vector_store is None:
            return []

        # Create hash of query text for exact match filtering
        query_hash = self._create_text_hash(text)

        # Search with larger k to allow filtering
        search_k = k * 3 if source_filter or confidence_threshold else k
        results = await self.vector_store.asimilarity_search_with_score(text, k=search_k)

        # Filter and format results
        filtered_examples = []
        for doc, score in results:
            metadata = doc.metadata
            example_text = metadata.get("text", "")

            # Exclude exact matches to avoid returning the same text we're trying to label
            if self._create_text_hash(example_text) == query_hash:
                continue

            # Apply filters
            if source_filter and metadata.get("source") != source_filter:
                continue
            if confidence_threshold and metadata.get("confidence", 1.0) < confidence_threshold:
                continue

            example = {
                "text": example_text,
                "label": metadata.get("label", ""),
                "source": metadata.get("source", ""),
                "confidence": metadata.get("confidence", 1.0),
                "similarity_score": 1 - score,  # Convert distance to similarity
                "metadata": {k: v for k, v in metadata.items()
                           if k not in ["text", "label", "source", "confidence"]}
            }
            filtered_examples.append(example)

            if len(filtered_examples) >= k:
                break

        return filtered_examples

    def get_label_distribution(self, source_filter: str | None = None) -> dict[str, int]:
        """
        Get distribution of labels in the knowledge store.

        Args:
            source_filter (str | None): Filter by source type.

        Returns:
            dict[str, int]: Label counts.

        Example:
            >>> dist = store.get_label_distribution(source_filter="human")
        """
        if not self.examples_path.exists():
            return {}

        try:
            df = pd.read_parquet(self.examples_path)
            if source_filter:
                df = df[df["kb_source"] == source_filter]

            # Find label column
            label_cols = [col for col in df.columns if "label" in col.lower()]
            if label_cols:
                return df[label_cols[0]].value_counts().to_dict()
        except Exception as e:
            logger.error(f"Failed to get label distribution: {e}")

        return {}

    def get_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive statistics about the knowledge store.

        Returns:
            dict: Statistics including counts, distributions, and metadata.

        Example:
            >>> stats = store.get_statistics()
            >>> print(f"Total examples: {stats['total_examples']}")
        """
        stats = self.metadata.copy()

        # Add current statistics
        stats["label_distribution"] = self.get_label_distribution()
        stats["source_distribution"] = {
            source: self.get_label_distribution(source)
            for source in self.metadata.get("sources", {}).keys()
        }

        if self.vector_store:
            stats["vector_store_size"] = self.vector_store.index.ntotal

        stats["last_updated"] = self.metadata.get("last_updated", "unknown")

        return stats

    def _create_documents(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        metadata_columns: list[str] | None,
        confidence_column: str | None
    ) -> list[Document]:
        """Create LangChain documents from DataFrame."""
        documents = []
        metadata_columns = metadata_columns or []

        for _, row in df.iterrows():
            # Base metadata
            metadata = {
                "text": str(row[text_column]),
                "label": str(row[label_column]),
                "source": row.get("kb_source", "unknown"),
                "added_at": row.get("kb_added_at", datetime.now().isoformat()),
                "dataset": self.dataset_name
            }

            # Add confidence if available
            if confidence_column and confidence_column in row:
                metadata["confidence"] = float(row[confidence_column])

            # Add additional metadata
            for col in metadata_columns:
                if col in row and pd.notna(row[col]):
                    metadata[f"meta_{col}"] = str(row[col])

            # Create enhanced text for better retrieval
            enhanced_text = str(row[text_column])
            if "category" in row and pd.notna(row["category"]):
                enhanced_text = f"[{row['category']}] {enhanced_text}"

            documents.append(Document(
                page_content=enhanced_text,
                metadata=metadata
            ))

        return documents

    def _update_metadata(self, df: pd.DataFrame, label_column: str, source: str) -> None:
        """Update internal metadata with new examples."""
        # Update counts
        self.metadata["total_examples"] += len(df)
        self.metadata.setdefault("sources", {})
        self.metadata["sources"][source] = self.metadata["sources"].get(source, 0) + len(df)

        # Update label distribution
        label_counts = df[label_column].value_counts().to_dict()
        for label, count in label_counts.items():
            self.metadata["label_distribution"][label] = \
                self.metadata["label_distribution"].get(label, 0) + count

        self.metadata["last_updated"] = datetime.now().isoformat()

    def _save_examples(self, df: pd.DataFrame) -> None:
        """Save examples to persistent storage."""
        try:
            if self.examples_path.exists():
                existing_df = pd.read_parquet(self.examples_path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            else:
                combined_df = df

            combined_df.to_parquet(self.examples_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save examples: {e}")

    def _save_store(self) -> None:
        """Save vector store and metadata."""
        try:
            # Save vector store
            if self.vector_store:
                self.vector_store.save_local(str(self.vector_store_path))

            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save knowledge store: {e}")

    def _load_store(self) -> None:
        """Load existing store from disk."""
        try:
            # Load vector store
            if self.vector_store_path.exists():
                self.vector_store = FAISS.load_local(
                    str(self.vector_store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)

            logger.info(f"Loaded knowledge store for {self.dataset_name}")

        except Exception as e:
            logger.warning(f"Could not load existing store: {e}")

    def clear(self) -> None:
        """Clear all data from the knowledge store."""
        import shutil

        if self.store_dir.exists():
            shutil.rmtree(self.store_dir)

        self.vector_store = None
        self.metadata = {
            "total_examples": 0,
            "sources": {},
            "label_distribution": {},
            "created_at": datetime.now().isoformat()
        }

        self.store_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cleared knowledge store for {self.dataset_name}")

    def check_duplicates(
        self,
        df: pd.DataFrame,
        text_column: str
    ) -> dict[str, Any]:
        """
        Check for duplicates in the provided DataFrame against existing data.

        Args:
            df (pd.DataFrame): DataFrame to check for duplicates.
            text_column (str): Column containing text.

        Returns:
            dict[str, Any]: Dictionary containing duplicate statistics.

        Example:
            >>> duplicate_info = store.check_duplicates(new_df, "text")
            >>> print(f"Duplicates found: {duplicate_info['duplicate_count']}")
        """
        existing_hashes = self._get_existing_text_hashes()
        df_copy = df.copy()
        df_copy['text_hash'] = df_copy[text_column].apply(self._create_text_hash)

        # Find duplicates
        duplicate_mask = df_copy['text_hash'].isin(existing_hashes)
        duplicates_df = df_copy[duplicate_mask]

        return {
            "total_examples": len(df),
            "duplicate_count": len(duplicates_df),
            "unique_count": len(df) - len(duplicates_df),
            "duplicate_percentage": (len(duplicates_df) / len(df)) * 100 if len(df) > 0 else 0,
            "duplicate_examples": duplicates_df[text_column].tolist() if len(duplicates_df) <= 10 else duplicates_df[text_column].head(10).tolist(),
            "showing_first_n_duplicates": min(10, len(duplicates_df))
        }

    def get_duplicate_statistics(self) -> dict[str, Any]:
        """
        Get statistics about duplicates in the knowledge store.

        Returns:
            dict[str, Any]: Statistics about potential internal duplicates.

        Example:
            >>> stats = store.get_duplicate_statistics()
            >>> print(f"Internal duplicates: {stats['internal_duplicate_count']}")
        """
        if not self.examples_path.exists():
            return {
                "total_examples": 0,
                "internal_duplicate_count": 0,
                "unique_text_count": 0
            }

        try:
            df = pd.read_parquet(self.examples_path)
            if df.empty:
                return {
                    "total_examples": 0,
                    "internal_duplicate_count": 0,
                    "unique_text_count": 0
                }

            # Find text column
            text_cols = [col for col in df.columns if 'text' in col.lower() and col != 'text_hash']
            if not text_cols:
                return {
                    "total_examples": len(df),
                    "internal_duplicate_count": 0,
                    "unique_text_count": len(df),
                    "error": "No text column found"
                }

            text_col = text_cols[0]

            # Check for internal duplicates
            if 'text_hash' not in df.columns:
                df['text_hash'] = df[text_col].apply(self._create_text_hash)

            total_examples = len(df)
            unique_hashes = df['text_hash'].nunique()
            internal_duplicates = total_examples - unique_hashes

            return {
                "total_examples": total_examples,
                "internal_duplicate_count": internal_duplicates,
                "unique_text_count": unique_hashes,
                "duplicate_percentage": (internal_duplicates / total_examples) * 100 if total_examples > 0 else 0
            }

        except Exception as e:
            logger.error(f"Failed to get duplicate statistics: {e}")
            return {
                "total_examples": 0,
                "internal_duplicate_count": 0,
                "unique_text_count": 0,
                "error": str(e)
            }

    def remove_duplicates(self, keep: str = "first") -> dict[str, Any]:
        """
        Remove duplicate examples from the knowledge store.

        Args:
            keep (str): Which duplicate to keep ('first', 'last', or 'highest_confidence').

        Returns:
            dict[str, Any]: Summary of duplicate removal operation.

        Example:
            >>> result = store.remove_duplicates(keep="highest_confidence")
            >>> print(f"Removed {result['duplicates_removed']} duplicates")
        """
        if not self.examples_path.exists():
            return {
                "duplicates_removed": 0,
                "remaining_examples": 0,
                "error": "No existing data found"
            }

        try:
            df = pd.read_parquet(self.examples_path)
            if df.empty:
                return {
                    "duplicates_removed": 0,
                    "remaining_examples": 0,
                    "error": "Knowledge store is empty"
                }

            initial_count = len(df)

            # Find text column
            text_cols = [col for col in df.columns if 'text' in col.lower() and col != 'text_hash']
            if not text_cols:
                return {
                    "duplicates_removed": 0,
                    "remaining_examples": initial_count,
                    "error": "No text column found"
                }

            text_col = text_cols[0]

            # Create text hashes if they don't exist
            if 'text_hash' not in df.columns:
                df['text_hash'] = df[text_col].apply(self._create_text_hash)

            # Handle different keep strategies
            if keep == "highest_confidence":
                # Sort by confidence (descending) and keep the highest confidence duplicate
                confidence_cols = [col for col in df.columns if 'confidence' in col.lower()]
                if confidence_cols:
                    confidence_col = confidence_cols[0]
                    df = df.sort_values(confidence_col, ascending=False)
                    df_deduplicated = df.drop_duplicates(subset=['text_hash'], keep='first')
                else:
                    # Fall back to 'first' if no confidence column
                    df_deduplicated = df.drop_duplicates(subset=['text_hash'], keep='first')
                    logger.warning("No confidence column found, falling back to keeping first duplicate")
            else:
                # Keep 'first' or 'last'
                df_deduplicated = df.drop_duplicates(subset=['text_hash'], keep=keep)

            final_count = len(df_deduplicated)
            duplicates_removed = initial_count - final_count

            if duplicates_removed > 0:
                # Save deduplicated data
                df_deduplicated.to_parquet(self.examples_path, index=False)

                # Rebuild vector store if it exists
                if self.vector_store is not None:
                    logger.info("Rebuilding vector store after duplicate removal...")

                    # Find label column for rebuilding
                    label_cols = [col for col in df_deduplicated.columns if 'label' in col.lower() and col not in ['text_hash']]
                    if label_cols:
                        label_col = label_cols[0]

                        # Rebuild documents
                        documents = self._create_documents(
                            df_deduplicated, text_col, label_col, None, None
                        )

                        # Create new vector store
                        self.vector_store = FAISS.from_documents(documents, self.embeddings)

                        # Save updated vector store
                        self._save_store()

                # Update metadata
                self.metadata["total_examples"] = final_count
                self.metadata["last_updated"] = datetime.now().isoformat()
                self._save_store()

                logger.info(f"Removed {duplicates_removed} duplicate examples from knowledge store")

            return {
                "duplicates_removed": duplicates_removed,
                "remaining_examples": final_count,
                "initial_examples": initial_count,
                "keep_strategy": keep,
                "vector_store_rebuilt": duplicates_removed > 0 and self.vector_store is not None
            }

        except Exception as e:
            logger.error(f"Failed to remove duplicates: {e}")
            return {
                "duplicates_removed": 0,
                "remaining_examples": 0,
                "error": str(e)
            }
