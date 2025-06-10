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
        source: str = "human",
        additional_metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Add labeled data to the knowledge base with enhanced metadata support.

        Args:
            df (pd.DataFrame): DataFrame containing text and labels.
            text_column (str): Name of column containing text.
            label_column (str): Name of column containing labels.
            source (str): Source of labels ("human" or "model").
            additional_metadata (dict[str, Any] | None): Additional metadata to attach to all examples.

        Example:
            >>> metadata = {"domain": "product_reviews", "quality_score": 0.95}
            >>> kb.add_labeled_data(labeled_df, "review", "sentiment", additional_metadata=metadata)
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
                "text_length": len(str(row[text_column])),
                "word_count": len(str(row[text_column]).split()),
            }

            # Add any additional metadata columns from the DataFrame
            for col in df.columns:
                if col not in [text_column, label_column]:
                    value = row[col]
                    if pd.notna(value):  # Only add non-null metadata
                        metadata[f"data_{col}"] = value

            # Add user-provided additional metadata
            if additional_metadata:
                for key, value in additional_metadata.items():
                    metadata[f"custom_{key}"] = value

            # Enhanced text for embedding: include context information
            enhanced_text = str(row[text_column])

            # Add contextual information to the embedding text if available
            context_fields = []
            if "category" in row and pd.notna(row["category"]):
                context_fields.append(f"Category: {row['category']}")
            if "domain" in row and pd.notna(row["domain"]):
                context_fields.append(f"Domain: {row['domain']}")
            if "topic" in row and pd.notna(row["topic"]):
                context_fields.append(f"Topic: {row['topic']}")

            if context_fields:
                enhanced_text = f"{' | '.join(context_fields)} | Text: {enhanced_text}"

            documents.append(
                Document(
                    page_content=enhanced_text,
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
            f"knowledge base with enhanced metadata"
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

    def get_similar_examples_with_metadata_filter(
        self,
        query_text: str,
        k: int | None = None,
        filter_source: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
        boost_metadata: dict[str, float] | None = None
    ) -> list[Document]:
        """
        Retrieve similar examples with advanced metadata filtering and boosting.

        Args:
            query_text (str): Text to find similar examples for.
            k (int | None): Number of examples to retrieve.
            filter_source (str | None): Filter by source ("human" or "model").
            metadata_filters (dict[str, Any] | None): Exact metadata matches required.
            boost_metadata (dict[str, float] | None): Metadata fields to boost similarity by multiplier.

        Returns:
            list[Document]: Similar examples with metadata-aware scoring.

        Example:
            >>> # Find examples from same domain with quality boost
            >>> filters = {"data_category": "electronics"}
            >>> boost = {"data_quality_score": 1.2}
            >>> examples = kb.get_similar_examples_with_metadata_filter(
            ...     "Great product!", metadata_filters=filters, boost_metadata=boost
            ... )
        """
        if self.vector_store is None:
            return []

        k = k or self.settings.max_examples_per_query
        search_k = k * 5  # Get more examples for better filtering

        # Enhanced query text with context if available
        enhanced_query = query_text
        if boost_metadata:
            context_parts = []
            for field, weight in boost_metadata.items():
                if weight > 1.0:  # Only boost positive weights
                    field_clean = field.replace("data_", "").replace("custom_", "")
                    context_parts.append(f"{field_clean.title()}")

            if context_parts:
                enhanced_query = f"{' '.join(context_parts)} | Text: {query_text}"

        similar_docs = self.vector_store.similarity_search_with_score(
            enhanced_query, k=search_k
        )

        # Filter by similarity threshold
        filtered_docs = [
            (doc, score) for doc, score in similar_docs
            if score <= (1 - self.settings.similarity_threshold)
        ]

        # Apply metadata filters
        if metadata_filters:
            filtered_docs = [
                (doc, score) for doc, score in filtered_docs
                if all(
                    doc.metadata.get(key) == value
                    for key, value in metadata_filters.items()
                )
            ]

        # Filter by source if specified
        if filter_source:
            filtered_docs = [
                (doc, score) for doc, score in filtered_docs
                if doc.metadata.get("source") == filter_source
            ]

        # Apply metadata boosting to scores
        if boost_metadata:
            boosted_docs = []
            for doc, score in filtered_docs:
                boost_factor = 1.0
                for field, multiplier in boost_metadata.items():
                    if field in doc.metadata:
                        # Boost based on metadata presence and value
                        if isinstance(doc.metadata[field], (int, float)):
                            # Numerical boosting
                            boost_factor *= (multiplier * float(doc.metadata[field]))
                        else:
                            # Categorical boosting
                            boost_factor *= multiplier

                # Apply boost to similarity (lower score is better for FAISS)
                boosted_score = score / boost_factor if boost_factor > 0 else score
                boosted_docs.append((doc, boosted_score))

            # Re-sort by boosted scores
            filtered_docs = sorted(boosted_docs, key=lambda x: x[1])

        # Return top k documents
        return [doc for doc, score in filtered_docs[:k]]

    def create_contextual_embedding(
        self,
        text: str,
        context: dict[str, Any] | None = None
    ) -> list[float]:
        """
        Create contextual embeddings by augmenting text with relevant metadata.

        Args:
            text (str): Original text to embed.
            context (dict[str, Any] | None): Context metadata to include in embedding.

        Returns:
            list[float]: Contextual embedding vector.

        Example:
            >>> context = {"domain": "healthcare", "category": "diagnosis"}
            >>> embedding = kb.create_contextual_embedding("Patient feels dizzy", context)
        """
        # Base text
        enhanced_text = text

        if context:
            # Add context information to influence the embedding
            context_parts = []

            # Domain context
            if "domain" in context:
                context_parts.append(f"Domain: {context['domain']}")

            # Category context
            if "category" in context:
                context_parts.append(f"Category: {context['category']}")

            # Intent context
            if "intent" in context or "user_intent" in context:
                intent = context.get("intent") or context.get("user_intent")
                context_parts.append(f"Intent: {intent}")

            # Additional semantic markers
            if "topic" in context:
                context_parts.append(f"Topic: {context['topic']}")

            if "sentiment" in context:
                context_parts.append(f"Sentiment: {context['sentiment']}")

            # Combine context with text
            if context_parts:
                enhanced_text = f"[{' | '.join(context_parts)}] {text}"

        # Generate embedding using the enhanced text
        try:
            embedding = self.embeddings.embed_query(enhanced_text)
            return embedding
        except Exception as e:
            logger.warning(f"Failed to create contextual embedding: {e}")
            # Fallback to regular embedding
            return self.embeddings.embed_query(text)

    def hybrid_search(
        self,
        query_text: str,
        context: dict[str, Any] | None = None,
        k: int | None = None,
        alpha: float = 0.7,
        metadata_filters: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        """
        Perform hybrid search combining semantic similarity and metadata matching.

        Args:
            query_text (str): Text query to search for.
            context (dict[str, Any] | None): Query context for enhanced embedding.
            k (int | None): Number of results to return.
            alpha (float): Weight for semantic similarity (0-1). (1-alpha) for metadata match.
            metadata_filters (dict[str, Any] | None): Required metadata matches.

        Returns:
            list[tuple[Document, float]]: Documents with hybrid scores.

        Example:
            >>> context = {"domain": "finance", "category": "investment"}
            >>> filters = {"data_quality_score": 0.8}
            >>> results = kb.hybrid_search("market volatility", context, metadata_filters=filters)
        """
        if self.vector_store is None:
            return []

        k = k or self.settings.max_examples_per_query
        search_k = k * 3  # Get more for better filtering

        # Create contextual embedding for query
        query_embedding = self.create_contextual_embedding(query_text, context)

        # Perform similarity search
        similar_docs = self.vector_store.similarity_search_with_score_by_vector(
            query_embedding, k=search_k
        )

        # Calculate hybrid scores
        hybrid_results = []
        for doc, semantic_score in similar_docs:
            # Calculate metadata similarity score
            metadata_score = 0.0
            metadata_matches = 0
            total_metadata_checks = 0

            if context:
                for key, value in context.items():
                    if key in ['boost', 'priority']:
                        continue
                    metadata_key = f"data_{key}"
                    total_metadata_checks += 1
                    if metadata_key in doc.metadata and doc.metadata[metadata_key] == value:
                        metadata_matches += 1

                if total_metadata_checks > 0:
                    metadata_score = metadata_matches / total_metadata_checks

            # Apply metadata filters
            if metadata_filters:
                passes_filter = all(
                    doc.metadata.get(key) == value
                    for key, value in metadata_filters.items()
                )
                if not passes_filter:
                    continue

            # Calculate hybrid score (lower is better for FAISS distance)
            # Convert FAISS distance to similarity and combine with metadata
            semantic_similarity = 1 / (1 + semantic_score)  # Convert distance to similarity
            hybrid_score = alpha * semantic_similarity + (1 - alpha) * metadata_score

            hybrid_results.append((doc, hybrid_score))

        # Sort by hybrid score (higher is better)
        hybrid_results.sort(key=lambda x: x[1], reverse=True)

        return hybrid_results[:k]
