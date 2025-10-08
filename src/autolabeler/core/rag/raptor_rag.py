"""RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

This module implements the RAPTOR algorithm which builds a hierarchical tree
structure of document summaries for multi-scale retrieval. The tree allows
retrieval at different levels of abstraction, from specific examples to
high-level themes.

Reference:
    Sarthi et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


class RAPTORConfig(BaseModel):
    """Configuration for RAPTOR.

    Attributes:
        max_tree_depth: Maximum depth of the tree (levels of abstraction)
        clustering_threshold: Distance threshold for agglomerative clustering
        min_cluster_size: Minimum size for a cluster to be summarized
        max_cluster_size: Maximum cluster size before splitting
        summary_length: Target length for summaries (in tokens)
        use_multi_level_retrieval: Retrieve from multiple tree levels
        level_weights: Weights for each tree level when combining results
    """

    max_tree_depth: int = Field(default=3, description='Maximum tree depth')
    clustering_threshold: float = Field(
        default=0.5, description='Clustering distance threshold'
    )
    min_cluster_size: int = Field(default=3, description='Minimum cluster size')
    max_cluster_size: int = Field(default=20, description='Maximum cluster size')
    summary_length: int = Field(default=100, description='Summary length in tokens')
    use_multi_level_retrieval: bool = Field(
        default=True, description='Use multi-level retrieval'
    )
    level_weights: list[float] | None = Field(
        default=None, description='Weights for each tree level'
    )


@dataclass
class RAPTORNode:
    """Node in the RAPTOR tree.

    Attributes:
        node_id: Unique identifier
        level: Level in tree (0=leaf, higher=more abstract)
        text: Text content (original or summary)
        embedding: Vector embedding
        children: List of child node IDs
        parent: Parent node ID
        is_leaf: Whether this is a leaf node (original document)
        metadata: Additional metadata
        cluster_id: ID of cluster this node belongs to at its level
    """

    node_id: str
    level: int
    text: str
    embedding: np.ndarray
    children: list[str]
    parent: str | None
    is_leaf: bool
    metadata: dict[str, Any]
    cluster_id: int | None = None


class RAPTORRAG:
    """RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

    This implementation builds a hierarchical tree where:
    - Leaf nodes are original documents/examples
    - Internal nodes are summaries of their children
    - Retrieval can happen at any level of abstraction

    Example:
        >>> config = RAPTORConfig(max_tree_depth=3)
        >>> raptor = RAPTORRAG(config)
        >>> raptor.build_tree(
        ...     examples_df,
        ...     text_col='text',
        ...     label_col='label',
        ...     embedding_fn=model.encode,
        ...     summarize_fn=llm_summarize
        ... )
        >>> results = raptor.retrieve(query_text, query_embedding, k=5)
    """

    def __init__(self, config: RAPTORConfig):
        """Initialize RAPTOR.

        Args:
            config: Configuration for RAPTOR behavior
        """
        self.config = config
        self.nodes: dict[str, RAPTORNode] = {}
        self.root_nodes: list[str] = []
        self.leaf_nodes: list[str] = []
        self.levels: dict[int, list[str]] = {}  # level -> list of node IDs

        # Set default level weights if not provided
        if self.config.level_weights is None:
            # Higher weight for lower (more specific) levels
            self.config.level_weights = [
                1.0,
                0.7,
                0.4,
            ][: self.config.max_tree_depth]

        logger.info('Initialized RAPTOR with config: %s', config)

    def build_tree(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        embedding_fn: Callable[[str], np.ndarray],
        summarize_fn: Callable[[list[str]], str],
        metadata_columns: list[str] | None = None,
    ) -> None:
        """Build RAPTOR tree from examples.

        Args:
            df: DataFrame containing examples
            text_column: Name of text column
            label_column: Name of label column
            embedding_fn: Function to compute embeddings
            summarize_fn: Function to generate summaries from list of texts
            metadata_columns: Optional list of metadata columns

        Example:
            >>> def summarize_cluster(texts):
            ...     # Use LLM to create summary
            ...     prompt = f"Summarize these examples: {texts}"
            ...     return llm.invoke(prompt)
            >>>
            >>> raptor.build_tree(
            ...     df,
            ...     text_column='text',
            ...     label_column='label',
            ...     embedding_fn=model.encode,
            ...     summarize_fn=summarize_cluster
            ... )
        """
        logger.info(f'Building RAPTOR tree from {len(df)} examples')

        # Step 1: Create leaf nodes from original examples
        leaf_embeddings = []
        for idx, row in df.iterrows():
            node_id = f'leaf_{idx}'
            text = str(row[text_column])
            label = str(row[label_column])

            # Compute embedding
            embedding = embedding_fn(text)
            leaf_embeddings.append(embedding)

            # Extract metadata
            metadata = {'label': label, 'original_index': idx}
            if metadata_columns:
                for col in metadata_columns:
                    if col in row:
                        metadata[col] = row[col]

            # Create leaf node
            node = RAPTORNode(
                node_id=node_id,
                level=0,
                text=text,
                embedding=embedding,
                children=[],
                parent=None,
                is_leaf=True,
                metadata=metadata,
            )

            self.nodes[node_id] = node
            self.leaf_nodes.append(node_id)

        self.levels[0] = self.leaf_nodes.copy()

        logger.info(f'Created {len(self.leaf_nodes)} leaf nodes')

        # Step 2: Build tree levels through recursive clustering and summarization
        current_level = 0
        current_nodes = self.leaf_nodes.copy()
        current_embeddings = np.array(leaf_embeddings)

        while current_level < self.config.max_tree_depth - 1 and len(current_nodes) > 1:
            logger.info(
                f'Building level {current_level + 1} from {len(current_nodes)} nodes'
            )

            # Cluster nodes at current level
            clusters = self._cluster_nodes(current_embeddings)

            # Create parent nodes for each cluster
            next_level_nodes = []
            next_level_embeddings = []

            for cluster_id, cluster_indices in clusters.items():
                if len(cluster_indices) < self.config.min_cluster_size:
                    # Skip small clusters
                    continue

                # Get texts from cluster
                cluster_node_ids = [current_nodes[i] for i in cluster_indices]
                cluster_texts = [self.nodes[nid].text for nid in cluster_node_ids]
                cluster_labels = [
                    self.nodes[nid].metadata.get('label', '')
                    for nid in cluster_node_ids
                ]

                # Generate summary
                summary_text = summarize_fn(cluster_texts)

                # Compute embedding for summary
                summary_embedding = embedding_fn(summary_text)

                # Create parent node
                parent_id = f'level{current_level + 1}_cluster{cluster_id}'
                parent_node = RAPTORNode(
                    node_id=parent_id,
                    level=current_level + 1,
                    text=summary_text,
                    embedding=summary_embedding,
                    children=cluster_node_ids,
                    parent=None,
                    is_leaf=False,
                    metadata={
                        'cluster_id': cluster_id,
                        'num_children': len(cluster_node_ids),
                        'dominant_label': max(
                            set(cluster_labels), key=cluster_labels.count
                        ),
                    },
                    cluster_id=cluster_id,
                )

                self.nodes[parent_id] = parent_node
                next_level_nodes.append(parent_id)
                next_level_embeddings.append(summary_embedding)

                # Update children to point to parent
                for child_id in cluster_node_ids:
                    self.nodes[child_id].parent = parent_id

            if not next_level_nodes:
                break

            # Store level information
            self.levels[current_level + 1] = next_level_nodes

            # Prepare for next iteration
            current_level += 1
            current_nodes = next_level_nodes
            current_embeddings = np.array(next_level_embeddings)

            logger.info(f'Created {len(next_level_nodes)} nodes at level {current_level}')

        # Store root nodes (top-level nodes with no parent)
        self.root_nodes = [
            node_id for node_id, node in self.nodes.items() if node.parent is None
        ]

        logger.info(
            f'Built tree with {len(self.levels)} levels, '
            f'{len(self.nodes)} total nodes, '
            f'{len(self.root_nodes)} root nodes'
        )

    def _cluster_nodes(
        self, embeddings: np.ndarray
    ) -> dict[int, list[int]]:
        """Cluster nodes using agglomerative clustering.

        Args:
            embeddings: Array of node embeddings

        Returns:
            Dictionary mapping cluster_id to list of node indices
        """
        if len(embeddings) < self.config.min_cluster_size:
            # Too few nodes to cluster
            return {0: list(range(len(embeddings)))}

        # Perform agglomerative clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.config.clustering_threshold,
            linkage='average',
        )

        labels = clustering.fit_predict(embeddings)

        # Group indices by cluster
        clusters = {}
        for idx, cluster_id in enumerate(labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(idx)

        # Split large clusters if needed
        final_clusters = {}
        next_cluster_id = max(clusters.keys()) + 1

        for cluster_id, indices in clusters.items():
            if len(indices) <= self.config.max_cluster_size:
                final_clusters[cluster_id] = indices
            else:
                # Recursively cluster large clusters
                sub_embeddings = embeddings[indices]
                sub_clusters = self._cluster_nodes(sub_embeddings)

                for sub_cluster_id, sub_indices in sub_clusters.items():
                    # Map sub-cluster indices back to original indices
                    original_indices = [indices[i] for i in sub_indices]
                    final_clusters[next_cluster_id] = original_indices
                    next_cluster_id += 1

        return final_clusters

    def retrieve(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        k: int = 5,
        levels: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant examples from RAPTOR tree.

        Args:
            query_text: Query text
            query_embedding: Embedding of query
            k: Number of examples to retrieve
            levels: Specific tree levels to search (None = all levels)

        Returns:
            List of retrieved examples with scores

        Example:
            >>> # Retrieve from all levels
            >>> results = raptor.retrieve(query_text, query_embedding, k=5)
            >>>
            >>> # Retrieve only from leaf level (most specific)
            >>> results = raptor.retrieve(query_text, query_embedding, k=5, levels=[0])
            >>>
            >>> # Retrieve from leaves and first summary level
            >>> results = raptor.retrieve(query_text, query_embedding, k=5, levels=[0, 1])
        """
        if levels is None:
            if self.config.use_multi_level_retrieval:
                levels = list(self.levels.keys())
            else:
                # Default to leaf level only
                levels = [0]

        # Collect candidates from specified levels
        all_candidates = []

        for level in levels:
            if level not in self.levels:
                logger.warning(f'Level {level} does not exist in tree')
                continue

            level_nodes = self.levels[level]
            level_weight = self.config.level_weights[
                min(level, len(self.config.level_weights) - 1)
            ]

            # Compute similarities for this level
            level_embeddings = np.array(
                [self.nodes[nid].embedding for nid in level_nodes]
            )

            similarities = cosine_similarity(
                query_embedding.reshape(1, -1), level_embeddings
            )[0]

            # Add candidates with level-weighted scores
            for i, node_id in enumerate(level_nodes):
                node = self.nodes[node_id]
                weighted_score = similarities[i] * level_weight

                candidate = {
                    'node_id': node_id,
                    'level': level,
                    'text': node.text,
                    'similarity_score': float(similarities[i]),
                    'weighted_score': float(weighted_score),
                    'is_leaf': node.is_leaf,
                    'metadata': node.metadata,
                }

                all_candidates.append(candidate)

        # Sort by weighted score and get top-k
        all_candidates.sort(key=lambda x: x['weighted_score'], reverse=True)
        top_k = all_candidates[:k]

        # For non-leaf nodes, we can also return their leaf descendants
        results = []
        for candidate in top_k:
            result = candidate.copy()

            # If not a leaf, add information about descendants
            if not candidate['is_leaf']:
                descendants = self._get_leaf_descendants(candidate['node_id'])
                result['num_descendants'] = len(descendants)
                result['descendant_samples'] = [
                    {
                        'text': self.nodes[d].text[:100],
                        'label': self.nodes[d].metadata.get('label', ''),
                    }
                    for d in descendants[:3]
                ]  # Sample of descendants

            results.append(result)

        return results

    def _get_leaf_descendants(self, node_id: str) -> list[str]:
        """Get all leaf node descendants of a given node.

        Args:
            node_id: ID of node to get descendants for

        Returns:
            List of leaf node IDs that are descendants
        """
        node = self.nodes[node_id]

        if node.is_leaf:
            return [node_id]

        # Recursively collect leaves from children
        leaves = []
        for child_id in node.children:
            leaves.extend(self._get_leaf_descendants(child_id))

        return leaves

    def retrieve_with_tree_collapse(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        k: int = 5,
        collapse_threshold: float = 0.8,
    ) -> list[dict[str, Any]]:
        """Retrieve with tree collapse strategy.

        This method starts from the root and recursively expands high-scoring
        nodes until reaching leaves, providing adaptive abstraction level.

        Args:
            query_text: Query text
            query_embedding: Embedding of query
            k: Number of leaf examples to retrieve
            collapse_threshold: Similarity threshold for expanding nodes

        Returns:
            List of leaf examples found through tree collapse

        Example:
            >>> results = raptor.retrieve_with_tree_collapse(
            ...     query_text, query_embedding, k=5, collapse_threshold=0.8
            ... )
        """
        # Start from root nodes
        candidates = [(nid, 1.0) for nid in self.root_nodes]  # (node_id, score)
        leaf_results = []

        while candidates and len(leaf_results) < k:
            # Get highest scoring candidate
            candidates.sort(key=lambda x: x[1], reverse=True)
            node_id, parent_score = candidates.pop(0)
            node = self.nodes[node_id]

            # Compute similarity for this node
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1), node.embedding.reshape(1, -1)
            )[0][0]

            combined_score = parent_score * similarity

            if node.is_leaf:
                # Add leaf to results
                leaf_results.append(
                    {
                        'node_id': node_id,
                        'text': node.text,
                        'label': node.metadata.get('label', ''),
                        'similarity_score': float(similarity),
                        'path_score': float(combined_score),
                        'metadata': node.metadata,
                    }
                )
            elif similarity >= collapse_threshold and node.children:
                # Expand this node - add children to candidates
                for child_id in node.children:
                    candidates.append((child_id, combined_score))
            else:
                # Don't expand - treat as terminal and get representative leaves
                descendants = self._get_leaf_descendants(node_id)
                for desc_id in descendants[: k - len(leaf_results)]:
                    desc_node = self.nodes[desc_id]
                    leaf_results.append(
                        {
                            'node_id': desc_id,
                            'text': desc_node.text,
                            'label': desc_node.metadata.get('label', ''),
                            'similarity_score': float(similarity),
                            'path_score': float(combined_score),
                            'from_summary': True,
                            'metadata': desc_node.metadata,
                        }
                    )

        return leaf_results[:k]

    def get_tree_statistics(self) -> dict[str, Any]:
        """Get statistics about the RAPTOR tree.

        Returns:
            Dictionary with tree statistics

        Example:
            >>> stats = raptor.get_tree_statistics()
            >>> print(f"Tree has {stats['num_levels']} levels")
        """
        stats = {
            'num_levels': len(self.levels),
            'total_nodes': len(self.nodes),
            'num_leaves': len(self.leaf_nodes),
            'num_roots': len(self.root_nodes),
            'nodes_per_level': {level: len(nodes) for level, nodes in self.levels.items()},
            'avg_children_per_node': np.mean(
                [len(node.children) for node in self.nodes.values() if not node.is_leaf]
            )
            if any(not node.is_leaf for node in self.nodes.values())
            else 0,
        }

        return stats

    def save_tree(self, output_path: Path) -> None:
        """Save RAPTOR tree to disk.

        Args:
            output_path: Path to save the tree

        Example:
            >>> raptor.save_tree(Path('knowledge_bases/raptor_tree.pkl'))
        """
        import pickle

        output_path.parent.mkdir(parents=True, exist_ok=True)

        tree_data = {
            'nodes': self.nodes,
            'root_nodes': self.root_nodes,
            'leaf_nodes': self.leaf_nodes,
            'levels': self.levels,
            'config': self.config,
        }

        with open(output_path, 'wb') as f:
            pickle.dump(tree_data, f)

        logger.info(f'Saved RAPTOR tree to {output_path}')

    @classmethod
    def load_tree(cls, input_path: Path) -> RAPTORRAG:
        """Load RAPTOR tree from disk.

        Args:
            input_path: Path to saved tree

        Returns:
            RAPTORRAG instance with loaded tree

        Example:
            >>> raptor = RAPTORRAG.load_tree(Path('knowledge_bases/raptor_tree.pkl'))
        """
        import pickle

        with open(input_path, 'rb') as f:
            tree_data = pickle.load(f)

        instance = cls(tree_data['config'])
        instance.nodes = tree_data['nodes']
        instance.root_nodes = tree_data['root_nodes']
        instance.leaf_nodes = tree_data['leaf_nodes']
        instance.levels = tree_data['levels']

        logger.info(f'Loaded RAPTOR tree from {input_path}')

        return instance
