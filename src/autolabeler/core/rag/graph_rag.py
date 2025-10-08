"""GraphRAG implementation for advanced knowledge retrieval.

This module implements Graph-based Retrieval Augmented Generation (GraphRAG)
which uses graph structures to capture relationships between documents and
enable more sophisticated retrieval patterns.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity


class GraphRAGConfig(BaseModel):
    """Configuration for GraphRAG.

    Attributes:
        similarity_threshold: Minimum similarity to create edge between nodes
        max_neighbors: Maximum number of neighbors to consider per node
        community_algorithm: Algorithm for community detection ('louvain', 'label_propagation')
        use_communities: Whether to use community structure for retrieval
        pagerank_alpha: Damping parameter for PageRank (0.85 standard)
        min_community_size: Minimum size for a community to be considered
        edge_weight_threshold: Minimum edge weight to include in graph
    """

    similarity_threshold: float = Field(
        default=0.7, description='Minimum similarity for edge creation'
    )
    max_neighbors: int = Field(default=10, description='Max neighbors per node')
    community_algorithm: str = Field(
        default='louvain', description='Community detection algorithm'
    )
    use_communities: bool = Field(default=True, description='Use community structure')
    pagerank_alpha: float = Field(default=0.85, description='PageRank damping factor')
    min_community_size: int = Field(
        default=3, description='Minimum community size'
    )
    edge_weight_threshold: float = Field(
        default=0.5, description='Minimum edge weight'
    )


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph.

    Attributes:
        node_id: Unique identifier for the node
        text: Text content of the node
        label: Label associated with the node
        embedding: Vector embedding of the text
        metadata: Additional metadata
        community_id: ID of the community this node belongs to
        pagerank_score: PageRank centrality score
    """

    node_id: str
    text: str
    label: str
    embedding: np.ndarray
    metadata: dict[str, Any]
    community_id: int | None = None
    pagerank_score: float = 0.0


class GraphRAG:
    """Graph-based Retrieval Augmented Generation.

    This class builds and maintains a knowledge graph where nodes are documents/examples
    and edges represent semantic similarity. It uses graph algorithms for enhanced
    retrieval that considers document relationships and community structure.

    Example:
        >>> config = GraphRAGConfig(similarity_threshold=0.7)
        >>> graph_rag = GraphRAG(config)
        >>> graph_rag.build_graph(examples_df, text_col='text', label_col='label')
        >>> results = graph_rag.retrieve(query_text, k=5)
    """

    def __init__(self, config: GraphRAGConfig):
        """Initialize GraphRAG.

        Args:
            config: Configuration for GraphRAG behavior
        """
        self.config = config
        self.graph = nx.Graph()
        self.nodes: dict[str, GraphNode] = {}
        self.embeddings_matrix: np.ndarray | None = None
        self.node_ids: list[str] = []
        self.communities: dict[int, list[str]] = {}

        logger.info('Initialized GraphRAG with config: %s', config)

    def build_graph(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        embedding_fn: callable,
        metadata_columns: list[str] | None = None,
    ) -> None:
        """Build knowledge graph from DataFrame.

        Args:
            df: DataFrame containing examples
            text_column: Name of text column
            label_column: Name of label column
            embedding_fn: Function to compute embeddings (takes text, returns np.ndarray)
            metadata_columns: Optional list of metadata columns to include

        Example:
            >>> from sentence_transformers import SentenceTransformer
            >>> model = SentenceTransformer('all-MiniLM-L6-v2')
            >>> graph_rag.build_graph(
            ...     df,
            ...     text_column='text',
            ...     label_column='label',
            ...     embedding_fn=model.encode
            ... )
        """
        logger.info(f'Building graph from {len(df)} examples')

        # Create nodes
        embeddings = []
        for idx, row in df.iterrows():
            node_id = f'node_{idx}'
            text = str(row[text_column])
            label = str(row[label_column])

            # Compute embedding
            embedding = embedding_fn(text)
            embeddings.append(embedding)

            # Extract metadata
            metadata = {}
            if metadata_columns:
                for col in metadata_columns:
                    if col in row:
                        metadata[col] = row[col]

            # Create node
            node = GraphNode(
                node_id=node_id,
                text=text,
                label=label,
                embedding=embedding,
                metadata=metadata,
            )

            self.nodes[node_id] = node
            self.node_ids.append(node_id)
            self.graph.add_node(node_id, data=node)

        # Store embeddings as matrix for efficient similarity computation
        self.embeddings_matrix = np.array(embeddings)

        # Build edges based on similarity
        self._build_edges()

        # Detect communities
        if self.config.use_communities:
            self._detect_communities()

        # Compute PageRank scores
        self._compute_pagerank()

        logger.info(
            f'Built graph with {len(self.nodes)} nodes, '
            f'{self.graph.number_of_edges()} edges, '
            f'{len(self.communities)} communities'
        )

    def _build_edges(self) -> None:
        """Build edges between nodes based on similarity."""
        logger.info('Building edges based on similarity...')

        # Compute pairwise similarities
        similarities = cosine_similarity(self.embeddings_matrix)

        # Create edges
        edge_count = 0
        for i, node_i_id in enumerate(self.node_ids):
            # Get top-k most similar nodes
            similar_indices = np.argsort(similarities[i])[::-1][
                1 : self.config.max_neighbors + 1
            ]

            for j in similar_indices:
                similarity = similarities[i, j]

                # Only create edge if above threshold
                if similarity >= self.config.similarity_threshold:
                    node_j_id = self.node_ids[j]

                    # Add edge with similarity as weight
                    if similarity >= self.config.edge_weight_threshold:
                        self.graph.add_edge(
                            node_i_id, node_j_id, weight=float(similarity)
                        )
                        edge_count += 1

        logger.info(f'Created {edge_count} edges')

    def _detect_communities(self) -> None:
        """Detect communities in the graph."""
        logger.info('Detecting communities...')

        if self.config.community_algorithm == 'louvain':
            try:
                import community as community_louvain

                partition = community_louvain.best_partition(self.graph)
            except ImportError:
                logger.warning(
                    'python-louvain not installed, falling back to label propagation'
                )
                self.config.community_algorithm = 'label_propagation'

        if self.config.community_algorithm == 'label_propagation':
            communities_generator = nx.community.label_propagation_communities(
                self.graph
            )
            partition = {}
            for comm_id, community_nodes in enumerate(communities_generator):
                for node in community_nodes:
                    partition[node] = comm_id

        # Assign community IDs to nodes
        self.communities = defaultdict(list)
        for node_id, comm_id in partition.items():
            self.nodes[node_id].community_id = comm_id
            self.communities[comm_id].append(node_id)

        # Filter out small communities
        filtered_communities = {
            comm_id: nodes
            for comm_id, nodes in self.communities.items()
            if len(nodes) >= self.config.min_community_size
        }

        logger.info(
            f'Detected {len(self.communities)} communities '
            f'({len(filtered_communities)} above minimum size)'
        )

        self.communities = filtered_communities

    def _compute_pagerank(self) -> None:
        """Compute PageRank scores for all nodes."""
        logger.info('Computing PageRank scores...')

        try:
            pagerank_scores = nx.pagerank(
                self.graph, alpha=self.config.pagerank_alpha, weight='weight'
            )

            for node_id, score in pagerank_scores.items():
                self.nodes[node_id].pagerank_score = score

            logger.info('PageRank computation complete')

        except Exception as e:
            logger.warning(f'Failed to compute PageRank: {e}')

    def retrieve(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        k: int = 5,
        use_pagerank: bool = True,
        community_boost: float = 0.2,
    ) -> list[dict[str, Any]]:
        """Retrieve most relevant examples using graph structure.

        Args:
            query_text: Query text
            query_embedding: Embedding of query text
            k: Number of examples to retrieve
            use_pagerank: Whether to boost scores by PageRank
            community_boost: Boost factor for nodes in same community

        Returns:
            List of retrieved examples with similarity scores and metadata

        Example:
            >>> results = graph_rag.retrieve(
            ...     query_text='example query',
            ...     query_embedding=model.encode('example query'),
            ...     k=5
            ... )
        """
        # Compute similarity to all nodes
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), self.embeddings_matrix
        )[0]

        # Score each node
        node_scores = []
        for i, node_id in enumerate(self.node_ids):
            node = self.nodes[node_id]
            base_similarity = similarities[i]

            # Start with base similarity
            score = base_similarity

            # Boost by PageRank if enabled
            if use_pagerank:
                score *= 1.0 + node.pagerank_score

            node_scores.append((node_id, score, base_similarity))

        # Find query's likely community (based on most similar nodes)
        query_community = None
        if self.config.use_communities and community_boost > 0:
            top_similar_nodes = sorted(node_scores, key=lambda x: x[2], reverse=True)[
                :3
            ]
            community_counts = defaultdict(int)
            for node_id, _, _ in top_similar_nodes:
                comm_id = self.nodes[node_id].community_id
                if comm_id is not None:
                    community_counts[comm_id] += 1

            if community_counts:
                query_community = max(community_counts, key=community_counts.get)

            # Boost nodes in same community
            for i, (node_id, score, base_sim) in enumerate(node_scores):
                node = self.nodes[node_id]
                if node.community_id == query_community:
                    node_scores[i] = (node_id, score * (1.0 + community_boost), base_sim)

        # Sort by final score and get top-k
        node_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = node_scores[:k]

        # Format results
        results = []
        for node_id, final_score, base_similarity in top_k:
            node = self.nodes[node_id]
            result = {
                'text': node.text,
                'label': node.label,
                'similarity_score': float(base_similarity),
                'graph_score': float(final_score),
                'pagerank_score': node.pagerank_score,
                'community_id': node.community_id,
                'metadata': node.metadata,
            }
            results.append(result)

        return results

    def retrieve_with_graph_walk(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        k: int = 5,
        walk_length: int = 2,
    ) -> list[dict[str, Any]]:
        """Retrieve examples using random walk from top matches.

        This method first finds the most similar nodes, then performs random walks
        from those nodes to discover related examples that might not be directly similar
        but are connected through the graph structure.

        Args:
            query_text: Query text
            query_embedding: Embedding of query text
            k: Number of examples to retrieve
            walk_length: Length of random walks from seed nodes

        Returns:
            List of retrieved examples

        Example:
            >>> results = graph_rag.retrieve_with_graph_walk(
            ...     query_text='example query',
            ...     query_embedding=model.encode('example query'),
            ...     k=5,
            ...     walk_length=2
            ... )
        """
        # First, get top-3 most similar nodes as seeds
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), self.embeddings_matrix
        )[0]

        seed_indices = np.argsort(similarities)[::-1][:3]
        seed_node_ids = [self.node_ids[i] for i in seed_indices]

        # Perform random walks and collect visited nodes
        visited_nodes = defaultdict(int)  # node_id -> visit count

        for seed_id in seed_node_ids:
            visited_nodes[seed_id] += 1

            # Perform multiple random walks from each seed
            for _ in range(5):  # 5 walks per seed
                current_node = seed_id

                for _ in range(walk_length):
                    # Get neighbors
                    neighbors = list(self.graph.neighbors(current_node))
                    if not neighbors:
                        break

                    # Choose next node based on edge weights
                    weights = [
                        self.graph[current_node][neighbor].get('weight', 1.0)
                        for neighbor in neighbors
                    ]
                    weights = np.array(weights)
                    weights = weights / weights.sum()  # Normalize to probabilities

                    next_node = np.random.choice(neighbors, p=weights)
                    visited_nodes[next_node] += 1
                    current_node = next_node

        # Score nodes by visit count and similarity
        node_scores = []
        for node_id, visit_count in visited_nodes.items():
            node_idx = self.node_ids.index(node_id)
            similarity = similarities[node_idx]
            # Combine visit count (exploration) with similarity (exploitation)
            score = 0.7 * similarity + 0.3 * (visit_count / max(visited_nodes.values()))
            node_scores.append((node_id, score, similarity))

        # Sort and get top-k
        node_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = node_scores[:k]

        # Format results
        results = []
        for node_id, final_score, base_similarity in top_k:
            node = self.nodes[node_id]
            result = {
                'text': node.text,
                'label': node.label,
                'similarity_score': float(base_similarity),
                'walk_score': float(final_score),
                'visit_count': visited_nodes[node_id],
                'metadata': node.metadata,
            }
            results.append(result)

        return results

    def get_community_summary(self, community_id: int) -> dict[str, Any]:
        """Get summary statistics for a community.

        Args:
            community_id: ID of the community

        Returns:
            Dictionary with community statistics

        Example:
            >>> summary = graph_rag.get_community_summary(0)
            >>> print(f"Community has {summary['size']} nodes")
        """
        if community_id not in self.communities:
            return {'error': f'Community {community_id} not found'}

        node_ids = self.communities[community_id]
        nodes = [self.nodes[nid] for nid in node_ids]

        # Label distribution
        label_counts = defaultdict(int)
        for node in nodes:
            label_counts[node.label] += 1

        # Average PageRank
        avg_pagerank = np.mean([node.pagerank_score for node in nodes])

        return {
            'community_id': community_id,
            'size': len(nodes),
            'label_distribution': dict(label_counts),
            'avg_pagerank': float(avg_pagerank),
            'node_ids': node_ids[:10],  # Sample of node IDs
        }

    def save_graph(self, output_path: Path) -> None:
        """Save graph structure to disk.

        Args:
            output_path: Path to save the graph

        Example:
            >>> graph_rag.save_graph(Path('knowledge_bases/graph.pkl'))
        """
        import pickle

        output_path.parent.mkdir(parents=True, exist_ok=True)

        graph_data = {
            'graph': self.graph,
            'nodes': self.nodes,
            'embeddings_matrix': self.embeddings_matrix,
            'node_ids': self.node_ids,
            'communities': self.communities,
            'config': self.config,
        }

        with open(output_path, 'wb') as f:
            pickle.dump(graph_data, f)

        logger.info(f'Saved graph to {output_path}')

    @classmethod
    def load_graph(cls, input_path: Path) -> GraphRAG:
        """Load graph structure from disk.

        Args:
            input_path: Path to saved graph

        Returns:
            GraphRAG instance with loaded graph

        Example:
            >>> graph_rag = GraphRAG.load_graph(Path('knowledge_bases/graph.pkl'))
        """
        import pickle

        with open(input_path, 'rb') as f:
            graph_data = pickle.load(f)

        instance = cls(graph_data['config'])
        instance.graph = graph_data['graph']
        instance.nodes = graph_data['nodes']
        instance.embeddings_matrix = graph_data['embeddings_matrix']
        instance.node_ids = graph_data['node_ids']
        instance.communities = graph_data['communities']

        logger.info(f'Loaded graph from {input_path}')

        return instance
