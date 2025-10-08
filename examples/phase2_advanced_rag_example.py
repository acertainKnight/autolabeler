"""Example: Using advanced RAG methods (GraphRAG and RAPTOR) for retrieval.

This example demonstrates how to use GraphRAG and RAPTOR for improved
example retrieval and labeling consistency.
"""

import pandas as pd
from autolabeler.config import Settings
from autolabeler.core.configs import LabelingConfig, AdvancedRAGConfig
from autolabeler.core.labeling import OptimizedLabelingService
from autolabeler.core.rag import GraphRAGConfig, RAPTORConfig


def main():
    """Demonstrate advanced RAG methods."""

    # Sample data with more examples for better graph structure
    training_data = pd.DataFrame({
        'text': [
            'This product is absolutely amazing!',
            'Terrible quality, very disappointed.',
            'Pretty good for the price.',
            'Best purchase I ever made!',
            'Would not recommend to anyone.',
            'Decent product, does the job.',
            'Outstanding service and quality!',
            'Waste of money, awful experience.',
            'Satisfied with my purchase.',
            'Horrible, returned immediately.',
            'Exceeded all my expectations!',
            'Poor quality control.',
            'Good value for money.',
            'Fantastic product!',
            'Not worth the price.',
        ],
        'label': [
            'positive', 'negative', 'neutral',
            'positive', 'negative', 'neutral',
            'positive', 'negative', 'neutral',
            'negative', 'positive', 'negative',
            'neutral', 'positive', 'negative',
        ],
    })

    test_queries = [
        'Great value and quality!',
        'Disappointing purchase.',
        'It\'s okay, nothing special.',
    ]

    print('=' * 80)
    print('Advanced RAG Methods Example')
    print('=' * 80)
    print()

    settings = Settings()

    # ==========================================================================
    # Example 1: Traditional RAG (baseline)
    # ==========================================================================
    print('1. Traditional RAG (Baseline)')
    print('-' * 80)

    traditional_config = LabelingConfig(use_rag=True, k_examples=3)
    rag_config_traditional = AdvancedRAGConfig(rag_mode='traditional')

    service_traditional = OptimizedLabelingService(
        dataset_name='rag_demo_traditional',
        settings=settings,
        config=traditional_config,
        rag_config=rag_config_traditional,
    )

    # Add examples
    service_traditional.knowledge_store.add_examples(
        training_data, 'text', 'label', source='human'
    )

    print(f'Added {len(training_data)} examples to knowledge base')
    print()

    # Test retrieval
    print('Testing retrieval with traditional RAG:')
    for query in test_queries[:1]:  # Just test one
        examples = service_traditional.knowledge_store.find_similar_examples(
            query, k=3
        )
        print(f'\nQuery: "{query}"')
        print('Retrieved examples:')
        for i, ex in enumerate(examples, 1):
            print(f'  {i}. {ex["text"][:50]}... (sim: {ex["similarity_score"]:.3f})')
    print()

    # ==========================================================================
    # Example 2: GraphRAG
    # ==========================================================================
    print('2. GraphRAG (Graph-based Retrieval)')
    print('-' * 80)

    graph_config = LabelingConfig(use_rag=True, k_examples=3)
    rag_config_graph = AdvancedRAGConfig(
        rag_mode='graph',
        graph_similarity_threshold=0.6,  # Lower threshold for more edges
        graph_max_neighbors=8,
        graph_use_communities=True,
    )

    service_graph = OptimizedLabelingService(
        dataset_name='rag_demo_graph',
        settings=settings,
        config=graph_config,
        rag_config=rag_config_graph,
    )

    # Add examples
    service_graph.knowledge_store.add_examples(
        training_data, 'text', 'label', source='human'
    )

    # Build GraphRAG index
    print('Building GraphRAG index...')
    graph_rag_config = GraphRAGConfig(
        similarity_threshold=rag_config_graph.graph_similarity_threshold,
        max_neighbors=rag_config_graph.graph_max_neighbors,
        use_communities=rag_config_graph.graph_use_communities,
    )
    service_graph.knowledge_store.build_graph_rag(graph_rag_config)
    print('GraphRAG index built successfully!')
    print()

    # Get graph statistics
    stats = service_graph.knowledge_store.get_stats()
    if stats['graph_rag']['enabled']:
        print('Graph Statistics:')
        print(f'  Nodes:       {stats["graph_rag"]["num_nodes"]}')
        print(f'  Edges:       {stats["graph_rag"]["num_edges"]}')
        print(f'  Communities: {stats["graph_rag"]["num_communities"]}')
        print()

    # Test retrieval with GraphRAG
    print('Testing retrieval with GraphRAG:')
    for query in test_queries[:1]:
        examples = service_graph.knowledge_store.find_similar_examples_advanced(
            query, k=3, rag_mode='graph'
        )
        print(f'\nQuery: "{query}"')
        print('Retrieved examples (with graph scoring):')
        for i, ex in enumerate(examples, 1):
            graph_score = ex['metadata'].get('graph_score', 0)
            pagerank = ex['metadata'].get('pagerank_score', 0)
            print(
                f'  {i}. {ex["text"][:50]}... '
                f'(sim: {ex["similarity_score"]:.3f}, '
                f'graph: {graph_score:.3f}, '
                f'pagerank: {pagerank:.4f})'
            )
    print()

    # ==========================================================================
    # Example 3: RAPTOR
    # ==========================================================================
    print('3. RAPTOR (Tree-based Hierarchical Retrieval)')
    print('-' * 80)

    raptor_config = LabelingConfig(use_rag=True, k_examples=3)
    rag_config_raptor = AdvancedRAGConfig(
        rag_mode='raptor',
        raptor_max_tree_depth=2,  # Build 2-level tree
        raptor_min_cluster_size=2,
        raptor_use_multi_level=True,
    )

    service_raptor = OptimizedLabelingService(
        dataset_name='rag_demo_raptor',
        settings=settings,
        config=raptor_config,
        rag_config=rag_config_raptor,
    )

    # Add examples
    service_raptor.knowledge_store.add_examples(
        training_data, 'text', 'label', source='human'
    )

    # Build RAPTOR tree
    print('Building RAPTOR tree...')

    # Simple summarization function for demo
    def summarize_cluster(texts):
        """Simple summarization by taking common theme."""
        if len(texts) == 0:
            return 'Empty cluster'
        # In practice, use an LLM here
        return f'Summary of {len(texts)} examples with mixed sentiment'

    raptor_rag_config = RAPTORConfig(
        max_tree_depth=rag_config_raptor.raptor_max_tree_depth,
        min_cluster_size=rag_config_raptor.raptor_min_cluster_size,
        use_multi_level_retrieval=rag_config_raptor.raptor_use_multi_level,
    )
    service_raptor.knowledge_store.build_raptor_rag(
        raptor_rag_config, summarize_cluster
    )
    print('RAPTOR tree built successfully!')
    print()

    # Get tree statistics
    stats = service_raptor.knowledge_store.get_stats()
    if stats['raptor']['enabled']:
        print('Tree Statistics:')
        print(f'  Levels:      {stats["raptor"]["num_levels"]}')
        print(f'  Total Nodes: {stats["raptor"]["total_nodes"]}')
        print(f'  Leaf Nodes:  {stats["raptor"]["num_leaves"]}')
        print()

    # Test retrieval with RAPTOR
    print('Testing retrieval with RAPTOR:')
    for query in test_queries[:1]:
        examples = service_raptor.knowledge_store.find_similar_examples_advanced(
            query, k=3, rag_mode='raptor'
        )
        print(f'\nQuery: "{query}"')
        print('Retrieved examples (multi-level):')
        for i, ex in enumerate(examples, 1):
            level = ex['metadata'].get('level', 0)
            is_leaf = ex['metadata'].get('is_leaf', True)
            print(
                f'  {i}. {ex["text"][:50]}... '
                f'(sim: {ex["similarity_score"]:.3f}, '
                f'level: {level}, '
                f'leaf: {is_leaf})'
            )
    print()

    # ==========================================================================
    # Comparison
    # ==========================================================================
    print('4. Comparison Summary')
    print('-' * 80)
    print()
    print('Method Characteristics:')
    print('  Traditional RAG:')
    print('    + Simple and fast')
    print('    + Works well with small datasets')
    print('    - No relationship modeling')
    print()
    print('  GraphRAG:')
    print('    + Models document relationships')
    print('    + Community detection for themes')
    print('    + PageRank for importance')
    print('    - More complex to build')
    print()
    print('  RAPTOR:')
    print('    + Multi-scale retrieval')
    print('    + Hierarchical abstraction')
    print('    + Good for diverse datasets')
    print('    - Requires summarization')
    print()

    print('=' * 80)
    print('Example complete!')
    print('=' * 80)


if __name__ == '__main__':
    main()
