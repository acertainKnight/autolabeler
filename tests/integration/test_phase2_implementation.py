"""Integration tests for Phase 2: DSPy optimization and Advanced RAG."""

import numpy as np
import pandas as pd
import pytest

# These imports will work when DSPy and other dependencies are installed
pytest.importorskip('dspy', reason='DSPy not installed')


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return pd.DataFrame({
        'text': [
            'This product is amazing!',
            'Terrible experience, would not recommend.',
            'Pretty good overall, happy with purchase.',
            'Worst product ever made.',
            'Absolutely love it, five stars!',
            'Not bad, but could be better.',
            'Completely satisfied with quality.',
            'Disappointed with the results.',
        ],
        'label': [
            'positive',
            'negative',
            'positive',
            'negative',
            'positive',
            'neutral',
            'positive',
            'negative',
        ],
    })


@pytest.mark.integration
@pytest.mark.requires_api
class TestDSPyOptimization:
    """Tests for DSPy prompt optimization."""

    def test_dspy_config_creation(self):
        """Test DSPy configuration creation."""
        from autolabeler.core.optimization import DSPyConfig

        config = DSPyConfig(
            model_name='gpt-4o-mini',
            num_candidates=5,
            num_trials=10,
        )

        assert config.model_name == 'gpt-4o-mini'
        assert config.num_candidates == 5
        assert config.num_trials == 10

    def test_dspy_optimizer_initialization(self):
        """Test DSPy optimizer initialization."""
        from autolabeler.core.optimization import DSPyOptimizer, DSPyConfig

        config = DSPyConfig(model_name='gpt-4o-mini')
        optimizer = DSPyOptimizer(config)

        assert optimizer.config == config
        assert optimizer.lm is not None

    @pytest.mark.slow
    def test_dspy_optimization_basic(self, sample_data):
        """Test basic DSPy optimization flow."""
        from autolabeler.core.optimization import DSPyOptimizer, DSPyConfig

        # Split data
        train_df = sample_data.iloc[:6]
        val_df = sample_data.iloc[6:]

        # Create optimizer with minimal config for testing
        config = DSPyConfig(
            model_name='gpt-4o-mini',
            num_candidates=2,  # Small for fast testing
            num_trials=2,  # Small for fast testing
        )
        optimizer = DSPyOptimizer(config)

        # Run optimization
        result = optimizer.optimize_labeling_prompt(
            train_df=train_df,
            val_df=val_df,
            text_column='text',
            label_column='label',
        )

        assert result is not None
        assert 0 <= result.validation_accuracy <= 1.0
        assert 0 <= result.train_accuracy <= 1.0
        assert result.optimization_cost >= 0
        assert result.best_prompt != ''


@pytest.mark.integration
class TestGraphRAG:
    """Tests for GraphRAG implementation."""

    def test_graph_rag_config_creation(self):
        """Test GraphRAG configuration."""
        from autolabeler.core.rag import GraphRAGConfig

        config = GraphRAGConfig(
            similarity_threshold=0.7,
            max_neighbors=10,
        )

        assert config.similarity_threshold == 0.7
        assert config.max_neighbors == 10

    def test_graph_rag_initialization(self):
        """Test GraphRAG initialization."""
        from autolabeler.core.rag import GraphRAG, GraphRAGConfig

        config = GraphRAGConfig()
        graph_rag = GraphRAG(config)

        assert graph_rag.config == config
        assert len(graph_rag.nodes) == 0

    def test_graph_rag_build_and_retrieve(self, sample_data):
        """Test GraphRAG build and retrieval."""
        from autolabeler.core.rag import GraphRAG, GraphRAGConfig

        # Mock embedding function
        def mock_embedding(text):
            # Simple mock: hash to fixed-size vector
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(384)

        config = GraphRAGConfig(similarity_threshold=0.5)
        graph_rag = GraphRAG(config)

        # Build graph
        graph_rag.build_graph(
            df=sample_data,
            text_column='text',
            label_column='label',
            embedding_fn=mock_embedding,
        )

        assert len(graph_rag.nodes) == len(sample_data)
        assert graph_rag.graph.number_of_nodes() == len(sample_data)

        # Test retrieval
        query_text = 'Great product, very satisfied!'
        query_embedding = mock_embedding(query_text)

        results = graph_rag.retrieve(
            query_text=query_text,
            query_embedding=query_embedding,
            k=3,
        )

        assert len(results) <= 3
        assert all('text' in r for r in results)
        assert all('label' in r for r in results)
        assert all('similarity_score' in r for r in results)


@pytest.mark.integration
class TestRAPTORRAG:
    """Tests for RAPTOR implementation."""

    def test_raptor_config_creation(self):
        """Test RAPTOR configuration."""
        from autolabeler.core.rag import RAPTORConfig

        config = RAPTORConfig(
            max_tree_depth=3,
            min_cluster_size=2,
        )

        assert config.max_tree_depth == 3
        assert config.min_cluster_size == 2

    def test_raptor_initialization(self):
        """Test RAPTOR initialization."""
        from autolabeler.core.rag import RAPTORRAG, RAPTORConfig

        config = RAPTORConfig()
        raptor = RAPTORRAG(config)

        assert raptor.config == config
        assert len(raptor.nodes) == 0

    def test_raptor_build_and_retrieve(self, sample_data):
        """Test RAPTOR build and retrieval."""
        from autolabeler.core.rag import RAPTORRAG, RAPTORConfig

        # Mock embedding function
        def mock_embedding(text):
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(384)

        # Mock summarization function
        def mock_summarize(texts):
            return f'Summary of {len(texts)} texts'

        config = RAPTORConfig(max_tree_depth=2, min_cluster_size=2)
        raptor = RAPTORRAG(config)

        # Build tree
        raptor.build_tree(
            df=sample_data,
            text_column='text',
            label_column='label',
            embedding_fn=mock_embedding,
            summarize_fn=mock_summarize,
        )

        assert len(raptor.leaf_nodes) == len(sample_data)
        assert len(raptor.levels) >= 1

        # Test retrieval
        query_text = 'Great product, very satisfied!'
        query_embedding = mock_embedding(query_text)

        results = raptor.retrieve(
            query_text=query_text,
            query_embedding=query_embedding,
            k=3,
        )

        assert len(results) <= 3
        assert all('text' in r for r in results)
        assert all('similarity_score' in r for r in results)


@pytest.mark.integration
class TestKnowledgeStoreAdvancedRAG:
    """Tests for KnowledgeStore with advanced RAG."""

    def test_knowledge_store_with_rag_mode(self, sample_data):
        """Test KnowledgeStore initialization with different RAG modes."""
        from autolabeler.config import Settings
        from autolabeler.core.knowledge import KnowledgeStore

        settings = Settings()

        # Test traditional mode
        store_traditional = KnowledgeStore(
            'test_traditional', settings, rag_mode='traditional'
        )
        assert store_traditional.rag_mode == 'traditional'

        # Test graph mode
        store_graph = KnowledgeStore('test_graph', settings, rag_mode='graph')
        assert store_graph.rag_mode == 'graph'

        # Test raptor mode
        store_raptor = KnowledgeStore('test_raptor', settings, rag_mode='raptor')
        assert store_raptor.rag_mode == 'raptor'

    def test_knowledge_store_advanced_retrieval(self, sample_data, tmp_path):
        """Test advanced RAG retrieval through KnowledgeStore."""
        from autolabeler.config import Settings
        from autolabeler.core.knowledge import KnowledgeStore

        settings = Settings()
        store = KnowledgeStore(
            'test_advanced', settings, store_dir=tmp_path / 'kb', rag_mode='traditional'
        )

        # Add examples
        store.add_examples(sample_data, 'text', 'label', source='human')

        # Test traditional retrieval
        results = store.find_similar_examples_advanced(
            'Great product!', k=3, rag_mode='traditional'
        )

        assert len(results) <= 3
        assert all('text' in r for r in results)


@pytest.mark.integration
@pytest.mark.requires_api
class TestOptimizedLabelingService:
    """Tests for OptimizedLabelingService."""

    def test_optimized_service_initialization(self):
        """Test OptimizedLabelingService initialization."""
        from autolabeler.config import Settings
        from autolabeler.core.labeling import OptimizedLabelingService
        from autolabeler.core.configs import (
            LabelingConfig,
            DSPyOptimizationConfig,
            AdvancedRAGConfig,
        )

        settings = Settings()
        config = LabelingConfig()
        dspy_config = DSPyOptimizationConfig(enabled=False)  # Disabled for test
        rag_config = AdvancedRAGConfig(rag_mode='traditional')

        service = OptimizedLabelingService(
            dataset_name='test',
            settings=settings,
            config=config,
            dspy_config=dspy_config,
            rag_config=rag_config,
        )

        assert service.dspy_config.enabled == False
        assert service.rag_config.rag_mode == 'traditional'

    def test_get_optimization_stats(self):
        """Test optimization statistics retrieval."""
        from autolabeler.config import Settings
        from autolabeler.core.labeling import OptimizedLabelingService
        from autolabeler.core.configs import DSPyOptimizationConfig, AdvancedRAGConfig

        settings = Settings()
        dspy_config = DSPyOptimizationConfig(enabled=False)
        rag_config = AdvancedRAGConfig(rag_mode='traditional')

        service = OptimizedLabelingService(
            dataset_name='test',
            settings=settings,
            dspy_config=dspy_config,
            rag_config=rag_config,
        )

        stats = service.get_optimization_stats()

        assert 'dspy_enabled' in stats
        assert 'rag_mode' in stats
        assert stats['rag_mode'] == 'traditional'


@pytest.mark.unit
class TestConfigClasses:
    """Tests for configuration classes."""

    def test_dspy_optimization_config(self):
        """Test DSPyOptimizationConfig."""
        from autolabeler.core.configs import DSPyOptimizationConfig

        config = DSPyOptimizationConfig(
            enabled=True,
            model_name='gpt-4o-mini',
            num_candidates=15,
        )

        assert config.enabled == True
        assert config.model_name == 'gpt-4o-mini'
        assert config.num_candidates == 15

    def test_advanced_rag_config(self):
        """Test AdvancedRAGConfig."""
        from autolabeler.core.configs import AdvancedRAGConfig

        config = AdvancedRAGConfig(
            rag_mode='graph',
            graph_similarity_threshold=0.8,
            raptor_max_tree_depth=4,
        )

        assert config.rag_mode == 'graph'
        assert config.graph_similarity_threshold == 0.8
        assert config.raptor_max_tree_depth == 4
