"""Shared fixtures for Phase 3 tests."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return pd.DataFrame({
        'text': [
            'This is a positive review',
            'This is a negative review',
            'Neutral feedback here',
            'Another positive comment',
            'Very negative experience',
        ] * 20,
        'label': [1, 0, 2, 1, 0] * 20,
        'confidence': np.random.uniform(0.6, 0.99, 100),
    })


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return np.random.randn(100, 384).astype(np.float32)


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    mock = Mock()
    mock.generate.return_value = {
        'text': 'Positive',
        'confidence': 0.85,
        'reasoning': 'The text expresses positive sentiment',
    }
    mock.batch_generate.return_value = [
        {'text': f'Label_{i}', 'confidence': 0.8 + i * 0.01}
        for i in range(10)
    ]
    return mock


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing."""
    mock = Mock()
    mock.encode.return_value = np.random.randn(384).astype(np.float32)
    mock.encode_batch.return_value = np.random.randn(10, 384).astype(np.float32)
    return mock


@pytest.fixture
def sample_preferences():
    """Sample preference data for DPO/RLHF."""
    return [
        {
            'prompt': 'Label this text: "Great product"',
            'chosen': 'Positive',
            'rejected': 'Negative',
            'margin': 0.8,
        },
        {
            'prompt': 'Label this text: "Terrible experience"',
            'chosen': 'Negative',
            'rejected': 'Positive',
            'margin': 0.9,
        },
        {
            'prompt': 'Label this text: "It works fine"',
            'chosen': 'Neutral',
            'rejected': 'Positive',
            'margin': 0.6,
        },
    ] * 10


@pytest.fixture
def constitutional_principles():
    """Sample constitutional principles."""
    return [
        {
            'name': 'accuracy',
            'description': 'Labels must be accurate and evidence-based',
            'check_fn': lambda x: x.get('confidence', 0) > 0.7,
        },
        {
            'name': 'consistency',
            'description': 'Similar inputs should have similar labels',
            'check_fn': lambda x: True,  # Simplified for testing
        },
        {
            'name': 'fairness',
            'description': 'Labels should not show demographic bias',
            'check_fn': lambda x: x.get('bias_score', 0) < 0.3,
        },
    ]


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test artifacts."""
    test_dir = tmp_path / 'phase3_tests'
    test_dir.mkdir()
    return test_dir


@pytest.fixture
def sample_drift_data():
    """Sample data for drift detection testing."""
    reference = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(5, 2, 1000),
        'feature_3': np.random.choice(['A', 'B', 'C'], 1000),
    })

    # Production data with drift
    production = pd.DataFrame({
        'feature_1': np.random.normal(0.5, 1.2, 500),  # Shifted mean
        'feature_2': np.random.normal(5, 2, 500),  # No drift
        'feature_3': np.random.choice(['A', 'B', 'C', 'D'], 500),  # New category
    })

    return reference, production


@pytest.fixture
def sample_agent_configs():
    """Sample agent configurations for multi-agent system."""
    return [
        {
            'agent_id': 'sentiment_specialist',
            'model': 'gpt-4',
            'specialization': 'sentiment_analysis',
            'confidence_threshold': 0.8,
            'temperature': 0.3,
        },
        {
            'agent_id': 'entity_extractor',
            'model': 'gpt-4',
            'specialization': 'entity_extraction',
            'confidence_threshold': 0.85,
            'temperature': 0.2,
        },
        {
            'agent_id': 'classification_expert',
            'model': 'claude-3-opus',
            'specialization': 'multi_class_classification',
            'confidence_threshold': 0.75,
            'temperature': 0.4,
        },
    ]


@pytest.fixture
def performance_data():
    """Sample performance data for benchmarking."""
    return {
        'latency_p50': 100.0,  # ms
        'latency_p95': 250.0,  # ms
        'latency_p99': 500.0,  # ms
        'throughput': 50.0,  # requests/sec
        'error_rate': 0.01,  # 1%
        'memory_usage': 512.0,  # MB
        'cpu_usage': 45.0,  # %
    }


@pytest.fixture
def mock_quality_estimator():
    """Mock quality estimator for STAPLE."""
    mock = Mock()
    mock.estimate_quality.return_value = {
        'consensus_labels': np.array([1, 0, 2] * 10),
        'quality_scores': np.array([0.85, 0.78, 0.92] * 10),
        'annotator_weights': np.array([0.9, 0.8, 0.85]),
        'converged': True,
        'iterations': 15,
    }
    return mock
