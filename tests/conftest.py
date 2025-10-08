"""Global pytest configuration and fixtures for AutoLabeler tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock
from typing import Any
import tempfile

# Try to import autolabeler components - they may not exist yet
try:
    from autolabeler.config import Settings
except ImportError:
    Settings = None


@pytest.fixture
def settings():
    """Create test settings with safe defaults."""
    if Settings is None:
        # Create a mock settings object for testing
        class MockSettings:
            llm_model = "gpt-3.5-turbo"
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            temperature = 0.1
            enable_dspy_optimization = False
            enable_advanced_rag = False
            openrouter_api_key = "test-key"
            max_examples_per_query = 5

        return MockSettings()

    return Settings(
        llm_model="gpt-3.5-turbo",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        temperature=0.1,
        enable_dspy_optimization=False,
        enable_advanced_rag=False,
        openrouter_api_key="test-key",
    )


@pytest.fixture
def sample_labeled_df():
    """Create sample labeled DataFrame for testing."""
    np.random.seed(42)

    data = {
        "text": [
            "This product is amazing! I love it.",
            "Terrible experience, would not recommend.",
            "It's okay, nothing special.",
            "Best purchase ever! Highly recommend.",
            "Waste of money, very disappointed.",
            "Pretty good, meets expectations.",
            "Excellent quality and fast shipping!",
            "Poor quality, broke after one use.",
            "Decent value for the price.",
            "Outstanding product, exceeded expectations!",
        ],
        "label": [
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative",
            "neutral",
            "positive",
        ],
        "confidence": [0.95, 0.90, 0.60, 0.98, 0.92, 0.65, 0.96, 0.88, 0.70, 0.97],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_unlabeled_df():
    """Create sample unlabeled DataFrame for testing."""
    data = {
        "text": [
            "I absolutely love this product!",
            "Not what I expected at all.",
            "It's pretty good overall.",
            "Very disappointed with the quality.",
            "Exceeded all my expectations!",
            "It's fine, nothing amazing.",
            "Fantastic purchase, very happy!",
            "Would not buy again.",
            "Good enough for the price.",
            "Perfect quality and value!",
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_calibration_data():
    """Generate sample calibration data with known miscalibration."""
    np.random.seed(42)
    n_samples = 1000

    # Simulate overconfident model (shifted toward 1.0)
    raw_scores = np.random.beta(8, 2, n_samples)

    # True labels (50/50 split)
    true_labels = np.random.binomial(1, 0.5, n_samples)

    # Predicted labels based on threshold
    predicted_labels = (raw_scores > 0.5).astype(int)

    return raw_scores, true_labels, predicted_labels


@pytest.fixture
def sample_multi_annotator_data():
    """Create sample data with multiple annotators for IAA testing."""
    data = {
        "text": [
            "This is great!",
            "This is terrible.",
            "This is okay.",
            "This is amazing!",
            "This is awful.",
        ],
        "annotator_1": ["positive", "negative", "neutral", "positive", "negative"],
        "annotator_2": ["positive", "negative", "neutral", "positive", "negative"],
        "annotator_3": ["positive", "negative", "positive", "positive", "negative"],
        "model_prediction": ["positive", "negative", "neutral", "positive", "neutral"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_synthetic_dataset():
    """Generate synthetic dataset for active learning tests."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate labels (simple linear classification)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["text"] = [f"Sample text {i}" for i in range(n_samples)]
    df["true_label"] = y

    # Generate embeddings (use subset of features)
    embeddings = X[:, :5]

    return df, embeddings, X, y


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing without API calls."""

    class MockLLMClient:
        def __init__(self):
            self.call_count = 0
            self.structured_output_mode = None

        def invoke(self, prompt: str | list[dict[str, Any]]) -> dict[str, Any]:
            self.call_count += 1
            return {
                "label": "positive",
                "confidence": 0.85,
                "reasoning": "Contains positive sentiment keywords",
            }

        def with_structured_output(self, schema, method="function_calling"):
            self.structured_output_mode = method
            return self

        def batch(self, prompts: list) -> list[dict[str, Any]]:
            return [self.invoke(p) for p in prompts]

    return MockLLMClient()


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing."""

    class MockEmbedder:
        def __init__(self, dim=384):
            self.dim = dim

        def encode(self, texts: list[str]) -> np.ndarray:
            # Generate deterministic fake embeddings based on text hash
            np.random.seed(sum(ord(c) for c in "".join(texts)))
            return np.random.randn(len(texts), self.dim)

    return MockEmbedder()


@pytest.fixture
def temp_storage_path(tmp_path):
    """Create temporary storage path for tests."""
    storage = tmp_path / "test_storage"
    storage.mkdir(exist_ok=True)
    return storage


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory with subdirectories."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (data_dir / "labeled").mkdir(exist_ok=True)
    (data_dir / "unlabeled").mkdir(exist_ok=True)
    (data_dir / "results").mkdir(exist_ok=True)
    (data_dir / "models").mkdir(exist_ok=True)

    return data_dir


@pytest.fixture
def sample_cost_data():
    """Create sample data with cost information."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "text": [f"Sample text {i}" for i in range(n_samples)],
        "label": np.random.choice(["positive", "negative", "neutral"], n_samples),
        "confidence": np.random.uniform(0.5, 1.0, n_samples),
        "llm_cost": np.random.uniform(0.001, 0.01, n_samples),  # Cost in dollars
        "latency_ms": np.random.uniform(100, 500, n_samples),
        "tokens_used": np.random.randint(50, 200, n_samples),
    }
    return pd.DataFrame(data)


# Helper functions for tests


def generate_synthetic_sentiment_data(n_samples: int = 100) -> pd.DataFrame:
    """Generate synthetic sentiment classification data."""
    np.random.seed(42)

    positive_templates = [
        "This is {adjective}!",
        "I {verb} this {noun}.",
        "{adjective} experience overall.",
    ]

    negative_templates = [
        "This is {adjective}.",
        "I {verb} this {noun}.",
        "Very {adjective} experience.",
    ]

    positive_adjectives = ["great", "excellent", "amazing", "fantastic", "wonderful"]
    negative_adjectives = ["terrible", "awful", "horrible", "disappointing", "poor"]
    verbs_positive = ["love", "enjoy", "recommend", "appreciate"]
    verbs_negative = ["hate", "dislike", "regret"]
    nouns = ["product", "service", "experience", "purchase"]

    data = []

    for _ in range(n_samples):
        if np.random.rand() > 0.5:
            # Positive sample
            template = np.random.choice(positive_templates)
            text = template.format(
                adjective=np.random.choice(positive_adjectives),
                verb=np.random.choice(verbs_positive),
                noun=np.random.choice(nouns),
            )
            label = "positive"
        else:
            # Negative sample
            template = np.random.choice(negative_templates)
            text = template.format(
                adjective=np.random.choice(negative_adjectives),
                verb=np.random.choice(verbs_negative),
                noun=np.random.choice(nouns),
            )
            label = "negative"

        data.append({"text": text, "label": label})

    return pd.DataFrame(data)


def compute_ece(confidence_scores: np.ndarray, true_labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidence_scores, bins) - 1

    ece = 0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_confidence = confidence_scores[mask].mean()
            bin_accuracy = true_labels[mask].mean()
            ece += mask.sum() / len(true_labels) * abs(bin_confidence - bin_accuracy)

    return ece


# Pytest configuration


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (slower, multiple components)")
    config.addinivalue_line("markers", "performance: Performance/benchmark tests")
    config.addinivalue_line("markers", "validation: Validation tests on benchmark datasets")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "requires_api: Tests requiring external API access")
