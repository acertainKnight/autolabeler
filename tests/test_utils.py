"""Test utilities for Phase 2 features.

This module provides utilities for:
- Synthetic data generation for active learning
- Mock labeling functions for weak supervision
- Test fixtures for DSPy and RAG components
- Performance benchmarking helpers
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""

    n_samples: int = 1000
    n_features: int = 10
    n_classes: int = 2
    class_balance: Optional[list[float]] = None
    noise_level: float = 0.1
    random_seed: int = 42


class SyntheticDataGenerator:
    """Generate synthetic datasets for testing active learning and weak supervision."""

    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        """Initialize generator with configuration.

        Args:
            config: Data generation configuration
        """
        self.config = config or SyntheticDataConfig()
        np.random.seed(self.config.random_seed)

    def generate_classification_data(
        self,
        task_type: str = 'binary'
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Generate synthetic classification data.

        Args:
            task_type: Type of classification ('binary', 'multiclass')

        Returns:
            Tuple of (features, labels, dataframe)
        """
        n = self.config.n_samples
        d = self.config.n_features

        # Generate features from multivariate normal
        X = np.random.randn(n, d)

        # Generate labels based on linear combination of features + noise
        if task_type == 'binary':
            # Binary classification
            weights = np.random.randn(d)
            logits = X @ weights + np.random.randn(n) * self.config.noise_level
            y = (logits > 0).astype(int)
        else:
            # Multiclass classification
            n_classes = self.config.n_classes
            weights = np.random.randn(d, n_classes)
            logits = X @ weights + np.random.randn(n, n_classes) * self.config.noise_level
            y = np.argmax(logits, axis=1)

        # Create DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(d)])
        df['text'] = [f'Sample text {i}' for i in range(n)]
        df['label'] = y
        df['true_label'] = y  # Keep ground truth

        # Add embeddings column (simplified)
        df['embedding'] = [X[i].tolist() for i in range(n)]

        return X, y, df

    def generate_sentiment_data(
        self,
        n_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate synthetic sentiment analysis data.

        Args:
            n_samples: Number of samples (uses config if None)

        Returns:
            DataFrame with text and sentiment labels
        """
        n = n_samples or self.config.n_samples

        # Templates for different sentiments
        positive_templates = [
            'This is {adj}!',
            'I {verb} this product.',
            '{adj} experience overall.',
            'Highly {verb} this!',
            'Very {adj} and {adj2}.',
        ]

        negative_templates = [
            'This is {adj}.',
            'I {verb} this product.',
            'Very {adj} experience.',
            'Would not {verb}.',
            '{adj} and {adj2}.',
        ]

        neutral_templates = [
            'This is {adj}.',
            'It is {adj}.',
            '{adj} product overall.',
            'Nothing {adj}.',
        ]

        positive_adj = ['excellent', 'amazing', 'fantastic', 'wonderful', 'great']
        positive_adj2 = ['impressive', 'outstanding', 'exceptional', 'superb']
        positive_verbs = ['love', 'enjoy', 'recommend', 'appreciate']

        negative_adj = ['terrible', 'awful', 'horrible', 'disappointing', 'poor']
        negative_adj2 = ['frustrating', 'unsatisfactory', 'subpar', 'inadequate']
        negative_verbs = ['hate', 'dislike', 'regret', 'recommend']

        neutral_adj = ['okay', 'average', 'acceptable', 'decent', 'fine', 'special']

        data = []
        for i in range(n):
            sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.4, 0.2])

            if sentiment_type == 'positive':
                template = np.random.choice(positive_templates)
                text = template.format(
                    adj=np.random.choice(positive_adj),
                    adj2=np.random.choice(positive_adj2),
                    verb=np.random.choice(positive_verbs)
                )
                label = 'positive'
            elif sentiment_type == 'negative':
                template = np.random.choice(negative_templates)
                text = template.format(
                    adj=np.random.choice(negative_adj),
                    adj2=np.random.choice(negative_adj2),
                    verb=np.random.choice(negative_verbs)
                )
                label = 'negative'
            else:
                template = np.random.choice(neutral_templates)
                text = template.format(adj=np.random.choice(neutral_adj))
                label = 'neutral'

            # Add noise (some mislabeled examples)
            if np.random.rand() < self.config.noise_level:
                labels = ['positive', 'negative', 'neutral']
                labels.remove(label)
                label = np.random.choice(labels)

            data.append({
                'text': text,
                'label': label,
                'true_label': label,
                'confidence': np.random.uniform(0.6, 1.0)
            })

        return pd.DataFrame(data)

    def generate_active_learning_pool(
        self,
        n_labeled: int = 100,
        n_unlabeled: int = 1000
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate initial pool for active learning.

        Args:
            n_labeled: Number of initially labeled samples
            n_unlabeled: Number of unlabeled samples

        Returns:
            Tuple of (labeled_df, unlabeled_df)
        """
        # Generate full dataset
        X, y, df = self.generate_classification_data()

        # Split into labeled and unlabeled
        indices = np.random.permutation(len(df))
        labeled_indices = indices[:n_labeled]
        unlabeled_indices = indices[n_labeled:n_labeled + n_unlabeled]

        labeled_df = df.iloc[labeled_indices].copy()
        unlabeled_df = df.iloc[unlabeled_indices].copy()

        # Remove labels from unlabeled set
        unlabeled_df['label'] = None

        return labeled_df, unlabeled_df

    def generate_weak_supervision_data(
        self,
        n_samples: Optional[int] = None,
        coverage: float = 0.8
    ) -> tuple[pd.DataFrame, list[dict]]:
        """Generate data for weak supervision testing.

        Args:
            n_samples: Number of samples (uses config if None)
            coverage: Fraction of samples covered by labeling functions

        Returns:
            Tuple of (dataframe, labeling_function_configs)
        """
        n = n_samples or self.config.n_samples

        # Generate sentiment data
        df = self.generate_sentiment_data(n)

        # Create labeling function configurations
        lf_configs = [
            {
                'name': 'keyword_positive',
                'description': 'Detects positive keywords',
                'keywords': ['excellent', 'amazing', 'great', 'love'],
                'label': 'positive',
                'accuracy': 0.85
            },
            {
                'name': 'keyword_negative',
                'description': 'Detects negative keywords',
                'keywords': ['terrible', 'awful', 'hate', 'horrible'],
                'label': 'negative',
                'accuracy': 0.82
            },
            {
                'name': 'exclamation_positive',
                'description': 'Exclamation marks indicate positive sentiment',
                'pattern': r'!',
                'label': 'positive',
                'accuracy': 0.70
            },
            {
                'name': 'length_negative',
                'description': 'Very short reviews are often negative',
                'condition': lambda text: len(text.split()) < 5,
                'label': 'negative',
                'accuracy': 0.65
            }
        ]

        # Apply labeling functions with coverage
        for lf in lf_configs:
            col_name = f"lf_{lf['name']}"
            df[col_name] = -1  # -1 means abstain

            # Apply with coverage
            mask = np.random.rand(len(df)) < coverage
            for idx in df[mask].index:
                text = df.loc[idx, 'text']

                # Check if labeling function applies
                applies = False
                if 'keywords' in lf:
                    applies = any(kw in text.lower() for kw in lf['keywords'])
                elif 'pattern' in lf:
                    import re
                    applies = bool(re.search(lf['pattern'], text))
                elif 'condition' in lf:
                    applies = lf['condition'](text)

                if applies:
                    # Apply with noise based on accuracy
                    if np.random.rand() < lf['accuracy']:
                        df.loc[idx, col_name] = lf['label']
                    else:
                        # Mislabel
                        wrong_labels = [l for l in ['positive', 'negative', 'neutral'] if l != lf['label']]
                        df.loc[idx, col_name] = np.random.choice(wrong_labels)

        return df, lf_configs


class MockLabelingFunction:
    """Mock labeling function for weak supervision tests."""

    def __init__(
        self,
        name: str,
        accuracy: float = 0.8,
        coverage: float = 0.7,
        polarity: int = 1
    ):
        """Initialize mock labeling function.

        Args:
            name: Function name
            accuracy: Accuracy on covered examples
            coverage: Fraction of examples covered
            polarity: Label polarity (1 for positive, -1 for negative, 0 for neutral)
        """
        self.name = name
        self.accuracy = accuracy
        self.coverage = coverage
        self.polarity = polarity

    def __call__(self, text: str) -> int:
        """Apply labeling function to text.

        Args:
            text: Input text

        Returns:
            Label (-1 for abstain, 0+ for labels)
        """
        # Random coverage
        if np.random.rand() > self.coverage:
            return -1  # Abstain

        # Apply with accuracy
        if np.random.rand() < self.accuracy:
            return self.polarity
        else:
            # Random mislabel
            return np.random.choice([0, 1, 2])


def create_mock_labeling_functions(n_functions: int = 5) -> list[MockLabelingFunction]:
    """Create a set of mock labeling functions.

    Args:
        n_functions: Number of functions to create

    Returns:
        List of mock labeling functions
    """
    functions = []
    for i in range(n_functions):
        lf = MockLabelingFunction(
            name=f'lf_{i}',
            accuracy=np.random.uniform(0.6, 0.9),
            coverage=np.random.uniform(0.5, 0.8),
            polarity=np.random.choice([0, 1, 2])
        )
        functions.append(lf)
    return functions


class MockDSPyModule:
    """Mock DSPy module for testing."""

    def __init__(self, name: str = 'mock_module'):
        """Initialize mock module.

        Args:
            name: Module name
        """
        self.name = name
        self.call_count = 0
        self.cost_per_call = 0.001

    def forward(self, **kwargs):
        """Mock forward pass.

        Args:
            **kwargs: Input arguments

        Returns:
            Mock prediction
        """
        self.call_count += 1
        return {
            'prediction': 'positive',
            'confidence': 0.85,
            'reasoning': 'Mock reasoning'
        }

    def __call__(self, **kwargs):
        """Make module callable."""
        return self.forward(**kwargs)


class MockRAGRetriever:
    """Mock RAG retriever for testing."""

    def __init__(self, n_docs: int = 5):
        """Initialize mock retriever.

        Args:
            n_docs: Number of documents to return
        """
        self.n_docs = n_docs
        self.documents = [
            {'text': f'Document {i}', 'score': 1.0 - i * 0.1}
            for i in range(n_docs)
        ]

    def retrieve(self, query: str, k: Optional[int] = None) -> list[dict]:
        """Mock retrieval.

        Args:
            query: Query text
            k: Number of documents to retrieve

        Returns:
            List of mock documents
        """
        k = k or self.n_docs
        return self.documents[:k]


def generate_mock_embeddings(
    n_samples: int,
    dim: int = 384,
    normalize: bool = True
) -> np.ndarray:
    """Generate mock embeddings for testing.

    Args:
        n_samples: Number of samples
        dim: Embedding dimension
        normalize: Whether to L2 normalize

    Returns:
        Array of shape (n_samples, dim)
    """
    embeddings = np.random.randn(n_samples, dim)

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

    return embeddings


def create_cost_tracker() -> dict:
    """Create a mock cost tracker for testing.

    Returns:
        Dictionary with cost tracking methods
    """
    costs = {'total': 0.0, 'calls': 0, 'tokens': 0}

    def track_call(tokens: int, cost: float):
        costs['total'] += cost
        costs['calls'] += 1
        costs['tokens'] += tokens

    def get_stats():
        return costs.copy()

    def reset():
        costs['total'] = 0.0
        costs['calls'] = 0
        costs['tokens'] = 0

    return {
        'track': track_call,
        'get_stats': get_stats,
        'reset': reset
    }


class PerformanceBenchmark:
    """Benchmark helper for performance tests."""

    def __init__(self):
        """Initialize benchmark."""
        self.metrics = {
            'latency': [],
            'cost': [],
            'accuracy': [],
            'throughput': []
        }

    def record(
        self,
        latency: Optional[float] = None,
        cost: Optional[float] = None,
        accuracy: Optional[float] = None,
        throughput: Optional[float] = None
    ):
        """Record metrics.

        Args:
            latency: Latency in seconds
            cost: Cost in dollars
            accuracy: Accuracy (0-1)
            throughput: Throughput (samples/sec)
        """
        if latency is not None:
            self.metrics['latency'].append(latency)
        if cost is not None:
            self.metrics['cost'].append(cost)
        if accuracy is not None:
            self.metrics['accuracy'].append(accuracy)
        if throughput is not None:
            self.metrics['throughput'].append(throughput)

    def get_summary(self) -> dict:
        """Get benchmark summary statistics.

        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        return summary

    def assert_improvement(
        self,
        baseline_value: float,
        metric: str,
        improvement_threshold: float = 0.1,
        direction: str = 'lower'
    ) -> bool:
        """Assert that metric shows improvement over baseline.

        Args:
            baseline_value: Baseline metric value
            metric: Metric name ('latency', 'cost', 'accuracy', 'throughput')
            improvement_threshold: Required improvement (0-1, e.g., 0.1 = 10%)
            direction: 'lower' means lower is better, 'higher' means higher is better

        Returns:
            True if improvement threshold met

        Raises:
            AssertionError: If improvement not met
        """
        values = self.metrics.get(metric, [])
        if not values:
            raise ValueError(f'No values recorded for metric: {metric}')

        mean_value = np.mean(values)

        if direction == 'lower':
            improvement = (baseline_value - mean_value) / baseline_value
            assert improvement >= improvement_threshold, (
                f'{metric} improvement {improvement:.2%} < threshold {improvement_threshold:.2%} '
                f'(baseline: {baseline_value:.4f}, current: {mean_value:.4f})'
            )
        else:  # higher
            improvement = (mean_value - baseline_value) / baseline_value
            assert improvement >= improvement_threshold, (
                f'{metric} improvement {improvement:.2%} < threshold {improvement_threshold:.2%} '
                f'(baseline: {baseline_value:.4f}, current: {mean_value:.4f})'
            )

        return True


def create_mock_experiment_config() -> dict:
    """Create a mock experiment configuration for testing.

    Returns:
        Dictionary with experiment configuration
    """
    return {
        'name': 'test_experiment',
        'model': {
            'name': 'gpt-3.5-turbo',
            'temperature': 0.1,
            'max_tokens': 150
        },
        'optimization': {
            'method': 'BootstrapFewShot',
            'max_bootstrapped_demos': 4,
            'max_labeled_demos': 8,
            'max_rounds': 3
        },
        'active_learning': {
            'strategy': 'uncertainty',
            'batch_size': 10,
            'max_iterations': 5
        },
        'weak_supervision': {
            'label_model': 'majority_vote',
            'min_coverage': 0.5
        },
        'cost_limits': {
            'max_cost_per_sample': 0.01,
            'max_total_cost': 10.0
        }
    }
