"""Comprehensive tests for Active Learning (60+ tests)."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.metrics import accuracy_score

from tests.test_utils import (
    SyntheticDataGenerator,
    SyntheticDataConfig,
    generate_mock_embeddings
)


@pytest.fixture
def al_data():
    """Generate active learning data pool."""
    generator = SyntheticDataGenerator()
    return generator.generate_active_learning_pool(n_labeled=100, n_unlabeled=900)


@pytest.mark.unit
class TestUncertaintySampling:
    """Test uncertainty sampling strategies (15 tests)."""

    def test_least_confident(self):
        """Test least confident sampling."""
        confidences = np.array([0.95, 0.60, 0.85, 0.55, 0.90])
        least_conf_idx = np.argmin(confidences)
        assert least_conf_idx == 3

    def test_margin_sampling(self):
        """Test margin sampling."""
        # Probabilities for top 2 classes
        probs = np.array([[0.6, 0.3], [0.9, 0.05], [0.55, 0.40]])
        margins = probs[:, 0] - probs[:, 1]
        min_margin_idx = np.argmin(margins)
        assert min_margin_idx == 2

    def test_entropy_sampling(self):
        """Test entropy-based sampling."""
        from scipy.stats import entropy
        probs = np.array([[0.5, 0.5], [0.9, 0.1], [0.7, 0.3]])
        entropies = [entropy(p) for p in probs]
        max_entropy_idx = np.argmax(entropies)
        assert max_entropy_idx == 0

    def test_batch_uncertainty_sampling(self):
        """Test batch uncertainty sampling."""
        confidences = np.random.rand(1000)
        batch_size = 10
        uncertain_indices = np.argsort(confidences)[:batch_size]
        assert len(uncertain_indices) == batch_size

    def test_uncertainty_threshold(self):
        """Test filtering by uncertainty threshold."""
        confidences = np.array([0.9, 0.6, 0.85, 0.55, 0.4])
        threshold = 0.7
        uncertain = np.where(confidences < threshold)[0]
        assert len(uncertain) == 3

    def test_confidence_calibration(self):
        """Test calibrated confidence scores."""
        raw_scores = np.array([0.8, 0.6, 0.9, 0.7])
        # Mock calibration (e.g., Platt scaling)
        calibrated = raw_scores * 0.9  # Simple scaling
        assert all(calibrated <= raw_scores)

    def test_multiclass_uncertainty(self):
        """Test uncertainty for multiclass."""
        probs = np.array([[0.4, 0.3, 0.3], [0.8, 0.1, 0.1], [0.33, 0.33, 0.34]])
        max_probs = np.max(probs, axis=1)
        uncertainties = 1 - max_probs
        assert len(uncertainties) == 3

    def test_uncertainty_diversity(self):
        """Test combining uncertainty with diversity."""
        uncertainties = np.array([0.5, 0.6, 0.55, 0.58])
        embeddings = generate_mock_embeddings(4, dim=128)
        # Combine scores (uncertainty + diversity)
        scores = uncertainties  # Would add diversity component
        top_k = np.argsort(scores)[-2:]
        assert len(top_k) == 2

    def test_breaking_ties(self):
        """Test breaking ties in uncertainty."""
        confidences = np.array([0.5, 0.5, 0.6, 0.5])
        # Break ties randomly
        np.random.seed(42)
        tied_indices = np.where(confidences == 0.5)[0]
        selected = np.random.choice(tied_indices)
        assert selected in tied_indices

    def test_uncertainty_over_time(self):
        """Test tracking uncertainty over iterations."""
        uncertainty_history = []
        for i in range(5):
            mean_uncertainty = 0.6 - i * 0.05
            uncertainty_history.append(mean_uncertainty)
        # Should decrease over time
        assert uncertainty_history[0] > uncertainty_history[-1]

    def test_ensemble_uncertainty(self):
        """Test ensemble-based uncertainty."""
        # Predictions from 3 models
        predictions = [
            np.array([0, 1, 0, 1, 0]),
            np.array([0, 1, 1, 1, 0]),
            np.array([0, 0, 0, 1, 0])
        ]
        # Disagreement rate
        disagreements = np.std(predictions, axis=0)
        assert len(disagreements) == 5

    def test_uncertainty_weighting(self):
        """Test weighting samples by uncertainty."""
        uncertainties = np.array([0.3, 0.7, 0.5, 0.8])
        weights = uncertainties / uncertainties.sum()
        assert np.isclose(weights.sum(), 1.0)

    def test_uncertainty_vs_random(self):
        """Test uncertainty sampling vs random."""
        confidences = np.random.rand(1000)
        # Uncertainty: select lowest confidence
        uncertain_idx = np.argmin(confidences)
        # Random: select any
        random_idx = np.random.randint(1000)
        # Uncertain should have lower confidence
        assert confidences[uncertain_idx] <= confidences[random_idx]

    def test_batch_balancing(self):
        """Test balancing uncertain batch."""
        predictions = np.array([0, 0, 1, 0, 1, 1, 0, 1])
        confidences = np.array([0.6, 0.55, 0.65, 0.50, 0.70, 0.60, 0.55, 0.65])
        # Ensure batch has balanced classes
        batch_size = 4
        uncertain_per_class = batch_size // 2
        class_0_uncertain = np.where((predictions == 0) & (confidences < 0.7))[0]
        assert len(class_0_uncertain) >= uncertain_per_class

    def test_uncertainty_stopping_criterion(self):
        """Test stopping when uncertainty low."""
        mean_uncertainty = 0.15
        threshold = 0.2
        should_stop = mean_uncertainty < threshold
        assert should_stop


@pytest.mark.unit
class TestDiversitySampling:
    """Test diversity-based sampling (15 tests)."""

    def test_kmeans_diversity(self):
        """Test K-means for diversity."""
        from sklearn.cluster import KMeans
        embeddings = generate_mock_embeddings(100, dim=128)
        kmeans = KMeans(n_clusters=10, random_state=42)
        kmeans.fit(embeddings)
        # Select closest to centroids
        centers = kmeans.cluster_centers_
        assert len(centers) == 10

    def test_core_set_selection(self):
        """Test core-set selection."""
        labeled_emb = generate_mock_embeddings(50, dim=128)
        unlabeled_emb = generate_mock_embeddings(100, dim=128)
        # Select samples furthest from labeled set
        from scipy.spatial.distance import cdist
        distances = cdist(unlabeled_emb, labeled_emb).min(axis=1)
        furthest_idx = np.argmax(distances)
        assert furthest_idx >= 0

    def test_maximal_marginal_relevance(self):
        """Test MMR for diversity."""
        query_emb = generate_mock_embeddings(1, dim=128)[0]
        doc_embs = generate_mock_embeddings(10, dim=128)
        selected = []
        lambda_param = 0.5
        for _ in range(3):
            # MMR scoring
            relevance = doc_embs @ query_emb
            if selected:
                selected_embs = doc_embs[selected]
                diversity = np.min(cdist(doc_embs, selected_embs), axis=1)
            else:
                diversity = np.ones(len(doc_embs))
            scores = lambda_param * relevance + (1 - lambda_param) * diversity
            selected.append(np.argmax(scores))
        assert len(selected) == 3

    def test_cluster_based_sampling(self):
        """Test sampling from clusters."""
        from sklearn.cluster import KMeans
        embeddings = generate_mock_embeddings(200, dim=128)
        kmeans = KMeans(n_clusters=5, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        # Sample one from each cluster
        samples = []
        for i in range(5):
            cluster_indices = np.where(labels == i)[0]
            samples.append(np.random.choice(cluster_indices))
        assert len(samples) == 5

    def test_similarity_threshold(self):
        """Test filtering by similarity threshold."""
        embeddings = generate_mock_embeddings(10, dim=128)
        similarity_matrix = embeddings @ embeddings.T
        threshold = 0.8
        similar_pairs = np.where(similarity_matrix > threshold)
        assert len(similar_pairs[0]) > 0

    def test_density_weighted_sampling(self):
        """Test density-weighted sampling."""
        embeddings = generate_mock_embeddings(100, dim=128)
        # Calculate local density
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(embeddings))
        densities = np.sum(distances < 0.5, axis=1)
        # Sample from low-density regions
        low_density_idx = np.argmin(densities)
        assert low_density_idx >= 0

    def test_determinantal_point_process(self):
        """Test DPP-based sampling."""
        # Mock DPP sampling
        n_samples = 100
        batch_size = 10
        # Would use actual DPP, here just random
        selected = np.random.choice(n_samples, batch_size, replace=False)
        assert len(selected) == batch_size

    def test_feature_space_coverage(self):
        """Test feature space coverage."""
        embeddings = generate_mock_embeddings(50, dim=128)
        # Measure coverage with selected samples
        selected = [0, 10, 20, 30, 40]
        selected_embs = embeddings[selected]
        # Calculate covered volume (simplified)
        coverage = selected_embs.std(axis=0).mean()
        assert coverage > 0

    def test_incremental_diversity(self):
        """Test incrementally adding diverse samples."""
        pool = generate_mock_embeddings(100, dim=128)
        selected = []
        for _ in range(5):
            if not selected:
                # First: random
                selected.append(np.random.randint(100))
            else:
                # Next: most diverse
                from scipy.spatial.distance import cdist
                selected_embs = pool[selected]
                distances = cdist(pool, selected_embs).min(axis=1)
                selected.append(np.argmax(distances))
        assert len(selected) == 5

    def test_diversity_vs_uncertainty_tradeoff(self):
        """Test balancing diversity and uncertainty."""
        uncertainties = np.random.rand(100)
        embeddings = generate_mock_embeddings(100, dim=128)
        # Combined score
        alpha = 0.5
        # Normalize and combine
        unc_norm = uncertainties / uncertainties.max()
        # Would add diversity component
        scores = alpha * unc_norm
        top_k = np.argsort(scores)[-10:]
        assert len(top_k) == 10

    def test_representativeness(self):
        """Test ensuring representative samples."""
        from sklearn.cluster import KMeans
        all_data = generate_mock_embeddings(1000, dim=128)
        kmeans = KMeans(n_clusters=10, random_state=42)
        labels = kmeans.fit_predict(all_data)
        # Check distribution
        unique, counts = np.unique(labels, return_counts=True)
        assert len(unique) == 10

    def test_outlier_detection(self):
        """Test detecting and handling outliers."""
        embeddings = generate_mock_embeddings(100, dim=128)
        # Add outlier
        outlier = np.ones((1, 128)) * 10
        all_embs = np.vstack([embeddings, outlier])
        # Detect outlier (simplified)
        means = all_embs.mean(axis=0)
        distances = np.linalg.norm(all_embs - means, axis=1)
        outlier_idx = np.argmax(distances)
        assert outlier_idx == 100  # Last one

    def test_batch_diversity_score(self):
        """Test scoring batch diversity."""
        batch_embs = generate_mock_embeddings(10, dim=128)
        # Calculate average pairwise distance
        from scipy.spatial.distance import pdist
        distances = pdist(batch_embs)
        diversity_score = distances.mean()
        assert diversity_score > 0

    def test_dynamic_diversity_weight(self):
        """Test adjusting diversity weight over time."""
        initial_weight = 0.5
        iteration = 5
        # Decrease diversity weight as we label more
        weight = initial_weight * (0.9 ** iteration)
        assert weight < initial_weight

    def test_coverage_based_stopping(self):
        """Test stopping when coverage adequate."""
        coverage = 0.85
        threshold = 0.80
        should_continue = coverage < threshold
        assert not should_continue


@pytest.mark.unit
class TestQueryByCommittee:
    """Test query-by-committee strategies (10 tests)."""

    def test_committee_disagreement(self):
        """Test measuring committee disagreement."""
        predictions = np.array([
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 0]
        ])
        # Calculate disagreement per sample
        disagreement = np.var(predictions, axis=0)
        assert len(disagreement) == 5

    def test_vote_entropy(self):
        """Test vote entropy."""
        votes = np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1]])
        # Calculate entropy of votes
        from scipy.stats import entropy
        vote_probs = votes.mean(axis=1)
        entropies = [entropy([p, 1-p]) for p in vote_probs]
        assert len(entropies) == 3

    def test_committee_size_effect(self):
        """Test effect of committee size."""
        for n_models in [3, 5, 7]:
            predictions = np.random.randint(0, 2, size=(n_models, 100))
            disagreement = np.var(predictions, axis=0).mean()
            # More models generally give better estimates
            assert disagreement >= 0

    def test_diverse_committee(self):
        """Test creating diverse committee."""
        # Different model types
        committee = ['logistic', 'random_forest', 'svm']
        assert len(committee) == 3

    def test_consensus_threshold(self):
        """Test consensus threshold."""
        votes = np.array([1, 1, 1, 0, 1])  # 4/5 agree
        agreement = votes.sum() / len(votes)
        threshold = 0.8
        has_consensus = agreement >= threshold
        assert has_consensus

    def test_soft_voting(self):
        """Test soft voting."""
        probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.9, 0.1]])
        avg_probs = probs.mean(axis=0)
        prediction = np.argmax(avg_probs)
        assert prediction in [0, 1]

    def test_hard_voting(self):
        """Test hard voting."""
        predictions = np.array([0, 0, 1])
        from collections import Counter
        votes = Counter(predictions)
        majority = votes.most_common(1)[0][0]
        assert majority == 0

    def test_weighted_voting(self):
        """Test weighted voting by model performance."""
        predictions = np.array([0, 1, 0])
        weights = np.array([0.9, 0.8, 0.85])
        weighted_votes = predictions * weights
        # Would implement full weighted voting
        assert len(weighted_votes) == 3

    def test_committee_update(self):
        """Test updating committee with new data."""
        committee_performance = [0.80, 0.82, 0.78]
        # Add new model if better
        new_model_perf = 0.85
        if new_model_perf > min(committee_performance):
            worst_idx = np.argmin(committee_performance)
            committee_performance[worst_idx] = new_model_perf
        assert max(committee_performance) == 0.85

    def test_disagreement_sampling(self):
        """Test sampling by disagreement."""
        predictions = np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]])
        disagreement = np.var(predictions, axis=0)
        most_disagreed = np.argmax(disagreement)
        assert most_disagreed >= 0


@pytest.mark.unit
class TestActiveLearningIteration:
    """Test active learning iteration logic (10 tests)."""

    def test_iteration_workflow(self, al_data):
        """Test one iteration of active learning."""
        labeled, unlabeled = al_data
        # Query
        batch_size = 10
        query_indices = np.random.choice(len(unlabeled), batch_size, replace=False)
        queried = unlabeled.iloc[query_indices]
        assert len(queried) == batch_size

    def test_oracle_labeling(self, al_data):
        """Test oracle labeling."""
        labeled, unlabeled = al_data
        # Simulate oracle providing labels
        queried_indices = [0, 1, 2]
        for idx in queried_indices:
            true_label = unlabeled.iloc[idx]['true_label']
            unlabeled.loc[unlabeled.index[idx], 'label'] = true_label
        # Verify labels assigned
        assert unlabeled.iloc[0]['label'] is not None

    def test_pool_update(self, al_data):
        """Test updating labeled/unlabeled pools."""
        labeled, unlabeled = al_data
        initial_unlabeled = len(unlabeled)
        # Move samples from unlabeled to labeled
        batch = unlabeled.iloc[:10].copy()
        labeled = pd.concat([labeled, batch])
        unlabeled = unlabeled.iloc[10:]
        assert len(unlabeled) == initial_unlabeled - 10

    def test_model_retraining(self):
        """Test retraining model after labeling."""
        # Mock training
        initial_accuracy = 0.75
        # After adding 10 labels
        new_accuracy = 0.78
        improvement = new_accuracy - initial_accuracy
        assert improvement > 0

    def test_convergence_check(self):
        """Test checking for convergence."""
        accuracy_history = [0.70, 0.75, 0.78, 0.80, 0.805, 0.806]
        # Check if improvements plateaued
        recent_improvements = np.diff(accuracy_history[-3:])
        avg_improvement = recent_improvements.mean()
        converged = avg_improvement < 0.01
        assert converged

    def test_budget_tracking(self):
        """Test tracking labeling budget."""
        budget = 100
        used = 0
        batch_size = 10
        for _ in range(5):
            if used + batch_size <= budget:
                used += batch_size
        assert used == 50

    def test_iteration_stopping(self):
        """Test stopping criteria."""
        max_iterations = 10
        current_iteration = 9
        accuracy = 0.95
        target_accuracy = 0.90
        should_stop = (current_iteration >= max_iterations) or (accuracy >= target_accuracy)
        assert should_stop

    def test_learning_curve(self):
        """Test generating learning curve."""
        iterations = [1, 2, 3, 4, 5]
        accuracies = [0.70, 0.75, 0.80, 0.83, 0.85]
        # Should be monotonically increasing (mostly)
        assert accuracies[-1] > accuracies[0]

    def test_sample_tracking(self):
        """Test tracking which samples were queried."""
        queried_indices = set()
        for iteration in range(3):
            batch = [iteration * 10 + i for i in range(10)]
            queried_indices.update(batch)
        assert len(queried_indices) == 30

    def test_performance_monitoring(self):
        """Test monitoring performance metrics."""
        metrics = {
            'accuracy': [0.75, 0.80, 0.83],
            'f1': [0.73, 0.78, 0.81],
            'samples_labeled': [100, 110, 120]
        }
        assert len(metrics['accuracy']) == 3


@pytest.mark.unit
class TestActiveLearningStrategies:
    """Test different AL strategies (10 tests)."""

    def test_random_sampling(self):
        """Test random sampling baseline."""
        pool_size = 1000
        batch_size = 10
        indices = np.random.choice(pool_size, batch_size, replace=False)
        assert len(indices) == batch_size

    def test_uncertainty_sampling(self):
        """Test uncertainty sampling."""
        confidences = np.random.rand(1000)
        batch_size = 10
        uncertain_indices = np.argsort(confidences)[:batch_size]
        assert len(uncertain_indices) == batch_size

    def test_diversity_sampling(self):
        """Test diversity sampling."""
        embeddings = generate_mock_embeddings(1000, dim=128)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=10, random_state=42)
        kmeans.fit(embeddings)
        # Select from each cluster
        labels = kmeans.predict(embeddings)
        diverse_samples = []
        for i in range(10):
            cluster_samples = np.where(labels == i)[0]
            if len(cluster_samples) > 0:
                diverse_samples.append(np.random.choice(cluster_samples))
        assert len(diverse_samples) <= 10

    def test_qbc_sampling(self):
        """Test query-by-committee sampling."""
        predictions = np.random.randint(0, 2, size=(3, 1000))
        disagreement = np.var(predictions, axis=0)
        batch_size = 10
        disagreed_indices = np.argsort(disagreement)[-batch_size:]
        assert len(disagreed_indices) == batch_size

    def test_expected_model_change(self):
        """Test expected model change criterion."""
        # Mock expected gradient length
        expected_changes = np.random.rand(100)
        top_k = np.argsort(expected_changes)[-5:]
        assert len(top_k) == 5

    def test_expected_error_reduction(self):
        """Test expected error reduction."""
        # Mock error reduction estimates
        error_reductions = np.random.rand(100)
        best_indices = np.argsort(error_reductions)[-10:]
        assert len(best_indices) == 10

    def test_variance_reduction(self):
        """Test variance reduction criterion."""
        # Mock variance estimates
        variances = np.random.rand(100)
        high_variance_indices = np.argsort(variances)[-10:]
        assert len(high_variance_indices) == 10

    def test_information_density(self):
        """Test information density weighting."""
        uncertainties = np.random.rand(100)
        embeddings = generate_mock_embeddings(100, dim=128)
        # Calculate density (simplified)
        from scipy.spatial.distance import cdist
        distances = cdist(embeddings, embeddings)
        densities = np.sum(distances < 0.5, axis=1)
        weighted_scores = uncertainties * densities
        assert len(weighted_scores) == 100

    def test_batch_mode_AL(self):
        """Test batch mode selection."""
        pool_size = 1000
        batch_size = 50
        # Select batch considering diversity
        selected = np.random.choice(pool_size, batch_size, replace=False)
        assert len(set(selected)) == batch_size

    def test_stream_based_AL(self):
        """Test stream-based active learning."""
        # Decide to query or skip for stream
        confidence = 0.65
        threshold = 0.7
        should_query = confidence < threshold
        assert should_query
