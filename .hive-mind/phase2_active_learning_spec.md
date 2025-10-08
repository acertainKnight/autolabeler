# Phase 2: Active Learning Loop - Technical Specification

**Document Version:** 1.0
**Date:** October 7, 2025
**Author:** ANALYST Agent (Hive Mind Swarm)
**Status:** Implementation Ready
**Dependencies:** Phase 1 Complete (Confidence Calibration, Quality Metrics)

---

## Executive Summary

This specification details the implementation of an **Active Learning (AL) loop** for AutoLabeler, targeting **40-70% cost reduction** through intelligent sample selection. By identifying and labeling only the most informative examples, Active Learning dramatically reduces annotation costs while maintaining or improving model performance.

### Key Goals
- Reduce labeling costs by 40-70% compared to random sampling
- Implement uncertainty, diversity, and committee-based sampling strategies
- Integrate seamlessly with existing LabelingService and EnsembleService
- Provide clear stopping criteria to prevent over-labeling
- Enable human-in-the-loop workflows for critical decisions

---

## 1. Background and Research

### 1.1 Active Learning Fundamentals

**Active Learning** is a machine learning paradigm where the model actively selects the most informative examples for labeling, rather than learning from a fixed dataset. The key insight is that **not all examples are equally valuable** for improving model performance.

**Core Concept:**
```
Instead of:  Random Sample → Label All → Train → Evaluate
Use:         Smart Selection → Label Few → Train → Repeat
```

### 1.2 State-of-the-Art Research (2024-2025)

From recent research:

1. **Uncertainty Sampling** (Settles, 2009; Updated 2024)
   - Query instances where the model is least confident
   - Proven to reduce labeling by 50-80% in text classification
   - Three main variants: least confident, margin sampling, entropy

2. **Diversity Sampling** (Sener & Savarese, 2018; TCM 2023)
   - Ensure selected samples are diverse to avoid redundancy
   - Core-set selection minimizes maximum distance to unlabeled data
   - TCM (Transductive Committee Machines) combines uncertainty + diversity

3. **Committee Disagreement** (Seung et al., 1992; Modern ensembles 2024)
   - Use ensemble of models to identify controversial examples
   - High disagreement = high information gain
   - Synergizes perfectly with AutoLabeler's existing ensemble capability

4. **Cold Start Strategies** (NAACL 2024)
   - Hybrid approaches for initial rounds
   - Combine diversity (exploration) with uncertainty (exploitation)
   - Critical for success with zero initial labels

### 1.3 Expected Cost Reductions

Based on research literature:

| Task Type | Random Sampling | Active Learning | Cost Reduction |
|-----------|----------------|-----------------|----------------|
| Binary Classification | 10,000 labels | 2,000-3,000 labels | 70-80% |
| Multi-class (5-10 classes) | 10,000 labels | 4,000-6,000 labels | 40-60% |
| Fine-grained (20+ classes) | 20,000 labels | 10,000-14,000 labels | 30-50% |
| **Average** | - | - | **40-70%** |

---

## 2. Architecture Design

### 2.1 Component Structure

```
src/autolabeler/core/active_learning/
├── __init__.py
├── sampler.py                    # Core ActiveLearningSampler class
├── strategies.py                 # Sampling strategy implementations
├── stopping_criteria.py          # When to stop active learning
└── human_review.py              # Human-in-the-loop integration
```

### 2.2 Integration with Existing Services

```
┌─────────────────────────────────────────────────────────────┐
│                    Active Learning Loop                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  ActiveLearningSampler (NEW)                                │
│  - select_samples()        ← Uses confidence from Phase 1    │
│  - should_stop()           ← Uses quality metrics            │
│  - update_model()                                            │
└─────────────────────────────────────────────────────────────┘
                    │                 │                 │
                    ▼                 ▼                 ▼
        ┌───────────────┐  ┌───────────────┐  ┌──────────────┐
        │ Uncertainty   │  │  Diversity    │  │ Committee    │
        │ Strategy      │  │  Strategy     │  │ Strategy     │
        └───────────────┘  └───────────────┘  └──────────────┘
                    │                 │                 │
                    └─────────────────┴─────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │  Existing Services              │
                    │  - LabelingService             │
                    │  - EnsembleService             │
                    │  - ConfidenceCalibrator        │
                    │  - QualityMonitor              │
                    └─────────────────────────────────┘
```

### 2.3 Data Flow

```
1. Initialize:
   - Start with small seed labeled dataset (50-100 examples)
   - Train initial model on seed data
   - Establish baseline performance

2. Active Learning Loop:
   a. Score unlabeled pool using current model
   b. Select most informative batch (strategy-dependent)
   c. Label selected samples (LLM or human)
   d. Add to training set
   e. Retrain model (optional: can defer to save cost)
   f. Evaluate stopping criteria
   g. Repeat or terminate

3. Termination:
   - Performance plateau detected
   - Budget exhausted
   - Target accuracy reached
   - Manual user stop
```

---

## 3. Sampling Strategies

### 3.1 Uncertainty Sampling

**Goal:** Select examples where the model is least confident.

#### 3.1.1 Least Confident Sampling
```python
score = 1 - max(P(y | x))
```
Select samples with lowest maximum probability.

**Use Case:** Binary classification, high-confidence predictions

#### 3.1.2 Margin Sampling
```python
score = P(y1 | x) - P(y2 | x)  # Difference between top 2 predictions
```
Select samples with smallest margin between top two classes.

**Use Case:** Multi-class with similar classes

#### 3.1.3 Entropy Sampling
```python
score = -Σ P(y | x) * log P(y | x)
```
Select samples with highest prediction entropy.

**Use Case:** Multi-class with many balanced classes

### 3.2 Diversity Sampling

**Goal:** Ensure selected samples cover diverse regions of the feature space.

#### 3.2.1 Core-Set Selection
```python
def core_set_selection(X_labeled, X_unlabeled, n_samples):
    """
    Select samples that minimize maximum distance to labeled data.

    Algorithm (Greedy):
    1. Compute embeddings for all samples
    2. For each unlabeled sample, find distance to nearest labeled sample
    3. Select sample with largest minimum distance
    4. Add to labeled set, repeat
    """
    embeddings_labeled = embed(X_labeled)
    embeddings_unlabeled = embed(X_unlabeled)

    selected = []
    for _ in range(n_samples):
        distances = compute_min_distances(
            embeddings_unlabeled,
            embeddings_labeled + selected
        )
        idx = argmax(distances)
        selected.append(embeddings_unlabeled[idx])

    return selected
```

**Use Case:** Cold start, high diversity needed

#### 3.2.2 Clustering-Based Diversity
```python
def cluster_diversity(X_unlabeled, n_samples):
    """
    Cluster unlabeled data and select one representative per cluster.
    """
    kmeans = KMeans(n_clusters=n_samples)
    clusters = kmeans.fit_predict(X_unlabeled)

    # Select sample nearest to each cluster center
    selected = []
    for i in range(n_samples):
        cluster_samples = X_unlabeled[clusters == i]
        center = kmeans.cluster_centers_[i]
        nearest = argmin(distance(cluster_samples, center))
        selected.append(cluster_samples[nearest])

    return selected
```

**Use Case:** Balanced exploration across data distribution

### 3.3 Committee Disagreement (Query-by-Committee)

**Goal:** Select examples where ensemble models disagree most.

```python
def committee_disagreement(predictions_list, method="vote_entropy"):
    """
    Calculate disagreement score from ensemble predictions.

    Args:
        predictions_list: List of predictions from each model
        method: "vote_entropy" or "kl_divergence"

    Returns:
        Disagreement scores (higher = more disagreement)
    """
    if method == "vote_entropy":
        # Count votes for each class
        vote_counts = Counter(predictions_list)
        total_votes = len(predictions_list)

        # Calculate entropy of vote distribution
        entropy = -sum(
            (count / total_votes) * log(count / total_votes)
            for count in vote_counts.values()
        )
        return entropy

    elif method == "kl_divergence":
        # Average KL divergence between each model and consensus
        consensus = average_predictions(predictions_list)
        divergences = [
            kl_div(pred, consensus)
            for pred in predictions_list
        ]
        return mean(divergences)
```

**Use Case:** Already have ensemble (AutoLabeler does!), maximize information gain

### 3.4 Hybrid Strategy (Recommended)

Combine uncertainty and diversity for best results:

```python
def hybrid_selection(
    X_unlabeled,
    predictions,
    n_samples,
    alpha=0.7  # Weight for uncertainty (1-alpha for diversity)
):
    """
    Hybrid strategy combining uncertainty and diversity.

    Algorithm:
    1. Score all samples by uncertainty
    2. Select top K*3 most uncertain (over-sample)
    3. From those, select n_samples using diversity
    4. Result: Uncertain AND diverse samples
    """
    # Step 1: Get top uncertain samples
    uncertainty_scores = calculate_uncertainty(predictions)
    top_uncertain_idx = argsort(uncertainty_scores)[-n_samples*3:]

    # Step 2: Apply diversity among uncertain samples
    X_candidates = X_unlabeled[top_uncertain_idx]
    diverse_idx = core_set_selection(X_candidates, n_samples)

    # Step 3: Combine scores
    final_scores = (
        alpha * uncertainty_scores[diverse_idx] +
        (1 - alpha) * diversity_scores[diverse_idx]
    )

    return argsort(final_scores)[-n_samples:]
```

**Use Case:** General purpose, best overall performance

---

## 4. Stopping Criteria

Active learning should terminate when additional labeling provides diminishing returns.

### 4.1 Performance Plateau Detection

```python
def detect_plateau(performance_history, patience=3, threshold=0.01):
    """
    Detect if performance has plateaued.

    Args:
        performance_history: List of accuracy/F1 scores over iterations
        patience: Number of iterations without improvement to trigger stop
        threshold: Minimum improvement considered significant

    Returns:
        bool: True if plateau detected
    """
    if len(performance_history) < patience + 1:
        return False

    recent_improvements = [
        performance_history[i] - performance_history[i-1]
        for i in range(-patience, 0)
    ]

    # Stop if all recent improvements below threshold
    return all(imp < threshold for imp in recent_improvements)
```

**Recommended:** `patience=3`, `threshold=0.01` (1% improvement)

### 4.2 Budget Exhaustion

```python
def check_budget(
    current_cost,
    max_budget,
    buffer=0.1  # Reserve 10% for final evaluation
):
    """
    Check if budget is exhausted.

    Returns:
        bool: True if should stop due to budget
    """
    return current_cost >= max_budget * (1 - buffer)
```

### 4.3 Target Performance Reached

```python
def target_reached(current_accuracy, target_accuracy):
    """
    Check if target performance achieved.

    Returns:
        bool: True if target reached
    """
    return current_accuracy >= target_accuracy
```

### 4.4 Uncertainty Threshold

```python
def low_uncertainty(pool_uncertainty, threshold=0.1):
    """
    Stop if remaining pool has low uncertainty.

    Indicates model is confident on all remaining examples.

    Returns:
        bool: True if average pool uncertainty below threshold
    """
    return mean(pool_uncertainty) < threshold
```

### 4.5 Recommended Combined Criteria

```python
def should_stop(state: ALState, config: ALConfig) -> tuple[bool, str]:
    """
    Combined stopping criteria with reasoning.

    Returns:
        (should_stop, reason)
    """
    # Check plateau
    if detect_plateau(
        state.performance_history,
        patience=config.patience,
        threshold=config.improvement_threshold
    ):
        return True, "performance_plateau"

    # Check budget
    if check_budget(state.current_cost, config.max_budget):
        return True, "budget_exhausted"

    # Check target
    if target_reached(state.current_accuracy, config.target_accuracy):
        return True, "target_reached"

    # Check low uncertainty
    if low_uncertainty(state.pool_uncertainty, config.uncertainty_threshold):
        return True, "low_uncertainty"

    # Check iteration limit
    if state.iteration >= config.max_iterations:
        return True, "max_iterations"

    return False, "continue"
```

---

## 5. Integration with Existing Services

### 5.1 LabelingService Integration

```python
class LabelingService:
    # ... existing code ...

    def label_batch_with_active_learning(
        self,
        unlabeled_df: pd.DataFrame,
        al_config: ActiveLearningConfig,
        text_column: str = "text"
    ) -> pd.DataFrame:
        """
        Label a dataset using active learning.

        Args:
            unlabeled_df: Unlabeled dataset
            al_config: Active learning configuration
            text_column: Column containing text to label

        Returns:
            Labeled dataset with AL metadata
        """
        # Initialize active learning sampler
        sampler = ActiveLearningSampler(
            labeling_service=self,
            config=al_config
        )

        # Run active learning loop
        results = sampler.run_active_learning_loop(
            unlabeled_df=unlabeled_df,
            text_column=text_column
        )

        return results
```

### 5.2 EnsembleService Integration (Committee-Based)

```python
class EnsembleService:
    # ... existing code ...

    def calculate_committee_disagreement(
        self,
        predictions: list[EnsembleResult]
    ) -> list[float]:
        """
        Calculate disagreement scores for committee-based active learning.

        Args:
            predictions: List of ensemble predictions

        Returns:
            Disagreement scores for each prediction
        """
        disagreement_scores = []

        for pred in predictions:
            if not pred.individual_predictions:
                disagreement_scores.append(0.0)
                continue

            # Extract labels from individual predictions
            labels = [
                p["label"]
                for p in pred.individual_predictions
            ]

            # Calculate vote entropy
            vote_counts = Counter(labels)
            total = len(labels)
            entropy = -sum(
                (count / total) * np.log(count / total)
                for count in vote_counts.values()
            )

            disagreement_scores.append(entropy)

        return disagreement_scores
```

### 5.3 ConfidenceCalibrator Integration

```python
# Use calibrated confidence scores for uncertainty sampling
calibrated_confidence = self.confidence_calibrator.calibrate(
    raw_confidence=predictions.confidence
)

# Uncertainty score = 1 - calibrated_confidence
uncertainty_scores = 1 - calibrated_confidence
```

---

## 6. Implementation Details

### 6.1 Core ActiveLearningSampler Class

```python
# src/autolabeler/core/active_learning/sampler.py

from dataclasses import dataclass
from typing import Callable, Literal
import numpy as np
import pandas as pd
from loguru import logger

from ..configs import ActiveLearningConfig
from ..labeling import LabelingService
from .strategies import (
    UncertaintySampler,
    DiversitySampler,
    CommitteeSampler,
    HybridSampler
)
from .stopping_criteria import StoppingCriteria


@dataclass
class ALState:
    """Active learning state tracking."""
    iteration: int = 0
    current_cost: float = 0.0
    current_accuracy: float = 0.0
    performance_history: list[float] = None
    labeled_indices: list[int] = None
    pool_uncertainty: list[float] = None

    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []
        if self.labeled_indices is None:
            self.labeled_indices = []
        if self.pool_uncertainty is None:
            self.pool_uncertainty = []


class ActiveLearningSampler:
    """
    Active learning sampler for intelligent sample selection.

    Implements multiple sampling strategies and stopping criteria
    to minimize labeling costs while maximizing model performance.

    Args:
        labeling_service: LabelingService for labeling operations
        config: Active learning configuration

    Example:
        >>> sampler = ActiveLearningSampler(labeling_service, config)
        >>> results = sampler.run_active_learning_loop(unlabeled_df)
    """

    def __init__(
        self,
        labeling_service: LabelingService,
        config: ActiveLearningConfig
    ):
        self.labeling_service = labeling_service
        self.config = config
        self.state = ALState()

        # Initialize strategy
        self.strategy = self._create_strategy(config.strategy)

        # Initialize stopping criteria
        self.stopping_criteria = StoppingCriteria(config)

        logger.info(
            f"ActiveLearningSampler initialized with strategy: {config.strategy}"
        )

    def _create_strategy(self, strategy_name: str):
        """Create sampling strategy instance."""
        strategies = {
            "uncertainty": UncertaintySampler,
            "diversity": DiversitySampler,
            "committee": CommitteeSampler,
            "hybrid": HybridSampler
        }

        if strategy_name not in strategies:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Choose from: {list(strategies.keys())}"
            )

        return strategies[strategy_name](self.config)

    def run_active_learning_loop(
        self,
        unlabeled_df: pd.DataFrame,
        text_column: str = "text",
        seed_labeled_df: pd.DataFrame | None = None,
        validation_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Run the complete active learning loop.

        Args:
            unlabeled_df: Unlabeled data pool
            text_column: Column containing text
            seed_labeled_df: Initial labeled examples (optional)
            validation_df: Validation set for evaluation (optional)

        Returns:
            DataFrame with selected and labeled samples
        """
        logger.info(
            f"Starting active learning loop with {len(unlabeled_df)} unlabeled examples"
        )

        # Initialize with seed data if provided
        if seed_labeled_df is not None:
            labeled_df = seed_labeled_df.copy()
            self.state.labeled_indices = seed_labeled_df.index.tolist()
        else:
            # Bootstrap with random seed
            seed_size = self.config.initial_seed_size
            seed_indices = np.random.choice(
                unlabeled_df.index,
                size=min(seed_size, len(unlabeled_df)),
                replace=False
            )
            labeled_df = unlabeled_df.loc[seed_indices].copy()
            unlabeled_df = unlabeled_df.drop(seed_indices)

        # Active learning iterations
        while True:
            self.state.iteration += 1
            logger.info(f"\n=== Active Learning Iteration {self.state.iteration} ===")

            # Check stopping criteria
            should_stop, reason = self.stopping_criteria.check(self.state)
            if should_stop:
                logger.info(f"Stopping active learning: {reason}")
                break

            # Select next batch
            selected_indices = self.select_batch(
                unlabeled_df=unlabeled_df,
                labeled_df=labeled_df,
                text_column=text_column
            )

            if len(selected_indices) == 0:
                logger.warning("No samples selected, stopping")
                break

            # Label selected batch
            selected_df = unlabeled_df.loc[selected_indices]
            labeled_batch = self._label_batch(selected_df, text_column)

            # Update datasets
            labeled_df = pd.concat([labeled_df, labeled_batch], ignore_index=True)
            unlabeled_df = unlabeled_df.drop(selected_indices)
            self.state.labeled_indices.extend(selected_indices.tolist())

            # Update state
            self.state.current_cost += self._calculate_batch_cost(labeled_batch)

            # Evaluate if validation set provided
            if validation_df is not None:
                accuracy = self._evaluate(labeled_df, validation_df, text_column)
                self.state.current_accuracy = accuracy
                self.state.performance_history.append(accuracy)

                logger.info(
                    f"Iteration {self.state.iteration}: "
                    f"Accuracy={accuracy:.3f}, "
                    f"Labeled={len(labeled_df)}, "
                    f"Cost=${self.state.current_cost:.2f}"
                )

            # Update pool uncertainty
            if len(unlabeled_df) > 0:
                pool_predictions = self._get_predictions(unlabeled_df, text_column)
                self.state.pool_uncertainty = [
                    1 - p.confidence for p in pool_predictions
                ]

        logger.info(
            f"\nActive learning complete. "
            f"Labeled {len(labeled_df)} examples "
            f"at ${self.state.current_cost:.2f} cost."
        )

        return labeled_df

    def select_batch(
        self,
        unlabeled_df: pd.DataFrame,
        labeled_df: pd.DataFrame,
        text_column: str
    ) -> pd.Index:
        """
        Select next batch of samples to label.

        Args:
            unlabeled_df: Unlabeled data pool
            labeled_df: Currently labeled data
            text_column: Column containing text

        Returns:
            Indices of selected samples
        """
        # Get predictions for unlabeled pool
        predictions = self._get_predictions(unlabeled_df, text_column)

        # Apply strategy to select samples
        selected_indices = self.strategy.select(
            unlabeled_df=unlabeled_df,
            labeled_df=labeled_df,
            predictions=predictions,
            batch_size=self.config.batch_size
        )

        return selected_indices

    def _get_predictions(
        self,
        df: pd.DataFrame,
        text_column: str
    ) -> list:
        """Get predictions for a dataset."""
        predictions = []
        for text in df[text_column]:
            result = self.labeling_service.label_text(text)
            predictions.append(result)
        return predictions

    def _label_batch(
        self,
        batch_df: pd.DataFrame,
        text_column: str
    ) -> pd.DataFrame:
        """Label a batch of samples."""
        return self.labeling_service.label_batch(
            df=batch_df,
            text_column=text_column
        )

    def _calculate_batch_cost(self, batch_df: pd.DataFrame) -> float:
        """Calculate cost of labeling a batch."""
        # Estimate based on token count and model pricing
        # This is a placeholder - actual implementation would use
        # real token counts from API responses
        avg_tokens = 500  # Estimate
        cost_per_1k_tokens = 0.001  # GPT-3.5-turbo pricing
        return len(batch_df) * (avg_tokens / 1000) * cost_per_1k_tokens

    def _evaluate(
        self,
        labeled_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        text_column: str
    ) -> float:
        """Evaluate current model performance."""
        # This is a simplified placeholder
        # Actual implementation would train a model and evaluate
        # For now, return mock accuracy
        return min(0.5 + 0.05 * self.state.iteration, 0.95)
```

### 6.2 Strategy Implementations

```python
# src/autolabeler/core/active_learning/strategies.py

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from ..configs import ActiveLearningConfig


class SamplingStrategy(ABC):
    """Base class for sampling strategies."""

    def __init__(self, config: ActiveLearningConfig):
        self.config = config

    @abstractmethod
    def select(
        self,
        unlabeled_df: pd.DataFrame,
        labeled_df: pd.DataFrame,
        predictions: list,
        batch_size: int
    ) -> pd.Index:
        """Select samples from unlabeled pool."""
        pass


class UncertaintySampler(SamplingStrategy):
    """Uncertainty-based sampling strategy."""

    def select(
        self,
        unlabeled_df: pd.DataFrame,
        labeled_df: pd.DataFrame,
        predictions: list,
        batch_size: int
    ) -> pd.Index:
        """Select samples with highest uncertainty."""
        # Calculate uncertainty scores
        uncertainty_scores = np.array([
            self._calculate_uncertainty(pred)
            for pred in predictions
        ])

        # Select top-k most uncertain
        top_indices = np.argsort(uncertainty_scores)[-batch_size:]
        return unlabeled_df.index[top_indices]

    def _calculate_uncertainty(self, prediction) -> float:
        """Calculate uncertainty score for a prediction."""
        method = self.config.uncertainty_method

        if method == "least_confident":
            return 1 - prediction.confidence

        elif method == "margin":
            # Requires access to top-2 probabilities
            # Placeholder for now
            return 1 - prediction.confidence

        elif method == "entropy":
            # Requires full probability distribution
            # Placeholder for now
            return 1 - prediction.confidence

        else:
            return 1 - prediction.confidence


class DiversitySampler(SamplingStrategy):
    """Diversity-based sampling strategy."""

    def __init__(self, config: ActiveLearningConfig):
        super().__init__(config)
        self.embedder = SentenceTransformer(config.embedding_model)

    def select(
        self,
        unlabeled_df: pd.DataFrame,
        labeled_df: pd.DataFrame,
        predictions: list,
        batch_size: int
    ) -> pd.Index:
        """Select diverse samples using clustering."""
        # Get embeddings
        texts = unlabeled_df[self.config.text_column].tolist()
        embeddings = self.embedder.encode(texts)

        # Cluster and select representatives
        kmeans = KMeans(n_clusters=batch_size, random_state=42)
        kmeans.fit(embeddings)

        # Select sample nearest to each cluster center
        selected_indices = []
        for i in range(batch_size):
            cluster_mask = kmeans.labels_ == i
            cluster_embeddings = embeddings[cluster_mask]
            center = kmeans.cluster_centers_[i]

            # Find nearest to center
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            nearest_idx = np.argmin(distances)

            # Map back to original index
            cluster_indices = np.where(cluster_mask)[0]
            selected_indices.append(cluster_indices[nearest_idx])

        return unlabeled_df.index[selected_indices]


class CommitteeSampler(SamplingStrategy):
    """Committee disagreement sampling strategy."""

    def select(
        self,
        unlabeled_df: pd.DataFrame,
        labeled_df: pd.DataFrame,
        predictions: list,
        batch_size: int
    ) -> pd.Index:
        """Select samples with highest committee disagreement."""
        # Calculate disagreement scores
        disagreement_scores = np.array([
            self._calculate_disagreement(pred)
            for pred in predictions
        ])

        # Select top-k with highest disagreement
        top_indices = np.argsort(disagreement_scores)[-batch_size:]
        return unlabeled_df.index[top_indices]

    def _calculate_disagreement(self, prediction) -> float:
        """Calculate committee disagreement score."""
        if not hasattr(prediction, 'individual_predictions'):
            return 0.0

        # Extract labels from ensemble members
        labels = [p.get('label') for p in prediction.individual_predictions]

        # Calculate vote entropy
        from collections import Counter
        vote_counts = Counter(labels)
        total = len(labels)

        entropy = -sum(
            (count / total) * np.log(count / total)
            for count in vote_counts.values()
        )

        return entropy


class HybridSampler(SamplingStrategy):
    """Hybrid sampling combining uncertainty and diversity."""

    def __init__(self, config: ActiveLearningConfig):
        super().__init__(config)
        self.uncertainty_sampler = UncertaintySampler(config)
        self.diversity_sampler = DiversitySampler(config)

    def select(
        self,
        unlabeled_df: pd.DataFrame,
        labeled_df: pd.DataFrame,
        predictions: list,
        batch_size: int
    ) -> pd.Index:
        """Select samples combining uncertainty and diversity."""
        alpha = self.config.hybrid_alpha  # Weight for uncertainty

        # Step 1: Select top-k*3 most uncertain samples
        uncertainty_scores = np.array([
            self.uncertainty_sampler._calculate_uncertainty(pred)
            for pred in predictions
        ])
        top_uncertain_idx = np.argsort(uncertainty_scores)[-(batch_size*3):]

        # Step 2: Among uncertain samples, select diverse ones
        uncertain_df = unlabeled_df.iloc[top_uncertain_idx]
        diverse_indices = self.diversity_sampler.select(
            unlabeled_df=uncertain_df,
            labeled_df=labeled_df,
            predictions=[predictions[i] for i in top_uncertain_idx],
            batch_size=batch_size
        )

        return diverse_indices
```

---

## 7. Configuration

### 7.1 ActiveLearningConfig Class

```python
# Add to src/autolabeler/core/configs.py

class ActiveLearningConfig(BaseModel):
    """Configuration for active learning."""

    # Strategy selection
    strategy: Literal["uncertainty", "diversity", "committee", "hybrid"] = Field(
        "hybrid",
        description="Active learning sampling strategy"
    )

    # Uncertainty parameters
    uncertainty_method: Literal["least_confident", "margin", "entropy"] = Field(
        "least_confident",
        description="Method for calculating uncertainty"
    )

    # Diversity parameters
    embedding_model: str = Field(
        "all-MiniLM-L6-v2",
        description="Embedding model for diversity sampling"
    )

    # Hybrid parameters
    hybrid_alpha: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Weight for uncertainty in hybrid strategy (1-alpha for diversity)"
    )

    # Sampling parameters
    batch_size: int = Field(
        50,
        gt=0,
        description="Number of samples to select per iteration"
    )
    initial_seed_size: int = Field(
        100,
        gt=0,
        description="Initial seed dataset size"
    )
    text_column: str = Field(
        "text",
        description="Column containing text to label"
    )

    # Stopping criteria
    max_iterations: int = Field(
        20,
        gt=0,
        description="Maximum active learning iterations"
    )
    max_budget: float = Field(
        100.0,
        gt=0,
        description="Maximum budget in USD"
    )
    target_accuracy: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description="Target accuracy to reach"
    )
    patience: int = Field(
        3,
        gt=0,
        description="Iterations without improvement before stopping"
    )
    improvement_threshold: float = Field(
        0.01,
        gt=0,
        description="Minimum improvement considered significant"
    )
    uncertainty_threshold: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Stop if pool uncertainty below this threshold"
    )

    # Human-in-the-loop
    enable_human_review: bool = Field(
        False,
        description="Enable human review for selected samples"
    )
    human_review_confidence_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Send samples below this confidence to human review"
    )
```

---

## 8. Usage Examples

### 8.1 Basic Active Learning

```python
from autolabeler import AutoLabeler
from autolabeler.core.configs import ActiveLearningConfig

# Initialize AutoLabeler
labeler = AutoLabeler(
    dataset_name="sentiment_analysis",
    task_description="Classify sentiment as positive or negative"
)

# Configure active learning
al_config = ActiveLearningConfig(
    strategy="hybrid",
    batch_size=50,
    max_budget=50.0,  # $50 budget
    target_accuracy=0.90
)

# Run active learning
results = labeler.label_with_active_learning(
    unlabeled_file="data/unlabeled.csv",
    output_file="results/al_labeled.csv",
    al_config=al_config
)

print(f"Labeled {len(results)} examples")
print(f"Total cost: ${results.metadata['total_cost']:.2f}")
print(f"Final accuracy: {results.metadata['final_accuracy']:.3f}")
```

### 8.2 Active Learning with Ensemble

```python
# Use committee disagreement strategy with existing ensemble
al_config = ActiveLearningConfig(
    strategy="committee",
    batch_size=30,
    max_iterations=10
)

# Requires ensemble service
labeler.enable_ensemble([
    {"model_name": "gpt-3.5-turbo", "temperature": 0.3},
    {"model_name": "gpt-4o-mini", "temperature": 0.5},
    {"model_name": "claude-3-haiku", "temperature": 0.4}
])

results = labeler.label_with_active_learning(
    unlabeled_file="data/unlabeled.csv",
    al_config=al_config
)
```

### 8.3 Cost Comparison Example

```python
import pandas as pd

# Baseline: Random sampling
random_results = labeler.label_batch(
    input_file="data/unlabeled_10k.csv",
    output_file="results/random_labeled.csv"
)
random_cost = random_results.metadata['total_cost']
print(f"Random sampling cost: ${random_cost:.2f}")

# Active learning
al_config = ActiveLearningConfig(
    strategy="hybrid",
    batch_size=50,
    target_accuracy=0.90  # Same target as random
)

al_results = labeler.label_with_active_learning(
    unlabeled_file="data/unlabeled_10k.csv",
    output_file="results/al_labeled.csv",
    al_config=al_config
)
al_cost = al_results.metadata['total_cost']
print(f"Active learning cost: ${al_cost:.2f}")

# Calculate savings
savings = (random_cost - al_cost) / random_cost * 100
print(f"Cost reduction: {savings:.1f}%")
```

---

## 9. Success Metrics

### 9.1 Primary Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Cost Reduction** | 40-70% | vs. random sampling baseline |
| **Label Efficiency** | 2-3× | Samples needed to reach target accuracy |
| **Convergence Speed** | <10 iterations | Iterations to plateau |
| **Time Savings** | 50-70% | vs. exhaustive labeling |

### 9.2 Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Final Accuracy** | ≥90% | On held-out test set |
| **Sample Diversity** | >0.8 | Cosine diversity score |
| **Uncertainty Reduction** | >50% | Pool uncertainty decrease |
| **Stop Criteria Accuracy** | >80% | Correct stop decisions |

---

## 10. Integration Testing

### 10.1 Unit Tests

```python
# tests/test_active_learning.py

def test_uncertainty_sampler():
    """Test uncertainty sampling strategy."""
    sampler = UncertaintySampler(config)
    # Test with mock predictions
    assert len(selected) == batch_size

def test_diversity_sampler():
    """Test diversity sampling strategy."""
    sampler = DiversitySampler(config)
    # Test cluster-based selection
    assert diversity_score(selected) > threshold

def test_committee_sampler():
    """Test committee disagreement strategy."""
    sampler = CommitteeSampler(config)
    # Test disagreement calculation
    assert all(disagreement > 0 for disagreement in scores)

def test_hybrid_sampler():
    """Test hybrid strategy."""
    sampler = HybridSampler(config)
    # Test combination of uncertainty + diversity
    assert len(selected) == batch_size
```

### 10.2 Integration Tests

```python
def test_active_learning_loop():
    """Test complete active learning loop."""
    sampler = ActiveLearningSampler(labeling_service, config)
    results = sampler.run_active_learning_loop(unlabeled_df)

    assert len(results) < len(unlabeled_df)  # Should label subset
    assert results.metadata['final_accuracy'] >= config.target_accuracy

def test_stopping_criteria():
    """Test various stopping criteria."""
    # Test plateau detection
    assert detect_plateau([0.8, 0.81, 0.809, 0.811]) == True

    # Test budget exhaustion
    assert check_budget(90, 100) == True
```

---

## 11. Performance Benchmarks

### 11.1 Expected Performance

| Dataset Size | Random Labels | AL Labels | Reduction | Time |
|--------------|---------------|-----------|-----------|------|
| 1,000 | 1,000 | 300-400 | 60-70% | 5-10 min |
| 10,000 | 10,000 | 2,000-4,000 | 60-80% | 30-60 min |
| 100,000 | 100,000 | 10,000-20,000 | 80-90% | 4-8 hours |

### 11.2 Cost Benchmarks

| Task | Random Cost | AL Cost | Savings |
|------|-------------|---------|---------|
| Sentiment (binary) | $500 | $100-150 | $350-400 (70-80%) |
| Topic Classification (10 classes) | $1,000 | $400-600 | $400-600 (40-60%) |
| Entity Recognition | $2,000 | $800-1,200 | $800-1,200 (40-60%) |

---

## 12. Next Steps

1. ✅ Specification complete
2. ⏭️ Implement `sampler.py` and `strategies.py`
3. ⏭️ Add `ActiveLearningConfig` to `configs.py`
4. ⏭️ Integrate with `LabelingService` and `EnsembleService`
5. ⏭️ Create comprehensive examples
6. ⏭️ Write unit and integration tests
7. ⏭️ Benchmark on real datasets
8. ⏭️ Document cost savings

---

**Specification Status:** ✅ COMPLETE - Ready for Implementation
**Next Document:** phase2_weak_supervision_spec.md
**Dependencies:** Phase 1 (Confidence Calibration, Quality Metrics)

---

*Generated by ANALYST Agent (Hive Mind Swarm)*
*Date: October 7, 2025*
*Working Directory: /home/nick/python/autolabeler*
