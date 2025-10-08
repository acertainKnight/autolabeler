# Phase 2: Weak Supervision Framework - Technical Specification

**Document Version:** 1.0
**Date:** October 7, 2025
**Author:** ANALYST Agent (Hive Mind Swarm)
**Status:** Implementation Ready
**Dependencies:** Phase 1 Complete, Active Learning Spec Complete

---

## Executive Summary

This specification details the implementation of a **Weak Supervision framework** for AutoLabeler, enabling **programmatic labeling at scale** with minimal manual annotation. By leveraging labeling functions (LFs), knowledge bases, and statistical aggregation, Weak Supervision can label datasets 10-100× faster than manual annotation while maintaining quality.

### Key Goals
- Generate training data 10-100× faster than manual annotation
- Implement Snorkel/FlyingSquid integration for label aggregation
- Support multiple labeling function types (keywords, regex, models, LLMs)
- Enable automatic LF generation using LLMs
- Provide quality metrics for labeling functions
- Achieve 70-85% accuracy on aggregated labels

---

## 1. Background and Research

### 1.1 Weak Supervision Fundamentals

**Weak Supervision** replaces slow, expensive manual labeling with **programmatic labeling** using imperfect heuristics called **labeling functions (LFs)**. The key insight is that **many noisy labels are better than few perfect labels**.

**Core Concept:**
```
Traditional:   Manual Annotation → 100% Accurate, Very Slow
Weak Supervision:  Programmatic LFs → 70% Accurate, Very Fast
                   + Statistical Aggregation → 85% Accurate, Fast
```

### 1.2 State-of-the-Art Research (2024-2025)

From recent research:

1. **Snorkel (Stanford, 2017-2024)**
   - Foundational weak supervision framework
   - Label model aggregates noisy LF outputs
   - Proven to match hand-labeling quality with 10-100× speedup
   - Used in production at Google, Apple, Intel

2. **FlyingSquid (2020-2024)**
   - Fast label aggregation (170× faster than Snorkel)
   - Based on triplet method for learning LF accuracies
   - Better handles correlations between LFs
   - Recommended for production use

3. **LLM-Generated Labeling Functions (NAACL 2024)**
   - Use LLMs to automatically generate LFs from examples
   - Reduces LF development time by 90%
   - AutoLabeler can leverage existing RAG examples

4. **Hybrid Approaches (2024)**
   - Combine weak supervision with active learning
   - Use WS for bulk labeling, AL for hard cases
   - Achieves best cost/quality tradeoff

### 1.3 Expected Performance

Based on research literature:

| Metric | Manual | Weak Supervision | Improvement |
|--------|--------|------------------|-------------|
| **Speed** | 100 labels/hour | 10,000+ labels/hour | 100× faster |
| **Cost** | $0.50/label | $0.005-0.01/label | 50-100× cheaper |
| **Accuracy** | 95-99% | 75-85% (aggregated) | -10-20% |
| **Coverage** | 100% | 60-90% | Some abstentions |

**Key Insight:** Trading 10-20% accuracy for 100× speed is often worthwhile, especially when combined with active learning for hard cases.

---

## 2. Architecture Design

### 2.1 Component Structure

```
src/autolabeler/core/weak_supervision/
├── __init__.py
├── labeling_functions.py       # LF management and definitions
├── snorkel_integrator.py      # Snorkel/FlyingSquid integration
├── lf_generator.py            # LLM-based LF generation
└── quality_analyzer.py        # LF quality metrics
```

### 2.2 Integration with Existing Services

```
┌─────────────────────────────────────────────────────────────┐
│                Weak Supervision Framework                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  WeakSupervisionService (NEW)                               │
│  - register_lf()           ← Manual or LLM-generated         │
│  - apply_lfs()             ← Apply to unlabeled data         │
│  - aggregate_labels()      ← FlyingSquid/Snorkel            │
│  - analyze_lf_quality()    ← Coverage, accuracy, conflicts  │
└─────────────────────────────────────────────────────────────┘
                    │                 │                 │
                    ▼                 ▼                 ▼
        ┌───────────────┐  ┌───────────────┐  ┌──────────────┐
        │ Labeling      │  │  Label        │  │ LF           │
        │ Functions     │  │  Aggregation  │  │ Generator    │
        └───────────────┘  └───────────────┘  └──────────────┘
                    │                 │                 │
                    └─────────────────┴─────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │  Existing Services              │
                    │  - LabelingService (for LLM LFs)│
                    │  - KnowledgeStore (for patterns)│
                    │  - EnsembleService (for models) │
                    └─────────────────────────────────┘
```

### 2.3 Data Flow

```
1. LF Development:
   a. Manually define LFs (keywords, regex, rules)
   b. OR: Use LLM to generate LFs from examples
   c. Register LFs in framework

2. LF Application:
   a. Apply all LFs to unlabeled dataset
   b. Each LF votes: POSITIVE, NEGATIVE, or ABSTAIN
   c. Result: Label matrix (n_samples × n_lfs)

3. Label Aggregation:
   a. Feed label matrix to FlyingSquid/Snorkel
   b. Learn LF accuracies and correlations
   c. Aggregate votes into final labels with confidence
   d. Output: Weakly labeled dataset

4. Quality Analysis:
   a. Calculate LF coverage (% non-abstain)
   b. Calculate LF accuracy (if dev set available)
   c. Identify conflicting LFs
   d. Recommend LF improvements

5. Refinement:
   a. Remove low-quality LFs
   b. Add new LFs for low-coverage regions
   c. Combine with active learning for hard cases
```

---

## 3. Labeling Function Types

### 3.1 Keyword-Based LFs

Simple pattern matching for specific words/phrases.

```python
@labeling_function()
def lf_positive_keywords(x):
    """Label as POSITIVE if contains positive keywords."""
    positive_words = {"excellent", "great", "amazing", "love", "best"}
    text_lower = x.text.lower()

    if any(word in text_lower for word in positive_words):
        return POSITIVE
    return ABSTAIN
```

**Pros:** Fast, interpretable, no dependencies
**Cons:** Brittle, misses context, low coverage
**Use Case:** High-precision rules for obvious cases

### 3.2 Regex-Based LFs

Pattern matching with regular expressions.

```python
@labeling_function()
def lf_email_pattern(x):
    """Label as CONTACT_INFO if contains email."""
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

    if re.search(email_pattern, x.text):
        return POSITIVE
    return ABSTAIN
```

**Pros:** More flexible than keywords, still fast
**Cons:** Hard to maintain, can be overly complex
**Use Case:** Structured patterns (emails, phone numbers, dates)

### 3.3 Heuristic Rules

Domain-specific logic.

```python
@labeling_function()
def lf_short_text(x):
    """Label short texts as LOW_QUALITY."""
    if len(x.text.split()) < 5:
        return LOW_QUALITY
    return ABSTAIN
```

**Pros:** Encode domain knowledge, interpretable
**Cons:** Requires expertise to develop
**Use Case:** Domain-specific constraints

### 3.4 External Model LFs

Use pre-trained models as labeling functions.

```python
@labeling_function(resources={"sentiment_model": sentiment_analyzer})
def lf_sentiment_model(x, sentiment_model):
    """Use sentiment model as labeling function."""
    score = sentiment_model(x.text)

    if score["label"] == "POSITIVE" and score["score"] > 0.7:
        return POSITIVE
    elif score["label"] == "NEGATIVE" and score["score"] > 0.7:
        return NEGATIVE
    return ABSTAIN
```

**Pros:** Leverage existing models, high accuracy
**Cons:** Slower, requires model download
**Use Case:** When pre-trained models are available

### 3.5 LLM-Based LFs

Use LLMs as labeling functions (zero-shot or few-shot).

```python
@labeling_function(resources={"llm": llm_client})
def lf_llm_zero_shot(x, llm):
    """Use LLM as zero-shot labeling function."""
    prompt = f"Is this text positive or negative? Text: {x.text}\nAnswer:"
    response = llm.complete(prompt, max_tokens=10)

    if "positive" in response.lower():
        return POSITIVE
    elif "negative" in response.lower():
        return NEGATIVE
    return ABSTAIN
```

**Pros:** Highest flexibility, can handle complex cases
**Cons:** Expensive, slower (use for hard cases only)
**Use Case:** Complex reasoning, rare classes

### 3.6 Knowledge Base LFs

Use existing labeled examples (RAG-based).

```python
@labeling_function(resources={"knowledge_store": kb})
def lf_similar_examples(x, knowledge_store):
    """Label based on similar examples in knowledge base."""
    similar = knowledge_store.find_similar_examples(x.text, k=3)

    if not similar:
        return ABSTAIN

    # Vote based on similar examples
    labels = [ex["label"] for ex in similar]
    most_common = Counter(labels).most_common(1)[0]

    if most_common[1] >= 2:  # At least 2/3 agree
        return label_to_int(most_common[0])

    return ABSTAIN
```

**Pros:** Leverage existing examples, adaptive
**Cons:** Requires labeled examples, slower
**Use Case:** When knowledge base is available (AutoLabeler has this!)

---

## 4. Label Aggregation

### 4.1 Majority Vote (Baseline)

Simple voting: most common label wins.

```python
def majority_vote(label_matrix):
    """
    Aggregate labels using majority vote.

    Args:
        label_matrix: (n_samples, n_lfs) with values in {-1, 0, 1}
                      -1 = NEGATIVE, 0 = ABSTAIN, 1 = POSITIVE

    Returns:
        aggregated_labels: (n_samples,) with majority label
    """
    aggregated = []
    for row in label_matrix:
        # Filter out abstentions (0)
        votes = [v for v in row if v != 0]

        if not votes:
            aggregated.append(0)  # All abstain
        else:
            # Most common vote
            aggregated.append(Counter(votes).most_common(1)[0][0])

    return np.array(aggregated)
```

**Pros:** Simple, interpretable
**Cons:** Treats all LFs equally, ignores accuracy
**Performance:** 60-70% accuracy typical

### 4.2 Snorkel Label Model

Generative model learning LF accuracies.

```python
from snorkel.labeling import LabelModel

def snorkel_aggregation(label_matrix, n_epochs=500):
    """
    Aggregate labels using Snorkel's label model.

    Args:
        label_matrix: (n_samples, n_lfs) label matrix
        n_epochs: Training epochs for label model

    Returns:
        aggregated_labels: (n_samples,) with learned labels
        label_probs: (n_samples, n_classes) with probabilities
    """
    # Initialize label model
    label_model = LabelModel(cardinality=2, verbose=True)

    # Fit on label matrix
    label_model.fit(
        L_train=label_matrix,
        n_epochs=n_epochs,
        log_freq=100,
        seed=42
    )

    # Predict
    preds = label_model.predict(L=label_matrix)
    probs = label_model.predict_proba(L=label_matrix)

    return preds, probs
```

**Pros:** Learns LF accuracies, handles correlations
**Cons:** Slower training, requires tuning
**Performance:** 75-85% accuracy typical

### 4.3 FlyingSquid (Recommended)

Fast triplet-based aggregation.

```python
from flyingsquid.label_model import LabelModel as FlyingSquidModel

def flyingsquid_aggregation(label_matrix):
    """
    Aggregate labels using FlyingSquid (170× faster than Snorkel).

    Args:
        label_matrix: (n_samples, n_lfs) label matrix

    Returns:
        aggregated_labels: (n_samples,) with learned labels
        label_probs: (n_samples, n_classes) with probabilities
    """
    # Initialize FlyingSquid model
    model = FlyingSquidModel(m=label_matrix.shape[1])

    # Fit (learns triplet parameters)
    model.fit(label_matrix)

    # Predict
    preds = model.predict(label_matrix)
    probs = model.predict_proba(label_matrix)

    return preds, probs
```

**Pros:** 170× faster than Snorkel, better quality
**Cons:** Less documentation, newer framework
**Performance:** 75-85% accuracy, <1 second for 10k samples

### 4.4 Weighted Voting

Weight LFs by accuracy on dev set.

```python
def weighted_voting(label_matrix, lf_weights):
    """
    Aggregate labels using weighted voting.

    Args:
        label_matrix: (n_samples, n_lfs) label matrix
        lf_weights: (n_lfs,) accuracy weights for each LF

    Returns:
        aggregated_labels: (n_samples,) with weighted labels
    """
    aggregated = []
    for row in label_matrix:
        # Weighted vote
        weighted_votes = defaultdict(float)
        for vote, weight in zip(row, lf_weights):
            if vote != 0:  # Not abstain
                weighted_votes[vote] += weight

        if not weighted_votes:
            aggregated.append(0)  # All abstain
        else:
            # Highest weighted vote
            aggregated.append(max(weighted_votes, key=weighted_votes.get))

    return np.array(aggregated)
```

**Pros:** Simple, effective with dev set
**Cons:** Requires labeled dev set
**Performance:** 70-80% accuracy typical

---

## 5. LF Quality Metrics

### 5.1 Coverage

Percentage of examples where LF does not abstain.

```python
def calculate_coverage(label_matrix, lf_idx):
    """
    Calculate coverage for a labeling function.

    Coverage = % of samples where LF votes (not abstain)

    Args:
        label_matrix: (n_samples, n_lfs) label matrix
        lf_idx: Index of LF to analyze

    Returns:
        coverage: Float in [0, 1]
    """
    lf_votes = label_matrix[:, lf_idx]
    non_abstain = np.sum(lf_votes != 0)
    return non_abstain / len(lf_votes)
```

**Interpretation:**
- Coverage < 0.1: Low coverage, may not be useful
- Coverage 0.1-0.3: Moderate coverage, specific patterns
- Coverage > 0.3: High coverage, broad patterns

### 5.2 Accuracy (requires dev set)

Percentage of correct votes (excluding abstentions).

```python
def calculate_accuracy(label_matrix, lf_idx, true_labels):
    """
    Calculate accuracy for a labeling function.

    Accuracy = % correct among non-abstain votes

    Args:
        label_matrix: (n_samples, n_lfs) label matrix
        lf_idx: Index of LF to analyze
        true_labels: (n_samples,) ground truth labels

    Returns:
        accuracy: Float in [0, 1]
    """
    lf_votes = label_matrix[:, lf_idx]

    # Filter to non-abstain
    non_abstain_mask = lf_votes != 0
    if np.sum(non_abstain_mask) == 0:
        return 0.0  # All abstain

    filtered_votes = lf_votes[non_abstain_mask]
    filtered_truth = true_labels[non_abstain_mask]

    correct = np.sum(filtered_votes == filtered_truth)
    return correct / len(filtered_votes)
```

**Interpretation:**
- Accuracy < 0.5: Worse than random, remove
- Accuracy 0.5-0.7: Weak signal, consider improving
- Accuracy > 0.7: Good quality, keep

### 5.3 Conflict Rate

Percentage of examples where LF disagrees with majority.

```python
def calculate_conflicts(label_matrix, lf_idx):
    """
    Calculate conflict rate for a labeling function.

    Conflict = % of votes that disagree with majority

    Args:
        label_matrix: (n_samples, n_lfs) label matrix
        lf_idx: Index of LF to analyze

    Returns:
        conflict_rate: Float in [0, 1]
    """
    lf_votes = label_matrix[:, lf_idx]
    majority_labels = majority_vote(label_matrix)

    # Compare only where LF voted
    non_abstain_mask = lf_votes != 0
    if np.sum(non_abstain_mask) == 0:
        return 0.0

    filtered_votes = lf_votes[non_abstain_mask]
    filtered_majority = majority_labels[non_abstain_mask]

    conflicts = np.sum(filtered_votes != filtered_majority)
    return conflicts / len(filtered_votes)
```

**Interpretation:**
- Conflict < 0.2: Agrees with others, good
- Conflict 0.2-0.4: Some disagreement, normal
- Conflict > 0.4: High disagreement, investigate

### 5.4 Overlap

Percentage of examples labeled by both LFs.

```python
def calculate_overlap(label_matrix, lf_idx1, lf_idx2):
    """
    Calculate overlap between two labeling functions.

    Overlap = % of examples labeled by both LFs

    Args:
        label_matrix: (n_samples, n_lfs) label matrix
        lf_idx1, lf_idx2: Indices of LFs to compare

    Returns:
        overlap: Float in [0, 1]
    """
    lf1_votes = label_matrix[:, lf_idx1]
    lf2_votes = label_matrix[:, lf_idx2]

    both_vote = np.sum((lf1_votes != 0) & (lf2_votes != 0))
    return both_vote / len(lf1_votes)
```

**Use Case:** Identify redundant LFs (high overlap + high agreement)

---

## 6. LLM-Based LF Generation

### 6.1 Pattern Discovery Prompt

```python
def generate_lfs_from_examples(
    examples: pd.DataFrame,
    text_column: str,
    label_column: str,
    num_lfs: int = 10
) -> list[str]:
    """
    Use LLM to generate labeling functions from examples.

    Args:
        examples: Labeled examples
        text_column: Column with text
        label_column: Column with labels
        num_lfs: Number of LFs to generate

    Returns:
        Generated LF code as strings
    """
    # Sample balanced examples
    samples_per_class = {}
    for label in examples[label_column].unique():
        samples_per_class[label] = examples[
            examples[label_column] == label
        ].sample(n=5)

    # Construct prompt
    prompt = f"""You are a data labeling expert. Given examples of labeled text data, generate {num_lfs} Python labeling functions that can programmatically label similar data.

LABELED EXAMPLES:
"""

    for label, sample_df in samples_per_class.items():
        prompt += f"\n--- Label: {label} ---\n"
        for text in sample_df[text_column]:
            prompt += f"- {text}\n"

    prompt += f"""

INSTRUCTIONS:
1. Analyze the examples and identify patterns (keywords, phrases, structure, length, etc.)
2. Generate {num_lfs} labeling functions in Python using this template:

```python
@labeling_function()
def lf_<descriptive_name>(x):
    \"\"\"<description of what this LF does>\"\"\"
    # Your pattern matching logic here
    if <condition>:
        return POSITIVE  # or NEGATIVE or other label
    return ABSTAIN
```

REQUIREMENTS:
- Each LF should capture a different pattern
- Use simple, interpretable logic (keywords, regex, length checks, etc.)
- LFs should have high precision (avoid false positives)
- Return ABSTAIN when uncertain
- Provide clear docstrings

GENERATE {num_lfs} LABELING FUNCTIONS:
"""

    # Get LLM response
    response = llm_client.complete(prompt, max_tokens=2000)

    # Parse generated LFs
    lfs = parse_lf_code(response)

    return lfs
```

### 6.2 Automatic LF Refinement

```python
def refine_lf_with_feedback(
    lf_code: str,
    label_matrix: np.ndarray,
    lf_idx: int,
    true_labels: np.ndarray
) -> str:
    """
    Use LLM to refine an underperforming LF.

    Args:
        lf_code: Original LF code
        label_matrix: Label matrix
        lf_idx: Index of LF
        true_labels: Ground truth labels

    Returns:
        Refined LF code
    """
    # Calculate performance
    accuracy = calculate_accuracy(label_matrix, lf_idx, true_labels)
    coverage = calculate_coverage(label_matrix, lf_idx)

    # Get error examples
    lf_votes = label_matrix[:, lf_idx]
    errors = []
    for i, (vote, truth) in enumerate(zip(lf_votes, true_labels)):
        if vote != 0 and vote != truth:  # Wrong vote
            errors.append(i)

    # Sample errors
    error_sample = np.random.choice(errors, size=min(5, len(errors)), replace=False)

    # Construct refinement prompt
    prompt = f"""You are improving a labeling function that has accuracy={accuracy:.2f} and coverage={coverage:.2f}.

CURRENT LABELING FUNCTION:
```python
{lf_code}
```

ERROR EXAMPLES (where LF was wrong):
"""

    for idx in error_sample:
        prompt += f"- True label: {true_labels[idx]}, LF predicted: {lf_votes[idx]}\n"

    prompt += """

TASK: Refine the labeling function to fix these errors while maintaining coverage.

REFINED LABELING FUNCTION:
```python
"""

    # Get refined LF
    response = llm_client.complete(prompt, max_tokens=500)
    refined_lf = parse_lf_code(response)[0]

    return refined_lf
```

---

## 7. Implementation Details

### 7.1 Core WeakSupervisionService Class

```python
# src/autolabeler/core/weak_supervision/snorkel_integrator.py

from typing import Callable, Literal
import numpy as np
import pandas as pd
from snorkel.labeling import PandasLFApplier, LFAnalysis, LabelModel
from snorkel.labeling import labeling_function
from loguru import logger

from ..configs import WeakSupervisionConfig
from ..base import ConfigurableComponent


# Label constants
ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1


class WeakSupervisionService(ConfigurableComponent):
    """
    Weak supervision service for programmatic labeling at scale.

    Implements Snorkel/FlyingSquid for aggregating noisy labeling functions
    into high-quality training labels.

    Args:
        dataset_name: Name of the dataset
        settings: Application settings
        config: Weak supervision configuration

    Example:
        >>> ws_service = WeakSupervisionService("sentiment", settings)
        >>> ws_service.register_lf(lf_positive_keywords, "positive_keywords")
        >>> ws_service.register_lf(lf_sentiment_model, "sentiment_model")
        >>> labels = ws_service.apply_and_aggregate(unlabeled_df)
    """

    def __init__(
        self,
        dataset_name: str,
        settings,
        config: WeakSupervisionConfig | None = None
    ):
        super().__init__(
            component_type="weak_supervision",
            dataset_name=dataset_name,
            settings=settings
        )
        self.config = config or WeakSupervisionConfig()
        self.labeling_functions: list[Callable] = []
        self.lf_names: list[str] = []
        self.label_model = None
        self.lf_analysis_cache = None

        logger.info(f"WeakSupervisionService initialized for '{dataset_name}'")

    def register_lf(
        self,
        lf: Callable,
        name: str,
        resources: dict | None = None
    ) -> None:
        """
        Register a labeling function.

        Args:
            lf: Labeling function (callable)
            name: Unique name for the LF
            resources: Optional resources (models, data, etc.)

        Example:
            >>> def lf_contains_urgent(x):
            ...     return POSITIVE if "urgent" in x.text.lower() else ABSTAIN
            >>> ws_service.register_lf(lf_contains_urgent, "contains_urgent")
        """
        # Wrap with Snorkel decorator if not already wrapped
        if not hasattr(lf, "name"):
            lf = labeling_function(name=name, resources=resources)(lf)

        self.labeling_functions.append(lf)
        self.lf_names.append(name)

        logger.info(f"Registered labeling function: {name}")

    def apply_lfs(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply all registered labeling functions to dataset.

        Args:
            df: DataFrame with data to label

        Returns:
            Label matrix (n_samples × n_lfs)

        Example:
            >>> label_matrix = ws_service.apply_lfs(unlabeled_df)
            >>> print(f"Shape: {label_matrix.shape}")
        """
        if not self.labeling_functions:
            raise ValueError("No labeling functions registered")

        logger.info(
            f"Applying {len(self.labeling_functions)} labeling functions "
            f"to {len(df)} examples"
        )

        # Apply LFs using Snorkel applier
        applier = PandasLFApplier(lfs=self.labeling_functions)
        L_matrix = applier.apply(df=df)

        logger.info(f"Label matrix shape: {L_matrix.shape}")

        return L_matrix

    def analyze_lfs(
        self,
        L_matrix: np.ndarray,
        Y_dev: np.ndarray | None = None
    ) -> pd.DataFrame:
        """
        Analyze labeling function quality.

        Args:
            L_matrix: Label matrix from apply_lfs()
            Y_dev: Optional ground truth labels for dev set

        Returns:
            DataFrame with LF statistics

        Example:
            >>> analysis = ws_service.analyze_lfs(L_matrix, Y_dev=dev_labels)
            >>> print(analysis[["Polarity", "Coverage", "Overlaps", "Conflicts"]])
        """
        logger.info("Analyzing labeling function quality")

        analysis = LFAnalysis(
            L=L_matrix,
            lfs=self.labeling_functions
        ).lf_summary(Y=Y_dev)

        self.lf_analysis_cache = analysis

        return analysis

    def aggregate_labels(
        self,
        L_matrix: np.ndarray,
        method: Literal["majority", "snorkel", "flyingsquid"] = "snorkel",
        n_epochs: int = 500
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Aggregate labels using statistical model.

        Args:
            L_matrix: Label matrix from apply_lfs()
            method: Aggregation method
            n_epochs: Training epochs (for snorkel/flyingsquid)

        Returns:
            (predictions, probabilities)
                predictions: (n_samples,) aggregated labels
                probabilities: (n_samples, n_classes) label probabilities

        Example:
            >>> preds, probs = ws_service.aggregate_labels(L_matrix, method="snorkel")
            >>> print(f"Predicted {len(preds)} labels")
        """
        logger.info(f"Aggregating labels using {method} method")

        if method == "majority":
            preds = self._majority_vote(L_matrix)
            # Majority vote doesn't provide probabilities
            probs = np.zeros((len(preds), 2))
            probs[np.arange(len(preds)), preds] = 1.0
            return preds, probs

        elif method == "snorkel":
            return self._snorkel_aggregation(L_matrix, n_epochs)

        elif method == "flyingsquid":
            return self._flyingsquid_aggregation(L_matrix)

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def apply_and_aggregate(
        self,
        df: pd.DataFrame,
        method: Literal["majority", "snorkel", "flyingsquid"] = "snorkel"
    ) -> pd.DataFrame:
        """
        Apply LFs and aggregate in one step.

        Args:
            df: DataFrame to label
            method: Aggregation method

        Returns:
            DataFrame with predicted labels and confidences

        Example:
            >>> labeled_df = ws_service.apply_and_aggregate(unlabeled_df)
            >>> print(labeled_df[["text", "ws_label", "ws_confidence"]])
        """
        # Apply LFs
        L_matrix = self.apply_lfs(df)

        # Aggregate
        preds, probs = self.aggregate_labels(L_matrix, method=method)

        # Add to dataframe
        result_df = df.copy()
        result_df["ws_label"] = preds
        result_df["ws_confidence"] = np.max(probs, axis=1)

        # Mark abstentions
        abstain_mask = np.all(L_matrix == ABSTAIN, axis=1)
        result_df.loc[abstain_mask, "ws_label"] = None

        logger.info(
            f"Labeled {len(result_df)} examples. "
            f"Abstentions: {np.sum(abstain_mask)} ({np.mean(abstain_mask)*100:.1f}%)"
        )

        return result_df

    def _majority_vote(self, L_matrix: np.ndarray) -> np.ndarray:
        """Simple majority voting."""
        preds = []
        for row in L_matrix:
            votes = row[row != ABSTAIN]
            if len(votes) == 0:
                preds.append(ABSTAIN)
            else:
                preds.append(np.bincount(votes).argmax())
        return np.array(preds)

    def _snorkel_aggregation(
        self,
        L_matrix: np.ndarray,
        n_epochs: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Snorkel label model aggregation."""
        # Determine cardinality (number of classes)
        unique_labels = np.unique(L_matrix)
        cardinality = len(unique_labels[unique_labels != ABSTAIN])

        # Initialize label model
        self.label_model = LabelModel(cardinality=cardinality, verbose=True)

        # Fit
        self.label_model.fit(
            L_train=L_matrix,
            n_epochs=n_epochs,
            log_freq=100,
            seed=42
        )

        # Predict
        preds = self.label_model.predict(L=L_matrix)
        probs = self.label_model.predict_proba(L=L_matrix)

        return preds, probs

    def _flyingsquid_aggregation(
        self,
        L_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """FlyingSquid aggregation (faster than Snorkel)."""
        try:
            from flyingsquid.label_model import LabelModel as FlyingSquidModel
        except ImportError:
            logger.warning("FlyingSquid not installed, falling back to Snorkel")
            return self._snorkel_aggregation(L_matrix, n_epochs=500)

        # Initialize
        model = FlyingSquidModel(m=L_matrix.shape[1])

        # Fit
        model.fit(L_matrix)

        # Predict
        preds = model.predict(L_matrix)
        probs = model.predict_proba(L_matrix)

        return preds, probs

    def save_lfs(self, output_path: str) -> None:
        """Save labeling functions to file."""
        # This would save LF code/metadata
        # Implementation depends on how LFs are stored
        logger.info(f"Saving {len(self.labeling_functions)} LFs to {output_path}")

    def load_lfs(self, input_path: str) -> None:
        """Load labeling functions from file."""
        logger.info(f"Loading LFs from {input_path}")
```

---

## 8. Configuration

### 8.1 WeakSupervisionConfig Class

```python
# Add to src/autolabeler/core/configs.py

class WeakSupervisionConfig(BaseModel):
    """Configuration for weak supervision."""

    # Aggregation method
    aggregation_method: Literal["majority", "snorkel", "flyingsquid"] = Field(
        "snorkel",
        description="Label aggregation method"
    )

    # Snorkel parameters
    n_epochs: int = Field(
        500,
        gt=0,
        description="Training epochs for Snorkel label model"
    )
    learning_rate: float = Field(
        0.01,
        gt=0,
        description="Learning rate for label model"
    )

    # LF generation parameters
    enable_lf_generation: bool = Field(
        True,
        description="Enable LLM-based LF generation"
    )
    num_generated_lfs: int = Field(
        10,
        gt=0,
        description="Number of LFs to generate with LLM"
    )
    lf_generation_model: str = Field(
        "gpt-4o-mini",
        description="Model for LF generation"
    )

    # Quality thresholds
    min_lf_coverage: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="Minimum coverage for LF to be kept"
    )
    min_lf_accuracy: float = Field(
        0.55,
        ge=0.0,
        le=1.0,
        description="Minimum accuracy for LF to be kept (if dev set available)"
    )
    max_lf_conflicts: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Maximum conflict rate for LF"
    )

    # Processing
    batch_size: int = Field(
        1000,
        gt=0,
        description="Batch size for applying LFs"
    )
    text_column: str = Field(
        "text",
        description="Column containing text"
    )

    # Output
    save_label_matrix: bool = Field(
        True,
        description="Save label matrix for debugging"
    )
    save_lf_analysis: bool = Field(
        True,
        description="Save LF analysis report"
    )
```

---

## 9. Usage Examples

### 9.1 Basic Weak Supervision

```python
from autolabeler import AutoLabeler
from autolabeler.core.weak_supervision import WeakSupervisionService, ABSTAIN, POSITIVE, NEGATIVE
from autolabeler.core.configs import WeakSupervisionConfig
from snorkel.labeling import labeling_function

# Define labeling functions
@labeling_function()
def lf_positive_keywords(x):
    positive_words = {"excellent", "great", "amazing", "love"}
    return POSITIVE if any(w in x.text.lower() for w in positive_words) else ABSTAIN

@labeling_function()
def lf_negative_keywords(x):
    negative_words = {"terrible", "awful", "hate", "worst"}
    return NEGATIVE if any(w in x.text.lower() for w in negative_words) else ABSTAIN

@labeling_function()
def lf_exclamation(x):
    # Multiple exclamation marks suggest strong sentiment
    if x.text.count("!") >= 2:
        # Need more context to determine polarity
        return POSITIVE if "!" in x.text[:len(x.text)//2] else ABSTAIN
    return ABSTAIN

# Initialize service
ws_service = WeakSupervisionService(
    dataset_name="sentiment",
    settings=settings
)

# Register LFs
ws_service.register_lf(lf_positive_keywords, "positive_keywords")
ws_service.register_lf(lf_negative_keywords, "negative_keywords")
ws_service.register_lf(lf_exclamation, "exclamation_marks")

# Load data
import pandas as pd
unlabeled_df = pd.read_csv("data/unlabeled.csv")

# Apply and aggregate
labeled_df = ws_service.apply_and_aggregate(
    unlabeled_df,
    method="snorkel"
)

# Save results
labeled_df.to_csv("results/weakly_labeled.csv", index=False)

print(f"Labeled {len(labeled_df)} examples")
print(f"Mean confidence: {labeled_df['ws_confidence'].mean():.3f}")
```

### 9.2 LF Quality Analysis

```python
# Apply LFs
L_matrix = ws_service.apply_lfs(unlabeled_df)

# Analyze with dev set
dev_df = pd.read_csv("data/dev.csv")
dev_labels = dev_df["label"].values

analysis = ws_service.analyze_lfs(L_matrix, Y_dev=dev_labels)

print(analysis)
# Output:
#              j  Polarity  Coverage  Overlaps  Conflicts  Correct  Incorrect  Emp. Acc.
# positive_keywords   0       [1]      0.234      0.145      0.089      102         18      0.850
# negative_keywords   1       [0]      0.189      0.134      0.076       85         12      0.876
# exclamation_marks   2  [0, 1, 2]      0.067      0.045      0.123       28         15      0.651

# Filter low-quality LFs
good_lfs = analysis[
    (analysis["Emp. Acc."] > 0.70) &
    (analysis["Coverage"] > 0.10)
]

print(f"Keeping {len(good_lfs)} high-quality LFs")
```

### 9.3 LLM-Generated LFs

```python
from autolabeler.core.weak_supervision import LFGenerator

# Initialize generator
lf_generator = LFGenerator(
    labeling_service=labeler.labeling_service,
    config=WeakSupervisionConfig()
)

# Generate LFs from examples
examples_df = pd.read_csv("data/seed_labeled.csv")

generated_lfs = lf_generator.generate_from_examples(
    examples=examples_df,
    text_column="text",
    label_column="label",
    num_lfs=10
)

print(f"Generated {len(generated_lfs)} labeling functions")

# Register generated LFs
for lf_code, lf_name in generated_lfs:
    # Execute LF code and register
    exec(lf_code)  # Be careful with exec in production!
    lf_func = locals()[lf_name]
    ws_service.register_lf(lf_func, lf_name)
```

### 9.4 Hybrid: Weak Supervision + Active Learning

```python
from autolabeler.core.configs import ActiveLearningConfig

# Step 1: Use weak supervision for bulk labeling
ws_config = WeakSupervisionConfig(aggregation_method="snorkel")
weakly_labeled_df = ws_service.apply_and_aggregate(unlabeled_df)

# Step 2: Filter high-confidence labels for training
high_conf_mask = weakly_labeled_df["ws_confidence"] > 0.8
training_df = weakly_labeled_df[high_conf_mask]

print(f"High-confidence training set: {len(training_df)} examples")

# Step 3: Use active learning for low-confidence examples
low_conf_df = weakly_labeled_df[~high_conf_mask]

al_config = ActiveLearningConfig(
    strategy="uncertainty",
    batch_size=50,
    max_iterations=5
)

al_labeled_df = labeler.label_with_active_learning(
    unlabeled_df=low_conf_df.drop(columns=["ws_label", "ws_confidence"]),
    al_config=al_config,
    seed_labeled_df=training_df
)

# Step 4: Combine
final_labeled_df = pd.concat([training_df, al_labeled_df], ignore_index=True)

print(f"Final labeled dataset: {len(final_labeled_df)} examples")
print(f"WS contribution: {len(training_df)} ({len(training_df)/len(final_labeled_df)*100:.1f}%)")
print(f"AL contribution: {len(al_labeled_df)} ({len(al_labeled_df)/len(final_labeled_df)*100:.1f}%)")
```

---

## 10. Success Metrics

### 10.1 Primary Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Speed** | 10-100× faster | vs. manual annotation |
| **Cost** | 50-100× cheaper | vs. manual annotation |
| **Coverage** | 60-90% | % of examples labeled (non-abstain) |
| **Aggregated Accuracy** | 75-85% | On test set |

### 10.2 LF Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **LF Coverage** | >10% per LF | Individual LF coverage |
| **LF Accuracy** | >70% per LF | Individual LF accuracy (on dev set) |
| **LF Diversity** | <30% overlap | Pairwise LF overlap |
| **Conflict Rate** | <20% | Average LF conflict rate |

---

## 11. Integration Testing

### 11.1 Unit Tests

```python
def test_lf_registration():
    """Test registering labeling functions."""
    ws_service = WeakSupervisionService("test", settings)
    ws_service.register_lf(lf_test, "test_lf")
    assert len(ws_service.labeling_functions) == 1

def test_lf_application():
    """Test applying LFs to data."""
    L_matrix = ws_service.apply_lfs(test_df)
    assert L_matrix.shape == (len(test_df), len(ws_service.labeling_functions))

def test_majority_vote():
    """Test majority vote aggregation."""
    L_matrix = np.array([[1, 1, 0], [-1, -1, 0], [1, -1, 0]])
    preds = ws_service._majority_vote(L_matrix)
    assert np.array_equal(preds, [1, -1, -1])  # Most common non-abstain

def test_snorkel_aggregation():
    """Test Snorkel label model."""
    preds, probs = ws_service.aggregate_labels(L_matrix, method="snorkel")
    assert len(preds) == len(L_matrix)
    assert probs.shape[0] == len(L_matrix)
```

---

## 12. Next Steps

1. ✅ Specification complete
2. ⏭️ Implement `snorkel_integrator.py`
3. ⏭️ Implement `labeling_functions.py` (utilities)
4. ⏭️ Implement `lf_generator.py` (LLM-based generation)
5. ⏭️ Add `WeakSupervisionConfig` to `configs.py`
6. ⏭️ Create example labeling functions
7. ⏭️ Write tests
8. ⏭️ Benchmark on datasets

---

**Specification Status:** ✅ COMPLETE - Ready for Implementation
**Companion Document:** phase2_active_learning_spec.md
**Dependencies:** Phase 1 (Quality Metrics, Confidence Calibration)

---

*Generated by ANALYST Agent (Hive Mind Swarm)*
*Date: October 7, 2025*
*Working Directory: /home/nick/python/autolabeler*
