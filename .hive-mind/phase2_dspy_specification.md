# Phase 2: DSPy Prompt Optimization - Technical Specification

**Document Version:** 1.0
**Last Updated:** 2025-10-07
**Status:** RESEARCH COMPLETE - READY FOR IMPLEMENTATION
**Owner:** RESEARCHER Agent

---

## Executive Summary

This specification details the integration of DSPy (Stanford's DSP research framework) with AutoLabeler's existing LabelingService to enable systematic prompt optimization using the state-of-the-art MIPROv2 optimizer. This integration will replace manual prompt engineering with algorithmic optimization, targeting **20-50% accuracy improvements** while maintaining compatibility with existing infrastructure.

### Key Objectives
- Integrate DSPy MIPROv2 optimizer with existing LabelingService
- Maintain backward compatibility with current prompt templates
- Enable automatic prompt optimization from labeled training data
- Provide CLI and programmatic interfaces for optimization workflows
- Support A/B testing of optimized vs baseline prompts

### Expected Impact
- **Accuracy Improvement:** +20-50% over hand-crafted prompts
- **Optimization Time:** <20 minutes per dataset
- **Optimization Cost:** $2-5 per optimization run
- **Reproducibility:** Deterministic with fixed random seed

---

## Table of Contents

1. [Background & Motivation](#background--motivation)
2. [Architecture Overview](#architecture-overview)
3. [Component Specifications](#component-specifications)
4. [Integration Points](#integration-points)
5. [Configuration Schema](#configuration-schema)
6. [API Design](#api-design)
7. [Implementation Guide](#implementation-guide)
8. [Testing Strategy](#testing-strategy)
9. [Migration Path](#migration-path)
10. [Dependencies](#dependencies)

---

## Background & Motivation

### Current State
AutoLabeler currently uses:
- Manual Jinja2 prompt templates
- LangChain for LLM interaction
- Instructor for structured output validation
- Static prompts without systematic optimization

**Limitations:**
- Manual prompt engineering is time-consuming
- No systematic way to improve prompts from data
- Difficult to compare prompt variants objectively
- No learning from successful predictions

### DSPy Overview

**DSPy** (Declarative Self-improving Language Programs) is a framework from Stanford that treats prompts as **learnable parameters** rather than hand-crafted strings.

**Key Features:**
- **Signatures:** Type-annotated interfaces for LLM tasks
- **Modules:** Composable building blocks (ChainOfThought, ReAct, etc.)
- **Optimizers:** Algorithms that learn better prompts from data
- **MIPROv2:** Multi-prompt Instruction Proposal Optimizer v2 (SOTA)

**MIPROv2 Process:**
1. **Bootstrapping:** Runs program across inputs, collects traces
2. **Grounded Proposal:** Generates instruction candidates from traces
3. **Discrete Search:** Uses Bayesian optimization to find best combinations

**Performance Benchmarks:**
- Raises ReAct from 24% → 51% on reasoning tasks (gpt-4o-mini)
- Requires 200+ examples to prevent overfitting
- Typical cost: $2-5 per optimization run
- Runtime: 15-25 minutes for 40 trials

---

## Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    LabelingService                          │
│  (Existing - Enhanced with DSPy Support)                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ├──> Current Path (Jinja2 + LangChain)
                  │    - label_text()
                  │    - label_dataframe()
                  │
                  └──> New DSPy Path
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              DSPyOptimizer (NEW)                            │
│  - optimize_labeling_prompt()                               │
│  - save_optimized_program()                                 │
│  - evaluate_program()                                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ├──> DSPy Core
                  │    - dspy.OpenAI / dspy.LM
                  │    - MIPROv2 optimizer
                  │    - ChainOfThought / ReAct modules
                  │
                  └──> Storage & Tracking
                       - OptimizedProgramStore
                       - PromptManager (existing)
                       - ABTestManager
```

### Integration Strategy

**Approach:** Additive, not disruptive
- DSPy runs **alongside** existing system
- Optimized prompts stored as JSON artifacts
- LabelingService can use either path
- Gradual migration via feature flags

---

## Component Specifications

### 1. DSPyOptimizer Class

**Location:** `src/autolabeler/core/prompt_optimization/dspy_optimizer.py`

**Responsibilities:**
- Initialize DSPy language models
- Convert AutoLabeler examples to DSPy format
- Run MIPROv2 optimization
- Evaluate optimized programs
- Save/load optimized artifacts

**API:**

```python
class DSPyOptimizer:
    """Optimize prompts using DSPy MIPROv2."""

    def __init__(
        self,
        config: DSPyConfig,
        labeling_service: LabelingService
    ):
        """
        Initialize optimizer.

        Args:
            config: DSPy optimization configuration
            labeling_service: Reference to labeling service for data access
        """

    def optimize_labeling_prompt(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        text_column: str,
        label_column: str,
        allowed_labels: list[str],
        num_candidates: int = 10,
        num_trials: int = 20,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 8,
        random_seed: int = 42
    ) -> DSPyOptimizationResult:
        """
        Optimize labeling prompt using MIPROv2.

        Process:
        1. Convert DataFrames to dspy.Example format
        2. Define LabelingSignature and LabelingModule
        3. Run MIPROv2 optimizer with specified parameters
        4. Evaluate on validation set
        5. Return results with cost tracking

        Returns:
            DSPyOptimizationResult with optimized module and metrics
        """

    def evaluate_program(
        self,
        program: dspy.Module,
        eval_df: pd.DataFrame,
        text_column: str,
        label_column: str
    ) -> dict[str, float]:
        """Evaluate DSPy program on test data."""

    def save_optimized_program(
        self,
        program: dspy.Module,
        result: DSPyOptimizationResult,
        output_path: Path
    ):
        """Save optimized program to disk."""

    def load_optimized_program(
        self,
        program_path: Path
    ) -> dspy.Module:
        """Load optimized program from disk."""
```

### 2. DSPy Task Modules

**LabelingSignature:**
```python
class LabelingSignature(dspy.Signature):
    """Classify text into predefined categories."""

    text: str = dspy.InputField(
        desc="Text to classify"
    )
    context_examples: str = dspy.InputField(
        desc="Similar labeled examples for reference",
        optional=True
    )
    allowed_labels: str = dspy.InputField(
        desc="Valid label categories"
    )
    label: str = dspy.OutputField(
        desc="Predicted label from allowed_labels"
    )
    reasoning: str = dspy.OutputField(
        desc="Step-by-step explanation of classification decision"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score between 0.0 and 1.0"
    )
```

**LabelingModule:**
```python
class LabelingModule(dspy.Module):
    """DSPy module for text labeling with RAG support."""

    def __init__(
        self,
        strategy: str = "cot",  # "cot" or "react"
        use_rag: bool = True
    ):
        super().__init__()

        if strategy == "cot":
            self.predictor = dspy.ChainOfThought(LabelingSignature)
        elif strategy == "react":
            self.predictor = dspy.ReAct(LabelingSignature)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.use_rag = use_rag

    def forward(self, text: str, allowed_labels: list[str]) -> dspy.Prediction:
        """
        Forward pass for labeling.

        Process:
        1. Retrieve similar examples if use_rag=True
        2. Format context and allowed_labels
        3. Run predictor
        4. Parse and validate output
        """

        # RAG retrieval (if enabled)
        context_examples = ""
        if self.use_rag:
            similar = self._retrieve_similar_examples(text)
            context_examples = self._format_examples(similar)

        # Format allowed labels
        labels_str = ", ".join(allowed_labels)

        # Predict
        prediction = self.predictor(
            text=text,
            context_examples=context_examples,
            allowed_labels=labels_str
        )

        return prediction
```

### 3. Metric Functions

```python
def accuracy_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None
) -> float:
    """Binary accuracy for label matching."""
    return 1.0 if example.label == prediction.label else 0.0

def weighted_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None
) -> float:
    """
    Weighted metric combining accuracy and confidence calibration.

    Score = accuracy * calibration_quality
    """
    correct = float(example.label == prediction.label)

    # Reward well-calibrated confidence
    confidence = prediction.confidence
    expected_conf = 1.0 if correct else 0.0
    calibration_error = abs(confidence - expected_conf)
    calibration_quality = 1.0 - calibration_error

    return correct * calibration_quality

def f1_metric(
    examples: list[dspy.Example],
    predictions: list[dspy.Prediction]
) -> float:
    """Macro F1 score across all labels."""
    from sklearn.metrics import f1_score

    y_true = [ex.label for ex in examples]
    y_pred = [pred.label for pred in predictions]

    return f1_score(y_true, y_pred, average='macro')
```

### 4. Configuration Schema

```python
from pydantic import BaseModel, Field

class DSPyConfig(BaseModel):
    """Configuration for DSPy optimization."""

    # LLM Configuration
    model_name: str = Field(
        "gpt-4o-mini",
        description="Model to use for DSPy (OpenAI, Anthropic, etc.)"
    )
    api_key: str | None = Field(
        None,
        description="API key (defaults to environment variable)"
    )
    base_url: str | None = Field(
        None,
        description="Custom base URL for LLM provider"
    )

    # Optimization Parameters
    num_candidates: int = Field(
        10,
        description="Number of instruction candidates per iteration"
    )
    num_trials: int = Field(
        20,
        description="Number of optimization trials (recommend 40+ for production)"
    )
    max_bootstrapped_demos: int = Field(
        4,
        description="Maximum bootstrapped demonstrations"
    )
    max_labeled_demos: int = Field(
        8,
        description="Maximum labeled demonstrations in prompts"
    )
    init_temperature: float = Field(
        1.0,
        description="Initial temperature for exploration"
    )

    # Module Configuration
    strategy: str = Field(
        "cot",
        description="Reasoning strategy: 'cot' (Chain-of-Thought) or 'react'"
    )
    use_rag: bool = Field(
        True,
        description="Use RAG for retrieving similar examples"
    )

    # Optimization Behavior
    metric: str = Field(
        "accuracy",
        description="Optimization metric: 'accuracy', 'f1', 'weighted'"
    )
    auto_mode: str = Field(
        "medium",
        description="Auto mode: 'light', 'medium', 'heavy'"
    )
    random_seed: int = Field(
        42,
        description="Random seed for reproducibility"
    )

    # Stopping Criteria
    min_improvement: float = Field(
        0.01,
        description="Minimum accuracy improvement to continue"
    )
    patience: int = Field(
        3,
        description="Trials without improvement before early stopping"
    )

    # Output Configuration
    save_intermediate: bool = Field(
        True,
        description="Save intermediate optimization results"
    )
    verbose: bool = Field(
        True,
        description="Print optimization progress"
    )


class DSPyOptimizationResult(BaseModel):
    """Result of DSPy optimization."""

    optimized_module: Any = Field(
        description="Optimized DSPy module (not serializable directly)"
    )
    best_prompt: str = Field(
        description="Best prompt instructions found"
    )

    # Performance Metrics
    baseline_accuracy: float = Field(
        description="Accuracy before optimization"
    )
    optimized_accuracy: float = Field(
        description="Accuracy after optimization"
    )
    accuracy_improvement: float = Field(
        description="Absolute accuracy improvement"
    )
    relative_improvement: float = Field(
        description="Relative improvement percentage"
    )

    # Cost Tracking
    total_cost: float = Field(
        description="Total optimization cost in USD"
    )
    num_trials: int = Field(
        description="Number of trials completed"
    )

    # Optimization Metadata
    optimization_time: float = Field(
        description="Optimization duration in seconds"
    )
    timestamp: str = Field(
        description="ISO timestamp of optimization"
    )
    config: DSPyConfig = Field(
        description="Configuration used"
    )

    # Saved Artifacts
    program_path: Path | None = Field(
        None,
        description="Path to saved program"
    )

    class Config:
        arbitrary_types_allowed = True
```

---

## Integration Points

### 1. LabelingService Integration

**Extend existing LabelingService:**

```python
# src/autolabeler/core/labeling/labeling_service.py

class LabelingService:

    def __init__(self, ...):
        # ... existing initialization ...

        # NEW: DSPy optimizer (lazy initialization)
        self._dspy_optimizer: DSPyOptimizer | None = None
        self.optimized_programs: dict[str, dspy.Module] = {}

    def get_dspy_optimizer(self) -> DSPyOptimizer:
        """Lazy initialization of DSPy optimizer."""
        if self._dspy_optimizer is None:
            self._dspy_optimizer = DSPyOptimizer(
                config=self._get_dspy_config(),
                labeling_service=self
            )
        return self._dspy_optimizer

    def optimize_prompts(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        text_column: str,
        label_column: str,
        config: DSPyConfig | None = None
    ) -> DSPyOptimizationResult:
        """
        Optimize prompts using DSPy MIPROv2.

        This is a high-level convenience method that:
        1. Validates input data
        2. Runs DSPy optimization
        3. Saves optimized program
        4. Updates internal state

        Returns:
            DSPyOptimizationResult
        """
        optimizer = self.get_dspy_optimizer()

        result = optimizer.optimize_labeling_prompt(
            train_df=train_df,
            val_df=val_df,
            text_column=text_column,
            label_column=label_column,
            allowed_labels=self.config.allowed_labels,
            **(config.dict() if config else {})
        )

        # Store optimized program
        program_id = f"{self.dataset_name}_optimized"
        self.optimized_programs[program_id] = result.optimized_module

        return result

    def label_text_with_dspy(
        self,
        text: str,
        program_id: str = "default",
        allowed_labels: list[str] | None = None
    ) -> LabelResponse:
        """
        Label text using optimized DSPy program.

        Args:
            text: Text to label
            program_id: ID of optimized program to use
            allowed_labels: Override allowed labels

        Returns:
            LabelResponse
        """
        if program_id not in self.optimized_programs:
            raise ValueError(f"No optimized program found: {program_id}")

        program = self.optimized_programs[program_id]
        labels = allowed_labels or self.config.allowed_labels

        prediction = program(text=text, allowed_labels=labels)

        return LabelResponse(
            label=prediction.label,
            confidence=prediction.confidence,
            reasoning=prediction.reasoning
        )
```

### 2. CLI Integration

**New commands:**

```bash
# Optimize prompts for a dataset
autolabeler dspy optimize \
    --train-data data/train.csv \
    --val-data data/val.csv \
    --text-column text \
    --label-column label \
    --num-trials 20 \
    --output config/optimized_prompt.json

# Evaluate optimized program
autolabeler dspy evaluate \
    --program config/optimized_prompt.json \
    --test-data data/test.csv \
    --text-column text \
    --label-column label

# Label with optimized program
autolabeler label \
    --input data/unlabeled.csv \
    --dspy-program config/optimized_prompt.json \
    --output results/labeled.csv

# A/B test: baseline vs optimized
autolabeler dspy ab-test \
    --program-a config/baseline.json \
    --program-b config/optimized.json \
    --test-data data/test.csv \
    --text-column text \
    --label-column label
```

**CLI Implementation:**

```python
# src/autolabeler/cli.py

@cli.group()
def dspy():
    """DSPy prompt optimization commands."""
    pass

@dspy.command()
@click.option("--train-data", type=click.Path(exists=True), required=True)
@click.option("--val-data", type=click.Path(exists=True), required=True)
@click.option("--text-column", default="text")
@click.option("--label-column", default="label")
@click.option("--num-trials", default=20)
@click.option("--output", type=click.Path(), required=True)
def optimize(
    train_data: str,
    val_data: str,
    text_column: str,
    label_column: str,
    num_trials: int,
    output: str
):
    """Optimize prompts using DSPy MIPROv2."""

    # Load data
    train_df = pd.read_csv(train_data)
    val_df = pd.read_csv(val_data)

    # Initialize service
    settings = Settings()
    service = LabelingService(
        dataset_name="optimization",
        settings=settings
    )

    # Run optimization
    click.echo(f"Starting DSPy optimization with {num_trials} trials...")
    result = service.optimize_prompts(
        train_df=train_df,
        val_df=val_df,
        text_column=text_column,
        label_column=label_column,
        config=DSPyConfig(num_trials=num_trials)
    )

    # Save results
    service.get_dspy_optimizer().save_optimized_program(
        program=result.optimized_module,
        result=result,
        output_path=Path(output)
    )

    click.echo(f"✅ Optimization complete!")
    click.echo(f"  Baseline accuracy: {result.baseline_accuracy:.1%}")
    click.echo(f"  Optimized accuracy: {result.optimized_accuracy:.1%}")
    click.echo(f"  Improvement: +{result.relative_improvement:.1%}")
    click.echo(f"  Cost: ${result.total_cost:.2f}")
    click.echo(f"  Saved to: {output}")
```

### 3. A/B Testing Infrastructure

```python
# src/autolabeler/core/prompt_optimization/ab_testing.py

from dataclasses import dataclass
from scipy.stats import ttest_ind

@dataclass
class ABTestResult:
    """Result of A/B test."""
    variant_a_accuracy: float
    variant_b_accuracy: float
    accuracy_lift: float
    p_value: float
    is_significant: bool
    winner: str  # "a", "b", or "tie"
    sample_size: int
    confidence_level: float = 0.95

class ABTestManager:
    """Manage A/B tests for prompt variants."""

    def run_ab_test(
        self,
        program_a: dspy.Module,
        program_b: dspy.Module,
        test_df: pd.DataFrame,
        text_column: str,
        label_column: str,
        significance_level: float = 0.05
    ) -> ABTestResult:
        """
        Run A/B test comparing two DSPy programs.

        Process:
        1. Evaluate both programs on same test set
        2. Compute accuracy for each
        3. Run t-test for statistical significance
        4. Determine winner
        """

        # Evaluate both programs
        results_a = self._evaluate_program(program_a, test_df, text_column, label_column)
        results_b = self._evaluate_program(program_b, test_df, text_column, label_column)

        # Extract binary results (correct/incorrect)
        binary_a = [1 if r else 0 for r in results_a]
        binary_b = [1 if r else 0 for r in results_b]

        # Compute metrics
        acc_a = np.mean(binary_a)
        acc_b = np.mean(binary_b)
        lift = acc_b - acc_a

        # Statistical test
        t_stat, p_value = ttest_ind(binary_a, binary_b)
        is_significant = p_value < significance_level

        # Determine winner
        if not is_significant:
            winner = "tie"
        else:
            winner = "b" if acc_b > acc_a else "a"

        return ABTestResult(
            variant_a_accuracy=acc_a,
            variant_b_accuracy=acc_b,
            accuracy_lift=lift,
            p_value=p_value,
            is_significant=is_significant,
            winner=winner,
            sample_size=len(test_df)
        )
```

---

## Implementation Guide

### Step 1: Install Dependencies

```bash
pip install dspy-ai>=2.5.0
```

### Step 2: Create Directory Structure

```
src/autolabeler/core/prompt_optimization/
├── __init__.py
├── dspy_optimizer.py       # Main optimizer class
├── dspy_modules.py          # Task-specific DSPy modules
├── metrics.py               # Evaluation metrics
├── ab_testing.py            # A/B testing infrastructure
└── utils.py                 # Helper functions
```

### Step 3: Implement DSPyOptimizer

See [Component Specifications](#component-specifications) above.

### Step 4: Integrate with LabelingService

See [Integration Points](#integration-points) above.

### Step 5: Add CLI Commands

See [CLI Integration](#cli-integration) above.

### Step 6: Create Example Workflows

```python
# examples/dspy_optimization_example.py

from autolabeler import Settings, LabelingService, DSPyConfig
import pandas as pd

# Load training data
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")

# Initialize service
settings = Settings()
service = LabelingService(
    dataset_name="sentiment",
    settings=settings
)

# Configure DSPy optimization
dspy_config = DSPyConfig(
    model_name="gpt-4o-mini",
    num_trials=20,
    strategy="cot",  # Chain-of-Thought
    use_rag=True,
    metric="f1"
)

# Run optimization
result = service.optimize_prompts(
    train_df=train_df,
    val_df=val_df,
    text_column="text",
    label_column="label",
    config=dspy_config
)

print(f"Baseline: {result.baseline_accuracy:.1%}")
print(f"Optimized: {result.optimized_accuracy:.1%}")
print(f"Improvement: +{result.relative_improvement:.1%}")
print(f"Cost: ${result.total_cost:.2f}")

# Use optimized program
test_text = "This product exceeded my expectations!"
response = service.label_text_with_dspy(
    text=test_text,
    program_id="sentiment_optimized"
)

print(f"Label: {response.label}")
print(f"Confidence: {response.confidence:.2f}")
print(f"Reasoning: {response.reasoning}")
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_dspy_optimizer.py

import pytest
from autolabeler.core.prompt_optimization import DSPyOptimizer, DSPyConfig

def test_dspy_optimizer_initialization():
    """Test DSPyOptimizer initializes correctly."""
    config = DSPyConfig(model_name="gpt-4o-mini")
    optimizer = DSPyOptimizer(config)

    assert optimizer.config.model_name == "gpt-4o-mini"
    assert optimizer.lm is not None

def test_example_conversion():
    """Test DataFrame to dspy.Example conversion."""
    df = pd.DataFrame({
        "text": ["Good product", "Bad service"],
        "label": ["positive", "negative"]
    })

    examples = optimizer._dataframe_to_examples(
        df, text_column="text", label_column="label"
    )

    assert len(examples) == 2
    assert examples[0].text == "Good product"
    assert examples[0].label == "positive"

@pytest.mark.slow
def test_optimization_run(sample_train_data, sample_val_data):
    """Integration test: full optimization run."""
    config = DSPyConfig(
        num_trials=5,  # Small for testing
        random_seed=42
    )

    optimizer = DSPyOptimizer(config)
    result = optimizer.optimize_labeling_prompt(
        train_df=sample_train_data,
        val_df=sample_val_data,
        text_column="text",
        label_column="label",
        allowed_labels=["positive", "negative"]
    )

    assert result.optimized_accuracy >= result.baseline_accuracy
    assert result.total_cost > 0
    assert result.num_trials == 5
```

### Integration Tests

```python
# tests/integration/test_dspy_labeling_service.py

def test_labeling_service_dspy_integration():
    """Test LabelingService can use DSPy for labeling."""
    service = LabelingService(
        dataset_name="test",
        settings=Settings()
    )

    # Optimize prompts
    result = service.optimize_prompts(
        train_df=train_data,
        val_df=val_data,
        text_column="text",
        label_column="label"
    )

    # Use optimized program
    response = service.label_text_with_dspy(
        text="Great product!",
        program_id="test_optimized"
    )

    assert response.label in ["positive", "negative"]
    assert 0 <= response.confidence <= 1
```

### Performance Tests

```python
# tests/performance/test_dspy_performance.py

@pytest.mark.benchmark
def test_optimization_cost(benchmark_dataset):
    """Verify optimization cost is within budget."""
    result = run_optimization(benchmark_dataset)

    assert result.total_cost < 10.0  # <$10 per optimization
    assert result.optimization_time < 1800  # <30 minutes

@pytest.mark.benchmark
def test_inference_latency():
    """Verify DSPy inference latency is acceptable."""
    program = load_optimized_program("test_program.json")

    latencies = []
    for text in test_texts:
        start = time.time()
        program(text=text, allowed_labels=["a", "b"])
        latencies.append(time.time() - start)

    p95_latency = np.percentile(latencies, 95)
    assert p95_latency < 2.0  # <2s p95
```

---

## Migration Path

### Phase 1: Parallel Deployment (Week 1)
- Deploy DSPy infrastructure alongside existing system
- No changes to existing workflows
- Internal testing only

### Phase 2: Opt-In Usage (Week 2)
- Enable DSPy via feature flag: `--use-dspy`
- Document usage patterns
- Collect user feedback

### Phase 3: A/B Testing (Week 3)
- Run A/B tests on production workloads
- Compare accuracy, latency, cost
- Build confidence in DSPy approach

### Phase 4: Gradual Rollout (Week 4-5)
- Default to DSPy for new projects
- Maintain backward compatibility
- Provide migration tools for existing projects

### Phase 5: Full Migration (Week 6+)
- DSPy becomes primary path
- Legacy path deprecated but available
- Complete documentation and training

---

## Dependencies

### Required Packages

```python
# phase2_dependencies.txt (DSPy section)

# DSPy Core
dspy-ai>=2.5.0

# Required by DSPy
openai>=1.0.0
anthropic>=0.15.0
backoff>=2.2.0

# Optimization
scipy>=1.10.0
numpy>=1.24.0

# Already in project
pydantic>=2.0.0
pandas>=2.0.0
```

### Version Compatibility

| Package | Minimum | Recommended | Notes |
|---------|---------|-------------|-------|
| dspy-ai | 2.5.0 | 2.5.x | Latest stable |
| openai | 1.0.0 | 1.x | For OpenAI models |
| scipy | 1.10.0 | 1.11.x | For A/B testing |
| Python | 3.10 | 3.11 | Type hints |

---

## Cost Analysis

### Optimization Costs

**Typical Run (20 trials):**
- Model: gpt-4o-mini
- Input tokens: ~500 per trial × 20 = 10,000 tokens
- Output tokens: ~200 per trial × 20 = 4,000 tokens
- Cost: ~$2-3 per optimization

**Heavy Run (40 trials):**
- Input tokens: 20,000
- Output tokens: 8,000
- Cost: ~$4-6 per optimization

**Best Practices:**
- Start with 20 trials, increase if needed
- Use validation set stopping criteria
- Cache intermediate results
- Reuse optimized programs across similar tasks

### Inference Costs

**No additional cost** - DSPy programs are just better prompts sent to the same LLMs.

---

## Success Metrics

### Optimization Quality
- **Target:** +20-50% accuracy improvement
- **Measurement:** Validation set accuracy
- **Acceptable:** +10% minimum

### Optimization Efficiency
- **Target:** <20 minutes optimization time
- **Target:** <$5 per optimization
- **Target:** 90% reproducibility (same seed)

### Integration Quality
- **Target:** Zero breaking changes to existing API
- **Target:** <10% latency overhead for DSPy path
- **Target:** 100% backward compatibility

### User Experience
- **Target:** <5 CLI commands to learn
- **Target:** Clear error messages and debugging
- **Target:** Complete documentation with examples

---

## Risks & Mitigation

### Risk 1: DSPy Dependency Instability
**Likelihood:** Medium
**Impact:** High
**Mitigation:**
- Pin to stable version (2.5.x)
- Extensive integration testing
- Fallback to baseline prompts

### Risk 2: Optimization Overfitting
**Likelihood:** Medium
**Impact:** Medium
**Mitigation:**
- Require minimum 200 training examples
- Use separate validation set
- Monitor train/val gap
- Implement early stopping

### Risk 3: Cost Overruns
**Likelihood:** Low
**Impact:** Medium
**Mitigation:**
- Set hard budget limits in code
- Track costs in real-time
- Provide cost estimates before optimization
- Allow user-defined trial limits

### Risk 4: Latency Regression
**Likelihood:** Low
**Impact:** Medium
**Mitigation:**
- Benchmark latency during testing
- Optimize DSPy module design
- Cache optimized programs
- Parallel inference where possible

---

## Future Enhancements

### V1.1: Advanced Strategies
- Multi-module pipelines (retrieval + reasoning)
- Custom optimization metrics
- Transfer learning across tasks

### V1.2: Multi-Model Optimization
- Optimize for multiple LLMs simultaneously
- Cost-aware model selection
- Cascade strategies (small → large models)

### V1.3: Continuous Learning
- Online optimization with new labeled data
- Feedback loops from human corrections
- Drift-aware re-optimization

---

## Appendix A: DSPy Resources

### Official Documentation
- **Main Site:** https://dspy.ai/
- **GitHub:** https://github.com/stanfordnlp/dspy
- **API Docs:** https://dspy.ai/api/
- **MIPROv2 Guide:** https://dspy.ai/deep-dive/optimizers/miprov2/

### Research Papers
- DSPy: Compiling Declarative Language Model Calls (ICLR 2024)
- MIPROv2: Multi-Prompt Instruction Proposal Optimizer

### Community Resources
- **Tutorials:** https://dspy.ai/tutorials/
- **Examples:** https://github.com/stanfordnlp/dspy/tree/main/examples
- **Discord:** https://discord.gg/VzS6RHHK6F

---

## Appendix B: Example Optimization Session

```python
# Complete end-to-end example

import pandas as pd
from autolabeler import Settings, LabelingService, DSPyConfig

# 1. Prepare data
train_df = pd.read_csv("sentiment_train.csv")  # 500 examples
val_df = pd.read_csv("sentiment_val.csv")      # 100 examples
test_df = pd.read_csv("sentiment_test.csv")    # 200 examples

# 2. Initialize service
settings = Settings()
service = LabelingService("sentiment", settings)

# 3. Configure optimization
config = DSPyConfig(
    model_name="gpt-4o-mini",
    num_trials=20,
    strategy="cot",
    use_rag=True,
    metric="f1",
    random_seed=42
)

# 4. Run optimization
print("Starting optimization...")
result = service.optimize_prompts(
    train_df=train_df,
    val_df=val_df,
    text_column="review_text",
    label_column="sentiment",
    config=config
)

# 5. View results
print(f"\nOptimization Results:")
print(f"  Baseline accuracy: {result.baseline_accuracy:.1%}")
print(f"  Optimized accuracy: {result.optimized_accuracy:.1%}")
print(f"  Improvement: +{result.relative_improvement:.1%}")
print(f"  Cost: ${result.total_cost:.2f}")
print(f"  Time: {result.optimization_time/60:.1f} minutes")

# 6. Evaluate on test set
optimizer = service.get_dspy_optimizer()
test_metrics = optimizer.evaluate_program(
    program=result.optimized_module,
    eval_df=test_df,
    text_column="review_text",
    label_column="sentiment"
)

print(f"\nTest Set Performance:")
print(f"  Accuracy: {test_metrics['accuracy']:.1%}")
print(f"  F1 Score: {test_metrics['f1']:.3f}")

# 7. Save optimized program
program_path = Path("config/sentiment_optimized.json")
optimizer.save_optimized_program(
    program=result.optimized_module,
    result=result,
    output_path=program_path
)

print(f"\n✅ Optimized program saved to: {program_path}")

# 8. Use optimized program for inference
test_review = "This product is amazing! Best purchase ever."
response = service.label_text_with_dspy(
    text=test_review,
    program_id="sentiment_optimized"
)

print(f"\nInference Example:")
print(f"  Text: {test_review}")
print(f"  Predicted: {response.label}")
print(f"  Confidence: {response.confidence:.2f}")
print(f"  Reasoning: {response.reasoning}")
```

**Expected Output:**
```
Starting optimization...
Optimization running: 20 trials...
[Progress bar: 100%]

Optimization Results:
  Baseline accuracy: 72.0%
  Optimized accuracy: 89.0%
  Improvement: +23.6%
  Cost: $2.34
  Time: 18.2 minutes

Test Set Performance:
  Accuracy: 87.5%
  F1 Score: 0.878

✅ Optimized program saved to: config/sentiment_optimized.json

Inference Example:
  Text: This product is amazing! Best purchase ever.
  Predicted: positive
  Confidence: 0.95
  Reasoning: The text contains strong positive sentiment indicators like "amazing" and "Best purchase ever", indicating clear satisfaction with the product.
```

---

**END OF SPECIFICATION**
