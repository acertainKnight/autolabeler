# AutoLabeler Enhancement - Quick Start Guide
## Get Started with Implementation in 30 Minutes

**Target Audience:** Developers ready to implement the enhancement strategy
**Time Required:** 30 minutes setup + ongoing development
**Prerequisites:** Python 3.10+, Git, basic familiarity with AutoLabeler

---

## TL;DR - Start Here

```bash
# 1. Set up environment
cd /home/nick/python/autolabeler
python -m venv venv-enhanced
source venv-enhanced/bin/activate  # On Windows: venv-enhanced\Scripts\activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Install Phase 1 dependencies
pip install instructor krippendorff plotly dash scikit-learn

# 4. Create feature branch
git checkout -b feature/phase1-quality-monitoring

# 5. Start with first task
# See "Week 1 Day 1-2" section below
```

---

## Phase 1 Implementation: Week-by-Week Breakdown

### Week 1: Quality Foundation

#### **Day 1-2: Structured Output Validation**

**Goal:** Implement Instructor-based validation with automatic retries

**Files to Create:**
```
src/autolabeler/core/validation/
├── __init__.py
├── output_validator.py
└── validation_strategies.py
```

**Step-by-Step:**

1. **Create the module structure:**
```bash
mkdir -p src/autolabeler/core/validation
touch src/autolabeler/core/validation/__init__.py
touch src/autolabeler/core/validation/output_validator.py
touch src/autolabeler/core/validation/validation_strategies.py
```

2. **Implement StructuredOutputValidator:**
```python
# src/autolabeler/core/validation/output_validator.py

from typing import Type, Callable, Any
from pydantic import BaseModel, ValidationError
import instructor
from openai import OpenAI
from loguru import logger

class StructuredOutputValidator:
    """Enhanced structured output validation using Instructor."""

    def __init__(self, client: OpenAI, max_retries: int = 3):
        self.client = instructor.from_openai(
            client,
            mode=instructor.Mode.FUNCTIONS
        )
        self.max_retries = max_retries

    def validate_and_retry(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        validation_rules: list[Callable] | None = None
    ) -> BaseModel:
        """Validate output with automatic retry."""
        for attempt in range(self.max_retries):
            try:
                # Attempt structured output generation
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    response_model=response_model,
                    messages=[{"role": "user", "content": prompt}]
                )

                # Run business rule validation
                if validation_rules:
                    for rule in validation_rules:
                        if not rule(response):
                            raise ValidationError("Business rule validation failed")

                return response

            except ValidationError as e:
                logger.warning(f"Validation failed (attempt {attempt+1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    # Construct error feedback for retry
                    error_feedback = f"\nPrevious attempt failed validation: {e}\nPlease correct and try again."
                    prompt = prompt + error_feedback
                else:
                    raise

        raise ValidationError("Max retries exceeded")
```

3. **Write tests:**
```python
# tests/unit/validation/test_output_validator.py

import pytest
from pydantic import BaseModel, Field
from autolabeler.core.validation import StructuredOutputValidator

class TestResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)

def test_successful_validation():
    """Test successful validation on first attempt."""
    # Implementation here
    pass

def test_retry_on_validation_failure():
    """Test retry logic when validation fails."""
    # Implementation here
    pass

def test_max_retries_exceeded():
    """Test that max retries is respected."""
    # Implementation here
    pass
```

4. **Run tests:**
```bash
pytest tests/unit/validation/ -v
```

5. **Integration with LabelingService:**
```python
# Modify src/autolabeler/core/labeling/labeling_service.py

from ..validation import StructuredOutputValidator

class LabelingService:
    def __init__(self, ...):
        # ... existing initialization ...

        # Add validator
        self.validator = StructuredOutputValidator(
            self._get_client_for_config(self.config),
            max_retries=3
        )

    def label_text(self, text: str, ...) -> LabelResponse:
        # ... existing code ...

        # Use validator instead of direct with_structured_output
        response = self.validator.validate_and_retry(
            rendered_prompt,
            LabelResponse
        )

        return response
```

**Checkpoint:** Run full test suite to ensure no regressions:
```bash
pytest tests/ -v
```

#### **Day 3-4: Confidence Calibration**

**Goal:** Implement temperature scaling and Platt scaling calibration

**Files to Create:**
```
src/autolabeler/core/quality/
├── __init__.py
├── confidence_calibrator.py
└── calibration_metrics.py
```

**Step-by-Step:**

1. **Create module structure:**
```bash
mkdir -p src/autolabeler/core/quality
touch src/autolabeler/core/quality/__init__.py
touch src/autolabeler/core/quality/confidence_calibrator.py
touch src/autolabeler/core/quality/calibration_metrics.py
```

2. **Implement ConfidenceCalibrator:**
```python
# src/autolabeler/core/quality/confidence_calibrator.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from typing import Literal
import pickle

CalibrationMethod = Literal["temperature_scaling", "platt_scaling", "isotonic_regression"]

class ConfidenceCalibrator:
    """Confidence score calibration."""

    def __init__(self, method: CalibrationMethod = "temperature_scaling"):
        self.method = method
        self.calibrator = None

    def fit(
        self,
        confidence_scores: np.ndarray,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray
    ) -> None:
        """Fit calibration model."""
        if self.method == "temperature_scaling":
            self._fit_temperature_scaling(confidence_scores, true_labels)
        elif self.method == "platt_scaling":
            self._fit_platt_scaling(confidence_scores, true_labels)
        else:  # isotonic_regression
            self._fit_isotonic_regression(confidence_scores, true_labels)

    def _fit_temperature_scaling(self, confidences, true_labels):
        """Fit temperature scaling (single parameter)."""
        # Implementation: Find temperature T that minimizes NLL
        from scipy.optimize import minimize

        def nll(temperature):
            scaled = np.clip(confidences / temperature, 0, 1)
            correct = (scaled == true_labels).astype(float)
            return -np.sum(correct * np.log(scaled + 1e-10))

        result = minimize(nll, x0=1.0, bounds=[(0.01, 10.0)])
        self.calibrator = {"temperature": result.x[0]}

    def calibrate(self, confidence_scores: np.ndarray) -> np.ndarray:
        """Apply calibration."""
        if self.calibrator is None:
            raise RuntimeError("Calibrator not fitted")

        if self.method == "temperature_scaling":
            temperature = self.calibrator["temperature"]
            return np.clip(confidence_scores / temperature, 0, 1)
        elif self.method == "platt_scaling":
            return self.calibrator.predict_proba(confidence_scores.reshape(-1, 1))[:, 1]
        else:  # isotonic
            return self.calibrator.transform(confidence_scores)
```

3. **Implement calibration metrics:**
```python
# src/autolabeler/core/quality/calibration_metrics.py

import numpy as np

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
            bin_weight = mask.sum() / len(true_labels)
            ece += bin_weight * abs(bin_confidence - bin_accuracy)

    return ece

def compute_brier_score(confidence_scores: np.ndarray, true_labels: np.ndarray) -> float:
    """Compute Brier score."""
    return np.mean((confidence_scores - true_labels) ** 2)
```

4. **Write tests:**
```bash
# tests/unit/quality/test_confidence_calibrator.py
# See TESTING_STRATEGY.md for complete test implementation
```

5. **Integration with LabelingService:**
```python
# Modify LabelingService to optionally use calibrator

class LabelingService:
    def __init__(self, ...):
        # ... existing initialization ...

        # Add calibrator (optional)
        self.calibrator = None
        if config.enable_confidence_calibration:
            self.calibrator = ConfidenceCalibrator(method="temperature_scaling")

    def label_text(self, text: str, ...) -> LabelResponse:
        # ... existing code to get response ...

        # Calibrate confidence if calibrator available
        if self.calibrator:
            response.confidence = self.calibrator.calibrate(
                np.array([response.confidence])
            )[0]

        return response

    def calibrate_on_validation_set(
        self,
        val_df: pd.DataFrame,
        text_column: str,
        label_column: str
    ) -> dict[str, float]:
        """Fit calibrator on validation set."""
        # Label validation set
        results = self.label_dataframe(val_df, text_column)

        # Fit calibrator
        confidences = results["predicted_label_confidence"].values
        true_labels = (results["predicted_label"] == results[label_column]).astype(int).values
        pred_labels = results["predicted_label"].values

        self.calibrator.fit(confidences, true_labels, pred_labels)

        # Evaluate calibration
        ece_before = compute_ece(confidences, true_labels)
        calibrated = self.calibrator.calibrate(confidences)
        ece_after = compute_ece(calibrated, true_labels)

        return {
            "ece_before": ece_before,
            "ece_after": ece_after,
            "improvement": ece_before - ece_after
        }
```

**Checkpoint:** Test calibration:
```bash
pytest tests/unit/quality/ -v
```

#### **Day 5: Quality Monitoring Dashboard**

**Goal:** Implement Krippendorff's alpha and basic quality dashboard

**Files to Create:**
```
src/autolabeler/core/quality/quality_monitor.py
```

**Step-by-Step:**

1. **Implement QualityMonitor:**
```python
# src/autolabeler/core/quality/quality_monitor.py

import krippendorff
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class QualityMonitor:
    """Comprehensive quality monitoring."""

    def calculate_krippendorff_alpha(
        self,
        df: pd.DataFrame,
        annotator_columns: list[str],
        level_of_measurement: str = "nominal"
    ) -> float:
        """Calculate Krippendorff's alpha."""
        # Prepare data for krippendorff library
        # Format: rows = annotators, columns = items
        reliability_data = df[annotator_columns].T.values

        alpha = krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement=level_of_measurement
        )

        return alpha

    def generate_dashboard(
        self,
        df: pd.DataFrame,
        output_path: Path
    ) -> Path:
        """Generate HTML dashboard."""
        # Create plotly figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Confidence Distribution",
                "Label Distribution",
                "Agreement Over Time",
                "Cost Analysis"
            )
        )

        # Add plots
        # ... implementation ...

        # Save to HTML
        fig.write_html(output_path)
        return output_path
```

2. **Create CLI command:**
```python
# Add to cli.py

@cli.command()
@click.option("--dataset-name", required=True)
@click.option("--results-file", required=True, type=click.Path(path_type=Path))
@click.option("--output-file", required=True, type=click.Path(path_type=Path))
def monitor_quality(dataset_name: str, results_file: Path, output_file: Path):
    """Generate quality monitoring dashboard."""
    from autolabeler.core.quality import QualityMonitor

    df = pd.read_csv(results_file)
    monitor = QualityMonitor()

    dashboard_path = monitor.generate_dashboard(df, output_file)
    logger.info(f"Dashboard generated: {dashboard_path}")
```

3. **Test:**
```bash
# Create sample data and test dashboard generation
python -m autolabeler.cli monitor-quality \
    --dataset-name test \
    --results-file test_results.csv \
    --output-file dashboard.html
```

**End of Week 1:** You now have:
- ✅ Structured output validation working
- ✅ Confidence calibration implemented
- ✅ Basic quality monitoring in place

---

### Week 2: Cost Tracking & Integration

#### **Day 1-3: Cost Tracking System**

**Goal:** Track LLM API costs and compute ROI metrics

**Files to Create:**
```
src/autolabeler/core/monitoring/
├── __init__.py
├── cost_tracker.py
└── pricing.py
```

**Implementation:**
```python
# src/autolabeler/core/monitoring/cost_tracker.py

class CostTracker:
    """Track LLM API costs and compute ROI."""

    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "claude-3": {"input": 0.015, "output": 0.075},
    }

    def __init__(self, budget_limit: float | None = None):
        self.budget_limit = budget_limit
        self.cost_history: list[dict] = []
        self.total_cost = 0.0

    def track_llm_call(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        success: bool
    ) -> float:
        """Track individual LLM call cost."""
        pricing = self.PRICING.get(model_name, {"input": 0.01, "output": 0.01})

        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost

        self.cost_history.append({
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": total_cost,
            "success": success,
            "timestamp": pd.Timestamp.now()
        })

        self.total_cost += total_cost

        # Check budget
        if self.budget_limit and self.total_cost > self.budget_limit:
            logger.warning(f"Budget limit exceeded: ${self.total_cost:.2f} > ${self.budget_limit:.2f}")

        return total_cost
```

#### **Day 4-5: Integration & Testing**

**Goal:** Integrate all Phase 1 components and run comprehensive tests

**Tasks:**
1. Integration testing across all Phase 1 components
2. Performance testing
3. Documentation updates
4. Demo preparation

**Checklist:**
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Performance tests within SLA
- [ ] Documentation updated
- [ ] Example notebooks created
- [ ] CLI commands working
- [ ] Demo prepared

**End of Week 2:** Phase 1 Complete! 🎉

---

## Useful Commands

### Development

```bash
# Run tests
pytest tests/unit/ -v                    # Unit tests only
pytest tests/integration/ -v             # Integration tests
pytest tests/performance/ -v --benchmark # Performance tests
pytest tests/ -v --cov=src/autolabeler  # All tests with coverage

# Code quality
black src/ tests/                        # Format code
ruff check src/ tests/                   # Lint code
mypy src/                                # Type checking

# Run specific test
pytest tests/unit/quality/test_confidence_calibrator.py::test_temperature_scaling -v
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/phase1-quality-monitoring

# Commit changes
git add .
git commit -m "feat: implement confidence calibration"

# Push to remote
git push origin feature/phase1-quality-monitoring

# Create pull request (via GitHub)
```

### CLI Testing

```bash
# Label with confidence calibration
autolabeler label \
    --dataset-name test \
    --input-file test_data.csv \
    --text-column text \
    --train-file labeled_data.csv \
    --label-column label \
    --output-file results.csv

# Generate quality dashboard
autolabeler monitor-quality \
    --dataset-name test \
    --results-file results.csv \
    --output-file dashboard.html
```

---

## Troubleshooting

### Common Issues

**Issue: Import errors after adding new modules**
```bash
# Solution: Reinstall in editable mode
pip install -e .
```

**Issue: Tests failing due to missing dependencies**
```bash
# Solution: Install dev dependencies
pip install -e ".[dev]"
pip install instructor krippendorff plotly dash
```

**Issue: Type checking errors**
```bash
# Solution: Install type stubs
pip install types-requests types-setuptools
```

**Issue: Slow tests**
```bash
# Solution: Run specific test categories
pytest tests/unit/ -v           # Fast unit tests only
pytest -m "not slow" -v         # Skip slow tests
```

---

## Phase 2 Preview

**Coming in Week 3-7:**

1. **DSPy Integration** - Systematic prompt optimization
2. **Advanced RAG** - GraphRAG, RAPTOR variants
3. **Active Learning** - TCM hybrid strategy
4. **Weak Supervision** - Snorkel + FlyingSquid
5. **Data Versioning** - DVC integration

**Preparation:**
```bash
# Install Phase 2 dependencies
pip install dspy-ai flyingsquid modAL dvc scikit-learn networkx
```

---

## Resources

### Documentation

- **Main Roadmap:** `.hive-mind/IMPLEMENTATION_ROADMAP.md`
- **Testing Strategy:** `.hive-mind/TESTING_STRATEGY.md`
- **API Specs:** `.hive-mind/API_SPECIFICATIONS.md`
- **Research Review:** `advanced-labeling.md`

### External Resources

- **Instructor:** https://python.useinstructor.com/
- **Krippendorff:** https://github.com/pln-fing-udelar/fast-krippendorff
- **Plotly:** https://plotly.com/python/
- **Pytest:** https://docs.pytest.org/

### Getting Help

- **Issues:** Create GitHub issue with `[enhancement]` tag
- **Questions:** Add comment to implementation roadmap
- **Slack:** #autolabeler-dev channel

---

## Success Checklist

### Phase 1 Complete When:

- [ ] Structured output validation reduces parsing failures to <1%
- [ ] Confidence calibration improves ECE by >50%
- [ ] Quality dashboard accessible and updating
- [ ] Cost tracking within 5% accuracy
- [ ] All tests passing with >75% coverage
- [ ] Documentation updated
- [ ] Demo successful

### Ready for Phase 2 When:

- [ ] Phase 1 merged to main branch
- [ ] No critical bugs in Phase 1 features
- [ ] Performance SLAs met
- [ ] Code reviewed and approved
- [ ] Release notes prepared

---

## Quick Reference

### Key Files

```
src/autolabeler/
├── core/
│   ├── validation/output_validator.py      # Structured output validation
│   ├── quality/
│   │   ├── confidence_calibrator.py        # Confidence calibration
│   │   └── quality_monitor.py              # Quality monitoring
│   └── monitoring/cost_tracker.py          # Cost tracking
```

### Key Commands

```bash
# Development cycle
black src/ && ruff check src/ && mypy src/ && pytest tests/unit/ -v

# Quick test
pytest tests/unit/quality/ -v -x  # Stop on first failure

# Coverage report
pytest tests/ --cov=src/autolabeler --cov-report=html
open htmlcov/index.html  # View coverage report
```

---

**Ready to start? Begin with Day 1-2: Structured Output Validation!**

Good luck! 🚀

---

**Document Control:**
- **Author:** TESTER/INTEGRATION AGENT
- **Version:** 1.0
- **Last Updated:** 2025-10-07
- **Next Review:** End of Week 1
