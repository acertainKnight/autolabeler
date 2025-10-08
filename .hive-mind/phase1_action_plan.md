# Phase 1 Action Plan - Quick Start Guide

**Status:** ✅ Research Complete - Ready for Implementation
**Timeline:** 2 weeks (10 working days)
**Risk Level:** LOW
**Dependencies:** All compatible with Python 3.12.8

---

## TL;DR - What to Do Next

1. **Install 5 new packages** (instructor, scipy, krippendorff, streamlit, plotly)
2. **Create quality module** for confidence calibration and agreement metrics
3. **Integrate instructor** for structured output validation
4. **Build Streamlit dashboard** for quality monitoring
5. **Add cost tracking** across all LLM calls

**Expected Benefits:**
- 90%+ reduction in validation failures (instructor)
- 20%+ improvement in confidence calibration (ECE)
- Real-time quality monitoring (dashboard)
- 100% cost visibility (tracking)
- Inter-rater reliability metrics (Krippendorff's alpha)

---

## Day-by-Day Implementation Plan

### Days 1-2: Foundation Setup

**Morning: Install Dependencies**
```bash
cd /home/nick/python/autolabeler

# Update pyproject.toml
# Add these to dependencies section:
#   'instructor>=1.11.3',
#   'scipy>=1.15.2,<1.16.0',
#   'krippendorff>=0.8.1',
#   'streamlit>=1.44.0',
#   'plotly>=6.3.1',

# Install
pip install instructor==1.11.3 scipy==1.15.2 krippendorff==0.8.1 streamlit==1.44.0 plotly==6.3.1

# Verify
python -c "import instructor, scipy, krippendorff, streamlit, plotly; print('All imports successful')"
```

**Afternoon: Create Directory Structure**
```bash
cd /home/nick/python/autolabeler/src/autolabeler/core

# Create new directories
mkdir -p quality/{__init__.py,confidence_calibrator.py,agreement_calculator.py,cost_tracker.py,quality_metrics.py}
mkdir -p dashboard/{__init__.py,components,utils}
mkdir -p validation/{__init__.py,structured_validator.py,retry_handler.py}

# Touch files to create them
touch quality/__init__.py
touch quality/confidence_calibrator.py
touch quality/agreement_calculator.py
touch quality/cost_tracker.py
touch quality/quality_metrics.py

touch dashboard/__init__.py
touch dashboard/app.py
touch dashboard/components/__init__.py
touch dashboard/utils/__init__.py

touch validation/structured_validator.py
touch validation/retry_handler.py
```

### Days 3-4: Confidence Calibration

**Implement:** `src/autolabeler/core/quality/confidence_calibrator.py`

**Key Functions:**
1. `TemperatureScaling.calibrate()` - Post-hoc temperature scaling
2. `PlattScaling.calibrate()` - Logistic regression calibration
3. `calculate_ece()` - Expected Calibration Error
4. `calculate_mce()` - Maximum Calibration Error
5. `create_calibration_curve()` - Reliability diagram data

**Integration Point:** `LabelingService.label_text()` - apply after LLM response

**Test:** Create calibration set from 500+ labeled examples, verify ECE improves

### Days 5-6: Agreement Metrics & Cost Tracking

**Implement:** `src/autolabeler/core/quality/agreement_calculator.py`

**Key Functions:**
1. `calculate_krippendorff_alpha()` - Inter-rater reliability
2. `calculate_pairwise_agreement()` - Cohen's kappa for pairs
3. `create_agreement_matrix()` - Heatmap data
4. `interpret_agreement()` - Threshold-based interpretation

**Implement:** `src/autolabeler/core/quality/cost_tracker.py`

**Key Functions:**
1. `track_api_call()` - Log tokens and cost per call
2. `get_cost_summary()` - Aggregate costs by model/time
3. `check_budget_alert()` - Alert when threshold exceeded
4. `export_cost_report()` - CSV/JSON export

**Integration Points:**
- `EnsembleService._consolidate_predictions()` - add Krippendorff's alpha
- `LabelingService.label_text()` - add cost tracking wrapper

### Days 7-8: Instructor Integration

**Implement:** `src/autolabeler/core/validation/structured_validator.py`

**Key Changes:**
```python
# Replace in LabelingService
from instructor import from_openai
from openai import OpenAI

class StructuredValidator:
    def __init__(self, model_name: str):
        self.client = from_openai(OpenAI())
        self.model_name = model_name

    def validate_response(
        self,
        prompt: str,
        response_model: type[BaseModel],
        max_retries: int = 3
    ) -> BaseModel:
        """Validate and retry with instructor."""
        return self.client.chat.completions.create(
            model=self.model_name,
            response_model=response_model,
            max_retries=max_retries,
            messages=[{"role": "user", "content": prompt}]
        )
```

**Testing:** Run existing test suite, verify no regressions

### Days 9-10: Streamlit Dashboard

**Implement:** `src/autolabeler/core/dashboard/app.py`

**Required Pages:**
1. **Overview** - Key metrics, cost summary, agreement scores
2. **Confidence Analysis** - Distribution, calibration curves, ECE/MCE
3. **Agreement Metrics** - Krippendorff's alpha, heatmaps, trends
4. **Cost Tracking** - Per-model costs, budget alerts, trends
5. **Model Comparison** - Side-by-side performance

**Example Structure:**
```python
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AutoLabeler Quality Dashboard", layout="wide")

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", [
    "Overview",
    "Confidence Analysis",
    "Agreement Metrics",
    "Cost Tracking",
    "Model Comparison"
])

if page == "Overview":
    st.title("AutoLabeler Quality Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", "10,245")
    col2.metric("Avg Confidence", "0.87", delta="0.03")
    col3.metric("Krippendorff's α", "0.73", delta="0.05")
    col4.metric("Total Cost", "$245.67", delta="-$12.34")

    # ... more visualizations
```

**Launch:** `streamlit run src/autolabeler/core/dashboard/app.py --server.port 8501`

### Days 11-12: Testing & Documentation

**Unit Tests:**
```bash
# Create test files
tests/core/quality/test_confidence_calibrator.py
tests/core/quality/test_agreement_calculator.py
tests/core/quality/test_cost_tracker.py
tests/core/validation/test_structured_validator.py
```

**Integration Tests:**
```bash
tests/integration/test_labeling_with_calibration.py
tests/integration/test_ensemble_with_agreement.py
tests/integration/test_cost_tracking_e2e.py
```

**Documentation:**
- Update README.md with Phase 1 features
- Create dashboard user guide
- Add configuration examples
- Document migration from old validation

### Days 13-14: Polish & Review

**Performance Testing:**
- Benchmark labeling service before/after
- Optimize any bottlenecks (target <10% overhead)
- Test dashboard with large datasets (10k+ predictions)

**Code Review:**
- Run ruff/black linters
- Review all integration points
- Verify backward compatibility
- Test rollback procedures

**Documentation Review:**
- API documentation complete?
- User guide clear?
- Configuration examples work?
- Migration guide tested?

---

## Critical Integration Points

### 1. LabelingService Enhancement

**File:** `/home/nick/python/autolabeler/src/autolabeler/core/labeling/labeling_service.py`

**Changes Required:**

**Line 365-370** (Replace structured output):
```python
# OLD
structured_llm = llm_client.with_structured_output(LabelResponse, method="function_calling")
response = structured_llm.invoke(rendered_prompt)

# NEW
from ..validation.structured_validator import StructuredValidator

validator = StructuredValidator(config.model_name)
response = validator.validate_response(
    prompt=rendered_prompt,
    response_model=LabelResponse,
    max_retries=3
)

# Add cost tracking
self.cost_tracker.track_api_call(
    model=config.model_name,
    tokens=response.usage.total_tokens,  # From OpenAI response
    timestamp=datetime.now()
)

# Add confidence calibration
response.confidence = self.confidence_calibrator.calibrate(
    raw_confidence=response.confidence,
    model_name=config.model_name
)
```

### 2. EnsembleService Enhancement

**File:** `/home/nick/python/autolabeler/src/autolabeler/core/ensemble/ensemble_service.py`

**Changes Required:**

**Line 235** (Add Krippendorff's alpha):
```python
# Add to _consolidate_predictions method
from ..quality.agreement_calculator import AgreementCalculator

agreement_calc = AgreementCalculator()
krippendorff_alpha = agreement_calc.calculate_krippendorff_alpha(
    predictions=individual_predictions
)

result.model_agreement = krippendorff_alpha
result.agreement_interpretation = agreement_calc.interpret_agreement(krippendorff_alpha)
```

### 3. EvaluationService Enhancement

**File:** `/home/nick/python/autolabeler/src/autolabeler/core/evaluation/evaluation_service.py`

**Changes Required:**

**Line 88-91** (Add ECE calculation):
```python
from ..quality.quality_metrics import calculate_ece

if confidence_column and confidence_column in valid_df.columns:
    results["confidence_analysis"] = evaluation_utils.analyze_confidence(
        y_true, y_pred, valid_df[confidence_column]
    )

    # Add ECE
    results["expected_calibration_error"] = calculate_ece(
        confidences=valid_df[confidence_column].values,
        predictions=y_pred.values,
        true_labels=y_true.values,
        n_bins=10
    )
```

---

## Configuration Updates

### Add to `/home/nick/python/autolabeler/src/autolabeler/core/configs.py`

```python
from pydantic import BaseModel, Field

class ConfidenceCalibrationConfig(BaseModel):
    """Configuration for confidence calibration."""
    enabled: bool = Field(True, description="Enable confidence calibration")
    method: str = Field("temperature_scaling", description="Calibration method (temperature_scaling, platt_scaling)")
    calibration_set_size: int = Field(500, description="Minimum samples for calibration")
    temperature: float = Field(1.0, description="Initial temperature (optimized during calibration)")


class QualityMonitoringConfig(BaseModel):
    """Configuration for quality monitoring."""
    enable_dashboard: bool = Field(True, description="Enable Streamlit dashboard")
    dashboard_port: int = Field(8501, description="Port for dashboard")
    calculate_krippendorff: bool = Field(True, description="Calculate Krippendorff's alpha")
    agreement_threshold: float = Field(0.67, description="Minimum acceptable agreement")
    ece_bins: int = Field(10, description="Number of bins for ECE calculation")


class CostTrackingConfig(BaseModel):
    """Configuration for cost tracking."""
    enabled: bool = Field(True, description="Enable cost tracking")
    track_tokens: bool = Field(True, description="Track token usage")
    cost_per_1k_tokens: dict[str, float] = Field(
        default_factory=lambda: {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-4o": 0.005,
            "gpt-4o-mini": 0.00015,
            "gpt-3.5-turbo": 0.0015,
            "claude-3-opus": 0.015,
            "claude-3-sonnet": 0.003,
            "claude-3-haiku": 0.00025,
        },
        description="Cost per 1K tokens by model"
    )
    budget_alert_threshold: float | None = Field(None, description="Alert when budget exceeded (USD)")
    export_path: str = Field("./results/cost_reports", description="Path for cost report exports")


class ValidationConfig(BaseModel):
    """Configuration for structured output validation."""
    enabled: bool = Field(True, description="Enable instructor validation")
    max_retries: int = Field(3, description="Maximum validation retry attempts")
    retry_delay: float = Field(1.0, description="Delay between retries (seconds)")
    fallback_to_manual: bool = Field(True, description="Fallback to manual parsing if validation fails")
```

### Update Settings

**File:** `/home/nick/python/autolabeler/src/autolabeler/config.py`

```python
from pydantic_settings import BaseSettings
from .core.configs import (
    ConfidenceCalibrationConfig,
    QualityMonitoringConfig,
    CostTrackingConfig,
    ValidationConfig
)

class Settings(BaseSettings):
    # ... existing settings ...

    # Phase 1 additions
    confidence_calibration: ConfidenceCalibrationConfig = ConfidenceCalibrationConfig()
    quality_monitoring: QualityMonitoringConfig = QualityMonitoringConfig()
    cost_tracking: CostTrackingConfig = CostTrackingConfig()
    validation: ValidationConfig = ValidationConfig()
```

---

## Testing Checklist

### Unit Tests
- [ ] ConfidenceCalibrator.calibrate() with temperature scaling
- [ ] ConfidenceCalibrator.calibrate() with Platt scaling
- [ ] calculate_ece() with known calibration data
- [ ] calculate_krippendorff_alpha() with test matrices
- [ ] CostTracker.track_api_call() and aggregation
- [ ] StructuredValidator with valid/invalid inputs
- [ ] Retry logic with max_retries

### Integration Tests
- [ ] LabelingService with instructor validation
- [ ] LabelingService with confidence calibration
- [ ] LabelingService with cost tracking
- [ ] EnsembleService with Krippendorff's alpha
- [ ] EvaluationService with ECE calculation
- [ ] Dashboard loads real data from services
- [ ] All visualizations render correctly

### Performance Tests
- [ ] Benchmark labeling_service.label_text() before/after
- [ ] Verify <10% performance overhead
- [ ] Dashboard handles 10,000+ predictions
- [ ] Cost tracking doesn't slow down labeling
- [ ] Memory usage within acceptable limits

### User Acceptance Tests
- [ ] Dashboard accessible at localhost:8501
- [ ] All metrics display correctly
- [ ] Export functionality works
- [ ] Configuration changes take effect
- [ ] Error messages are clear and actionable

---

## Success Criteria

### Technical Metrics
- ✅ Validation success rate >98% (instructor)
- ✅ ECE reduction >20% after calibration
- ✅ Krippendorff's alpha calculation <100ms
- ✅ Cost tracking covers 100% of API calls
- ✅ Dashboard load time <2 seconds

### Quality Metrics
- ✅ Inter-rater agreement (Krippendorff's α) >0.67
- ✅ Expected Calibration Error (ECE) <0.05
- ✅ Validation retry success rate >90%
- ✅ Cost prediction accuracy >95%

### User Experience Metrics
- ✅ Dashboard uptime >99%
- ✅ Visualization rendering <1 second
- ✅ Real-time metric updates <5 seconds
- ✅ Export functionality works for all formats

---

## Rollback Plan

If issues arise during Phase 1 implementation:

### Immediate Rollback (Emergency)
```bash
# Revert to pre-Phase 1 commit
git checkout <pre-phase1-commit-hash>
pip install -r requirements-old.txt
```

### Feature-Level Rollback
```python
# In config.py or settings
confidence_calibration.enabled = False
quality_monitoring.enable_dashboard = False
cost_tracking.enabled = False
validation.enabled = False
```

### Gradual Rollback
1. Disable dashboard (least critical)
2. Disable cost tracking (monitoring only)
3. Disable confidence calibration (impacts accuracy)
4. Disable instructor validation (last resort)

---

## Support and Resources

### Documentation
- Full research report: `/home/nick/python/autolabeler/.hive-mind/phase1_research_report.md`
- This action plan: `/home/nick/python/autolabeler/.hive-mind/phase1_action_plan.md`

### Package Documentation
- **instructor:** https://python.useinstructor.com/
- **scipy:** https://docs.scipy.org/doc/scipy/
- **krippendorff:** https://pypi.org/project/krippendorff/
- **streamlit:** https://docs.streamlit.io/
- **plotly:** https://plotly.com/python/

### Research References
- Advanced labeling research: `/home/nick/python/autolabeler/advanced-labeling.md`
- Confidence calibration: NAACL 2024 survey (Geng et al.)
- Krippendorff's alpha: Original papers + production guidelines
- Structured output: instructor package examples

---

## Next Steps

1. **Review this action plan** with team/stakeholders
2. **Create git branch** for Phase 1 work: `git checkout -b phase1-quick-wins`
3. **Start Day 1 tasks** - install dependencies and create directory structure
4. **Daily check-ins** to track progress and address blockers
5. **Week 2 milestone** - dashboard functional, metrics tracking
6. **Final review** before merging to main

**Questions or blockers?** Document in `.hive-mind/phase1_blockers.md`

---

**Action Plan Status:** ✅ Ready for Implementation
**Created:** October 7, 2025
**By:** RESEARCHER Agent (Hive Mind Swarm)
