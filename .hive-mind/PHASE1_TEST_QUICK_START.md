# Phase 1 Testing Quick Start Guide
## Get Testing in 5 Minutes

**Version:** 1.0
**Date:** 2025-10-07
**Status:** Ready to Use

---

## Quick Start

### 1. Install Dependencies

```bash
cd /home/nick/python/autolabeler
pip install -e ".[dev]"
```

This installs:
- pytest, pytest-cov, pytest-benchmark
- scipy (for calibration tests)
- scikit-learn (for active learning tests)
- All other dev dependencies

### 2. Run Your First Test

```bash
# Run the quick test suite (unit tests only, ~30 seconds)
./scripts/run_phase1_tests.sh quick
```

### 3. View Coverage

```bash
# Run with coverage report
./scripts/run_phase1_tests.sh coverage

# Open HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

---

## Common Commands

### Running Tests

```bash
# Quick unit tests (fastest)
pytest tests/test_unit/ -m unit

# All tests
pytest tests/

# Specific test file
pytest tests/test_unit/test_confidence_calibrator.py -v

# Specific test
pytest tests/test_unit/test_confidence_calibrator.py::TestConfidenceCalibrator::test_fit_temperature_scaling

# With coverage
pytest tests/ --cov=src/autolabeler --cov-report=term-missing

# Performance benchmarks
pytest tests/test_performance/ --benchmark-only
```

### Using the Test Script

```bash
# Show all options
./scripts/run_phase1_tests.sh help

# Quick tests (fail fast)
./scripts/run_phase1_tests.sh quick

# Unit tests only
./scripts/run_phase1_tests.sh unit

# Integration tests
./scripts/run_phase1_tests.sh integration

# Performance tests
./scripts/run_phase1_tests.sh performance

# Validation tests
./scripts/run_phase1_tests.sh validation

# Full suite
./scripts/run_phase1_tests.sh all

# CI mode (what runs in GitHub Actions)
./scripts/run_phase1_tests.sh ci

# Coverage report
./scripts/run_phase1_tests.sh coverage
```

---

## Test Structure At-a-Glance

```
tests/
├── conftest.py                          # Fixtures & config
├── test_unit/                           # Fast tests (~30s)
│   ├── test_confidence_calibrator.py    # Calibration tests
│   ├── test_quality_monitor.py          # Monitoring tests
│   └── test_structured_output_validator.py  # Validation tests
├── test_integration/                    # Slower tests (~2min)
│   └── test_phase1_integration.py       # End-to-end tests
├── test_performance/                    # Benchmarks (~5min)
│   └── test_phase1_performance.py       # SLA validation
└── test_validation/                     # Acceptance (~10min)
    └── test_phase1_success_criteria.py  # Success criteria
```

---

## Writing Your First Test

### 1. Create Test File

```python
# tests/test_unit/test_my_component.py
import pytest

@pytest.mark.unit
class TestMyComponent:
    """Test suite for MyComponent."""

    def test_basic_functionality(self, settings):
        """Test basic functionality."""
        # Arrange
        component = MyComponent(settings)

        # Act
        result = component.process("input")

        # Assert
        assert result is not None
        assert result.value > 0
```

### 2. Use Fixtures

```python
def test_with_mock_data(self, sample_labeled_df):
    """Test with mock labeled data."""
    # sample_labeled_df is provided by conftest.py
    assert len(sample_labeled_df) == 10
    assert "text" in sample_labeled_df.columns
```

### 3. Run Your Test

```bash
pytest tests/test_unit/test_my_component.py -v
```

---

## Available Fixtures

Use these fixtures in your tests (they're in `conftest.py`):

```python
def test_example(
    settings,                    # Test settings
    sample_labeled_df,           # Labeled DataFrame (10 samples)
    sample_unlabeled_df,         # Unlabeled DataFrame (10 samples)
    sample_calibration_data,     # Calibration data (1000 samples)
    sample_multi_annotator_data, # Multi-annotator data (5 items)
    sample_cost_data,            # Cost tracking data (100 samples)
    mock_llm_client,             # Mock LLM (no API calls)
    temp_storage_path,           # Temporary directory
):
    # Your test code here
    pass
```

---

## Test Markers

Mark your tests for selective execution:

```python
@pytest.mark.unit           # Fast unit test
@pytest.mark.integration    # Integration test
@pytest.mark.performance    # Performance/benchmark test
@pytest.mark.validation     # Validation/acceptance test
@pytest.mark.slow           # Slow test (>10 seconds)
@pytest.mark.requires_api   # Requires external API
```

Run by marker:
```bash
pytest tests/ -m unit        # Run only unit tests
pytest tests/ -m "not slow"  # Skip slow tests
```

---

## Debugging Tests

### Run with Verbose Output

```bash
pytest tests/test_unit/test_confidence_calibrator.py -vv
```

### Show Print Statements

```bash
pytest tests/test_unit/test_confidence_calibrator.py -s
```

### Stop on First Failure

```bash
pytest tests/test_unit/ -x
```

### Drop into Debugger on Failure

```bash
pytest tests/test_unit/test_confidence_calibrator.py --pdb
```

### Run Last Failed Tests

```bash
pytest tests/ --lf
```

---

## Coverage Tips

### Check Coverage for Specific File

```bash
pytest tests/test_unit/test_confidence_calibrator.py \
    --cov=src/autolabeler/core/quality/confidence_calibrator.py \
    --cov-report=term-missing
```

### Generate HTML Coverage Report

```bash
pytest tests/ --cov=src/autolabeler --cov-report=html
open htmlcov/index.html
```

### Fail if Coverage Below Threshold

```bash
pytest tests/ --cov=src/autolabeler --cov-fail-under=75
```

---

## Performance Testing

### Run Benchmarks

```bash
pytest tests/test_performance/ --benchmark-only
```

### Compare Benchmarks

```bash
# Run baseline
pytest tests/test_performance/ --benchmark-save=baseline

# Make changes, then compare
pytest tests/test_performance/ --benchmark-compare=baseline
```

### Benchmark Specific Test

```bash
pytest tests/test_performance/test_phase1_performance.py::TestPhase1PerformanceSLA::test_confidence_calibration_latency --benchmark-only
```

---

## CI/CD Integration

### What Runs in GitHub Actions

On every push/PR:
1. Unit tests (Python 3.10, 3.11, 3.12)
2. Integration tests
3. Code quality (Black, Ruff, codespell)

On PRs:
4. Performance benchmarks (with PR comment)

On main branch:
5. Full test suite with coverage

### Simulate CI Locally

```bash
./scripts/run_phase1_tests.sh ci
```

---

## Troubleshooting

### Tests Not Found

```bash
# Ensure pytest can find tests
pytest --collect-only tests/
```

### Import Errors

```bash
# Reinstall in editable mode
pip install -e ".[dev]"
```

### Fixture Not Found

```bash
# Check if fixture is in conftest.py
grep -r "def fixture_name" tests/conftest.py
```

### Coverage Not Working

```bash
# Check coverage config
cat pyproject.toml | grep -A 10 "tool.coverage"

# Verify source path
pytest tests/ --cov=src/autolabeler --cov-report=term
```

---

## Next Steps

### 1. Implement Phase 1 Components

The tests are ready. Now implement:
- `src/autolabeler/core/quality/structured_output_validator.py`
- `src/autolabeler/core/quality/confidence_calibrator.py`
- `src/autolabeler/core/quality/quality_monitor.py`

### 2. Run Tests Against Implementation

```bash
# As you implement, run tests
pytest tests/test_unit/test_confidence_calibrator.py -v
pytest tests/test_unit/test_quality_monitor.py -v
pytest tests/test_unit/test_structured_output_validator.py -v
```

### 3. Check Coverage

```bash
./scripts/run_phase1_tests.sh coverage
```

### 4. Run Full Suite

```bash
./scripts/run_phase1_tests.sh all
```

### 5. Commit and Push

```bash
git add .
git commit -m "Implement Phase 1 quality control components"
git push
```

GitHub Actions will automatically run the full test suite.

---

## Resources

- **Test README:** `/tests/README.md` - Comprehensive guide
- **Testing Strategy:** `/.hive-mind/TESTING_STRATEGY.md` - Full strategy
- **Implementation Summary:** `/.hive-mind/PHASE1_TESTING_IMPLEMENTATION_SUMMARY.md`
- **pytest docs:** https://docs.pytest.org/
- **pytest-cov docs:** https://pytest-cov.readthedocs.io/

---

## Quick Reference Card

```bash
# Essential Commands
pytest tests/                              # Run all tests
pytest tests/test_unit/ -m unit            # Run unit tests
pytest tests/ --cov --cov-report=html      # Coverage report
./scripts/run_phase1_tests.sh quick       # Quick tests
./scripts/run_phase1_tests.sh ci          # CI simulation

# Test Markers
-m unit           # Unit tests
-m integration    # Integration tests
-m performance    # Performance tests
-m validation     # Validation tests
-m "not slow"     # Exclude slow tests

# Debugging
-v                # Verbose
-vv               # Very verbose
-s                # Show print statements
-x                # Stop on first failure
--pdb             # Drop to debugger on failure
--lf              # Run last failed

# Coverage
--cov=src/autolabeler                      # Enable coverage
--cov-report=html                          # HTML report
--cov-report=term-missing                  # Show missing lines
--cov-fail-under=75                        # Fail if <75%

# Performance
--benchmark-only                           # Run benchmarks only
--benchmark-save=name                      # Save benchmark
--benchmark-compare=name                   # Compare to saved
```

---

**Ready to test!** Start with `./scripts/run_phase1_tests.sh quick` and go from there.
