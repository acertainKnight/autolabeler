# AutoLabeler Test Suite - Phase 1

## Overview

This directory contains a comprehensive test suite for Phase 1 of the AutoLabeler enhancement initiative. The test suite validates the quality control system including structured output validation, confidence calibration, quality monitoring, and cost tracking.

## Test Structure

```
tests/
├── conftest.py                          # Global fixtures and configuration
├── test_unit/                           # Unit tests (fast, isolated)
│   ├── test_confidence_calibrator.py    # ConfidenceCalibrator tests
│   ├── test_quality_monitor.py          # QualityMonitor tests
│   └── test_structured_output_validator.py  # StructuredOutputValidator tests
├── test_integration/                    # Integration tests
│   └── test_phase1_integration.py       # Phase 1 component integration
├── test_performance/                    # Performance/benchmark tests
│   └── test_phase1_performance.py       # Phase 1 performance SLAs
└── test_validation/                     # Validation/acceptance tests
    └── test_phase1_success_criteria.py  # Phase 1 success criteria
```

## Test Categories

### Unit Tests (`-m unit`)
- **Purpose**: Test individual components in isolation
- **Coverage**: 70-75% of total tests (~300 tests)
- **Execution Time**: <30 seconds
- **Focus Areas**:
  - ConfidenceCalibrator: Calibration methods, ECE computation
  - QualityMonitor: Krippendorff's alpha, CQAA, anomaly detection
  - StructuredOutputValidator: Pydantic validation, retry logic

### Integration Tests (`-m integration`)
- **Purpose**: Test components working together
- **Coverage**: 15-20% of total tests (~80 tests)
- **Execution Time**: <2 minutes
- **Focus Areas**:
  - Validation → Calibration pipeline
  - Calibration → Monitoring integration
  - End-to-end quality control workflow
  - Multi-annotator agreement tracking

### Performance Tests (`-m performance`)
- **Purpose**: Validate performance SLAs
- **Coverage**: 5-10% of total tests (~20 tests)
- **Execution Time**: <5 minutes
- **Focus Areas**:
  - Latency: p95 <2s for single label
  - Throughput: >50 items/minute
  - Calibration: <100ms for 1000 samples
  - Memory usage: <100MB for 10K annotations

### Validation Tests (`-m validation`)
- **Purpose**: Validate success criteria and acceptance tests
- **Coverage**: 5% of total tests (~15 tests)
- **Execution Time**: <10 minutes
- **Focus Areas**:
  - Parsing failure rate <1%
  - ECE <0.05
  - Test coverage >70%
  - Regression prevention

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test categories
```bash
# Unit tests only (fast)
pytest tests/test_unit/ -m unit

# Integration tests
pytest tests/test_integration/ -m integration

# Performance tests
pytest tests/test_performance/ -m performance

# Validation tests
pytest tests/test_validation/ -m validation
```

### Run with coverage
```bash
pytest tests/ --cov=src/sibyls --cov-report=html --cov-report=term-missing
```

### Run performance benchmarks
```bash
pytest tests/test_performance/ --benchmark-only --benchmark-sort=mean
```

### Run specific test file
```bash
pytest tests/test_unit/test_confidence_calibrator.py -v
```

### Run specific test
```bash
pytest tests/test_unit/test_confidence_calibrator.py::TestConfidenceCalibrator::test_fit_temperature_scaling -v
```

## Test Markers

Tests are organized with pytest markers:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (multiple components)
- `@pytest.mark.performance` - Performance/benchmark tests
- `@pytest.mark.validation` - Validation/acceptance tests
- `@pytest.mark.slow` - Slow-running tests (>10 seconds)
- `@pytest.mark.requires_api` - Tests requiring external API access

## Fixtures

### Common Fixtures (in conftest.py)

- `settings` - Test settings configuration
- `sample_labeled_df` - Sample labeled DataFrame
- `sample_unlabeled_df` - Sample unlabeled DataFrame
- `sample_calibration_data` - Calibration test data (1000 samples)
- `sample_multi_annotator_data` - Multi-annotator agreement data
- `sample_synthetic_dataset` - Synthetic dataset for active learning
- `mock_llm_client` - Mock LLM client (no API calls)
- `mock_embedding_model` - Mock embedding model
- `temp_storage_path` - Temporary storage directory
- `sample_cost_data` - Sample cost tracking data

### Helper Functions

- `generate_synthetic_sentiment_data(n_samples)` - Generate synthetic sentiment data
- `compute_ece(confidence_scores, true_labels, n_bins)` - Compute Expected Calibration Error

## Success Criteria

Phase 1 tests validate the following success criteria:

1. **Parsing Reliability**: <1% parsing failures with structured output validation
2. **Confidence Calibration**: ECE <0.05 after calibration
3. **Agreement Metrics**: Krippendorff's alpha operational and accurate
4. **Test Coverage**: >70% overall test coverage
5. **Performance**: p95 latency <2s, throughput >50 items/min
6. **Cost Tracking**: Accurate per-annotation cost tracking
7. **Quality Monitoring**: Anomaly detection and dashboard generation operational

## CI/CD Integration

Tests are automatically run in GitHub Actions on:
- Push to `main`, `develop`, or `feature/phase1-*` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

### Workflow Jobs

1. **unit-tests**: Fast unit tests on Python 3.10, 3.11, 3.12
2. **integration-tests**: Integration tests after unit tests pass
3. **performance-tests**: Performance benchmarks on PRs
4. **code-quality**: Black, Ruff, codespell checks
5. **full-test-suite**: Complete test suite on main branch

### Coverage Requirements

- Minimum coverage: 75%
- Coverage is measured with pytest-cov
- Reports uploaded to Codecov
- HTML reports available as artifacts

## Best Practices

### Writing New Tests

1. **Use appropriate markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.
2. **Use fixtures**: Leverage existing fixtures from conftest.py
3. **Mock external dependencies**: Use mock LLM clients, don't make real API calls
4. **Test edge cases**: Include boundary conditions and error cases
5. **Use parametrize**: Test multiple inputs with `@pytest.mark.parametrize`
6. **Clear assertions**: Use descriptive assertion messages
7. **Fast tests**: Keep unit tests under 1 second each

### Example Test

```python
import pytest

@pytest.mark.unit
class TestMyComponent:
    """Test suite for MyComponent."""

    def test_basic_functionality(self, settings):
        """Test basic functionality with clear description."""
        component = MyComponent(settings)
        result = component.process("input")

        assert result is not None, "Result should not be None"
        assert result.value > 0, "Value should be positive"

    @pytest.mark.parametrize("input_value,expected", [
        (1, 2),
        (5, 10),
        (10, 20),
    ])
    def test_multiple_inputs(self, input_value, expected):
        """Test with multiple input values."""
        component = MyComponent()
        result = component.double(input_value)

        assert result == expected
```

## Troubleshooting

### Tests failing locally but passing in CI
- Check Python version (tests run on 3.10, 3.11, 3.12)
- Check for floating point precision issues
- Check for timezone or locale issues
- Verify all dependencies are installed: `pip install -e ".[dev]"`

### Slow tests
- Use `-m unit` to run only fast unit tests
- Use `--maxfail=1` to stop on first failure
- Use `-x` to exit on first failure
- Check for unnecessary API calls or I/O operations

### Coverage issues
- Run with `--cov-report=html` to see which lines are missing
- Check that test imports match source structure
- Verify coverage config in pyproject.toml
- Use `# pragma: no cover` for truly untestable lines

## Maintenance

### Updating Tests

When adding new Phase 1 features:
1. Add unit tests for the new component
2. Add integration tests for component interactions
3. Add performance tests if applicable
4. Update validation tests for new success criteria
5. Update this README with new fixtures or patterns

### Test Data

- Mock data is generated in conftest.py
- Real benchmark datasets should go in `tests/fixtures/`
- Keep test data small and representative
- Use seeds for reproducibility (e.g., `np.random.seed(42)`)

## References

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [Phase 1 Implementation Plan](../.hive-mind/MASTER_IMPLEMENTATION_PLAN.md)
- [Testing Strategy](../.hive-mind/TESTING_STRATEGY.md)

## Contact

For questions or issues with the test suite, please refer to the [Testing Strategy](.hive-mind/TESTING_STRATEGY.md) document or open an issue.
