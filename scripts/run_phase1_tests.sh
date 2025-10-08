#!/bin/bash
# Phase 1 Test Execution Script
# Runs comprehensive test suite for Phase 1 components

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}AutoLabeler Phase 1 Test Suite${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found${NC}"
    echo "Please install with: pip install -e \".[dev]\""
    exit 1
fi

# Parse command line arguments
TEST_MODE=${1:-"all"}
VERBOSE=${2:-""}

# Function to run tests with timing
run_test_suite() {
    local name=$1
    local command=$2

    echo -e "${YELLOW}Running $name...${NC}"
    start_time=$(date +%s)

    if eval "$command"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo -e "${GREEN}✓ $name passed (${duration}s)${NC}"
        echo ""
        return 0
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo -e "${RED}✗ $name failed (${duration}s)${NC}"
        echo ""
        return 1
    fi
}

# Main test execution
case "$TEST_MODE" in
    "unit")
        echo "Running unit tests only..."
        run_test_suite "Unit Tests" "pytest tests/test_unit/ -m unit -v $VERBOSE"
        ;;

    "integration")
        echo "Running integration tests..."
        run_test_suite "Integration Tests" "pytest tests/test_integration/ -m integration -v $VERBOSE"
        ;;

    "performance")
        echo "Running performance tests..."
        run_test_suite "Performance Tests" "pytest tests/test_performance/ -m performance --benchmark-only $VERBOSE"
        ;;

    "validation")
        echo "Running validation tests..."
        run_test_suite "Validation Tests" "pytest tests/test_validation/ -m validation -v $VERBOSE"
        ;;

    "quick")
        echo "Running quick test suite (unit tests only)..."
        run_test_suite "Quick Unit Tests" "pytest tests/test_unit/ -m unit -v --maxfail=3 $VERBOSE"
        ;;

    "coverage")
        echo "Running tests with coverage..."
        run_test_suite "Coverage Tests" "pytest tests/ --cov=src/autolabeler --cov-report=html --cov-report=term-missing --cov-fail-under=75 $VERBOSE"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Coverage report generated: htmlcov/index.html${NC}"
        fi
        ;;

    "ci")
        echo "Running CI test suite..."

        # Unit tests
        if ! run_test_suite "Unit Tests" "pytest tests/test_unit/ -m unit -v --tb=short"; then
            exit 1
        fi

        # Integration tests
        if ! run_test_suite "Integration Tests" "pytest tests/test_integration/ -m integration -v --tb=short --maxfail=3"; then
            exit 1
        fi

        # Code quality (if available)
        if command -v black &> /dev/null && command -v ruff &> /dev/null; then
            echo -e "${YELLOW}Running code quality checks...${NC}"
            black --check src/ tests/ 2>/dev/null || echo -e "${YELLOW}⚠ Black formatting issues found${NC}"
            ruff check src/ tests/ 2>/dev/null || echo -e "${YELLOW}⚠ Ruff linting issues found${NC}"
            echo ""
        fi

        # Coverage check
        if ! run_test_suite "Coverage Check" "pytest tests/ --cov=src/autolabeler --cov-report=term-missing --cov-fail-under=75 -v"; then
            echo -e "${RED}Coverage below 75% threshold${NC}"
            exit 1
        fi

        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}All CI tests passed! ✓${NC}"
        echo -e "${GREEN}========================================${NC}"
        ;;

    "all")
        echo "Running complete test suite..."

        FAILED=0

        # Unit tests
        run_test_suite "Unit Tests" "pytest tests/test_unit/ -m unit -v" || FAILED=1

        # Integration tests
        run_test_suite "Integration Tests" "pytest tests/test_integration/ -m integration -v" || FAILED=1

        # Performance tests
        run_test_suite "Performance Tests" "pytest tests/test_performance/ -m performance -v" || FAILED=1

        # Validation tests
        run_test_suite "Validation Tests" "pytest tests/test_validation/ -m validation -v" || FAILED=1

        # Coverage report
        echo -e "${YELLOW}Generating coverage report...${NC}"
        pytest tests/ --cov=src/autolabeler --cov-report=html --cov-report=term-missing > /dev/null 2>&1 || true

        if [ $FAILED -eq 0 ]; then
            echo -e "${GREEN}========================================${NC}"
            echo -e "${GREEN}All tests passed! ✓${NC}"
            echo -e "${GREEN}Coverage report: htmlcov/index.html${NC}"
            echo -e "${GREEN}========================================${NC}"
        else
            echo -e "${RED}========================================${NC}"
            echo -e "${RED}Some tests failed ✗${NC}"
            echo -e "${RED}========================================${NC}"
            exit 1
        fi
        ;;

    "help"|"-h"|"--help")
        echo "Usage: ./run_phase1_tests.sh [MODE] [OPTIONS]"
        echo ""
        echo "Modes:"
        echo "  all          - Run complete test suite (default)"
        echo "  unit         - Run unit tests only (fast)"
        echo "  integration  - Run integration tests"
        echo "  performance  - Run performance/benchmark tests"
        echo "  validation   - Run validation/acceptance tests"
        echo "  quick        - Run quick unit tests (fail fast)"
        echo "  coverage     - Run tests with coverage report"
        echo "  ci           - Run CI test suite (unit + integration + coverage)"
        echo "  help         - Show this help message"
        echo ""
        echo "Options:"
        echo "  -v           - Verbose output"
        echo "  -vv          - Very verbose output"
        echo ""
        echo "Examples:"
        echo "  ./run_phase1_tests.sh unit              # Run unit tests"
        echo "  ./run_phase1_tests.sh coverage -v       # Coverage with verbose"
        echo "  ./run_phase1_tests.sh quick             # Fast unit tests"
        echo "  ./run_phase1_tests.sh ci                # Full CI suite"
        ;;

    *)
        echo -e "${RED}Unknown test mode: $TEST_MODE${NC}"
        echo "Use './run_phase1_tests.sh help' for usage information"
        exit 1
        ;;
esac
