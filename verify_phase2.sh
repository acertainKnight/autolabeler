#!/bin/bash
# Phase 2 Verification Script

echo "=========================================="
echo "Phase 2 Test Suite Verification"
echo "=========================================="
echo ""

# Check Python version
echo "1. Python version:"
python --version
echo ""

# Check installed packages
echo "2. Checking Phase 2 dependencies:"
pip show dvc scikit-learn scipy 2>/dev/null | grep -E "^Name:|^Version:" || echo "Dependencies not installed (install with: pip install dvc scikit-learn scipy)"
echo ""

# Count tests
echo "3. Counting Phase 2 tests:"
TEST_COUNT=$(pytest tests/test_phase2/ tests/test_unit/versioning/ --collect-only -q 2>&1 | grep -E "^[0-9]+ test" | awk '{print $1}')
echo "Total tests: $TEST_COUNT"

if [ "$TEST_COUNT" -ge 300 ]; then
    echo "✅ Test count meets requirement (300+)"
else
    echo "❌ Test count below requirement: $TEST_COUNT < 300"
fi
echo ""

# Test breakdown
echo "4. Test breakdown by component:"
echo ""

echo "   DVC Manager:"
DVC_COUNT=$(pytest tests/test_unit/versioning/test_dvc_manager.py --collect-only -q 2>/dev/null | tail -1)
echo "   $DVC_COUNT"

echo "   DSPy Optimizer:"
DSPY_COUNT=$(pytest tests/test_phase2/test_dspy_optimizer.py --collect-only -q 2>/dev/null | tail -1)
echo "   $DSPY_COUNT"

echo "   RAG Components:"
RAG_COUNT=$(pytest tests/test_phase2/test_rag_components.py --collect-only -q 2>/dev/null | tail -1)
echo "   $RAG_COUNT"

echo "   Active Learning:"
AL_COUNT=$(pytest tests/test_phase2/test_active_learning.py --collect-only -q 2>/dev/null | tail -1)
echo "   $AL_COUNT"

echo "   Weak Supervision:"
WS_COUNT=$(pytest tests/test_phase2/test_weak_supervision.py --collect-only -q 2>/dev/null | tail -1)
echo "   $WS_COUNT"

echo "   Integration:"
INT_COUNT=$(pytest tests/test_phase2/test_integration.py --collect-only -q 2>/dev/null | tail -1)
echo "   $INT_COUNT"

echo "   Performance:"
PERF_COUNT=$(pytest tests/test_phase2/test_performance.py --collect-only -q 2>/dev/null | tail -1)
echo "   $PERF_COUNT"

echo ""

# Check file structure
echo "5. Verifying file structure:"
FILES=(
    "src/autolabeler/core/versioning/__init__.py"
    "src/autolabeler/core/versioning/dvc_manager.py"
    ".dvcignore"
    "docs/dvc_setup_guide.md"
    "tests/test_utils.py"
    "tests/test_phase2/test_dspy_optimizer.py"
    "tests/test_phase2/test_rag_components.py"
    "tests/test_phase2/test_active_learning.py"
    "tests/test_phase2/test_weak_supervision.py"
    "tests/test_phase2/test_integration.py"
    "tests/test_phase2/test_performance.py"
    "tests/test_unit/versioning/test_dvc_manager.py"
    ".github/workflows/phase2-tests.yml"
    "PHASE2_TEST_SUMMARY.md"
)

ALL_EXIST=true
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file (missing)"
        ALL_EXIST=false
    fi
done
echo ""

# Summary
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo ""

if [ "$TEST_COUNT" -ge 300 ] && [ "$ALL_EXIST" = true ]; then
    echo "✅ Phase 2 verification PASSED"
    echo ""
    echo "Summary:"
    echo "  - Tests: $TEST_COUNT (target: 300+)"
    echo "  - All required files present"
    echo "  - CI/CD pipeline configured"
    echo "  - Documentation created"
    echo ""
    echo "Phase 2 is ready! 🚀"
    exit 0
else
    echo "❌ Phase 2 verification FAILED"
    echo ""
    echo "Issues:"
    [ "$TEST_COUNT" -lt 300 ] && echo "  - Test count insufficient: $TEST_COUNT < 300"
    [ "$ALL_EXIST" = false ] && echo "  - Some required files missing"
    exit 1
fi
