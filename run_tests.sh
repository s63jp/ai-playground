#!/usr/bin/env bash
# Full verification: run ALL test suites + health check. Exit code is non-zero on any failure.
#
# Usage:
#   ./run_tests.sh           # Run all tests
#   ./run_tests.sh --fast    # Skip XPU/GPU tests and slow markers
#
set -euo pipefail
cd "$(dirname "$0")"

# Find the Python interpreter
PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"
[ -x "$PY" ] || { echo "ERROR: No Python interpreter found in .venv"; exit 1; }

# Parse arguments
FAST_MODE=false
PYTEST_ARGS=()
HEALTH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast)
            FAST_MODE=true
            PYTEST_ARGS+=("-m" "not xpu")
            HEALTH_ARGS+=("--no-xpu")
            shift
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

# Discover all test files
TEST_FILES=$(find tests -name 'test_*.py' -type f | sort)
if [[ -z "$TEST_FILES" ]]; then
    echo "ERROR: No test files found in tests/"
    exit 1
fi

echo "========================================"
echo "Running AI Playground Test Suite"
echo "========================================"
echo "Mode: $([ "$FAST_MODE" = true ] && echo 'FAST (skipping XPU tests)' || echo 'FULL')"
echo "Python: $PY"
echo "Test files:"
echo "$TEST_FILES" | sed 's/^/  /'
echo ""

FAILED=0
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Run pytest on all test files
echo "== Running pytest =="
PYTEST_OUTPUT=$("$PY" -m pytest $TEST_FILES -v -p no:cacheprovider -W ignore::DeprecationWarning "${PYTEST_ARGS[@]}" 2>&1) || true

# Parse pytest output for summary
echo "$PYTEST_OUTPUT"

# Extract counts from pytest output
if [[ "$PYTEST_OUTPUT" =~ ([0-9]+)\ passed ]]; then
    PASSED_TESTS=${BASH_REMATCH[1]}
    TOTAL_TESTS=$((TOTAL_TESTS + PASSED_TESTS))
fi
if [[ "$PYTEST_OUTPUT" =~ ([0-9]+)\ failed ]]; then
    FAILED_TESTS=${BASH_REMATCH[1]}
    TOTAL_TESTS=$((TOTAL_TESTS + FAILED_TESTS))
fi
if [[ "$PYTEST_OUTPUT" =~ ([0-9]+)\ skipped ]]; then
    SKIPPED_TESTS=${BASH_REMATCH[1]}
    TOTAL_TESTS=$((TOTAL_TESTS + SKIPPED_TESTS))
fi

if [[ $FAILED_TESTS -gt 0 ]]; then
    FAILED=1
fi

echo ""
echo "== Running health check =="
if ! "$PY" scripts/health_check.py "${HEALTH_ARGS[@]}"; then
    echo "Health check failed"
    FAILED=1
fi

echo ""
echo "========================================"
echo "TEST SUMMARY"
echo "========================================"
echo "Passed:  $PASSED_TESTS"
echo "Failed:  $FAILED_TESTS"
echo "Skipped: $SKIPPED_TESTS"
echo "Total:   $TOTAL_TESTS"
echo ""

if [[ $FAILED -eq 0 ]]; then
    echo "✓ All tests passed"
    exit 0
else
    echo "✗ Some tests failed"
    exit 1
fi
