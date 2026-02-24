#!/bin/bash
# run_lsf_integrationtest.sh
# Run each test (either file or individual test function) in separate LSF jobs with coverage + logs.
#
# Environment variables:
#   TEST_DIR    - Base directory for tests (default: current directory or first argument)
#   TEST_FILE   - Test file to run (default: integrationtests/test_base_set.py)
#   VENV_PATH   - Path to Python virtual environment (default: .venv)

set -euo pipefail

TEST_DIR="${TEST_DIR:-${1:-$(pwd)}}"
TEST_FILE="${1:-${TEST_FILE:-integrationtests/test_base_set.py}}"
VENV_PATH="${VENV_PATH:-.venv}"
LOG_DIR="$TEST_DIR/logs"
COV_DIR="$TEST_DIR/.coverage_jobs"
mkdir -p "$LOG_DIR" "$COV_DIR"


all_tests=$(cd "$TEST_DIR" && \
  pytest --collect-only -q "$TEST_FILE" 2>/dev/null | \
  grep -E '^integrationtests/test_base_set\.py::' || true)

echo "$all_tests"

for test in $all_tests; do
    # Normalize name (pytest nodeid may include "::class::test_func")
    test_name=$(echo "$test" | tr '/:' '_')
    out="$LOG_DIR/${test_name}.out"
    err="$LOG_DIR/${test_name}.err"


    echo "Submitting job for $test"
    hash=$(echo -n "$test" | sha1sum | cut -c1-10)
    job_name="tt_${hash}"
    bsub -gpu num=1 -R "rusage[ngpus=1,cpu=4,mem=32GB]" \
         -J "terratorch_${job_name}" \
         -oo "$out" -eo "$err" \
         "cd $TEST_DIR && \
          source $VENV_PATH/bin/activate && \
          pytest -s -v $test"
done

echo "All jobs submitted. Monitor with: bjobs -u \$USER"
