#!/bin/bash

# Script to run a specific hidden states test file
set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <test_file>"
    echo "Examples:"
    echo "  $0 test_hidden_states_engine_core.py"
    echo "  $0 test_hidden_states_model_runner.py"
    echo "  $0 test_hidden_states_api.py"
    echo "  $0 test_hidden_states_integration.py"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up environment for hidden states test: $1"

# Activate virtual environment
source .venv/bin/activate

# Set V1 engine flag
export VLLM_USE_V1=1

echo "Running $1..."
echo "Note: This test is expected to fail until implementation is complete."
echo

# Run specific test file with verbose output
python -m pytest "tests/v1/hidden_states/$1" -v --tb=short -s

echo
echo "Test run completed."