#!/bin/bash

# Script to validate the structure of hidden states tests without running them
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Validating hidden states test structure..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating minimal virtual environment for validation..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install minimal dependencies for syntax checking
pip install pytest > /dev/null 2>&1 || echo "Installing pytest..."
pip install pytest > /dev/null 2>&1

echo "Checking test file syntax and imports..."

# List of test files to validate
TEST_FILES=(
    "tests/v1/hidden_states/test_hidden_states_engine_core.py"
    "tests/v1/hidden_states/test_hidden_states_model_runner.py" 
    "tests/v1/hidden_states/test_hidden_states_api.py"
    "tests/v1/hidden_states/test_hidden_states_integration.py"
    "tests/v1/hidden_states/conftest.py"
)

for test_file in "${TEST_FILES[@]}"; do
    if [ -f "$test_file" ]; then
        echo "✓ Found: $test_file"
        # Try to compile the Python file to check syntax
        python -m py_compile "$test_file" 2>/dev/null && echo "  ✓ Syntax OK" || echo "  ✗ Syntax Error"
    else
        echo "✗ Missing: $test_file"
    fi
done

echo
echo "Test structure validation complete."
echo "Note: Import errors are expected until vLLM is fully installed."