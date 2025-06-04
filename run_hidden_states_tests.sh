#!/bin/bash

# Script to run hidden states tests with proper environment setup
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up environment for hidden states tests..."

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Installing basic test dependencies..."
    pip install pytest pytest-asyncio > /dev/null 2>&1
else
    source .venv/bin/activate
fi

# Set V1 engine flag
export VLLM_USE_V1=1

echo "Running hidden states test suite..."
echo "Note: Tests are designed as implementation specifications."
echo "Current implementation status from DESIGN.md:"
echo "✅ Data structures extended (EngineCoreRequest, ModelRunnerOutput, etc.)"
echo "🔄 ZMQ pipeline partially implemented"
echo "❌ Model forward pass integration not started"
echo "❌ API integration not started"
echo

# Check if we want to run all tests or specific categories
if [ "$1" = "--fast" ]; then
    echo "Running only basic structure tests (faster)..."
    python -m pytest tests/v1/hidden_states/test_hidden_states_engine_core.py::test_engine_core_basic_hidden_states -v --tb=short
elif [ "$1" = "--data-structures" ]; then
    echo "Running data structure tests..."
    python -m pytest tests/v1/hidden_states/test_hidden_states_model_runner.py -v --tb=short -k "structure"
elif [ "$1" = "--current" ]; then
    echo "Running tests for currently implemented features..."
    python -m pytest tests/v1/hidden_states/ -v --tb=short -k "without_hidden_states or structure"
else
    echo "Running all hidden states tests..."
    echo "Use --fast for quick test, --data-structures for structure tests, --current for implemented features"
    python -m pytest tests/v1/hidden_states/ -v --tb=short
fi

echo
echo "Test run completed."
echo "For test alignment with DESIGN.md, see: ai-guidance/DESIGN.md"