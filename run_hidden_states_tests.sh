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
fi

source .venv/bin/activate

# Set V1 engine flag
export VLLM_USE_V1=1

echo "Running hidden states test suite..."
echo "Note: These tests are expected to fail until implementation is complete."
echo

# Run all hidden states tests with verbose output
python -m pytest tests/v1/hidden_states/ -v --tb=short

echo
echo "Test run completed. Check output above for failure details."