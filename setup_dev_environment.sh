#!/bin/bash

# Script to set up the development environment for vLLM hidden states implementation
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up vLLM development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

echo "Installing dependencies..."

# Install basic dependencies
pip install jinja2

# Set build configuration
export MAX_JOBS=6

# Install ninja build system (requires sudo)
echo "Installing ninja-build (requires sudo)..."
sudo apt install ninja-build -y

# Install vLLM in editable mode
echo "Installing vLLM in editable mode (this may take several minutes)..."
pip install -e .

# Install test dependencies
echo "Installing test dependencies..."
pip install -r requirements/test.txt
pip install pytest pytest-asyncio

echo
echo "Development environment setup complete!"
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
echo "  export VLLM_USE_V1=1"
echo
echo "To run hidden states tests:"
echo "  ./run_hidden_states_tests.sh"
echo "  ./run_single_hidden_states_test.sh <test_file>"