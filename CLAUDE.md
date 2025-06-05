# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vLLM is a high-throughput, memory-efficient inference and serving engine for Large Language Models. It's a PyTorch Foundation hosted project originally developed at UC Berkeley.

## Key Commands

### Development Setup
```bash
# Install development dependencies
pip install -r requirements/dev.txt

# Install pre-commit hooks (replaces old format.sh)
pre-commit install

# Build from source
pip install -e .
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test directory
pytest tests/core/

# Run single test file
pytest tests/test_outputs.py -v

# Hidden states specific tests (current branch)
./run_hidden_states_tests.sh
./run_single_hidden_states_test.sh [test_name]
```

### Code Quality
```bash
# Linting and formatting (via pre-commit)
pre-commit run --all-files

# Type checking
tools/mypy.sh

# Manual ruff check
ruff check vllm/
```

## Architecture Overview

### V1 vs V0 Architecture
- **V0**: Legacy architecture in most of `vllm/` (engine/, worker/, etc.)
- **V1**: Next-generation architecture in `vllm/v1/` with cleaner separation, better performance
- **Current Branch**: Implementing hidden states extraction in V1 only

### Core Components
- **Engine** (`vllm/engine/`, `vllm/v1/engine/`): Request orchestration and execution
- **Model Executor** (`vllm/model_executor/`): Model loading and execution
- **Workers** (`vllm/worker/`): Distributed execution across devices  
- **Attention** (`vllm/attention/`): PagedAttention and attention backends
- **Core** (`vllm/core/`): Scheduling and block management

### Hidden States Implementation (Current Branch)
- **Architecture**: ZMQ-based post-sampling extraction
- **Location**: V1 engine only (`vllm/v1/`)
- **Test Suite**: 38 comprehensive tests in various test directories
- **Status**: Phase 1 complete, core functionality implemented

## Development Patterns

### Code Style
- Follow Google Python/C++ style guides
- Use pre-commit hooks for automatic formatting
- Line length: 80 characters (ruff configured)
- Type hints required for new code

### Testing Requirements  
- Write tests before implementation (TDD approach)
- Place tests in `tests/` matching source structure
- Use pytest fixtures from `conftest.py` files
- Include integration tests for API changes

### Commit Requirements
- Use DCO sign-off: `git commit -s`
- Prefix titles: `[Core]`, `[Model]`, `[Frontend]`, etc.
- Write clear, descriptive commit messages

### Performance Considerations
- Prefer V1 architecture for new features
- Consider CUDA graph compatibility
- Minimize memory allocations in hot paths
- Test performance impact of changes

## File Organization

### Key Entry Points
- `vllm/__init__.py`: Main library interface
- `vllm/engine/llm_engine.py`: V0 engine core
- `vllm/v1/engine/core.py`: V1 engine core
- `vllm/entrypoints/`: API servers and CLI

### Model Support
- `vllm/model_executor/models/`: Model implementations
- Models auto-registered via `@MODELS.register_model()` decorator
- Support for quantization, LoRA, multimodal inputs

### Testing Structure
- `tests/`: Matches source directory structure
- `tests/conftest.py`: Shared fixtures and utilities  
- `tests/v1/`: V1-specific tests including hidden states

## Current Development Context

This branch implements hidden states extraction for the V1 engine:
- **Feature**: Extract hidden states from any layer post-sampling
- **Architecture**: Separate ZMQ-based requests to avoid generation pipeline impact
- **Scope**: V1 engine only (not backward compatible with V0)
- **Testing**: Comprehensive test suite covering engine, API, and integration scenarios

## Build System

- **Build Backend**: setuptools with setuptools-scm for versioning
- **Dependencies**: Managed via requirements/*.txt files
- **CUDA Kernels**: Built via CMake and PyTorch extensions
- **Platform Support**: CUDA, ROCm, CPU, TPU, XPU with platform-specific backends