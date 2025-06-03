# Hidden States Test Suite for vLLM v1

This directory contains comprehensive tests for the hidden states functionality in vLLM v1 engine.

## Overview

These tests are designed to **fail initially** until the hidden states implementation is complete. They serve as a specification for the expected behavior and will guide the implementation process.

## Test Structure

### Core Test Files

1. **`test_hidden_states_engine_core.py`**
   - Tests hidden states extraction at the EngineCore level
   - Verifies basic functionality, multiple requests, and performance
   - Tests various prompts and sampling parameters

2. **`test_hidden_states_model_runner.py`**
   - Tests hidden states handling in the ModelRunner
   - Focuses on data structure extensions and memory management
   - Tests batch processing and conditional extraction logic

3. **`test_hidden_states_api.py`**
   - Tests OpenAI-compatible API integration
   - Covers both `/v1/chat/completions` and `/v1/completions` endpoints
   - Tests streaming and non-streaming responses

4. **`test_hidden_states_integration.py`**
   - End-to-end integration tests
   - Performance impact measurement
   - Memory management and error handling
   - Consistency and serialization tests

5. **`conftest.py`**
   - Shared fixtures and utilities
   - Mock classes for testing
   - Performance monitoring tools

## Expected Implementation Changes

The tests assume the following changes will be made during implementation:

### Data Structure Extensions

```python
# EngineCoreRequest
class EngineCoreRequest:
    return_hidden_states: bool = False
    hidden_states_for_tokens: Optional[list[int]] = None

# ModelRunnerOutput  
@dataclass
class ModelRunnerOutput:
    last_hidden_states: Optional[dict[str, torch.Tensor]] = None
    hidden_states_positions: Optional[dict[str, list[int]]] = None

# EngineCoreOutput
class EngineCoreOutput:
    hidden_states: Optional[list[float]] = None
```

### API Extensions

```python
# Request payloads
{
    "return_hidden_states": true,  # New optional field
    # ... existing fields
}

# Response format
{
    "choices": [{
        "message": {
            "content": "...",
            "hidden_states": [0.1, 0.2, 0.3, ...]  # New optional field
        }
    }]
}
```

## Running the Tests

### Prerequisites

```bash
# Ensure V1 is enabled
export VLLM_USE_V1=1

# Install test dependencies
pip install pytest pytest-asyncio
```

### Run All Hidden States Tests

```bash
# From the vllm root directory
pytest tests/v1/hidden_states/ -v
```

### Run Specific Test Categories

```bash
# Engine core tests
pytest tests/v1/hidden_states/test_hidden_states_engine_core.py -v

# Model runner tests  
pytest tests/v1/hidden_states/test_hidden_states_model_runner.py -v

# API tests
pytest tests/v1/hidden_states/test_hidden_states_api.py -v

# Integration tests
pytest tests/v1/hidden_states/test_hidden_states_integration.py -v
```

### Run with Coverage

```bash
pytest tests/v1/hidden_states/ --cov=vllm.v1 --cov-report=html
```

## Test Categories and Expected Behavior

### 1. Basic Functionality Tests
- ✅ **Should pass now**: Tests without hidden states (baseline functionality)
- ❌ **Will fail**: Tests requesting hidden states until implementation

### 2. Data Structure Tests
- ❌ **Will fail**: Tests for extended data structures
- ❌ **Will fail**: Tensor shape and type validation
- ✅ **Should pass now**: Memory efficiency calculations

### 3. Performance Tests
- ✅ **Should pass now**: Baseline performance measurements
- ❌ **Will fail**: Performance comparison with hidden states
- ❌ **Will fail**: Memory usage validation

### 4. API Tests
- ✅ **Should pass now**: Standard API requests (without hidden states)
- ❌ **Will fail**: API requests with `return_hidden_states=true`
- ❌ **Will fail**: Response validation with hidden states

### 5. Integration Tests
- ❌ **Will fail**: End-to-end hidden states extraction
- ❌ **Will fail**: Serialization/deserialization tests
- ✅ **Should pass now**: Error handling for unsupported features

## Implementation Guidance

### Phase 1: Core Infrastructure
1. Extend `EngineCoreRequest` with hidden states fields
2. Modify `ModelRunnerOutput` to include hidden states data
3. Update `EngineCoreOutput` for ZMQ serialization

### Phase 2: Model Integration  
1. Add hidden states extraction to model forward pass
2. Implement conditional extraction in `GPUModelRunner`
3. Add memory management for hidden states tensors

### Phase 3: API Integration
1. Extend OpenAI API schemas
2. Add request parameter validation
3. Implement response formatting with hidden states

### Phase 4: Optimization
1. Add memory pooling for hidden states
2. Optimize serialization for ZMQ transfer
3. Ensure torch.compile compatibility

## Debugging Failed Tests

When tests fail during implementation:

1. **Check the error message** - Tests include detailed assertions about expected behavior
2. **Look for TODO comments** - These indicate code that needs to be uncommented when features are implemented
3. **Run subset of tests** - Focus on one component at a time
4. **Use performance monitoring** - Built-in fixtures help identify bottlenecks

## Contributing

When adding new tests:

1. Follow the existing naming convention
2. Add appropriate TODO comments for unimplemented features
3. Include both positive and negative test cases
4. Add performance and memory usage validations
5. Update this README if adding new test categories

## Implementation Status Tracking

| Component | Test File | Status | Notes |
|-----------|-----------|--------|-------|
| EngineCore | `test_hidden_states_engine_core.py` | ❌ Not implemented | Core extraction logic needed |
| ModelRunner | `test_hidden_states_model_runner.py` | ❌ Not implemented | Data structure extensions needed |
| API Layer | `test_hidden_states_api.py` | ❌ Not implemented | OpenAI API extensions needed |
| Integration | `test_hidden_states_integration.py` | ❌ Not implemented | End-to-end pipeline needed |

✅ = Implemented and passing  
❌ = Not implemented (tests failing as expected)  
⚠️ = Partially implemented  

---

*This test suite serves as both a specification and validation for the hidden states feature implementation in vLLM v1.*