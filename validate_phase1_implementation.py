#!/usr/bin/env python3

"""
Validation script for Phase 1 hidden states implementation.

This script tests the extended data structures without requiring
the full vLLM installation or model loading.
"""

import sys
from typing import Optional


def test_engine_core_request_fields():
    """Test that EngineCoreRequest has the new hidden states fields."""
    try:
        from vllm.v1.engine import EngineCoreRequest
        from vllm.sampling_params import SamplingParams
        
        # Test creation with new fields
        sampling_params = SamplingParams(max_tokens=10)
        
        request = EngineCoreRequest(
            request_id="test_id",
            prompt_token_ids=[1, 2, 3],
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=sampling_params,
            eos_token_id=2,
            arrival_time=1.0,
            lora_request=None,
            cache_salt=None,
            return_hidden_states=True,
            hidden_states_for_tokens=[0, 1, 2]
        )
        
        assert hasattr(request, 'return_hidden_states')
        assert hasattr(request, 'hidden_states_for_tokens')
        assert request.return_hidden_states == True
        assert request.hidden_states_for_tokens == [0, 1, 2]
        
        print("✓ EngineCoreRequest: hidden states fields added successfully")
        return True
        
    except Exception as e:
        print(f"✗ EngineCoreRequest test failed: {e}")
        return False


def test_engine_core_output_fields():
    """Test that EngineCoreOutput has the new hidden states field."""
    try:
        from vllm.v1.engine import EngineCoreOutput
        
        # Test creation with new field
        output = EngineCoreOutput(
            request_id="test_id",
            new_token_ids=[1, 2],
            hidden_states=[0.1, 0.2, 0.3, 0.4]
        )
        
        assert hasattr(output, 'hidden_states')
        assert output.hidden_states == [0.1, 0.2, 0.3, 0.4]
        
        print("✓ EngineCoreOutput: hidden states field added successfully")
        return True
        
    except Exception as e:
        print(f"✗ EngineCoreOutput test failed: {e}")
        return False


def test_model_runner_output_fields():
    """Test that ModelRunnerOutput has the new hidden states fields."""
    try:
        from vllm.v1.outputs import ModelRunnerOutput
        import torch
        
        # Test creation with new fields
        hidden_states_tensor = torch.randn(1, 4096)  # [1, hidden_size]
        
        output = ModelRunnerOutput(
            req_ids=["test_id"],
            req_id_to_index={"test_id": 0},
            sampled_token_ids=[[1, 2]],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            last_hidden_states={"test_id": hidden_states_tensor},
            hidden_states_positions={"test_id": [0]}
        )
        
        assert hasattr(output, 'last_hidden_states')
        assert hasattr(output, 'hidden_states_positions')
        assert "test_id" in output.last_hidden_states
        assert torch.equal(output.last_hidden_states["test_id"], hidden_states_tensor)
        assert output.hidden_states_positions["test_id"] == [0]
        
        print("✓ ModelRunnerOutput: hidden states fields added successfully")
        return True
        
    except Exception as e:
        print(f"✗ ModelRunnerOutput test failed: {e}")
        return False


def test_empty_model_runner_output():
    """Test that EMPTY_MODEL_RUNNER_OUTPUT includes new fields."""
    try:
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
        
        assert hasattr(EMPTY_MODEL_RUNNER_OUTPUT, 'last_hidden_states')
        assert hasattr(EMPTY_MODEL_RUNNER_OUTPUT, 'hidden_states_positions')
        assert EMPTY_MODEL_RUNNER_OUTPUT.last_hidden_states is None
        assert EMPTY_MODEL_RUNNER_OUTPUT.hidden_states_positions is None
        
        print("✓ EMPTY_MODEL_RUNNER_OUTPUT: updated with hidden states fields")
        return True
        
    except Exception as e:
        print(f"✗ EMPTY_MODEL_RUNNER_OUTPUT test failed: {e}")
        return False


def main():
    """Run all Phase 1 validation tests."""
    print("Phase 1 Hidden States Implementation Validation")
    print("=" * 50)
    
    tests = [
        test_engine_core_request_fields,
        test_engine_core_output_fields,
        test_model_runner_output_fields,
        test_empty_model_runner_output,
    ]
    
    results = []
    for test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    print()
    print("Summary:")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All Phase 1 data structure extensions completed successfully!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())