#!/usr/bin/env python3
"""
Simple test script to verify hidden states extraction is working.
This script tests the core functionality without the complex engine core setup.
"""

import os
import sys
import torch
from typing import Optional
import vllm
from time import sleep

# Set V1 engine flag
os.environ["VLLM_USE_V1"] = "1"

def test_hidden_states_model_runner():
    """Test the ModelRunnerOutput structure with hidden states."""
    print("Testing ModelRunnerOutput with hidden states...")
    
    from vllm.v1.outputs import ModelRunnerOutput
    
    # Test creating ModelRunnerOutput with hidden states
    hidden_size = 2048
    mock_hidden_states = torch.randn(1, hidden_size, dtype=torch.float32)
    
    output = ModelRunnerOutput(
        req_ids=["test_req_1"],
        req_id_to_index={"test_req_1": 0},
        sampled_token_ids=[[123]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        # Test the new hidden states fields
        last_hidden_states={"test_req_1": mock_hidden_states},
        hidden_states_positions={"test_req_1": [0]},
    )
    
    # Verify the fields exist and work correctly
    assert hasattr(output, 'last_hidden_states')
    assert hasattr(output, 'hidden_states_positions')
    assert output.last_hidden_states is not None
    assert "test_req_1" in output.last_hidden_states
    assert torch.equal(output.last_hidden_states["test_req_1"], mock_hidden_states)
    assert output.hidden_states_positions["test_req_1"] == [0]
    
    print("✅ ModelRunnerOutput with hidden states: PASSED")
    return True

def test_data_structures_flow():
    """Test that the data structures pass hidden states correctly."""
    print("Testing data structures flow...")
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.request import Request
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.worker.gpu_input_batch import CachedRequestState
    from vllm import SamplingParams
    import time
    
    # Test EngineCoreRequest with hidden states
    engine_request = EngineCoreRequest(
        request_id="test_123",
        prompt_token_ids=[1, 2, 3],
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=SamplingParams(max_tokens=5),
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        return_hidden_states=True,
        hidden_states_for_tokens=[-1],
    )
    
    # Test conversion to Request
    request = Request.from_engine_core_request(engine_request)
    assert hasattr(request, 'return_hidden_states')
    assert hasattr(request, 'hidden_states_for_tokens')
    assert request.return_hidden_states == True
    assert request.hidden_states_for_tokens == [-1]
    
    # Test conversion to NewRequestData
    new_req_data = NewRequestData.from_request(request, block_ids=[[1, 2, 3]])
    assert hasattr(new_req_data, 'return_hidden_states')
    assert hasattr(new_req_data, 'hidden_states_for_tokens')
    assert new_req_data.return_hidden_states == True
    assert new_req_data.hidden_states_for_tokens == [-1]
    
    # Test CachedRequestState creation
    cached_state = CachedRequestState(
        req_id="test_123",
        prompt_token_ids=[1, 2, 3],
        mm_inputs=[],
        mm_positions=[],
        sampling_params=SamplingParams(max_tokens=5),
        generator=None,
        block_ids=[[1, 2, 3]],
        num_computed_tokens=0,
        output_token_ids=[],
        lora_request=None,
        return_hidden_states=new_req_data.return_hidden_states,
        hidden_states_for_tokens=new_req_data.hidden_states_for_tokens,
    )
    
    assert hasattr(cached_state, 'return_hidden_states')
    assert hasattr(cached_state, 'hidden_states_for_tokens')
    assert cached_state.return_hidden_states == True
    assert cached_state.hidden_states_for_tokens == [-1]
    
    print("✅ Data structures flow: PASSED")
    return True


def test_zmq_pipeline_structures():
    """Test ZMQ pipeline data structures."""
    print("Testing ZMQ pipeline structures...")
    
    from vllm.v1.engine import HiddenStatesExtractionRequest, EngineCoreRequestType
    from vllm.v1.engine.output_processor import OutputProcessorOutput, CompletedRequestInfo
    from vllm.v1.engine import EngineCoreRequest
    from vllm import SamplingParams
    import time
    
    # Test HiddenStatesExtractionRequest creation
    hs_request = HiddenStatesExtractionRequest(
        request_id="hs_test_request_123",
        original_request_id="original_request_456",
        sequence_tokens=[1, 2, 3, 4, 5],
        target_position=-1,
        arrival_time=time.time(),
    )
    
    assert hs_request.request_id == "hs_test_request_123"
    assert hs_request.original_request_id == "original_request_456"
    assert hs_request.target_position == -1
    
    # Test CompletedRequestInfo
    original_request = EngineCoreRequest(
        request_id="original_123",
        prompt_token_ids=[1, 2, 3],
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=SamplingParams(max_tokens=5),
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        return_hidden_states=True,
        hidden_states_for_tokens=None
    )
    
    completed_info = CompletedRequestInfo(
        request_id="original_123",
        original_request=original_request,
        sequence_tokens=[1, 2, 3, 4, 5],
        final_token_position=4
    )
    
    assert completed_info.request_id == "original_123"
    assert completed_info.original_request.return_hidden_states == True
    
    # Test request type
    assert hasattr(EngineCoreRequestType, 'HIDDEN_STATES_EXTRACT')
    assert EngineCoreRequestType.HIDDEN_STATES_EXTRACT.value == b'\x05'
    
    print("✅ ZMQ pipeline structures: PASSED")
    return True
        


def test_hidden_states_actual_request():
    """Test retrieving hidden states via an actual engine call."""
    print("Testing actual engine hidden states extraction via actual engine call...")

    llm = vllm.LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        enable_lora=False,
        max_num_seqs=16,
        max_loras=4,
        max_model_len=400,
        gpu_memory_utilization=0.2,  #avoid OOM
        quantization=None,
        trust_remote_code=True,
        enable_chunked_prefill=True)

    prompt = "The capital of France is"
    sampling_params = vllm.SamplingParams(temperature=0,
                                          return_hidden_states=True,
                                          hidden_states_for_tokens=[-1],
                                          max_tokens=10)
    outputs = llm.generate(
        prompt,
        sampling_params)
    
    output = outputs[0]
    
    hidden_states = getattr(output, "hidden_states", None)
    assert hidden_states is not None, "Engine output missing hidden_states"
    print(hidden_states)
    print("✅ Actual engine hidden states extraction: PASSED")


    sleep(5)
    return True


def wrap_test(test_func):
    try:
        return test_func()
    except Exception as e:
        import traceback
        print(f"❌ Test failed: {e}")
        print(traceback.format_exc())
        return False

def main():
    """Run all tests."""
    print("🔍 Testing Hidden States Implementation")
    print("=" * 50)
    
    all_passed = True
    
    # Test individual components
    all_passed &= wrap_test(test_hidden_states_model_runner)
    all_passed &= wrap_test(test_data_structures_flow)
    all_passed &= wrap_test(test_zmq_pipeline_structures)
    all_passed &= wrap_test(test_hidden_states_actual_request)
    
    print("=" * 50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())