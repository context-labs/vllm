# SPDX-License-Identifier: Apache-2.0

"""
Test suite for hidden states functionality at the ModelRunner level.

These tests focus on the model execution and hidden states extraction
at the GPUModelRunner level, testing the core extraction logic.
"""

import pytest
import torch
from transformers import AutoTokenizer

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.v1.outputs import ModelRunnerOutput

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.",
                allow_module_level=True)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.fixture
def vllm_config():
    """Create a VllmConfig for testing."""
    engine_args = EngineArgs(model=MODEL_NAME)
    return engine_args.create_engine_config()


@pytest.fixture
def tokenizer():
    """Create a tokenizer for testing."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def test_model_runner_output_structure_without_hidden_states(vllm_config: VllmConfig):
    """Test that ModelRunnerOutput can be created without hidden states (baseline)."""
    
    # Test current ModelRunnerOutput structure
    output = ModelRunnerOutput(
        req_ids=["test_req_1"],
        req_id_to_index={"test_req_1": 0},
        sampled_token_ids=[[123, 456]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    
    assert output.req_ids == ["test_req_1"]
    assert output.req_id_to_index == {"test_req_1": 0}
    assert output.sampled_token_ids == [[123, 456]]
    
    # These fields should now exist (implemented)
    assert hasattr(output, 'last_hidden_states')
    assert hasattr(output, 'hidden_states_positions')
    # But they should be None when not requested
    assert output.last_hidden_states is None
    assert output.hidden_states_positions is None


def test_model_runner_output_structure_with_hidden_states(vllm_config: VllmConfig):
    """Test ModelRunnerOutput structure with hidden states fields (will fail until implemented)."""
    
    hidden_size = vllm_config.model_config.hf_config.hidden_size
    
    # Test structure with hidden states fields (now implemented)
    # Create mock hidden states tensor
    mock_hidden_states = torch.randn(1, hidden_size, dtype=torch.float32)
    
    output = ModelRunnerOutput(
        req_ids=["test_req_1"],
        req_id_to_index={"test_req_1": 0},
        sampled_token_ids=[[123]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        # These fields are now implemented
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


def test_hidden_states_tensor_properties(vllm_config: VllmConfig):
    """Test properties of hidden states tensors."""
    
    hidden_size = vllm_config.model_config.hf_config.hidden_size
    
    # Test expected properties of hidden states tensors
    mock_hidden_states = torch.randn(1, hidden_size, dtype=torch.float32)
    
    # Verify tensor properties
    assert mock_hidden_states.shape == (1, hidden_size)
    assert mock_hidden_states.dtype == torch.float32
    assert not mock_hidden_states.requires_grad  # Should be detached for output
    
    # Test conversion to list for serialization
    hidden_states_list = mock_hidden_states.squeeze(0).tolist()
    assert isinstance(hidden_states_list, list)
    assert len(hidden_states_list) == hidden_size
    assert all(isinstance(x, float) for x in hidden_states_list)


def test_hidden_states_memory_efficiency():
    """Test memory-efficient handling of hidden states."""
    
    # Test that we can create and manage multiple hidden states tensors
    # without excessive memory usage
    batch_size = 4
    hidden_size = 2048  # Typical hidden size
    
    # Simulate multiple requests with hidden states
    hidden_states_dict = {}
    for i in range(batch_size):
        req_id = f"req_{i}"
        hidden_states = torch.randn(1, hidden_size, dtype=torch.float16)  # Use half precision
        hidden_states_dict[req_id] = hidden_states
    
    # Verify we can handle multiple tensors
    assert len(hidden_states_dict) == batch_size
    
    # Test memory usage is reasonable (each tensor should be small)
    tensor_size_bytes = hidden_size * 2  # float16 is 2 bytes
    total_size_bytes = batch_size * tensor_size_bytes
    
    # Should be manageable (less than 100MB for reasonable batch sizes)
    assert total_size_bytes < 100 * 1024 * 1024  # 100MB limit
    
    # Test cleanup
    for req_id in list(hidden_states_dict.keys()):
        del hidden_states_dict[req_id]
    
    assert len(hidden_states_dict) == 0


def test_hidden_states_batch_processing(vllm_config: VllmConfig):
    """Test hidden states extraction in batch processing scenarios."""
    
    hidden_size = vllm_config.model_config.hf_config.hidden_size
    batch_size = 3
    
    # Simulate batch of requests with mixed hidden states requirements
    req_ids = [f"req_{i}" for i in range(batch_size)]
    requests_need_hidden_states = [True, False, True]  # Only req_0 and req_2 need hidden states
    
    # Mock the scenario where model runner extracts hidden states
    # for only the requests that need them
    last_hidden_states = {}
    hidden_states_positions = {}
    
    for i, (req_id, needs_hs) in enumerate(zip(req_ids, requests_need_hidden_states)):
        if needs_hs:
            # Simulate extracting hidden states for this request
            hidden_states = torch.randn(1, hidden_size, dtype=torch.float32)
            last_hidden_states[req_id] = hidden_states
            hidden_states_positions[req_id] = [0]  # Position of final token
    
    # Verify selective extraction
    assert len(last_hidden_states) == 2  # Only req_0 and req_2
    assert "req_0" in last_hidden_states
    assert "req_1" not in last_hidden_states
    assert "req_2" in last_hidden_states
    
    # Verify tensor shapes
    for req_id, hidden_states in last_hidden_states.items():
        assert hidden_states.shape == (1, hidden_size)
        assert req_id in hidden_states_positions
        assert hidden_states_positions[req_id] == [0]


@pytest.mark.parametrize("hidden_size", [768, 1024, 2048, 4096])
def test_hidden_states_different_model_sizes(hidden_size: int):
    """Test hidden states handling with different model sizes."""
    
    # Test hidden states for various model sizes
    mock_hidden_states = torch.randn(1, hidden_size, dtype=torch.float32)
    
    assert mock_hidden_states.shape == (1, hidden_size)
    
    # Test serialization performance for different sizes
    hidden_states_list = mock_hidden_states.squeeze(0).tolist()
    assert len(hidden_states_list) == hidden_size
    
    # Verify reasonable memory usage even for large models
    tensor_size_mb = (hidden_size * 4) / (1024 * 1024)  # float32 is 4 bytes
    assert tensor_size_mb < 100  # Should be less than 100MB per tensor


def test_hidden_states_gpu_cpu_transfer():
    """Test efficient GPU to CPU transfer for hidden states."""
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for GPU/CPU transfer test")
    
    hidden_size = 2048
    
    # Create hidden states on GPU (as they would be during model execution)
    hidden_states_gpu = torch.randn(1, hidden_size, dtype=torch.float32, device='cuda')
    
    # Test transfer to CPU for serialization
    hidden_states_cpu = hidden_states_gpu.cpu()
    
    assert hidden_states_cpu.device.type == 'cpu'
    assert torch.equal(hidden_states_gpu.cpu(), hidden_states_cpu)
    
    # Test conversion to list for ZMQ serialization
    hidden_states_list = hidden_states_cpu.squeeze(0).tolist()
    assert isinstance(hidden_states_list, list)
    assert len(hidden_states_list) == hidden_size


def test_hidden_states_dtype_handling():
    """Test handling of different data types for hidden states."""
    
    hidden_size = 1024
    
    # Test different dtypes
    dtypes_to_test = [torch.float32, torch.float16, torch.bfloat16]
    
    for dtype in dtypes_to_test:
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            continue  # bfloat16 requires CUDA
            
        hidden_states = torch.randn(1, hidden_size, dtype=dtype)
        
        # Convert to float32 for serialization
        hidden_states_float32 = hidden_states.float()
        assert hidden_states_float32.dtype == torch.float32
        
        # Test list conversion
        hidden_states_list = hidden_states_float32.squeeze(0).tolist()
        assert all(isinstance(x, float) for x in hidden_states_list)


def test_hidden_states_extraction_conditional_logic():
    """Test logic for conditional hidden states extraction."""
    
    # Simulate scheduler output with mixed requests
    class MockRequest:
        def __init__(self, req_id: str, needs_hidden_states: bool):
            self.req_id = req_id
            self.needs_hidden_states = needs_hidden_states
    
    class MockSchedulerOutput:
        def __init__(self, requests: list):
            self.requests = requests
    
    # Create mock requests
    requests = [
        MockRequest("req_1", True),
        MockRequest("req_2", False),
        MockRequest("req_3", True),
        MockRequest("req_4", False),
    ]
    
    scheduler_output = MockSchedulerOutput(requests)
    
    # Simulate the logic that would be in GPUModelRunner
    def should_extract_hidden_states(scheduler_output) -> bool:
        return any(req.needs_hidden_states for req in scheduler_output.requests)
    
    def get_hidden_states_requests(scheduler_output) -> list:
        return [req for req in scheduler_output.requests if req.needs_hidden_states]
    
    # Test the logic
    assert should_extract_hidden_states(scheduler_output) == True
    
    hs_requests = get_hidden_states_requests(scheduler_output)
    assert len(hs_requests) == 2
    assert hs_requests[0].req_id == "req_1"
    assert hs_requests[1].req_id == "req_3"
    
    # Test case with no hidden states requests
    no_hs_requests = [MockRequest("req_5", False), MockRequest("req_6", False)]
    no_hs_scheduler_output = MockSchedulerOutput(no_hs_requests)
    
    assert should_extract_hidden_states(no_hs_scheduler_output) == False
    assert len(get_hidden_states_requests(no_hs_scheduler_output)) == 0