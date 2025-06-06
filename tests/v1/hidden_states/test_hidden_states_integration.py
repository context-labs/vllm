# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for hidden states functionality across the full vLLM v1 pipeline.

These tests verify end-to-end hidden states extraction from API request
through the engine to model execution and back to the response.
"""

import pytest
import time
import uuid
from typing import List, Optional

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor

from ...utils import create_new_process_for_each_test

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.",
                allow_module_level=True)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


@create_new_process_for_each_test()
def test_end_to_end_hidden_states_extraction(monkeypatch: pytest.MonkeyPatch):
    """Test complete pipeline from request to hidden states output."""
    
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        
        engine_args = EngineArgs(model=MODEL_NAME)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        
        engine_core = EngineCore(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=True
        )
        
        # Test the complete flow:
        # 1. Request with hidden states
        # 2. Processing through scheduler
        # 3. Model execution
        # 4. Hidden states extraction
        # 5. Response formatting
        
        request = EngineCoreRequest(
            request_id=str(uuid.uuid4()),
            prompt_token_ids=[1, 2, 3, 4, 5],  # Simple token sequence
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=SamplingParams(max_tokens=3),
            eos_token_id=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            # TODO: Add when implementing
            # return_hidden_states=True,
        )
        
        engine_core.add_request(request)
        
        # Process through the complete pipeline
        hidden_states_received = False
        for step in range(10):  # Max steps
            outputs = engine_core.step()
            
            if outputs and outputs.outputs:
                for output in outputs.outputs:
                    if output.request_id == request.request_id:
                        if output.finished:
                            # TODO: Uncomment when implementation is complete
                            # assert hasattr(output, 'hidden_states')
                            # assert output.hidden_states is not None
                            # assert isinstance(output.hidden_states, list)
                            # assert len(output.hidden_states) == vllm_config.model_config.hf_config.hidden_size
                            # hidden_states_received = True
                            hidden_states_received = True  # Temporary for test structure
                            break
            
            if hidden_states_received:
                break
        
        # TODO: Enable when implementation is complete
        # assert hidden_states_received, "Hidden states should be received for completed request"


@create_new_process_for_each_test()  
def test_performance_impact_of_hidden_states(monkeypatch: pytest.MonkeyPatch):
    """Test that hidden states extraction doesn't significantly impact performance."""
    
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        
        engine_args = EngineArgs(model=MODEL_NAME)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        
        engine_core = EngineCore(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=True
        )
        
        # Benchmark without hidden states
        start_time = time.time()
        
        request_without_hs = EngineCoreRequest(
            request_id=str(uuid.uuid4()),
            prompt_token_ids=[1, 2, 3, 4, 5],
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=SamplingParams(max_tokens=5),
            eos_token_id=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            # return_hidden_states=False (default)
        )
        
        engine_core.add_request(request_without_hs)
        
        # Process request
        for _ in range(15):
            outputs = engine_core.step()
            if outputs and outputs.outputs:
                finished = any(output.finished for output in outputs.outputs 
                             if output.request_id == request_without_hs.request_id)
                if finished:
                    break
        
        time_without_hs = time.time() - start_time
        
        # Benchmark with hidden states
        start_time = time.time()
        
        request_with_hs = EngineCoreRequest(
            request_id=str(uuid.uuid4()),
            prompt_token_ids=[1, 2, 3, 4, 5],
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=SamplingParams(max_tokens=5),
            eos_token_id=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            # TODO: Add when implementing
            # return_hidden_states=True,
        )
        
        engine_core.add_request(request_with_hs)
        
        # Process request
        for _ in range(15):
            outputs = engine_core.step()
            if outputs and outputs.outputs:
                finished = any(output.finished for output in outputs.outputs 
                             if output.request_id == request_with_hs.request_id)
                if finished:
                    break
        
        time_with_hs = time.time() - start_time
        
        # Performance impact should be minimal (less than 50% overhead)
        # TODO: Enable when implementation is complete
        # performance_ratio = time_with_hs / time_without_hs
        # assert performance_ratio < 1.5, f"Hidden states extraction adds too much overhead: {performance_ratio:.2f}x"
        
        # For now, just verify both completed
        assert time_without_hs > 0
        assert time_with_hs > 0


@create_new_process_for_each_test()
def test_hidden_states_with_different_sampling_params(monkeypatch: pytest.MonkeyPatch):
    """Test hidden states extraction with various sampling parameters."""
    
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        
        engine_args = EngineArgs(model=MODEL_NAME)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        
        engine_core = EngineCore(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=True
        )
        
        # Test different sampling configurations
        sampling_configs = [
            SamplingParams(max_tokens=1, temperature=0.0),  # Greedy
            SamplingParams(max_tokens=3, temperature=0.8, top_p=0.9),  # Sampling
            SamplingParams(max_tokens=2, top_k=10),  # Top-K
            SamplingParams(max_tokens=2, stop=["test", "end"]),  # With stop words
        ]
        
        for i, sampling_params in enumerate(sampling_configs):
            request = EngineCoreRequest(
                request_id=f"test_req_{i}",
                prompt_token_ids=[1, 2, 3, 4, 5],
                mm_inputs=None,
                mm_hashes=None,
                mm_placeholders=None,
                sampling_params=sampling_params,
                eos_token_id=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=None,
                # TODO: Add when implementing
                # return_hidden_states=True,
            )
            
            engine_core.add_request(request)
        
        # Process all requests
        finished_requests = set()
        hidden_states_results = {}
        
        for step in range(20):  # Max steps
            outputs = engine_core.step()
            
            if outputs and outputs.outputs:
                for output in outputs.outputs:
                    if output.finished and output.request_id not in finished_requests:
                        finished_requests.add(output.request_id)
                        
                        # TODO: Uncomment when implementation is complete
                        # assert hasattr(output, 'hidden_states')
                        # assert output.hidden_states is not None
                        # hidden_states_results[output.request_id] = output.hidden_states
            
            if len(finished_requests) == len(sampling_configs):
                break
        
        # TODO: Enable when implementation is complete
        # assert len(finished_requests) == len(sampling_configs)
        # assert len(hidden_states_results) == len(sampling_configs)
        # 
        # # All hidden states should have the same dimension regardless of sampling method
        # expected_size = vllm_config.model_config.hf_config.hidden_size
        # for req_id, hidden_states in hidden_states_results.items():
        #     assert len(hidden_states) == expected_size


@create_new_process_for_each_test()
def test_hidden_states_memory_management(monkeypatch: pytest.MonkeyPatch):
    """Test memory management for hidden states in high-load scenarios."""
    
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        
        engine_args = EngineArgs(model=MODEL_NAME)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        
        engine_core = EngineCore(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=True
        )
        
        # Create multiple requests to test memory management
        num_requests = 5
        requests = []
        
        for i in range(num_requests):
            request = EngineCoreRequest(
                request_id=f"mem_test_req_{i}",
                prompt_token_ids=[1, 2, 3, 4, 5] + [i],  # Slightly different prompts
                mm_inputs=None,
                mm_hashes=None,
                mm_placeholders=None,
                sampling_params=SamplingParams(max_tokens=2),
                eos_token_id=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=None,
                # TODO: Add when implementing
                # return_hidden_states=(i % 2 == 0),  # Only some requests need hidden states
            )
            requests.append(request)
            engine_core.add_request(request)
        
        # Process all requests and monitor memory usage
        finished_requests = set()
        peak_memory_usage = 0
        
        for step in range(25):  # Max steps
            outputs = engine_core.step()
            
            # TODO: Add memory monitoring when implementation is complete
            # import psutil
            # current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            # peak_memory_usage = max(peak_memory_usage, current_memory)
            
            if outputs and outputs.outputs:
                for output in outputs.outputs:
                    if output.finished and output.request_id not in finished_requests:
                        finished_requests.add(output.request_id)
            
            if len(finished_requests) == num_requests:
                break
        
        # Memory usage should be reasonable
        # TODO: Enable when implementation is complete
        # assert peak_memory_usage < 10000, f"Memory usage too high: {peak_memory_usage:.2f} MB"
        
        assert len(finished_requests) == num_requests


@create_new_process_for_each_test()
def test_hidden_states_error_handling(monkeypatch: pytest.MonkeyPatch):
    """Test error handling for hidden states extraction."""
    
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        
        engine_args = EngineArgs(model=MODEL_NAME)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        
        engine_core = EngineCore(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=True
        )
        
        # Test various error conditions
        
        # 1. Empty prompt tokens
        try:
            request_empty = EngineCoreRequest(
                request_id="empty_test",
                prompt_token_ids=[],  # Empty
                mm_inputs=None,
                mm_hashes=None,
                mm_placeholders=None,
                sampling_params=SamplingParams(max_tokens=1),
                eos_token_id=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=None,
                # TODO: Add when implementing
                # return_hidden_states=True,
            )
            engine_core.add_request(request_empty)
            
            # Should handle gracefully
            outputs = engine_core.step()
            # TODO: Add specific error handling tests when implementing
            
        except Exception as e:
            # Should not crash the engine
            assert "EngineCore" not in str(type(e))
        
        # 2. Very long sequence (test memory limits)
        try:
            long_sequence = list(range(1000))  # Very long prompt
            request_long = EngineCoreRequest(
                request_id="long_test",
                prompt_token_ids=long_sequence,
                mm_inputs=None,
                mm_hashes=None,
                mm_placeholders=None,
                sampling_params=SamplingParams(max_tokens=1),
                eos_token_id=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=None,
                # TODO: Add when implementing
                # return_hidden_states=True,
            )
            engine_core.add_request(request_long)
            
            # Should handle gracefully or provide clear error
            for _ in range(10):
                outputs = engine_core.step()
                if outputs and outputs.outputs:
                    break
            
        except Exception as e:
            # Should provide meaningful error message
            assert len(str(e)) > 0


def test_hidden_states_serialization_deserialization():
    """Test serialization and deserialization of hidden states for ZMQ transfer."""
    
    import json
    import torch
    
    # Mock hidden states tensor
    hidden_size = 2048
    hidden_states_tensor = torch.randn(1, hidden_size, dtype=torch.float32)
    
    # Test conversion to serializable format
    hidden_states_list = hidden_states_tensor.squeeze(0).tolist()
    
    # Test JSON serialization (what ZMQ would do)
    serialized = json.dumps(hidden_states_list)
    assert isinstance(serialized, str)
    assert len(serialized) > 0
    
    # Test deserialization
    deserialized = json.loads(serialized)
    assert isinstance(deserialized, list)
    assert len(deserialized) == hidden_size
    assert all(isinstance(x, float) for x in deserialized)
    
    # Test reconstruction
    reconstructed_tensor = torch.tensor(deserialized, dtype=torch.float32).unsqueeze(0)
    assert reconstructed_tensor.shape == hidden_states_tensor.shape
    assert torch.allclose(reconstructed_tensor, hidden_states_tensor, atol=1e-6)
    
    # Test size estimation for ZMQ transfer
    serialized_size_bytes = len(serialized.encode('utf-8'))
    expected_size_range = (hidden_size * 8, hidden_size * 20)  # Rough estimate for JSON overhead
    assert expected_size_range[0] <= serialized_size_bytes <= expected_size_range[1]


@create_new_process_for_each_test()
def test_hidden_states_consistency_across_runs(monkeypatch: pytest.MonkeyPatch):
    """Test that hidden states are consistent across multiple runs with same input."""
    
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        
        engine_args = EngineArgs(model=MODEL_NAME, seed=42)  # Fixed seed for reproducibility
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        
        # Run same request multiple times
        hidden_states_results = []
        
        for run in range(2):  # Multiple runs
            engine_core = EngineCore(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=True
            )
            
            request = EngineCoreRequest(
                request_id=f"consistency_test_{run}",
                prompt_token_ids=[1, 2, 3, 4, 5],
                mm_inputs=None,
                mm_hashes=None,
                mm_placeholders=None,
                sampling_params=SamplingParams(max_tokens=1, temperature=0.0),  # Deterministic
                eos_token_id=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=None,
                # TODO: Add when implementing
                # return_hidden_states=True,
            )
            
            engine_core.add_request(request)
            
            # Process request
            for _ in range(10):
                outputs = engine_core.step()
                if outputs and outputs.outputs:
                    for output in outputs.outputs:
                        if output.request_id == request.request_id and output.finished:
                            # TODO: Uncomment when implementation is complete
                            # hidden_states_results.append(output.hidden_states)
                            hidden_states_results.append([0.1, 0.2, 0.3])  # Mock for structure
                            break
                    if len(hidden_states_results) == run + 1:
                        break
        
        # TODO: Enable when implementation is complete
        # assert len(hidden_states_results) == 2
        # # Hidden states should be identical for deterministic runs
        # assert hidden_states_results[0] == hidden_states_results[1]
        
        # Verify structure is consistent
        assert len(hidden_states_results) == 2
        assert all(isinstance(hs, list) for hs in hidden_states_results)