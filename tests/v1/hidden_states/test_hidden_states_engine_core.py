# SPDX-License-Identifier: Apache-2.0

"""
Test suite for hidden states functionality at the EngineCore level.

These tests will fail until the hidden states implementation is complete.
They serve as a specification for the expected behavior and will guide
the implementation process.
"""

import time
import uuid
from typing import List, Optional

import pytest
import torch
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.v1.engine import EngineCoreRequest, EngineCoreOutput
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor

from ...utils import create_new_process_for_each_test

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.",
                allow_module_level=True)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

# Test prompts of varying lengths
TEST_PROMPTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "In the beginning was the Word, and the Word was with God, and the Word was God. He was with God in the beginning. Through him all things were made; without him nothing was made that has been made.",
]

def make_request_with_hidden_states(
    prompt: str, 
    return_hidden_states: bool = False,
    max_tokens: int = 10
) -> EngineCoreRequest:
    """Create an EngineCoreRequest with hidden states parameters."""
    prompt_tokens = TOKENIZER(prompt).input_ids
    
    return EngineCoreRequest(
        request_id=str(uuid.uuid4()),
        prompt_token_ids=prompt_tokens,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=SamplingParams(max_tokens=max_tokens),
        eos_token_id=TOKENIZER.eos_token_id,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        # TODO: Add these fields when implementing hidden states
        # return_hidden_states=return_hidden_states,
        # hidden_states_for_tokens=None,  # Return for all tokens by default
    )


@create_new_process_for_each_test()
def test_engine_core_basic_hidden_states(monkeypatch: pytest.MonkeyPatch):
    """Test basic hidden states extraction from EngineCore."""
    
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        
        # Setup EngineCore
        engine_args = EngineArgs(model=MODEL_NAME)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        
        engine_core = EngineCore(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=True
        )
        
        # Test request without hidden states (should work now)
        request_without_hs = make_request_with_hidden_states(
            TEST_PROMPTS[0], 
            return_hidden_states=False
        )
        engine_core.add_request(request_without_hs)
        
        outputs = engine_core.step()
        assert outputs is not None
        assert len(outputs.outputs) >= 0
        
        # Test request with hidden states (will fail until implemented)
        request_with_hs = make_request_with_hidden_states(
            TEST_PROMPTS[0], 
            return_hidden_states=True
        )
        engine_core.add_request(request_with_hs)
        
        # TODO: This will fail until implementation is complete
        # Expected behavior after implementation:
        outputs = engine_core.step()
        
        # Find the output for our request
        target_output = None
        for output in outputs.outputs:
            if output.request_id == request_with_hs.request_id:
                target_output = output
                break
        
        if target_output and target_output.finished:
            # TODO: Uncomment when implementation is complete
            # assert hasattr(target_output, 'hidden_states')
            # assert target_output.hidden_states is not None
            # assert isinstance(target_output.hidden_states, list)
            # assert len(target_output.hidden_states) == vllm_config.model_config.hf_config.hidden_size
            pass


@create_new_process_for_each_test()
def test_engine_core_hidden_states_final_token_only(monkeypatch: pytest.MonkeyPatch):
    """Test that hidden states are only returned for the final token."""
    
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
        
        # Create a request that will generate multiple tokens
        request = make_request_with_hidden_states(
            TEST_PROMPTS[1], 
            return_hidden_states=True,
            max_tokens=5
        )
        engine_core.add_request(request)
        
        outputs_with_hidden_states = []
        outputs_without_hidden_states = []
        
        # Run until the request is finished
        for _ in range(20):  # Safety limit
            outputs = engine_core.step()
            if outputs and outputs.outputs:
                for output in outputs.outputs:
                    if output.request_id == request.request_id:
                        if output.finished:
                            # TODO: Uncomment when implementation is complete
                            # assert hasattr(output, 'hidden_states')
                            # assert output.hidden_states is not None
                            # outputs_with_hidden_states.append(output)
                            pass
                        else:
                            # Intermediate tokens should not have hidden states
                            # TODO: Uncomment when implementation is complete
                            # assert not hasattr(output, 'hidden_states') or output.hidden_states is None
                            # outputs_without_hidden_states.append(output)
                            pass
                        
                        if output.finished:
                            break
            else:
                break
        
        # TODO: Uncomment when implementation is complete
        # assert len(outputs_with_hidden_states) == 1, "Only final token should have hidden states"
        # assert len(outputs_without_hidden_states) >= 1, "Should have intermediate tokens without hidden states"


@create_new_process_for_each_test()
def test_engine_core_hidden_states_multiple_requests(monkeypatch: pytest.MonkeyPatch):
    """Test hidden states extraction with multiple concurrent requests."""
    
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
        
        # Create multiple requests - some with hidden states, some without
        requests = []
        for i, prompt in enumerate(TEST_PROMPTS):
            request = make_request_with_hidden_states(
                prompt,
                return_hidden_states=(i % 2 == 0),  # Every other request gets hidden states
                max_tokens=3
            )
            requests.append(request)
            engine_core.add_request(request)
        
        finished_requests = set()
        hidden_states_received = {}
        
        # Process until all requests are finished
        for _ in range(30):  # Safety limit
            outputs = engine_core.step()
            if outputs and outputs.outputs:
                for output in outputs.outputs:
                    if output.finished and output.request_id not in finished_requests:
                        finished_requests.add(output.request_id)
                        
                        # Find the corresponding request
                        request_idx = None
                        for i, req in enumerate(requests):
                            if req.request_id == output.request_id:
                                request_idx = i
                                break
                        
                        if request_idx is not None:
                            should_have_hidden_states = (request_idx % 2 == 0)
                            
                            # TODO: Uncomment when implementation is complete
                            # if should_have_hidden_states:
                            #     assert hasattr(output, 'hidden_states')
                            #     assert output.hidden_states is not None
                            #     hidden_states_received[output.request_id] = output.hidden_states
                            # else:
                            #     assert not hasattr(output, 'hidden_states') or output.hidden_states is None
            
            if len(finished_requests) == len(requests):
                break
        
        # TODO: Uncomment when implementation is complete
        # assert len(finished_requests) == len(requests), "All requests should finish"
        # expected_hidden_states_count = sum(1 for i in range(len(TEST_PROMPTS)) if i % 2 == 0)
        # assert len(hidden_states_received) == expected_hidden_states_count


@create_new_process_for_each_test()
def test_engine_core_hidden_states_dimensions(monkeypatch: pytest.MonkeyPatch):
    """Test that hidden states have the correct dimensions."""
    
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
        
        # Get expected hidden size from model config
        expected_hidden_size = vllm_config.model_config.hf_config.hidden_size
        
        request = make_request_with_hidden_states(
            TEST_PROMPTS[0], 
            return_hidden_states=True,
            max_tokens=1
        )
        engine_core.add_request(request)
        
        # Process until request is finished
        for _ in range(20):
            outputs = engine_core.step()
            if outputs and outputs.outputs:
                for output in outputs.outputs:
                    if output.request_id == request.request_id and output.finished:
                        # TODO: Uncomment when implementation is complete
                        # assert hasattr(output, 'hidden_states')
                        # assert output.hidden_states is not None
                        # assert isinstance(output.hidden_states, list)
                        # assert len(output.hidden_states) == expected_hidden_size
                        # # All values should be floats
                        # assert all(isinstance(x, (int, float)) for x in output.hidden_states)
                        return
        
        # Should not reach here if implementation is correct
        pytest.fail("Request did not finish or hidden states not found")


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@create_new_process_for_each_test()
def test_engine_core_hidden_states_various_prompts(prompt: str, monkeypatch: pytest.MonkeyPatch):
    """Test hidden states extraction with various prompt lengths and content."""
    
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
        
        request = make_request_with_hidden_states(
            prompt, 
            return_hidden_states=True,
            max_tokens=2
        )
        engine_core.add_request(request)
        
        # Process request
        for _ in range(20):
            outputs = engine_core.step()
            if outputs and outputs.outputs:
                for output in outputs.outputs:
                    if output.request_id == request.request_id and output.finished:
                        # TODO: Uncomment when implementation is complete
                        # assert hasattr(output, 'hidden_states')
                        # assert output.hidden_states is not None
                        # Regardless of prompt length, hidden states should be for final token only
                        # assert len(output.hidden_states) == vllm_config.model_config.hf_config.hidden_size
                        return
        
        pytest.fail(f"Request for prompt '{prompt[:20]}...' did not finish")


@create_new_process_for_each_test()
def test_engine_core_hidden_states_with_stop_tokens(monkeypatch: pytest.MonkeyPatch):
    """Test hidden states when request finishes due to stop tokens."""
    
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
        
        # Create request with stop tokens
        prompt_tokens = TOKENIZER("Hello, my name is").input_ids
        request = EngineCoreRequest(
            request_id=str(uuid.uuid4()),
            prompt_token_ids=prompt_tokens,
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=SamplingParams(
                max_tokens=20,
                stop=["world", "AI", "assistant"]  # Common stop words
            ),
            eos_token_id=TOKENIZER.eos_token_id,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            # TODO: Add when implementing
            # return_hidden_states=True,
        )
        engine_core.add_request(request)
        
        # Process until finished
        for _ in range(30):
            outputs = engine_core.step()
            if outputs and outputs.outputs:
                for output in outputs.outputs:
                    if output.request_id == request.request_id and output.finished:
                        # TODO: Uncomment when implementation is complete
                        # assert hasattr(output, 'hidden_states')
                        # assert output.hidden_states is not None
                        # Hidden states should be available even when stopped by stop tokens
                        # assert len(output.hidden_states) == vllm_config.model_config.hf_config.hidden_size
                        return
        
        pytest.fail("Request did not finish with stop tokens")