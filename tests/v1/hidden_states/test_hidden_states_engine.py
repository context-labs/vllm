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
import pytest

# Set V1 engine flag
os.environ["VLLM_USE_V1"] = "1"

model_dir = "meta-llama/Llama-3.1-8B-Instruct"
eagle_dir = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
eagle3_dir = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


def _test_hidden_states(llm, prompts, n = 1):
    sampling_params = vllm.SamplingParams(temperature=1,
                                          n=n,
                                          return_hidden_states=True,
                                          hidden_states_for_tokens=[-1],
                                          max_tokens=10)

    outputs = llm.generate(
        prompts,
        sampling_params)

    _assert_hidden_states(outputs)

def _assert_hidden_states(outputs):
    for i,output in enumerate(outputs):
        print("Output:")
        hidden_states = getattr(output, "hidden_states", None)
        assert hidden_states is not None, "Engine output missing hidden_states"    

def _assert_no_hidden_states(outputs):
    for i,output in enumerate(outputs):
        hidden_states = getattr(output, "hidden_states", None)
        assert hidden_states is None, "Engine output should not have hidden_states"


def test_no_hidden_states_when_not_requested():
    llm = vllm.LLM(
        model=model_dir,
        max_model_len=400,
        quantization=None,
        trust_remote_code=True,
        enable_chunked_prefill=True)

    prompts = ["What is the meaning of life? Respond with an essay."]

    sampling_params = vllm.SamplingParams(temperature=1,
                                          n=1,
                                          max_tokens=1) 

    outputs = llm.generate(prompts, sampling_params)

    _assert_no_hidden_states(outputs)

# todo: test that requesting hidden states without enabling server arg -> error
# todo: test that default hidden states position is -1 

def test_last_token_with_truncated_response():
    
    llm = vllm.LLM(
        model=model_dir,
        max_model_len=400,
        trust_remote_code=True)

    prompts = ["What is the meaning of life? Respond with an essay."]

    sampling_params = vllm.SamplingParams(temperature=1,
                                          n=1,
                                          max_tokens=1,
                                          return_hidden_states=True,
                                          hidden_states_for_tokens=[-1]) 

    outputs = llm.generate(prompts, sampling_params)

    for i,output in enumerate(outputs):
        hidden_states = getattr(output, "hidden_states", None)
        assert hidden_states is not None, "Engine output missing hidden_states"                                               

def test_last_token_hidden_states_engine_request():
    """Test retrieving hidden states via an actual engine call."""
    print("Testing actual engine hidden states extraction via actual engine call...")

    llm = vllm.LLM(
        model=model_dir,
        max_model_len=400,
        trust_remote_code=True)

    _test_hidden_states(llm, ["The capital of France is"])

def test_last_token_hidden_states_multiple_prompts():
    """Test retrieving hidden states via parallel sampling."""
    print("Testing parallel sampling hidden states extraction...")

    llm = vllm.LLM(
        model=model_dir,
        max_model_len=400,
        trust_remote_code=True)

    prompts = ["The capital of France is", "The capital of Spain is"]

    _test_hidden_states(llm, prompts)

def test_last_token_hidden_states_parallel_sampling():
    """Test retrieving hidden states via parallel sampling."""
    print("Testing parallel sampling hidden states extraction...")

    llm = vllm.LLM(
        model=model_dir,
        max_model_len=400,
        trust_remote_code=True)

    _test_hidden_states(llm, ["The capital of France is"], n = 2)



@pytest.mark.skip(reason="Speculative decoding not implemented for v1")
def test_hidden_states_with_eagle():
    llm = vllm.LLM(
        model=model_dir,
        max_model_len=400,
        trust_remote_code=True,
        speculative_config={
            "model": eagle_dir,
            "draft_tensor_parallel_size": 1,
        })

    prompts = ["What is the meaning of life?"]

    _test_hidden_states(llm, prompts)

def test_hidden_states_enforce_eager():
    llm = vllm.LLM(
        model=model_dir,
        max_model_len=400,
        trust_remote_code=True,
        enforce_eager=True)

    prompts = ["The capital of France is"]

    _test_hidden_states(llm, prompts)


def main():
    test_no_hidden_states_when_not_requested()
    test_last_token_with_truncated_response()
    test_last_token_hidden_states_engine_request()
    test_last_token_hidden_states_multiple_prompts()
    test_last_token_hidden_states_parallel_sampling()
    test_hidden_states_enforce_eager()

if __name__ == "__main__":
    sys.exit(main())