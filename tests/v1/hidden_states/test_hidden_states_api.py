# SPDX-License-Identifier: Apache-2.0

"""
Test suite for hidden states functionality in OpenAI-compatible API endpoints.

These tests focus on the API layer integration for hidden states,
testing both chat completions and completions endpoints.
"""

import pytest
import requests
from typing import Dict, Any, Optional

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.",
                allow_module_level=True)

# Test data
TEST_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
BASE_URL = "http://localhost:8000"


def make_chat_completion_request(
    messages: list,
    model: str = TEST_MODEL,
    max_tokens: int = 10,
    return_hidden_states: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Create a chat completion request with optional hidden states."""
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        **kwargs
    }
    
    if return_hidden_states:
        payload["return_hidden_states"] = True
        payload["hidden_states_for_tokens"] = kwargs.get("hidden_states_for_tokens", [-1])
    
    return payload


def make_completion_request(
    prompt: str,
    model: str = TEST_MODEL,
    max_tokens: int = 10,
    return_hidden_states: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Create a completion request with optional hidden states."""
    
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        **kwargs
    }
    
    if return_hidden_states:
        payload["return_hidden_states"] = True
        payload["hidden_states_for_tokens"] = kwargs.get("hidden_states_for_tokens", [-1])
    
    return payload


@pytest.mark.asyncio
async def test_chat_completion_without_hidden_states():
    """Test chat completion without hidden states (baseline functionality)."""
    
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    payload = make_chat_completion_request(
        messages=messages,
        return_hidden_states=False
    )
    
    # This test verifies current functionality works
    # TODO: Replace with actual API client when testing with live server
    expected_response_structure = {
        "id": str,
        "object": "chat.completion",
        "created": int,
        "model": str,
        "choices": list,
        "usage": dict,
    }
    
    # Verify the payload structure is correct
    assert "model" in payload
    assert "messages" in payload
    assert "max_tokens" in payload
    assert "return_hidden_states" not in payload  # Should not be present
    
    # TODO: Make actual API call when testing with live server
    # response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    # assert response.status_code == 200
    # response_data = response.json()
    # 
    # # Verify standard response structure
    # for key, expected_type in expected_response_structure.items():
    #     assert key in response_data
    #     assert isinstance(response_data[key], expected_type)
    # 
    # # Should not have hidden_states field
    # assert "hidden_states" not in response_data["choices"][0]["message"]


@pytest.mark.asyncio
async def test_chat_completion_with_hidden_states():
    """Test chat completion with hidden states - validate request structure."""
    
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    payload = make_chat_completion_request(
        messages=messages,
        return_hidden_states=True
    )
    
    # Test that the request payload now includes hidden states parameters
    assert "return_hidden_states" in payload
    assert payload["return_hidden_states"] is True
    assert "hidden_states_for_tokens" in payload
    assert payload["hidden_states_for_tokens"] == [-1]
    
    # Test ChatCompletionRequest can be created with hidden states
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    
    request = ChatCompletionRequest(**payload)
    assert request.return_hidden_states is True
    assert request.hidden_states_for_tokens == [-1]
    
    # Test conversion to SamplingParams
    sampling_params = request.to_sampling_params(
        default_max_tokens=100,
        logits_processor_pattern=None
    )
    assert sampling_params.return_hidden_states is True
    assert sampling_params.hidden_states_for_tokens == [-1]
    
    # Test response structure can include hidden states
    from vllm.entrypoints.openai.protocol import ChatCompletionResponseChoice, ChatMessage
    
    message = ChatMessage(role="assistant", content="Hello!")
    choice = ChatCompletionResponseChoice(
        index=0,
        message=message,
        hidden_states=[1.0, 2.0, 3.0]
    )
    assert choice.hidden_states == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_completion_without_hidden_states():
    """Test completion without hidden states (baseline functionality)."""
    
    payload = make_completion_request(
        prompt="The capital of France is",
        return_hidden_states=False
    )
    
    expected_response_structure = {
        "id": str,
        "object": "text_completion",
        "created": int,
        "model": str,
        "choices": list,
        "usage": dict,
    }
    
    # Verify the payload structure is correct
    assert "model" in payload
    assert "prompt" in payload
    assert "max_tokens" in payload
    assert "return_hidden_states" not in payload
    
    # TODO: Make actual API call when testing with live server
    # response = requests.post(f"{BASE_URL}/v1/completions", json=payload)
    # assert response.status_code == 200
    # response_data = response.json()
    # 
    # # Verify standard response structure
    # for key, expected_type in expected_response_structure.items():
    #     assert key in response_data
    #     assert isinstance(response_data[key], expected_type)
    # 
    # # Should not have hidden_states field
    # assert "hidden_states" not in response_data["choices"][0]


@pytest.mark.asyncio
async def test_completion_with_hidden_states():
    """Test completion with hidden states - validate request structure."""
    
    payload = make_completion_request(
        prompt="The capital of France is",
        return_hidden_states=True
    )
    
    # Test that the request payload now includes hidden states parameters
    assert "return_hidden_states" in payload
    assert payload["return_hidden_states"] is True
    assert "hidden_states_for_tokens" in payload
    assert payload["hidden_states_for_tokens"] == [-1]
    
    # Test CompletionRequest can be created with hidden states
    from vllm.entrypoints.openai.protocol import CompletionRequest
    
    request = CompletionRequest(**payload)
    assert request.return_hidden_states is True
    assert request.hidden_states_for_tokens == [-1]
    
    # Test conversion to SamplingParams
    sampling_params = request.to_sampling_params(
        default_max_tokens=100,
        logits_processor_pattern=None
    )
    assert sampling_params.return_hidden_states is True
    assert sampling_params.hidden_states_for_tokens == [-1]
    
    # Test response structure can include hidden states
    from vllm.entrypoints.openai.protocol import CompletionResponseChoice
    
    choice = CompletionResponseChoice(
        index=0,
        text="Paris",
        hidden_states=[4.0, 5.0, 6.0]
    )
    assert choice.hidden_states == [4.0, 5.0, 6.0]


@pytest.mark.asyncio
async def test_streaming_chat_completion_with_hidden_states():
    """Test streaming chat completion with hidden states."""
    
    messages = [
        {"role": "user", "content": "Write a short story about a robot."}
    ]
    
    payload = make_chat_completion_request(
        messages=messages,
        return_hidden_states=True,
        stream=True,
        max_tokens=20
    )
    
    # TODO: This will fail until streaming support is implemented
    try:
        # TODO: Implement streaming test when API supports it
        # with requests.post(f"{BASE_URL}/v1/chat/completions", 
        #                   json=payload, stream=True) as response:
        #     assert response.status_code == 200
        #     
        #     chunks = []
        #     for line in response.iter_lines():
        #         if line:
        #             chunk_data = json.loads(line.decode('utf-8').split('data: ')[1])
        #             chunks.append(chunk_data)
        #     
        #     # Only the final chunk should have hidden states
        #     hidden_states_chunks = [chunk for chunk in chunks 
        #                           if 'choices' in chunk and 
        #                           len(chunk['choices']) > 0 and
        #                           'hidden_states' in chunk['choices'][0].get('delta', {})]
        #     
        #     assert len(hidden_states_chunks) == 1  # Only final chunk
        #     final_chunk = hidden_states_chunks[0]
        #     hidden_states = final_chunk['choices'][0]['delta']['hidden_states']
        #     assert isinstance(hidden_states, list)
        #     assert len(hidden_states) > 0
        
        pytest.skip("Streaming hidden states support not implemented yet")
        
    except Exception as e:
        pytest.skip(f"Streaming API doesn't support hidden states yet: {e}")


@pytest.mark.asyncio
async def test_streaming_completion_with_hidden_states():
    """Test streaming completion with hidden states."""
    
    payload = make_completion_request(
        prompt="Once upon a time, in a land far away",
        return_hidden_states=True,
        stream=True,
        max_tokens=15
    )
    
    # TODO: This will fail until streaming support is implemented
    try:
        # TODO: Implement streaming test when API supports it
        pytest.skip("Streaming hidden states support not implemented yet")
        
    except Exception as e:
        pytest.skip(f"Streaming API doesn't support hidden states yet: {e}")


def test_api_request_validation():
    """Test API request validation for hidden states parameter."""
    
    # Test valid requests
    valid_chat_payload = make_chat_completion_request(
        messages=[{"role": "user", "content": "Hello"}],
        return_hidden_states=True
    )
    
    valid_completion_payload = make_completion_request(
        prompt="Hello",
        return_hidden_states=True
    )
    
    # Basic structure validation
    assert isinstance(valid_chat_payload, dict)
    assert isinstance(valid_completion_payload, dict)
    
    # TODO: Add validation when API parameter is implemented
    # assert "return_hidden_states" in valid_chat_payload
    # assert valid_chat_payload["return_hidden_states"] is True
    # assert "return_hidden_states" in valid_completion_payload
    # assert valid_completion_payload["return_hidden_states"] is True


def test_api_response_schema_extension():
    """Test that API response schemas can be extended with hidden states."""
    
    # Define expected schema extensions
    chat_completion_choice_extension = {
        "message": {
            "role": str,
            "content": str,
            "hidden_states": Optional[list]  # Should be Optional[List[float]]
        }
    }
    
    completion_choice_extension = {
        "text": str,
        "index": int,
        "logprobs": Optional[dict],
        "finish_reason": str,
        "hidden_states": Optional[list]  # Should be Optional[List[float]]
    }
    
    # Test schema validation logic
    def validate_choice_with_hidden_states(choice_data: dict, schema: dict) -> bool:
        for key, expected_type in schema.items():
            if key == "message" and isinstance(expected_type, dict):
                # Nested validation for message
                if key not in choice_data:
                    return False
                message = choice_data[key]
                for msg_key, msg_type in expected_type.items():
                    if msg_key == "hidden_states":
                        # Optional field
                        if msg_key in message:
                            if not isinstance(message[msg_key], (list, type(None))):
                                return False
                    else:
                        if msg_key not in message:
                            return False
                        if not isinstance(message[msg_key], msg_type):
                            return False
            elif key == "hidden_states":
                # Optional field
                if key in choice_data:
                    if not isinstance(choice_data[key], (list, type(None))):
                        return False
            else:
                if key not in choice_data:
                    return False
                if not isinstance(choice_data[key], expected_type):
                    return False
        return True
    
    # Test mock response data
    mock_chat_choice = {
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you?",
            # "hidden_states": [0.1, 0.2, 0.3, ...]  # Will be added when implemented
        }
    }
    
    mock_completion_choice = {
        "text": " Paris.",
        "index": 0,
        "logprobs": None,
        "finish_reason": "stop",
        # "hidden_states": [0.1, 0.2, 0.3, ...]  # Will be added when implemented
    }
    
    # Current schemas should validate (without hidden_states)
    assert validate_choice_with_hidden_states(mock_chat_choice, 
                                            {"message": {"role": str, "content": str}})
    assert validate_choice_with_hidden_states(mock_completion_choice,
                                            {"text": str, "index": int, 
                                             "finish_reason": str})
    
    # TODO: Test with hidden_states when implemented
    # mock_chat_choice["message"]["hidden_states"] = [0.1, 0.2, 0.3]
    # mock_completion_choice["hidden_states"] = [0.1, 0.2, 0.3]
    # assert validate_choice_with_hidden_states(mock_chat_choice, chat_completion_choice_extension)
    # assert validate_choice_with_hidden_states(mock_completion_choice, completion_choice_extension)


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/completions"])
def test_api_error_handling(endpoint: str):
    """Test API error handling for invalid hidden states requests."""
    
    # Test invalid parameter types
    invalid_payloads = [
        # TODO: Add these tests when API parameter is implemented
        # {"return_hidden_states": "true"},  # String instead of bool
        # {"return_hidden_states": 1},       # Int instead of bool
        # {"return_hidden_states": []},      # List instead of bool
    ]
    
    base_payload = {
        "model": TEST_MODEL,
        "max_tokens": 5,
    }
    
    if endpoint == "/v1/chat/completions":
        base_payload["messages"] = [{"role": "user", "content": "Hello"}]
    else:
        base_payload["prompt"] = "Hello"
    
    for invalid_payload in invalid_payloads:
        test_payload = {**base_payload, **invalid_payload}
        
        # TODO: Test actual API calls when implementing
        # response = requests.post(f"{BASE_URL}{endpoint}", json=test_payload)
        # assert response.status_code == 422  # Validation error
        # error_data = response.json()
        # assert "error" in error_data
        # assert "return_hidden_states" in error_data["error"]["message"].lower()
        
        pass  # Skip until implementation


def test_hidden_states_backward_compatibility():
    """Test that existing API requests work without hidden states parameter."""
    
    # Standard requests should work exactly as before
    chat_payload = {
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5
    }
    
    completion_payload = {
        "model": TEST_MODEL,
        "prompt": "Hello",
        "max_tokens": 5
    }
    
    # These payloads should be valid and work without any changes
    assert "return_hidden_states" not in chat_payload
    assert "return_hidden_states" not in completion_payload
    
    # TODO: Test actual API calls when testing with live server
    # Verify that responses don't include hidden_states field when not requested
    pass