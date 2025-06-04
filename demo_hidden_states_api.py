#!/usr/bin/env python3
"""
Demo script showing vLLM Hidden States API structure and usage

This script demonstrates the API request/response structures without requiring a running server.
It shows how to construct requests and what the responses look like.

Usage:
    python demo_hidden_states_api.py
"""

import json
from typing import Dict, Any

def demo_chat_completion_request() -> Dict[str, Any]:
    """Demonstrate chat completion request with hidden states."""
    
    print("🚀 Chat Completion Request with Hidden States")
    print("=" * 50)
    
    # Standard request without hidden states
    standard_request = {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 10,
        "temperature": 0.7
    }
    
    print("📤 Standard Request (without hidden states):")
    print(json.dumps(standard_request, indent=2))
    print()
    
    # Request with hidden states
    hidden_states_request = {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 10,
        "temperature": 0.7,
        "return_hidden_states": True,
        "hidden_states_for_tokens": [-1]  # Extract for last token
    }
    
    print("📤 Request with Hidden States:")
    print(json.dumps(hidden_states_request, indent=2))
    print()
    
    # Simulated standard response
    standard_response = {
        "id": "chatcmpl-123456789",
        "object": "chat.completion",
        "created": 1699999999,
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The capital of France is Paris."
                },
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 7,
            "total_tokens": 15
        }
    }
    
    print("📥 Standard Response (without hidden states):")
    print(json.dumps(standard_response, indent=2))
    print()
    
    # Simulated response with hidden states
    hidden_states_response = {
        "id": "chatcmpl-123456789",
        "object": "chat.completion",
        "created": 1699999999,
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The capital of France is Paris."
                },
                "logprobs": None,
                "finish_reason": "stop",
                "hidden_states": [
                    0.1234, -0.5678, 0.9012, -0.3456, 0.7890,
                    -0.2345, 0.6789, -0.4567, 0.8901, 0.2345,
                    # ... (representing 4096-dimensional vector)
                    # "... (4086 more values) ..."
                ]
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 7,
            "total_tokens": 15
        }
    }
    
    # Truncate hidden states for display
    truncated_response = hidden_states_response.copy()
    truncated_response["choices"][0]["hidden_states"] = (
        hidden_states_response["choices"][0]["hidden_states"][:10] + 
        ["... (4086 more values) ..."]
    )
    
    print("📥 Response with Hidden States:")
    print(json.dumps(truncated_response, indent=2))
    print()
    
    return hidden_states_request, hidden_states_response


def demo_completion_request() -> Dict[str, Any]:
    """Demonstrate completion request with hidden states."""
    
    print("🚀 Completion Request with Hidden States")
    print("=" * 50)
    
    # Standard request without hidden states
    standard_request = {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "temperature": 0.7
    }
    
    print("📤 Standard Request (without hidden states):")
    print(json.dumps(standard_request, indent=2))
    print()
    
    # Request with hidden states
    hidden_states_request = {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "temperature": 0.7,
        "return_hidden_states": True,
        "hidden_states_for_tokens": [-1]  # Extract for last token
    }
    
    print("📤 Request with Hidden States:")
    print(json.dumps(hidden_states_request, indent=2))
    print()
    
    # Simulated standard response
    standard_response = {
        "id": "cmpl-123456789",
        "object": "text_completion",
        "created": 1699999999,
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "choices": [
            {
                "index": 0,
                "text": " Paris.",
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 6,
            "completion_tokens": 2,
            "total_tokens": 8
        }
    }
    
    print("📥 Standard Response (without hidden states):")
    print(json.dumps(standard_response, indent=2))
    print()
    
    # Simulated response with hidden states
    hidden_states_response = {
        "id": "cmpl-123456789",
        "object": "text_completion",
        "created": 1699999999,
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "choices": [
            {
                "index": 0,
                "text": " Paris.",
                "logprobs": None,
                "finish_reason": "stop",
                "hidden_states": [
                    0.2468, -0.1357, 0.8024, -0.5791, 0.3146,
                    -0.7913, 0.4680, -0.9257, 0.1835, 0.6429,
                    # ... (representing 4096-dimensional vector)
                ]
            }
        ],
        "usage": {
            "prompt_tokens": 6,
            "completion_tokens": 2,
            "total_tokens": 8
        }
    }
    
    # Truncate hidden states for display
    truncated_response = hidden_states_response.copy()
    truncated_response["choices"][0]["hidden_states"] = (
        hidden_states_response["choices"][0]["hidden_states"][:10] + 
        ["... (4086 more values) ..."]
    )
    
    print("📥 Response with Hidden States:")
    print(json.dumps(truncated_response, indent=2))
    print()
    
    return hidden_states_request, hidden_states_response


def demo_streaming_response() -> None:
    """Demonstrate streaming response with hidden states."""
    
    print("🚀 Streaming Response with Hidden States")
    print("=" * 50)
    
    print("📤 Streaming Request:")
    streaming_request = {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [
            {"role": "user", "content": "Write a short story about a robot."}
        ],
        "max_tokens": 20,
        "temperature": 0.7,
        "stream": True,
        "return_hidden_states": True,
        "hidden_states_for_tokens": [-1]
    }
    print(json.dumps(streaming_request, indent=2))
    print()
    
    print("📥 Streaming Response chunks:")
    print("data: " + json.dumps({
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1699999999,
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "logprobs": None,
            "finish_reason": None
        }]
    }))
    print()
    
    print("data: " + json.dumps({
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1699999999,
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "choices": [{
            "index": 0,
            "delta": {"content": "Once"},
            "logprobs": None,
            "finish_reason": None
        }]
    }))
    print()
    
    print("data: " + json.dumps({
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1699999999,
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "choices": [{
            "index": 0,
            "delta": {"content": " upon"},
            "logprobs": None,
            "finish_reason": None
        }]
    }))
    print()
    
    print("... (more chunks) ...")
    print()
    
    # Final chunk with hidden states
    final_chunk = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1699999999,
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "choices": [{
            "index": 0,
            "delta": {"content": " end."},
            "logprobs": None,
            "finish_reason": "stop",
            "hidden_states": [0.1234, -0.5678, 0.9012, "... (4093 more values) ..."]
        }]
    }
    
    print("data: " + json.dumps(final_chunk))
    print()
    print("data: [DONE]")
    print()


def demo_advanced_features() -> None:
    """Demonstrate advanced hidden states features."""
    
    print("🚀 Advanced Hidden States Features")
    print("=" * 50)
    
    # Multiple token positions
    print("📤 Request for Multiple Token Positions:")
    multi_token_request = {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "prompt": "The quick brown fox jumps over the lazy dog",
        "max_tokens": 5,
        "temperature": 0.7,
        "return_hidden_states": True,
        "hidden_states_for_tokens": [0, 5, 10, -1]  # First, 6th, 11th, and last tokens
    }
    print(json.dumps(multi_token_request, indent=2))
    print()
    
    print("📥 Response with Multiple Hidden States:")
    multi_token_response = {
        "id": "cmpl-123456789",
        "object": "text_completion",
        "created": 1699999999,
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "choices": [
            {
                "index": 0,
                "text": " and runs away.",
                "logprobs": None,
                "finish_reason": "stop",
                "hidden_states": {
                    "0": [0.1, -0.2, 0.3, "... (4093 more values) ..."],     # Token at position 0
                    "5": [0.4, -0.5, 0.6, "... (4093 more values) ..."],     # Token at position 5
                    "10": [0.7, -0.8, 0.9, "... (4093 more values) ..."],    # Token at position 10
                    "-1": [0.2, -0.3, 0.4, "... (4093 more values) ..."]     # Last token
                }
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 4,
            "total_tokens": 13
        }
    }
    print(json.dumps(multi_token_response, indent=2))
    print()


def demo_validation_examples() -> None:
    """Show API validation examples."""
    
    print("🚀 API Validation Examples")
    print("=" * 50)
    
    print("✅ Valid Requests:")
    valid_requests = [
        {
            "description": "Basic hidden states request",
            "request": {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "messages": [{"role": "user", "content": "Hello"}],
                "return_hidden_states": True
            }
        },
        {
            "description": "Hidden states for specific tokens",
            "request": {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "messages": [{"role": "user", "content": "Hello"}],
                "return_hidden_states": True,
                "hidden_states_for_tokens": [0, -1]
            }
        },
        {
            "description": "No hidden states (backward compatible)",
            "request": {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        }
    ]
    
    for example in valid_requests:
        print(f"• {example['description']}:")
        print(f"  {json.dumps(example['request'])}")
        print()
    
    print("❌ Invalid Requests (would return 422 validation error):")
    invalid_requests = [
        {
            "description": "Wrong type for return_hidden_states",
            "request": {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "messages": [{"role": "user", "content": "Hello"}],
                "return_hidden_states": "true"  # Should be boolean
            }
        },
        {
            "description": "Wrong type for hidden_states_for_tokens",
            "request": {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "messages": [{"role": "user", "content": "Hello"}],
                "return_hidden_states": True,
                "hidden_states_for_tokens": "-1"  # Should be list of integers
            }
        }
    ]
    
    for example in invalid_requests:
        print(f"• {example['description']}:")
        print(f"  {json.dumps(example['request'])}")
        print()


def main():
    """Run all demos."""
    
    print("🎯 vLLM Hidden States API Demo")
    print("=" * 60)
    print()
    
    # Basic demos
    demo_chat_completion_request()
    print("\n" + "=" * 60 + "\n")
    
    demo_completion_request()
    print("\n" + "=" * 60 + "\n")
    
    demo_streaming_response()
    print("\n" + "=" * 60 + "\n")
    
    demo_advanced_features()
    print("\n" + "=" * 60 + "\n")
    
    demo_validation_examples()
    print("=" * 60)
    
    print("\n🎉 Demo Complete!")
    print("\n📚 Key Points:")
    print("   • Add 'return_hidden_states': true to enable hidden states extraction")
    print("   • Use 'hidden_states_for_tokens': [-1] to get final token hidden states")
    print("   • Hidden states appear in the 'hidden_states' field of response choices")
    print("   • Supports both chat completions and completions endpoints")
    print("   • Streaming responses include hidden states in the final chunk")
    print("   • Multiple token positions can be specified for extraction")
    print("   • Fully backward compatible - existing requests work unchanged")
    print("\n🚀 To test with a live server:")
    print("   1. Start server: VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.2-1B-Instruct")
    print("   2. Run test: python test_hidden_states_api_client.py")
    print("   3. Or use curl: ./test_hidden_states_curl.sh")


if __name__ == "__main__":
    main()