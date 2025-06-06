#!/usr/bin/env python3
"""
Debug script to test hidden states API integration step by step.
This version starts its own vLLM server with V1 engine.
"""

import os
import sys
import time
import json
import requests
import contextlib
from typing import Dict, Any

# Add the tests directory to the path so we can import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))
from tests.utils import RemoteOpenAIServer

# Test configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

def test_completion_hidden_states(server):
    """Test completion API with hidden states."""
    print("🔍 Testing /v1/completions with hidden states...")
    
    url = server.url_for("v1", "completions")
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "temperature": 0.7,
        "return_hidden_states": True,
        "hidden_states_for_tokens": [-1]  # Last token
    }
    
    print(f"📤 Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📥 Response keys: {list(data.keys())}")
            
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                print(f"🎯 Choice keys: {list(choice.keys())}")
                print(f"📝 Generated text: '{choice.get('text', '')}'")
                
                if "hidden_states" in choice:
                    hidden_states = choice["hidden_states"]
                    if hidden_states is not None:
                        print(f"✅ Hidden states found: type={type(hidden_states)}, length={len(hidden_states) if isinstance(hidden_states, list) else 'N/A'}")
                        if isinstance(hidden_states, list) and len(hidden_states) > 0:
                            print(f"   First few values: {hidden_states[:5]}")
                    else:
                        print("❌ Hidden states field is None")
                else:
                    print("❌ Hidden states field not present")
            else:
                print("❌ No choices in response")
        else:
            print(f"❌ Error response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_chat_completion_hidden_states(server):
    """Test chat completion API with hidden states."""
    print("\n🔍 Testing /v1/chat/completions with hidden states...")
    
    url = server.url_for("v1", "chat/completions")
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 5,
        "temperature": 0.7,
        "return_hidden_states": True,
        "hidden_states_for_tokens": [-1]  # Last token
    }
    
    print(f"📤 Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📥 Response keys: {list(data.keys())}")
            
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                print(f"🎯 Choice keys: {list(choice.keys())}")
                print(f"📝 Generated text: '{choice.get('message', {}).get('content', '')}'")
                
                if "hidden_states" in choice:
                    hidden_states = choice["hidden_states"]
                    if hidden_states is not None:
                        print(f"✅ Hidden states found: type={type(hidden_states)}, length={len(hidden_states) if isinstance(hidden_states, list) else 'N/A'}")
                        if isinstance(hidden_states, list) and len(hidden_states) > 0:
                            print(f"   First few values: {hidden_states[:5]}")
                    else:
                        print("❌ Hidden states field is None")
                else:
                    print("❌ Hidden states field not present")
            else:
                print("❌ No choices in response")
        else:
            print(f"❌ Error response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_completion_streaming_hidden_states(server):
    """Test completion API with streaming and hidden states."""
    print("\n🔍 Testing /v1/completions with streaming and hidden states...")
    
    url = server.url_for("v1", "completions")
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "temperature": 0.7,
        "stream": True,
        "return_hidden_states": True,
        "hidden_states_for_tokens": [-1]  # Last token
    }
    
    print(f"📤 Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            chunks = []
            generated_text = ""
            found_hidden_states = False
            hidden_states_chunk = None
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    chunk_data = line[6:]  # Remove "data: " prefix
                    if chunk_data == "[DONE]":
                        print("📄 Stream finished with [DONE]")
                        break
                    
                    try:
                        chunk = json.loads(chunk_data)
                        chunks.append(chunk)
                        
                        if "choices" in chunk and chunk["choices"]:
                            choice = chunk["choices"][0]
                            if "text" in choice:
                                generated_text += choice["text"]
                            
                            # Check for hidden states in final chunk
                            if choice.get("finish_reason") is not None:
                                print(f"📋 Final chunk finish_reason: {choice['finish_reason']}")
                                if "hidden_states" in choice and choice["hidden_states"] is not None:
                                    found_hidden_states = True
                                    hidden_states_chunk = chunk
                                    print(f"✅ Hidden states found in final chunk: length={len(choice['hidden_states'])}")
                                    print(f"   First few values: {choice['hidden_states'][:5]}")
                                else:
                                    print("❌ No hidden states in final chunk")
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Failed to parse chunk: {e}")
            
            print(f"📝 Complete generated text: '{generated_text}'")
            print(f"📊 Total chunks received: {len(chunks)}")
            if found_hidden_states:
                print("✅ Streaming with hidden states: SUCCESS")
            else:
                print("❌ Streaming with hidden states: FAILED - No hidden states found")
                
        else:
            print(f"❌ Error response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_chat_completion_streaming_hidden_states(server):
    """Test chat completion API with streaming and hidden states."""
    print("\n🔍 Testing /v1/chat/completions with streaming and hidden states...")
    
    url = server.url_for("v1", "chat/completions")
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 5,
        "temperature": 0.7,
        "stream": True,
        "return_hidden_states": True,
        "hidden_states_for_tokens": [-1]  # Last token
    }
    
    print(f"📤 Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            chunks = []
            generated_text = ""
            found_hidden_states = False
            hidden_states_chunk = None
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    chunk_data = line[6:]  # Remove "data: " prefix
                    if chunk_data == "[DONE]":
                        print("📄 Stream finished with [DONE]")
                        break
                    
                    try:
                        chunk = json.loads(chunk_data)
                        chunks.append(chunk)
                        
                        if "choices" in chunk and chunk["choices"]:
                            choice = chunk["choices"][0]
                            if "delta" in choice and "content" in choice["delta"]:
                                if choice["delta"]["content"]:
                                    generated_text += choice["delta"]["content"]
                            
                            # Check for hidden states in final chunk
                            if choice.get("finish_reason") is not None:
                                print(f"📋 Final chunk finish_reason: {choice['finish_reason']}")
                                delta = choice.get("delta", {})
                                if "hidden_states" in delta and delta["hidden_states"] is not None:
                                    found_hidden_states = True
                                    hidden_states_chunk = chunk
                                    print(f"✅ Hidden states found in final chunk delta: length={len(delta['hidden_states'])}")
                                    print(f"   First few values: {delta['hidden_states'][:5]}")
                                else:
                                    print("❌ No hidden states in final chunk delta")
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Failed to parse chunk: {e}")
            
            print(f"📝 Complete generated text: '{generated_text}'")
            print(f"📊 Total chunks received: {len(chunks)}")
            if found_hidden_states:
                print("✅ Chat streaming with hidden states: SUCCESS")
            else:
                print("❌ Chat streaming with hidden states: FAILED - No hidden states found")
                
        else:
            print(f"❌ Error response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_streaming_parallel_sampling(server):
    """Test streaming with parallel sampling (n>1) and hidden states."""
    print("\n🔍 Testing streaming with parallel sampling (n=2) and hidden states...")
    
    url = server.url_for("v1", "completions")
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "prompt": "The capital of France is",
        "max_tokens": 3,
        "temperature": 0.8,
        "n": 2,  # Parallel sampling
        "stream": True,
        "return_hidden_states": True,
        "hidden_states_for_tokens": [-1]  # Last token
    }
    
    print(f"📤 Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            chunks = []
            choice_texts = {}  # Track text per choice index
            choice_hidden_states = {}  # Track hidden states per choice
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    chunk_data = line[6:]  # Remove "data: " prefix
                    if chunk_data == "[DONE]":
                        print("📄 Stream finished with [DONE]")
                        break
                    
                    try:
                        chunk = json.loads(chunk_data)
                        chunks.append(chunk)
                        
                        if "choices" in chunk and chunk["choices"]:
                            choice = chunk["choices"][0]
                            choice_idx = choice.get("index", 0)
                            
                            # Initialize tracking for this choice
                            if choice_idx not in choice_texts:
                                choice_texts[choice_idx] = ""
                            
                            if "text" in choice:
                                choice_texts[choice_idx] += choice["text"]
                            
                            # Check for hidden states in final chunk
                            if choice.get("finish_reason") is not None:
                                print(f"📋 Choice {choice_idx} final chunk - finish_reason: {choice['finish_reason']}")
                                if "hidden_states" in choice and choice["hidden_states"] is not None:
                                    choice_hidden_states[choice_idx] = choice["hidden_states"]
                                    print(f"✅ Choice {choice_idx} hidden states: length={len(choice['hidden_states'])}")
                                    print(f"   First few values: {choice['hidden_states'][:3]}")
                                else:
                                    print(f"❌ Choice {choice_idx} missing hidden states")
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Failed to parse chunk: {e}")
            
            print(f"📊 Total chunks received: {len(chunks)}")
            print(f"📝 Generated texts:")
            for idx, text in choice_texts.items():
                print(f"   Choice {idx}: '{text}'")
            
            expected_choices = 2
            if len(choice_hidden_states) == expected_choices:
                print(f"✅ Parallel sampling (n={expected_choices}) with hidden states: SUCCESS")
                print(f"   Hidden states received for {len(choice_hidden_states)} choices")
            else:
                print(f"❌ Parallel sampling failed: Expected {expected_choices} choices with hidden states, got {len(choice_hidden_states)}")
                
        else:
            print(f"❌ Error response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def check_server_health(server):
    """Check if vLLM server is running and responsive."""
    print("🏥 Checking server health...")
    
    try:
        response = requests.get(server.url_for("health"), timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy")
            return True
        else:
            print(f"❌ Server unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        return False

def check_models(server):
    """Check available models."""
    print("📋 Checking available models...")
    
    try:
        response = requests.get(server.url_for("v1", "models"), timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            print(f"✅ Available models: {models}")
            if MODEL_NAME in models:
                print(f"✅ Target model {MODEL_NAME} is available")
                return True
            else:
                print(f"❌ Target model {MODEL_NAME} not found")
                return False
        else:
            print(f"❌ Failed to get models: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to check models: {e}")
        return False

def run_debug_tests():
    """Run the debug tests with a self-managed server."""
    print("🚀 Hidden States API Debug Script")
    print("=" * 50)
    print("🔧 Starting vLLM server with V1 engine...")
    
    # Server arguments similar to the integration test
    server_args = [
        "--max-model-len", "2048",
        "--max-num-seqs", "128", 
        "--enforce-eager",  # Disable CUDA graphs for debugging
    ]
    
    # Environment to force V1 engine
    env_dict = {"VLLM_USE_V1": "1"}
    
    try:
        with RemoteOpenAIServer(MODEL_NAME, server_args, env_dict=env_dict) as server:
            print(f"✅ Server started at {server.url_for('')}")
            
            # Give the server a moment to fully initialize
            print("⏳ Waiting for server to be ready...")
            time.sleep(2)
            
            # Basic health checks
            if not check_server_health(server):
                print("❌ Server health check failed")
                return False
                
            if not check_models(server):
                print("❌ Model availability check failed")
                return False
            
            # Test APIs - Non-streaming
            test_completion_hidden_states(server)
            test_chat_completion_hidden_states(server)
            
            # Test APIs - Streaming  
            test_completion_streaming_hidden_states(server)
            test_chat_completion_streaming_hidden_states(server)
            test_streaming_parallel_sampling(server)
            
            print("\n🏁 Debug complete!")
            return True
            
    except Exception as e:
        print(f"❌ Failed to start server or run tests: {e}")
        return False

if __name__ == "__main__":
    success = run_debug_tests()
    sys.exit(0 if success else 1)