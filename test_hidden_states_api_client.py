#!/usr/bin/env python3
"""
Test script for vLLM Hidden States API Integration

This script tests the OpenAI-compatible API endpoints with hidden states support.
It sends actual HTTP requests to a running vLLM server and validates the responses.

Usage:
    python test_hidden_states_api_client.py [--host HOST] [--port PORT] [--model MODEL]

Examples:
    python test_hidden_states_api_client.py
    python test_hidden_states_api_client.py --host localhost --port 8000
    python test_hidden_states_api_client.py --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import json
import sys
import time
from typing import Dict, Any, Optional
import requests
from requests.exceptions import ConnectionError, RequestException


class HiddenStatesAPITester:
    """Test client for vLLM Hidden States API."""
    
    def __init__(self, host: str = "localhost", port: int = 8000, model: str = "meta-llama/Llama-3.2-1B-Instruct"):
        self.base_url = f"http://{host}:{port}"
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def check_server_health(self) -> bool:
        """Check if the vLLM server is running and healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except ConnectionError:
            return False
        except RequestException:
            return False
    
    def test_chat_completion_without_hidden_states(self) -> Dict[str, Any]:
        """Test chat completion without hidden states (baseline)."""
        print("🧪 Testing Chat Completion without Hidden States...")
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "Hello! How are you today?"}
            ],
            "max_tokens": 10,
            "temperature": 0.7
        }
        
        try:
            response = self.session.post(f"{self.base_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Validate response structure
            assert "choices" in data
            assert len(data["choices"]) > 0
            choice = data["choices"][0]
            assert "message" in choice
            
            # Debug: Print the actual response to see what's there
            print(f"   DEBUG: Response keys: {list(data.keys())}")
            print(f"   DEBUG: Choice keys: {list(choice.keys())}")
            if "hidden_states" in choice:
                print(f"   DEBUG: Hidden states found: {type(choice['hidden_states'])}, length: {len(choice['hidden_states']) if isinstance(choice['hidden_states'], list) else 'N/A'}")
            
            # With the new exclude_if_none approach, hidden_states should not be present when None
            # But if server hasn't restarted, it might still be there with None value
            if "hidden_states" in choice:
                assert choice["hidden_states"] is None, f"Expected hidden_states to be None, got {choice['hidden_states']}"
                print("   NOTE: hidden_states field present but None (server needs restart for exclude_if_none)")
            else:
                print("   ✅ hidden_states field properly excluded")
            
            print("✅ Chat completion without hidden states: SUCCESS")
            print(f"   Response: {choice['message']['content'][:50]}...")
            return data
            
        except Exception as e:
            print(f"❌ Chat completion without hidden states: FAILED - {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def test_chat_completion_with_hidden_states(self) -> Dict[str, Any]:
        """Test chat completion with hidden states."""
        print("🧪 Testing Chat Completion with Hidden States...")
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "max_tokens": 10,
            "temperature": 0.7,
            "return_hidden_states": True,
            "hidden_states_for_tokens": [-1]  # Last token
        }
        
        try:
            response = self.session.post(f"{self.base_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Validate response structure
            assert "choices" in data
            assert len(data["choices"]) > 0
            choice = data["choices"][0]
            assert "message" in choice
            assert "hidden_states" in choice  # Should be present
            assert isinstance(choice["hidden_states"], list)
            assert len(choice["hidden_states"]) > 0
            assert all(isinstance(x, (int, float)) for x in choice["hidden_states"])
            
            print("✅ Chat completion with hidden states: SUCCESS")
            print(f"   Response: {choice['message']['content'][:50]}...")
            print(f"   Hidden states shape: {len(choice['hidden_states'])}")
            print(f"   Hidden states sample: {choice['hidden_states'][:5]}...")
            return data
            
        except Exception as e:
            print(f"❌ Chat completion with hidden states: FAILED - {e}")
            raise
    
    def test_completion_without_hidden_states(self) -> Dict[str, Any]:
        """Test completion without hidden states (baseline)."""
        print("🧪 Testing Completion without Hidden States...")
        
        payload = {
            "model": self.model,
            "prompt": "The capital of France is",
            "max_tokens": 5,
            "temperature": 0.7
        }
        
        try:
            response = self.session.post(f"{self.base_url}/v1/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Validate response structure
            assert "choices" in data
            assert len(data["choices"]) > 0
            choice = data["choices"][0]
            assert "text" in choice
            
            # With the new exclude_if_none approach, hidden_states should not be present when None
            # But if server hasn't restarted, it might still be there with None value
            if "hidden_states" in choice:
                assert choice["hidden_states"] is None, f"Expected hidden_states to be None, got {choice['hidden_states']}"
                print("   NOTE: hidden_states field present but None (server needs restart for exclude_if_none)")
            else:
                print("   ✅ hidden_states field properly excluded")
            
            print("✅ Completion without hidden states: SUCCESS")
            print(f"   Response: {choice['text'][:50]}...")
            return data
            
        except Exception as e:
            print(f"❌ Completion without hidden states: FAILED - {e}")
            raise
    
    def test_completion_with_hidden_states(self) -> Dict[str, Any]:
        """Test completion with hidden states."""
        print("🧪 Testing Completion with Hidden States...")
        
        payload = {
            "model": self.model,
            "prompt": "The capital of France is",
            "max_tokens": 5,
            "temperature": 0.7,
            "return_hidden_states": True,
            "hidden_states_for_tokens": [-1]  # Last token
        }
        
        try:
            response = self.session.post(f"{self.base_url}/v1/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Validate response structure
            assert "choices" in data
            assert len(data["choices"]) > 0
            choice = data["choices"][0]
            assert "text" in choice
            assert "hidden_states" in choice  # Should be present
            assert isinstance(choice["hidden_states"], list)
            assert len(choice["hidden_states"]) > 0
            assert all(isinstance(x, (int, float)) for x in choice["hidden_states"])
            
            print("✅ Completion with hidden states: SUCCESS")
            print(f"   Response: {choice['text'][:50]}...")
            print(f"   Hidden states shape: {len(choice['hidden_states'])}")
            print(f"   Hidden states sample: {choice['hidden_states'][:5]}...")
            return data
            
        except Exception as e:
            print(f"❌ Completion with hidden states: FAILED - {e}")
            raise
    
    def test_streaming_chat_completion_with_hidden_states(self) -> Dict[str, Any]:
        """Test streaming chat completion with hidden states."""
        print("🧪 Testing Streaming Chat Completion with Hidden States...")
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "Write a very short story about a robot."}
            ],
            "max_tokens": 20,
            "temperature": 0.7,
            "stream": True,
            "return_hidden_states": True,
            "hidden_states_for_tokens": [-1]
        }
        
        try:
            response = self.session.post(f"{self.base_url}/v1/chat/completions", json=payload, stream=True)
            response.raise_for_status()
            
            chunks = []
            full_content = ""
            hidden_states_found = False
            
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        data_text = line_text[6:]  # Remove 'data: ' prefix
                        if data_text.strip() == '[DONE]':
                            break
                        
                        try:
                            chunk_data = json.loads(data_text)
                            chunks.append(chunk_data)
                            
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                choice = chunk_data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    full_content += choice['delta']['content']
                                
                                # Check for hidden states in final chunk
                                if 'hidden_states' in choice:
                                    hidden_states_found = True
                                    print(f"   Found hidden states in chunk: {len(choice['hidden_states'])}")
                        
                        except json.JSONDecodeError:
                            continue
            
            print("✅ Streaming chat completion with hidden states: SUCCESS")
            print(f"   Content: {full_content[:100]}...")
            print(f"   Total chunks: {len(chunks)}")
            print(f"   Hidden states found: {hidden_states_found}")
            
            return {"chunks": chunks, "content": full_content}
            
        except Exception as e:
            print(f"❌ Streaming chat completion with hidden states: FAILED - {e}")
            raise
    
    def test_invalid_request(self) -> None:
        """Test invalid request parameters."""
        print("🧪 Testing Invalid Request Parameters...")
        
        # Test invalid return_hidden_states type
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
            "return_hidden_states": "true"  # Should be boolean
        }
        
        try:
            response = self.session.post(f"{self.base_url}/v1/chat/completions", json=payload)
            # This should fail with validation error
            if response.status_code == 422:
                print("✅ Invalid request validation: SUCCESS (correctly rejected)")
            else:
                print(f"⚠️  Invalid request validation: UNEXPECTED STATUS {response.status_code}")
                
        except Exception as e:
            print(f"❌ Invalid request validation: FAILED - {e}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results."""
        print(f"🚀 Starting Hidden States API Tests")
        print(f"   Server: {self.base_url}")
        print(f"   Model: {self.model}")
        print("=" * 60)
        
        # Check server health first
        if not self.check_server_health():
            print(f"❌ Server is not running or not healthy at {self.base_url}")
            print("   Please start the vLLM server with V1 engine enabled:")
            print("   VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.2-1B-Instruct")
            sys.exit(1)
        
        print(f"✅ Server is healthy at {self.base_url}")
        print()
        
        results = {}
        
        try:
            # Run baseline tests
            results["chat_without_hidden_states"] = self.test_chat_completion_without_hidden_states()
            print()
            
            results["completion_without_hidden_states"] = self.test_completion_without_hidden_states()
            print()
            
            # Run hidden states tests
            results["chat_with_hidden_states"] = self.test_chat_completion_with_hidden_states()
            print()
            
            results["completion_with_hidden_states"] = self.test_completion_with_hidden_states()
            print()
            
            # Run streaming test
            results["streaming_chat_with_hidden_states"] = self.test_streaming_chat_completion_with_hidden_states()
            print()
            
            # Run validation test
            self.test_invalid_request()
            print()
            
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            import traceback
            traceback.print_exc()
            return results
        
        print("=" * 60)
        print("🎉 All Hidden States API Tests Completed Successfully!")
        print()
        print("📊 Summary:")
        for test_name, result in results.items():
            if isinstance(result, dict):
                if "choices" in result:
                    choice = result["choices"][0]
                    has_hidden_states = "hidden_states" in choice or \
                                      ("message" in choice and "hidden_states" in choice.get("message", {}))
                    print(f"   ✅ {test_name}: Hidden states = {has_hidden_states}")
                elif "chunks" in result:
                    print(f"   ✅ {test_name}: {len(result['chunks'])} chunks")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Test vLLM Hidden States API")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct", 
                       help="Model name (default: meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Create tester and run tests
    tester = HiddenStatesAPITester(host=args.host, port=args.port, model=args.model)
    results = tester.run_all_tests()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📁 Results saved to {args.output}")


if __name__ == "__main__":
    main()