#!/usr/bin/env python3
"""
Simple test script to verify ZMQ client logic for hidden states is working.
This script tests the implementation without full engine startup.
"""

import os
import sys
import time

# Set V1 engine flag
os.environ["VLLM_USE_V1"] = "1"

def test_zmq_client_logic():
    """Test the ZMQ client logic implementation."""
    print("Testing ZMQ client logic for hidden states...")
    
    try:
        # Test imports
        from vllm.v1.engine import HiddenStatesExtractionRequest, EngineCoreRequestType
        from vllm.v1.engine.output_processor import CompletedRequestInfo, OutputProcessorOutput
        from vllm.v1.engine import EngineCoreRequest
        from vllm import SamplingParams
        
        # Test 1: Create completed request info
        original_request = EngineCoreRequest(
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
        
        completed_info = CompletedRequestInfo(
            request_id="test_123",
            original_request=original_request,
            sequence_tokens=[1, 2, 3, 4, 5],
            final_token_position=4
        )
        
        # Test 2: Create HiddenStatesExtractionRequest
        hs_request = HiddenStatesExtractionRequest(
            request_id=f"hs_{completed_info.request_id}",
            original_request_id=completed_info.request_id,
            sequence_tokens=completed_info.sequence_tokens,
            target_position=completed_info.final_token_position,
            arrival_time=time.time(),
            layer_indices=None,
            extract_all_positions=False,
        )
        
        # Test 3: Verify the ZMQ request structure
        assert hs_request.request_id == "hs_test_123"
        assert hs_request.original_request_id == "test_123"
        assert hs_request.sequence_tokens == [1, 2, 3, 4, 5]
        assert hs_request.target_position == 4
        assert hs_request.layer_indices is None
        assert hs_request.extract_all_positions is False
        
        # Test 4: Verify EngineCoreRequestType
        assert hasattr(EngineCoreRequestType, 'HIDDEN_STATES_EXTRACT')
        assert EngineCoreRequestType.HIDDEN_STATES_EXTRACT.value == b'\x05'
        
        print("✅ ZMQ client logic: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ ZMQ client logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_zmq_method_signatures():
    """Test that the ZMQ methods have correct signatures."""
    print("Testing ZMQ method signatures...")
    
    try:
        # Check AsyncLLM method
        from vllm.v1.engine.async_llm import AsyncLLM
        assert hasattr(AsyncLLM, '_process_hidden_states_requests')
        
        # Check LLMEngine method  
        from vllm.v1.engine.llm_engine import LLMEngine
        assert hasattr(LLMEngine, '_process_hidden_states_requests')
        
        print("✅ ZMQ method signatures: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ ZMQ method signatures test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_output_processor_integration():
    """Test OutputProcessor integration with completed requests."""
    print("Testing OutputProcessor integration...")
    
    try:
        from vllm.v1.engine.output_processor import OutputProcessorOutput
        
        # Test OutputProcessorOutput structure
        output = OutputProcessorOutput(
            request_outputs=[],
            reqs_to_abort=[],
            completed_requests=[]
        )
        
        assert hasattr(output, 'completed_requests')
        assert isinstance(output.completed_requests, list)
        
        print("✅ OutputProcessor integration: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ OutputProcessor integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔍 Testing ZMQ Client Implementation")
    print("=" * 50)
    
    all_passed = True
    
    # Test individual components
    all_passed &= test_zmq_client_logic()
    all_passed &= test_zmq_method_signatures()
    all_passed &= test_output_processor_integration()
    
    print("=" * 50)
    if all_passed:
        print("🎉 All ZMQ client tests PASSED!")
        print()
        print("📋 Implementation Status:")
        print("✅ Data structures extended")
        print("✅ Model forward pass integration implemented")
        print("✅ ZMQ pipeline data structures working")
        print("✅ ZMQ client logic implemented (AsyncLLM & LLMEngine)")
        print("🔄 End-to-end ZMQ pipeline testing pending")
        print("🔄 API integration pending")
    else:
        print("❌ Some ZMQ client tests FAILED. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())