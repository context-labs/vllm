# SPDX-License-Identifier: Apache-2.0

"""
Configuration and fixtures for hidden states tests.
"""

import pytest
import torch
from transformers import AutoTokenizer

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs

# Test configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TEST_SEED = 42


@pytest.fixture(scope="session")
def tokenizer():
    """Provide a tokenizer for testing."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="session")
def vllm_config():
    """Provide a VllmConfig for testing."""
    engine_args = EngineArgs(model=MODEL_NAME, seed=TEST_SEED)
    return engine_args.create_engine_config()


@pytest.fixture
def sample_hidden_states(vllm_config: VllmConfig):
    """Generate sample hidden states tensor for testing."""
    hidden_size = vllm_config.model_config.hf_config.hidden_size
    return torch.randn(1, hidden_size, dtype=torch.float32)


@pytest.fixture
def sample_prompt_tokens(tokenizer):
    """Generate sample prompt tokens for testing."""
    prompts = [
        "Hello world",
        "The quick brown fox",
        "In the beginning was the Word"
    ]
    return [tokenizer(prompt).input_ids for prompt in prompts]


class MockHiddenStatesExtractor:
    """Mock class for testing hidden states extraction logic."""
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
    
    def extract_hidden_states(self, 
                            request_ids: list[str], 
                            model_output: torch.Tensor) -> dict[str, torch.Tensor]:
        """Mock hidden states extraction."""
        return {
            req_id: torch.randn(1, self.hidden_size, dtype=torch.float32)
            for req_id in request_ids
        }
    
    def should_extract_hidden_states(self, requests: list) -> bool:
        """Mock logic for determining if hidden states should be extracted."""
        return any(getattr(req, 'return_hidden_states', False) for req in requests)


@pytest.fixture
def mock_hidden_states_extractor(vllm_config: VllmConfig):
    """Provide a mock hidden states extractor for testing."""
    hidden_size = vllm_config.model_config.hf_config.hidden_size
    return MockHiddenStatesExtractor(hidden_size)


class HiddenStatesTestUtils:
    """Utility functions for hidden states testing."""
    
    @staticmethod
    def validate_hidden_states_tensor(tensor: torch.Tensor, expected_hidden_size: int) -> bool:
        """Validate a hidden states tensor."""
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.shape != (1, expected_hidden_size):
            return False
        if tensor.dtype != torch.float32:
            return False
        return True
    
    @staticmethod
    def validate_hidden_states_list(hidden_states: list, expected_hidden_size: int) -> bool:
        """Validate a hidden states list (serialized format)."""
        if not isinstance(hidden_states, list):
            return False
        if len(hidden_states) != expected_hidden_size:
            return False
        if not all(isinstance(x, (int, float)) for x in hidden_states):
            return False
        return True
    
    @staticmethod
    def convert_tensor_to_list(tensor: torch.Tensor) -> list[float]:
        """Convert hidden states tensor to serializable list."""
        return tensor.squeeze(0).tolist()
    
    @staticmethod
    def convert_list_to_tensor(hidden_states: list[float]) -> torch.Tensor:
        """Convert hidden states list back to tensor."""
        return torch.tensor(hidden_states, dtype=torch.float32).unsqueeze(0)
    
    @staticmethod
    def estimate_serialized_size(hidden_states: list[float]) -> int:
        """Estimate serialized size in bytes for ZMQ transfer."""
        import json
        return len(json.dumps(hidden_states).encode('utf-8'))


@pytest.fixture
def hidden_states_utils():
    """Provide hidden states test utilities."""
    return HiddenStatesTestUtils


# Test data generators
def generate_test_requests(num_requests: int = 3, 
                         with_hidden_states: bool = True) -> list[dict]:
    """Generate test request data."""
    requests = []
    for i in range(num_requests):
        request = {
            "request_id": f"test_req_{i}",
            "prompt_token_ids": [1, 2, 3, 4, 5, i],
            "max_tokens": 5,
            "return_hidden_states": with_hidden_states and (i % 2 == 0)
        }
        requests.append(request)
    return requests


@pytest.fixture
def sample_test_requests():
    """Provide sample test requests."""
    return generate_test_requests()


# Performance monitoring utilities
class PerformanceMonitor:
    """Simple performance monitoring for tests."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_usage = []
    
    def start(self):
        import time
        self.start_time = time.time()
    
    def stop(self):
        import time
        self.end_time = time.time()
    
    def elapsed_time(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def record_memory(self):
        try:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.memory_usage.append(memory_mb)
        except ImportError:
            # psutil not available
            pass
    
    def peak_memory(self) -> float:
        return max(self.memory_usage) if self.memory_usage else 0.0


@pytest.fixture
def performance_monitor():
    """Provide a performance monitor for tests."""
    return PerformanceMonitor()