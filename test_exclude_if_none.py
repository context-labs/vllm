#!/usr/bin/env python3
"""
Quick test to validate the exclude_if_none functionality
"""

import sys
sys.path.insert(0, '/home/kyle/code/vllm-hidden-states-context/vllm')

from vllm.entrypoints.openai.protocol import ChatCompletionResponseChoice, ChatMessage

# Test creating a ChatCompletionResponseChoice without hidden_states
choice = ChatCompletionResponseChoice(
    index=0,
    message=ChatMessage(role="assistant", content="Hello!"),
    finish_reason="stop"
)

print("Choice created successfully")
print(f"Choice fields: {list(choice.model_fields.keys())}")
print(f"Choice exclude_if_none_fields: {choice.exclude_if_none_fields}")

# Serialize to dict
choice_dict = choice.model_dump()
print(f"Serialized keys: {list(choice_dict.keys())}")
print(f"hidden_states in dict: {'hidden_states' in choice_dict}")

if 'hidden_states' in choice_dict:
    print(f"hidden_states value: {choice_dict['hidden_states']}")

# Test with hidden_states
choice_with_hs = ChatCompletionResponseChoice(
    index=0,
    message=ChatMessage(role="assistant", content="Hello!"),
    finish_reason="stop",
    hidden_states=[1.0, 2.0, 3.0]
)

choice_with_hs_dict = choice_with_hs.model_dump()
print(f"\nWith hidden states - Serialized keys: {list(choice_with_hs_dict.keys())}")
print(f"hidden_states in dict: {'hidden_states' in choice_with_hs_dict}")
if 'hidden_states' in choice_with_hs_dict:
    print(f"hidden_states value: {choice_with_hs_dict['hidden_states']}")