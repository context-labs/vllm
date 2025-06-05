# Hidden States API Integration Investigation Summary

## Problem Statement
The hidden states API integration test was failing because the `/v1/completions` endpoint was not returning a `hidden_states` field in the response when `return_hidden_states: true` and `hidden_states_for_tokens: [-1]` were sent.

## Root Cause Analysis

After investigating the complete vLLM v1 pipeline, I found that the hidden states functionality is **fully implemented** from the engine core through the model runner, but there was a **critical bug in the API response formatting** in both completion and chat completion endpoints.

### The Bug

In both `/home/kyle/code/vllm-hidden-states-context/vllm/vllm/entrypoints/openai/serving_completion.py` and `/home/kyle/code/vllm-hidden-states-context/vllm/vllm/entrypoints/openai/serving_chat.py`, the code was incorrectly trying to access hidden states using the **output choice index** instead of the **token position**:

```python
# INCORRECT (original code)
if (hasattr(final_res, 'hidden_states') and 
    final_res.hidden_states is not None and 
    output.index in final_res.hidden_states):
    choice_kwargs["hidden_states"] = final_res.hidden_states[output.index]
```

### The Issue Explanation

The `RequestOutput.hidden_states` field is structured as:
```python
hidden_states: dict[int, list[float]]  # token_position -> hidden_state_vector
```

But the code was using `output.index` (which is the choice/sequence index, typically 0) as a key to look up hidden states, when it should have been using the actual token positions where hidden states were extracted.

## Complete Data Flow (Working Correctly)

1. **API Request**: `{"return_hidden_states": true, "hidden_states_for_tokens": [-1]}`
2. **Request Processing**: Parameters flow through `CompletionRequest.to_sampling_params()` 
3. **V1 Engine Core**: Creates `Request` with `return_hidden_states=True`
4. **GPU Model Runner**: Extracts hidden states from model activations for specified token positions
5. **ModelRunnerOutput**: Contains `last_hidden_states: dict[str, torch.Tensor]` (req_id -> tensor)
6. **Scheduler**: Converts tensors to `EngineCoreOutput.hidden_states: list[float]` 
7. **Output Processor**: Converts to `RequestOutput.hidden_states: dict[int, list[float]]` (position -> vector)
8. **API Response Formatting**: **THIS IS WHERE THE BUG WAS** - incorrectly accessing the dict

## Fixes Implemented

### 1. Fixed Completion API (`serving_completion.py`)

```python
# NEW (fixed code)
if (hasattr(final_res, 'hidden_states') and 
    final_res.hidden_states is not None and 
    request.return_hidden_states):
    # Hidden states are keyed by token position, not output index
    if final_res.hidden_states:
        if request.hidden_states_for_tokens:
            # Handle -1 as last token position
            requested_positions = []
            total_tokens = len(final_res.prompt_token_ids or []) + len(output.token_ids)
            for pos in request.hidden_states_for_tokens:
                if pos == -1:
                    # Last token position (convert to absolute position)
                    requested_positions.append(total_tokens - 1)
                else:
                    requested_positions.append(pos)
            
            # Find the first available position from the requested ones
            for pos in requested_positions:
                if pos in final_res.hidden_states:
                    choice_kwargs["hidden_states"] = final_res.hidden_states[pos]
                    break
        else:
            # No specific positions requested, use last available
            last_pos = max(final_res.hidden_states.keys())
            choice_kwargs["hidden_states"] = final_res.hidden_states[last_pos]
```

### 2. Fixed Chat Completion API (`serving_chat.py`)

Applied the same fix to the `chat_completion_full_generator` method.

## Key Insights

1. **Hidden states extraction is fully implemented** in the V1 engine - the bug was only in the API response formatting
2. **Token position mapping**: `-1` means "last token" and gets converted to the absolute position
3. **Data structure**: `RequestOutput.hidden_states` maps token positions to hidden state vectors
4. **Multiple requests**: Each completion choice needs to calculate its own final token position
5. **Backward compatibility**: The fix maintains full backward compatibility with existing API behavior

## Files Modified

1. `/home/kyle/code/vllm-hidden-states-context/vllm/vllm/entrypoints/openai/serving_completion.py`
2. `/home/kyle/code/vllm-hidden-states-context/vllm/vllm/entrypoints/openai/serving_chat.py`

## Expected Result

After these fixes, API requests with `return_hidden_states: true` should properly return hidden state vectors in the response:

```json
{
  "choices": [
    {
      "text": "Paris.",
      "hidden_states": [0.1234, -0.5678, 0.9012, ...],  // 4096-dimensional vector
      "finish_reason": "stop"
    }
  ]
}
```

## Testing

The debug script `/home/kyle/code/vllm-hidden-states-context/vllm/debug_hidden_states_api.py` can be used to verify the fix works correctly once a V1 server is running.

## Next Steps

1. Test the fix with a running vLLM V1 server
2. Verify that the integration tests now pass
3. Consider adding more comprehensive error handling for edge cases
4. Review the TODO comment about supporting multiple token positions in the output processor