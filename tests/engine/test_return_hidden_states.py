import pytest
import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_return_hidden_states(model: str):
    # This test checks if stepping the LLM successfully runs iterations
    # and returns hidden states for each request.
    engine_args = EngineArgs(model=model,
                             return_hidden_states=True,
                             enforce_eager=True)

    engine = LLMEngine.from_engine_args(engine_args)
    tokenizer = engine.tokenizer.tokenizer
    sampling_params = SamplingParams()
    prompt = (
        "You are a helpful assistant. Please tell me the capital of France in a few words.")
    prompt_tokens = tokenizer(prompt)['input_ids']
    engine.add_request("0", prompt, sampling_params)

    finished = False
    i = 0
    while not finished and i < 100:
        (step_out,) = engine.step()
        assert isinstance(step_out, RequestOutput)
        if step_out.finished:
            finished = True
            assert step_out.outputs[0].hidden_states is not None
        else:
            assert step_out.outputs[0].hidden_states is None
        i += 1