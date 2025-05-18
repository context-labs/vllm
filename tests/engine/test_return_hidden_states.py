import pytest
import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

@pytest.mark.parametrize("model", ["facebook/opt-125m"])
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_return_hidden_states_single_step(model: str, enforce_eager: bool):
    # This test checks if stepping the LLM successfully runs iterations
    # and returns hidden states for the last token if single-step model is enabled.
    engine_args = EngineArgs(model=model,
                             return_hidden_states=True,
                             num_scheduler_steps=1,
                             enforce_eager=enforce_eager)

    engine = LLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams()
    prompt = (
        "You are a helpful assistant. Please tell me the capital of France in a few words.")
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

@pytest.mark.parametrize("model", ["facebook/opt-125m"])
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_return_hidden_states_multi_step(model: str, enforce_eager: bool):
    # This test checks if stepping the LLM successfully runs iterations
    # and returns hidden states for the last token if multi-step mode is enabled.
    engine_args = EngineArgs(model=model,
                             return_hidden_states=True,
                             num_scheduler_steps=2,
                             enforce_eager=enforce_eager)

    engine = LLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams()
    prompt = (
        "You are a helpful assistant. Please tell me the capital of France in a few words.")
    engine.add_request("0", prompt, sampling_params)

    finished = False
    i = 0
    while not finished and i < 100:
        steps = engine.step()
        for step_out in steps:
            assert isinstance(step_out, RequestOutput)
            if step_out.finished:
                finished = True
                assert step_out.outputs[0].hidden_states is not None
            else:
                assert step_out.outputs[0].hidden_states is None
            i += 1