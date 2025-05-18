# SPDX-License-Identifier: Apache-2.0

# imports for guided decoding tests
import json
import re
from typing import Optional

import jsonschema
import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
import requests
import torch
from openai import BadRequestError, OpenAI

from ...utils import RemoteOpenAIServer
from .test_completion import zephyr_lora_added_tokens_files  # noqa: F401
from .test_completion import zephyr_lora_files  # noqa: F401

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()

@pytest.fixture(scope="module", params=[False,True])  
def return_hidden_states_param(request):
    return request.param

@pytest.fixture(scope="module", params = [False])
def server(
        request,
        monkeypatch_module,
        zephyr_lora_files,  #noqa: F811
        zephyr_lora_added_tokens_files,
        return_hidden_states_param : str
):
    use_v1 = request.param
    monkeypatch_module.setenv('VLLM_USE_V1', '1' if use_v1 else '0')

    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        # lora config below
        "--enable-lora",
        "--lora-modules",
        f"zephyr-lora={zephyr_lora_files}",
        f"zephyr-lora2={zephyr_lora_added_tokens_files}",
        "--max-lora-rank",
        "64",
        "--max-cpu-loras",
        "2",
        "--max-num-seqs",
        "128",
    ]

    if return_hidden_states_param:
        args.append("--return-hidden-states")

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server

@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client

@pytest.mark.asyncio
async def test_single_chat_session(client: openai.AsyncOpenAI, return_hidden_states_param: str):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=10,
        logprobs=True,
        top_logprobs=5)
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]

    if return_hidden_states_param:
        assert choice.hidden_states is not None, "hidden_states should be returned"
    else:
        assert not hasattr(choice, "hidden_states"), "hidden_states should not be returned"

    message = chat_completion.choices[0].message

    messages.append({"role": "assistant", "content": message.content})

    # test multi-turn dialogue
    messages.append({"role": "user", "content": "express your result in json"})
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=10,
    )
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0

    if return_hidden_states_param:
        assert choice.hidden_states is not None, "hidden_states should be returned"
    else:
        assert not hasattr(choice, "hidden_states"), "hidden_states should not be returned"



@pytest.mark.asyncio
async def test_chat_streaming(client: openai.AsyncOpenAI, return_hidden_states_param: str):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    # test streaming
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        stream=True,
    )
    hidden_states = None
    async for chunk in stream:
        delta = chunk.choices[0].delta
        hidden_states = hidden_states or (delta.hidden_states if hasattr(delta, "hidden_states") else None)

    if return_hidden_states_param:
        assert hidden_states is not None, "hidden_states should be returned"
    else:
        assert hidden_states is None, "hidden_states should not be returned"

