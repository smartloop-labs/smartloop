"""A decode that runs out of device memory is retried once, after eviction.

The answer shares the device with whatever earlier steps left resident, and
with the embedding model citation prep loads alongside it.  When llama.cpp
cannot find room it returns a non-zero ``llama_decode``; freeing the auxiliary
models gives it that room, so the stream evicts them and tries again before
giving up.  What already reached the client is kept: the retry asks for the
rest of the answer rather than starting it over.
"""
from unittest import mock

import pytest

from langchain_core.outputs import GenerationChunk

from smartloop.api.routers.chat import stream_chunks
from tests.unit.test_stream_opening_buffer import _content, _request, _service


@pytest.fixture(autouse=True)
def _no_telemetry():
    with mock.patch("smartloop.api.routers.chat.get_api_client"):
        yield


def _failing_once(first, second):
    prompts: list[str] = []

    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        prompts.append(prompt)
        chunks = first if len(prompts) == 1 else second
        for text in chunks:
            yield GenerationChunk(text=text)
        if len(prompts) == 1:
            raise RuntimeError("llama_decode returned -3")

    return stream_completion, prompts


@pytest.mark.asyncio
async def test_an_oom_before_anything_is_sent_starts_over():
    stream, prompts = _failing_once(["Con"], ["Contoso Ltd. billed Jordan Rivera."])
    svc = _service(stream)
    raw = [event async for event in stream_chunks(_request(), svc)]

    assert "".join(_content(raw)) == "Contoso Ltd. billed Jordan Rivera."
    assert len(prompts) == 2 and prompts[0] == prompts[1]
    svc.unload_auxiliary_models.assert_any_call(include_embedding=True)


@pytest.mark.asyncio
async def test_an_oom_mid_answer_continues_it():
    sent = "Contoso Ltd. billed Jordan Rivera "
    stream, prompts = _failing_once([sent], ["for contract work."])
    svc = _service(stream)
    raw = [event async for event in stream_chunks(_request(), svc)]

    text = "".join(_content(raw))
    assert text == sent + "for contract work."
    assert "**Error:**" not in text
    # The retry quotes back what the client already has.
    assert prompts[1].startswith(prompts[0]) and sent in prompts[1][len(prompts[0]):]


@pytest.mark.asyncio
async def test_a_second_oom_reaches_the_client():
    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        raise RuntimeError("llama_decode returned -3")
        yield  # pragma: no cover

    svc = _service(stream_completion)
    raw = [event async for event in stream_chunks(_request(), svc)]
    assert "**Error:** llama_decode returned -3" in "".join(_content(raw))


@pytest.mark.asyncio
async def test_other_errors_are_not_retried():
    calls = []

    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        calls.append(prompt)
        raise ValueError("bad template")
        yield  # pragma: no cover

    svc = _service(stream_completion)
    raw = [event async for event in stream_chunks(_request(), svc)]
    assert len(calls) == 1
    assert "**Error:** bad template" in "".join(_content(raw))
