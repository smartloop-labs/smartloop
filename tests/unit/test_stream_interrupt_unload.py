"""Interrupting a stream must not free models out from under llama.cpp.

``cancelled`` is cooperative — the producer thread only observes it between
tokens — so the consumer's ``finally`` runs while that thread is still inside
``llama_sampler_sample``.  Freeing the llama.cpp context there segfaults the
process:

    llama_sampler_sample + ffi_call_SYSV + ... + thread_run
    EXC_BAD_ACCESS (SIGSEGV) at 0x14c23c000

preceded by ``malloc: pointer being freed was not allocated``.  So the unload
has to run on the producer thread, after its generator is done.
"""
import asyncio
import threading

from unittest import mock

import pytest

from langchain_core.outputs import GenerationChunk

from smartloop.api.model.chat import ChatCompletionRequest, ChatMessage
from smartloop.api.routers.chat import stream_chunks
from smartloop.model_host.client import InProcessModelClient


def _service(stream_completion):
    """A ModelService stand-in that reaches the producer with no orchestration."""
    svc = mock.MagicMock()
    svc.stream_completion.side_effect = stream_completion
    svc.get_relevant_history.return_value = []
    svc.get_project_skills.return_value = []
    svc.get_asset_images.return_value = []
    svc.get_project_info.return_value = None
    svc.get_project_id.return_value = "proj"
    svc.count_tokens.return_value = 10
    svc.get_response_max_context_tokens.return_value = 4096
    svc.get_max_context_tokens.return_value = 4096
    svc.maybe_write_output.return_value = None
    # Chat generates through the model client, whose default implementation is
    # the in-process one — so put the real client in front of the fake service
    # and the test drives exactly the path production takes.
    svc.model_client = InProcessModelClient(svc)
    return svc


@pytest.fixture(autouse=True)
def _no_telemetry():
    """The usage tail calls the platform client; keep the test offline."""
    with mock.patch("smartloop.api.routers.chat.get_api_client"):
        yield


def _request():
    # No user message → the orchestrator block is skipped and we land on the
    # response stream directly, which is the path under test.
    return ChatCompletionRequest(
        model="sl-mini",
        messages=[ChatMessage(role="system", content="be brief")],
        stream=True,
    )


@pytest.mark.asyncio
async def test_interrupt_defers_unload_until_the_producer_stops():
    """The unload must wait for the thread that is inside llama.cpp."""
    entered = threading.Event()   # producer is back inside stream_completion
    release = threading.Event()   # let it produce its second token
    finished = threading.Event()  # producer's generator is done

    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        try:
            yield GenerationChunk(text="hello")
            entered.set()
            # Stands in for a blocking llama.cpp sampler call: the interrupt
            # arrives while control is *inside* here.
            release.wait(10)
            yield GenerationChunk(text=" world")
        finally:
            finished.set()

    svc = _service(stream_completion)
    gen = stream_chunks(_request(), svc)

    # Consume up to the first content chunk. That leaves the generator
    # suspended at the yield *inside* its try/finally — so aclose() below runs
    # the same `finally` a real client disconnect would, which is the whole
    # point. (Breaking out earlier suspends it before the try, and no cleanup
    # runs at all — the interrupt path would go untested.)
    async for event in gen:
        if "hello" in event:
            break

    # ...and the producer thread is now blocked back inside stream_completion.
    for _ in range(500):
        if entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert entered.is_set(), "producer never re-entered stream_completion"

    await gen.aclose()  # the interrupt

    assert svc.unload_auxiliary_models.call_count == 0, (
        "freed the model while the producer thread was still inside llama.cpp"
    )

    # Let the producer finish; only then may the models go.
    release.set()
    for _ in range(500):
        if svc.unload_auxiliary_models.call_count:
            break
        await asyncio.sleep(0.01)

    assert finished.is_set()
    assert svc.unload_auxiliary_models.call_count == 1


@pytest.mark.asyncio
async def test_normal_completion_still_unloads():
    """The happy path must keep freeing the auxiliary models."""
    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        yield GenerationChunk(text="hello")

    svc = _service(stream_completion)

    async for _ in stream_chunks(_request(), svc):
        pass

    for _ in range(500):
        if svc.unload_auxiliary_models.call_count:
            break
        await asyncio.sleep(0.01)
    assert svc.unload_auxiliary_models.call_count == 1
