"""A generation that never produces must not hang the client forever.

The orchestration pass has had a wall-clock budget for a while; the response
stream that follows it did not.  Its consumer waited on the producer thread's
queue with a bare ``await queue.get()``, so a producer that never put anything
there — wedged inside llama.cpp, or still queued behind an earlier generation
on the single-worker ``_executor`` — left the request open with no error, no
finish_reason and no ``[DONE]``.  That is the "it gets stuck and stays stuck"
report: the spinner runs until the user kills the client.

``slp_stream_stall_timeout_s`` bounds the *silence*, not the answer: every
chunk restarts the clock, so a long generation never trips it.
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
    svc.model_client = InProcessModelClient(svc)
    return svc


@pytest.fixture(autouse=True)
def _no_telemetry():
    with mock.patch("smartloop.api.routers.chat.get_api_client"):
        yield


@pytest.fixture
def _short_budget(monkeypatch):
    """Half a second of silence stands in for the shipped five minutes."""
    monkeypatch.setenv("SLP_STREAM_STALL_TIMEOUT_S", "0.5")
    yield


def _request():
    # No user message → no orchestration; the response stream is the path
    # under test.
    return ChatCompletionRequest(
        model="sl-mini",
        messages=[ChatMessage(role="system", content="be brief")],
        stream=True,
    )


@pytest.mark.asyncio
async def test_a_producer_that_never_yields_ends_the_stream(_short_budget):
    release = threading.Event()  # never set: the producer is wedged

    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        release.wait(30)
        yield GenerationChunk(text="too late")

    svc = _service(stream_completion)

    events = []
    try:
        async for event in stream_chunks(_request(), svc):
            events.append(event)
    finally:
        release.set()  # let the worker unwind so the pool is usable again

    body = "".join(events)
    assert "Generation stopped." in body, "the user was told nothing"
    assert '"finish_reason":"stop"' in body.replace(" ", "")
    assert events[-1] == "data: [DONE]\n\n", "the stream never terminated"


@pytest.mark.asyncio
async def test_the_wedged_producer_is_told_to_stop(_short_budget):
    """Cooperative cancellation still has to be signalled on the way out."""
    release = threading.Event()
    seen_cancel = threading.Event()

    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        release.wait(30)
        if cancelled is not None and cancelled.is_set():
            seen_cancel.set()
        return
        yield  # pragma: no cover - generator marker

    svc = _service(stream_completion)

    async for _ in stream_chunks(_request(), svc):
        pass

    release.set()
    for _ in range(500):
        if seen_cancel.is_set():
            break
        await asyncio.sleep(0.01)
    assert seen_cancel.is_set(), "the abandoned producer was never cancelled"


@pytest.mark.asyncio
async def test_a_slow_but_live_stream_is_not_cut_off(_short_budget):
    """Each chunk restarts the clock — only silence trips the watchdog."""
    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        for word in ("one ", "two ", "three ", "four "):
            # Each gap is most of the budget; the whole answer is well past it.
            threading.Event().wait(0.35)
            yield GenerationChunk(text=word)

    svc = _service(stream_completion)

    body = "".join([event async for event in stream_chunks(_request(), svc)])

    assert "Generation stopped." not in body, "cut off a stream that was alive"
    for word in ("one", "two", "three", "four"):
        assert word in body


@pytest.mark.asyncio
async def test_the_watchdog_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SLP_STREAM_STALL_TIMEOUT_S", "0")

    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        threading.Event().wait(0.6)  # longer than the budget the others use
        yield GenerationChunk(text="hello")

    svc = _service(stream_completion)

    body = "".join([event async for event in stream_chunks(_request(), svc)])
    assert "hello" in body
    assert "Generation stopped." not in body
