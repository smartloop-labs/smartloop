"""The opening of a tool-synthesis answer is buffered until it looks real.

Small models often answer a tool payload with nothing but ``` and an EOS.  The
retry that fixes that is only a *replacement* while nothing has reached the
client yet — once the fence is out, the correction can only be appended after
it, which reads as two answers.  So the opening is held until 20 alphanumeric
characters say the model is actually writing, then flushed and streamed
normally.  Streaming from the first token was tried and did not measurably
improve how the answer opens; the retry is worth the held opening.

Regular chat — no tool context — streams from the first chunk either way.
"""
import json
import time

from unittest import mock

import pytest

from langchain_core.outputs import GenerationChunk

from smartloop.api.model.chat import ChatCompletionRequest, ChatMessage
from smartloop.api.routers.chat import stream_chunks
from smartloop.model_host.client import InProcessModelClient

CONTEXT = "TOOL RESULTS:\nContoso Ltd. billed Jordan Rivera for contract work."


def _service(stream_completion, generate="", context=CONTEXT):
    svc = mock.MagicMock()
    svc.stream_completion.side_effect = stream_completion
    svc.generate_completion.return_value = generate
    # Tool context is what turns buffering on.
    svc.orchestrate.return_value = (context, None, None, [], None, False)
    svc.get_relevant_history.return_value = []
    svc.get_project_skills.return_value = []
    svc.get_asset_images.return_value = []
    svc.get_project_info.return_value = None
    svc.get_project_id.return_value = "proj"
    svc.count_tokens.return_value = 10
    svc.get_response_max_context_tokens.return_value = 4096
    svc.get_max_context_tokens.return_value = 4096
    svc.maybe_write_output.return_value = None
    svc._model.workspace.workspace_dir = "/tmp/ws"
    svc.model_client = InProcessModelClient(svc)
    return svc


@pytest.fixture(autouse=True)
def _no_telemetry():
    with mock.patch("smartloop.api.routers.chat.get_api_client"):
        yield


def _request():
    return ChatCompletionRequest(
        model="sl-mini",
        messages=[ChatMessage(role="user", content="who was the invoice for?")],
        stream=True,
    )


def _content(raw: list) -> list[str]:
    """The content deltas, in order."""
    out = []
    for chunk in raw:
        for line in chunk.splitlines():
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            event = json.loads(line[6:])
            for choice in event.get("choices") or []:
                text = (choice.get("delta") or {}).get("content")
                if text:
                    out.append(text)
    return out


@pytest.mark.asyncio
async def test_a_degenerate_answer_is_replaced_not_appended():
    """The stray fence never reaches the client, so the retry stands alone."""
    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        yield GenerationChunk(text="```")

    answer = "Contoso Ltd. billed Jordan Rivera."
    svc = _service(stream_completion, generate=answer)
    raw = [event async for event in stream_chunks(_request(), svc)]

    assert "".join(_content(raw)) == answer
    # The retry asks with the tool payload and a minimal prompt, not the
    # original one.
    prompt = svc.generate_completion.call_args.args[0]
    assert "Contoso Ltd. billed Jordan Rivera for contract work." in prompt


@pytest.mark.asyncio
async def test_a_retry_that_is_also_degenerate_falls_back_to_guidance():
    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        yield GenerationChunk(text="```")

    svc = _service(stream_completion, generate="``")
    raw = [event async for event in stream_chunks(_request(), svc)]
    assert "**No response generated.**" in "".join(_content(raw))


@pytest.mark.asyncio
async def test_the_opening_is_held_then_flushed_once_the_answer_looks_real():
    """Nothing goes out on the preamble; the whole opening arrives together.

    The producer pauses after a markdown preamble carrying seven alphanumerics
    — under the threshold — and the consumer, whose loop is free for that
    whole pause, must still have received nothing.  What follows crosses the
    threshold, and the held chunks are flushed in order.
    """
    seen: list[str] = []
    # What the consumer had in hand while the producer was paused.
    at_pause: list[list[str]] = []

    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        yield GenerationChunk(text="## ")
        yield GenerationChunk(text="Summary")
        yield GenerationChunk(text="\n\n")
        time.sleep(0.3)
        at_pause.append(list(seen))
        yield GenerationChunk(text="Contoso billed Jordan Rivera today.")

    svc = _service(stream_completion)
    async for event in stream_chunks(_request(), svc):
        seen.extend(_content([event]))

    assert at_pause == [[]]
    assert seen == ["## ", "Summary", "\n\n", "Contoso billed Jordan Rivera today."]


@pytest.mark.asyncio
async def test_regular_chat_streams_from_the_first_chunk():
    """No tool context, no buffering — the deltas are the generated chunks."""
    chunks = ["The ", "invoice ", "was ", "for ", "Jordan ", "Rivera."]

    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        for text in chunks:
            yield GenerationChunk(text=text)

    svc = _service(stream_completion, context=None)
    raw = [event async for event in stream_chunks(_request(), svc)]
    assert _content(raw) == chunks
