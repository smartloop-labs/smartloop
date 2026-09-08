"""A tool-synthesis answer streams from its first token.

The opening of such an answer used to be withheld until 20 alphanumeric
characters had been generated, so that a degenerate output (small models often
emit just ```) could be *replaced* rather than retracted.  The retry was worth
having; paying for it with the opening of every answer that turns out fine was
not — and a markdown preamble ("## Summary\n\n**") counted none of its
characters toward the threshold, so the wait had no upper bound the reader
could feel: dead air, then a burst, then normal streaming.

Chunks now go out as they arrive.  A degenerate output is still caught at the
end of the stream, and corrected by appending the retry to what already
reached the client.
"""
import json
import threading

from unittest import mock

import pytest

from langchain_core.outputs import GenerationChunk

from smartloop.api.model.chat import ChatCompletionRequest, ChatMessage
from smartloop.api.routers.chat import stream_chunks
from smartloop.model_host.client import InProcessModelClient

CONTEXT = "TOOL RESULTS:\nContoso Ltd. billed Jordan Rivera for contract work."


def _service(stream_completion, generate=""):
    svc = mock.MagicMock()
    svc.stream_completion.side_effect = stream_completion
    svc.generate_completion.return_value = generate
    # Tool context is what used to turn buffering on.
    svc.orchestrate.return_value = (CONTEXT, None, None, [], None, False)
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
async def test_the_opening_reaches_the_client_while_the_model_is_still_generating():
    """The reader sees the first words before the rest of the answer exists.

    The producer blocks after a markdown preamble that carries only seven
    alphanumerics — under the old 20-character threshold, nothing at all had
    been sent by this point.
    """
    release = threading.Event()
    # Whether the consumer had the opening in hand while the producer was
    # parked — False if it only arrived once generation moved on, which is
    # what withholding it looked like.
    released: list[bool] = []

    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        yield GenerationChunk(text="## ")
        yield GenerationChunk(text="Summary")
        yield GenerationChunk(text="\n\n")
        released.append(release.wait(3))
        yield GenerationChunk(text="Contoso billed Jordan Rivera.")

    svc = _service(stream_completion)
    seen: list[str] = []
    try:
        async for event in stream_chunks(_request(), svc):
            seen.extend(_content([event]))
            if "".join(seen).strip() == "## Summary":
                # Arrived while the producer is parked on `release`.
                release.set()
    finally:
        release.set()

    assert released == [True]
    assert "".join(seen) == "## Summary\n\nContoso billed Jordan Rivera."


@pytest.mark.asyncio
async def test_every_chunk_is_forwarded_as_it_arrives():
    """One delta per generated chunk — nothing is coalesced on the way out."""
    chunks = ["The ", "invoice ", "was ", "for ", "Jordan ", "Rivera."]

    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        for text in chunks:
            yield GenerationChunk(text=text)

    svc = _service(stream_completion)
    raw = [event async for event in stream_chunks(_request(), svc)]
    assert _content(raw) == chunks


@pytest.mark.asyncio
async def test_a_degenerate_answer_is_corrected_by_appending_the_retry():
    """The stray fence is already on the client, so the retry follows it.

    And it follows a *closed* fence: an unclosed ``` swallows everything after
    it in markdown, which would hide the replacement it exists to deliver.
    """
    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        yield GenerationChunk(text="```")

    answer = "Contoso Ltd. billed Jordan Rivera."
    svc = _service(stream_completion, generate=answer)
    raw = [event async for event in stream_chunks(_request(), svc)]

    assert "".join(_content(raw)) == f"```\n```\n\n{answer}"
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
async def test_an_answer_that_is_fine_is_not_retried():
    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        yield GenerationChunk(text="Jordan Rivera.")

    svc = _service(stream_completion, generate="should not be used")
    raw = [event async for event in stream_chunks(_request(), svc)]
    assert "".join(_content(raw)) == "Jordan Rivera."
    svc.generate_completion.assert_not_called()
