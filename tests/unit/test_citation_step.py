"""Checking the sources is a visible step, and never runs on the interrupt path.

Verification embeds the finished answer, which takes long enough to notice, and
it lands at the worst possible moment: the text has stopped moving and the
citations are not up yet.  Silently, that pause reads as a hang.  So it is
announced like every other phase of the turn — `chat.status`, step
`citations` — and it happens inside the stream, before the client is told the
turn is complete.

It is also deliberately *not* in the generator's `finally`, where it used to
sit.  Announcing it means yielding around it, and that block also runs on
client disconnect, where a yield raises "async generator ignored
GeneratorExit" and takes the session write down with it.  (Awaiting there is
fine; yielding is not.)  An interrupted answer still gets its sources checked
on the way out, with the free lexical test rather than the embedder.
"""
import asyncio
import json
import threading

from unittest import mock

import pytest

from langchain_core.outputs import GenerationChunk

from smartloop.api.model.chat import ChatCompletionRequest, ChatMessage
from smartloop.api.routers.chat import stream_chunks
from smartloop.constants import OPENING_TOKENS
from smartloop.model_host.client import InProcessModelClient

DOC = "invoice-contoso.pdf"
CHUNK = "Invoice from Contoso Ltd. to Jordan Rivera for contract work."


def _citation():
    return {"document_id": DOC, "document_name": DOC,
            "_extract": CHUNK, "_chunks": [CHUNK], "_vectors": [None]}


def _service(stream_completion, citations):
    svc = mock.MagicMock()
    svc.stream_completion.side_effect = stream_completion
    svc.orchestrate.return_value = (
        f"DOCUMENT CONTEXT\n{CHUNK}", None, None, citations, None, False,
    )
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
    # Chat generates through the model client, whose default implementation is
    # the in-process one — so put the real client in front of the fake service
    # and the test drives exactly the path production takes.
    svc.model_client = InProcessModelClient(svc)
    return svc


@pytest.fixture(autouse=True)
def _no_telemetry():
    with mock.patch("smartloop.api.routers.chat.get_api_client"):
        yield


@pytest.fixture(autouse=True)
def _encoder(monkeypatch):
    """No embedding model here — verification takes its lexical fallback.

    Yielded as a spy so a test can assert the embedder was never reached.
    """
    spy = mock.Mock(return_value=None)
    monkeypatch.setattr("smartloop.db.embeddings.encode_texts", spy)
    return spy


def _request(session_id=None):
    return ChatCompletionRequest(
        model="sl-mini",
        messages=[ChatMessage(role="user", content="who was the invoice for?")],
        stream=True,
        session_id=session_id,
    )


def _events(raw: list) -> list:
    """The SSE payloads, decoded, in order."""
    out = []
    for chunk in raw:
        for line in chunk.splitlines():
            if line.startswith("data: "):
                body = line[6:]
                if body != "[DONE]":
                    out.append(json.loads(body))
    return out


def _fake_vectors(texts):
    """Deterministic stand-in vectors — shape is all these tests care about."""
    import numpy as np
    return np.asarray(
        [[float(len(t) % 7) + 1, float(len(t) % 3) + 1] for t in texts],
        dtype=np.float32,
    )


async def _eventually(predicate, timeout=5.0) -> bool:
    """Wait for work that was deliberately fired without being awaited."""
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


def _steps(events: list) -> list:
    return [(e["step"], e["status"]) for e in events
            if e.get("object") == "chat.status"]


async def _run(citations, answer=f"The invoice was for Jordan Rivera, {CHUNK}",
               session_id=None):
    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        yield GenerationChunk(text=answer)

    svc = _service(stream_completion, citations)
    raw = [event async for event in stream_chunks(_request(session_id), svc)]
    return svc, _events(raw)


@pytest.mark.asyncio
async def test_the_source_check_is_announced_as_a_step():
    _, events = await _run([_citation()])
    assert ("citations", "processing") in _steps(events)
    assert ("citations", "completed") in _steps(events)


@pytest.mark.asyncio
async def test_the_step_opens_after_the_text_and_closes_before_the_turn_does():
    # The whole point of the step is to cover the gap between the last token
    # and the citations, so it has to sit in exactly that gap.
    _, events = await _run([_citation()])
    kinds = [
        "content" if e.get("object") == "chat.completion.chunk"
        and e["choices"][0]["delta"].get("content") else
        e.get("step") if e.get("object") == "chat.status" else
        e.get("object")
        for e in events
    ]
    last_content = max(i for i, k in enumerate(kinds) if k == "content")
    opened = kinds.index("citations")
    emitted = kinds.index("chat.citations")
    complete = max(i for i, k in enumerate(kinds) if k == "streaming")
    assert last_content < opened < emitted < complete


@pytest.mark.asyncio
async def test_the_closing_message_reports_what_survived():
    _, events = await _run([_citation()])
    done = [e for e in events
            if e.get("object") == "chat.status" and e["step"] == "citations"
            and e["status"] == "completed"]
    assert done[0]["message"] == "1 source(s) cited"


@pytest.mark.asyncio
async def test_an_answer_that_used_nothing_says_so_and_emits_no_citations():
    _, events = await _run([_citation()], answer="The Hasselblad has a big sensor.")
    done = [e for e in events
            if e.get("object") == "chat.status" and e["step"] == "citations"
            and e["status"] == "completed"]
    assert done[0]["message"] == "No sources used"
    assert not [e for e in events if e.get("object") == "chat.citations"]


@pytest.mark.asyncio
async def test_a_turn_with_no_sources_skips_the_step_entirely():
    _, events = await _run(None)
    assert not [s for s in _steps(events) if s[0] == "citations"]


@pytest.mark.asyncio
async def test_the_verified_set_is_what_reaches_the_session():
    svc, _ = await _run([_citation()], session_id="sess-1")
    # Only the streamed assistant turn carries citations.
    saved = [c for c in svc.append_to_session.call_args_list
             if c.kwargs.get("citations")]
    assert [c["document_name"] for c in saved[0].kwargs["citations"]] == [DOC]
    assert not any(k.startswith("_") for k in saved[0].kwargs["citations"][0])


@pytest.mark.asyncio
async def test_an_interrupt_writes_the_session_without_loading_the_embedder(_encoder):
    """A stopped answer is still checked — with the free test, on the way out."""
    entered = threading.Event()
    release = threading.Event()

    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        yield GenerationChunk(text=f"The invoice was for Jordan Rivera, {CHUNK}")
        entered.set()
        release.wait(10)
        yield GenerationChunk(text=" and more")

    svc = _service(stream_completion, [_citation()])
    gen = stream_chunks(_request(session_id="sess-1"), svc)

    async for event in gen:
        if "Jordan Rivera" in event and "chat.status" not in event:
            break
    for _ in range(500):
        if entered.is_set():
            break
        await asyncio.sleep(0.01)

    await gen.aclose()   # the interrupt

    saved = [c for c in svc.append_to_session.call_args_list
             if c.kwargs.get("citations")]
    assert saved, "the interrupt path lost the session write"
    assert [c["document_name"] for c in saved[0].kwargs["citations"]] == [DOC]
    release.set()


# ---------------------------------------------------------------------------
# When the embedding model is loaded.
#
# From a real run: the GGUF took ~3.4s to come back, and every bit of it landed
# after the last token, where the user is waiting for citations to appear.  Two
# things caused that.  `unload_auxiliary_models` freed the embedder the instant
# generation ended — on the premise that "retrieval runs at the start of a turn
# and is finished well before the response is", which stopped being true when
# verification started needing it at the end.  And nothing loaded it until
# verification asked, by which point there was nothing left to overlap with.
#
#   [embeddings] Loaded llama-cpp:gpustack/bge-m3-GGUF/bge-m3-Q4_K_M.gguf
#   [citations] 'sp923.pdf' distance=0.563 cutoff=0.273 — dropping   (+3.4s)
#
# Verification cannot start before the answer is complete.  The load can.


@pytest.mark.asyncio
async def test_the_producer_keeps_the_embedder_when_there_are_sources_to_check():
    svc, _ = await _run([_citation()])
    assert await _eventually(lambda: svc.unload_auxiliary_models.call_args is not None)
    assert svc.unload_auxiliary_models.call_args.kwargs == {"include_embedding": False}
    # ...and the consumer frees it once verification is done, so the
    # between-chats footprint is unchanged.  Fired without awaiting, so that
    # the citations event does not queue behind a GPU context teardown.
    assert await _eventually(lambda: svc.unload_embedding_model.call_count == 1)


@pytest.mark.asyncio
async def test_a_turn_with_no_sources_frees_the_embedder_as_before():
    svc, _ = await _run(None)
    assert await _eventually(lambda: svc.unload_auxiliary_models.call_args is not None)
    assert svc.unload_auxiliary_models.call_args.kwargs == {"include_embedding": True}


@pytest.mark.asyncio
async def test_an_interrupt_frees_the_embedder_nobody_will_use():
    """The producer left it loaded for a check that is no longer happening."""
    entered = threading.Event()
    release = threading.Event()

    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        yield GenerationChunk(text=f"The invoice was for Jordan Rivera, {CHUNK}")
        entered.set()
        release.wait(10)
        yield GenerationChunk(text=" and more")

    svc = _service(stream_completion, [_citation()])
    gen = stream_chunks(_request(session_id="sess-1"), svc)
    async for event in gen:
        if "Jordan Rivera" in event and "chat.status" not in event:
            break
    for _ in range(500):
        if entered.is_set():
            break
        await asyncio.sleep(0.01)

    await gen.aclose()

    assert await _eventually(lambda: svc.unload_embedding_model.call_count == 1)
    release.set()


@pytest.mark.asyncio
async def test_the_sources_are_embedded_while_the_answer_is_still_streaming():
    """The half of verification that does not depend on the answer."""
    encoded = []
    sources_done = threading.Event()

    def encode(texts, base_dir=None):
        encoded.append(list(texts))
        sources_done.set()
        return _fake_vectors(texts)

    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        yield GenerationChunk(text="The invoice ")
        sources_done.wait(10)      # the sources are being encoded right now
        yield GenerationChunk(text=f"was for Jordan Rivera, {CHUNK}")

    svc = _service(stream_completion, [_citation()])
    with mock.patch("smartloop.db.embeddings.encode_texts", encode):
        [e async for e in stream_chunks(_request(), svc)]

    assert sources_done.is_set(), "sources were not encoded until after the answer"
    assert encoded[0] == [CHUNK]


@pytest.mark.asyncio
async def test_once_the_answer_arrives_only_the_answer_is_embedded():
    # The measured failure: three web citations carry a whole fetched page and
    # no vectors, so verification was encoding two dozen 1200-character windows
    # after the last token, with the user watching.  Pre-embedding leaves this
    # step one encode — the answer, which is the only thing that could not have
    # been done earlier.
    encoded = []

    def encode(texts, base_dir=None):
        encoded.append(list(texts))
        return _fake_vectors(texts)

    answer = f"The invoice was for Jordan Rivera, {CHUNK}"
    with mock.patch("smartloop.db.embeddings.encode_texts", encode):
        await _run([_citation()], answer=answer)

    assert encoded == [[CHUNK], [answer]]


@pytest.mark.asyncio
async def test_a_web_style_source_is_windowed_and_embedded_up_front():
    """A citation with only a fetched page, which is what a web result is."""
    encoded = []

    def encode(texts, base_dir=None):
        encoded.append(list(texts))
        return _fake_vectors(texts)

    page = {"document_id": "https://example.test/x", "url": "https://example.test/x",
            "document_name": "https://example.test/x",
            "_extract": CHUNK + " Navigation, footer, cookie banner. " * 200}
    with mock.patch("smartloop.db.embeddings.encode_texts", encode):
        await _run([page])

    assert len(encoded) == 2, "the page was still being encoded after the answer"
    assert len(encoded[0]) > 1, "a long page should window before it is encoded"


@pytest.mark.asyncio
async def test_an_interrupt_never_embeds_the_answer():
    """Sources may already be encoded; the answer nobody waited for is not."""
    entered = threading.Event()
    release = threading.Event()
    encoded = []

    def encode(texts, base_dir=None):
        encoded.append(list(texts))
        return _fake_vectors(texts)

    answer = f"The invoice was for Jordan Rivera, {CHUNK}"

    def stream_completion(prompt, model_name=None, cancelled=None, capability=None,
                          thinking=None):
        yield GenerationChunk(text=answer)
        entered.set()
        release.wait(10)
        yield GenerationChunk(text=" and more")

    svc = _service(stream_completion, [_citation()])
    with mock.patch("smartloop.db.embeddings.encode_texts", encode):
        gen = stream_chunks(_request(session_id="sess-1"), svc)
        async for event in gen:
            if "Jordan Rivera" in event and "chat.status" not in event:
                break
        assert await _eventually(entered.is_set)
        await gen.aclose()

    assert not any(answer in t for batch in encoded for t in batch)
    release.set()


class TestTheSourceEncodeWaitsForTheOpening:
    """Overlapping the encode with generation is right; starting it first is not.

    The embedder is a second GGUF and it loads onto the same device the answer
    is decoding on — about a second and a half of it, from the server log:

        21:46:47,018  [models] RESPONSE → qwen3.5-2b
        21:46:47,020  [embeddings] Using cached GGUF … bge-m3-Q4_K_M.gguf
        21:46:48,382  [embeddings] Loaded llama-cpp:… (n_gpu_layers=-1)

    Landing that on the first tokens is the one place it shows: the answer
    opens slowly and then speeds up, which reads as the model warming up.
    """

    def _service_with(self, chunks, monkeypatch):
        """(service, started) — *started* is set when the encode is reached."""
        started = threading.Event()

        def prepare(citations, model_service):
            started.set()

        monkeypatch.setattr(
            "smartloop.api.routers.chat._prepare_citation_sources", prepare,
        )

        def stream_completion(prompt, model_name=None, cancelled=None,
                              capability=None, thinking=None):
            for text in chunks:
                yield GenerationChunk(text=text)

        return _service(stream_completion, [_citation()]), started

    @pytest.mark.asyncio
    async def test_it_does_not_start_before_the_opening_is_out(self, monkeypatch):
        # Two hundred one-token chunks: long enough to have an opening and a
        # rest, like the answers this was noticed on.
        svc, started = self._service_with(["tok " for _ in range(200)], monkeypatch)

        stream = stream_chunks(_request(), svc)
        seen = 0
        try:
            async for event in stream:
                deltas = [
                    e for e in _events([event])
                    if (e.get("choices") or [{}])[0].get("delta", {}).get("content")
                ]
                if not deltas:
                    continue
                seen += 1
                if seen == 1:
                    # One token of the answer is out; the device belongs to
                    # the model that is writing it.
                    assert not started.wait(0.25)
                if seen > OPENING_TOKENS:
                    break
        finally:
            await stream.aclose()

        assert started.wait(5), "the encode still has to happen"

    @pytest.mark.asyncio
    async def test_a_short_answer_still_gets_its_sources_encoded(self, monkeypatch):
        """Nothing left to overlap with — but the encode still has to happen."""
        svc, started = self._service_with(["Jordan Rivera."], monkeypatch)

        [event async for event in stream_chunks(_request(), svc)]

        assert started.is_set()
