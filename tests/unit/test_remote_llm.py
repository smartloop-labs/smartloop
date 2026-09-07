"""A per-project agent's orchestrator runs on the host's model, not on nothing.

Routing the final response through a model client was only half of it. The
orchestrator does most of the inference — classification, skill matching,
planning, every DAG decision node — and it takes ``BaseLLM`` objects directly.
It also fails *quietly* without one: ``orchestrate`` logs "BYPASSED" and falls
back to raw generation, so a child with no weights would keep answering while
silently losing skills, tools and planning. These tests are what stops that
being how anyone finds out.
"""

import threading

import pytest
from langchain_core.outputs import GenerationChunk

from smartloop.model_host.remote_llm import (
    DEFAULT_MAX_CONTEXT_TOKENS, RemoteLLM,
)
from smartloop.types import ModelCapability


class FakeClient:
    """The host, as a child sees it."""

    def __init__(self) -> None:
        self.calls: list = []
        self.chunks = ["think", "ing", " done"]
        self.fail_with: Exception | None = None
        self.tokens = 42
        self.context = 8192

    def generate(self, prompt, model_name=None, capability=None):
        self.calls.append(("generate", prompt, capability))
        if self.fail_with:
            raise self.fail_with
        return f"answered {prompt}"

    def stream(self, prompt, model_name=None, capability=None, cancelled=None):
        self.calls.append(("stream", prompt, capability))
        for chunk in self.chunks:
            if cancelled is not None and cancelled.is_set():
                return
            yield chunk

    def count_tokens(self, text):
        self.calls.append(("count_tokens", text))
        if self.fail_with:
            raise self.fail_with
        return self.tokens

    def response_max_context_tokens(self, capability=None):
        self.calls.append(("response_max_context_tokens", capability))
        if self.fail_with:
            raise self.fail_with
        return self.context

    def model_label(self, capability=None):
        self.calls.append(("model_label", capability))
        if self.fail_with:
            raise self.fail_with
        return "sl-mini (2B)"


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def llm(client) -> RemoteLLM:
    return RemoteLLM(client, project_id="alpha")


class TestItLooksLikeAnLLM:
    def test_it_is_a_base_llm(self, llm):
        from smartloop.llms.base_llm import BaseLLM

        assert isinstance(llm, BaseLLM)

    def test_it_names_itself_distinctly(self, llm):
        assert llm._llm_type == "smartloop-remote"

    def test_it_carries_the_project(self, llm):
        assert llm.project_id == "alpha"

    def test_it_inherits_the_prompt_parsing_every_llm_shares(self, llm):
        """The orchestrator builds tagged prompts; those must still parse."""
        messages = llm.create_messages_from_prompt(
            "<system>be brief</system><user>hello</user>"
        )
        assert messages == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hello"},
        ]


class TestGeneration:
    def test_invoke_reaches_the_host(self, llm, client):
        assert llm.invoke("who?") == "answered who?"
        assert ("generate", "who?", None) in client.calls

    def test_an_empty_answer_matches_the_local_contract(self, llm, client):
        client.generate = lambda *a, **k: "   "
        assert llm.invoke("x") == "No response generated"

    def test_a_failure_is_returned_not_raised(self, llm, client):
        """A DAG node treats a bad answer as a dead branch; an exception there
        would take down a whole plan over one step."""
        client.fail_with = RuntimeError("host is gone")
        assert "Error generating response" in llm.invoke("x")

    def test_the_eot_marker_is_trimmed_as_it_is_locally(self, llm, client):
        client.generate = lambda *a, **k: "the answer<|eot_id|>"
        assert llm.invoke("x") == "the answer"

    def test_streaming_yields_generation_chunks(self, llm):
        chunks = list(llm._stream("go"))
        assert all(isinstance(c, GenerationChunk) for c in chunks)
        assert "".join(c.text for c in chunks) == "thinking done"

    def test_cancellation_travels_with_the_stream(self, llm, client):
        cancelled = threading.Event()
        llm._cancelled = cancelled

        produced = []
        for chunk in llm._stream("go"):
            produced.append(chunk.text)
            cancelled.set()

        assert len(produced) < len(client.chunks)


class TestCapabilityStaysOnTheHost:
    def test_the_capability_is_sent_with_the_request(self, client):
        llm = RemoteLLM(client, capability=ModelCapability.VISION)

        llm.invoke("describe this")

        assert ("generate", "describe this", ModelCapability.VISION) in client.calls

    def test_no_capability_means_the_default_model(self, llm, client):
        llm.invoke("x")
        assert ("generate", "x", None) in client.calls

    def test_the_context_size_is_asked_for_this_capability(self, client):
        llm = RemoteLLM(client, capability=ModelCapability.SKILLS)

        llm.get_max_tokens()

        assert ("response_max_context_tokens", ModelCapability.SKILLS) in client.calls


class TestWhatTheUserIsToldTheModelIs:
    """The progress SSE names the answering model; it must not name a class.

    The regression: "Selected RemoteLLM for the response", because both
    ``model_label`` helpers read GGUF metadata off ``llm.model`` — which for a
    remote handle is the transport — and fell through to ``type(llm).__name__``.
    """

    def test_the_handle_reports_the_hosts_label(self, llm):
        assert llm.model_label == "sl-mini (2B)"

    def test_the_service_helper_uses_it(self, llm):
        from smartloop.api.services.model_service import ModelService

        assert ModelService._model_label(llm) == "sl-mini (2B)"

    def test_the_router_helper_uses_it(self, llm):
        from smartloop.agents.orchestration.model_router import ModelRouter

        assert ModelRouter.model_label(llm) == "sl-mini (2B)"

    def test_the_label_is_asked_for_this_capability(self, client):
        RemoteLLM(client, capability=ModelCapability.VISION).model_label
        assert ("model_label", ModelCapability.VISION) in client.calls

    def test_the_sse_message_names_the_model(self, llm):
        from smartloop.agents.orchestration.model_router import ModelRouter

        messages: list = []
        router = ModelRouter.__new__(ModelRouter)
        router.emit = lambda step, status, message: messages.append(message)
        router._event_label = lambda cap, purpose: ("the response", "resp")
        router._model_event_seen = lambda key: False

        router.emit_model_selected(None, llm, purpose="the response")

        assert messages == ["Selected sl-mini (2B) for the response"]
        assert "RemoteLLM" not in messages[0]

    def test_the_label_is_the_model_that_actually_answers(self):
        """Identical to the single-process SSE, not an approximation of it.

        The label must come from the same ``_llm_for_response`` chain the
        answer does — pinned model, capability model, large fallback, default —
        because that chain is what decides which model writes the response.  A
        cheaper "what is resident" guess is free to name a different one.
        """
        from smartloop.api.services.model_service import ModelService

        svc = ModelService.__new__(ModelService)
        asked: list = []

        class _GGUF:
            metadata = {"general.name": "Some Model", "general.size_label": "7B"}

        class _Chosen:
            model = _GGUF()

        def _selection(capability):
            asked.append(capability)
            return _Chosen()

        svc._llm_for_response = _selection

        label = svc.response_model_label(ModelCapability.SKILLS)

        # Same selection call the response makes...
        assert asked == [ModelCapability.SKILLS]
        # ...and the same label the old SSE would have carried for it.
        assert label == ModelService._model_label(_Chosen())

    def test_the_label_follows_the_fallback_rather_than_the_request(self):
        """When the capability model is unavailable the answer falls back —
        and the label has to follow it, or the user is told the wrong model."""
        from smartloop.api.services.model_service import ModelService

        svc = ModelService.__new__(ModelService)

        class _GGUF:
            metadata = {"general.name": "Default Model", "general.size_label": "2B"}

        class _Fallback:
            model = _GGUF()

        # The chain resolved past the requested capability to the default.
        svc._llm_for_response = lambda capability: _Fallback()

        assert svc.response_model_label(ModelCapability.VISION) == \
            ModelService._model_label(_Fallback())

    def test_a_local_llm_still_reads_its_gguf_metadata(self):
        """The explicit label must not shadow the existing path."""
        from smartloop.agents.orchestration.model_router import ModelRouter

        class _GGUF:
            metadata = {"general.name": "Slp Orchestra Mini",
                        "general.size_label": "2B"}

        class _Local:
            model = _GGUF()

        assert ModelRouter.model_label(_Local()) == "sl-mini (2B)"

    def test_an_unreachable_host_falls_back_rather_than_raising(self, llm, client):
        client.fail_with = RuntimeError("gone")
        from smartloop.agents.orchestration.model_router import ModelRouter

        assert ModelRouter.model_label(llm) == "RemoteLLM"


class TestSizing:
    def test_tokens_are_counted_by_the_host(self, llm, client):
        assert llm.get_num_tokens("some text") == 42
        assert ("count_tokens", "some text") in client.calls

    def test_the_context_window_comes_from_the_host(self, llm):
        assert llm.get_max_tokens() == 8192

    def test_an_unreachable_host_counts_zero_rather_than_raising(self, llm, client):
        client.fail_with = RuntimeError("gone")
        assert llm.get_num_tokens("x") == 0

    def test_an_unreachable_host_falls_back_to_a_usable_window(self, llm, client):
        client.fail_with = RuntimeError("gone")
        assert llm.get_max_tokens() == DEFAULT_MAX_CONTEXT_TOKENS

    def test_a_host_reporting_nothing_falls_back(self, llm, client):
        client.context = 0
        assert llm.get_max_tokens() == DEFAULT_MAX_CONTEXT_TOKENS

    def test_the_template_is_left_to_whoever_owns_the_model(self, llm):
        messages = [{"role": "user", "content": "hi"}]
        assert llm._apply_chat_template(messages) == messages


class TestTheServiceHandsOutRemoteHandles:
    """A child must not answer 'no model' and then load one of its own."""

    @pytest.fixture
    def service(self, client, monkeypatch):
        from smartloop.api.services.model_service import ModelService

        svc = ModelService.__new__(ModelService)
        svc._project_caches = {}
        svc._project_caches_lock = threading.Lock()
        svc._model_client = client
        svc._model_client_lock = threading.Lock()
        svc._llm = None
        svc._model = None
        svc._model_name = None

        # The service asks whether the client is a RemoteModelClient; make the
        # fake answer to that question rather than reimplementing the check.
        import smartloop.model_host.client as client_mod
        monkeypatch.setattr(client_mod, "RemoteModelClient", FakeClient)
        return svc

    def test_it_recognises_a_remote_device(self, service, client):
        assert service._remote_device() is client

    def test_an_in_process_client_is_not_remote(self, monkeypatch):
        from smartloop.api.services.model_service import ModelService
        from smartloop.model_host.client import InProcessModelClient

        svc = ModelService.__new__(ModelService)
        svc._model_client = InProcessModelClient(object())
        svc._model_client_lock = threading.Lock()

        assert svc._remote_device() is None

    def test_is_loaded_is_false_until_this_process_holds_a_handle(
        self, service, client,
    ):
        """The regression that produced "[orchestrator] BYPASSED (llm=False)".

        Answering with the host's half alone told the caller there was nothing
        to load.  ``load_model`` then never ran here, ``_llm`` stayed None, and
        the orchestrator fell back to raw generation — losing skills, tools and
        planning without failing.
        """
        client.is_loaded = lambda: True
        service._llm = None

        assert service.is_loaded() is False

    def test_is_loaded_is_true_once_both_halves_hold(self, service, client):
        client.is_loaded = lambda: True
        service._llm = service._remote_llm()

        assert service.is_loaded() is True

    def test_is_loaded_is_false_when_the_host_has_no_weights(self, service, client):
        client.is_loaded = lambda: False
        service._llm = service._remote_llm()

        assert service.is_loaded() is False

    def test_is_loaded_is_false_when_the_host_is_unreachable(self, service, client):
        def boom():
            raise RuntimeError("gone")

        client.is_loaded = boom
        service._llm = service._remote_llm()
        assert service.is_loaded() is False

    def test_token_counting_goes_to_the_host(self, service, client):
        assert service.count_tokens("hello there") == 42

    def test_the_context_window_goes_to_the_host(self, service, client):
        client.max_context_tokens = lambda: 4096
        assert service.get_max_context_tokens() == 4096

    def test_a_capability_request_becomes_a_remote_handle(self, service, client):
        service.get_project_id = lambda: "alpha"

        llm = service._get_llm_for_capability(ModelCapability.VISION)

        assert isinstance(llm, RemoteLLM)
        assert llm.capability == ModelCapability.VISION

    def test_orchestration_binds_a_handle_rather_than_bypassing(
        self, service, client, caplog,
    ):
        """Never degrade to raw generation over a handle we can just build."""
        from smartloop.api.services.model_service import ModelService

        service._model = object()          # workspace model present, LLM not
        service._llm = None
        service._model_name = None
        service.get_project_id = lambda: "alpha"

        # Reach the guard at the top of orchestrate without running the whole
        # pipeline: it is the first thing the method does.
        try:
            ModelService.orchestrate(service, "a question")
        except Exception:
            pass  # the rest of the pipeline is not what this asserts

        assert isinstance(service._llm, RemoteLLM)
        assert "BYPASSED" not in caplog.text

    def test_the_large_model_becomes_a_remote_handle(self, service, client):
        service.get_project_id = lambda: "alpha"

        llm = service._get_large_llm()

        assert isinstance(llm, RemoteLLM)
        assert llm.capability == ModelCapability.SKILLS
