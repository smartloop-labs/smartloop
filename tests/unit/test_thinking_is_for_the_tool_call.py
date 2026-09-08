"""Reasoning is spent on the tool call, and nowhere else.

A <think> block is how a tool call is course-corrected into the right tool with
the right arguments: the model reads the catalog, works out which call the
request means and what to pass it, and the monologue is stripped before the
JSON is parsed. By the time the answer is written that decision is made and its
results are already in the context — so reasoning there buys nothing and costs
the stream, because the filter withholds every token of a thought the reader
will never see.

So the flag travels with the request rather than living on the model: the
resolve call asks for it, the answer asks against it, and neither depends on
what the other left behind.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from smartloop.agents.mcp.mcp_agent import MCPAgent, clear_resolution_cache
from smartloop.llms.llama_cpp import LlamaCpp


class _Flag:
    """Stands in for the ``Llama`` the reasoning flag is bound onto."""

    def __init__(self, enabled: bool) -> None:
        self._slp_thinking = enabled


def _wrapper(enabled: bool) -> LlamaCpp:
    llm = LlamaCpp.__new__(LlamaCpp)
    object.__setattr__(llm, "__dict__", {"model": _Flag(enabled)})
    return llm


class TestTheFlagIsScopedToOneCall:
    def test_a_call_that_asks_to_reason_does_not_leave_it_on(self):
        llm = _wrapper(False)

        with llm._thinking_override(True):
            assert llm.model._slp_thinking is True

        assert llm.model._slp_thinking is False

    def test_a_call_that_asks_not_to_reason_does_not_leave_it_off(self):
        """The picker's setting survives an answer that turned it down."""
        llm = _wrapper(True)

        with llm._thinking_override(False):
            assert llm.model._slp_thinking is False

        assert llm.model._slp_thinking is True

    def test_the_setting_is_restored_when_the_call_raises(self):
        llm = _wrapper(False)

        try:
            with llm._thinking_override(True):
                raise RuntimeError("oom")
        except RuntimeError:
            pass

        assert llm.model._slp_thinking is False

    def test_asking_for_nothing_changes_nothing(self):
        llm = _wrapper(True)

        with llm._thinking_override(None):
            assert llm.model._slp_thinking is True

        assert llm.model._slp_thinking is True

    def test_a_model_that_cannot_reason_is_left_alone(self):
        llm = LlamaCpp.__new__(LlamaCpp)
        object.__setattr__(llm, "__dict__", {"model": object()})

        with llm._thinking_override(True):
            pass  # no attribute to set, and no exception either


class TestReasoningIsOptInPerCall:
    """A call that does not ask to reason does not reason.

    Short decision calls were already covered by the budget floor, but
    classification, planning, per-document extraction and synthesis all leave
    room to think and none of them use the thought: they parse a label, a plan
    or a paragraph out of the reply and drop the monologue.  Inheriting the
    model's loaded flag made every one of those calls slower on a model the
    user had turned reasoning on for, with nothing to show for it.
    """

    def test_a_call_that_says_nothing_does_not_reason(self):
        assert LlamaCpp._requested_thinking({}) is False

    def test_an_explicit_none_is_not_an_opt_in(self):
        """What a call carries over the model host socket when nobody asked."""
        assert LlamaCpp._requested_thinking({"thinking": None}) is False

    def test_asking_for_it_is_honoured(self):
        assert LlamaCpp._requested_thinking({"thinking": True}) is True

    def test_the_loaded_flag_does_not_decide_a_call(self):
        llm = _wrapper(True)   # a model loaded with reasoning on

        with llm._thinking_override(llm._requested_thinking({})):
            assert llm.model._slp_thinking is False

        # Restored, so the model is left as it was found.
        assert llm.model._slp_thinking is True


class TestWhoAsksForWhat:
    def test_the_resolve_call_asks_to_reason(self):
        clear_resolution_cache()
        llm = MagicMock()
        llm.invoke = MagicMock(
            return_value='{"tool": 1, "arguments": {"q": "pitch"}}'
        )
        llm.supports_thinking = True
        llm.model_label = "test-model"
        server = MagicMock()
        server.id, server.name = "s1", "notion"
        server.enabled, server.system = True, False
        server.tools = [{
            "name": "notion-search", "description": "Search",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string", "description": "query"}},
                "required": ["q"],
            },
        }]

        MCPAgent(llm).resolve("outline the pitch", [server])
        clear_resolution_cache()

        assert llm.invoke.call_args.kwargs["thinking"] is True

    def test_the_answer_asks_not_to(self):
        """Pinned at the call site, since this is the one the user waits on."""
        import inspect

        from smartloop.api.routers import chat

        src = inspect.getsource(chat)
        assert "cancelled=cancelled, thinking=False," in src, (
            "the streamed answer must ask for no reasoning"
        )
        # …and the non-streaming answer, which shares the model.
        assert src.count("thinking=False") >= 3


class TestTheFlagReachesTheModelHost:
    """A project agent holds a socket, not the weights, so the flag has to
    travel with the request rather than being set on a local model."""

    def test_generate_carries_it(self):
        from smartloop.model_host.remote_llm import RemoteLLM

        client = MagicMock()
        client.generate.return_value = "answered"
        llm = RemoteLLM(client, project_id="p", logger_=logging.getLogger("t"))

        llm.invoke("hello", max_tokens=1024, thinking=True)

        assert client.generate.call_args.kwargs["thinking"] is True
        assert client.generate.call_args.kwargs["max_tokens"] == 1024

    def test_stream_carries_it(self):
        from smartloop.model_host.remote_llm import RemoteLLM

        client = MagicMock()
        client.stream.return_value = iter(["a", "b"])
        llm = RemoteLLM(client, project_id="p", logger_=logging.getLogger("t"))

        list(llm._stream("hello", thinking=False))

        assert client.stream.call_args.kwargs["thinking"] is False

    def test_the_host_serves_what_the_frame_asked_for(self):
        from smartloop.model_host.host import _OPERATIONS

        served = MagicMock()
        served.generate.return_value = "ok"
        _OPERATIONS["generate"](served, {
            "prompt": "hi", "thinking": True, "max_tokens": 1024,
        })

        assert served.generate.call_args.kwargs["thinking"] is True
        assert served.generate.call_args.kwargs["max_tokens"] == 1024


class TestTheServiceForwardsIt:
    def _svc(self, llm):
        import contextlib

        from smartloop.api.services.model_service import ModelService

        svc = ModelService.__new__(ModelService)
        svc._llm_for_response = lambda capability=None: llm
        svc._inference_slot = contextlib.nullcontext
        svc._model_label = staticmethod(lambda _llm: "test-model")
        svc._cap_name = staticmethod(lambda c: str(c))
        return svc

    def test_a_streamed_answer_passes_the_flag_down(self):
        from smartloop.api.services.model_service import ModelService

        llm = MagicMock()
        llm.stream.return_value = iter([SimpleNamespace(text="hi")])
        svc = self._svc(llm)

        list(ModelService.stream_completion(svc, "prompt", thinking=False))

        assert llm.stream.call_args.kwargs["thinking"] is False

    def test_a_non_streamed_answer_passes_it_too(self):
        from smartloop.api.services.model_service import ModelService

        llm = MagicMock()
        llm.invoke.return_value = "answer"
        svc = self._svc(llm)

        ModelService.generate_completion(svc, "prompt", thinking=False)

        assert llm.invoke.call_args.kwargs["thinking"] is False
