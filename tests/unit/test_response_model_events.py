"""The model that writes the answer reports itself, like every other one.

The response model is resolved by the streaming layer *after* orchestration
finishes, so it never went through ModelRouter and surfaced nothing at all: a
cold load sat between the last orchestration event and the first token, as
dead air exactly where the answer was expected to start.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.api.services.model_service import ModelService
from smartloop.types import ModelCapability


def _llm(name):
    llm = MagicMock()
    llm.model.metadata = {"general.name": name, "general.size_label": ""}
    return llm


def _service(*, capability_llm=None, available=True, pinned=None,
             pinned_llm=None, large_llm=None, skip=None):
    svc = object.__new__(ModelService)
    svc._llm = _llm("Slp Orchestra Mini")
    svc._response_router_instance = None
    svc._get_llm_for_capability = MagicMock(return_value=capability_llm)
    svc._get_large_llm = MagicMock(return_value=large_llm)
    svc.has_capability_model = MagicMock(return_value=available)
    svc.last_capability_skip = MagicMock(return_value=skip)
    svc._pinned_model = MagicMock(return_value=pinned)
    svc._pinned_llm = MagicMock(return_value=pinned_llm)
    return svc


def _events(svc):
    sink = MagicMock()
    svc.begin_response_request(sink)
    return sink


def _messages(sink):
    return [c.args[2] for c in sink.call_args_list]


class TestTheLoadIsAnnouncedFirst:
    def test_a_pending_capability_load_warns_before_it_blocks(self):
        svc = _service(available=True)
        sink = _events(svc)

        svc.announce_response_model(ModelCapability.GENERAL)

        assert _messages(sink) == ["Loading model for the response…"]

    def test_nothing_is_announced_when_no_model_provides_it(self):
        svc = _service(available=False)
        sink = _events(svc)

        svc.announce_response_model(ModelCapability.GENERAL)

        assert _messages(sink) == []

    def test_a_pinned_model_is_announced_even_without_a_capability(self):
        """A pin loads whatever the request asked for, capability or not."""
        svc = _service(available=False, pinned=MagicMock())
        sink = _events(svc)

        svc.announce_response_model(None)

        assert _messages(sink) == ["Loading model for the response…"]

    def test_plain_chat_announces_nothing(self):
        svc = _service(available=False)
        sink = _events(svc)

        svc.announce_response_model(None)

        assert _messages(sink) == []


class TestTheChosenModelIsNamed:
    def test_a_capability_model_is_named(self):
        svc = _service(capability_llm=_llm("Gemma 3n E2B"))
        sink = _events(svc)

        svc._llm_for_response(ModelCapability.GENERAL)

        assert _messages(sink) == ["Selected Gemma 3n E2B for the response"]

    def test_the_large_model_fallback_is_named(self):
        svc = _service(capability_llm=None, large_llm=_llm("Ministral 3B"))
        sink = _events(svc)

        svc._llm_for_response(ModelCapability.GENERAL)

        assert _messages(sink) == ["Selected Ministral 3B for the response"]

    def test_a_pinned_model_is_named(self):
        svc = _service(pinned=MagicMock(), pinned_llm=_llm("Gemma 3n E2B"))
        sink = _events(svc)

        svc._llm_for_response(None)

        assert _messages(sink) == ["Selected Gemma 3n E2B for the response"]

    def test_the_capability_tag_never_reaches_the_user(self):
        svc = _service(capability_llm=_llm("Gemma 3n E2B"))
        sink = _events(svc)

        svc._llm_for_response(ModelCapability.GENERAL)

        assert not any("general" in m.lower() for m in _messages(sink))


class TestFallingBackSaysWhy:
    def test_memory_is_named(self):
        svc = _service(capability_llm=None, skip=("gemma4-e2b", "memory"))
        sink = _events(svc)

        svc._llm_for_response(ModelCapability.GENERAL)

        assert _messages(sink) == [
            "Selected sl-mini — not enough memory for gemma4-e2b"
        ]

    def test_a_failed_pin_also_falls_back_with_a_reason(self):
        svc = _service(
            pinned=MagicMock(), pinned_llm=None, skip=("gemma4-e2b", "memory"),
        )
        sink = _events(svc)

        svc._llm_for_response(ModelCapability.GENERAL)

        assert "not enough memory" in _messages(sink)[0]

    def test_plain_chat_on_the_resident_model_says_nothing(self):
        """Nothing loaded and nothing overridden — there is nothing to report."""
        svc = _service()
        sink = _events(svc)

        svc._llm_for_response(None)

        assert _messages(sink) == []


class TestPerRequestState:
    def test_each_request_reports_its_model_again(self):
        svc = _service(capability_llm=_llm("Gemma 3n E2B"))

        first = _events(svc)
        svc._llm_for_response(ModelCapability.GENERAL)
        second = _events(svc)
        svc._llm_for_response(ModelCapability.GENERAL)

        assert _messages(first) == ["Selected Gemma 3n E2B for the response"]
        assert _messages(second) == ["Selected Gemma 3n E2B for the response"]

    def test_one_request_reports_once(self):
        """stream_completion resolves the model again after sizing the window."""
        svc = _service(capability_llm=_llm("Gemma 3n E2B"))
        sink = _events(svc)

        svc._llm_for_response(ModelCapability.GENERAL)
        svc._llm_for_response(ModelCapability.GENERAL)

        assert _messages(sink) == ["Selected Gemma 3n E2B for the response"]

    def test_the_router_tracks_a_model_switch(self):
        svc = _service(capability_llm=None)
        _events(svc)
        svc._llm = _llm("Ministral 3B")

        svc._llm_for_response(ModelCapability.GENERAL)

        assert svc._response_router.orchestrator_llm is svc._llm


def test_capability_names_are_logged_by_value_not_member():
    """A log line naming ModelCapability.GENERAL is one nobody can grep for."""
    assert ModelService._cap_name(ModelCapability.GENERAL) == "general"
    assert ModelService._cap_name("dp") == "dp"
    assert ModelService._cap_name(None) == "None"
