"""``enable_thinking`` reaches the model's chat template, and vision opts out.

``Llama.create_chat_completion`` has no ``**kwargs`` to forward template
variables through, so the flag is bound onto the resolved chat handler at load
time — that is the only hook a hybrid-reasoning model's template reads, and it
covers the call sites that reach ``create_chat_completion`` directly (vision
Q&A, OCR) as well as the LLM wrapper.
"""

import logging
from unittest.mock import MagicMock

import pytest

from smartloop.llms.llama_cpp import THINKING_MIN_BUDGET, LlamaCpp
from smartloop.llms.llm_factory import LlmFactory


class _Wrapper:
    """``LlamaCpp``'s budget guard over a fake model.

    ``LlamaCpp`` is a pydantic model that needs a real llama.cpp context to
    construct, and the guard only touches ``self.model``.
    """

    _thinking_for_budget = LlamaCpp._thinking_for_budget

    def __init__(self, model):
        self.model = model


class _FakeLlama:
    """Just enough of ``Llama`` for handler resolution."""

    def __init__(self, handler):
        self.chat_handler = handler
        self.chat_format = "chat_template.default"
        self._chat_handlers = {}


def _handler():
    calls = []

    def handler(**kwargs):
        calls.append(kwargs)
        return {"choices": []}

    return handler, calls


def test_bound_flag_is_passed_to_every_completion():
    handler, calls = _handler()
    llm = _FakeLlama(handler)

    LlmFactory._bind_thinking(llm, True, logging.getLogger(__name__))
    llm.chat_handler(llama=llm, messages=[])

    assert calls[0]["enable_thinking"] is True


def test_thinking_can_be_bound_off():
    handler, calls = _handler()
    llm = _FakeLlama(handler)

    LlmFactory._bind_thinking(llm, False, logging.getLogger(__name__))
    llm.chat_handler(llama=llm, messages=[])

    assert calls[0]["enable_thinking"] is False


def test_a_caller_can_still_override_the_bound_default():
    handler, calls = _handler()
    llm = _FakeLlama(handler)

    LlmFactory._bind_thinking(llm, True, logging.getLogger(__name__))
    llm.chat_handler(llama=llm, messages=[], enable_thinking=False)

    assert calls[0]["enable_thinking"] is False


def test_a_short_output_budget_suppresses_reasoning():
    # Decision calls (8-256 tokens for a yes/no) would spend the whole budget
    # inside <think> and return a truncated thought instead of an answer.
    handler, calls = _handler()
    llm = _FakeLlama(handler)
    LlmFactory._bind_thinking(llm, True, logging.getLogger(__name__))

    wrapper = _Wrapper(llm)

    with wrapper._thinking_for_budget(16):
        llm.chat_handler(llama=llm, messages=[])
    assert calls[-1]["enable_thinking"] is False

    # ...and the model's default is back afterwards.
    with wrapper._thinking_for_budget(None):
        llm.chat_handler(llama=llm, messages=[])
    assert calls[-1]["enable_thinking"] is True


def test_a_prompt_that_fills_the_context_suppresses_reasoning():
    # No explicit cap, but a 38k-token prompt left only a sliver of headroom:
    # the guard sees the effective budget, not just what the caller asked for.
    handler, calls = _handler()
    llm = _FakeLlama(handler)
    LlmFactory._bind_thinking(llm, True, logging.getLogger(__name__))

    wrapper = _Wrapper(llm)

    with wrapper._thinking_for_budget(200):
        llm.chat_handler(llama=llm, messages=[])
    assert calls[-1]["enable_thinking"] is False


def test_a_full_output_budget_keeps_reasoning():
    handler, calls = _handler()
    llm = _FakeLlama(handler)
    LlmFactory._bind_thinking(llm, True, logging.getLogger(__name__))

    wrapper = _Wrapper(llm)

    with wrapper._thinking_for_budget(THINKING_MIN_BUDGET * 8):
        llm.chat_handler(llama=llm, messages=[])
    assert calls[-1]["enable_thinking"] is True


def test_a_non_reasoning_model_is_left_alone():
    # Nothing was bound, so the budget guard must not invent a flag.
    handler, calls = _handler()
    llm = _FakeLlama(handler)

    wrapper = _Wrapper(llm)

    with wrapper._thinking_for_budget(16):
        llm.chat_handler(llama=llm, messages=[])
    assert "enable_thinking" not in calls[-1]


def test_binding_failure_leaves_the_model_usable():
    # A llama-cpp that resolves handlers differently must not fail the load —
    # the model just keeps its chat template's own default.
    llm = MagicMock()
    del llm._chat_handlers
    llm.chat_handler = None
    LlmFactory._bind_thinking(llm, True, logging.getLogger(__name__))


def test_a_model_without_the_capability_is_never_bound():
    assert LlmFactory.thinking_for("gemma3-4b") is None
    assert LlmFactory.thinking_for(None) is None


def test_unknown_workspace_model_is_left_alone():
    assert LlmFactory.thinking_for("some-user-added-model") is None


def test_the_bootstrap_model_answers_the_capability_question():
    # Its class declares nothing of its own, which broke every model load with
    # "type object 'GemmaItTextModel' has no attribute 'default_capabilities'".
    # SLPModel now supplies the floor, and the answer is still no.
    assert LlmFactory.supports_thinking("sl-mini") is False
    assert LlmFactory.thinking_for("sl-mini") is None


def test_capabilities_are_never_empty_and_sl_mini_is_unchanged():
    from smartloop.config.supported_models import SUPPORTED_MODELS
    from smartloop.types import ModelCapability
    from smartloop.workspace.workspace import WorkspaceModel

    for name, model_cls in SUPPORTED_MODELS.items():
        assert model_cls.default_capabilities(), name

    # The base default has to match what capability inference already gave a
    # 1B text model, or adding it would silently re-scope the orchestrator.
    sl_mini = WorkspaceModel(
        model_name="sl-mini", model_id="smartloop-ai/slp-orchestra-mini",
    )
    assert sl_mini.effective_capabilities == [ModelCapability.GENERAL]


def test_every_supported_model_resolves():
    # Guards against a model class that answers the capability question but
    # not the default one (or vice versa) breaking loads for that model.
    from smartloop.config.supported_models import SUPPORTED_MODELS

    for name in SUPPORTED_MODELS.keys():
        assert LlmFactory.thinking_for(name) in (True, False, None)


def test_nothing_can_force_reasoning_onto_a_model_without_it():
    # gemma3-4b's template never reads enable_thinking, so honoring the flag
    # would just bind a no-op wrapper and mislead the caller.
    assert LlmFactory.thinking_for("gemma3-4b", override=True) is None
