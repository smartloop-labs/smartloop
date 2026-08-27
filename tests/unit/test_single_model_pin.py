"""When the user selects exactly one model, that model runs the whole request.

Capability routing exists to spread a request across several selected models.
With a single selection there is nothing to route between, so every path that
would otherwise fall back to the bootstrap orchestrator model — an unresolved
capability, no larger agent model, a response with no capability tag — resolves
to the selected model instead.  Two or more selections keep today's routing
untouched.

The pin assumes the selected model can hold a text conversation.  One that
declares only ``vision`` (qwen2-vl-2b) cannot, so it is pinned to the vision
steps it was chosen for and nothing else.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.api.services.model_service import ModelService

BOOTSTRAP = "sl-mini"


def _ws_model(name, capabilities, enabled=True, downloaded=True):
    m = MagicMock(name=f"ws_model_{name}")
    m.model_name = name
    m.model_id = f"org/{name}"
    m.effective_capabilities = capabilities
    m._enabled = enabled
    m._downloaded = downloaded
    return m


def _service(*models):
    """A ModelService whose workspace holds *models* plus the bootstrap model."""
    all_models = [_ws_model(BOOTSTRAP, ["general"]), *models]

    workspace = MagicMock(name="workspace")
    workspace.models = all_models
    workspace.current_project = MagicMock(name="project")
    workspace.effective_model_enabled.side_effect = (
        lambda name, project=None: next(
            (m._enabled for m in all_models if m.model_name == name), False)
    )
    workspace.effective_model_downloaded.side_effect = (
        lambda name: next(
            (m._downloaded for m in all_models if m.model_name == name), False)
    )
    # Nothing switched off per project, so the enabled set is the declared set.
    # The pin reads this rather than ``effective_capabilities`` so a capability a
    # project has turned off cannot be pinned — see test_model_service_capabilities.
    workspace.effective_model_capabilities.side_effect = (
        lambda name, project=None: next(
            (list(m.effective_capabilities) for m in all_models
             if m.model_name == name), [])
    )
    workspace.ranked_models_for.side_effect = lambda cap, project=None: [
        m for m in all_models
        if m._enabled and m._downloaded and cap in m.effective_capabilities
    ]

    svc = object.__new__(ModelService)
    svc._model = MagicMock(name="model", workspace=workspace)
    svc._llm = MagicMock(name="bootstrap_llm")
    svc._model_name = BOOTSTRAP
    svc._large_llm = None
    svc._large_model_name = None
    svc._large_model_attempted = False
    svc._detect_large_model = MagicMock(name="_detect_large_model")
    # Stand in for the GGUF resolve + fit guard + llama.cpp load.
    svc._load_workspace_llm = MagicMock(
        name="_load_workspace_llm",
        side_effect=lambda ws_model, reason: MagicMock(name=f"llm_{ws_model.model_name}"),
    )
    return svc


# ── which selection counts ────────────────────────────────────────────────

def test_single_selection_is_pinned():
    svc = _service(_ws_model("big-doc-model", ["general", "dp"]))
    assert svc._pinned_model().model_name == "big-doc-model"


def test_bootstrap_model_does_not_count_as_a_selection():
    """It is always enabled and hidden from the picker, so it never blocks a pin."""
    svc = _service(_ws_model("big-doc-model", ["general", "dp"]))
    assert [m.model_name for m in svc._selected_models()] == ["big-doc-model"]


def test_two_selections_are_not_pinned():
    svc = _service(
        _ws_model("big-doc-model", ["general", "dp"]),
        _ws_model("chat-model", ["general"]),
    )
    assert svc._pinned_model() is None


def test_disabled_model_is_not_a_selection():
    svc = _service(
        _ws_model("big-doc-model", ["general", "dp"]),
        _ws_model("chat-model", ["general"], enabled=False),
    )
    assert svc._pinned_model().model_name == "big-doc-model"


def test_undownloaded_model_is_not_a_selection():
    svc = _service(
        _ws_model("big-doc-model", ["general", "dp"]),
        _ws_model("chat-model", ["general"], downloaded=False),
    )
    assert svc._pinned_model().model_name == "big-doc-model"


def test_no_selection_is_not_pinned():
    svc = _service()
    assert svc._pinned_model() is None


# ── what the pin replaces ─────────────────────────────────────────────────

def test_pinned_model_serves_a_capability_it_does_not_declare():
    """The point of the fix: no fallback to the model the user did not pick."""
    svc = _service(_ws_model("chat-model", ["general"]))

    llm = svc._get_llm_for_capability("dp")

    assert llm is not None
    assert svc._load_workspace_llm.call_args[0][0].model_name == "chat-model"


def test_unloadable_pinned_model_falls_back_rather_than_failing():
    svc = _service(_ws_model("chat-model", ["general"]))
    svc._load_workspace_llm.side_effect = lambda ws_model, reason: None

    assert svc._get_llm_for_capability("dp") is None


def test_pin_is_the_large_agent_model():
    """Even a small selection: _detect_large_model only promotes bigger models."""
    svc = _service(_ws_model("chat-model", ["general"]))

    llm = svc._get_large_llm()

    assert llm is not None
    svc._detect_large_model.assert_not_called()


def test_pin_writes_the_response_when_no_capability_is_tagged():
    svc = _service(_ws_model("chat-model", ["general"]))

    assert svc._llm_for_response(None) is not svc._llm


# ── two or more selections keep today's routing ───────────────────────────

def test_multi_selection_still_routes_by_capability():
    svc = _service(
        _ws_model("big-doc-model", ["general", "dp"]),
        _ws_model("chat-model", ["general"]),
    )

    svc._get_llm_for_capability("dp")

    assert svc._load_workspace_llm.call_args[0][0].model_name == "big-doc-model"


def test_multi_selection_with_no_capable_model_still_falls_back():
    svc = _service(
        _ws_model("big-doc-model", ["general", "dp"]),
        _ws_model("chat-model", ["general"]),
    )

    assert svc._get_llm_for_capability("vision") is None


def test_multi_selection_response_still_uses_the_bootstrap_llm_untagged():
    svc = _service(
        _ws_model("big-doc-model", ["general", "dp"]),
        _ws_model("chat-model", ["general"]),
    )

    assert svc._llm_for_response(None) is svc._llm


def test_multi_selection_large_model_detection_still_runs():
    svc = _service(
        _ws_model("big-doc-model", ["general", "dp"]),
        _ws_model("chat-model", ["general"]),
    )

    svc._get_large_llm()

    svc._detect_large_model.assert_called_once()


# ── a pin that cannot be loaded falls back to the default model ───────────

@pytest.fixture
def broken_pin():
    """One model selected, but it has no GGUF / doesn't fit / faulted."""
    svc = _service(_ws_model("chat-model", ["general"]))
    svc._load_workspace_llm.side_effect = lambda ws_model, reason: None
    return svc


def test_failed_pin_leaves_the_capability_caller_to_fall_back(broken_pin):
    assert broken_pin._get_llm_for_capability("dp") is None


def test_failed_pin_does_not_retry_through_large_model_detection(broken_pin):
    """The pinned model is the only candidate detection could find."""
    assert broken_pin._get_large_llm() is None
    broken_pin._detect_large_model.assert_not_called()


def test_failed_pin_answers_on_the_default_model_untagged(broken_pin):
    assert broken_pin._llm_for_response(None) is broken_pin._llm


def test_failed_pin_answers_on_the_default_model_with_a_capability(broken_pin):
    assert broken_pin._llm_for_response("dp") is broken_pin._llm


# ── a vision-only selection is pinned only to vision work ─────────────────

@pytest.fixture
def vision_only():
    """One model selected, and it declares nothing but ``vision``."""
    return _service(_ws_model("qwen2-vl-2b", ["vision"]))


def test_a_vision_only_selection_is_pinned_for_a_vision_step(vision_only):
    """It is what the user picked it for, so the image step is still its own."""
    assert vision_only._pinned_model("vision").model_name == "qwen2-vl-2b"


def test_a_vision_only_selection_is_not_pinned_for_a_text_step(vision_only):
    """The regression: a vision-only model pinned to 'general' or 'dp' hands
    text work to a model that cannot do it."""
    assert vision_only._pinned_model("general") is None
    assert vision_only._pinned_model("dp") is None


def test_a_vision_only_selection_is_not_pinned_without_a_capability(vision_only):
    """The planner, the skill matcher and the relevance judge ask with no
    capability at all — they are text nodes and must not land on it."""
    assert vision_only._pinned_model() is None
    assert vision_only._pinned_llm() is None


def test_a_text_step_falls_back_to_the_default_model(vision_only):
    """No selected model declares 'dp', so the caller gets None and answers on
    the default model rather than on the vision-only pin."""
    assert vision_only._get_llm_for_capability("dp") is None
    vision_only._load_workspace_llm.assert_not_called()


def test_a_vision_step_still_loads_the_selection(vision_only):
    llm = vision_only._get_llm_for_capability("vision")

    assert llm is not None
    assert vision_only._load_workspace_llm.call_args[0][0].model_name == "qwen2-vl-2b"


def test_the_response_to_a_text_step_is_written_by_the_default_model(vision_only):
    """Nothing selected declares 'dp', so the chain runs out at the default —
    it does not stop at the pin on the way."""
    assert vision_only._llm_for_response(None) is vision_only._llm
    assert vision_only._llm_for_response("dp") is vision_only._llm


def test_a_general_step_resolves_to_the_bootstrap_model_not_the_selection(vision_only):
    vision_only._llm_for_response("general")

    assert vision_only._load_workspace_llm.call_args[0][0].model_name == BOOTSTRAP


def test_the_response_to_a_vision_step_is_written_by_the_selection(vision_only):
    assert vision_only._llm_for_response("vision") is not vision_only._llm


def test_the_vision_only_selection_is_not_the_large_agent_model(vision_only):
    """Agent/synthesis work is text, so detection runs instead of the pin."""
    vision_only._get_large_llm()

    vision_only._detect_large_model.assert_called_once()


def test_a_text_capability_is_not_announced_as_available(vision_only):
    """has_capability_model gates the "Loading model for X…" event: promising a
    load the vision-only pin will not do is the dishonesty it guards against."""
    assert vision_only.has_capability_model("dp") is False
    assert vision_only.has_capability_model("vision") is True


def test_a_general_capable_selection_still_serves_every_capability():
    """The pin is unchanged for anything that declares 'general'."""
    svc = _service(_ws_model("chat-model", ["general"]))

    assert svc._pinned_model("vision").model_name == "chat-model"
    assert svc._pinned_model("dp").model_name == "chat-model"
    assert svc._pinned_model().model_name == "chat-model"
