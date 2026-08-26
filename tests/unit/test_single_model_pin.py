"""When the user selects exactly one model, that model runs the whole request.

Capability routing exists to spread a request across several selected models.
With a single selection there is nothing to route between, so every path that
would otherwise fall back to the bootstrap orchestrator model — an unresolved
capability, no larger agent model, a response with no capability tag — resolves
to the selected model instead.  Two or more selections keep today's routing
untouched.
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
    svc = _service(_ws_model("big-doc-model", ["dp"]))
    assert svc._pinned_model().model_name == "big-doc-model"


def test_bootstrap_model_does_not_count_as_a_selection():
    """It is always enabled and hidden from the picker, so it never blocks a pin."""
    svc = _service(_ws_model("big-doc-model", ["dp"]))
    assert [m.model_name for m in svc._selected_models()] == ["big-doc-model"]


def test_two_selections_are_not_pinned():
    svc = _service(
        _ws_model("big-doc-model", ["dp"]),
        _ws_model("chat-model", ["general"]),
    )
    assert svc._pinned_model() is None


def test_disabled_model_is_not_a_selection():
    svc = _service(
        _ws_model("big-doc-model", ["dp"]),
        _ws_model("chat-model", ["general"], enabled=False),
    )
    assert svc._pinned_model().model_name == "big-doc-model"


def test_undownloaded_model_is_not_a_selection():
    svc = _service(
        _ws_model("big-doc-model", ["dp"]),
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
        _ws_model("big-doc-model", ["dp"]),
        _ws_model("chat-model", ["general"]),
    )

    svc._get_llm_for_capability("dp")

    assert svc._load_workspace_llm.call_args[0][0].model_name == "big-doc-model"


def test_multi_selection_with_no_capable_model_still_falls_back():
    svc = _service(
        _ws_model("big-doc-model", ["dp"]),
        _ws_model("chat-model", ["general"]),
    )

    assert svc._get_llm_for_capability("vision") is None


def test_multi_selection_response_still_uses_the_bootstrap_llm_untagged():
    svc = _service(
        _ws_model("big-doc-model", ["dp"]),
        _ws_model("chat-model", ["general"]),
    )

    assert svc._llm_for_response(None) is svc._llm


def test_multi_selection_large_model_detection_still_runs():
    svc = _service(
        _ws_model("big-doc-model", ["dp"]),
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
