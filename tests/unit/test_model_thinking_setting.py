"""Reasoning is a per-model setting, resolved like ``enabled``.

Thinking belongs to the model, not to a request: a model the user switches it
on for in the picker reasons on every call, so the preference is stored on the
workspace model (and overridable per project) and resolved through the same
project → workspace → default cascade the ``enabled`` flag uses.
"""

import pytest

from smartloop.config.app_settings import AppSettings
from smartloop.workspace.workspace import Project, Workspace, WorkspaceModel

BOOTSTRAP_MODEL = AppSettings().default_model


def _make_workspace() -> Workspace:
    """Workspace with one reasoning-capable model and one without."""
    return Workspace(
        id="ws1",
        models=[
            WorkspaceModel(
                model_name="qwen3.5-2b",
                model_id="Qwen/Qwen3.5-2B",
                downloaded=True,
                enabled=True,
            ),
            WorkspaceModel(
                model_name="gemma3-4b",
                model_id="google/gemma-3-4b-it",
                downloaded=True,
                enabled=True,
            ),
        ],
        projects=[Project(id="p1", project_dir="/tmp/p1", current=True, name="p1")],
    )


class TestSupport:
    def test_only_models_declaring_the_capability_support_it(self):
        ws = _make_workspace()
        assert ws.supports_thinking("qwen3.5-2b") is True
        assert ws.supports_thinking("gemma3-4b") is False

    def test_an_unsupported_model_resolves_to_none(self):
        # None means "bind nothing" — gemma3's template never reads
        # enable_thinking, so a stored preference would do nothing.
        ws = _make_workspace()
        assert ws.effective_model_thinking("gemma3-4b") is None


class TestResolution:
    def test_off_until_switched_on(self):
        ws = _make_workspace()
        assert ws.effective_model_thinking("qwen3.5-2b") is False

    def test_the_workspace_setting_turns_it_on(self):
        ws = _make_workspace()
        ws.get_model("qwen3.5-2b").thinking = True
        assert ws.effective_model_thinking("qwen3.5-2b") is True

    def test_a_project_overrides_the_workspace_setting(self):
        ws = _make_workspace()
        project = ws.projects[0]
        ws.get_model("qwen3.5-2b").thinking = True

        ws.set_project_model_thinking(project, "qwen3.5-2b", False)

        assert ws.effective_model_thinking("qwen3.5-2b", project) is False
        assert ws.effective_model_thinking("qwen3.5-2b") is True

    def test_clearing_the_project_override_falls_back(self):
        ws = _make_workspace()
        project = ws.projects[0]
        ws.get_model("qwen3.5-2b").thinking = True
        ws.set_project_model_thinking(project, "qwen3.5-2b", False)

        ws.set_project_model_thinking(project, "qwen3.5-2b", None)

        assert ws.effective_model_thinking("qwen3.5-2b", project) is True

    def test_the_env_setting_applies_when_nothing_is_stored(self, monkeypatch):
        monkeypatch.setenv("SLP_THINKING_ENABLED", "1")
        ws = _make_workspace()
        assert ws.effective_model_thinking("qwen3.5-2b") is True

    def test_a_stored_setting_beats_the_env_setting(self, monkeypatch):
        monkeypatch.setenv("SLP_THINKING_ENABLED", "1")
        ws = _make_workspace()
        ws.get_model("qwen3.5-2b").thinking = False
        assert ws.effective_model_thinking("qwen3.5-2b") is False


class TestPinning:
    def test_moving_the_default_does_not_change_existing_projects(self):
        # Same hazard as ``enabled``: a project with no entry reads through to
        # the workspace default, so switching that default would retroactively
        # turn reasoning on for projects the user never touched.
        ws = _make_workspace()
        project = ws.projects[0]

        ws.pin_model_for_existing_projects("qwen3.5-2b")
        ws.get_model("qwen3.5-2b").thinking = True

        assert ws.effective_model_thinking("qwen3.5-2b", project) is False
        assert ws.effective_model_thinking("qwen3.5-2b") is True


class TestPersistence:
    def test_the_setting_survives_a_round_trip(self):
        ws = _make_workspace()
        ws.get_model("qwen3.5-2b").thinking = True

        reloaded = Workspace.model_validate(ws.model_dump())

        assert reloaded.effective_model_thinking("qwen3.5-2b") is True

    def test_a_config_written_before_the_setting_existed_still_loads(self):
        ws = _make_workspace()
        raw = ws.model_dump()
        for m in raw["models"]:
            m.pop("thinking", None)

        reloaded = Workspace.model_validate(raw)

        assert reloaded.effective_model_thinking("qwen3.5-2b") is False
