"""Reasoning is a per-model setting, resolved like ``enabled``.

Thinking belongs to the model, not to a request: a model the user switches it
on for in the picker reasons on every call, so the preference is stored on the
workspace model (and overridable per project) and resolved through the same
project -> workspace -> default cascade the ``enabled`` flag uses.
"""

from smartloop.workspace.workspace import Project, Workspace, WorkspaceModel


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
    def test_an_unsupported_model_resolves_to_none(self):
        # None means "bind nothing" — gemma3's template never reads
        # enable_thinking, so a stored preference would do nothing.
        ws = _make_workspace()
        assert ws.effective_model_thinking("gemma3-4b") is None
