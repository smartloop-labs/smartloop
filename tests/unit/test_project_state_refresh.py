"""What a process holding a project sees after its owner writes to it.

Under supervision a project's own state is written by the child pinned to it,
while the supervisor holds a copy taken when it first learned the project
existed.  The model host runs in the supervisor and resolves model selection
from that copy, so nothing invalidating it means a model disabled in a project
keeps answering for that project — observed live: the picker showed one model,
the host loaded the one that had just been turned off, and only a restart
reconciled them.
"""

import json
import os

import pytest

from smartloop.workspace.workspace import (
    Project, ProjectModel, Workspace, project_state_path,
)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SLP_HOME", str(tmp_path))
    monkeypatch.delenv("SLP_LEGACY_CONFIG", raising=False)
    Workspace.reset()
    yield tmp_path
    Workspace.reset()


def _workspace(home) -> Workspace:
    ws_dir = os.path.join(str(home), "ws-1")
    ws = Workspace(id="ws-1", workspace_dir=ws_dir,
                   models_dir=os.path.join(ws_dir, "models"))
    ws.projects = [
        Project(
            id=pid,
            project_dir=os.path.join(ws_dir, pid),
            current=(pid == "alpha"),
            name=pid.title(),
            model_settings=[
                ProjectModel(model_name="gemma4-e4b", enabled=True),
                ProjectModel(model_name="ministral3-3b", enabled=True),
            ],
        )
        for pid in ("alpha", "beta")
    ]
    for project in ws.projects:
        os.makedirs(project.project_dir, exist_ok=True)
    ws.save()
    return ws


def _child_writes(home, project_id: str, **changes) -> None:
    """Simulate the pinned child persisting its own project.json."""
    path = project_state_path(os.path.join(str(home), "ws-1", project_id))
    doc = json.loads(open(path).read())
    doc["project"].update(changes)
    with open(path, "w") as fh:
        json.dump(doc, fh)
    # Guarantee a distinct mtime even on a coarse-grained filesystem clock.
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


class TestPickingUpTheOwnersWrite:
    def test_a_disabled_model_is_seen(self, home):
        ws = _workspace(home)
        alpha = ws.projects[0]
        assert ws.effective_model_enabled("gemma4-e4b", alpha) is True

        _child_writes(home, "alpha", model_settings=[
            {"model_name": "gemma4-e4b", "enabled": False},
            {"model_name": "ministral3-3b", "enabled": True},
        ])

        assert ws.refresh_project("alpha") is True
        assert ws.effective_model_enabled("gemma4-e4b", ws.projects[0]) is False
        assert ws.effective_model_enabled("ministral3-3b", ws.projects[0]) is True

    def test_the_project_object_is_updated_in_place(self, home):
        """A caller already holding the Project must see the new state.

        ``current_project`` hands out the object itself, and the host resolves
        through it while the request runs.  Replacing the list entry would
        leave that caller reading a detached copy.
        """
        ws = _workspace(home)
        held = ws.projects[0]

        _child_writes(home, "alpha", model_settings=[
            {"model_name": "gemma4-e4b", "enabled": False},
        ])
        ws.refresh_project("alpha")

        assert held is ws.projects[0]
        assert [(m.model_name, m.enabled) for m in held.model_settings] == [
            ("gemma4-e4b", False),
        ]

    def test_an_unchanged_file_is_not_re_read(self, home):
        """The host calls this per frame; the usual cost must be one stat."""
        ws = _workspace(home)
        assert ws.refresh_project("alpha") is True    # first look, never seen
        assert ws.refresh_project("alpha") is False
        assert ws.refresh_project("alpha") is False

    def test_the_selection_stays_the_workspaces_answer(self, home):
        """``current`` is workspace-owned; a project's own file cannot claim it."""
        ws = _workspace(home)
        assert ws.projects[0].current is True and ws.projects[1].current is False

        _child_writes(home, "beta", current=True, name="Renamed")
        ws.refresh_project("beta")

        assert ws.projects[1].current is False, "a child claimed the selection"
        assert ws.projects[1].name == "Renamed", "its own fields still arrive"

    def test_an_unknown_project_is_a_no_op(self, home):
        ws = _workspace(home)
        assert ws.refresh_project("nope") is False
        assert ws.refresh_project("") is False

    def test_an_unreadable_file_does_not_raise(self, home):
        ws = _workspace(home)
        path = project_state_path(ws.projects[0].project_dir)
        with open(path, "w") as fh:
            fh.write("{not json")
        assert ws.refresh_project("alpha") is False
        assert ws.projects[0].name == "Alpha", "the copy we have survives"

    def test_two_instances_keep_their_own_vintage(self, home):
        """The mtime map is per instance.

        A shared one lets whichever instance looked first mark the file seen,
        and the other then skips a read it genuinely needs.
        """
        ws = _workspace(home)
        other = Workspace.load()
        _child_writes(home, "alpha", name="Renamed")

        assert ws.refresh_project("alpha") is True
        assert other.refresh_project("alpha") is True
