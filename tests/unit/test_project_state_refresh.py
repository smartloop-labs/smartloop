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
from types import SimpleNamespace

import pytest

from smartloop.workspace.context import pin_project
from smartloop.workspace.workspace import (
    Project, ProjectModel, Workspace, project_state_path,
)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SLP_HOME", str(tmp_path))
    monkeypatch.delenv("SLP_LEGACY_CONFIG", raising=False)
    Workspace.reset()
    yield tmp_path
    pin_project(None)
    Workspace.reset()
    # The service is a process-wide singleton; one built here with a temp
    # workspace would otherwise answer for the next test that asks for it.
    from smartloop.api.services.model_service import ModelService
    ModelService._instance = None


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


class TestSeeingAModelRegisteredAfterTheChildStarted:
    """The mirror: the workspace owner's write, reaching a running child.

    Weights are workspace-wide — one shared ``models_dir``, one registry in
    ``config.json``, written only by whoever owns that file.  A child holds the
    workspace it loaded at spawn, and ``_ensure_workspace`` re-reads from disk
    only while no model pipeline exists, so a model downloaded meanwhile was
    invisible to it: missing from ``GET /v1/projects/<id>/models`` and a 404
    from the ``PATCH`` that would enable it, both answered by the child.
    """

    def _seeded(self, home) -> Workspace:
        """A workspace with one model already registered, as on disk."""
        ws = _workspace(home)
        ws.add_model("gemma4-e4b", "google/gemma-4-e4b-it")
        ws.save()
        return ws

    def _owner_registers(self, home, model_name: str) -> None:
        """The process that owns config.json downloads and registers a model."""
        owner = Workspace.load()
        owner.add_model(model_name, f"vendor/{model_name}")
        owner.save()

    def test_the_new_model_arrives(self, home):
        ws = self._seeded(home)
        assert ws.refresh_models() is True          # first look
        assert [m.model_name for m in ws.models] == ["gemma4-e4b"]

        self._owner_registers(home, "ministral3-3b")

        assert ws.refresh_models() is True
        assert ws.get_model("ministral3-3b") is not None

    def test_the_weights_are_found_under_this_home(self, home):
        """Nothing persisted is trusted as a location — same rule as load()."""
        ws = self._seeded(home)
        ws.refresh_models()
        self._owner_registers(home, "ministral3-3b")
        ws.refresh_models()

        assert ws.get_model("ministral3-3b").models_dir == ws.models_dir

    def test_an_unchanged_record_is_not_re_read(self, home):
        ws = self._seeded(home)
        assert ws.refresh_models() is True
        assert ws.refresh_models() is False

    def test_the_project_list_is_left_alone(self, home):
        """Only the registry is taken.

        A child holds project state its siblings own; adopting a sibling's
        record from the shared file would undo the split that keeps one
        process from overwriting another's work.
        """
        ws = self._seeded(home)
        ws.refresh_models()
        before = [p.id for p in ws.projects]
        held = ws.projects[0]

        owner = Workspace.load()
        owner.projects[0].name = "RenamedByTheOwner"
        owner.save()

        ws.refresh_models()
        assert [p.id for p in ws.projects] == before
        assert ws.projects[0] is held
        assert held.name == "Alpha"

    def test_a_malformed_record_keeps_what_we_have(self, home):
        ws = self._seeded(home)
        ws.refresh_models()
        with open(Workspace.config_file, "w") as fh:
            fh.write("{not json")

        assert ws.refresh_models() is False
        assert [m.model_name for m in ws.models] == ["gemma4-e4b"]


class TestWhoRefreshesTheRegistry:
    """Only a child re-reads it.

    The owner must not re-read its own record out from under mutations it has
    not saved yet — ``/v1/init`` registers a model well before it persists one,
    and a refresh triggered by anything else would drop the registration.
    """

    def _service(self, workspace):
        from smartloop.api.services.model_service import ModelService

        ModelService._instance = None
        made = ModelService.__new__(ModelService)
        ModelService.__init__(made)
        made._model = SimpleNamespace(workspace=workspace, save=workspace.save)
        made._ensure_model = lambda: None
        return made

    def test_a_child_picks_up_the_download(self, home):
        ws = _workspace(home)
        ws.add_model("gemma4-e4b", "google/gemma-4-e4b-it")
        ws.save()
        service = self._service(ws)
        pin_project("alpha")
        service._ensure_workspace()

        owner = Workspace.load()          # the owner, unpinned in its own process
        pin_project(None)
        owner.add_model("ministral3-3b", "vendor/ministral3-3b")
        owner.save()
        pin_project("alpha")

        assert service._ensure_workspace().get_model("ministral3-3b") is not None

    def test_the_owner_keeps_its_unsaved_registration(self, home):
        """The /v1/init shape: register, then something else writes the file."""
        ws = _workspace(home)
        ws.add_model("gemma4-e4b", "google/gemma-4-e4b-it")
        ws.save()
        service = self._service(ws)          # unpinned: this process owns the record
        ws.add_model("ministral3-3b", "vendor/ministral3-3b")   # not saved yet

        # Someone else touches config.json in the meantime.
        other = Workspace.load()
        other.save()

        assert service._ensure_workspace().get_model("ministral3-3b") is not None, (
            "the owner re-read its own record and lost an unsaved registration"
        )
