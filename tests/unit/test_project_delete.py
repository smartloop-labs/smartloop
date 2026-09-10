"""Deleting a project, in the process that owns the workspace record.

Removing a project is a write to ``config.json``: the list of projects lives
nowhere else.  A per-project agent child cannot make that write — a pinned
``Workspace.save()`` writes one ``project.json`` and nothing else — and least of
all for the project it is pinned to, which is the one leaving the list.  So the
endpoint is answered by the supervisor, and these are the guarantees that gives.
"""

import json
import os
from types import SimpleNamespace

import pytest

from smartloop.workspace.context import delegate_projects, pin_project
from smartloop.workspace.workspace import (
    MCPServer, Project, Workspace, read_project_state,
)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SLP_HOME", str(tmp_path))
    monkeypatch.delenv("SLP_LEGACY_CONFIG", raising=False)
    Workspace.reset()
    yield tmp_path
    pin_project(None)
    delegate_projects(False)
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
            mcp_servers=[MCPServer(id=f"{pid}-srv", name=f"{pid} server")],
        )
        for pid in ("alpha", "beta")
    ]
    for project in ws.projects:
        os.makedirs(project.documents_dir, exist_ok=True)
    ws.save()
    return ws


@pytest.fixture
def service(home):
    """A ModelService wired to a real workspace but no model.

    ``_ensure_model`` is what would otherwise pull the weights in; everything
    ``delete_project`` touches after it is the workspace and the filesystem.
    """
    from smartloop.api.services.model_service import ModelService

    ModelService._instance = None
    made = ModelService.__new__(ModelService)
    ModelService.__init__(made)
    workspace = _workspace(home)
    made._model = SimpleNamespace(workspace=workspace, save=workspace.save)
    made._ensure_model = lambda: None
    yield made
    ModelService._instance = None


def _config(home) -> dict:
    return json.loads((home / "config.json").read_text())


def _project_ids(home) -> list:
    return [p["id"] for p in _config(home)["workspace"]["projects"]]


class TestTheRecordAndTheDirectoryGoTogether:
    def test_the_project_leaves_the_workspace_record(self, service, home):
        service.delete_project("beta")

        assert _project_ids(home) == ["alpha"]
        assert [p.id for p in Workspace.load().projects] == ["alpha"]

    def test_the_directory_is_removed(self, service, home):
        service.delete_project("beta")

        assert not os.path.exists(os.path.join(str(home), "ws-1", "beta"))

    def test_it_does_not_come_back_on_the_next_load(self, service, home):
        """The regression: the tree went, the record stayed, the UI relisted it."""
        service.delete_project("beta")
        Workspace.reset()

        assert [p.id for p in Workspace.load().projects] == ["alpha"]

    def test_the_other_project_is_untouched(self, service, home):
        service.delete_project("beta")

        alpha = read_project_state(os.path.join(str(home), "ws-1", "alpha"))
        assert [s["name"] for s in alpha["mcp_servers"]] == ["alpha server"]

    def test_deleting_the_selected_project_hands_the_selection_on(self, service, home):
        service.delete_project("alpha")

        assert Workspace.load().current_project.id == "beta"

    def test_the_singleton_is_dropped(self, service, home):
        """Anything resolving the workspace afresh must not still see it."""
        service.delete_project("beta")

        assert Workspace.get_instance() is None

    def test_a_system_project_is_refused(self, service, home):
        service._model.workspace.projects[1].system = True

        with pytest.raises(ValueError, match="System projects"):
            service.delete_project("beta")

        assert os.path.exists(os.path.join(str(home), "ws-1", "beta"))

    def test_an_unknown_project_is_refused(self, service):
        with pytest.raises(ValueError, match="Project not found"):
            service.delete_project("gamma")


class TestUnderSupervision:
    """The supervisor has delegated the project files; it still owns this one."""

    def test_the_delete_persists_while_projects_are_delegated(self, service, home):
        delegate_projects(True)

        service.delete_project("beta")

        assert _project_ids(home) == ["alpha"]
        assert not os.path.exists(os.path.join(str(home), "ws-1", "beta"))

    def test_it_does_not_republish_a_delegated_project(self, service, home):
        """Deleting one project must not write another one's file back."""
        delegate_projects(True)
        # A child adds a server after the service loaded its workspace.
        child = Workspace.load()
        pin_project("alpha")
        child.projects[0].mcp_servers.append(MCPServer(id="new", name="from the child"))
        child.save()
        pin_project(None)

        service.delete_project("beta")

        alpha = read_project_state(os.path.join(str(home), "ws-1", "alpha"))
        assert [s["name"] for s in alpha["mcp_servers"]] == [
            "alpha server", "from the child",
        ]


class TestFindingTheProject:
    def test_a_project_created_by_another_process_can_be_deleted(self, service, home):
        """The cached workspace can predate the CLI, or an import."""
        on_disk = Workspace.load()
        ws_dir = os.path.join(str(home), "ws-1")
        gamma = Project(id="gamma", project_dir=os.path.join(ws_dir, "gamma"),
                        current=False, name="Gamma")
        os.makedirs(gamma.documents_dir, exist_ok=True)
        on_disk.projects.append(gamma)
        on_disk.save()

        service.delete_project("gamma")

        assert _project_ids(home) == ["alpha", "beta"]
        assert not os.path.exists(os.path.join(ws_dir, "gamma"))
