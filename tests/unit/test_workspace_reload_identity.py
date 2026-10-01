"""``reload()`` must refresh the objects callers hold, not replace them.

A chat request opens with ``Workspace.reload()``.  Background passes — MCP
provisioning, document indexing — hold a ``Project`` or a record inside one
across that reload and write to it afterwards.  While reload rebound
``self.projects`` to a freshly-parsed list, those writes landed on an object
the workspace no longer contained and the next ``save()`` serialized the list
that had replaced them.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from smartloop.workspace import workspace as ws_mod
from smartloop.workspace.workspace import (
    Document, MCPServer, MCPServerStatus, MCPServerType, Project,
    RemoteMCPServerConfig, Workspace, project_state_path,
)


def _workspace(home, servers=None, documents=None):
    return Workspace(
        id="w1",
        workspace_dir=os.path.join(home, "w1"),
        models_dir=os.path.join(home, "w1", "models"),
        projects=[Project(
            id="p1", name="proj", current=True,
            project_dir=os.path.join(home, "w1", "p1"),
            mcp_servers=servers if servers is not None else [],
            documents=documents if documents is not None else [],
        )],
    )


def _server(**kw):
    kw.setdefault("id", "s1")
    kw.setdefault("name", "notion")
    kw.setdefault("server_type", MCPServerType.REMOTE)
    kw.setdefault(
        "remote_config", RemoteMCPServerConfig(server_url="https://mcp.notion.com/mcp"),
    )
    return MCPServer(**kw)


@pytest.fixture
def home(tmp_path):
    """A temp framework home with the class-level config path pointed at it."""
    root = str(tmp_path)
    config_file = os.path.join(root, "config.json")
    with patch.object(Workspace, "config_file", config_file), \
         patch.object(ws_mod, "_settings", SimpleNamespace(home_dir=root)):
        yield root


def _rewrite_on_disk(home, mutate):
    """Stand in for another process writing this project's own file."""
    path = project_state_path(os.path.join(home, "w1", "p1"))
    with open(path) as fh:
        data = json.load(fh)
    mutate(data["project"])
    with open(path, "w") as fh:
        json.dump(data, fh)


class TestTheObjectsSurviveAReload:

    def test_the_project_is_the_same_object(self, home):
        ws = _workspace(home, servers=[_server()])
        ws.save()
        held = ws.projects[0]

        ws.reload()

        assert ws.projects[0] is held

    def test_the_server_is_the_same_object(self, home):
        ws = _workspace(home, servers=[_server()])
        ws.save()
        held = ws.projects[0].mcp_servers[0]

        ws.reload()

        assert ws.projects[0].mcp_servers[0] is held

    def test_a_document_is_the_same_object(self, home):
        ws = _workspace(home, documents=[Document(id="d1", path="/tmp/a.pdf")])
        ws.save()
        held = ws.projects[0].documents[0]

        ws.reload()

        assert ws.projects[0].documents[0] is held

    def test_a_write_after_a_reload_reaches_the_disk(self, home):
        """The bug in its original shape: a pass mutates what it is holding."""
        ws = _workspace(home, servers=[_server()])
        ws.save()
        held = ws.projects[0].mcp_servers[0]

        ws.reload()                       # a chat request, mid-pass
        held.account = "chris@smartloop.ai"
        held.status = MCPServerStatus.READY
        ws.save()

        reloaded = Workspace.load()
        assert reloaded.projects[0].mcp_servers[0].account == "chris@smartloop.ai"


class TestItStillReadsFromDisk:
    """In-place must not mean stale: reload exists to pick up other writers."""

    def test_a_field_another_process_wrote_is_visible_on_the_held_object(self, home):
        ws = _workspace(home, servers=[_server(account=None)])
        ws.save()
        held = ws.projects[0].mcp_servers[0]

        def set_account(project):
            project["mcp_servers"][0]["account"] = "someone@else.com"
        _rewrite_on_disk(home, set_account)
        ws.reload()

        assert held.account == "someone@else.com"

    def test_a_server_added_elsewhere_appears(self, home):
        ws = _workspace(home, servers=[_server()])
        ws.save()

        def add(project):
            second = dict(project["mcp_servers"][0], id="s2", name="linear")
            project["mcp_servers"].append(second)
        _rewrite_on_disk(home, add)
        ws.reload()

        assert [s.id for s in ws.projects[0].mcp_servers] == ["s1", "s2"]

    def test_a_server_deleted_elsewhere_is_dropped(self, home):
        ws = _workspace(home, servers=[_server(), _server(id="s2", name="linear")])
        ws.save()

        def drop(project):
            project["mcp_servers"] = [
                s for s in project["mcp_servers"] if s["id"] != "s1"
            ]
        _rewrite_on_disk(home, drop)
        ws.reload()

        assert [s.id for s in ws.projects[0].mcp_servers] == ["s2"]

    def test_disk_order_wins(self, home):
        ws = _workspace(home, servers=[_server(), _server(id="s2", name="linear")])
        ws.save()

        def swap(project):
            project["mcp_servers"].reverse()
        _rewrite_on_disk(home, swap)
        ws.reload()

        assert [s.id for s in ws.projects[0].mcp_servers] == ["s2", "s1"]


class TestRefreshProjectKeepsThemToo:
    """The same promise on the narrower, mtime-gated path."""

    def test_the_server_survives_a_refresh(self, home):
        ws = _workspace(home, servers=[_server(account=None)])
        ws.save()
        held = ws.projects[0].mcp_servers[0]

        def set_account(project):
            project["mcp_servers"][0]["account"] = "chris@smartloop.ai"
        _rewrite_on_disk(home, set_account)

        assert ws.refresh_project("p1") is True
        assert ws.projects[0].mcp_servers[0] is held
        assert held.account == "chris@smartloop.ai"
