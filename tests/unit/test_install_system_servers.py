"""Regression tests: install_system_tools_server must find an existing record
by its stable id, not its display name.

Records saved under an earlier name (e.g. "Smartloop" before the constant
was unified to "smartloop") were never matched by the name-based lookup, so
their tool list stayed stale — a project created before grep was added kept
advertising a server without it, and tool resolution failed with
"Tool 'grep' not found on server 'Smartloop'".
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from smartloop.api.services.model_service import ModelService
from smartloop.agents.planner import (
    FETCH_SERVER_ID,
    SYSTEM_TOOLS_SERVER_ID,
    SYSTEM_TOOLS_SERVER_NAME,
)
from smartloop.workspace import (
    MCPServer,
    MCPServerConfig,
    MCPServerType,
    MCPToolDefinition,
)


def _svc(project):
    svc = MagicMock()
    svc._get_project = MagicMock(return_value=project)
    svc._SYSTEM_TOOL_DEFS = ModelService._SYSTEM_TOOL_DEFS
    svc._model = MagicMock()
    return svc


def _server(server_id, name, tool_names):
    return MCPServer(
        id=server_id,
        name=name,
        server_type=MCPServerType.LOCAL,
        config=MCPServerConfig(command="python", args=["-m", "x"]),
        enabled=True,
        system=True,
        tools=[
            MCPToolDefinition(name=n, description=n, input_schema={"type": "object"})
            for n in tool_names
        ],
    )


def test_stale_renamed_record_is_refreshed_not_duplicated():
    stale = _server(
        SYSTEM_TOOLS_SERVER_ID, "Smartloop",
        ["bash", "file-read", "file-write", "explore"],  # pre-grep snapshot
    )
    project = SimpleNamespace(id="p1", mcp_servers=[stale])
    svc = _svc(project)

    ModelService.install_system_tools_server(svc, "p1")

    assert len(project.mcp_servers) == 1
    assert stale.name == SYSTEM_TOOLS_SERVER_NAME
    assert "grep" in [t.name for t in stale.tools]
    svc._model.save.assert_called_once()


def test_current_record_is_left_untouched():
    current = _server(
        SYSTEM_TOOLS_SERVER_ID, SYSTEM_TOOLS_SERVER_NAME,
        [t.name for t in ModelService._SYSTEM_TOOL_DEFS],
    )
    project = SimpleNamespace(id="p1", mcp_servers=[current])
    svc = _svc(project)

    ModelService.install_system_tools_server(svc, "p1")

    assert len(project.mcp_servers) == 1
    svc._model.save.assert_not_called()


def test_missing_record_is_registered():
    project = SimpleNamespace(id="p1", mcp_servers=[])
    svc = _svc(project)

    ModelService.install_system_tools_server(svc, "p1")

    assert len(project.mcp_servers) == 1
    installed = project.mcp_servers[0]
    assert installed.id == SYSTEM_TOOLS_SERVER_ID
    assert installed.name == SYSTEM_TOOLS_SERVER_NAME
    assert "grep" in [t.name for t in installed.tools]
    svc._model.save.assert_called_once()


def test_legacy_fetch_server_is_removed():
    # Fetching moved from an MCP server to the in-process FetchAgent, so a
    # previously installed fetch server must be dropped from the project.
    stale = _server(FETCH_SERVER_ID, "fetch", ["fetch"])
    system = _server(
        SYSTEM_TOOLS_SERVER_ID, SYSTEM_TOOLS_SERVER_NAME,
        [t.name for t in ModelService._SYSTEM_TOOL_DEFS],
    )
    project = SimpleNamespace(id="p1", mcp_servers=[stale, system])
    svc = _svc(project)

    ModelService.remove_fetch_server(svc, "p1")

    assert [s.id for s in project.mcp_servers] == [SYSTEM_TOOLS_SERVER_ID]
    svc._model.save.assert_called_once()


def test_remove_fetch_server_noop_when_absent():
    project = SimpleNamespace(id="p1", mcp_servers=[])
    svc = _svc(project)

    ModelService.remove_fetch_server(svc, "p1")

    assert project.mcp_servers == []
    svc._model.save.assert_not_called()


def test_sweep_legacy_fetch_server_removes_and_persists():
    # `remove_fetch_server` only runs for the actively loaded project, so wild
    # projects that are merely listed keep their stale `__fetch__` entry. The
    # sweep on `_get_project` cleans any project on access.
    stale = _server(FETCH_SERVER_ID, "fetch", ["fetch"])
    other = _server("server-a", "Server A", ["do_thing"])
    project = SimpleNamespace(id="p1", mcp_servers=[stale, other])
    svc = _svc(project)

    ModelService._sweep_legacy_fetch_server(svc, project)

    assert [s.id for s in project.mcp_servers] == ["server-a"]
    svc._model.save.assert_called_once()

    # Idempotent: a second sweep neither removes nor saves again.
    ModelService._sweep_legacy_fetch_server(svc, project)
    assert [s.id for s in project.mcp_servers] == ["server-a"]
    svc._model.save.assert_called_once()


def test_sweep_legacy_fetch_server_noop_when_absent():
    project = SimpleNamespace(id="p1", mcp_servers=[])
    svc = _svc(project)

    ModelService._sweep_legacy_fetch_server(svc, project)

    assert project.mcp_servers == []
    svc._model.save.assert_not_called()
