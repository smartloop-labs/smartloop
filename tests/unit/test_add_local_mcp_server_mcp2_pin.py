"""Auto-fix for MCP SDK 2.x breakage in local MCP server discovery.

``mcp`` 2.0.0 removed ``mcp.server.fastmcp``, so legacy PyPI servers that
declare an unbounded ``mcp>=1.x`` dependency die at startup with
``ModuleNotFoundError: No module named 'mcp.server.fastmcp'``. The discovery
retry must inject ``--with mcp<2`` into the uv invocation and persist the
pinned args into the server config so runtime tool calls use it too.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from smartloop.api.services.model_service import ModelService
from smartloop.mcp.client import MCPConnectResult, MCPDiscoveryError, MCPLaunchError
from smartloop.workspace import MCPToolDefinition

_MCP2_TRACEBACK = (
    "MCP process exited (code 1). stderr: Traceback (most recent call last):\n"
    "  File \"/Users/x/.cache/uv/archive-v0/xxx/bin/weather-mcp-pypi\", line 6, in <module>\n"
    "    from weather_mcp_pypi import main\n"
    "  File \".../weather_mcp_pypi/__init__.py\", line 2, in <module>\n"
    "    from mcp.server.fastmcp import FastMCP\n"
    "ModuleNotFoundError: No module named 'mcp.server.fastmcp'"
)

_MCP2_MCPERROR_TRACEBACK = (
    "MCP process exited (code 1). stderr: port *\n"
    "  File \".../mcp_weather_server/tools/tools_time.py\", line 12, in <module>\n"
    "    from .. import utils\n"
    "  File \".../mcp_weather_server/utils.py\", line 7, in <module>\n"
    "    from mcp import McpError\n"
    "ImportError: cannot import name 'McpError' from 'mcp' (...)"
)

_UVX_HINT_EXEC_MISMATCH = (
    "MCP process exited (code 1). stderr: An executable named `mcp-weather-server` "
    "is not provided by package `mcp-weather-server`.\n"
    "The following executables are available:\n"
    "- mcp_weather_server\n\n"
    "Use `uv tool run --from mcp-weather-server mcp_weather_server` instead."
)

_MCP_TYPES_MISSING = (
    "MCP process exited (code 1). stderr: Traceback (most recent call last):\n"
    "    from mcp_types import Something\n"
    "ModuleNotFoundError: No module named 'mcp_types'"
)

_OK = MCPConnectResult(
    success=True,
    tools=[MCPToolDefinition(name="get_weather", description="d", input_schema={"type": "object"})],
)


def _svc(project):
    svc = MagicMock()
    svc._get_project = MagicMock(return_value=project)
    svc._model = MagicMock()
    # MagicMock would otherwise auto-mock static-method lookups; bind the real
    # helpers so the retry logic runs against actual implementations.
    svc._mcp2_incompat_error = ModelService._mcp2_incompat_error
    svc._pin_mcp1x_args = ModelService._pin_mcp1x_args
    svc._friendly_discovery_error = ModelService._friendly_discovery_error
    return svc


def _project():
    return SimpleNamespace(id="p1", mcp_servers=[])


def _register_and_discover(svc, **kwargs):
    """Register a local server, then run the discovery pass.

    Registration is non-blocking: it normalises the launch config, persists the
    connection and queues provisioning.  The launch retries under test live in
    the discovery step the MCP task runs, so both halves are exercised here the
    way they run in sequence in production.
    """
    ModelService.add_local_mcp_server(svc, **kwargs)
    project = svc._get_project.return_value
    server = project.mcp_servers[-1]
    ModelService._discover_local_mcp_tools(svc, server)
    return server


def _patch_client(side_effect):
    patcher = patch("smartloop.api.services.model_service.LocalMCPClient")
    mock_cls = patcher.start()
    client = MagicMock()
    client.connect_and_discover_tools.side_effect = side_effect
    mock_cls.return_value.__enter__.return_value = client
    return patcher, mock_cls


def test_mcp2_breakage_retries_with_mcp_pin():
    project = _project()
    svc = _svc(project)
    patcher, mock_cls = _patch_client([MCPLaunchError(_MCP2_TRACEBACK), _OK])
    try:
        _register_and_discover(
            svc,
            command="uv",
            args=["tool", "run", "--from", "weather-mcp-pypi", "weather-mcp-pypi"],
            project_id="p1",
        )
    finally:
        patcher.stop()

    assert mock_cls.call_count == 2
    retry_args = mock_cls.call_args_list[1].kwargs["args"]
    assert retry_args == [
        "tool", "run", "--with", "mcp<2",
        "--from", "weather-mcp-pypi", "weather-mcp-pypi",
    ]
    assert project.mcp_servers[0].config.args == retry_args


def test_uvx_bare_command_gets_pin_at_front():
    project = _project()
    svc = _svc(project)
    patcher, mock_cls = _patch_client([MCPLaunchError(_MCP2_TRACEBACK), _OK])
    try:
        _register_and_discover(
            svc,
            command="uvx",
            args=["weather-mcp-pypi"],
            project_id="p1",
        )
    finally:
        patcher.stop()

    assert mock_cls.call_count == 2
    retry_args = mock_cls.call_args_list[1].kwargs["args"]
    assert retry_args == [
        "tool", "run", "--with", "mcp<2",
        "--from", "weather-mcp-pypi", "weather-mcp-pypi",
    ]


def test_no_pin_when_with_flag_already_present():
    project = _project()
    svc = _svc(project)
    args = ["tool", "run", "--from", "pkg", "--with", "mcp>=1", "pkg"]
    patcher, mock_cls = _patch_client([MCPLaunchError(_MCP2_TRACEBACK), _OK])
    try:
        with pytest.raises(MCPDiscoveryError):
            _register_and_discover(
                svc, command="uv", args=args, project_id="p1",
            )
    finally:
        patcher.stop()

    assert mock_cls.call_count == 1


def test_non_uv_launcher_not_pinned():
    project = _project()
    svc = _svc(project)
    patcher, mock_cls = _patch_client([MCPLaunchError(_MCP2_TRACEBACK)])
    try:
        with pytest.raises(MCPDiscoveryError):
            _register_and_discover(
                svc, command="python", args=["-m", "my_server"], project_id="p1",
            )
    finally:
        patcher.stop()

    assert mock_cls.call_count == 1


def test_mcperror_import_breakage_retries_with_mcp_pin():
    project = _project()
    svc = _svc(project)
    patcher, mock_cls = _patch_client([MCPLaunchError(_MCP2_MCPERROR_TRACEBACK), _OK])
    try:
        _register_and_discover(
            svc,
            command="uv",
            args=["tool", "run", "--from", "mcp-weather-server", "mcp-weather-server"],
            project_id="p1",
        )
    finally:
        patcher.stop()

    assert mock_cls.call_count == 2
    retry_args = mock_cls.call_args_list[1].kwargs["args"]
    assert retry_args == [
        "tool", "run", "--with", "mcp<2",
        "--from", "mcp-weather-server", "mcp-weather-server",
    ]
    assert project.mcp_servers[0].config.args == retry_args


def test_mcp_types_missing_not_pinned():
    project = _project()
    svc = _svc(project)
    patcher, mock_cls = _patch_client([MCPLaunchError(_MCP_TYPES_MISSING)])
    try:
        with pytest.raises(MCPDiscoveryError):
            _register_and_discover(
                svc, command="uv", args=["tool", "run", "--from", "pkg", "pkg"], project_id="p1",
            )
    finally:
        patcher.stop()

    assert mock_cls.call_count == 1


def test_hint_fix_then_mcperror_retries_with_pin():
    # The app registers the bare exec name, so the *first* failure is uv's
    # "Use `uv tool run --from <pkg> <exec>` instead" hint. The hint fix swaps
    # in the right executable, but that retry then dies importing McpError —
    # the pin must be evaluated on the retry error too.
    project = _project()
    svc = _svc(project)
    patcher, mock_cls = _patch_client(
        [MCPLaunchError(_UVX_HINT_EXEC_MISMATCH), MCPLaunchError(_MCP2_MCPERROR_TRACEBACK), _OK]
    )
    try:
        _register_and_discover(
            svc,
            command="uv",
            args=["tool", "run", "mcp-weather-server"],
            project_id="p1",
        )
    finally:
        patcher.stop()

    assert mock_cls.call_count == 3
    calls = [c.kwargs["args"] for c in mock_cls.call_args_list]
    assert calls[1] == ["tool", "run", "--from", "mcp-weather-server", "mcp_weather_server"]
    assert calls[2] == [
        "tool", "run", "--with", "mcp<2",
        "--from", "mcp-weather-server", "mcp_weather_server",
    ]
    assert project.mcp_servers[0].config.args == calls[2]
    assert project.mcp_servers[0].name == "mcp-weather-server"
