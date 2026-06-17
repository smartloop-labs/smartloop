"""Regression test: the built-in "smartloop" system tools server must be
included in ``mcp_tool_contexts`` so the planner can target it.

Background: the default skills installed at bootstrap (``file-read``,
``file-write``, ``bash``) reference the system tools. Previously the
system tools server was filtered out of ``mcp_tool_contexts``, so the
planner could never resolve those tools. This test pins the fixed
behaviour: the system server is present in ``mcp_tool_contexts``.
"""

from unittest.mock import MagicMock

from smartloop.agents.orchestrator import SLPOrchestrator
from smartloop.agents.planner import (
    SYSTEM_TOOLS_SERVER_ID,
    SYSTEM_TOOLS_SERVER_NAME,
)
from smartloop.workspace import (
    MCPServer,
    MCPServerConfig,
    MCPServerType,
    MCPToolDefinition,
    RemoteMCPServerConfig,
)


def _system_tools_server() -> MCPServer:
    """Mirror the server installed by model_service.install_system_tools_server."""
    return MCPServer(
        id=SYSTEM_TOOLS_SERVER_ID,
        name=SYSTEM_TOOLS_SERVER_NAME,
        description="Built-in tools for file operations, search, and shell commands",
        server_type=MCPServerType.LOCAL,
        config=MCPServerConfig(
            command="python",
            args=["-m", "smartloop.api.mcp.system_server"],
        ),
        enabled=True,
        system=True,
        tools=[
            MCPToolDefinition(
                name="file-read",
                description="Read a file and return its content",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
        ],
    )


def _fetch_server() -> MCPServer:
    return MCPServer(
        id="fetch-srv",
        name="fetch",
        server_type=MCPServerType.REMOTE,
        remote_config=RemoteMCPServerConfig(server_url="https://mcp.example.com"),
        enabled=True,
        tools=[
            MCPToolDefinition(
                name="fetch",
                description="Fetch a URL and return its content as markdown",
                input_schema={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            ),
        ],
    )


def test_build_context_includes_system_tools_server():
    """The system tools server must appear in mcp_tool_contexts alongside
    third-party servers — not be filtered out (the original bug)."""
    orch = SLPOrchestrator(orchestrator_llm=MagicMock())

    ctx = orch._build_context(
        user_message="read /tmp/foo.txt",
        mcp_servers=[_system_tools_server(), _fetch_server()],
        cwd="/tmp",
    )

    server_ids = {c.server_id for c in ctx.mcp_tool_contexts}
    assert SYSTEM_TOOLS_SERVER_ID in server_ids, (
        "system tools server was filtered out of mcp_tool_contexts"
    )
    assert "fetch-srv" in server_ids
