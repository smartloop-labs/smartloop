"""Regression tests: the built-in "smartloop" system tools server must
resolve through the same ```mcp-block path as third-party servers (e.g.
fetch).

Background: the default skills installed at bootstrap (``file-read``,
``file-write``, ``bash``) carry ```mcp blocks like
``smartloop.file-read(path)``.  The orchestrator resolves those blocks
against ``mcp_tool_contexts``.  Previously the system tools server was
filtered out of that list, so the blocks could never resolve and the
step was silently skipped (``[mcp-block] Server 'smartloop' not found``).

These tests pin the fixed behaviour: the system server is included in
``mcp_tool_contexts`` and ``smartloop.<tool>`` blocks resolve to a
``PlanStep`` targeting the system server.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.agents.orchestrator import SLPOrchestrator
from smartloop.agents.planner import (
    MCPToolContext,
    StepType,
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


def _file_read_context() -> MCPToolContext:
    return MCPToolContext(
        server_id=SYSTEM_TOOLS_SERVER_ID,
        server_name=SYSTEM_TOOLS_SERVER_NAME,
        tools=[{
            "name": "file-read",
            "description": "Read a file and return its content",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }],
    )


def _file_read_block() -> dict:
    """A parsed ```mcp block equivalent to ``smartloop.file-read(path)``."""
    return {
        "server_name": SYSTEM_TOOLS_SERVER_NAME,
        "tool_name": "file-read",
        "inferred_args": ["path"],
        "default_args": {},
    }


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


def test_mcp_block_resolves_system_tool():
    """A ``smartloop.file-read`` block resolves to a system-tools PlanStep
    when the system server is present in the contexts — the same path fetch
    skills already use."""
    steps = SLPOrchestrator._mcp_blocks_to_plan_steps(
        [_file_read_block()],
        skill_name="file-read",
        mcp_tool_contexts=[_file_read_context()],
    )

    assert len(steps) == 1
    step = steps[0]
    assert step.step_type == StepType.TOOL
    assert step.tool_name == "file-read"
    assert step.server_id == SYSTEM_TOOLS_SERVER_ID
    assert step.server_name == SYSTEM_TOOLS_SERVER_NAME


@pytest.mark.parametrize("tool_name", ["file-read", "file-write", "grep", "bash"])
def test_mcp_block_resolves_all_system_tools(tool_name):
    """Every built-in tool — file-read/file-write/grep/bash — resolves to a
    TOOL step on the smartloop server through the unified mcp-block path."""
    ctx = MCPToolContext(
        server_id=SYSTEM_TOOLS_SERVER_ID,
        server_name=SYSTEM_TOOLS_SERVER_NAME,
        tools=[{"name": tool_name, "description": tool_name, "input_schema": {"type": "object"}}],
    )
    block = {
        "server_name": SYSTEM_TOOLS_SERVER_NAME,
        "tool_name": tool_name,
        "inferred_args": [],
        "default_args": {},
    }
    steps = SLPOrchestrator._mcp_blocks_to_plan_steps(
        [block], skill_name=tool_name, mcp_tool_contexts=[ctx],
    )
    assert len(steps) == 1
    assert steps[0].step_type == StepType.TOOL
    assert steps[0].tool_name == tool_name
    assert steps[0].server_id == SYSTEM_TOOLS_SERVER_ID


def test_mcp_block_unresolved_without_system_context():
    """Without the system server in the contexts the block cannot resolve
    and is skipped — the regression this fix prevents (the server is now
    always present via _build_context)."""
    steps = SLPOrchestrator._mcp_blocks_to_plan_steps(
        [_file_read_block()],
        skill_name="file-read",
        mcp_tool_contexts=[],
    )
    assert steps == []
