"""One call resolves the tool AND its arguments; the extractor is the fallback.

Resolution and extraction used to be two calls: retrieval picked a tool from
embedded descriptions, then a second prompt worked out the arguments for it
with no knowledge of why it had been picked.  ``MCPAgent.resolve`` answers both
at once, so the step runs what came back instead of asking again — and only
falls back to :class:`MCPArgumentExtractor` when what came back does not cover
what the tool requires.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from smartloop.agents.mcp.argument_extractor import MCPArgumentExtractor
from smartloop.agents.mcp.mcp_agent import ResolvedCall
from smartloop.agents.orchestration.context import _OrchestratorContext
from smartloop.agents.orchestration.step_executor import StepExecutor
from smartloop.agents.planner import PlanStep, StepType
from smartloop.workspace import MCPServerType, MCPToolDefinition

_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "The issue title"},
        "priority": {"type": "number", "description": "1-4, lower is higher"},
    },
    "required": ["title"],
}


@pytest.fixture
def wired():
    """A StepExecutor with everything but resolution and extraction stubbed."""
    server = MagicMock()
    server.id = "srv-1"
    server.name = "linear"
    server.enabled = True
    server.server_type = MCPServerType.LOCAL
    server.tools = [MCPToolDefinition(
        name="create_issue", description="Create an issue", input_schema=_SCHEMA,
    )]

    tool_caller = MagicMock()
    tool_caller.find_server.return_value = server
    tool_caller.find_tool_schema.return_value = {
        "name": "create_issue", "input_schema": _SCHEMA,
    }
    tool_caller.get_server_url.return_value = ""
    tool_caller.validate_argument_values.return_value = []

    dep_analyzer = MagicMock()
    dep_analyzer.invoke.return_value = SimpleNamespace(
        has_dependencies=False, can_execute=False,
    )

    # Real extractor (its URL/id helpers are used inline) with the LLM pass
    # spied on, so "did it ask a second time?" is answerable.
    arg_extractor = MCPArgumentExtractor(MagicMock())
    arg_extractor.invoke = MagicMock(return_value={"title": "from the extractor"})

    mcp_agent = MagicMock()

    executor = StepExecutor(
        orchestrator_llm=MagicMock(),
        tool_caller=tool_caller,
        arg_extractor=arg_extractor,
        dep_analyzer=dep_analyzer,
        result_formatter=MagicMock(),
        explorer=MagicMock(),
        mcp_agent=mcp_agent,
        response=MagicMock(),
    )
    ctx = _OrchestratorContext(
        user_message="file an issue: Login is broken",
        mcp_servers=[server],
        available_servers=[server],
        effective_cwd="/tmp",
        project_dir="/tmp/project",
    )
    step = PlanStep(
        step=1, step_type=StepType.TOOL,
        instruction="file an issue: Login is broken",
    )
    return SimpleNamespace(
        executor=executor, ctx=ctx, step=step, server=server,
        mcp_agent=mcp_agent, arg_extractor=arg_extractor,
    )


def test_a_complete_resolution_is_called_as_returned(wired):
    wired.mcp_agent.resolve.return_value = ResolvedCall(
        server_id="srv-1", tool_name="create_issue",
        arguments={"title": "Login is broken"}, reasoned=True,
    )

    resolved = wired.executor.resolve_tool_args(wired.ctx, wired.step)

    assert resolved == (wired.server, "create_issue", {"title": "Login is broken"})
    wired.arg_extractor.invoke.assert_not_called()


def test_a_partial_resolution_falls_back_to_the_extractor(wired):
    """A required argument still missing is the one case worth a second pass."""
    wired.mcp_agent.resolve.return_value = ResolvedCall(
        server_id="srv-1", tool_name="create_issue", arguments={},
    )

    resolved = wired.executor.resolve_tool_args(wired.ctx, wired.step)

    assert resolved == (wired.server, "create_issue", {"title": "from the extractor"})
    wired.arg_extractor.invoke.assert_called_once()


def test_resolution_reads_what_earlier_steps_returned(wired):
    """A chained call takes its arguments from the prior result, so the prior
    result has to reach the call that picks the tool."""
    wired.mcp_agent.resolve.return_value = ResolvedCall(
        server_id="srv-1", tool_name="create_issue",
        arguments={"title": "Login is broken"},
    )

    wired.executor.resolve_tool_args(
        wired.ctx, wired.step,
        previous_results=[{"tool_name": "list_teams", "result": "team ENG"}],
    )

    context = wired.mcp_agent.resolve.call_args.kwargs["context"]
    assert "list_teams" in context and "team ENG" in context


def test_an_unresolvable_step_is_skipped(wired):
    wired.mcp_agent.resolve.return_value = None
    wired.executor.tool_caller.find_server.return_value = None

    assert wired.executor.resolve_tool_args(wired.ctx, wired.step) is None
