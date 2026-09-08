"""The client is told which call is being made, not just which model ran.

A TOOL step's whole decision is *which call*: server, tool, arguments.  The
progress stream carried "Selected <model> for the response" and nothing else,
so a resolution that reasoned for several seconds looked like a stall, and a
call that went out with the wrong arguments was visible only in the server log.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from smartloop.agents.nodes import NodeResult
from smartloop.agents.orchestration.context import _OrchestratorContext
from smartloop.agents.orchestration.pipeline_builder import PipelineBuilder
from smartloop.agents.planner import PlanStep, StepType


@pytest.fixture
def built():
    models = MagicMock()
    steps = MagicMock()
    server = MagicMock()
    server.id = "srv-1"
    server.name = "linear"

    builder = PipelineBuilder(
        models=models,
        planner=MagicMock(),
        explorer=MagicMock(),
        executor=MagicMock(),
        tool_caller=MagicMock(),
        fetch_agent=MagicMock(),
        web_search_agent=MagicMock(),
        judge=MagicMock(),
        doc_fanout=MagicMock(),
        skills=MagicMock(),
        steps=steps,
        tool_fanout=MagicMock(),
        response=MagicMock(),
    )
    ctx = _OrchestratorContext(
        user_message="file an issue: Login is broken",
        mcp_servers=[server],
        available_servers=[server],
        effective_cwd="/tmp",
    )
    step = PlanStep(
        step=1, step_type=StepType.TOOL, tool_name="create_issue",
        server_id="srv-1", server_name="linear",
        instruction="file an issue: Login is broken",
    )
    return SimpleNamespace(
        builder=builder, models=models, steps=steps, ctx=ctx, step=step,
        server=server,
    )


def _messages(models):
    return [c.args[2] for c in models.emit.call_args_list]


def test_resolution_reports_the_call_it_settled_on(built):
    built.steps.resolve_tool_args.return_value = (
        built.server, "create_issue", {"title": "Login is broken"},
    )
    handler = built.builder.make_resolve_handler(
        built.ctx, built.step, search_ctx=MagicMock(),
    )

    handler({})

    assert "Resolved linear.create_issue(title='Login is broken')" in _messages(built.models)


def test_the_wait_is_explained_before_the_call_is_known(built):
    """A reasoning resolution takes seconds; say what is being decided."""
    built.steps.resolve_tool_args.return_value = (
        built.server, "create_issue", {"title": "Login is broken"},
    )
    handler = built.builder.make_resolve_handler(
        built.ctx, built.step, search_ctx=MagicMock(),
    )

    handler({})

    # The capability model here is a MagicMock, which reads as reasoning-capable
    # (the flag attribute is present), so the wait is named as reasoning.
    assert _messages(built.models)[0] == "Reasoning about which tool to call"


def test_a_step_that_resolves_to_nothing_says_so(built):
    built.steps.resolve_tool_args.return_value = None
    handler = built.builder.make_resolve_handler(
        built.ctx, built.step, search_ctx=MagicMock(),
    )

    result = handler({})

    assert result.data["skip"] is True
    assert any("No tool call could be resolved" in m for m in _messages(built.models))


def test_the_call_is_reported_as_it_goes_out(built):
    built.steps.call_tool.return_value = ("formatted", "raw")
    built.builder.tool_caller.find_server.return_value = built.server
    handler = built.builder._make_execute_handler(built.ctx, built.step)

    handler({"resolve": NodeResult(node_id="resolve", success=True, data={
        "skip": False,
        "tool_name": "create_issue",
        "server_id": "srv-1",
        "arguments": {"title": "Login is broken"},
    })})

    assert "Calling linear.create_issue(title='Login is broken')" in _messages(built.models)
