"""A connection whose grant has died must not be offered by any consumer.

Three places build a list of connections for the model: the planner's catalog
(``StepExecutor.build_available_servers``), the collect pass
(``Explorer._run_mcp``) and the MCP agent's tool catalog (``MCPAgent._catalog``).
Each grew its own filter and only the planner's checked ``status``, so a dead
grant went on being offered by the other two — it keeps its cached tool list, so
it looks healthy, and every call 401s into a silent skip.

Driven from one fixture on purpose: the bug was three copies disagreeing, so the
test that matters is the one that asks all three the same question.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from smartloop.agents.mcp.mcp_agent import MCPAgent, clear_resolution_cache
from smartloop.agents.mcp.explorer import Explorer
from smartloop.agents.orchestration.step_executor import StepExecutor
from smartloop.agents.planner import mcp_connection_is_usable


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_resolution_cache()
    yield
    clear_resolution_cache()


def _server(status="ready", enabled=True, server_id="s1"):
    """A remote connection with one discovered tool — the dead-grant shape.

    A connection whose authorization died keeps everything discovery found, so
    the only thing separating it from a healthy one is ``status``.
    """
    return SimpleNamespace(
        id=server_id, name="notion", enabled=enabled, system=False,
        status=status, last_error=None,
        description="Notion", summary=None, server_type="remote",
        catalog_slug=None, pkg=None, remote_config=None, config=None,
        tools=[SimpleNamespace(
            name="search", description="Search pages", input_schema={},
        )],
    )


# --------------------------------------------------------------- the predicate

class TestThePredicate:

    def test_a_failed_connection_is_not_usable(self):
        assert mcp_connection_is_usable(_server(status="failed")) is False

    def test_a_ready_connection_is(self):
        assert mcp_connection_is_usable(_server()) is True

    def test_a_disabled_connection_is_not(self):
        assert mcp_connection_is_usable(_server(enabled=False)) is False

    def test_a_pending_connection_still_is(self):
        """Provisioning has not finished; nothing says it is broken."""
        assert mcp_connection_is_usable(_server(status="pending")) is True

    def test_a_server_with_no_status_is(self):
        """Records written before the field existed, and test stand-ins."""
        bare = SimpleNamespace(id="s1", name="notion", enabled=True)
        assert mcp_connection_is_usable(bare) is True

    def test_the_enum_and_its_bare_value_agree(self):
        """Status arrives as the enum in the app and as a string in fixtures."""
        from smartloop.workspace.workspace import MCPServerStatus
        assert mcp_connection_is_usable(_server(status=MCPServerStatus.FAILED)) is False
        assert mcp_connection_is_usable(_server(status=MCPServerStatus.READY)) is True


# --------------------------------------------------------------- the consumers

class TestNoConsumerOffersIt:

    def test_the_planner_catalog_leaves_it_out(self):
        executor = StepExecutor.__new__(StepExecutor)
        assert executor.build_available_servers([_server()]) != []
        assert executor.build_available_servers([_server(status="failed")]) == []

    def test_the_mcp_agent_catalog_leaves_it_out(self):
        agent = MCPAgent.__new__(MCPAgent)
        assert agent._catalog([_server()]) != []
        assert agent._catalog([_server(status="failed")]) == []

    def test_the_collect_pass_leaves_it_out(self):
        """The path the dead-grant commit called the main one."""
        explorer = Explorer.__new__(Explorer)
        explorer.arg_extractor = MagicMock()
        explorer.tool_caller = MagicMock()
        explorer.mcp_agent = MagicMock()

        result = explorer._run_mcp(
            "what is in notion?", [_server(status="failed")], project_dir=None,
        )

        assert result == ""
        # Not merely an empty answer — it must never have been asked.
        explorer.mcp_agent.resolve.assert_not_called()
        explorer.tool_caller.call_tool.assert_not_called()
