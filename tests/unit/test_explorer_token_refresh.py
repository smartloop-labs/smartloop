"""The collect path renews an expired MCP token, like every other call path.

``Explorer._run_mcp`` called the tool with whatever token was stored and returned
``""`` on failure.  Every other MCP call site — both ``StepExecutor`` paths and
the content crawl — retries once behind a refresh, so a connected app whose token
had simply aged out looked to retrieval like an app with nothing in it, while the
same app answered a planned execute step perfectly well.
"""

from unittest.mock import MagicMock, patch

from smartloop.agents.mcp.explorer import Explorer
from smartloop.mcp.client import MCPToolCallResult


def _text(body):
    return MCPToolCallResult(success=True, content=[{"type": "text", "text": body}])


_UNAUTHORIZED = MCPToolCallResult(
    success=False, error="Authentication failed (401).", auth_expired=True,
)


class _Server:
    id = "s1"
    name = "monday"
    tools = []


def _explorer(tool_caller):
    arg_extractor = MagicMock()
    arg_extractor.invoke = MagicMock(return_value={})
    explorer = Explorer(MagicMock(), tool_caller, arg_extractor)
    explorer.mcp_agent = MagicMock()
    explorer.mcp_agent.select_tool = MagicMock(return_value=("s1", "get_board"))
    return explorer


def _run(explorer):
    with patch.object(Explorer, "_is_read_only_tool", return_value=True):
        return explorer._run_mcp("what is on my board", [_Server()], "/tmp/project")


class TestAnExpiredTokenIsRenewed:
    def test_the_call_is_retried_on_the_fresh_token(self):
        caller = MagicMock()
        caller.get_access_token = MagicMock(return_value="expired")
        caller.call_tool = MagicMock(side_effect=[_UNAUTHORIZED, _text("Sprint board")])
        caller.refresh_access_token = MagicMock(return_value="fresh")

        assert "Sprint board" in _run(_explorer(caller))

        caller.refresh_access_token.assert_called_once()
        assert caller.call_tool.call_args_list[1].kwargs["access_token"] == "fresh"

    def test_nothing_to_refresh_with_is_still_a_silent_skip(self):
        caller = MagicMock()
        caller.get_access_token = MagicMock(return_value="expired")
        caller.call_tool = MagicMock(return_value=_UNAUTHORIZED)
        caller.refresh_access_token = MagicMock(return_value=None)

        assert _run(_explorer(caller)) == ""
        assert caller.call_tool.call_count == 1

    def test_an_ordinary_failure_does_not_spend_a_refresh_token(self):
        """Only a real 401 sets ``auth_expired``; a tool error is not one."""
        caller = MagicMock()
        caller.get_access_token = MagicMock(return_value="good")
        caller.call_tool = MagicMock(
            return_value=MCPToolCallResult(success=False, error="board 401 not found"),
        )
        caller.refresh_access_token = MagicMock()

        assert _run(_explorer(caller)) == ""
        caller.refresh_access_token.assert_not_called()
