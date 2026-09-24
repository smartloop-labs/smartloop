"""Every MCP call path renews an expired token, through one shared helper.

``Explorer._call_mcp_tool`` — the main MCP answering path since indexing was
replaced by live tool calls — called the tool with whatever token was stored and
returned ``""`` on failure, where every failure is a deliberate silent skip.  So
a connected app whose token had simply aged out looked to retrieval like an app
with nothing in it, while the same app answered a planned execute step fine
because *that* path retried.

The retry now lives in :meth:`MCPToolCaller.call_with_refresh`, so the six call
paths cannot drift apart again — which they had: two guarded on a server URL,
one retried with the wrong tool name, and this one had no retry at all.
"""

from unittest.mock import MagicMock, patch

from smartloop.agents.mcp.explorer import Explorer
from smartloop.agents.mcp.mcp_tool_caller import MCPToolCaller
from smartloop.mcp.client import MCPToolCallResult


def _text(body):
    return MCPToolCallResult(success=True, content=[{"type": "text", "text": body}])


_UNAUTHORIZED = MCPToolCallResult(
    success=False, error="Authentication failed (401).", auth_expired=True,
)


def _server(access_token="expired", refresh_token="rt"):
    from smartloop.workspace.workspace import (
        MCPOAuthCredentials, MCPServer, MCPServerType, RemoteMCPServerConfig,
    )
    return MCPServer(
        id="s1", name="monday", server_type=MCPServerType.REMOTE,
        remote_config=RemoteMCPServerConfig(
            server_url="https://example/mcp",
            token_endpoint="https://example/token",
        ),
        oauth_credentials=MCPOAuthCredentials(
            client_id="c", access_token=access_token, refresh_token=refresh_token,
        ),
    )


class TestTheHelperRenewsOnce:
    def test_the_call_is_retried_on_the_fresh_token(self):
        caller = MCPToolCaller(None, None)
        caller.call_tool = MagicMock(side_effect=[_UNAUTHORIZED, _text("Sprint board")])
        caller.refresh_access_token = MagicMock(return_value="fresh")

        result, refreshed = caller.call_with_refresh(_server(), "get_board", {})

        assert refreshed is True
        assert result.content[0]["text"] == "Sprint board"
        assert caller.call_tool.call_args_list[1].kwargs["access_token"] == "fresh"

    def test_the_failed_token_is_offered_to_the_refresh(self):
        """So a redemption that lost the race can return the winner's token."""
        caller = MCPToolCaller(None, None)
        caller.call_tool = MagicMock(return_value=_UNAUTHORIZED)
        caller.refresh_access_token = MagicMock(return_value=None)

        caller.call_with_refresh(_server(access_token="expired"), "get_board", {})

        assert caller.refresh_access_token.call_args.kwargs["stale_token"] == "expired"

    def test_nothing_to_refresh_with_is_still_a_silent_skip(self):
        caller = MCPToolCaller(None, None)
        caller.call_tool = MagicMock(return_value=_UNAUTHORIZED)
        caller.refresh_access_token = MagicMock(return_value=None)

        result, refreshed = caller.call_with_refresh(_server(), "get_board", {})

        assert result.success is False
        assert refreshed is False
        assert caller.call_tool.call_count == 1

    def test_an_ordinary_failure_does_not_spend_a_refresh_token(self):
        """Only a real 401 sets ``auth_expired``; a tool error is not one."""
        caller = MCPToolCaller(None, None)
        caller.call_tool = MagicMock(
            return_value=MCPToolCallResult(success=False, error="board 401 not found"),
        )
        caller.refresh_access_token = MagicMock()

        result, refreshed = caller.call_with_refresh(_server(), "get_board", {})

        assert refreshed is False
        caller.refresh_access_token.assert_not_called()


class TestTheCollectPathUsesIt:
    """The path that had no retry at all now goes through the helper."""

    def _explorer(self, caller):
        explorer = Explorer(MagicMock(), caller, MagicMock())
        return explorer

    def test_an_expired_token_still_returns_the_records(self):
        caller = MCPToolCaller(None, None)
        caller.call_tool = MagicMock(side_effect=[_UNAUTHORIZED, _text("Sprint board")])
        caller.refresh_access_token = MagicMock(return_value="fresh")

        text = self._explorer(caller)._call_mcp_tool(
            _server(), "get_board", {"id": "1"}, "what is on my board",
        )

        assert "Sprint board" in text
        caller.refresh_access_token.assert_called_once()

    def test_a_dead_grant_is_still_a_silent_skip(self):
        caller = MCPToolCaller(None, None)
        caller.call_tool = MagicMock(return_value=_UNAUTHORIZED)
        caller.refresh_access_token = MagicMock(return_value=None)

        text = self._explorer(caller)._call_mcp_tool(
            _server(), "get_board", {"id": "1"}, "what is on my board",
        )

        assert text == ""

    def test_the_follow_up_fetch_retries_with_its_own_tool(self):
        """It was retried with the *listing* tool's name, not the detail tool's."""
        caller = MCPToolCaller(None, None)
        caller.call_tool = MagicMock(side_effect=[_UNAUTHORIZED, _text("the body")])
        caller.refresh_access_token = MagicMock(return_value="fresh")

        with patch(
            "smartloop.mcp.call_plans.follow_up_calls",
            return_value=[("get_item", {"id": "7"})],
        ):
            text = self._explorer(caller)._fetch_record_details(
                _server(), "list_items", "payload",
            )

        assert "the body" in text
        assert caller.call_tool.call_args_list[1].kwargs["tool_name"] == "get_item"
