"""A connection that can no longer authenticate stops reporting itself healthy.

``status``/``last_error`` were written only by the provisioning pass, so a
runtime auth failure changed nothing: a connection whose grant the provider had
retired kept reporting ``ready`` with its full tool list forever, and
``build_available_servers`` never consulted ``status`` anyway, so the planner
went on offering tools that could only 401 — into a failure path that swallows
the error.  To the user that reads as "that app has nothing in it".

Only a refusal that re-minting cannot recover from counts.  A network blip or a
5xx must leave the connection alone; failing it over one bad round-trip would
cost someone a working connection.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from smartloop.agents.mcp.mcp_tool_caller import MCPToolCaller
from smartloop.mcp.oauth import TokenExchangeError
from smartloop.workspace.workspace import (
    MCPAuthType, MCPOAuthCredentials, MCPServer, MCPServerStatus,
    MCPServerType, MCPToolDefinition, RemoteMCPServerConfig,
)


def _server(*, refresh_token="rt", token=None):
    return MCPServer(
        id="s1", name="monday", server_type=MCPServerType.REMOTE,
        remote_config=RemoteMCPServerConfig(
            server_url="https://mcp.monday.com/mcp",
            auth_type=MCPAuthType.OAUTH if refresh_token else MCPAuthType.TOKEN,
            token_endpoint="https://auth.monday.com/token",
            token=token,
        ),
        oauth_credentials=(
            MCPOAuthCredentials(client_id="cid", refresh_token=refresh_token)
            if refresh_token else None
        ),
        tools=[MCPToolDefinition(name="get_board", description="", input_schema={})],
    )


def _refusal(code, status=400):
    resp = httpx.Response(
        status, json={"error": code} if code else {},
        request=httpx.Request("POST", "https://auth.monday.com/token"),
    )
    return httpx.HTTPStatusError("refused", request=resp.request, response=resp)


def _posting(exc):
    http = MagicMock()
    http.post = MagicMock(side_effect=exc)
    return patch("httpx.Client", return_value=http)


class TestOnlyADeadGrantFailsTheConnection:
    @pytest.mark.parametrize("code", ["invalid_grant", "invalid_client"])
    def test_a_retired_grant_fails_it(self, code):
        server = _server()
        saves = []
        caller = MCPToolCaller(None, None, persist=lambda: saves.append(1))

        with _posting(_refusal(code)):
            assert caller.refresh_access_token(server) is None

        assert server.status == MCPServerStatus.FAILED
        assert "Reconnect" in server.last_error
        assert saves == [1]

    def test_a_server_error_leaves_it_alone(self):
        """A 5xx is the provider having a bad day, not the grant expiring."""
        server = _server()
        caller = MCPToolCaller(None, None)

        with _posting(_refusal("server_error", status=503)):
            assert caller.refresh_access_token(server) is None

        assert server.status == MCPServerStatus.READY
        assert server.last_error is None

    def test_an_unreachable_endpoint_leaves_it_alone(self):
        server = _server()
        caller = MCPToolCaller(None, None)

        with _posting(httpx.ConnectError("no route")):
            assert caller.refresh_access_token(server) is None

        assert server.status == MCPServerStatus.READY

    def test_a_refusal_with_no_error_code_leaves_it_alone(self):
        server = _server()
        caller = MCPToolCaller(None, None)

        with _posting(_refusal(None)):
            assert caller.refresh_access_token(server) is None

        assert server.status == MCPServerStatus.READY

    def test_the_same_failure_twice_is_written_once(self):
        server = _server()
        saves = []
        caller = MCPToolCaller(None, None, persist=lambda: saves.append(1))

        with _posting(_refusal("invalid_grant")):
            caller.refresh_access_token(server)
            caller.refresh_access_token(server)

        assert saves == [1]


class TestARejectedPastedCredentialFailsIt:
    """There is no refresh to fall back on, so the 401 is the verdict."""

    def test_a_pasted_token_that_stopped_working_is_reported(self):
        server = _server(refresh_token=None, token="ghp_xxx")
        caller = MCPToolCaller(None, None)

        assert caller.refresh_access_token(server) is None

        assert server.status == MCPServerStatus.FAILED
        assert "Enter a new token" in server.last_error

    def test_an_oauth_connection_with_no_refresh_token_is_left_alone(self):
        """Nothing to re-enter: it needs re-authorizing, not a new paste."""
        server = _server(refresh_token=None)
        caller = MCPToolCaller(None, None)

        assert caller.refresh_access_token(server) is None

        assert server.status == MCPServerStatus.READY


class TestThePlannerStopsOfferingIt:
    def test_a_failed_connection_is_not_offered(self):
        from smartloop.agents.orchestration.step_executor import StepExecutor

        server = _server()
        server.status = MCPServerStatus.FAILED
        server.last_error = "This connection's authorization has expired."

        available = StepExecutor.build_available_servers(MagicMock(), [server])

        assert available == []

    def test_a_healthy_connection_still_is(self):
        from smartloop.agents.orchestration.step_executor import StepExecutor

        available = StepExecutor.build_available_servers(MagicMock(), [_server()])

        assert [s["name"] for s in available] == ["monday"]
