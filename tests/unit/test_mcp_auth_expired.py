"""A token has expired when the server said 401, not when the word appears.

Every MCP call site decided a token had expired by asking ``"401" in
result.error`` — a substring test against a message that embeds the server's own
response body.  A tool returning a document that mentions 401 spent a refresh
token on nothing and retried a call that had not failed for auth reasons at all.

``auth_expired`` is set by the transport from the HTTP status instead.
"""

from unittest.mock import patch

from smartloop.mcp.client import MCPToolCallResult


class TestOnlyARealUnauthorizedTriggersARefresh:
    """``auth_expired`` is set by the transport from the HTTP status."""

    def test_a_body_mentioning_401_is_not_an_auth_failure(self):
        result = MCPToolCallResult(
            success=False,
            error="Tool call failed: the upstream returned error code 401 in its payload",
        )
        assert result.auth_expired is False

    def test_the_transport_flags_a_real_401(self):
        from smartloop.mcp.client import MCPClient, MCPConnectionError

        client = MCPClient()
        with patch.object(
            MCPClient, "_ensure_session",
            side_effect=MCPConnectionError("Authentication failed (401).", status_code=401),
        ):
            result = client.call_tool("https://example/mcp", "whoami")

        assert result.success is False
        assert result.auth_expired is True

    def test_a_403_is_not_an_expired_token(self):
        """Insufficient permissions is about the grant, not the token's age."""
        from smartloop.mcp.client import MCPClient, MCPConnectionError

        client = MCPClient()
        with patch.object(
            MCPClient, "_ensure_session",
            side_effect=MCPConnectionError("Insufficient permissions", status_code=403),
        ):
            result = client.call_tool("https://example/mcp", "whoami")

        assert result.success is False
        assert result.auth_expired is False
