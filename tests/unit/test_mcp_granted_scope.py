"""What the grant came back with, kept rather than discarded.

RFC 6749 §5.1 lets the token response name the scope actually granted, which
need not be the scope asked for — a server may narrow the set and say so. The
value was parsed into ``TokenResponse`` and then dropped by every caller, so a
down-scoped grant was invisible until a tool call failed, and the page shown
when a connection produced no tools could only guess that scopes were why.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from smartloop.agents.mcp.mcp_tool_caller import MCPToolCaller
from smartloop.mcp.client import MCPConnectResult

from tests.unit.test_mcp_credential_persistence import _reply, _server


class TestTheCredentialCarriesIt:
    def test_the_model_records_a_granted_scope(self):
        from smartloop.workspace.workspace import MCPOAuthCredentials

        creds = MCPOAuthCredentials(client_id="cid", scope="profile email")
        assert creds.scope == "profile email"

    def test_it_defaults_to_nothing_for_credentials_written_before_it_existed(self):
        from smartloop.workspace.workspace import MCPOAuthCredentials

        assert MCPOAuthCredentials(client_id="cid").scope is None


class TestARefreshKeepsItCurrent:
    def test_a_narrowed_refresh_is_recorded(self):
        """§6 lets a server narrow on refresh; the stored scope must follow."""
        server = _server(scope="profile email offline_access")
        with _reply({
            "access_token": "new", "token_type": "Bearer",
            "expires_in": 3600, "scope": "profile offline_access",
        }):
            MCPToolCaller(None, None).refresh_access_token(server)
        assert server.oauth_credentials.scope == "profile offline_access"

    def test_a_refresh_that_names_no_scope_leaves_the_grant_alone(self):
        """Omitting it means "unchanged", not "nothing"."""
        server = _server(scope="profile email offline_access")
        with _reply({"access_token": "new", "token_type": "Bearer", "expires_in": 3600}):
            MCPToolCaller(None, None).refresh_access_token(server)
        assert server.oauth_credentials.scope == "profile email offline_access"


def _complete(connect_result, granted_scope):
    """Run the redirect callback to completion and capture what was persisted."""
    from smartloop.api.routers import mcp_servers as router_mod

    class _FakeOAuth:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def exchange_code_for_tokens(self, **kwargs):
            return MagicMock(
                access_token="at", token_type="Bearer",
                refresh_token="rt", expires_in=3600, scope=granted_scope,
            )

    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.connect_and_discover_tools = MagicMock(return_value=connect_result)

    model_service = MagicMock()
    model_service.add_remote_mcp_server = MagicMock(
        return_value={"id": "s1", "name": "Clera", "tools": []}
    )

    router_mod._pending_oauth_sessions.clear()
    router_mod._completed_oauth_responses.clear()
    router_mod._pending_oauth_sessions["st"] = {
        "created_at": time.time(),
        "project_id": "p1",
        "server_url": "https://mcp.getclera.com/",
        "state": "st",
        "code_verifier": "cv",
        "code_challenge": "chal",
        "scope": "profile email offline_access",
        "client_id": "cid",
        "client_secret": None,
        "token_endpoint": "https://clerk.getclera.com/oauth/token",
        "authorization_endpoint": "https://clerk.getclera.com/oauth/authorize",
        "resource": "https://mcp.getclera.com",
        "callback_url": "http://localhost:38540/v1/mcp/oauth/callback",
        "name": "Clera",
        "return_url": None,
        "description": None,
        "tags": None,
        "catalog_slug": None,
    }
    with patch("smartloop.mcp.MCPOAuthService", _FakeOAuth), patch(
        "smartloop.mcp.MCPClient", return_value=client
    ):
        resp = asyncio.run(
            router_mod.mcp_oauth_callback(
                code="c", state="st", model_service=model_service
            )
        )
    return resp, model_service.add_remote_mcp_server


class TestTheCallbackPersistsIt:
    def test_the_granted_scope_reaches_the_record(self):
        tool = MagicMock(name="t", description="d", input_schema={})
        _, saved = _complete(
            MCPConnectResult(success=True, tools=[tool]), "profile email"
        )
        assert saved.call_args.kwargs["scope"] == "profile email"

    def test_a_grant_naming_no_scope_persists_nothing(self):
        tool = MagicMock(name="t", description="d", input_schema={})
        _, saved = _complete(MCPConnectResult(success=True, tools=[tool]), None)
        assert saved.call_args.kwargs["scope"] is None


class TestTheEmptyToolPageStopsGuessing:
    def test_it_names_what_was_actually_granted(self):
        resp, _ = _complete(
            MCPConnectResult(success=True, tools=[]), "profile email"
        )
        body = resp.body.decode()
        assert resp.status_code == 400
        assert "profile email" in body

    def test_it_says_so_when_the_grant_named_no_scopes(self):
        resp, _ = _complete(MCPConnectResult(success=True, tools=[]), None)
        body = resp.body.decode()
        assert "no scopes" in body

    def test_a_transport_failure_still_reports_the_transport(self):
        """A dead endpoint is not a scope problem and must not read as one."""
        resp, _ = _complete(
            MCPConnectResult(success=False, error="HTTP error: 404"), "profile"
        )
        body = resp.body.decode()
        assert "HTTP error: 404" in body
        assert "granted" not in body
