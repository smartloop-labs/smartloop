"""RFC 8707 resource indicators in the MCP OAuth flow.

MCP auth (spec rev 2025-06-18) requires the client to name the MCP server it wants
a token for, in both the authorization request and every token request. Servers that
enforce it — monday.com rejects the authorize call outright with "invalid request:
resource parameter required" — are unreachable without this, and a refresh that
drops the indicator is rejected even though the original grant succeeded.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smartloop.api.model.mcp import MCPRegisterRequest
from smartloop.mcp.oauth import MCPOAuthService, ProtectedResourceMetadata, OAuthMetadata


class _FakeResponse:
    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class _RecordingClient:
    """Records POST bodies so tests can assert on what was actually sent."""

    def __init__(self, json_data=None):
        self.posts = []
        self._json = json_data or {"access_token": "at", "token_type": "Bearer"}

    def post(self, url, data=None, headers=None, **kwargs):
        self.posts.append((url, data))
        return _FakeResponse(self._json)


class TestCanonicalResource:
    def test_path_is_significant(self):
        """/mcp and /sse are distinct resources on the same host."""
        assert (
            MCPOAuthService.canonical_resource("https://mcp.monday.com/mcp")
            == "https://mcp.monday.com/mcp"
        )
        assert (
            MCPOAuthService.canonical_resource("https://mcp.monday.com/sse")
            == "https://mcp.monday.com/sse"
        )

    def test_scheme_and_host_are_lowercased(self):
        assert (
            MCPOAuthService.canonical_resource("HTTPS://MCP.Monday.COM/mcp")
            == "https://mcp.monday.com/mcp"
        )

    def test_fragment_is_dropped(self):
        """RFC 8707 §2 forbids a fragment in a resource indicator."""
        assert (
            MCPOAuthService.canonical_resource("https://mcp.example.com/mcp#frag")
            == "https://mcp.example.com/mcp"
        )

    def test_bare_host_has_no_trailing_slash(self):
        assert (
            MCPOAuthService.canonical_resource("https://mcp.example.com/")
            == "https://mcp.example.com"
        )

    def test_query_is_preserved(self):
        assert (
            MCPOAuthService.canonical_resource("https://mcp.example.com/mcp?v=2")
            == "https://mcp.example.com/mcp?v=2"
        )


class TestAuthorizationRequest:
    def test_resource_is_sent_when_given(self):
        url = MCPOAuthService().build_authorization_url(
            authorization_endpoint="https://auth.monday.com/oauth2/authorize",
            client_id="cid",
            redirect_uri="http://localhost:38540/v1/mcp/oauth/callback",
            state="st",
            code_challenge="cc",
            resource="https://mcp.monday.com/mcp",
        )
        assert "resource=https%3A%2F%2Fmcp.monday.com%2Fmcp" in url

    def test_resource_is_omitted_when_absent(self):
        url = MCPOAuthService().build_authorization_url(
            authorization_endpoint="https://auth.example.com/authorize",
            client_id="cid",
            redirect_uri="http://localhost/cb",
            state="st",
            code_challenge="cc",
        )
        assert "resource=" not in url


class TestTokenRequests:
    def test_code_exchange_sends_resource(self):
        client = _RecordingClient()
        MCPOAuthService(http_client=client).exchange_code_for_tokens(
            token_endpoint="https://auth.monday.com/oauth_ms/oauth/token",
            code="code",
            redirect_uri="http://localhost/cb",
            client_id="cid",
            code_verifier="cv",
            resource="https://mcp.monday.com/mcp",
        )
        _, body = client.posts[0]
        assert body["resource"] == "https://mcp.monday.com/mcp"

    def test_refresh_sends_resource(self):
        client = _RecordingClient()
        MCPOAuthService(http_client=client).refresh_access_token(
            token_endpoint="https://auth.monday.com/oauth_ms/oauth/token",
            refresh_token="rt",
            client_id="cid",
            resource="https://mcp.monday.com/mcp",
        )
        _, body = client.posts[0]
        assert body["resource"] == "https://mcp.monday.com/mcp"

    def test_refresh_omits_resource_when_absent(self):
        """Pre-existing servers registered without an indicator keep working."""
        client = _RecordingClient()
        MCPOAuthService(http_client=client).refresh_access_token(
            token_endpoint="https://auth.example.com/token",
            refresh_token="rt",
            client_id="cid",
        )
        _, body = client.posts[0]
        assert "resource" not in body


class TestStoredServerRefresh:
    """``MCPToolCaller.refresh_access_token`` runs off the persisted config."""

    def _server(self, resource):
        from smartloop.workspace.workspace import (
            MCPAuthType,
            MCPOAuthCredentials,
            MCPServer,
            MCPServerType,
            RemoteMCPServerConfig,
        )

        return MCPServer(
            id="s1",
            name="monday",
            server_type=MCPServerType.REMOTE,
            remote_config=RemoteMCPServerConfig(
                server_url="https://mcp.monday.com/mcp",
                auth_type=MCPAuthType.OAUTH,
                token_endpoint="https://auth.monday.com/oauth_ms/oauth/token",
                resource=resource,
            ),
            oauth_credentials=MCPOAuthCredentials(client_id="cid", refresh_token="rt"),
        )

    def _refresh(self, server):
        from smartloop.agents.mcp.mcp_tool_caller import MCPToolCaller

        client = _RecordingClient()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=client)
        ctx.__exit__ = MagicMock(return_value=False)
        with patch("httpx.Client", return_value=ctx):
            token = MCPToolCaller.refresh_access_token(server)
        return token, client.posts

    def test_persisted_resource_is_replayed(self):
        token, posts = self._refresh(self._server("https://mcp.monday.com/mcp"))
        assert token == "at"
        assert posts[0][1]["resource"] == "https://mcp.monday.com/mcp"

    def test_server_without_resource_refreshes_unchanged(self):
        token, posts = self._refresh(self._server(None))
        assert token == "at"
        assert "resource" not in posts[0][1]


class TestRegisterFlowPlumbing:
    """The register endpoint must derive an indicator and carry it to the callback."""

    def _register(self, prm_resource, server_url="https://mcp.monday.com/mcp"):
        from smartloop.api.routers import mcp_servers as router_mod

        recorded = {}

        class _FakeOAuth:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def discover_protected_resource_metadata(self, url, metadata_url=None):
                return ProtectedResourceMetadata(
                    resource=prm_resource,
                    authorization_servers=["https://auth.monday.com/mcp"],
                )

            def discover_oauth_metadata(self, url):
                return OAuthMetadata(
                    issuer="https://auth.monday.com/mcp",
                    authorization_endpoint="https://auth.monday.com/oauth2/authorize",
                    token_endpoint="https://auth.monday.com/oauth_ms/oauth/token",
                    registration_endpoint="https://auth.monday.com/oauth_ms/oauth/register",
                )

            def register_client(self, endpoint, redirect_uri):
                return MagicMock(client_id="cid", client_secret=None)

            def build_authorization_url(self, **kwargs):
                recorded.update(kwargs)
                return "https://auth.monday.com/oauth2/authorize?stub"

            canonical_resource = staticmethod(MCPOAuthService.canonical_resource)
            generate_pkce = staticmethod(MCPOAuthService.generate_pkce)
            generate_state = staticmethod(MCPOAuthService.generate_state)

        # Direct (unauthenticated) connection has to fail for the OAuth path to run.
        failing_client = MagicMock()
        failing_client.__enter__ = MagicMock(side_effect=RuntimeError("401"))
        failing_client.__exit__ = MagicMock(return_value=False)

        router_mod._pending_oauth_sessions.clear()
        with patch("smartloop.mcp.MCPOAuthService", _FakeOAuth), patch(
            "smartloop.mcp.MCPClient", return_value=failing_client
        ), patch.object(
            router_mod, "_try_register_smartloop_mcp_server",
            new=AsyncMock(return_value=None),
        ):
            resp = asyncio.run(
                router_mod._register_remote_mcp_server(
                    request=MCPRegisterRequest(server_url=server_url, server_type="remote"),
                    model_service=MagicMock(),
                    http_request=None,
                    project_id="p1",
                )
            )
        session = next(iter(router_mod._pending_oauth_sessions.values()))
        return resp, recorded, session

    def test_prm_resource_wins(self):
        resp, recorded, session = self._register("https://mcp.monday.com/mcp")
        assert recorded["resource"] == "https://mcp.monday.com/mcp"
        assert session["resource"] == "https://mcp.monday.com/mcp"
        assert resp.resource == "https://mcp.monday.com/mcp"

    def test_falls_back_to_canonical_server_url(self):
        """A server publishing no PRM ``resource`` still gets an indicator."""
        _, recorded, session = self._register(None, server_url="https://mcp.example.com/mcp")
        assert recorded["resource"] == "https://mcp.example.com/mcp"
        assert session["resource"] == "https://mcp.example.com/mcp"
