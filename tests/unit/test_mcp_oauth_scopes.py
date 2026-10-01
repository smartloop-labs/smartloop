"""Which scopes the authorization request asks for.

The flow used to join every scope the *authorization server* advertised and ask
for all of them, consulting the protected resource's list only when the server
published none. That is backwards twice over. RFC 9728 §2 defines the PRM list
as the scopes "used in authorization requests to request access to this
protected resource", while RFC 8414's is a catalog of everything the server
knows across every resource it fronts — and §7.2 asks clients for "as limited a
scope as possible" either way.

Clera is the case that exposed it: the resource wants three scopes, Clerk
advertises seven, and a DCR-registered client is not permitted ``openid``. So
the authorize call was refused outright with ``invalid_scope`` on a connection
that needed nothing unusual.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smartloop.api.model.mcp import MCPRegisterRequest
from smartloop.mcp.oauth import (
    MCPOAuthError,
    MCPOAuthService,
    OAuthMetadata,
    ProtectedResourceMetadata,
)

# What the two servers actually publish, fetched from them while diagnosing this.
CLERA_PRM_SCOPES = ["profile", "email", "offline_access"]
CLERK_AS_SCOPES = [
    "openid", "profile", "email", "public_metadata",
    "private_metadata", "offline_access", "user:org:read",
]


def _register(prm, as_scopes, server_url="https://mcp.getclera.com/"):
    """Drive the register endpoint down its OAuth path and capture the authorize call."""
    from smartloop.api.routers import mcp_servers as router_mod

    recorded = {}

    class _FakeOAuth:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def discover_protected_resource_metadata(self, url, metadata_url=None):
            if prm is None:
                raise MCPOAuthError("no PRM published")
            return prm

        def discover_oauth_metadata(self, url):
            return OAuthMetadata(
                issuer="https://clerk.getclera.com",
                authorization_endpoint="https://clerk.getclera.com/oauth/authorize",
                token_endpoint="https://clerk.getclera.com/oauth/token",
                registration_endpoint="https://clerk.getclera.com/oauth/register",
                scopes_supported=as_scopes,
            )

        def register_client(self, endpoint, redirect_uri):
            return MagicMock(client_id="cid", client_secret=None)

        def build_authorization_url(self, **kwargs):
            recorded.update(kwargs)
            return "https://clerk.getclera.com/oauth/authorize?stub"

        canonical_resource = staticmethod(MCPOAuthService.canonical_resource)
        generate_pkce = staticmethod(MCPOAuthService.generate_pkce)
        generate_state = staticmethod(MCPOAuthService.generate_state)

    # The unauthenticated probe has to fail for the OAuth path to run at all.
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
        asyncio.run(
            router_mod._register_remote_mcp_server(
                request=MCPRegisterRequest(server_url=server_url, server_type="remote"),
                model_service=MagicMock(),
                project_id="p1",
            )
        )
    session = next(iter(router_mod._pending_oauth_sessions.values()))
    return recorded, session


class TestScopeSelection:
    def test_the_resource_list_is_what_is_asked_for(self):
        """The Clera case, end to end: three scopes requested, not Clerk's seven."""
        recorded, _ = _register(
            ProtectedResourceMetadata(
                resource="https://mcp.getclera.com/",
                authorization_servers=["https://clerk.getclera.com"],
                scopes_supported=CLERA_PRM_SCOPES,
            ),
            as_scopes=CLERK_AS_SCOPES,
        )
        assert recorded["scope"] == "profile email offline_access"

    def test_a_scope_only_the_server_advertises_is_never_requested(self):
        """``openid`` is in Clerk's catalog and forbidden to DCR clients."""
        recorded, _ = _register(
            ProtectedResourceMetadata(
                resource="https://mcp.getclera.com/",
                authorization_servers=["https://clerk.getclera.com"],
                scopes_supported=CLERA_PRM_SCOPES,
            ),
            as_scopes=CLERK_AS_SCOPES,
        )
        for catalog_only in ("openid", "private_metadata", "public_metadata", "user:org:read"):
            assert catalog_only not in recorded["scope"]

    def test_a_resource_declaring_nothing_asks_for_nothing(self):
        """No parameter at all, so the server applies its own default."""
        recorded, _ = _register(
            ProtectedResourceMetadata(
                resource="https://mcp.example.com/mcp",
                authorization_servers=["https://as.example.com"],
                scopes_supported=[],
            ),
            as_scopes=CLERK_AS_SCOPES,
        )
        assert recorded["scope"] is None

    def test_no_resource_metadata_at_all_asks_for_nothing(self):
        recorded, _ = _register(None, as_scopes=CLERK_AS_SCOPES)
        assert recorded["scope"] is None

    def test_the_server_catalog_is_not_a_fallback(self):
        """Even with no PRM scopes and a rich catalog, we do not help ourselves."""
        recorded, _ = _register(
            ProtectedResourceMetadata(
                resource="https://mcp.example.com/mcp",
                authorization_servers=["https://as.example.com"],
            ),
            as_scopes=["read", "write", "admin"],
        )
        assert recorded["scope"] is None


class TestSessionCarriesTheRequest:
    """The callback needs both to reissue an authorization the server refused."""

    def test_scope_and_challenge_survive_the_redirect(self):
        recorded, session = _register(
            ProtectedResourceMetadata(
                resource="https://mcp.getclera.com/",
                authorization_servers=["https://clerk.getclera.com"],
                scopes_supported=CLERA_PRM_SCOPES,
            ),
            as_scopes=CLERK_AS_SCOPES,
        )
        assert session["scope"] == "profile email offline_access"
        assert session["code_challenge"] == recorded["code_challenge"]

    def test_an_empty_scope_is_recorded_as_none(self):
        _, session = _register(None, as_scopes=CLERK_AS_SCOPES)
        assert session["scope"] is None
