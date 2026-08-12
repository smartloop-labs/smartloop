"""Tests for MCP OAuth metadata discovery (RFC 9728 / RFC 8414).

These exercise the discovery URL handling that unblocks path-scoped OAuth servers
such as Trello (https://mcp.trello.com/v1). A fake httpx client is injected via the
``MCPOAuthService(http_client=...)`` constructor so no real network calls are made.
"""

import httpx
import pytest

from smartloop.mcp.client import MCPClient
from smartloop.mcp.oauth import (
    MCPOAuthService,
    MetadataDiscoveryError,
    ResourceMetadataDiscoveryError,
)


class _FakeResponse:
    """Stand-in for an httpx.Response returned from ``client.get``."""

    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = {} if json_data is None else json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            req = httpx.Request("GET", "https://example.test")
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=req,
                response=httpx.Response(self.status_code, request=req),
            )

    def json(self):
        return self._json


class _FakeClient:
    """Maps an exact URL -> _FakeResponse / Exception. Unmapped URLs default to 404.

    Records every requested URL in ``get_calls`` so tests can assert ordering.
    """

    def __init__(self, routes):
        self._routes = routes
        self.get_calls = []

    def get(self, url, **kwargs):
        self.get_calls.append(url)
        handler = self._routes.get(url, _FakeResponse(404))
        if isinstance(handler, Exception):
            raise handler
        return handler


def _service(routes):
    return MCPOAuthService(http_client=_FakeClient(routes))


class TestPRMDiscovery:
    def test_advertised_metadata_url_is_tried_first(self):
        advertised = "https://mcp.trello.com/v1/custom-prm"
        routes = {
            advertised: _FakeResponse(
                json_data={
                    "resource": "https://mcp.trello.com/v1",
                    "authorization_servers": ["https://auth.atlassian.com/abc123"],
                }
            )
        }
        svc = _service(routes)
        prm = svc.discover_protected_resource_metadata(
            "https://mcp.trello.com/v1", metadata_url=advertised
        )
        assert prm.authorization_servers == ["https://auth.atlassian.com/abc123"]
        # The advertised URL must be attempted before any guessed well-known URL.
        assert svc._http_client.get_calls[0] == advertised

    def test_base_503_falls_through_to_path_scoped_200(self):
        """Trello's base PRM URL persistently 503s; the path-scoped one returns 200."""
        base = "https://mcp.trello.com/.well-known/oauth-protected-resource"
        scoped = "https://mcp.trello.com/.well-known/oauth-protected-resource/v1"
        routes = {
            base: _FakeResponse(503),
            scoped: _FakeResponse(
                json_data={
                    "authorization_servers": ["https://auth.atlassian.com/abc123"],
                    "scopes_supported": ["read", "write"],
                }
            ),
        }
        svc = _service(routes)
        prm = svc.discover_protected_resource_metadata("https://mcp.trello.com/v1")
        assert prm.authorization_servers == ["https://auth.atlassian.com/abc123"]
        assert base in svc._http_client.get_calls  # the 503 did not abort discovery

    def test_metadata_url_failure_falls_back_to_guessed(self):
        advertised = "https://mcp.trello.com/v1/custom-prm"
        base = "https://mcp.trello.com/.well-known/oauth-protected-resource"
        routes = {
            advertised: _FakeResponse(503),
            base: _FakeResponse(
                json_data={"authorization_servers": ["https://as.example.com"]}
            ),
        }
        svc = _service(routes)
        prm = svc.discover_protected_resource_metadata(
            "https://mcp.trello.com/v1", metadata_url=advertised
        )
        assert prm.authorization_servers == ["https://as.example.com"]

    def test_all_endpoints_fail_raises(self):
        svc = _service({})  # everything 404s
        with pytest.raises(ResourceMetadataDiscoveryError):
            svc.discover_protected_resource_metadata("https://mcp.trello.com/v1")


class TestOAuthMetadataDiscovery:
    def test_path_append_form_resolves_for_atlassian(self):
        issuer = "https://auth.atlassian.com/abc123"
        working = f"{issuer}/.well-known/oauth-authorization-server"
        routes = {
            # Both well-known-inserted forms fail for Atlassian.
            "https://auth.atlassian.com/.well-known/oauth-authorization-server": _FakeResponse(404),
            "https://auth.atlassian.com/.well-known/oauth-authorization-server/abc123": _FakeResponse(404),
            working: _FakeResponse(
                json_data={
                    "issuer": issuer,
                    "authorization_endpoint": f"{issuer}/authorize",
                    "token_endpoint": f"{issuer}/token",
                    "registration_endpoint": f"{issuer}/register",
                }
            ),
        }
        svc = _service(routes)
        meta = svc.discover_oauth_metadata(issuer)
        assert meta.registration_endpoint == f"{issuer}/register"
        assert working in svc._http_client.get_calls

    def test_bare_url_does_not_shadow_path_scoped_dcr_metadata(self):
        """Atlassian's real shape: the bare URL 200s without a registration_endpoint
        while the path-scoped URL carries DCR. We must return the DCR-capable doc."""
        issuer = "https://auth.atlassian.com/abc123"
        host_insert_with_path = (
            "https://auth.atlassian.com/.well-known/oauth-authorization-server/abc123"
        )
        bare = "https://auth.atlassian.com/.well-known/oauth-authorization-server"
        routes = {
            bare: _FakeResponse(
                json_data={
                    "issuer": "https://auth.atlassian.com",
                    "authorization_endpoint": "https://auth.atlassian.com/authorize",
                    "token_endpoint": "https://auth.atlassian.com/token",
                    # no registration_endpoint
                }
            ),
            host_insert_with_path: _FakeResponse(
                json_data={
                    "issuer": issuer,
                    "authorization_endpoint": f"{issuer}/authorize",
                    "token_endpoint": f"{issuer}/token",
                    "registration_endpoint": f"{issuer}/dcr/register",
                }
            ),
        }
        svc = _service(routes)
        meta = svc.discover_oauth_metadata(issuer)
        assert meta.registration_endpoint == f"{issuer}/dcr/register"

    def test_no_path_issuer_resolves_via_host_insert(self):
        """Regression guard: issuers without a path still resolve at the base URL."""
        base = "https://mcp.linear.app/.well-known/oauth-authorization-server"
        routes = {
            base: _FakeResponse(
                json_data={
                    "issuer": "https://mcp.linear.app",
                    "authorization_endpoint": "https://mcp.linear.app/authorize",
                    "token_endpoint": "https://mcp.linear.app/token",
                    "registration_endpoint": "https://mcp.linear.app/register",
                }
            )
        }
        svc = _service(routes)
        meta = svc.discover_oauth_metadata("https://mcp.linear.app")
        assert meta.registration_endpoint == "https://mcp.linear.app/register"

    def test_no_path_issuer_only_emits_host_insert_url(self):
        svc = MCPOAuthService()
        urls = svc._get_metadata_urls_to_try("https://mcp.linear.app")
        assert urls == ["https://mcp.linear.app/.well-known/oauth-authorization-server"]

    def test_path_issuer_emits_all_candidate_forms(self):
        svc = MCPOAuthService()
        urls = svc._get_metadata_urls_to_try("https://auth.atlassian.com/abc123")
        # Path-bearing forms first; the bare (path-stripped) URL is a last resort.
        assert urls == [
            "https://auth.atlassian.com/.well-known/oauth-authorization-server/abc123",
            "https://auth.atlassian.com/abc123/.well-known/oauth-authorization-server",
            "https://auth.atlassian.com/abc123/.well-known/openid-configuration",
            "https://auth.atlassian.com/.well-known/oauth-authorization-server",
        ]

    def test_all_endpoints_fail_raises(self):
        svc = _service({})
        with pytest.raises(MetadataDiscoveryError):
            svc.discover_oauth_metadata("https://auth.atlassian.com/abc123")


class TestResourceMetadataHeaderParsing:
    def test_extracts_resource_metadata_url(self):
        header = 'Bearer resource_metadata="https://mcp.trello.com/v1/prm", resource="https://mcp.trello.com/v1"'
        assert (
            MCPClient._parse_resource_metadata_from_www_authenticate(header)
            == "https://mcp.trello.com/v1/prm"
        )

    def test_does_not_confuse_with_plain_resource_field(self):
        header = 'Bearer resource="https://mcp.trello.com/v1"'
        assert MCPClient._parse_resource_metadata_from_www_authenticate(header) is None
        # And the plain-resource parser must not grab the metadata field.
        header2 = 'Bearer resource_metadata="https://mcp.trello.com/v1/prm"'
        assert MCPClient._parse_resource_from_www_authenticate(header2) is None

    def test_unquoted_form(self):
        # Mirrors the existing resource= parser: bare value, trailing comma stripped.
        assert (
            MCPClient._parse_resource_metadata_from_www_authenticate(
                "Bearer resource_metadata=https://x/prm"
            )
            == "https://x/prm"
        )
        assert (
            MCPClient._parse_resource_metadata_from_www_authenticate(
                "Bearer resource_metadata=https://x/prm,"
            )
            == "https://x/prm"
        )

    def test_empty_header_returns_none(self):
        assert MCPClient._parse_resource_metadata_from_www_authenticate("") is None
