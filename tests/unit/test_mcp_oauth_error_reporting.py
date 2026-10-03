"""What the user is told when a provider refuses authorization.

FastAPI binds only the query parameters a route declares.  The callbacks
declared ``code``, ``state`` and ``error`` and nothing else, so the rest of the
RFC 6749 §4.1.2.1 error response — ``error_description`` and ``error_uri`` —
arrived on the redirect and was dropped on the floor.  ``error`` alone comes
from a fixed enum, so the page could only ever say ``invalid_scope``: the
sentence naming *which* scope went in the bin, and nothing was logged either,
because the branch returns before any logging.
"""
import asyncio
import logging
from unittest.mock import MagicMock

import httpx
import pytest

from smartloop.mcp.oauth import TokenExchangeError, _token_error


def _redirect(**params):
    from smartloop.api.routers import mcp_servers as router_mod

    resp = asyncio.run(
        router_mod.mcp_oauth_callback(model_service=MagicMock(), **params)
    )
    return resp.status_code, resp.body.decode()


class TestAuthorizationErrorPage:
    def test_description_reaches_the_page(self):
        """The Clera failure: the enum says nothing, the description says 'openid'."""
        status, body = _redirect(
            error="invalid_scope",
            error_description=(
                "The requested scope is invalid, unknown, or malformed. The "
                "OAuth 2.0 Client is not allowed to request scope 'openid'."
            ),
        )
        assert status == 400
        assert "invalid_scope" in body
        assert "not allowed to request scope &#x27;openid&#x27;" in body

    def test_error_alone_still_renders(self):
        status, body = _redirect(error="access_denied")
        assert status == 400
        assert "access_denied" in body

    def test_http_error_uri_becomes_a_link(self):
        _, body = _redirect(
            error="invalid_scope",
            error_uri="https://clerk.com/docs/oauth/scoped-access",
        )
        assert '<a href="https://clerk.com/docs/oauth/scoped-access"' in body

    def test_non_http_error_uri_is_never_linked(self):
        """``error_uri`` arrives from a server we discovered, not one we configured."""
        _, body = _redirect(error="invalid_scope", error_uri="javascript:alert(1)")
        assert "<a href" not in body
        assert "javascript:alert(1)" in body

    def test_description_is_escaped(self):
        _, body = _redirect(
            error="invalid_scope", error_description="<script>alert(1)</script>"
        )
        assert "<script>" not in body
        assert "&lt;script&gt;" in body

    def test_the_refusal_is_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="smartloop.api.routers.mcp_servers"):
            _redirect(
                error="invalid_scope",
                error_description="not allowed to request scope 'openid'",
                error_uri="https://example.com/why",
            )
        logged = caplog.text
        assert "invalid_scope" in logged
        assert "openid" in logged
        assert "https://example.com/why" in logged


class TestTokenEndpointErrorBody:
    """RFC 6749 §5.2 carries the same three fields on the token endpoint."""

    def _error(self, payload, status=400):
        request = httpx.Request("POST", "https://as.example.com/token")
        response = httpx.Response(status, json=payload, request=request)
        exc = httpx.HTTPStatusError("boom", request=request, response=response)
        return _token_error(exc, "Token exchange failed")

    def test_description_and_uri_are_both_reported(self):
        err = self._error({
            "error": "invalid_scope",
            "error_description": "scope 'openid' is not permitted",
            "error_uri": "https://example.com/why",
        })
        assert isinstance(err, TokenExchangeError)
        assert err.error_code == "invalid_scope"
        assert "scope 'openid' is not permitted" in str(err)
        assert "https://example.com/why" in str(err)

    def test_the_error_code_still_drives_dead_grant_detection(self):
        assert self._error({"error": "invalid_grant"}).grant_is_dead
        assert not self._error({"error": "invalid_scope"}).grant_is_dead

    def test_a_body_that_is_not_json_leaves_the_status_line_as_the_story(self):
        request = httpx.Request("POST", "https://as.example.com/token")
        response = httpx.Response(500, text="<html>nope</html>", request=request)
        exc = httpx.HTTPStatusError("boom", request=request, response=response)
        err = _token_error(exc, "Token refresh failed")
        assert err.error_code is None
        assert "Token refresh failed: HTTP 500" in str(err)

    def test_a_bare_error_code_is_used_as_its_own_description(self):
        err = self._error({"error": "invalid_client"})
        assert "invalid_client" in str(err)
