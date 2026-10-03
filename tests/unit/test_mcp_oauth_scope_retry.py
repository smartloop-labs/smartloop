"""Reissuing an authorization the server refused on scope.

The PRM list is what RFC 9728 §2 says to ask for, but it describes the resource
and not the client: a client registered through DCR can be permitted less than
the resource advertises. When that happens the error names at most one
offending scope, so there is nothing to negotiate against — the one useful
retry asks for none and lets the server apply the default for the client it
registered.
"""
import asyncio
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import pytest


def _session(**overrides):
    base = {
        "created_at": 0.0,
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
    base.update(overrides)
    return base


def _refuse(error="invalid_scope", session=None, state="st"):
    """Deliver an authorization error to the callback and return the response."""
    from smartloop.api.routers import mcp_servers as router_mod

    router_mod._pending_oauth_sessions.clear()
    router_mod._completed_oauth_responses.clear()
    if session is not None:
        router_mod._pending_oauth_sessions["st"] = session
    resp = asyncio.run(
        router_mod.mcp_oauth_callback(
            state=state, error=error, model_service=MagicMock()
        )
    )
    return resp, router_mod._pending_oauth_sessions.get("st")


class TestRetry:
    def test_a_refused_scope_is_reissued_without_one(self):
        resp, _ = _refuse(session=_session())
        assert resp.status_code == 302
        target = resp.headers["location"]
        assert target.startswith("https://clerk.getclera.com/oauth/authorize")
        assert "scope" not in parse_qs(urlparse(target).query)

    def test_the_reissue_keeps_everything_else_the_grant_needs(self):
        resp, _ = _refuse(session=_session())
        params = parse_qs(urlparse(resp.headers["location"]).query)
        assert params["state"] == ["st"]
        assert params["client_id"] == ["cid"]
        assert params["code_challenge"] == ["chal"]
        assert params["code_challenge_method"] == ["S256"]
        # RFC 8707 — dropping the indicator would make the retry fail differently.
        assert params["resource"] == ["https://mcp.getclera.com"]

    def test_the_session_records_what_the_retry_asked_for(self):
        _, session = _refuse(session=_session())
        assert session["scope_retry"] is True
        assert session["scope"] is None

    def test_a_second_refusal_gives_up(self):
        """Otherwise a server refusing the empty request bounces the browser forever."""
        from smartloop.api.routers import mcp_servers as router_mod

        first, _ = _refuse(session=_session())
        assert first.status_code == 302

        # Same session, now carrying the retry mark the first refusal left.
        second = asyncio.run(
            router_mod.mcp_oauth_callback(
                state="st", error="invalid_scope", model_service=MagicMock()
            )
        )
        assert second.status_code == 400
        assert "invalid_scope" in second.body.decode()


class TestWhenNotToRetry:
    def test_another_error_is_reported_not_retried(self):
        resp, _ = _refuse(error="access_denied", session=_session())
        assert resp.status_code == 400
        assert "access_denied" in resp.body.decode()

    def test_nothing_to_drop_means_nothing_to_retry(self):
        """The request already named no scopes; asking again changes nothing."""
        resp, _ = _refuse(session=_session(scope=None))
        assert resp.status_code == 400

    def test_a_session_without_a_challenge_is_not_rebuilt(self):
        """A client driving the flow itself owns its own PKCE pair."""
        resp, _ = _refuse(session=_session(code_challenge=None))
        assert resp.status_code == 400

    def test_an_unknown_state_is_reported(self):
        resp, _ = _refuse(session=None)
        assert resp.status_code == 400

    def test_an_error_without_state_is_reported(self):
        resp, _ = _refuse(session=_session(), state=None)
        assert resp.status_code == 400
