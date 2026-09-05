"""Tests for the /v1/auth API router."""

from unittest.mock import MagicMock, patch

import pytest

from smartloop.api.routers.auth import UserInfo
from smartloop.auth import credential_store

_RESOLVE_USER = "smartloop.api.routers.auth._resolve_user"
_PLATFORM_SERVICE = "smartloop.api.routers.auth.get_platform_service"


@pytest.fixture(autouse=True)
def _no_network():
    """Keep the auth router off the real platform API.

    ``_refresh_platform_cache`` resolves the service through the module, not
    through FastAPI's dependency graph, so the ``platform_service`` override in
    ``conftest`` does not reach it — every credential-setting test here was
    quietly calling out to api.smartloop.ai and relying on the handler's
    try/except to swallow the result.

    That stopped being survivable once a 401 started redeeming the refresh
    token: a test storing a fake token got a real 401, refreshed against the
    real endpoint, got a second 401, and had both credentials cleared out from
    under its own assertions.
    """
    with patch(_PLATFORM_SERVICE, return_value=MagicMock()):
        yield


@pytest.fixture(autouse=True)
def _no_platform_call():
    """Prevent /auth/status from hitting the real platform API."""
    patcher = patch(_RESOLVE_USER, return_value=None)
    patcher.start()
    yield patcher
    patcher.stop()


def _unauthorized() -> PermissionError:
    """A platform 401 as the client actually raises it, status attached."""
    exc = PermissionError("Authentication failed")
    exc.status_code = 401
    return exc


class TestAuthStatus:
    def test_no_credentials(self, client):
        resp = client.get("/v1/auth/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is False
        assert data["method"] is None
        assert data["has_developer_token"] is False
        assert data["has_access_token"] is False
        assert data["user"] is None

    def test_developer_token_only(self, client):
        credential_store.set_token("sk-dev-123")
        resp = client.get("/v1/auth/status")
        data = resp.json()
        assert data["authenticated"] is True
        assert data["method"] == "developer_token"
        assert data["has_developer_token"] is True
        assert data["has_access_token"] is False

    def test_access_token_only(self, client):
        credential_store.set_access_token("jwt-abc")
        resp = client.get("/v1/auth/status")
        data = resp.json()
        assert data["authenticated"] is True
        assert data["method"] == "access_token"
        assert data["has_developer_token"] is False
        assert data["has_access_token"] is True

    def test_both_tokens_prefers_access(self, client):
        credential_store.set_token("sk-dev-123")
        credential_store.set_access_token("jwt-abc")
        resp = client.get("/v1/auth/status")
        data = resp.json()
        assert data["method"] == "access_token"
        assert data["has_developer_token"] is True
        assert data["has_access_token"] is True

    def test_status_includes_user_info(self, client):
        credential_store.set_access_token("jwt-abc")
        user = UserInfo(email="chris@example.com", name="Chris")
        with patch(_RESOLVE_USER, return_value=user):
            resp = client.get("/v1/auth/status")
        data = resp.json()
        assert data["user"] == {"email": "chris@example.com", "name": "Chris"}

    def test_status_user_null_on_platform_failure(self, client):
        credential_store.set_access_token("jwt-abc")
        resp = client.get("/v1/auth/status")
        data = resp.json()
        assert data["authenticated"] is True
        assert data["user"] is None

class TestExpiredAccessToken:
    """What ``_resolve_user`` still owns after the refresh moved into the client.

    Redeeming the refresh token used to happen here, which is why this was the
    only path in the app that recovered from an expired token.  It now happens in
    :meth:`SLPPlatformClient._authed`, so a ``PermissionError`` arriving here has
    already survived a refresh attempt — see
    ``tests/unit/test_platform_client.py::TestRefreshOn401`` for that half.  What
    is left is the disposal: drop the dead access token so later requests fall
    back to the developer token, and leave a *transient* failure alone.
    """

    _SERVICE = _PLATFORM_SERVICE

    def _status(self, _no_platform_call, client, get_user):
        # Stop the autouse fixture so the real _resolve_user runs.
        #
        # Patched at the *service* boundary, not the client: the service caches
        # the user in the workspace DB with a TTL, so patching the client left
        # a warm cache to satisfy later calls — no request, no 401, nothing to
        # assert.  That cache leaking between tests is what made this file fail
        # intermittently depending on execution order.
        _no_platform_call.stop()
        try:
            service = MagicMock()
            service.get_current_user = MagicMock(**get_user)
            with patch(self._SERVICE, return_value=service):
                return client.get("/v1/auth/status").json()
        finally:
            _no_platform_call.start()

    def test_a_dead_access_token_is_dropped_for_the_developer_token(
        self, _no_platform_call, client,
    ):
        credential_store.set_access_token("expired-jwt")
        credential_store.set_token("sk-dev-valid")

        data = self._status(
            _no_platform_call, client,
            get_user={"side_effect": _unauthorized()},
        )

        assert credential_store.get_access_token() is None
        assert data["authenticated"] is True
        assert data["method"] == "developer_token"

    def test_the_refresh_token_is_not_collateral(
        self, _no_platform_call, client,
    ):
        """Only the refresh *endpoint* gets to decide the refresh token is dead.

        A 401 from ``/v1/users/me`` says nothing about it, and the refresh agent
        has already cleared it if its own call was refused.
        """
        credential_store.set_access_token("expired-jwt")
        credential_store.set_refresh_token("still-good-rt")

        self._status(
            _no_platform_call, client,
            get_user={"side_effect": _unauthorized()},
        )

        assert credential_store.get_access_token() is None
        assert credential_store.get_refresh_token() == "still-good-rt"

    def test_only_a_401_discards_the_token(self, _no_platform_call, client):
        """403 and 404 arrive here as PermissionError too — they are not the
        token's fault, and an API deploy that moves a route should not log
        everyone out of the desktop client."""
        for status in (403, 404):
            credential_store.set_access_token("good-jwt")
            exc = PermissionError(f"API error ({status})")
            exc.status_code = status

            data = self._status(
                _no_platform_call, client, get_user={"side_effect": exc},
            )

            assert credential_store.get_access_token() == "good-jwt", status
            assert data["user"] is None

    def test_an_outage_is_not_a_logout(self, _no_platform_call, client):
        """A platform blip now arrives as ConnectionError, not PermissionError.

        It used to arrive as the latter — every non-2xx took the endpoint's
        error class — and this path would then throw away a perfectly good
        access token because the platform had a bad minute.
        """
        credential_store.set_access_token("good-jwt")

        data = self._status(
            _no_platform_call, client,
            get_user={"side_effect": ConnectionError("platform unreachable")},
        )

        assert credential_store.get_access_token() == "good-jwt"
        assert data["authenticated"] is True
        assert data["user"] is None


class TestSetToken:
    def test_set_developer_token_default(self, client):
        resp = client.post("/v1/auth/token", json={"token": "sk-dev-123"})
        assert resp.status_code == 200
        assert resp.json()["has_developer_token"] is True
        assert credential_store.get_token() == "sk-dev-123"

    def test_set_access_token(self, client):
        resp = client.post("/v1/auth/token", json={
            "token": "jwt-abc",
            "type": "access_token",
        })
        assert resp.status_code == 200
        assert resp.json()["has_access_token"] is True
        assert credential_store.get_access_token() == "jwt-abc"

    def test_missing_token_returns_422(self, client):
        resp = client.post("/v1/auth/token", json={})
        assert resp.status_code == 422


class TestClearToken:
    def test_clear_all_default(self, client):
        credential_store.set_token("sk-dev-123")
        credential_store.set_access_token("jwt-abc")
        credential_store.set_refresh_token("rt-xyz")

        resp = client.delete("/v1/auth/token")
        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is False
        assert credential_store.get_token() is None
        assert credential_store.get_access_token() is None
        assert credential_store.get_refresh_token() is None

    def test_clear_developer_only(self, client):
        credential_store.set_token("sk-dev-123")
        credential_store.set_access_token("jwt-abc")

        resp = client.delete("/v1/auth/token?type=developer_token")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_developer_token"] is False
        assert data["has_access_token"] is True

    def test_clear_access_only(self, client):
        credential_store.set_token("sk-dev-123")
        credential_store.set_access_token("jwt-abc")
        credential_store.set_refresh_token("rt-xyz")

        resp = client.delete("/v1/auth/token?type=access_token")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_developer_token"] is True
        assert data["has_access_token"] is False
        assert credential_store.get_refresh_token() is None


class TestOAuthCallback:
    def test_callback_with_access_token_only(self, client):
        resp = client.post("/v1/auth/callback", json={
            "access_token": "jwt-from-webapp",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_access_token"] is True
        assert credential_store.get_access_token() == "jwt-from-webapp"
        assert credential_store.get_refresh_token() is None

    def test_callback_with_refresh_token(self, client):
        resp = client.post("/v1/auth/callback", json={
            "access_token": "jwt-from-webapp",
            "refresh_token": "rt-from-webapp",
        })
        assert resp.status_code == 200
        assert credential_store.get_access_token() == "jwt-from-webapp"
        assert credential_store.get_refresh_token() == "rt-from-webapp"

    def test_callback_missing_access_token_returns_422(self, client):
        resp = client.post("/v1/auth/callback", json={})
        assert resp.status_code == 422

    def test_callback_overwrites_existing(self, client):
        credential_store.set_access_token("old-jwt")
        resp = client.post("/v1/auth/callback", json={
            "access_token": "new-jwt",
        })
        assert resp.status_code == 200
        assert credential_store.get_access_token() == "new-jwt"

    def test_the_safari_fallback_stores_both_and_redirects(self, client):
        """Safari blocks the HTTPS→HTTP fetch, so the webapp navigates instead.

        Untested until now, which is uncomfortable for the one login path a
        whole browser depends on.
        """
        resp = client.get(
            "/v1/auth/callback",
            params={"access_token": "jwt-via-get", "refresh_token": "rt-via-get"},
            follow_redirects=False,
        )

        assert resp.status_code == 307
        assert "mode=desktop" in resp.headers["location"]
        assert credential_store.get_access_token() == "jwt-via-get"
        assert credential_store.get_refresh_token() == "rt-via-get"

    def test_the_safari_fallback_needs_an_access_token(self, client):
        resp = client.get("/v1/auth/callback", follow_redirects=False)
        assert resp.status_code == 422


class TestSigningInClearsTheSearchGate:
    """Whether hosted search will serve is cached for a minute, per account.

    Signing in with that cache holding the logged-out "no" left web search
    skipping itself for the first minute of the session — the orchestrator asks
    the gate before it runs the query, so the whole step disappeared.
    ``reset_status_cache`` existed for exactly this and nothing called it.
    """

    def _gate_says_no(self):
        from smartloop.search import providers

        providers.reset_status_cache()
        with patch.object(providers, "_platform_client") as pc:
            pc.return_value.web_search_status = MagicMock(
                side_effect=PermissionError("not signed in"))
            credential_store.set_token("sk-dev")  # so the gate gets past has_credentials
            assert providers.hosted_search_available() is False
        credential_store.clear_token()

    def test_a_login_reopens_the_gate(self, client):
        from smartloop.search import providers

        self._gate_says_no()

        client.post("/v1/auth/callback", json={"access_token": "jwt-fresh"})

        assert providers._status_cache is None, (
            "the stale 'search is off' survived the login"
        )

    def test_a_logout_reopens_it_too(self, client):
        from smartloop.search import providers

        credential_store.set_access_token("jwt")
        self._gate_says_no()

        client.delete("/v1/auth/token")

        assert providers._status_cache is None
