"""Tests for the /v1/auth API router."""

from unittest.mock import MagicMock, patch

import pytest

from smartloop.api.routers.auth import UserInfo
from smartloop.auth import credential_store

_RESOLVE_USER = "smartloop.api.routers.auth._resolve_user"


@pytest.fixture(autouse=True)
def _no_platform_call():
    """Prevent /auth/status from hitting the real platform API."""
    patcher = patch(_RESOLVE_USER, return_value=None)
    patcher.start()
    yield patcher
    patcher.stop()


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
    """A 401 on /v1/users/me means the access token is dead — but a refresh
    token can mint a new one, so it is redeemed before anything is thrown away.
    Deleting first logged the user out until the background loop's next tick,
    up to a full refresh interval later, with a working refresh token on disk.
    """

    _SERVICE = "smartloop.api.routers.auth.get_platform_service"
    _REFRESH = "smartloop.platform.client.SLPPlatformClient.refresh_access_token"

    def _status(self, _no_platform_call, client, get_user, refresh):
        # Stop the autouse fixture so the real _resolve_user runs.
        #
        # Patched at the *service* boundary, not the client: the service caches
        # the user in the workspace DB with a TTL, so patching the client left
        # a warm cache to satisfy later calls — no request, no 401, nothing to
        # assert.  That cache leaking between tests is what made this file fail
        # intermittently depending on execution order.
        #
        # And both calls are patched so the test never reaches the network: it
        # otherwise POSTed a fake refresh token to api.smartloop.ai and its
        # result depended on what came back.
        _no_platform_call.stop()
        try:
            service = MagicMock()
            service.get_current_user = MagicMock(**get_user)
            with patch(self._SERVICE, return_value=service), \
                    patch(self._REFRESH, **refresh):
                return client.get("/v1/auth/status").json()
        finally:
            _no_platform_call.start()

    def test_a_valid_refresh_token_restores_the_session(
        self, _no_platform_call, client,
    ):
        credential_store.set_access_token("expired-jwt")
        credential_store.set_refresh_token("old-rt")

        data = self._status(
            _no_platform_call, client,
            # 401 first, then the retry on the refreshed token succeeds.
            get_user={"side_effect": [
                PermissionError("Authentication failed"),
                {"email": "chris@example.com", "name": "Chris"},
            ]},
            refresh={"return_value": {
                "access_token": "fresh-jwt", "refresh_token": "new-rt",
            }},
        )

        assert credential_store.get_access_token() == "fresh-jwt"
        assert credential_store.get_refresh_token() == "new-rt"
        assert data["authenticated"] is True
        assert data["method"] == "access_token"
        assert data["user"] == {"email": "chris@example.com", "name": "Chris"}

    def test_a_rotated_refresh_token_is_optional(self, _no_platform_call, client):
        """The endpoint may return only a new access token."""
        credential_store.set_access_token("expired-jwt")
        credential_store.set_refresh_token("old-rt")

        self._status(
            _no_platform_call, client,
            get_user={"side_effect": [
                PermissionError("Authentication failed"),
                {"email": "chris@example.com"},
            ]},
            refresh={"return_value": {"access_token": "fresh-jwt"}},
        )

        assert credential_store.get_access_token() == "fresh-jwt"
        assert credential_store.get_refresh_token() == "old-rt"

    def test_a_dead_refresh_token_clears_both_and_falls_back(
        self, _no_platform_call, client,
    ):
        """Retrying a rejected refresh token cannot help — re-login required."""
        credential_store.set_access_token("expired-jwt")
        credential_store.set_refresh_token("old-rt")
        credential_store.set_token("sk-dev-valid")

        data = self._status(
            _no_platform_call, client,
            get_user={"side_effect": PermissionError("Authentication failed")},
            refresh={"side_effect": PermissionError("Authentication failed")},
        )

        assert credential_store.get_access_token() is None
        assert credential_store.get_refresh_token() is None
        assert data["authenticated"] is True
        assert data["method"] == "developer_token"

    def test_no_refresh_token_still_purges_the_stale_one(
        self, _no_platform_call, client,
    ):
        credential_store.set_access_token("expired-jwt")
        credential_store.set_token("sk-dev-valid")

        data = self._status(
            _no_platform_call, client,
            get_user={"side_effect": PermissionError("Authentication failed")},
            refresh={"side_effect": AssertionError("must not be called")},
        )

        assert credential_store.get_access_token() is None
        assert data["method"] == "developer_token"

    def test_a_refresh_outage_leaves_the_refresh_token_alone(
        self, _no_platform_call, client,
    ):
        """A network failure is not proof the token is bad — keep it to retry."""
        credential_store.set_access_token("expired-jwt")
        credential_store.set_refresh_token("old-rt")

        self._status(
            _no_platform_call, client,
            get_user={"side_effect": PermissionError("Authentication failed")},
            refresh={"side_effect": ConnectionError("platform unreachable")},
        )

        assert credential_store.get_access_token() is None
        assert credential_store.get_refresh_token() == "old-rt"


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
