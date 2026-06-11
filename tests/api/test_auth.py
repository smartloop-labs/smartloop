"""Tests for the /v1/auth API router."""

from unittest.mock import patch

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

    def test_status_clears_expired_access_token_on_401(self, _no_platform_call, client):
        """When /v1/users/me returns 401, the stale access token is purged."""
        credential_store.set_access_token("expired-jwt")
        credential_store.set_refresh_token("old-rt")
        credential_store.set_token("sk-dev-valid")

        # Stop the autouse fixture so real _resolve_user runs, then
        # make the underlying platform call raise a 401-style error.
        _no_platform_call.stop()
        with patch(
            "smartloop.platform.client.SLPPlatformClient.get_current_user",
            side_effect=PermissionError("Authentication failed"),
        ):
            resp = client.get("/v1/auth/status")
        _no_platform_call.start()

        data = resp.json()
        assert credential_store.get_access_token() is None
        assert credential_store.get_refresh_token() == "old-rt"  # kept for future refresh
        # Falls back to developer token
        assert data["authenticated"] is True
        assert data["method"] == "developer_token"


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
