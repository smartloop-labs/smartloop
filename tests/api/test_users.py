"""Tests for the /v1/users API router."""

from unittest.mock import patch

import pytest

_GET_CURRENT_USER = "smartloop.platform.client.SLPPlatformClient.get_current_user"
_UPDATE_CURRENT_USER = "smartloop.platform.client.SLPPlatformClient.update_current_user"


class TestGetMe:
    def test_returns_full_profile(self, client):
        mock_data = {
            "email": "chris@example.com",
            "name": "Chris",
            "is_premium": True,
            "is_paid": True,
            "number_of_tokens_processed": 5000,
            "selected_model": "slp-orchestra-mini",
        }
        with patch(_GET_CURRENT_USER, return_value=mock_data):
            resp = client.get("/v1/users/me")

        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "chris@example.com"
        assert data["name"] == "Chris"
        assert data["is_premium"] is True
        assert data["is_paid"] is True
        assert data["number_of_tokens_processed"] == 5000
        assert data["selected_model"] == "slp-orchestra-mini"

    def test_optional_fields_default(self, client):
        mock_data = {"email": "chris@example.com"}
        with patch(_GET_CURRENT_USER, return_value=mock_data):
            resp = client.get("/v1/users/me")

        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "chris@example.com"
        assert data["name"] is None
        assert data["is_premium"] is False
        assert data["is_paid"] is False
        assert data["number_of_tokens_processed"] == 0
        assert data["selected_model"] is None

    def test_401_when_not_authenticated(self, client):
        with patch(_GET_CURRENT_USER, side_effect=PermissionError("Not authenticated")):
            resp = client.get("/v1/users/me")

        assert resp.status_code == 401
        assert resp.json()["detail"] == "Not authenticated"

    def test_502_when_platform_unavailable(self, client):
        with patch(_GET_CURRENT_USER, side_effect=ConnectionError("Connection refused")):
            resp = client.get("/v1/users/me")

        assert resp.status_code == 502
        assert resp.json()["detail"] == "Platform API unavailable"


class TestUpdateMe:
    def test_updates_selected_model(self, client):
        mock_data = {
            "email": "chris@example.com",
            "name": "Chris",
            "selected_model": "gemma3-4b",
        }
        with patch(_UPDATE_CURRENT_USER, return_value=mock_data):
            resp = client.patch("/v1/users/me", json={"selected_model": "gemma3-4b"})

        assert resp.status_code == 200
        assert resp.json()["selected_model"] == "gemma3-4b"

    def test_422_when_no_fields(self, client):
        resp = client.patch("/v1/users/me", json={})
        assert resp.status_code == 422

    def test_401_when_not_authenticated(self, client):
        with patch(_UPDATE_CURRENT_USER, side_effect=PermissionError("Not authenticated")):
            resp = client.patch("/v1/users/me", json={"selected_model": "slp-orchestra-mini"})

        assert resp.status_code == 401
