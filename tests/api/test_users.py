"""Tests for the /v1/users API router."""


class TestGetMe:
    def test_returns_full_profile(self, client, platform_service):
        platform_service.get_current_user.return_value = {
            "email": "chris@example.com",
            "name": "Chris",
            "is_premium": True,
            "is_paid": True,
            "number_of_tokens_processed": 5000,
            "default_temperature": 0.5,
            "default_memory": True,
        }
        resp = client.get("/v1/users/me")

        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "chris@example.com"
        assert data["name"] == "Chris"
        assert data["is_premium"] is True
        assert data["is_paid"] is True
        assert data["number_of_tokens_processed"] == 5000
        assert data["default_temperature"] == 0.5
        assert "selected_model" not in data

    def test_optional_fields_default(self, client, platform_service):
        platform_service.get_current_user.return_value = {"email": "chris@example.com"}
        resp = client.get("/v1/users/me")

        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "chris@example.com"
        assert data["name"] is None
        assert data["is_premium"] is False
        assert data["is_paid"] is False
        assert data["number_of_tokens_processed"] == 0
        assert data["default_temperature"] == 0.3
        assert data["default_memory"] is True

    def test_401_when_not_authenticated(self, client, platform_service):
        platform_service.get_current_user.side_effect = PermissionError("Not authenticated")
        resp = client.get("/v1/users/me")

        assert resp.status_code == 401
        assert resp.json()["detail"] == "Not authenticated"

    def test_502_when_platform_unavailable(self, client, platform_service):
        platform_service.get_current_user.side_effect = ConnectionError("Connection refused")
        resp = client.get("/v1/users/me")

        assert resp.status_code == 502
        assert resp.json()["detail"] == "Platform API unavailable"


class TestUpdateMe:
    def test_updates_default_temperature(self, client, platform_service):
        platform_service.update_current_user.return_value = {
            "email": "chris@example.com",
            "name": "Chris",
            "default_temperature": 0.9,
        }
        resp = client.patch("/v1/users/me", json={"default_temperature": 0.9})

        assert resp.status_code == 200
        assert resp.json()["default_temperature"] == 0.9

    def test_422_when_no_fields(self, client):
        resp = client.patch("/v1/users/me", json={})
        assert resp.status_code == 422

    def test_401_when_not_authenticated(self, client, platform_service):
        platform_service.update_current_user.side_effect = PermissionError("Not authenticated")
        resp = client.patch("/v1/users/me", json={"default_temperature": 0.4})

        assert resp.status_code == 401
