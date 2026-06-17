"""Tests for the /v1/models API router."""


class TestListModels:
    def test_returns_model_list(self, client, platform_service):
        # slp-orchestra-mini is the bootstrap (framework core) model; /v1/models
        # lists the additional models a user can add, so the bootstrap model is
        # filtered out of the result.
        platform_service.list_models.return_value = [
            {"name": "slp-orchestra-mini", "model_id": "google/gemma-3-1b-it"},
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it"},
        ]
        resp = client.get("/v1/models")

        assert resp.status_code == 200
        data = resp.json()
        names = [m["name"] for m in data]
        assert "slp-orchestra-mini" not in names  # bootstrap model excluded
        assert names == ["gemma3-4b"]

    def test_returns_empty_list(self, client, platform_service):
        platform_service.list_models.return_value = []
        resp = client.get("/v1/models")

        assert resp.status_code == 200
        assert resp.json() == []

    def test_502_when_platform_unavailable(self, client, platform_service):
        platform_service.list_models.side_effect = ConnectionError("Connection refused")
        resp = client.get("/v1/models")

        assert resp.status_code == 502
        assert resp.json()["detail"] == "Platform API unavailable"
