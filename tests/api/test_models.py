"""Tests for the /v1/models API router."""

from unittest.mock import patch

import pytest

_LIST_MODELS = "smartloop.platform.client.SLPPlatformClient.list_models"


class TestListModels:
    def test_returns_model_list(self, client):
        # slp-orchestra-mini is the bootstrap (framework core) model; /v1/models
        # lists the additional models a user can add, so the bootstrap model is
        # filtered out of the result.
        mock_data = [
            {"name": "slp-orchestra-mini", "model_id": "google/gemma-3-1b-it"},
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it"},
        ]
        with patch(_LIST_MODELS, return_value=mock_data):
            resp = client.get("/v1/models")

        assert resp.status_code == 200
        data = resp.json()
        names = [m["name"] for m in data]
        assert "slp-orchestra-mini" not in names  # bootstrap model excluded
        assert names == ["gemma3-4b"]

    def test_returns_empty_list(self, client):
        with patch(_LIST_MODELS, return_value=[]):
            resp = client.get("/v1/models")

        assert resp.status_code == 200
        assert resp.json() == []

    def test_502_when_platform_unavailable(self, client):
        with patch(_LIST_MODELS, side_effect=ConnectionError("Connection refused")):
            resp = client.get("/v1/models")

        assert resp.status_code == 502
        assert resp.json()["detail"] == "Platform API unavailable"
