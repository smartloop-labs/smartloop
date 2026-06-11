"""Tests for the /v1/models API router."""

from unittest.mock import patch

import pytest

_LIST_MODELS = "smartloop.platform.client.SLPPlatformClient.list_models"


class TestListModels:
    def test_returns_model_list(self, client):
        mock_data = [
            {"name": "gemma3-1b", "model_id": "google/gemma-3-1b-it"},
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it"},
        ]
        with patch(_LIST_MODELS, return_value=mock_data):
            resp = client.get("/v1/models")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["name"] == "gemma3-1b"
        assert data[0]["model_id"] == "google/gemma-3-1b-it"
        assert data[1]["name"] == "gemma3-4b"

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
