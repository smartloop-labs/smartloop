"""Tests for the /v1/storage API router."""

from unittest.mock import MagicMock, patch

import pytest

from smartloop.api.main import app
from smartloop.api.services import get_model_service

_USAGE = "smartloop.api.routers.storage.storage_service.compute_usage"
_CLEANUP = "smartloop.api.routers.storage.storage_service.run_cleanup"

SAMPLE_USAGE = {
    "home_dir": "/home/.smartloop",
    "total": 1000,
    "files": 10,
    "reclaimable": 300,
    "kept": 700,
    "categories": {
        # The group subtotals partition the category: in_use + system +
        # reclaimable + overhead == bytes.  Kept realistic so this sample does
        # not drift from what compute_usage actually returns.
        "models": {
            "bytes": 800, "files": 4, "reclaimable": 300, "kept": 500, "items": [],
            "in_use": 400, "system": 90, "overhead": 10,
        },
    },
}


@pytest.fixture
def model_service():
    """A MagicMock ModelService bound via FastAPI dependency override."""
    mock = MagicMock()
    app.dependency_overrides[get_model_service] = lambda: mock
    yield mock
    app.dependency_overrides.pop(get_model_service, None)


class TestGetStorageUsage:
    def test_returns_the_usage_report(self, client, model_service):
        with patch(_USAGE, return_value=SAMPLE_USAGE):
            resp = client.get("/v1/storage/usage")

        assert resp.status_code == 200
        assert resp.json() == SAMPLE_USAGE

    def test_passes_the_model_service_through(self, client, model_service):
        with patch(_USAGE, return_value=SAMPLE_USAGE) as usage:
            client.get("/v1/storage/usage")

        assert usage.call_args.kwargs["model_service"] is model_service

    def test_503_when_workspace_is_not_initialized(self, client, model_service):
        with patch(_USAGE, side_effect=RuntimeError("Workspace not initialized")):
            resp = client.get("/v1/storage/usage")

        assert resp.status_code == 503
        assert resp.json()["detail"] == "Workspace not initialized"

    def test_500_on_unexpected_failure(self, client, model_service):
        with patch(_USAGE, side_effect=OSError("disk gone")):
            resp = client.get("/v1/storage/usage")

        assert resp.status_code == 500


class TestCleanupStorage:
    def test_defaults_to_cleaning_nothing(self, client, model_service):
        with patch(_CLEANUP, return_value={"freed": 0}) as cleanup:
            resp = client.post("/v1/storage/cleanup", json={})

        assert resp.status_code == 200
        kwargs = cleanup.call_args.kwargs
        assert not any([
            kwargs["models"], kwargs["versions"], kwargs["logs"],
            kwargs["cache"], kwargs["orphan_projects"], kwargs["dry_run"],
        ])
        assert kwargs["model_ids"] is None

    def test_forwards_every_flag(self, client, model_service):
        body = {
            "models": True, "model_ids": ["google/gemma-3-4b-it"], "versions": True,
            "logs": True, "cache": True, "orphan_projects": True, "dry_run": True,
        }
        with patch(_CLEANUP, return_value={"freed": 0}) as cleanup:
            resp = client.post("/v1/storage/cleanup", json=body)

        assert resp.status_code == 200
        kwargs = cleanup.call_args.kwargs
        for key, value in body.items():
            assert kwargs[key] == value
        assert kwargs["model_service"] is model_service

    def test_returns_the_cleanup_report(self, client, model_service):
        report = {
            "dry_run": False, "freed": 512, "freed_by_category": {"logs": 512},
            "deleted": [{"category": "logs", "path": "/home/.smartloop/server.log.1", "bytes": 512}],
            "failed": [], "models_marked": [], "restart_required": False,
            "usage": SAMPLE_USAGE,
        }
        with patch(_CLEANUP, return_value=report):
            resp = client.post("/v1/storage/cleanup", json={"logs": True})

        assert resp.status_code == 200
        assert resp.json() == report

    def test_400_when_a_model_id_is_not_removable(self, client, model_service):
        with patch(_CLEANUP, side_effect=ValueError("Not removable: google/gemma-4-e4b-it")):
            resp = client.post(
                "/v1/storage/cleanup",
                json={"models": True, "model_ids": ["google/gemma-4-e4b-it"]},
            )

        assert resp.status_code == 400
        assert "Not removable" in resp.json()["detail"]

    def test_503_when_the_model_is_busy(self, client, model_service):
        with patch(_CLEANUP, side_effect=RuntimeError("Model is busy generating")):
            resp = client.post("/v1/storage/cleanup", json={"models": True})

        assert resp.status_code == 503

    def test_422_on_a_malformed_body(self, client, model_service):
        resp = client.post("/v1/storage/cleanup", json={"models": "yes please"})

        assert resp.status_code == 422
