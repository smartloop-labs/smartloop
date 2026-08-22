"""Tests for the /v1/models API router."""

from unittest.mock import MagicMock

import pytest

from smartloop.api.main import app
from smartloop.api.services import get_model_service
from smartloop.workspace.workspace import Workspace, WorkspaceModel


def _ws_model(model_name: str, *, downloaded: bool, enabled: bool = True):
    return WorkspaceModel(
        model_name=model_name,
        model_id=f"test/{model_name}",
        downloaded=downloaded,
        enabled=enabled,
    )


@pytest.fixture
def workspace_models():
    """Override the model service so /v1/models sees a known workspace.

    Yields the workspace's ``models`` list for the test to append to; the
    override returns the same workspace every call, so appends made after the
    client is created still take effect.  This is a real ``Workspace`` rather
    than a stub because the router resolves ``enabled`` through
    ``effective_model_enabled`` (project override -> workspace default).
    """
    workspace = Workspace(
        id="ws-test",
        workspace_dir="/tmp/ws-test",
        models_dir="/tmp/ws-test/models",
    )
    svc = MagicMock()
    svc._ensure_workspace.return_value = workspace
    app.dependency_overrides[get_model_service] = lambda: svc
    yield workspace.models
    app.dependency_overrides.pop(get_model_service, None)


class TestListModels:
    def test_returns_model_list(self, client, platform_service):
        # sl-mini is the bootstrap (framework core) model; /v1/models lists the
        # additional models a user can add, so the bootstrap model is filtered
        # out of the result.
        platform_service.list_models.return_value = [
            {"name": "sl-mini", "model_id": "smartloop-ai/slp-orchestra-mini"},
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it"},
        ]
        resp = client.get("/v1/models")

        assert resp.status_code == 200
        data = resp.json()
        names = [m["name"] for m in data]
        assert "sl-mini" not in names  # bootstrap model excluded
        assert names == ["gemma3-4b"]

    def test_bootstrap_model_excluded_under_its_legacy_name(self, client, platform_service):
        # The cloud catalog still publishes the pre-rename name; it must be
        # canonicalized so the bootstrap filter still catches it.
        platform_service.list_models.return_value = [
            {"name": "slp-orchestra-mini", "model_id": "smartloop-ai/slp-orchestra-mini"},
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it"},
        ]
        resp = client.get("/v1/models")

        assert resp.status_code == 200
        names = [m["name"] for m in resp.json()]
        assert names == ["gemma3-4b"]

    def test_excludes_paid_models_outside_plan(self, client, platform_service):
        platform_service.list_models.return_value = [
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it",
             "access_level": "free", "is_accessible": True},
            {"name": "gemma3-12b", "model_id": "google/gemma-3-12b-it",
             "access_level": "pro", "is_accessible": False},
            {"name": "gemma3-27b", "model_id": "google/gemma-3-27b-it",
             "access_level": "premium", "is_accessible": False},
        ]
        resp = client.get("/v1/models")

        assert resp.status_code == 200
        assert [m["name"] for m in resp.json()] == ["gemma3-4b"]

    def test_keeps_free_models_when_not_logged_in(self, client, platform_service):
        # Anonymous callers get is_accessible=False on everything, so the free
        # tier must not be gated on that flag.
        platform_service.list_models.return_value = [
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it",
             "access_level": "free", "is_accessible": False},
            {"name": "gemma3-12b", "model_id": "google/gemma-3-12b-it",
             "access_level": "pro", "is_accessible": False},
        ]
        resp = client.get("/v1/models")

        assert resp.status_code == 200
        assert [m["name"] for m in resp.json()] == ["gemma3-4b"]

    def test_keeps_unclassified_models(self, client, platform_service):
        # No access_level from the cloud — treat as free, not as gated.
        platform_service.list_models.return_value = [
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it",
             "is_accessible": False},
        ]
        resp = client.get("/v1/models")

        assert resp.status_code == 200
        assert [m["name"] for m in resp.json()] == ["gemma3-4b"]

    def test_keeps_inaccessible_model_already_downloaded(
        self, client, platform_service, workspace_models
    ):
        # Weights are on disk and still usable, so a plan downgrade must not
        # hide the model.
        workspace_models.append(_ws_model("gemma3-12b", downloaded=True))
        platform_service.list_models.return_value = [
            {"name": "gemma3-12b", "model_id": "google/gemma-3-12b-it",
             "access_level": "pro", "is_accessible": False},
        ]
        resp = client.get("/v1/models")

        assert resp.status_code == 200
        assert [m["name"] for m in resp.json()] == ["gemma3-12b"]

    def test_excludes_inaccessible_model_registered_but_not_downloaded(
        self, client, platform_service, workspace_models
    ):
        workspace_models.append(_ws_model("gemma3-12b", downloaded=False))
        platform_service.list_models.return_value = [
            {"name": "gemma3-12b", "model_id": "google/gemma-3-12b-it",
             "access_level": "pro", "is_accessible": False},
        ]
        resp = client.get("/v1/models")

        assert resp.status_code == 200
        assert resp.json() == []

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
