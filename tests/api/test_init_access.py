"""Tests for the download access gate on /v1/init.

Free models download with no token at all, so ``/v1/init`` must not ask for
credentials.  The only gate left is the plan, and it answers with a real HTTP
403 — before the SSE stream opens — so the client can render an upgrade prompt
off the status code instead of parsing an error event.
"""

import json
from unittest.mock import MagicMock

import pytest

from smartloop.api.main import app
from smartloop.api.routers import init as init_router
from smartloop.api.services import get_model_service
from smartloop.platform import ModelAccessError
from smartloop.workspace.workspace import Workspace

FREE_MODEL = "gemma3-4b"
PAID_MODEL = "gemma4-e4b"


@pytest.fixture
def model_service(tmp_path):
    """A model service whose workspace has no weights on disk.

    ``models_dir`` points into an empty tmp dir, so ``_gguf_exists`` is False
    and every request in this module takes the "needs download" path.
    """
    workspace = Workspace(
        id="ws-test",
        workspace_dir=str(tmp_path),
        models_dir=str(tmp_path / "models"),
    )
    svc = MagicMock()
    svc._ensure_workspace.return_value = workspace
    app.dependency_overrides[get_model_service] = lambda: svc
    yield svc
    app.dependency_overrides.pop(get_model_service, None)


def _catalog(name, access_level, is_accessible):
    return [{
        "name": name,
        "model_id": f"google/{name}",
        "access_level": access_level,
        "is_accessible": is_accessible,
    }]


def _init_status(client, model_name):
    """POST /v1/init and return the status code without draining the stream.

    The gate runs before the ``StreamingResponse`` is built, so the status is
    settled by the time the headers arrive; closing here keeps the test from
    running a real download for the cases that are allowed through.
    """
    with client.stream("POST", "/v1/init", json={"model_name": model_name}) as resp:
        return resp.status_code


class TestInitPlanGate:
    def test_403_for_a_model_off_the_plan(self, client, platform_service, model_service):
        platform_service.list_models.return_value = _catalog(PAID_MODEL, "pro", False)

        with client.stream("POST", "/v1/init", json={"model_name": PAID_MODEL}) as resp:
            assert resp.status_code == 403
            assert "not available on your current plan" in resp.read().decode()

    def test_no_gate_for_a_free_model_without_credentials(
        self, client, platform_service, model_service
    ):
        # The old behaviour was a 400 ("Authentication required") for any
        # download without a token. A free model must now get through.
        platform_service.list_models.return_value = _catalog(FREE_MODEL, "free", False)

        assert _init_status(client, FREE_MODEL) == 200

    def test_accessible_paid_model_is_not_gated(
        self, client, platform_service, model_service
    ):
        platform_service.list_models.return_value = _catalog(PAID_MODEL, "pro", True)

        assert _init_status(client, PAID_MODEL) == 200

    def test_unknown_to_the_catalog_is_not_gated(
        self, client, platform_service, model_service
    ):
        # Fail open: a missing catalog entry must not block a download the
        # platform would have served.
        platform_service.list_models.return_value = []

        assert _init_status(client, PAID_MODEL) == 200

    def test_unclassified_model_is_not_gated(
        self, client, platform_service, model_service
    ):
        # No access_level from the cloud — treat as free, as /v1/models does.
        platform_service.list_models.return_value = _catalog(PAID_MODEL, None, False)

        assert _init_status(client, PAID_MODEL) == 200

    def test_unreachable_platform_is_not_gated(
        self, client, platform_service, model_service
    ):
        platform_service.list_models.side_effect = ConnectionError("offline")

        assert _init_status(client, PAID_MODEL) == 200


class TestLateDenialOnTheStream:
    """The pre-stream gate reads a cached catalog, so the platform's own 403 can
    still land mid-stream. The status line is 200 by then, so the code has to
    ride on the event for the client to handle it the same way."""

    def test_error_event_carries_the_403(
        self, client, platform_service, model_service, monkeypatch
    ):
        # Catalog says it's free; the platform disagrees when asked for URLs.
        platform_service.list_models.return_value = _catalog(FREE_MODEL, "free", True)

        def _denying_downloader(workspace, logger):
            downloader = MagicMock()
            downloader.download.side_effect = ModelAccessError(
                f"Model '{FREE_MODEL}' is not available on your current plan."
            )
            return downloader

        monkeypatch.setattr(init_router, "SLPArtifactDownloader", _denying_downloader)

        resp = client.post("/v1/init", json={"model_name": FREE_MODEL})

        assert resp.status_code == 200
        events = [
            json.loads(line[len("data: "):])
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        error = next(e for e in events if e["status"] == "error")
        assert error["code"] == 403
        assert "not available on your current plan" in error["message"]
