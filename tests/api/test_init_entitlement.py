"""/v1/init must answer a refused download with a status code, not a stream.

Once StreamingResponse begins, the request has committed to 200 and a refusal
can only be described inside the body — where a client that keys off the status
will never see it. So entitlement is resolved before the stream opens, and only
failures that happen mid-transfer arrive as SSE error frames (tagged with the
status they would have been).
"""

import json
from unittest.mock import MagicMock

import pytest

from smartloop.api.main import app
from smartloop.api.routers import init as init_router
from smartloop.api.services import get_model_service
from smartloop.workspace.workspace import Workspace, WorkspaceModel


@pytest.fixture
def model_service(tmp_path):
    workspace = Workspace(
        id="ws-test",
        workspace_dir=str(tmp_path),
        models_dir=str(tmp_path / "models"),
    )
    workspace.models.append(WorkspaceModel(
        model_name="gemma3-12b",
        model_id="google/gemma-3-12b-it",
        downloaded=False,
    ))
    svc = MagicMock()
    svc._ensure_workspace.return_value = workspace
    app.dependency_overrides[get_model_service] = lambda: svc
    yield svc
    app.dependency_overrides.pop(get_model_service, None)


def _stub_downloader(monkeypatch, resolve_side_effect):
    """Replace the downloader the router builds with one that fails on resolve."""
    made = MagicMock()
    made.resolve_artifacts.side_effect = resolve_side_effect
    monkeypatch.setattr(init_router, "SLPArtifactDownloader", lambda *a, **k: made)
    return made


def _tagged(cls, message, status):
    exc = cls(message)
    exc.status_code = status
    return exc


class TestEntitlementIsSettledBeforeTheStream:
    def test_refusal_is_a_403_not_a_200_stream(self, client, model_service, monkeypatch):
        _stub_downloader(monkeypatch, _tagged(
            PermissionError, "Model 'gemma3-12b' is not available on your current plan.", 403))

        resp = client.post("/v1/init", json={"model_name": "gemma3-12b"})

        assert resp.status_code == 403
        assert "current plan" in resp.json()["detail"]

    def test_rate_limit_surfaces_as_429(self, client, model_service, monkeypatch):
        # Not 403: retrying fixes this one, and calling it a plan problem would
        # send the user to a billing page instead of back in a minute.
        _stub_downloader(monkeypatch, _tagged(
            ConnectionError, "Too many download requests. Please wait a moment and try again.", 429))

        resp = client.post("/v1/init", json={"model_name": "gemma3-12b"})

        assert resp.status_code == 429

    def test_platform_unreachable_is_502(self, client, model_service, monkeypatch):
        _stub_downloader(monkeypatch, ConnectionError("Connection error: refused"))

        resp = client.post("/v1/init", json={"model_name": "gemma3-12b"})

        assert resp.status_code == 502

    def test_unknown_model_is_still_400(self, client, model_service):
        # The only remaining 400 on this route. It is not fixed by signing in,
        # so a client must be able to tell it apart from an entitlement refusal.
        resp = client.post("/v1/init", json={"model_name": "no-such-model"})

        assert resp.status_code == 400
        assert "Unknown model" in resp.json()["detail"]


class TestMidStreamFailureCarriesItsStatus:
    def test_error_frame_is_tagged_with_a_code(self, client, model_service, monkeypatch):
        """A failure after the stream opened still has to be classifiable."""
        made = _stub_downloader(monkeypatch, None)
        made.resolve_artifacts.side_effect = None
        made.resolve_artifacts.return_value = {"model.gguf": "https://signed/url"}

        async def boom(*args, **kwargs):
            raise _tagged(PermissionError, "Download rejected (401 Unauthorized).", 401)
            yield  # pragma: no cover - makes this an async generator

        monkeypatch.setattr(init_router, "_download_and_stream", boom)

        resp = client.post("/v1/init", json={"model_name": "gemma3-12b"})

        assert resp.status_code == 200  # already committed; the body carries the failure
        frames = [json.loads(line[len("data: "):])
                  for line in resp.text.splitlines() if line.startswith("data: ")]
        errors = [f for f in frames if f.get("status") == "error"]
        assert errors, f"no error frame in stream: {resp.text!r}"
        assert errors[0]["code"] == 401
        assert "401" in errors[0]["message"]
