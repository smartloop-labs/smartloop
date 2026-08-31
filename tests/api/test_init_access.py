"""/v1/init must answer a refused download with a status code, not a stream.

Free models download with no token at all, so there is no credential gate left
on this route.  What remains is entitlement, and it is resolved against the
platform *before* the SSE stream opens: once StreamingResponse begins, the
request has committed to 200 and a refusal can only be described inside a body
that a client keying off the status will never read.

Only failures that happen mid-transfer arrive as SSE error frames, tagged with
the status they would have been.
"""

import json
from unittest.mock import MagicMock

import pytest

from smartloop.api.main import app
from smartloop.api.routers import init as init_router
from smartloop.api.services import get_model_service
from smartloop.platform import ModelAccessError
from smartloop.workspace.workspace import Workspace, WorkspaceModel

FREE_MODEL = "gemma3-4b"
PAID_MODEL = "gemma3-12b"


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
    workspace.models.append(WorkspaceModel(
        model_name=PAID_MODEL,
        model_id="google/gemma-3-12b-it",
        downloaded=False,
    ))
    svc = MagicMock()
    svc._ensure_workspace.return_value = workspace
    app.dependency_overrides[get_model_service] = lambda: svc
    yield svc
    app.dependency_overrides.pop(get_model_service, None)


def _stub_downloader(monkeypatch, side_effect=None, artifacts=None):
    """Replace the downloader the router builds with one under our control."""
    made = MagicMock()
    made.resolve_artifacts.side_effect = side_effect
    if side_effect is None:
        made.resolve_artifacts.return_value = artifacts or {"model.gguf": "https://signed/url"}
    monkeypatch.setattr(init_router, "SLPArtifactDownloader", lambda *a, **k: made)
    return made


def _tagged(cls, message, status):
    exc = cls(message)
    exc.status_code = status
    return exc


def _frames(resp):
    return [json.loads(line[len("data: "):])
            for line in resp.text.splitlines() if line.startswith("data: ")]


class TestEntitlementIsSettledBeforeTheStream:
    def test_refusal_is_a_403_not_a_200_stream(self, client, model_service, monkeypatch):
        _stub_downloader(monkeypatch, side_effect=ModelAccessError(
            f"Model '{PAID_MODEL}' is not available on your current plan."))

        resp = client.post("/v1/init", json={"model_name": PAID_MODEL})

        assert resp.status_code == 403
        assert "current plan" in resp.json()["detail"]

    def test_rate_limit_surfaces_as_429(self, client, model_service, monkeypatch):
        # Not 403: retrying fixes this one, and calling it a plan problem would
        # send the user to a billing page instead of back in a minute.
        _stub_downloader(monkeypatch, side_effect=_tagged(
            ConnectionError, "Too many download requests. Please wait a moment and try again.", 429))

        resp = client.post("/v1/init", json={"model_name": PAID_MODEL})

        assert resp.status_code == 429

    def test_platform_unreachable_is_502(self, client, model_service, monkeypatch):
        _stub_downloader(monkeypatch, side_effect=ConnectionError("Connection error: refused"))

        resp = client.post("/v1/init", json={"model_name": PAID_MODEL})

        assert resp.status_code == 502

    def test_unknown_model_is_still_400(self, client, model_service):
        # The only remaining 400 on this route. It is not fixed by signing in,
        # so a client must be able to tell it apart from an entitlement refusal.
        resp = client.post("/v1/init", json={"model_name": "no-such-model"})

        assert resp.status_code == 400
        assert "Unknown model" in resp.json()["detail"]

    def test_free_model_is_resolved_without_credentials(
        self, client, model_service, monkeypatch
    ):
        # The point of DEV-383: the old behaviour was a 400 ("Authentication
        # required") for any download without a token. A free model must now get
        # as far as asking the platform for URLs.
        made = _stub_downloader(monkeypatch)

        with client.stream("POST", "/v1/init", json={"model_name": FREE_MODEL}) as resp:
            assert resp.status_code == 200

        made.resolve_artifacts.assert_called_once()

    def test_resolved_urls_are_handed_to_the_download(
        self, client, model_service, monkeypatch
    ):
        """The download route is rate-limited, so the URLs are minted once."""
        artifacts = {"model.gguf": "https://signed/url"}
        made = _stub_downloader(monkeypatch, artifacts=artifacts)

        client.post("/v1/init", json={"model_name": PAID_MODEL})

        assert made.download.call_args.kwargs["artifacts"] == artifacts


class TestMidStreamFailureCarriesItsStatus:
    """A failure after the stream opened still has to be classifiable.

    The status line is 200 by then, so the code rides on the event for the
    client to handle it the same way it handles the pre-stream status.
    """

    def test_late_plan_denial_is_tagged_403(self, client, model_service, monkeypatch):
        # A plan that changed between resolving the URLs and using them.
        made = _stub_downloader(monkeypatch)
        made.download.side_effect = ModelAccessError(
            f"Model '{PAID_MODEL}' is not available on your current plan.")

        resp = client.post("/v1/init", json={"model_name": PAID_MODEL})

        assert resp.status_code == 200
        error = next(f for f in _frames(resp) if f.get("status") == "error")
        assert error["code"] == 403
        assert "not available on your current plan" in error["message"]

    def test_expired_url_is_tagged_with_its_own_status(
        self, client, model_service, monkeypatch
    ):
        _stub_downloader(monkeypatch)

        async def boom(*args, **kwargs):
            raise _tagged(PermissionError, "Download rejected (401 Unauthorized).", 401)
            yield  # pragma: no cover - makes this an async generator

        monkeypatch.setattr(init_router, "_download_and_stream", boom)

        resp = client.post("/v1/init", json={"model_name": PAID_MODEL})

        assert resp.status_code == 200  # already committed; the body carries the failure
        errors = [f for f in _frames(resp) if f.get("status") == "error"]
        assert errors, f"no error frame in stream: {resp.text!r}"
        assert errors[0]["code"] == 401


class TestMidStreamFailureIsClassifiedByWhatFixesIt:
    """The code has to answer "would trying again help?", not just "it broke".

    Every one of these used to arrive as a bare 500 or, worse, a 403 — which the
    desktop client renders as "Sign in to download this model" for a signed-out
    user and "Your plan doesn't include this model" for a signed-in one. A full
    disk telling someone to upgrade their plan is the failure this guards.
    """

    @pytest.mark.parametrize("exc,expected", [
        # Aggregated by _aggregate_error once the transfer gives up: a dropped
        # socket, an expired signed URL, an upstream 5xx.
        (ConnectionError("Failed to download 1 file(s): model.gguf"), 502),
        (init_router.ModelAccessError("not on your plan"), 403),
        # An OSError subclass, and emphatically not an entitlement verdict.
        (PermissionError("[Errno 13] Permission denied: '/models'"), 500),
        (OSError(28, "No space left on device"), 500),
        (RuntimeError("model.gguf: MD5 verification failed"), 500),
    ])
    def test_status_reflects_the_remedy(self, exc, expected):
        assert init_router._status_for(exc) == expected

    def test_upstream_status_still_wins(self):
        """A platform error carries its own status and keeps it."""
        assert init_router._status_for(
            _tagged(ConnectionError, "Too many download requests.", 429)) == 429
        assert init_router._status_for(
            _tagged(PermissionError, "Sign in to download this model.", 401)) == 401

    def test_dropped_socket_reaches_the_client_as_502(
        self, client, model_service, monkeypatch
    ):
        _stub_downloader(monkeypatch)

        async def boom(*args, **kwargs):
            raise ConnectionError("Failed to download 1 file(s): model.gguf")
            yield  # pragma: no cover - makes this an async generator

        monkeypatch.setattr(init_router, "_download_and_stream", boom)

        resp = client.post("/v1/init", json={"model_name": PAID_MODEL})

        assert resp.status_code == 200
        errors = [f for f in _frames(resp) if f.get("status") == "error"]
        assert errors, f"no error frame in stream: {resp.text!r}"
        assert errors[0]["code"] == 502

    def test_local_disk_refusal_is_not_a_sign_in_prompt(
        self, client, model_service, monkeypatch
    ):
        _stub_downloader(monkeypatch)

        async def boom(*args, **kwargs):
            raise PermissionError("[Errno 13] Permission denied: '/models'")
            yield  # pragma: no cover - makes this an async generator

        monkeypatch.setattr(init_router, "_download_and_stream", boom)

        resp = client.post("/v1/init", json={"model_name": PAID_MODEL})

        errors = [f for f in _frames(resp) if f.get("status") == "error"]
        assert errors, f"no error frame in stream: {resp.text!r}"
        assert errors[0]["code"] == 500
