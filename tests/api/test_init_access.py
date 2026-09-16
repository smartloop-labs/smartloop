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
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from smartloop.api.main import app
from smartloop.api.routers import init as init_router
from smartloop.api.routers._errors import status_for
from smartloop.api.services import ModelService, get_model_service
from smartloop.platform import ModelAccessError
from smartloop.workspace.workspace import Workspace, WorkspaceModel

FREE_MODEL = "gemma3-4b"
PAID_MODEL = "gemma3-12b"


@pytest.fixture
def model_service(tmp_path_factory, monkeypatch):
    """A model service whose workspace has no weights on disk.

    ``models_dir`` points into an empty temp dir, so ``_gguf_exists`` is False
    and every request in this module takes the "needs download" path.

    Deliberately not the test's ``tmp_path``: ``no_real_downloads`` scans that
    tree afterwards for stray ``.gguf`` files, and one test here plants a fake
    one to mean "these weights are already on disk".  The guard cannot tell
    that apart from a download that escaped the stub, and it should not have to
    — so the workspace keeps its files where the guard is not looking, the same
    arrangement ``test_init_promotes_a_downloaded_model`` uses.
    """
    home = tmp_path_factory.mktemp("slp-home")
    # ``save()`` writes ``Workspace.config_file``, which resolves from
    # ``SLP_HOME`` and *not* from ``workspace_dir`` — config.json sits one level
    # above a workspace, so it cannot be derived from it.  Setting the two dirs
    # below therefore sandboxes everything except the file the routes here
    # actually persist, and ``/v1/init`` now really does persist it.  Without
    # this the suite overwrites the developer's own ``~/.smartloop/config.json``
    # with this fixture's workspace.
    monkeypatch.setenv("SLP_HOME", str(home))
    workspace = Workspace(
        id="ws-test",
        workspace_dir=str(home),
        models_dir=str(home / "models"),
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
        (ModelAccessError("not on your plan"), 403),
        # An OSError subclass, and emphatically not an entitlement verdict.
        (PermissionError("[Errno 13] Permission denied: '/models'"), 500),
        (OSError(28, "No space left on device"), 500),
        (RuntimeError("model.gguf: MD5 verification failed"), 500),
    ])
    def test_status_reflects_the_remedy(self, exc, expected):
        assert status_for(exc) == expected

    def test_upstream_status_still_wins(self):
        """A platform error carries its own status and keeps it."""
        assert status_for(
            _tagged(ConnectionError, "Too many download requests.", 429)) == 429
        assert status_for(
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


class TestInitOnAFreshProcess:
    """/v1/init must not need a prior request to have warmed the service.

    ``ModelService`` is a process-wide singleton whose ``_model`` starts as None;
    ``_ensure_model()`` is what assigns it, and it runs *after* the download.
    The router used to call ``_model.save()`` immediately after
    ``mark_model_downloaded``, so on a fresh process the weights landed on disk
    and the client still got an error frame. Both branches hit that line, so the
    retry failed identically: "model_ready", then error, forever.
    """

    @pytest.fixture
    def cold_service(self, model_service):
        """A service in the state a just-started process is really in.

        ``_model`` is None until ``_ensure_model()`` runs, so the mock assigns it
        there rather than pretending it was always available — that ordering is
        the whole point of the test.
        """
        workspace = model_service._ensure_workspace.return_value
        model_service._model = None

        def ensure_model():
            model_service._model = MagicMock(workspace=workspace)

        model_service._ensure_model.side_effect = ensure_model
        return model_service

    def test_download_completes_without_a_warmed_model(
        self, client, cold_service, monkeypatch
    ):
        _stub_downloader(monkeypatch)

        async def streamed(*args, **kwargs):
            yield init_router._sse_event({"status": "downloading", "message": "..."})

        monkeypatch.setattr(init_router, "_download_and_stream", streamed)

        resp = client.post("/v1/init", json={"model_name": PAID_MODEL})

        assert resp.status_code == 200
        errors = [f for f in _frames(resp) if f.get("status") == "error"]
        assert not errors, f"fresh-process init errored: {errors}"
        cold_service.mark_model_downloaded.assert_called_once()
        assert cold_service.mark_model_downloaded.call_args.args[0] == PAID_MODEL

    def test_already_downloaded_also_survives_a_fresh_process(
        self, client, cold_service
    ):
        """The model_ready branch dereferenced the same None."""
        # Where the route will look, rather than a path built to match it: the
        # workspace resolves ``model_dir`` from its own ``models_dir``, so
        # asking it keeps the sentinel and the lookup from drifting apart.
        workspace = cold_service._ensure_workspace.return_value
        model_dir = Path(workspace.get_model(PAID_MODEL).model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "weights.gguf").write_bytes(b"\0")

        resp = client.post("/v1/init", json={"model_name": PAID_MODEL})

        frames = _frames(resp)
        assert any(f.get("status") == "model_ready" for f in frames), frames
        assert not [f for f in frames if f.get("status") == "error"], frames

    def test_the_download_flag_reaches_disk(self, client, cold_service, monkeypatch):
        """The flag is only useful if it is persisted, and it must not need _model.

        ``mark_model_downloaded`` resolves the workspace itself and saves it, so
        the flag survives a restart whether or not anything warmed the service.
        """
        workspace = cold_service._ensure_workspace.return_value
        cold_service.mark_model_downloaded.side_effect = \
            lambda name, workspace=None: ModelService.mark_model_downloaded(
                cold_service, name, workspace=workspace)
        saves = []
        monkeypatch.setattr(type(workspace), "save", lambda self: saves.append(1))
        _stub_downloader(monkeypatch)

        async def streamed(*args, **kwargs):
            yield init_router._sse_event({"status": "downloading", "message": "..."})

        monkeypatch.setattr(init_router, "_download_and_stream", streamed)

        resp = client.post("/v1/init", json={"model_name": PAID_MODEL})

        assert not [f for f in _frames(resp) if f.get("status") == "error"]
        assert saves, "the workspace was never persisted"
        assert workspace.get_model(PAID_MODEL).downloaded is True

    def test_the_router_hands_over_its_own_workspace(
        self, client, cold_service, monkeypatch
    ):
        """The flag has to land on the instance the router mutated.

        This fixture's ``_ensure_workspace`` is a mock returning one object, so
        the test above cannot tell the two instances apart — it passes whether
        or not the workspace is threaded through.  Asserting the hand-off is
        what pins it here; ``test_the_flag_survives_a_second_load`` covers the
        real two-instance behaviour.
        """
        _stub_downloader(monkeypatch)

        async def streamed(*args, **kwargs):
            yield init_router._sse_event({"status": "downloading", "message": "..."})

        monkeypatch.setattr(init_router, "_download_and_stream", streamed)

        resp = client.post("/v1/init", json={"model_name": PAID_MODEL})

        assert resp.status_code == 200
        _, kwargs = cold_service.mark_model_downloaded.call_args
        assert kwargs.get("workspace") is cold_service._ensure_workspace.return_value

    def test_the_flag_survives_a_second_load(self, tmp_path, monkeypatch):
        """A cold ``_ensure_workspace`` builds a *fresh* workspace every call.

        The router registers the model on its own instance and deliberately
        does not save (a refused download must strand nothing in config.json),
        so resolving a second instance inside ``mark_model_downloaded`` read
        the model back off disk *without* that registration, skipped the flag
        for a model it could not find, and saved the stale copy over it —
        leaving config.json with no record of the model at all.
        """
        monkeypatch.setattr(Workspace, "config_file", str(tmp_path / "config.json"))
        Workspace(
            id="ws-1",
            workspace_dir=str(tmp_path / "ws-1"),
            models_dir=str(tmp_path / "ws-1" / "models"),
        ).save()

        svc = MagicMock()
        svc._model = None
        svc._ensure_workspace = lambda cfg=None: ModelService._ensure_workspace(svc, cfg)

        # What the router does before the stream opens.
        routers_workspace = svc._ensure_workspace()
        routers_workspace.add_model(PAID_MODEL, "google/gemma-3-12b-it", enabled=False)
        assert svc._ensure_workspace() is not routers_workspace, \
            "a cold _ensure_workspace is supposed to return a fresh instance"

        # What the stream does after the transfer.
        ModelService.mark_model_downloaded(svc, PAID_MODEL, workspace=routers_workspace)

        reloaded = Workspace.load()
        assert [m.model_name for m in reloaded.models] == [PAID_MODEL], \
            "the router's add_model was clobbered by a stale instance"
        assert reloaded.get_model(PAID_MODEL).downloaded is True
