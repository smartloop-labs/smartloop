"""Stopping a download from a second request, and what the stream then says.

Two halves: ``DELETE /v1/downloads/{name}`` finding the running transfer, and
``/v1/init`` answering the cancel with a terminal ``cancelled`` frame instead of
an error — and, more importantly, without any of the nine things it does after
a download completes.
"""

import asyncio
import contextlib
import json
import threading
from unittest.mock import MagicMock, patch

import pytest

import smartloop.api.routers.init as init_router_module
from smartloop.api.main import app
from smartloop.api.services import get_model_service
from smartloop.model.downloader import cancellation
from smartloop.model.downloader.artifact_downloader import SLPArtifactDownloader
from smartloop.model.downloader.cancellation import DownloadCancelled

_DISCARD = "smartloop.api.routers.init.storage_service.discard_partial_download"

MODEL_NAME = "gemma3-4b"
MODEL_ID = "google/gemma-3-4b-it"


@pytest.fixture(autouse=True)
def _clean_registry():
    cancellation.reset_for_tests()
    yield
    cancellation.reset_for_tests()


def _workspace_model(model_id=MODEL_ID, name=MODEL_NAME):
    entry = MagicMock()
    entry.model_id = model_id
    entry.model_name = name
    entry.model_dir = f"/nowhere/{model_id}"
    return entry


@pytest.fixture
def model_service():
    """A MagicMock ModelService bound via FastAPI dependency override."""
    mock = MagicMock()
    workspace = MagicMock()
    workspace.get_model.return_value = _workspace_model()
    mock._ensure_workspace.return_value = workspace
    app.dependency_overrides[get_model_service] = lambda: mock
    yield mock
    app.dependency_overrides.pop(get_model_service, None)


class TestResolvingTheModel:
    def test_it_uses_the_workspace_record(self, client, model_service):
        cancellation.begin(MODEL_ID)
        resp = client.delete(f"/v1/downloads/{MODEL_NAME}")

        assert resp.status_code == 200
        assert resp.json()["model_id"] == MODEL_ID

    def test_it_falls_back_to_the_catalog_before_registration(self, client, model_service):
        """A cancel can land before ``/v1/init`` has added the workspace entry."""
        model_service._ensure_workspace.return_value.get_model.return_value = None

        resp = client.delete("/v1/downloads/sl-mini")

        assert resp.status_code == 200
        assert resp.json()["model_id"]

    def test_an_unknown_model_is_a_400(self, client, model_service):
        model_service._ensure_workspace.return_value.get_model.return_value = None

        resp = client.delete("/v1/downloads/not-a-model")

        assert resp.status_code == 400
        assert "not-a-model" in resp.json()["detail"]

    def test_a_workspace_failure_is_a_503(self, client, model_service):
        model_service._ensure_workspace.side_effect = RuntimeError("Workspace not initialized")

        resp = client.delete(f"/v1/downloads/{MODEL_NAME}")

        assert resp.status_code == 503


class TestCancelling:
    def test_an_active_download_is_told_to_stop(self, client, model_service):
        token = cancellation.begin(MODEL_ID)

        resp = client.delete(f"/v1/downloads/{MODEL_NAME}")

        assert resp.status_code == 200
        assert resp.json() == {"cancelled": True, "status": "cancelling",
                               "model_name": MODEL_NAME, "model_id": MODEL_ID}
        assert token.cancelled

    def test_nothing_running_is_a_200_not_a_404(self, client, model_service):
        """The client races by construction: a progress frame arrives, the user
        clicks, the transfer finishes before the request lands.  A 404 would
        push the picker into an error path for an ordinary outcome."""
        resp = client.delete(f"/v1/downloads/{MODEL_NAME}")

        assert resp.status_code == 200
        assert resp.json()["cancelled"] is False
        assert resp.json()["status"] == "not_found"

    def test_a_second_cancel_is_idempotent(self, client, model_service):
        cancellation.begin(MODEL_ID)

        first = client.delete(f"/v1/downloads/{MODEL_NAME}")
        second = client.delete(f"/v1/downloads/{MODEL_NAME}")

        assert first.json() == second.json()

    def test_it_deletes_nothing_itself(self, client, model_service):
        """Cleanup belongs to the download thread — the only one that knows
        every worker has stopped.  Unlinking here would race an open write."""
        cancellation.begin(MODEL_ID)

        with patch(_DISCARD) as discard:
            client.delete(f"/v1/downloads/{MODEL_NAME}")

        discard.assert_not_called()

    def test_it_returns_before_the_threads_stop(self, client, model_service):
        """A stalled socket can take 60s to die; the route must not wait."""
        cancellation.begin(MODEL_ID)

        resp = client.delete(f"/v1/downloads/{MODEL_NAME}")

        assert resp.json()["status"] == "cancelling"      # not "stopped"

    def test_it_cancels_the_model_the_caller_named(self, client, model_service):
        other = cancellation.begin("google/gemma-4-e4b-it")
        mine = cancellation.begin(MODEL_ID)

        client.delete(f"/v1/downloads/{MODEL_NAME}")

        assert mine.cancelled
        assert not other.cancelled


class TestWaitingForItToStop:
    def test_it_reports_stopped_once_the_bytes_are_gone(self, client, model_service):
        token = cancellation.begin(MODEL_ID)
        threading.Timer(0.05, lambda: cancellation.end(MODEL_ID, token)).start()

        resp = client.delete(f"/v1/downloads/{MODEL_NAME}?wait_ms=5000")

        assert resp.json()["status"] == "stopped"

    def test_it_still_reports_cancelling_when_the_wait_runs_out(self, client, model_service):
        cancellation.begin(MODEL_ID)

        resp = client.delete(f"/v1/downloads/{MODEL_NAME}?wait_ms=50")

        assert resp.json() == {"cancelled": True, "status": "cancelling",
                               "model_name": MODEL_NAME, "model_id": MODEL_ID}

    def test_a_negative_wait_is_rejected(self, client, model_service):
        resp = client.delete(f"/v1/downloads/{MODEL_NAME}?wait_ms=-1")
        assert resp.status_code == 422


def _frames(resp):
    """Parse an SSE body into ``[(event, data), …]``."""
    out = []
    event = "message"
    for line in resp.text.splitlines():
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            out.append((event, json.loads(line[len("data: "):])))
            event = "message"
    return out


class TestTheInitStreamAnswersACancel:
    """``/v1/init``'s download raises, and the stream has to end honestly.

    The transfer is stubbed to raise rather than really run: the autouse
    ``no_real_downloads`` fixture already replaces ``download`` for this whole
    package, and what is under test is the router's handling, not the byte
    loops (``tests/unit/test_download_cancel.py`` covers those).
    """

    @pytest.fixture
    def cancelled_download(self, monkeypatch):
        def _raise(self, model_id=None, progress_callback=None, **kwargs):
            raise DownloadCancelled("download cancelled")

        monkeypatch.setattr(SLPArtifactDownloader, "download", _raise)
        monkeypatch.setattr(SLPArtifactDownloader, "resolve_artifacts",
                            lambda self, model_id: {"weights.gguf": "https://x/f"})

    def test_it_ends_with_a_cancelled_event(self, client, model_service, cancelled_download):
        with patch(_DISCARD, return_value={"removed": True}):
            resp = client.post("/v1/init", json={"model_name": MODEL_NAME})

        assert resp.status_code == 200
        events = [name for name, _ in _frames(resp)]
        assert "cancelled" in events

    def test_it_emits_no_error_frame(self, client, model_service, cancelled_download):
        """A cancel read as an error shows a failure toast for the user's own
        button."""
        with patch(_DISCARD, return_value={"removed": True}):
            resp = client.post("/v1/init", json={"model_name": MODEL_NAME})

        assert "error" not in [name for name, _ in _frames(resp)]

    def test_it_emits_no_complete_frame(self, client, model_service, cancelled_download):
        with patch(_DISCARD, return_value={"removed": True}):
            resp = client.post("/v1/init", json={"model_name": MODEL_NAME})

        assert "complete" not in [name for name, _ in _frames(resp)]

    def test_it_discards_the_partial_bytes(self, client, model_service, cancelled_download):
        with patch(_DISCARD, return_value={"removed": True}) as discard:
            client.post("/v1/init", json={"model_name": MODEL_NAME})

        assert discard.call_args.kwargs["model_id"] == MODEL_ID

    def test_a_failed_discard_does_not_become_an_error_frame(self, client, model_service, cancelled_download):
        """Bytes left behind are the abandoned-download case the storage sweep
        already collects; the cancel itself still succeeded."""
        with patch(_DISCARD, side_effect=OSError("busy")):
            resp = client.post("/v1/init", json={"model_name": MODEL_NAME})

        events = [name for name, _ in _frames(resp)]
        assert "cancelled" in events and "error" not in events

    def test_it_does_not_mark_the_model_downloaded(self, client, model_service, cancelled_download):
        """The head of a list of nine side effects the early return skips."""
        with patch(_DISCARD, return_value={"removed": True}):
            client.post("/v1/init", json={"model_name": MODEL_NAME})

        model_service.mark_model_downloaded.assert_not_called()

    def test_it_does_not_promote_the_app_default(self, client, model_service, cancelled_download):
        """A model earns its place among the app-defaults by arriving."""
        with patch(_DISCARD, return_value={"removed": True}):
            client.post("/v1/init", json={"model_name": MODEL_NAME})

        workspace = model_service._ensure_workspace.return_value
        workspace.promote_model_to_workspace_default.assert_not_called()

    def test_the_registration_is_released(self, client, model_service, cancelled_download):
        with patch(_DISCARD, return_value={"removed": True}):
            client.post("/v1/init", json={"model_name": MODEL_NAME})

        assert cancellation.active_keys() == []

    def test_a_completed_download_also_releases_it(self, client, model_service):
        """The stub stands in for a successful transfer."""
        client.post("/v1/init", json={"model_name": MODEL_NAME})

        assert cancellation.active_keys() == []




class TestADroppedStreamIsNotACancel:
    """A window reload must not kill a multi-GB transfer.

    This is what claiming and releasing the registration on the *executor
    thread* buys.  Hung off the SSE generator instead, the unwinding a closed
    stream causes there would unregister a download still consuming bandwidth,
    and the user's next cancel would be answered ``not_found``.

    Driven against ``_download_and_stream`` rather than through ``/v1/init``:
    the generator is where the ownership question lives, and the route around it
    would drag in the catalog seed, the model pipeline and the project setup for
    a property none of them participate in.
    """

    def test_the_transfer_outlives_the_stream_and_still_cleans_up_after_itself(self):
        started = threading.Event()
        release = threading.Event()

        downloader = MagicMock()

        def _blocking(model_id=None, progress_callback=None, **kwargs):
            started.set()
            # Bounded, so a regression cannot wedge the suite.
            release.wait(10)

        downloader.download.side_effect = _blocking

        async def drive():
            agen = init_router_module._download_and_stream(
                downloader, MODEL_ID, MODEL_NAME, model_service=MagicMock())

            async def consume():
                async for _ in agen:
                    pass

            task = asyncio.create_task(consume())
            loop = asyncio.get_running_loop()
            # Wait for the transfer to really be under way: the opening frame is
            # yielded before the executor task is even submitted, so closing the
            # stream at that point would prove nothing.
            assert await loop.run_in_executor(None, started.wait, 5), \
                "the download never started"

            task.cancel()                           # the client walks away
            with contextlib.suppress(asyncio.CancelledError):
                await task

            token = cancellation.find(MODEL_ID)
            assert token is not None, "a dropped stream unregistered a live download"
            assert not token.cancelled, "a dropped stream cancelled a live download"

            # And the other half: the executor thread releases the registration
            # on its own, though the generator that claimed it is long gone.
            release.set()
            assert await loop.run_in_executor(None, token.done.wait, 5), \
                "the transfer never released its registration"
            assert cancellation.find(MODEL_ID) is None

        try:
            asyncio.run(drive())
        finally:
            release.set()
