"""API test fixtures.

Nothing here may reach the network, and one guarantee needs stating outright:
**no test downloads a model.**  The routers under test build their own
``SLPArtifactDownloader`` rather than receiving one, so overriding a FastAPI
dependency does not stop them — and ``/v1/init`` hands the transfer to
``run_in_executor``, whose thread outlives the response.  A test that closed
the stream to avoid "running a real download" therefore left one running: the
weights kept arriving, into a ``tmp_path`` pytest would later try to delete,
long after the assertion had passed.  A full run wrote tens of gigabytes of
real GGUFs this way and filled the disk.

So the download methods are stubbed for every test in this package, at the
class rather than at any one importing module — three call sites build a
downloader (``/v1/init``, ``/v1/bootstrap``, and ``GenericModel.init`` by way
of ``ModelFactory``), and a stub attached to one of them silently misses the
others.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from smartloop.api.main import app
from smartloop.model.downloader.artifact_downloader import SLPArtifactDownloader
from smartloop.model.downloader.slp_downloader import SLPDownloader
from smartloop.platform import get_platform_service

#: Files a real download leaves behind, complete or interrupted.
_WEIGHT_SUFFIXES = (".gguf", ".part")


@pytest.fixture(autouse=True)
def no_real_downloads(monkeypatch, tmp_path):
    """Stub every path that would fetch weights, and prove none was taken.

    Autouse and un-opt-out-able by design: a download is never what an API
    test is asserting, and the failure mode when one slips through is silent —
    the test passes, and the machine fills up minutes later from a thread
    nobody is watching.

    The stubs stand in for a *successful* download, so a route that goes on to
    register the model still behaves as it would in life.  What they do not do
    is touch the network or the disk.
    """
    calls: list[tuple] = []

    def _fake_download(self, model_id, progress_callback=None, **kwargs):
        calls.append(("download", model_id))
        return Path(getattr(self, "models_dir", str(tmp_path)))

    def _fake_bootstrap(self, model_name=None, progress_callback=None, **kwargs):
        calls.append(("bootstrap_download", model_name))
        return Path(getattr(self, "models_dir", str(tmp_path)))

    monkeypatch.setattr(SLPArtifactDownloader, "download", _fake_download)
    monkeypatch.setattr(SLPArtifactDownloader, "bootstrap_download", _fake_bootstrap)
    # ``GenericModel.init`` downloads through the base class, so a model
    # pipeline built during a test reaches a different method than the routes.
    monkeypatch.setattr(SLPDownloader, "download", _fake_download, raising=False)

    yield calls

    # The stub can stop matching — a route that starts building its downloader
    # somewhere new, or a rename — and the only symptom would be the disk
    # filling again. Assert the outcome rather than trusting the patch.
    leaked = [
        p for p in tmp_path.rglob("*")
        if p.is_file() and p.suffix in _WEIGHT_SUFFIXES
    ]
    assert not leaked, (
        "a real model download escaped the stub and wrote "
        f"{[str(p) for p in leaked]} — check whether a new code path builds "
        "its own downloader"
    )


@pytest.fixture
def platform_service():
    """A MagicMock PlatformService bound via FastAPI dependency override.

    Tests can configure return values or side_effects on
    ``platform_service.get_current_user`` / ``update_current_user`` /
    ``list_models`` to drive router behavior. No real DB or HTTP I/O.
    """
    mock = MagicMock()
    app.dependency_overrides[get_platform_service] = lambda: mock
    yield mock
    app.dependency_overrides.pop(get_platform_service, None)


@pytest.fixture
def client(platform_service):
    """FastAPI test client — in-process, no real server socket."""
    return TestClient(app)
