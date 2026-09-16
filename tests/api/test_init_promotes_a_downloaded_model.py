"""A model becomes a workspace app-default by arriving, not by being enabled.

``/v1/init`` is where a model enters the workspace.  When it is fetched for a
project the workspace record is registered *disabled* on purpose, so a
cancelled download strands nothing — which leaves something to flip once the
weights actually land.

That flip used to be attached to the wrong event.  ``update_project_model``
promoted on any enable, and under supervision that route is proxied to the
child pinned to the project — a process that writes only its own
``project.json``.  So the promotion was discarded and ``ws.enabled`` stayed
false forever: every model ever downloaded from inside a project was off for
projects created later, and shown disabled in the workspace list.

Promoting here instead puts it in the one process that owns ``config.json``
(``/v1/init`` names no project in its path, so it is never proxied) and ties it
to the one event that means "this model is new to the workspace".  Switching on
a model that was already here stays project-scoped — see
``TestEnablingAModelIsProjectScoped`` in ``test_project_models.py``.
"""

import os
from unittest.mock import MagicMock

import pytest

from smartloop.api.main import app
from smartloop.api.routers import init as init_router
from smartloop.api.services import get_model_service
from smartloop.workspace.workspace import Workspace, WorkspaceModel

MODEL = "gemma3-4b"
MODEL_ID = "google/gemma-3-4b-it"
PROJECT = "p1"


@pytest.fixture
def workspace(tmp_path_factory, monkeypatch):
    """A real workspace under a temp home, with no weights on disk.

    Real rather than mocked because the assertions are about what ``save()``
    leaves in ``config.json`` — the workspace record is the whole subject.

    Deliberately not the ``tmp_path`` of the test: ``no_real_downloads`` scans
    that tree afterwards for stray ``.gguf`` files, and one test here plants a
    fake one to mean "these weights are already on disk".  The guard cannot
    tell that apart from a download that escaped the stub, and it should not
    have to — so the fixture keeps its files where the guard is not looking.
    """
    home = tmp_path_factory.mktemp("slp-home")
    monkeypatch.setenv("SLP_HOME", str(home))
    Workspace.reset()
    ws = Workspace(
        id="ws-test",
        workspace_dir=str(home / "ws-test"),
        models_dir=str(home / "ws-test" / "models"),
    )
    yield ws
    Workspace.reset()


@pytest.fixture
def model_service(workspace, monkeypatch):
    svc = MagicMock()
    svc._ensure_workspace.return_value = workspace
    # The same instance the router saves, so ``current_project`` resolves to
    # None rather than to a mock the SSE encoder cannot serialise.
    svc._model.workspace = workspace
    # Off the critical path and a real cloud call otherwise.
    monkeypatch.setattr(init_router, "_seed_model_catalog", lambda: None)
    app.dependency_overrides[get_model_service] = lambda: svc
    yield svc
    app.dependency_overrides.pop(get_model_service, None)


def _stub_downloader(monkeypatch):
    """A downloader that resolves and 'fetches' without touching the network."""
    made = MagicMock()
    made.resolve_artifacts.return_value = {"model.gguf": "https://signed/url"}
    monkeypatch.setattr(init_router, "SLPArtifactDownloader", lambda *a, **k: made)
    return made


def _register(workspace, *, enabled: bool, downloaded: bool) -> WorkspaceModel:
    ws_model = workspace.add_model(MODEL, MODEL_ID, enabled=enabled)
    ws_model.downloaded = downloaded
    return ws_model


def _put_weights_on_disk(ws_model) -> None:
    """Make ``_gguf_exists`` true, which is what 'already available' means."""
    os.makedirs(ws_model.model_dir, exist_ok=True)
    open(os.path.join(ws_model.model_dir, "model.gguf"), "w").close()


class TestADownloadPromotesTheModel:
    def test_a_project_scoped_download_flips_the_workspace_default(
        self, client, workspace, model_service, monkeypatch
    ):
        _stub_downloader(monkeypatch)

        resp = client.post("/v1/init", json={"model_name": MODEL, "project_id": PROJECT})

        assert resp.status_code == 200
        # Registered disabled by the route, then promoted once the weights landed.
        assert workspace.get_model(MODEL).enabled is True

    def test_it_survives_the_reload_the_next_request_does(
        self, client, workspace, model_service, monkeypatch
    ):
        """The point of doing it here: this process may write ``config.json``."""
        _stub_downloader(monkeypatch)

        client.post("/v1/init", json={"model_name": MODEL, "project_id": PROJECT})

        reloaded = Workspace.load()
        assert reloaded.get_model(MODEL).enabled is True

    def test_a_workspace_scoped_download_is_still_enabled(
        self, client, workspace, model_service, monkeypatch
    ):
        """No project named (CLI/TUI): registered enabled, promote is a no-op."""
        _stub_downloader(monkeypatch)

        client.post("/v1/init", json={"model_name": MODEL})

        assert workspace.get_model(MODEL).enabled is True


class TestNothingElsePromotes:
    def test_an_already_available_model_is_not_promoted(
        self, client, workspace, model_service, monkeypatch
    ):
        """No download ran, so nothing says this model is new to the workspace.

        This is the case the rule excludes: the user is switching on something
        that was already here, which is their project's business and not the
        default for every project made afterwards.
        """
        ws_model = _register(workspace, enabled=False, downloaded=True)
        _put_weights_on_disk(ws_model)

        resp = client.post("/v1/init", json={"model_name": MODEL, "project_id": PROJECT})

        assert resp.status_code == 200
        assert workspace.get_model(MODEL).enabled is False

    def test_an_already_enabled_model_is_left_alone(
        self, client, workspace, model_service, monkeypatch
    ):
        """Re-downloading must not re-pin every project that has no entry."""
        _register(workspace, enabled=True, downloaded=False)
        _stub_downloader(monkeypatch)

        client.post("/v1/init", json={"model_name": MODEL, "project_id": PROJECT})

        assert workspace.get_model(MODEL).enabled is True
