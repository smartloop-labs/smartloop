"""Tests for the per-project model endpoints (/v1/projects/{id}/models)."""

import json
import pathlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from smartloop.api.main import app
from smartloop.config.app_settings import AppSettings
from smartloop.workspace import workspace as ws_module
from smartloop.workspace.workspace import Project, ProjectModel, Workspace, WorkspaceModel
from smartloop.platform import get_platform_service


@pytest.fixture()
def isolated_workspace(tmp_path, monkeypatch):
    """Write a config.json to a temp dir and point the app's workspace at it.

    Both projects share the workspace (app-default) model registry; their
    per-project settings start empty so effective state falls back to defaults.
    The always-on bootstrap model is registered too, so the list assertions
    below also cover its exclusion.
    """
    home_dir = tmp_path / "dot_smartloop"
    home_dir.mkdir()

    ws_id = "ws-test"
    workspace = Workspace(
        id=ws_id,
        workspace_dir=str(home_dir / ws_id),
        models_dir=str(home_dir / ws_id / "models"),
        models=[
            WorkspaceModel(
                model_name=AppSettings().default_model,
                model_id="smartloop/slp-orchestra-mini",
                downloaded=True,
                enabled=True,
            ),
            WorkspaceModel(
                model_name="gemma3-4b",
                model_id="google/gemma-3-4b-it",
                downloaded=True,
                enabled=True,
            ),
            WorkspaceModel(
                model_name="qwen2-vl-2b",
                model_id="Qwen/Qwen2-VL-2B-Instruct",
                downloaded=True,
                enabled=False,
                is_active=True,
            ),
        ],
        projects=[
            Project(id="p1", project_dir="/tmp/p1", current=True, name="p1"),
            Project(id="p2", project_dir="/tmp/p2", current=False, name="p2"),
        ],
    )

    config_file = home_dir / "config.json"
    ws_data = workspace.model_dump(mode="json")
    ws_data.pop("current_project", None)
    config_file.write_text(json.dumps({"id": "gemma-3-4b-it", "workspace": ws_data}))

    settings = AppSettings()
    settings.slp_home = str(home_dir)
    monkeypatch.setattr(ws_module, "_settings", settings)
    monkeypatch.setattr(Workspace, "config_file", str(config_file))

    # Fresh model service so _ensure_workspace() re-reads the isolated config.
    import smartloop.api.services as services
    monkeypatch.setattr(services, "_model_service", None)

    return {"workspace": workspace, "config_file": str(config_file)}


@pytest.fixture()
def client(platform_service):
    app.dependency_overrides[get_platform_service] = lambda: platform_service
    yield TestClient(app)
    app.dependency_overrides.pop(get_platform_service, None)



def _saved_project(config_file: str, project_id: str) -> dict:
    """A project's persisted record, read back from its own file.

    Per-project state no longer lives in ``config.json`` — that keeps a thin
    index, and each project owns a ``project.json`` in its own directory.  The
    index is what says where, so this walks the same path a load does.
    """
    index = json.loads(pathlib.Path(config_file).read_text())["workspace"]["projects"]
    entry = next(p for p in index if p["id"] == project_id)
    state = pathlib.Path(entry["project_dir"]) / "project.json"
    if not state.is_file():
        # Nothing has saved this project yet — the index entry is all there is,
        # which is itself the answer for a test asserting nothing was written.
        return entry
    return json.loads(state.read_text())["project"]

class TestListProjectModels:
    def test_lists_models_with_workspace_defaults(self, isolated_workspace, client):
        resp = client.get("/v1/projects/p1/models")

        assert resp.status_code == 200
        data = {m["model_name"]: m for m in resp.json()}
        assert set(data) == {"gemma3-4b", "qwen2-vl-2b"}
        assert data["gemma3-4b"]["enabled"] is True
        assert data["gemma3-4b"]["downloaded"] is True
        assert data["qwen2-vl-2b"]["enabled"] is False
        assert data["qwen2-vl-2b"]["downloaded"] is True
        assert data["qwen2-vl-2b"]["is_active"] is True

    def test_404_for_unknown_project(self, isolated_workspace, client):
        resp = client.get("/v1/projects/does-not-exist/models")
        assert resp.status_code == 404

    def test_projects_are_independent(self, isolated_workspace, client):
        assert client.patch(
            "/v1/projects/p1/models/qwen2-vl-2b", json={"enabled": True}
        ).status_code == 200

        p1 = {m["model_name"]: m for m in client.get("/v1/projects/p1/models").json()}
        p2 = {m["model_name"]: m for m in client.get("/v1/projects/p2/models").json()}

        assert p1["qwen2-vl-2b"]["enabled"] is True
        # p2 was never touched — still the workspace default.
        assert p2["qwen2-vl-2b"]["enabled"] is False


class TestUpdateProjectModel:
    def test_updates_enabled_state_per_project(self, isolated_workspace, client):
        resp = client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"enabled": False}
        )

        assert resp.status_code == 200
        assert resp.json()["enabled"] is False
        assert resp.json()["model_name"] == "gemma3-4b"
        assert resp.json()["capabilities"] != []

    def test_downloaded_is_read_only(self, isolated_workspace, client):
        """A client cannot assert download state — it describes the disk.

        The field is not on the request model, so it is ignored rather than
        written, and the response keeps reporting the workspace-wide value.
        """
        resp = client.patch(
            "/v1/projects/p1/models/qwen2-vl-2b", json={"downloaded": False}
        )

        assert resp.status_code == 200
        assert resp.json()["downloaded"] is True
        # Nothing per-project was written for it.
        entry = next(
            m for m in client.get("/v1/projects/p1/models").json()
            if m["model_name"] == "qwen2-vl-2b"
        )
        assert entry["downloaded"] is True

    def test_active_flag_deactivates_other_project_models(self, isolated_workspace, client):
        resp = client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"is_active": True}
        )
        assert resp.status_code == 200
        assert resp.json()["is_active"] is True

        p1 = {m["model_name"]: m for m in client.get("/v1/projects/p1/models").json()}
        assert p1["gemma3-4b"]["is_active"] is True
        assert p1["qwen2-vl-2b"]["is_active"] is False

    def test_settings_update_merges_into_project_entry(self, isolated_workspace, client):
        resp = client.patch(
            "/v1/projects/p1/models/gemma3-4b",
            json={"settings": {"temperature": 0.3}},
        )
        assert resp.status_code == 200
        assert resp.json()["settings"] == {"temperature": 0.3}

        listed = {m["model_name"]: m for m in client.get("/v1/projects/p1/models").json()}
        assert listed["gemma3-4b"]["settings"] == {"temperature": 0.3}

    def test_404_for_unknown_model(self, isolated_workspace, client):
        resp = client.patch(
            "/v1/projects/p1/models/not-a-model", json={"enabled": False}
        )
        assert resp.status_code == 404

    def test_404_for_unknown_project(self, isolated_workspace, client):
        resp = client.patch(
            "/v1/projects/nope/models/gemma3-4b", json={"enabled": False}
        )
        assert resp.status_code == 404

    def test_patch_persists_to_disk(self, isolated_workspace, client):
        client.patch("/v1/projects/p1/models/gemma3-4b", json={"enabled": False})

        from smartloop.workspace.workspace import Workspace as WS
        with patch.object(WS, "config_file", isolated_workspace["config_file"]):
            reloaded = WS.load()
        project = next(p for p in reloaded.projects if p.id == "p1")
        entry = next(m for m in project.model_settings if m.model_name == "gemma3-4b")
        assert entry.enabled is False


class TestMarkModelDownloaded:
    def test_marks_the_workspace_and_writes_no_project_state(self, isolated_workspace):
        """One flag, workspace-wide — every project sees it at once."""
        from unittest.mock import patch as _patch
        from smartloop.api.services import get_model_service
        from smartloop.workspace.workspace import Workspace as WS

        with _patch.object(WS, "config_file", isolated_workspace["config_file"]):
            ws = WS.load()
        ws.get_model("gemma3-4b").downloaded = False
        # Give both projects seeded entries, so a stray per-project write would show.
        for project in ws.projects:
            ws.sync_project_models(project)

        service = get_model_service()
        with _patch.object(service, "_ensure_workspace", return_value=ws):
            service.mark_model_downloaded("gemma3-4b")
        ws.save()

        with _patch.object(WS, "config_file", isolated_workspace["config_file"]):
            reloaded = WS.load()
        assert reloaded.get_model("gemma3-4b").downloaded is True
        # Visible from every project, with no per-project copy to drift.
        for project in reloaded.projects:
            assert reloaded.effective_model_downloaded("gemma3-4b") is True
            assert any(
                m.model_name == "gemma3-4b" for m in project.model_settings
            )
        assert "downloaded" not in ProjectModel.model_fields


class TestBootstrapModelIsNotConfigurablePerProject:
    """The orchestrator model is always enabled and hidden from the list.

    It has no per-project entry, so PATCHing it by name is rejected rather
    than silently writing state that would shadow its always-on default.
    """

    def test_it_is_absent_from_the_list(self, isolated_workspace, client):
        resp = client.get("/v1/projects/p1/models")

        assert resp.status_code == 200
        assert AppSettings().default_model not in {m["model_name"] for m in resp.json()}

    def test_patch_is_rejected(self, isolated_workspace, client):
        resp = client.patch(
            f"/v1/projects/p1/models/{AppSettings().default_model}",
            json={"enabled": False},
        )

        assert resp.status_code == 400
        assert "always enabled" in resp.json()["detail"]

    def test_patch_under_the_legacy_name_is_rejected(self, isolated_workspace, client):
        # An older client still addresses the orchestrator model as
        # slp-orchestra-mini; the path param is canonicalized so the same
        # always-on guard applies.
        resp = client.patch(
            "/v1/projects/p1/models/slp-orchestra-mini", json={"enabled": False}
        )

        assert resp.status_code == 400
        assert "always enabled" in resp.json()["detail"]

    def test_rejected_patch_writes_nothing(self, isolated_workspace, client):
        bootstrap = AppSettings().default_model
        client.patch(f"/v1/projects/p1/models/{bootstrap}", json={"enabled": False})

        for project_id in ("p1", "p2"):
            saved = _saved_project(isolated_workspace["config_file"], project_id)
            names = {m["model_name"] for m in saved.get("model_settings", [])}
            assert bootstrap not in names


class TestEnablingAModelIsProjectScoped:
    """Switching a model on for a project says nothing about the workspace.

    A model earns a place among the app-defaults by being *downloaded* — that
    is the act that adds it to the workspace, and ``/v1/init`` promotes it
    there (see ``test_init_promotes_a_downloaded_model.py``).  Turning on a
    model that is already present is one project's preference and must not
    reach ``config.json``.

    It could not anyway, which is how the old behaviour went unnoticed: under
    supervision this route is proxied to the child pinned to the project, and
    a child writes only its own ``project.json``.  The promotion attempted
    here was discarded at save, so a model downloaded from inside a project
    stayed disabled workspace-wide forever.
    """

    def test_enable_does_not_flip_the_workspace_default(self, isolated_workspace, client):
        assert client.patch(
            "/v1/projects/p1/models/qwen2-vl-2b", json={"enabled": True}
        ).status_code == 200

        ws = {m["model_name"]: m for m in client.get("/v1/models/workspace").json()}
        assert ws["qwen2-vl-2b"]["enabled"] is False

    def test_the_enable_still_lands_on_the_project(self, isolated_workspace, client):
        client.patch("/v1/projects/p1/models/qwen2-vl-2b", json={"enabled": True})

        p1 = {m["model_name"]: m for m in client.get("/v1/projects/p1/models").json()}
        assert p1["qwen2-vl-2b"]["enabled"] is True

        p1_saved = _saved_project(isolated_workspace["config_file"], "p1")
        entry = next(m for m in p1_saved["model_settings"] if m["model_name"] == "qwen2-vl-2b")
        assert entry["enabled"] is True

    def test_sibling_projects_are_untouched(self, isolated_workspace, client):
        client.patch("/v1/projects/p1/models/qwen2-vl-2b", json={"enabled": True})

        p2 = {m["model_name"]: m for m in client.get("/v1/projects/p2/models").json()}
        assert p2["qwen2-vl-2b"]["enabled"] is False

    def test_a_project_created_afterwards_does_not_inherit_it(self, isolated_workspace, client):
        """The workspace default never moved, so there is nothing to inherit."""
        from smartloop.api.services import get_model_service

        client.patch("/v1/projects/p1/models/qwen2-vl-2b", json={"enabled": True})

        # Each request reloads config.json, so the new project has to be saved.
        workspace = get_model_service()._ensure_workspace()
        workspace.projects.append(
            Project(id="p3", project_dir="/tmp/p3", current=False, name="p3")
        )
        workspace.save()

        p3 = {m["model_name"]: m for m in client.get("/v1/projects/p3/models").json()}
        assert p3["qwen2-vl-2b"]["enabled"] is False

    def test_disable_leaves_the_workspace_default_alone_too(self, isolated_workspace, client):
        """Neither direction reaches the workspace record from this route."""
        client.patch("/v1/projects/p1/models/gemma3-4b", json={"enabled": False})

        ws = {m["model_name"]: m for m in client.get("/v1/models/workspace").json()}
        assert ws["gemma3-4b"]["enabled"] is True


class TestWorkspaceDefaultEdit:
    def test_workspace_toggle_leaves_existing_projects_alone(self, isolated_workspace, client):
        """The Settings-level editor sets the default for future projects only.

        p1 and p2 have no entry for gemma3-4b and read through to the workspace
        default, so disabling it there would otherwise switch it off for both.
        """
        assert client.patch(
            "/v1/models/workspace/gemma3-4b", json={"enabled": False}
        ).status_code == 200

        for project_id in ("p1", "p2"):
            models = {m["model_name"]: m for m in client.get(f"/v1/projects/{project_id}/models").json()}
            assert models["gemma3-4b"]["enabled"] is True
