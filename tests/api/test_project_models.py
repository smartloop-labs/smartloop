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

        saved = json.loads(pathlib.Path(isolated_workspace["config_file"]).read_text())
        for project in saved["workspace"]["projects"]:
            names = {m["model_name"] for m in project.get("model_settings", [])}
            assert bootstrap not in names


class TestEnablePromotesWorkspaceDefault:
    """Enabling a model in a project makes it a default for future projects.

    Downloads are workspace-wide, so a model pulled for one project should be
    on offer to the ones created afterwards without a second download.  The
    projects that already exist are left alone: had the user wanted the model
    there, they would have enabled it and triggered the download themselves.
    """

    def test_enable_flips_the_workspace_default(self, isolated_workspace, client):
        assert client.patch(
            "/v1/projects/p1/models/qwen2-vl-2b", json={"enabled": True}
        ).status_code == 200

        ws = {m["model_name"]: m for m in client.get("/v1/models/workspace").json()}
        assert ws["qwen2-vl-2b"]["enabled"] is True

    def test_existing_projects_are_pinned_not_promoted(self, isolated_workspace, client):
        client.patch("/v1/projects/p1/models/qwen2-vl-2b", json={"enabled": True})

        p2 = {m["model_name"]: m for m in client.get("/v1/projects/p2/models").json()}
        assert p2["qwen2-vl-2b"]["enabled"] is False

        # p2 had no entry for the model before the enable; it must have gained an
        # explicit one, otherwise it would read through to the new default.
        saved = json.loads(pathlib.Path(isolated_workspace["config_file"]).read_text())
        p2_saved = next(p for p in saved["workspace"]["projects"] if p["id"] == "p2")
        entry = next(m for m in p2_saved["model_settings"] if m["model_name"] == "qwen2-vl-2b")
        assert entry["enabled"] is False

    def test_project_created_afterwards_inherits_the_model(self, isolated_workspace, client):
        from smartloop.api.services import get_model_service

        client.patch("/v1/projects/p1/models/qwen2-vl-2b", json={"enabled": True})

        # Each request reloads config.json, so the new project has to be saved.
        workspace = get_model_service()._ensure_workspace()
        workspace.projects.append(
            Project(id="p3", project_dir="/tmp/p3", current=False, name="p3")
        )
        workspace.save()

        p3 = {m["model_name"]: m for m in client.get("/v1/projects/p3/models").json()}
        assert p3["qwen2-vl-2b"]["enabled"] is True

    def test_disable_does_not_demote_the_default(self, isolated_workspace, client):
        """Promotion is one-way — one project opting out keeps the default."""
        client.patch("/v1/projects/p1/models/qwen2-vl-2b", json={"enabled": True})
        client.patch("/v1/projects/p1/models/qwen2-vl-2b", json={"enabled": False})

        ws = {m["model_name"]: m for m in client.get("/v1/models/workspace").json()}
        assert ws["qwen2-vl-2b"]["enabled"] is True


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


class TestProjectModelCapabilities:
    """Per-capability toggles on a project's model.

    ``capabilities`` is what the model supports (read-only, from the model
    class); ``enabled_capabilities`` is the subset switched on for this project.
    A client renders one control per supported capability and reads the enabled
    set to know which are on, so both have to be reported.
    """

    def test_all_capabilities_start_enabled(self, isolated_workspace, client):
        entry = next(
            m for m in client.get("/v1/projects/p1/models").json()
            if m["model_name"] == "gemma3-4b"
        )

        assert entry["capabilities"] == ["general", "dp"]
        assert entry["enabled_capabilities"] == ["general", "dp"]

    def test_patch_toggles_one_capability(self, isolated_workspace, client):
        resp = client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"dp": False}}
        )

        assert resp.status_code == 200
        # The supported set is unchanged — the client still needs it to offer a
        # way to switch dp back on.
        assert resp.json()["capabilities"] == ["general", "dp"]
        assert resp.json()["enabled_capabilities"] == ["general"]

    def test_the_list_reflects_the_toggle(self, isolated_workspace, client):
        client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"dp": False}}
        )

        entry = next(
            m for m in client.get("/v1/projects/p1/models").json()
            if m["model_name"] == "gemma3-4b"
        )
        assert entry["capabilities"] == ["general", "dp"]
        assert entry["enabled_capabilities"] == ["general"]

    def test_toggling_back_on_restores_it(self, isolated_workspace, client):
        client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"dp": False}}
        )

        resp = client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"dp": True}}
        )

        assert resp.status_code == 200
        assert resp.json()["enabled_capabilities"] == ["general", "dp"]

    def test_projects_are_independent(self, isolated_workspace, client):
        client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"dp": False}}
        )

        p2 = {m["model_name"]: m for m in client.get("/v1/projects/p2/models").json()}
        assert p2["gemma3-4b"]["enabled_capabilities"] == ["general", "dp"]

    def test_a_toggle_leaves_the_other_fields_alone(self, isolated_workspace, client):
        resp = client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"dp": False}}
        )

        assert resp.json()["enabled"] is True
        assert resp.json()["downloaded"] is True
        assert resp.json()["settings"] == {}

    def test_disabling_the_model_keeps_its_capability_state(self, isolated_workspace, client):
        """``enabled`` is the master switch; the toggles persist underneath it."""
        client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"dp": False}}
        )
        client.patch("/v1/projects/p1/models/gemma3-4b", json={"enabled": False})

        resp = client.patch("/v1/projects/p1/models/gemma3-4b", json={"enabled": True})
        assert resp.json()["enabled_capabilities"] == ["general"]

    def test_400_when_disabling_the_last_capability(self, isolated_workspace, client):
        client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"dp": False}}
        )

        resp = client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"general": False}}
        )

        assert resp.status_code == 400
        assert "last capability" in resp.json()["detail"]

    def test_400_when_disabling_the_only_capability_of_a_single_capability_model(
        self, isolated_workspace, client,
    ):
        resp = client.patch(
            "/v1/projects/p1/models/qwen2-vl-2b", json={"capabilities": {"vision": False}}
        )

        assert resp.status_code == 400
        assert "last capability" in resp.json()["detail"]

    def test_400_for_a_capability_the_model_does_not_support(self, isolated_workspace, client):
        resp = client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"vision": False}}
        )

        assert resp.status_code == 400
        assert "does not support" in resp.json()["detail"]

    def test_422_for_a_value_that_is_not_a_capability(self, isolated_workspace, client):
        """Typing the request key as the enum rejects it before the handler."""
        resp = client.patch(
            "/v1/projects/p1/models/gemma3-4b",
            json={"capabilities": {"not-a-capability": False}},
        )

        assert resp.status_code == 422

    def test_a_rejected_toggle_leaves_the_project_untouched(self, isolated_workspace, client):
        """A partly-applied map must not leak into the in-memory workspace."""
        resp = client.patch(
            "/v1/projects/p1/models/gemma3-4b",
            json={"capabilities": {"dp": False, "general": False}},
        )

        assert resp.status_code == 400
        entry = next(
            m for m in client.get("/v1/projects/p1/models").json()
            if m["model_name"] == "gemma3-4b"
        )
        assert entry["enabled_capabilities"] == ["general", "dp"]

    def test_toggle_persists_to_disk(self, isolated_workspace, client):
        client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"dp": False}}
        )

        from smartloop.workspace.workspace import Workspace as WS
        with patch.object(WS, "config_file", isolated_workspace["config_file"]):
            reloaded = WS.load()
        project = next(p for p in reloaded.projects if p.id == "p1")
        entry = next(m for m in project.model_settings if m.model_name == "gemma3-4b")
        assert entry.disabled_capabilities == ["dp"]

    def test_settings_cannot_reach_capability_state(self, isolated_workspace, client):
        """``settings`` is an unvalidated catch-all, so the deny-list is not in it.

        Were it stored there, this request would empty the model's capability
        set behind the guards above.
        """
        resp = client.patch(
            "/v1/projects/p1/models/gemma3-4b",
            json={"settings": {"disabled_capabilities": ["general", "dp"]}},
        )

        assert resp.status_code == 200
        assert resp.json()["enabled_capabilities"] == ["general", "dp"]

    def test_bootstrap_model_is_rejected(self, isolated_workspace, client):
        resp = client.patch(
            f"/v1/projects/p1/models/{AppSettings().default_model}",
            json={"capabilities": {"general": False}},
        )

        assert resp.status_code == 400
        assert "always enabled" in resp.json()["detail"]


class TestCapabilityToggleUnloadsTheModel:
    """Switching a capability off has to free the resident instance.

    The project gate is read when a model is *selected*, but ``get_vision_llm``
    and ``_get_large_llm`` short-circuit on their name-keyed cache before that
    sweep runs.  Without an unload here a model already in VRAM would keep
    serving the capability the user just switched off, which is the same reason
    the ``enabled`` flag unloads.
    """

    @pytest.fixture()
    def unload(self):
        from smartloop.api.services.model_service import ModelService
        with patch.object(ModelService, "unload_disabled_capability_slots") as m:
            yield m

    def test_disabling_a_capability_unloads(self, isolated_workspace, client, unload):
        resp = client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"dp": False}}
        )

        assert resp.status_code == 200
        unload.assert_called_once_with("gemma3-4b")

    def test_enabling_a_capability_does_not_unload(self, isolated_workspace, client, unload):
        """Turning one back on only widens the gate, so the instance stays valid."""
        client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"dp": False}}
        )
        unload.reset_mock()

        resp = client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"dp": True}}
        )

        assert resp.status_code == 200
        unload.assert_not_called()

    def test_a_no_op_patch_does_not_unload(self, isolated_workspace, client, unload):
        """Re-sending the state a capability is already in must not drop VRAM."""
        resp = client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"dp": True}}
        )

        assert resp.status_code == 200
        unload.assert_not_called()

    def test_a_rejected_patch_does_not_unload(self, isolated_workspace, client, unload):
        """The 400 leaves the deny-list untouched, so nothing should be freed."""
        client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"dp": False}}
        )
        unload.reset_mock()

        resp = client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"general": False}}
        )

        assert resp.status_code == 400
        unload.assert_not_called()

    def test_a_partially_rejected_map_does_not_unload(self, isolated_workspace, client, unload):
        """The whole map is rolled back on rejection, so it is not a shrink."""
        resp = client.patch(
            "/v1/projects/p1/models/gemma3-4b",
            json={"capabilities": {"dp": False, "general": False}},
        )

        assert resp.status_code == 400
        unload.assert_not_called()


class TestCatalogAgreesWithTheProjectView:
    """GET /v1/models?project_id= must resolve the same enabled set.

    The catalog view and the project view render the same controls, so a
    capability switched off in one has to read as off in the other.
    """

    @pytest.fixture(autouse=True)
    def catalog(self, platform_service):
        """The cloud catalog behind GET /v1/models (see the router docstring)."""
        platform_service.list_models.return_value = [
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it"},
            {"name": "qwen2-vl-2b", "model_id": "Qwen/Qwen2-VL-2B-Instruct"},
        ]

    def test_enabled_capabilities_follow_the_project(self, isolated_workspace, client):
        client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"dp": False}}
        )

        listed = {m["name"]: m for m in client.get("/v1/models?project_id=p1").json()}
        assert listed["gemma3-4b"]["capabilities"] == ["general", "dp"]
        assert listed["gemma3-4b"]["enabled_capabilities"] == ["general"]

    def test_without_a_project_the_full_set_is_reported(self, isolated_workspace, client):
        client.patch(
            "/v1/projects/p1/models/gemma3-4b", json={"capabilities": {"dp": False}}
        )

        # No project_id — nothing to resolve the toggles against, so the
        # workspace (app-default) view reports everything supported.
        listed = {m["name"]: m for m in client.get("/v1/models").json()}
        assert listed["gemma3-4b"]["enabled_capabilities"] == ["general", "dp"]

    def test_a_model_absent_from_the_workspace_reports_its_class_defaults(
        self, isolated_workspace, client, platform_service,
    ):
        """Nothing can have been toggled off for a model not registered here."""
        platform_service.list_models.return_value = [
            {"name": "gemma4-e4b", "model_id": "google/gemma-4-e4b-it"},
        ]

        listed = {m["name"]: m for m in client.get("/v1/models?project_id=p1").json()}
        assert listed["gemma4-e4b"]["enabled_capabilities"] == ["general", "dp", "vision"]
