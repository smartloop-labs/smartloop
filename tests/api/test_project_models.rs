//! Port of `tests/api/test_project_models.py`.
//!
//! Tests for the per-project model endpoints (`/v1/projects/{id}/models`).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use serde_json::{Value, json};
use serial_test::serial;

use smartloop::api::services::get_model_service;
use smartloop::config::AppSettings;
use smartloop::workspace::context::{delegate_projects, pin_project};
use smartloop::workspace::workspace::{
    Project, ProjectModel, Workspace, WorkspaceModel, set_settings_home_dir,
};

use crate::common::workspace_api::{NoRealDownloads, PlatformServiceOverride, TestClient};
use crate::shared::workspace_support::EnvGuard;

fn s(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn default_model() -> String {
    AppSettings::new().default_model
}

/// The `isolated_workspace` fixture: writes a config.json to a temp dir and
/// points the app's workspace at it, for the fixture's lifetime.
///
/// Both projects share the workspace (app-default) model registry; their
/// per-project settings start empty so effective state falls back to
/// defaults.  The always-on bootstrap model is registered too, so the list
/// assertions below also cover its exclusion.
struct IsolatedWorkspace {
    #[allow(dead_code)]
    workspace: Workspace,
    config_file: String,
    // Drop order: the overrides go first, then the env, then the temp dir.
    _client_override: PlatformServiceOverride,
    _downloads: NoRealDownloads,
    _env: Vec<EnvGuard>,
    _tmp: tempfile::TempDir,
}

impl Drop for IsolatedWorkspace {
    fn drop(&mut self) {
        Workspace::set_config_file(None);
        set_settings_home_dir(None);
        Workspace::reset();
        pin_project(None);
        delegate_projects(false);
    }
}

fn isolated_workspace() -> IsolatedWorkspace {
    pin_project(None);
    delegate_projects(false);
    let tmp = tempfile::tempdir().unwrap();
    let downloads = NoRealDownloads::install(tmp.path());
    let home_dir = tmp.path().join("dot_smartloop");
    fs::create_dir(&home_dir).unwrap();

    let ws_id = "ws-test";
    let workspace = Workspace::new(ws_id)
        .with_workspace_dir(s(&home_dir.join(ws_id)))
        .with_models_dir(s(&home_dir.join(ws_id).join("models")))
        .with_models(vec![
            WorkspaceModel {
                model_name: default_model(),
                model_id: "smartloop/slp-orchestra-mini".to_string(),
                downloaded: true,
                enabled: true,
                ..Default::default()
            },
            WorkspaceModel {
                model_name: "gemma3-4b".to_string(),
                model_id: "google/gemma-3-4b-it".to_string(),
                downloaded: true,
                enabled: true,
                ..Default::default()
            },
            WorkspaceModel {
                model_name: "qwen2-vl-2b".to_string(),
                model_id: "Qwen/Qwen2-VL-2B-Instruct".to_string(),
                downloaded: true,
                enabled: false,
                is_active: true,
                ..Default::default()
            },
        ])
        .with_projects(vec![
            Project {
                id: "p1".to_string(),
                project_dir: "/tmp/p1".to_string(),
                current: true,
                name: Some("p1".to_string()),
                ..Default::default()
            },
            Project {
                id: "p2".to_string(),
                project_dir: "/tmp/p2".to_string(),
                current: false,
                name: Some("p2".to_string()),
                ..Default::default()
            },
        ]);

    let config_file = home_dir.join("config.json");
    let ws_data = serde_json::to_value(&workspace).unwrap();
    fs::write(&config_file, serde_json::to_string(&json!({"id": "gemma-3-4b-it", "workspace": ws_data})).unwrap())
        .unwrap();

    // `_settings.slp_home = home_dir`, with nothing in the environment
    // overriding it, and `Workspace.config_file = config_file`.
    let env = vec![
        EnvGuard::set("SLP_HOME", &home_dir),
        EnvGuard::remove("SLP_CONFIG_FILE"),
        EnvGuard::remove("SLP_LEGACY_CONFIG"),
    ];
    set_settings_home_dir(Some(s(&home_dir)));
    Workspace::set_config_file(Some(s(&config_file)));
    Workspace::reset();

    // A service with no model pipeline, so `ensure_workspace()` re-reads the
    // isolated config on every request (`services._model_service = None`).
    *get_model_service().model.write() = None;

    IsolatedWorkspace {
        workspace,
        config_file: s(&config_file),
        _client_override: PlatformServiceOverride::install(),
        _downloads: downloads,
        _env: env,
        _tmp: tmp,
    }
}

/// A project's persisted record, read back from its own file.
///
/// Per-project state no longer lives in `config.json` — that keeps a thin
/// index, and each project owns a `project.json` in its own directory.  The
/// index is what says where, so this walks the same path a load does.
fn saved_project(config_file: &str, project_id: &str) -> Value {
    let config: Value = serde_json::from_str(&fs::read_to_string(config_file).unwrap()).unwrap();
    let entry = config["workspace"]["projects"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["id"] == project_id)
        .cloned()
        .unwrap();
    let state = Path::new(entry["project_dir"].as_str().unwrap()).join("project.json");
    if !state.is_file() {
        // Nothing has saved this project yet — the index entry is all there
        // is, which is itself the answer for a test asserting nothing was
        // written.
        return entry;
    }
    let saved: Value = serde_json::from_str(&fs::read_to_string(state).unwrap()).unwrap();
    saved["project"].clone()
}

/// `{m["model_name"]: m for m in resp.json()}`.
fn by_name(body: &Value) -> HashMap<String, Value> {
    body.as_array()
        .expect("a list of models")
        .iter()
        .map(|m| (m["model_name"].as_str().unwrap().to_string(), m.clone()))
        .collect()
}

async fn project_models(client: &TestClient, project_id: &str) -> HashMap<String, Value> {
    by_name(client.get(&format!("/v1/projects/{project_id}/models")).await.json())
}

mod test_list_project_models {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_lists_models_with_workspace_defaults() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        let resp = client.get("/v1/projects/p1/models").await;

        assert_eq!(resp.status_code(), 200);
        let data = by_name(resp.json());
        let names: HashSet<&str> = data.keys().map(String::as_str).collect();
        assert_eq!(names, HashSet::from(["gemma3-4b", "qwen2-vl-2b"]));
        assert_eq!(data["gemma3-4b"]["enabled"], true);
        assert_eq!(data["gemma3-4b"]["downloaded"], true);
        assert_eq!(data["qwen2-vl-2b"]["enabled"], false);
        assert_eq!(data["qwen2-vl-2b"]["downloaded"], true);
        assert_eq!(data["qwen2-vl-2b"]["is_active"], true);
        assert_eq!(data["gemma3-4b"]["type"], "local");
        assert_eq!(data["gemma3-4b"]["url"], Value::Null);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_404_for_unknown_project() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        let resp = client.get("/v1/projects/does-not-exist/models").await;
        assert_eq!(resp.status_code(), 404);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_projects_are_independent() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        assert_eq!(
            client.patch("/v1/projects/p1/models/qwen2-vl-2b", json!({"enabled": true})).await.status_code(),
            200
        );

        let p1 = project_models(&client, "p1").await;
        let p2 = project_models(&client, "p2").await;

        assert_eq!(p1["qwen2-vl-2b"]["enabled"], true);
        // p2 was never touched — still the workspace default.
        assert_eq!(p2["qwen2-vl-2b"]["enabled"], false);
    }
}

mod test_update_project_model {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_updates_enabled_state_per_project() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        let resp = client.patch("/v1/projects/p1/models/gemma3-4b", json!({"enabled": false})).await;

        assert_eq!(resp.status_code(), 200);
        assert_eq!(resp.json()["enabled"], false);
        assert_eq!(resp.json()["model_name"], "gemma3-4b");
        assert_ne!(resp.json()["capabilities"], json!([]));
    }

    /// A client cannot assert download state — it describes the disk.
    ///
    /// The field is not on the request model, so it is ignored rather than
    /// written, and the response keeps reporting the workspace-wide value.
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_downloaded_is_read_only() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        let resp = client.patch("/v1/projects/p1/models/qwen2-vl-2b", json!({"downloaded": false})).await;

        assert_eq!(resp.status_code(), 200);
        assert_eq!(resp.json()["downloaded"], true);
        // Nothing per-project was written for it.
        let entry = project_models(&client, "p1").await["qwen2-vl-2b"].clone();
        assert_eq!(entry["downloaded"], true);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_active_flag_deactivates_other_project_models() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        let resp = client.patch("/v1/projects/p1/models/gemma3-4b", json!({"is_active": true})).await;
        assert_eq!(resp.status_code(), 200);
        assert_eq!(resp.json()["is_active"], true);

        let p1 = project_models(&client, "p1").await;
        assert_eq!(p1["gemma3-4b"]["is_active"], true);
        assert_eq!(p1["qwen2-vl-2b"]["is_active"], false);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_settings_update_merges_into_project_entry() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        let resp = client
            .patch("/v1/projects/p1/models/gemma3-4b", json!({"settings": {"temperature": 0.3}}))
            .await;
        assert_eq!(resp.status_code(), 200);
        assert_eq!(resp.json()["settings"], json!({"temperature": 0.3}));

        let listed = project_models(&client, "p1").await;
        assert_eq!(listed["gemma3-4b"]["settings"], json!({"temperature": 0.3}));
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_404_for_unknown_model() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        let resp = client.patch("/v1/projects/p1/models/not-a-model", json!({"enabled": false})).await;
        assert_eq!(resp.status_code(), 404);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_404_for_unknown_project() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        let resp = client.patch("/v1/projects/nope/models/gemma3-4b", json!({"enabled": false})).await;
        assert_eq!(resp.status_code(), 404);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_patch_persists_to_disk() {
        let ws = isolated_workspace();
        let client = TestClient::new();

        client.patch("/v1/projects/p1/models/gemma3-4b", json!({"enabled": false})).await;

        Workspace::set_config_file(Some(ws.config_file.clone()));
        let reloaded = Workspace::load().expect("a workspace on disk");
        let project = reloaded.projects.iter().find(|p| p.id == "p1").unwrap();
        let entry = project.model_settings.iter().find(|m| m.model_name == "gemma3-4b").unwrap();
        assert!(!entry.enabled);
    }
}

mod test_mark_model_downloaded {
    use super::*;

    use std::sync::Arc;

    use parking_lot::RwLock;

    /// One flag, workspace-wide — every project sees it at once.
    #[test]
    #[serial]
    fn test_marks_the_workspace_and_writes_no_project_state() {
        let isolated = isolated_workspace();

        Workspace::set_config_file(Some(isolated.config_file.clone()));
        let mut ws = Workspace::load().expect("a workspace on disk");
        ws.get_model_mut(Some("gemma3-4b")).unwrap().downloaded = false;
        // Give both projects seeded entries, so a stray per-project write would show.
        let ids: Vec<String> = ws.projects.iter().map(|p| p.id.clone()).collect();
        for id in &ids {
            ws.sync_project_models(id);
        }
        let ws = Arc::new(RwLock::new(ws));

        let service = get_model_service();
        // `patch.object(service, "_ensure_workspace", return_value=ws)`.
        service.mark_model_downloaded("gemma3-4b", Some(Arc::clone(&ws))).unwrap();
        ws.read().save().unwrap();

        Workspace::set_config_file(Some(isolated.config_file.clone()));
        let reloaded = Workspace::load().expect("a workspace on disk");
        assert!(reloaded.get_model(Some("gemma3-4b")).unwrap().downloaded);
        // Visible from every project, with no per-project copy to drift.
        for project in &reloaded.projects {
            assert!(reloaded.effective_model_downloaded("gemma3-4b"));
            assert!(project.model_settings.iter().any(|m| m.model_name == "gemma3-4b"));
        }
        // `"downloaded" not in ProjectModel.model_fields`.
        let fields = serde_json::to_value(ProjectModel::new("gemma3-4b")).unwrap();
        assert!(fields.get("downloaded").is_none());
    }
}

/// The orchestrator model is always enabled and hidden from the list.
///
/// It has no per-project entry, so PATCHing it by name is rejected rather
/// than silently writing state that would shadow its always-on default.
mod test_bootstrap_model_is_not_configurable_per_project {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_it_is_absent_from_the_list() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        let resp = client.get("/v1/projects/p1/models").await;

        assert_eq!(resp.status_code(), 200);
        assert!(!by_name(resp.json()).contains_key(&default_model()));
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_patch_is_rejected() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        let resp = client
            .patch(&format!("/v1/projects/p1/models/{}", default_model()), json!({"enabled": false}))
            .await;

        assert_eq!(resp.status_code(), 400);
        assert!(resp.json()["detail"].as_str().unwrap().contains("always enabled"));
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_patch_under_the_legacy_name_is_rejected() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        // An older client still addresses the orchestrator model as
        // slp-orchestra-mini; the path param is canonicalized so the same
        // always-on guard applies.
        let resp = client.patch("/v1/projects/p1/models/slp-orchestra-mini", json!({"enabled": false})).await;

        assert_eq!(resp.status_code(), 400);
        assert!(resp.json()["detail"].as_str().unwrap().contains("always enabled"));
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_rejected_patch_writes_nothing() {
        let ws = isolated_workspace();
        let client = TestClient::new();
        let bootstrap = default_model();
        client.patch(&format!("/v1/projects/p1/models/{bootstrap}"), json!({"enabled": false})).await;

        for project_id in ["p1", "p2"] {
            let saved = saved_project(&ws.config_file, project_id);
            let names: HashSet<String> = saved
                .get("model_settings")
                .and_then(Value::as_array)
                .map(|ms| ms.iter().map(|m| m["model_name"].as_str().unwrap().to_string()).collect())
                .unwrap_or_default();
            assert!(!names.contains(&bootstrap));
        }
    }
}

/// Switching a model on for a project says nothing about the workspace.
///
/// A model earns a place among the app-defaults by being *downloaded* — that
/// is the act that adds it to the workspace, and `/v1/init` promotes it there
/// (see `test_init_promotes_a_downloaded_model`).  Turning on a model that is
/// already present is one project's preference and must not reach
/// `config.json`.
///
/// It could not anyway, which is how the old behaviour went unnoticed: under
/// supervision this route is proxied to the child pinned to the project, and a
/// child writes only its own `project.json`.  The promotion attempted here was
/// discarded at save, so a model downloaded from inside a project stayed
/// disabled workspace-wide forever.
mod test_enabling_a_model_is_project_scoped {
    use super::*;

    async fn workspace_models(client: &TestClient) -> HashMap<String, Value> {
        by_name(client.get("/v1/models/workspace").await.json())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_enable_does_not_flip_the_workspace_default() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        assert_eq!(
            client.patch("/v1/projects/p1/models/qwen2-vl-2b", json!({"enabled": true})).await.status_code(),
            200
        );

        let ws = workspace_models(&client).await;
        assert_eq!(ws["qwen2-vl-2b"]["enabled"], false);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_the_enable_still_lands_on_the_project() {
        let ws = isolated_workspace();
        let client = TestClient::new();

        client.patch("/v1/projects/p1/models/qwen2-vl-2b", json!({"enabled": true})).await;

        let p1 = project_models(&client, "p1").await;
        assert_eq!(p1["qwen2-vl-2b"]["enabled"], true);

        let p1_saved = saved_project(&ws.config_file, "p1");
        let entry = p1_saved["model_settings"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["model_name"] == "qwen2-vl-2b")
            .cloned()
            .unwrap();
        assert_eq!(entry["enabled"], true);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_sibling_projects_are_untouched() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        client.patch("/v1/projects/p1/models/qwen2-vl-2b", json!({"enabled": true})).await;

        let p2 = project_models(&client, "p2").await;
        assert_eq!(p2["qwen2-vl-2b"]["enabled"], false);
    }

    /// The workspace default never moved, so there is nothing to inherit.
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_a_project_created_afterwards_does_not_inherit_it() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        client.patch("/v1/projects/p1/models/qwen2-vl-2b", json!({"enabled": true})).await;

        // Each request reloads config.json, so the new project has to be saved.
        let workspace = get_model_service().ensure_workspace(None).unwrap();
        {
            let mut workspace = workspace.write();
            workspace.projects.push(Project {
                id: "p3".to_string(),
                project_dir: "/tmp/p3".to_string(),
                current: false,
                name: Some("p3".to_string()),
                ..Default::default()
            });
            workspace.save().unwrap();
        }

        let p3 = project_models(&client, "p3").await;
        assert_eq!(p3["qwen2-vl-2b"]["enabled"], false);
    }

    /// Neither direction reaches the workspace record from this route.
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_disable_leaves_the_workspace_default_alone_too() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        client.patch("/v1/projects/p1/models/gemma3-4b", json!({"enabled": false})).await;

        let ws = workspace_models(&client).await;
        assert_eq!(ws["gemma3-4b"]["enabled"], true);
    }
}

mod test_workspace_default_edit {
    use super::*;

    /// The Settings-level editor sets the default for future projects only.
    ///
    /// p1 and p2 have no entry for gemma3-4b and read through to the
    /// workspace default, so disabling it there would otherwise switch it off
    /// for both.
    #[tokio::test(flavor = "multi_thread")]
    #[serial]
    async fn test_workspace_toggle_leaves_existing_projects_alone() {
        let _ws = isolated_workspace();
        let client = TestClient::new();

        assert_eq!(
            client.patch("/v1/models/workspace/gemma3-4b", json!({"enabled": false})).await.status_code(),
            200
        );

        for project_id in ["p1", "p2"] {
            let models = project_models(&client, project_id).await;
            assert_eq!(models["gemma3-4b"]["enabled"], true);
        }
    }
}
