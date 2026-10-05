//! Port of `tests/api/test_init_promotes_a_downloaded_model.py`.
//!
//! A model becomes a workspace app-default by arriving, not by being enabled.
//!
//! `/v1/init` is where a model enters the workspace.  When it is fetched for a
//! project the workspace record is registered *disabled* on purpose, so a
//! cancelled download strands nothing — which leaves something to flip once
//! the weights actually land.
//!
//! That flip used to be attached to the wrong event.  `update_project_model`
//! promoted on any enable, and under supervision that route is proxied to the
//! child pinned to the project — a process that writes only its own
//! `project.json`.  So the promotion was discarded and `ws.enabled` stayed
//! false forever: every model ever downloaded from inside a project was off
//! for projects created later, and shown disabled in the workspace list.
//!
//! Promoting here instead puts it in the one process that owns `config.json`
//! (`/v1/init` names no project in its path, so it is never proxied) and ties
//! it to the one event that means "this model is new to the workspace".
//! Switching on a model that was already here stays project-scoped — see
//! `test_enabling_a_model_is_project_scoped` in `test_project_models`.

use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::json;
use serial_test::serial;

use smartloop::api::services::ModelService;
use smartloop::model::model_manager::Model;
use smartloop::workspace::workspace::{SharedWorkspace, Workspace, WorkspaceModel};

use crate::common::api_core::{EnvGuard, ModelServiceOverride, NoRealDownloads, TestClient};

const MODEL: &str = "gemma3-4b";
const MODEL_ID: &str = "google/gemma-3-4b-it";
const PROJECT: &str = "p1";

/// `workspace`: a real workspace under a temp home, with no weights on disk.
///
/// Real rather than mocked because the assertions are about what `save()`
/// leaves in `config.json` — the workspace record is the whole subject.
///
/// Deliberately not the download guard's directory: `no_real_downloads` scans
/// that tree afterwards for stray `.gguf` files, and one test here plants a
/// fake one to mean "these weights are already on disk".  The guard cannot
/// tell that apart from a download that escaped the stub, and it should not
/// have to — so the fixture keeps its files where the guard is not looking.
struct WorkspaceFixture {
    _home: tempfile::TempDir,
    _env: EnvGuard,
    workspace: SharedWorkspace,
}

impl WorkspaceFixture {
    fn new() -> Self {
        let home = tempfile::tempdir().unwrap();
        let mut env = EnvGuard::new();
        env.set("SLP_HOME", &home.path().to_string_lossy());
        Workspace::reset();
        let mut ws = Workspace::new("ws-test");
        ws.workspace_dir = home.path().join("ws-test").to_string_lossy().into_owned();
        ws.models_dir = home.path().join("ws-test").join("models").to_string_lossy().into_owned();
        WorkspaceFixture { _home: home, _env: env, workspace: Arc::new(RwLock::new(ws)) }
    }
}

impl Drop for WorkspaceFixture {
    fn drop(&mut self) {
        if std::thread::panicking() {
            let _ = std::panic::catch_unwind(Workspace::reset);
        } else {
            Workspace::reset();
        }
    }
}

/// `model_service`: a service whose model pipeline holds the fixture's
/// workspace — the same instance the router saves, so `current_project`
/// resolves against it — bound as the dependency.  The catalog seed the route
/// fires is answered by the client's platform fake (an empty catalog), so it
/// is off the network.
fn model_service(workspace: &WorkspaceFixture) -> ModelServiceOverride {
    let svc = ModelService::new_bare();
    *svc.model.write() = Some(Arc::new(Model::new("ws-test", Arc::clone(&workspace.workspace), None)));
    ModelServiceOverride::install(svc)
}

/// A downloader that resolves and "fetches" without touching the network.
fn stub_downloader(downloads: &NoRealDownloads) {
    downloads.resolve_artifacts_with(&[("model.gguf", "https://signed/url")]);
}

fn register(workspace: &SharedWorkspace, enabled: bool, downloaded: bool) -> WorkspaceModel {
    let mut ws = workspace.write();
    ws.add_model(MODEL, MODEL_ID, None, enabled);
    let models_dir = ws.models_dir.clone();
    let ws_model = ws.get_model_mut(Some(MODEL)).expect("registered");
    ws_model.downloaded = downloaded;
    if ws_model.models_dir.is_empty() {
        ws_model.models_dir = models_dir;
    }
    ws_model.clone()
}

/// Make `gguf_exists` true, which is what "already available" means.
fn put_weights_on_disk(ws_model: &WorkspaceModel) {
    let model_dir = ws_model.model_dir();
    assert!(!model_dir.is_empty(), "the model has no directory");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::File::create(Path::new(&model_dir).join("model.gguf")).unwrap();
}

fn enabled(workspace: &SharedWorkspace) -> bool {
    workspace.read().get_model(Some(MODEL)).expect("registered").enabled
}

mod test_a_download_promotes_the_model {
    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_a_project_scoped_download_flips_the_workspace_default() {
        let client = TestClient::new();
        let workspace = WorkspaceFixture::new();
        let _svc = model_service(&workspace);
        stub_downloader(&client.downloads);

        let resp = client.post("/v1/init", json!({"model_name": MODEL, "project_id": PROJECT})).await;

        assert_eq!(resp.status_code, 200);
        // Registered disabled by the route, then promoted once the weights landed.
        assert!(enabled(&workspace.workspace));
    }

    /// The point of doing it here: this process may write `config.json`.
    #[tokio::test]
    #[serial]
    async fn test_it_survives_the_reload_the_next_request_does() {
        let client = TestClient::new();
        let workspace = WorkspaceFixture::new();
        let _svc = model_service(&workspace);
        stub_downloader(&client.downloads);

        client.post("/v1/init", json!({"model_name": MODEL, "project_id": PROJECT})).await;

        let reloaded = Workspace::load().expect("a saved workspace");
        assert!(reloaded.get_model(Some(MODEL)).expect("registered").enabled);
    }

    /// No project named (CLI/TUI): registered enabled, promote is a no-op.
    #[tokio::test]
    #[serial]
    async fn test_a_workspace_scoped_download_is_still_enabled() {
        let client = TestClient::new();
        let workspace = WorkspaceFixture::new();
        let _svc = model_service(&workspace);
        stub_downloader(&client.downloads);

        client.post("/v1/init", json!({"model_name": MODEL})).await;

        assert!(enabled(&workspace.workspace));
    }
}

mod test_nothing_else_promotes {
    use super::*;

    /// No download ran, so nothing says this model is new to the workspace.
    ///
    /// This is the case the rule excludes: the user is switching on something
    /// that was already here, which is their project's business and not the
    /// default for every project made afterwards.
    #[tokio::test]
    #[serial]
    async fn test_an_already_available_model_is_not_promoted() {
        let client = TestClient::new();
        let workspace = WorkspaceFixture::new();
        let _svc = model_service(&workspace);
        let ws_model = register(&workspace.workspace, false, true);
        put_weights_on_disk(&ws_model);

        let resp = client.post("/v1/init", json!({"model_name": MODEL, "project_id": PROJECT})).await;

        assert_eq!(resp.status_code, 200);
        assert!(!enabled(&workspace.workspace));
    }

    /// Re-downloading must not re-pin every project that has no entry.
    #[tokio::test]
    #[serial]
    async fn test_an_already_enabled_model_is_left_alone() {
        let client = TestClient::new();
        let workspace = WorkspaceFixture::new();
        let _svc = model_service(&workspace);
        register(&workspace.workspace, true, false);
        stub_downloader(&client.downloads);

        client.post("/v1/init", json!({"model_name": MODEL, "project_id": PROJECT})).await;

        assert!(enabled(&workspace.workspace));
    }
}
