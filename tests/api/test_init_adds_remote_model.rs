//! `/v1/init` for a remote model registers it and completes at once.
//!
//! The studio adds a cloud model through the same action that downloads a
//! local one.  There is nothing to fetch, so the stream is just the ready and
//! complete events, and entitlement comes from the catalog's `is_accessible`.

use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::json;
use serial_test::serial;

use smartloop::api::services::ModelService;
use smartloop::model::model_manager::Model;
use smartloop::workspace::workspace::{SharedWorkspace, Workspace};

use crate::common::api_core::{EnvGuard, ModelServiceOverride, TestClient};

const MODEL: &str = "kimi-k2";
const PROJECT: &str = "p1";

struct WorkspaceFixture {
    _home: tempfile::TempDir,
    _env: EnvGuard,
    workspace: SharedWorkspace,
    _svc: ModelServiceOverride,
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
        let workspace: SharedWorkspace = Arc::new(RwLock::new(ws));
        let svc = ModelService::new_bare();
        *svc.model.write() = Some(Arc::new(Model::new("ws-test", Arc::clone(&workspace), None)));
        WorkspaceFixture { _home: home, _env: env, workspace, _svc: ModelServiceOverride::install(svc) }
    }

    fn signed_in(&mut self) {
        self._env.set("SLP_DEVELOPER_TOKEN", "dev-token");
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

fn catalog(client: &TestClient, access_level: &str, is_accessible: bool) {
    client.platform().set_list_models(json!([{
        "name": MODEL, "model_id": null, "type": "remote",
        "url": "https://edge.smartloop.ai",
        "access_level": access_level, "is_accessible": is_accessible,
        "capabilities": ["general", "coding"], "context_length": 131072,
    }]));
}

#[tokio::test]
#[serial]
async fn test_a_remote_model_is_registered_without_a_download() {
    let client = TestClient::new();
    let fixture = WorkspaceFixture::new();
    catalog(&client, "free_login", true);

    let resp = client.post("/v1/init", json!({"model_name": MODEL, "project_id": PROJECT})).await;

    assert_eq!(resp.status_code, 200);
    let body = resp.text();
    assert!(body.contains("\"model_ready\""), "{body}");
    assert!(body.contains("\"completed\""), "{body}");

    let ws = fixture.workspace.read();
    let model = ws.get_model(Some(MODEL)).expect("registered");
    assert!(model.is_remote());
    assert!(model.downloaded);
    assert!(!model.is_active);
    assert_eq!(model.url().as_deref(), Some("https://edge.smartloop.ai"));
    // "general" and "coding" in the catalog row each read as `text`, once.
    assert_eq!(model.effective_capabilities(), ["text"]);
    assert_eq!(model.context_length(), Some(131072));
    // Project-scoped: left for the follow-up PATCH, and never promoted.
    assert!(!model.enabled);
}

#[tokio::test]
#[serial]
async fn test_a_project_less_add_is_enabled_workspace_wide() {
    let client = TestClient::new();
    let fixture = WorkspaceFixture::new();
    catalog(&client, "free", true);

    let resp = client.post("/v1/init", json!({"model_name": MODEL})).await;

    assert_eq!(resp.status_code, 200);
    assert!(fixture.workspace.read().get_model(Some(MODEL)).expect("registered").enabled);
}

#[tokio::test]
#[serial]
async fn test_a_signed_out_user_is_asked_to_sign_in() {
    let client = TestClient::new();
    let fixture = WorkspaceFixture::new();
    catalog(&client, "free_login", false);

    let resp = client.post("/v1/init", json!({"model_name": MODEL})).await;

    assert_eq!(resp.status_code, 401);
    assert!(fixture.workspace.read().get_model(Some(MODEL)).is_none());
}

#[tokio::test]
#[serial]
async fn test_a_signed_in_user_outside_the_plan_is_refused() {
    let client = TestClient::new();
    let mut fixture = WorkspaceFixture::new();
    fixture.signed_in();
    catalog(&client, "pro", false);

    let resp = client.post("/v1/init", json!({"model_name": MODEL})).await;

    assert_eq!(resp.status_code, 403);
    assert!(fixture.workspace.read().get_model(Some(MODEL)).is_none());
}

#[tokio::test]
#[serial]
async fn test_a_model_missing_from_the_catalog_is_still_unknown() {
    let client = TestClient::new();
    let _fixture = WorkspaceFixture::new();
    client.platform().set_list_models(json!([]));

    let resp = client.post("/v1/init", json!({"model_name": "no-such-model"})).await;

    assert_eq!(resp.status_code, 400);
}
