//! Port of `tests/api/test_models.py`.
//!
//! Tests for the /v1/models API router.

use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::{json, Value};
use serial_test::serial;

use smartloop::api::services::ModelService;
use smartloop::model::model_manager::Model;
use smartloop::platform::PlatformError;
use smartloop::workspace::workspace::{SharedWorkspace, Workspace, WorkspaceModel};

use crate::common::api_core::{ModelServiceOverride, TestClient};

fn ws_model(model_name: &str, downloaded: bool, enabled: bool) -> WorkspaceModel {
    WorkspaceModel {
        model_name: model_name.to_string(),
        model_id: format!("test/{model_name}"),
        downloaded,
        enabled,
        ..Default::default()
    }
}

/// `workspace_models`: override the model service so /v1/models sees a known
/// workspace.
///
/// The test appends to the workspace's `models`; the override returns the same
/// workspace every call, so appends made after the client is created still
/// take effect.  This is a real `Workspace` rather than a stub because the
/// router resolves `enabled` through `effective_model_enabled` (project
/// override -> workspace default).
struct WorkspaceModels {
    workspace: SharedWorkspace,
    _override: ModelServiceOverride,
}

impl WorkspaceModels {
    fn new() -> Self {
        let mut workspace = Workspace::new("ws-test");
        workspace.workspace_dir = "/tmp/ws-test".to_string();
        workspace.models_dir = "/tmp/ws-test/models".to_string();
        let workspace: SharedWorkspace = Arc::new(RwLock::new(workspace));
        // `svc._ensure_workspace.return_value = workspace`: a service whose
        // model pipeline holds this workspace answers it.
        let svc = ModelService::new_bare();
        *svc.model.write() = Some(Arc::new(Model::new("ws-test", Arc::clone(&workspace), None)));
        WorkspaceModels { workspace, _override: ModelServiceOverride::install(svc) }
    }

    fn append(&self, model: WorkspaceModel) {
        self.workspace.write().models.push(model);
    }
}

fn names(body: &Value) -> Vec<String> {
    body.as_array()
        .expect("a list")
        .iter()
        .map(|m| m["name"].as_str().expect("a name").to_string())
        .collect()
}

mod test_list_models {
    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_returns_model_list() {
        let client = TestClient::new();
        // sl-mini is the bootstrap (framework core) model; /v1/models lists the
        // additional models a user can add, so the bootstrap model is filtered
        // out of the result.
        client.platform().set_list_models(json!([
            {"name": "sl-mini", "model_id": "smartloop-ai/slp-orchestra-mini"},
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it"},
        ]));
        let resp = client.get("/v1/models").await;

        assert_eq!(resp.status_code, 200);
        let names = names(&resp.json());
        assert!(!names.contains(&"sl-mini".to_string())); // bootstrap model excluded
        assert_eq!(names, vec!["gemma3-4b"]);
    }

    #[tokio::test]
    #[serial]
    async fn test_bootstrap_model_excluded_under_its_legacy_name() {
        let client = TestClient::new();
        // The cloud catalog still publishes the pre-rename name; it must be
        // canonicalized so the bootstrap filter still catches it.
        client.platform().set_list_models(json!([
            {"name": "slp-orchestra-mini", "model_id": "smartloop-ai/slp-orchestra-mini"},
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it"},
        ]));
        let resp = client.get("/v1/models").await;

        assert_eq!(resp.status_code, 200);
        assert_eq!(names(&resp.json()), vec!["gemma3-4b"]);
    }

    #[tokio::test]
    #[serial]
    async fn test_lists_paid_models_outside_plan() {
        let client = TestClient::new();
        // The picker shows what an upgrade would buy, so a Pro model the plan
        // doesn't grant is listed with is_accessible=False rather than hidden.
        // Access is enforced at download time (403), not here.
        client.platform().set_list_models(json!([
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it",
             "access_level": "free", "is_accessible": true},
            {"name": "gemma3-12b", "model_id": "google/gemma-3-12b-it",
             "access_level": "pro", "is_accessible": false},
        ]));
        let resp = client.get("/v1/models").await;

        assert_eq!(resp.status_code, 200);
        let body = resp.json();
        let mut got = names(&body);
        got.sort();
        assert_eq!(got, vec!["gemma3-12b", "gemma3-4b"]);
        let pro = body.as_array().unwrap().iter().find(|m| m["name"] == "gemma3-12b").unwrap();
        assert_eq!(pro["access_level"], "pro");
        assert_eq!(pro["is_accessible"], false);
    }

    #[tokio::test]
    #[serial]
    async fn test_lists_internal_models() {
        let client = TestClient::new();
        // The cloud filters INTERNAL out for everyone but superusers, so a row
        // that reaches us is one the caller is entitled to see. It must survive
        // validation — a local enum missing the member would 500 the list.
        client.platform().set_list_models(json!([
            {"name": "gemma3-12b", "model_id": "google/gemma-3-12b-it",
             "access_level": "internal", "is_accessible": true},
        ]));
        let resp = client.get("/v1/models").await;

        assert_eq!(resp.status_code, 200);
        let levels: Vec<Value> = resp.json().as_array().unwrap().iter().map(|m| m["access_level"].clone()).collect();
        assert_eq!(levels, vec![json!("internal")]);
    }

    #[tokio::test]
    #[serial]
    async fn test_an_unknown_access_level_costs_the_level_not_the_model() {
        let client = TestClient::new();
        // An access level added cloud-side ahead of a release drops to null and
        // the model is still listed — losing the tier beats losing the entry.
        client.platform().set_list_models(json!([
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it",
             "access_level": "free", "is_accessible": true},
            {"name": "gemma3-27b", "model_id": "google/gemma-3-27b-it",
             "access_level": "enterprise", "is_accessible": false},
        ]));
        let resp = client.get("/v1/models").await;

        assert_eq!(resp.status_code, 200);
        let body = resp.json();
        let mut got = names(&body);
        got.sort();
        assert_eq!(got, vec!["gemma3-27b", "gemma3-4b"]);
        let unknown = body.as_array().unwrap().iter().find(|m| m["name"] == "gemma3-27b").unwrap();
        assert_eq!(unknown["access_level"], Value::Null);
    }

    #[tokio::test]
    #[serial]
    async fn test_skips_unparseable_entry_without_dropping_the_rest() {
        let client = TestClient::new();
        // A row the client cannot represent at all -- here a required field the
        // cloud stopped sending -- costs that row, not the whole catalog.
        client.platform().set_list_models(json!([
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it",
             "access_level": "free", "is_accessible": true},
            {"name": "gemma3-27b", "access_level": "pro", "is_accessible": false},
        ]));
        let resp = client.get("/v1/models").await;

        assert_eq!(resp.status_code, 200);
        assert_eq!(names(&resp.json()), vec!["gemma3-4b"]);
    }

    #[tokio::test]
    #[serial]
    async fn test_lists_everything_when_not_logged_in() {
        let client = TestClient::new();
        // What the cloud actually sends an anonymous caller: the free tier is
        // accessible (it downloads with no token) and pro is not.  The pro
        // model must still be listed, because the picker shows what an upgrade
        // would buy — so is_accessible describes a model, it does not decide
        // whether one appears.
        client.platform().set_list_models(json!([
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it",
             "access_level": "free", "is_accessible": true},
            {"name": "gemma3-12b", "model_id": "google/gemma-3-12b-it",
             "access_level": "pro", "is_accessible": false},
        ]));
        let resp = client.get("/v1/models").await;

        assert_eq!(resp.status_code, 200);
        let mut got = names(&resp.json());
        got.sort();
        assert_eq!(got, vec!["gemma3-12b", "gemma3-4b"]);
    }

    #[tokio::test]
    #[serial]
    async fn test_keeps_unclassified_models() {
        let client = TestClient::new();
        // No access_level from the cloud — treat as free, not as gated.
        client.platform().set_list_models(json!([
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it",
             "is_accessible": false},
        ]));
        let resp = client.get("/v1/models").await;

        assert_eq!(resp.status_code, 200);
        assert_eq!(names(&resp.json()), vec!["gemma3-4b"]);
    }

    #[tokio::test]
    #[serial]
    async fn test_keeps_inaccessible_model_already_downloaded() {
        let client = TestClient::new();
        let workspace_models = WorkspaceModels::new();
        // Weights are on disk and still usable, so a plan downgrade must not
        // hide the model.
        workspace_models.append(ws_model("gemma3-12b", true, true));
        client.platform().set_list_models(json!([
            {"name": "gemma3-12b", "model_id": "google/gemma-3-12b-it",
             "access_level": "pro", "is_accessible": false},
        ]));
        let resp = client.get("/v1/models").await;

        assert_eq!(resp.status_code, 200);
        assert_eq!(names(&resp.json()), vec!["gemma3-12b"]);
    }

    #[tokio::test]
    #[serial]
    async fn test_lists_inaccessible_model_registered_but_not_downloaded() {
        let client = TestClient::new();
        let workspace_models = WorkspaceModels::new();
        // Registered, not yet on disk, and off the plan — still listed, with
        // the local `enabled` state resolved as for any other entry.
        workspace_models.append(ws_model("gemma3-12b", false, true));
        client.platform().set_list_models(json!([
            {"name": "gemma3-12b", "model_id": "google/gemma-3-12b-it",
             "access_level": "pro", "is_accessible": false},
        ]));
        let resp = client.get("/v1/models").await;

        assert_eq!(resp.status_code, 200);
        assert_eq!(names(&resp.json()), vec!["gemma3-12b"]);
    }

    #[tokio::test]
    #[serial]
    async fn test_returns_empty_list() {
        let client = TestClient::new();
        client.platform().set_list_models(json!([]));
        let resp = client.get("/v1/models").await;

        assert_eq!(resp.status_code, 200);
        assert_eq!(resp.json(), json!([]));
    }

    #[tokio::test]
    #[serial]
    async fn test_502_when_platform_unavailable() {
        let client = TestClient::new();
        client.platform().fail_list_models(PlatformError::connection("Connection refused"));
        let resp = client.get("/v1/models").await;

        assert_eq!(resp.status_code, 502);
        assert_eq!(resp.json()["detail"], "Platform API unavailable");
    }
}
