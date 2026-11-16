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
    async fn test_a_remote_model_keeps_the_catalogs_capabilities() {
        let client = TestClient::new();
        // A remote model's capabilities come from the catalog, minus ones
        // this build can't route, the retired "general" and "coding" (what
        // a turn is about, not a capability) and the internal-only
        // "thinking".
        client.platform().set_list_models(json!([
            {"name": "kimi-k2", "model_id": null, "type": "remote",
             "access_level": "free_login", "is_accessible": true,
             "capabilities": ["general", "coding", "thinking", "teleport"],
             "context_length": 131072},
        ]));
        let resp = client.get("/v1/models").await;

        assert_eq!(resp.status_code, 200);
        let body = resp.json();
        let remote = &body[0];
        assert_eq!(remote["type"], "remote");
        // "general" and "coding" read as `text`; thinking stays internal;
        // "teleport" is nothing.
        assert_eq!(remote["capabilities"], json!(["text"]));
        assert_eq!(remote["context_length"], 131072);
        assert_eq!(remote["access_level"], "free_login");
    }

    #[tokio::test]
    #[serial]
    async fn test_models_report_their_parameter_size() {
        let client = TestClient::new();
        client.platform().set_list_models(json!([
            {"name": "remote-sized", "model_id": null, "type": "remote",
             "is_accessible": true, "capabilities": ["general"], "param_size_b": 355.0,
             "param_size_estimated": true, "active_param_size_b": 32.0, "attention": "quantum"},
            {"name": "gemma4-e2b", "model_id": "google/gemma-4-e2b-it", "is_accessible": true},
        ]));
        let resp = client.get("/v1/models").await;

        let body = resp.json();
        let by_name = |name: &str| body.as_array().unwrap().iter().find(|m| m["name"] == name).unwrap().clone();
        assert_eq!(by_name("remote-sized")["param_size_b"], 355.0);
        assert_eq!(by_name("remote-sized")["param_size_estimated"], true);
        assert_eq!(by_name("remote-sized")["active_param_size_b"], 32.0);
        // An attention type this build doesn't know doesn't drop the row.
        assert_eq!(by_name("remote-sized")["attention"], Value::Null);
        // A catalog without the fields still reports the built-in size, as
        // an exact one.
        assert_eq!(by_name("gemma4-e2b")["param_size_b"], 2.0);
        assert_eq!(by_name("gemma4-e2b")["param_size_estimated"], false);
    }

    #[tokio::test]
    #[serial]
    async fn test_a_catalog_only_local_model_is_listed_with_its_spec() {
        let client = TestClient::new();
        client.platform().set_list_models(json!([
            {"name": "catalog-only-8b", "model_id": "org/catalog-only-8b", "type": "local",
             "is_accessible": true, "capabilities": ["general", "dp", "vision"],
             "param_size_b": 8.0, "model_version": 4, "vision_chat_handler": "Gemma4ChatHandler"},
        ]));
        let resp = client.get("/v1/models").await;

        let entry = &resp.json()[0];
        assert_eq!(entry["name"], "catalog-only-8b");
        assert_eq!(entry["capabilities"], json!(["text", "vision"]));
        assert_eq!(entry["param_size_b"], 8.0);
    }

    #[tokio::test]
    #[serial]
    async fn test_a_missing_url_falls_back_to_the_platform() {
        let client = TestClient::new();
        client.platform().set_list_models(json!([
            {"name": "kimi-k2", "model_id": "", "type": "remote"},
        ]));
        let resp = client.get("/v1/models").await;

        let expected = smartloop::config::app_settings::AppSettings::new().slp_api_base;
        assert_eq!(resp.json()[0]["url"], json!(expected));
    }

    #[tokio::test]
    #[serial]
    async fn test_free_login_sorts_between_free_and_pro() {
        let client = TestClient::new();
        client.platform().set_list_models(json!([
            {"name": "pro-model", "model_id": "", "access_level": "pro"},
            {"name": "login-model", "model_id": "", "access_level": "free_login"},
            {"name": "free-model", "model_id": "", "access_level": "free"},
        ]));
        let resp = client.get("/v1/models").await;

        assert_eq!(names(&resp.json()), vec!["free-model", "login-model", "pro-model"]);
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
    async fn test_a_cloud_copy_the_plan_does_not_cover_is_still_reported() {
        let client = TestClient::new();
        // Listed both ways, the cloud copy locked: it's `cloud_variant`, so a
        // client can show it as there but locked, though not `cloud`.
        client.platform().set_list_models(json!([
            {"name": "qwen3.8-27b", "model_id": "Qwen/Qwen3.8-27B", "type": "local",
             "access_level": "free", "is_accessible": true},
            {"name": "qwen3.8-27b", "model_id": null, "type": "remote", "url": "https://edge.smartloop.ai",
             "access_level": "pro", "is_accessible": false},
            {"name": "gemma3-4b", "model_id": "google/gemma-3-4b-it", "type": "local",
             "access_level": "free", "is_accessible": true},
        ]));
        let resp = client.get("/v1/models").await;

        assert_eq!(resp.status_code, 200);
        let body = resp.json();
        let rows = body.as_array().unwrap();
        let row = |name: &str| rows.iter().find(|m| m["name"] == name).cloned().unwrap();
        assert_eq!(rows.iter().filter(|m| m["name"] == "qwen3.8-27b").count(), 1, "one entry: {body}");
        let qwen = row("qwen3.8-27b");
        assert_eq!(qwen["cloud_variant"], true, "{qwen}");
        assert_eq!(qwen["cloud"], false, "{qwen}");
        assert_eq!(row("gemma3-4b")["cloud_variant"], false);
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
