//! A remote model in the workspace registry: no weights on disk, catalog
//! capabilities, never the active model, ranked after every local model.
//!
//! Its endpoint, capabilities and size come from its spec, which these tests
//! register under a name of their own: the spec registry is process-wide.

use serde_json::json;
use smartloop::model::spec::MODEL_SPECS;
use smartloop::types::ModelKind;
use smartloop::workspace::workspace::{Workspace, WorkspaceModel};

use crate::shared::platform_support::obj;

fn workspace() -> Workspace {
    let mut ws = Workspace::new("ws-remote");
    ws.models_dir = "/tmp/ws-remote/models".to_string();
    ws
}

fn caps(names: &[&str]) -> Vec<String> {
    names.iter().map(|c| c.to_string()).collect()
}

/// Publish a remote model's catalog row to the spec registry.
fn remote_spec(name: &str, row: serde_json::Value) {
    let mut row = obj(row);
    row.insert("name".into(), json!(name));
    row.insert("type".into(), json!("remote"));
    MODEL_SPECS.hydrate(&[row]);
}

#[test]
fn test_registered_remote_model_is_ready_and_never_active() {
    remote_spec(
        "remote-ready",
        json!({"url": "https://api.smartloop.ai", "capabilities": ["general", "coding"], "context_length": 131072}),
    );
    let mut ws = workspace();
    let added = ws.add_remote_model("remote-ready", true);

    assert!(added.is_remote());
    assert!(added.downloaded);
    assert!(!added.is_active);
    assert_eq!(added.url().as_deref(), Some("https://api.smartloop.ai"));
    assert_eq!(added.context_length(), Some(131072));
}

#[test]
fn test_details_follow_the_catalog_not_the_registration() {
    remote_spec("remote-refresh", json!({"url": "https://old.example", "capabilities": ["general"]}));
    let mut ws = workspace();
    ws.add_remote_model("remote-refresh", false);
    remote_spec("remote-refresh", json!({"url": "https://api.smartloop.ai", "capabilities": ["coding"]}));
    let again = ws.add_remote_model("remote-refresh", true);

    assert_eq!(ws.models.len(), 1);
    assert_eq!(again.url().as_deref(), Some("https://api.smartloop.ai"));
    assert_eq!(again.effective_capabilities(), caps(&["coding"]));
    // Re-adding keeps the user's enabled choice.
    assert!(!again.enabled);
}

#[test]
fn test_a_remote_model_without_a_url_uses_the_platform_api() {
    remote_spec("remote-no-url", json!({"capabilities": ["general"]}));
    let mut ws = workspace();
    let added = ws.add_remote_model("remote-no-url", true);
    assert_eq!(added.url(), Some(smartloop::config::AppSettings::new().slp_api_base));
}

#[test]
fn test_a_remote_model_has_no_weights_path() {
    let mut ws = workspace();
    let added = ws.add_remote_model("remote-no-weights", true);

    // An empty model_id must not resolve to the models directory itself.
    assert_eq!(added.model_dir(), "");
    assert_eq!(added.model_path(), "");
    assert_eq!(added.bin_dir(), "");
    assert_eq!(added.gguf_path(), "");
}

#[test]
fn test_remote_capabilities_come_from_the_catalog() {
    remote_spec("deepseek-coder-remote", json!({"capabilities": ["general"]}));
    let model = WorkspaceModel {
        model_name: "deepseek-coder-remote".to_string(),
        kind: ModelKind::Remote,
        ..Default::default()
    };
    // Name inference would add "coding"; the catalog's list wins.
    assert_eq!(model.effective_capabilities(), caps(&["general"]));
}

#[test]
fn test_a_remote_model_without_a_spec_declares_nothing() {
    let model = WorkspaceModel {
        model_name: "remote-coder-unknown".to_string(),
        kind: ModelKind::Remote,
        ..Default::default()
    };
    assert!(model.effective_capabilities().is_empty());
}

#[test]
fn test_a_remote_model_ranks_on_its_active_params() {
    remote_spec(
        "remote-sized",
        json!({"capabilities": ["general"], "param_size_b": 428.0, "active_param_size_b": 23.0}),
    );
    let mut ws = workspace();
    let added = ws.add_remote_model("remote-sized", true);

    assert_eq!(added.param_size_b(), 428.0);
    assert_eq!(added.rank(), (23.0, 0));
}

#[test]
fn test_remote_models_rank_among_themselves_by_active_params() {
    remote_spec("remote-light", json!({"capabilities": ["general"], "param_size_b": 900.0, "active_param_size_b": 23.0}));
    remote_spec("remote-heavy", json!({"capabilities": ["general"], "param_size_b": 753.0, "active_param_size_b": 40.0}));
    let mut ws = workspace();
    ws.add_remote_model("remote-light", true);
    ws.add_remote_model("remote-heavy", true);
    assert_eq!(ranked_names(&ws, "general"), vec!["remote-heavy", "remote-light"]);
}

#[test]
fn test_a_remote_model_is_never_the_largest_downloaded_one() {
    remote_spec("remote-giant", json!({"capabilities": ["general"], "param_size_b": 753.0}));
    let mut ws = workspace();
    ws.add_remote_model("remote-giant", true);
    assert!(ws.largest_downloaded_model(None).is_none());

    ws.add_model("gemma4-e2b", "gemma4-e2b", None, true);
    ws.get_model_mut(Some("gemma4-e2b")).unwrap().downloaded = true;
    assert_eq!(ws.largest_downloaded_model(None).unwrap().model_name, "gemma4-e2b");
}

fn ranked_names(ws: &Workspace, capability: &str) -> Vec<String> {
    ws.ranked_models_for(capability, None).into_iter().map(|m| m.model_name).collect()
}

#[test]
fn test_remote_models_rank_after_local_ones() {
    remote_spec("remote-ranked", json!({"capabilities": ["general", "coding"], "param_size_b": 1000.0}));
    let mut ws = workspace();
    ws.add_remote_model("remote-ranked", true);
    ws.add_model("gemma4-e4b", "gemma4-e4b", None, true);
    ws.get_model_mut(Some("gemma4-e4b")).unwrap().downloaded = true;
    assert_eq!(ranked_names(&ws, "general"), vec!["gemma4-e4b", "remote-ranked"]);
}

#[test]
fn test_a_capability_only_remote_models_declare_ranks_them() {
    remote_spec("remote-coder", json!({"capabilities": ["general", "coding"]}));
    let mut ws = workspace();
    ws.add_remote_model("remote-coder", true);
    assert_eq!(ranked_names(&ws, "coding"), vec!["remote-coder"]);
}

#[test]
fn test_a_local_model_persists_without_remote_fields() {
    let model = WorkspaceModel { model_name: "gemma3-4b".into(), model_id: "google/gemma-3-4b-it".into(), ..Default::default() };
    let value = serde_json::to_value(&model).unwrap();
    assert!(value.get("type").is_none());

    let remote = WorkspaceModel { kind: ModelKind::Remote, ..model };
    let value = serde_json::to_value(&remote).unwrap();
    assert_eq!(value["type"], "remote");
    // The endpoint lives in the spec, so nothing else is stored.
    assert!(value.get("url").is_none());
    let back: WorkspaceModel = serde_json::from_value(value).unwrap();
    assert!(back.is_remote());
}
