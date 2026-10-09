//! A remote model in the workspace registry: no weights on disk, catalog
//! capabilities, never the active model, never auto-routed.

use smartloop::types::ModelKind;
use smartloop::workspace::workspace::{Workspace, WorkspaceModel};

fn workspace() -> Workspace {
    let mut ws = Workspace::new("ws-remote");
    ws.models_dir = "/tmp/ws-remote/models".to_string();
    ws
}

fn caps(names: &[&str]) -> Vec<String> {
    names.iter().map(|c| c.to_string()).collect()
}

#[test]
fn test_registered_remote_model_is_ready_and_never_active() {
    let mut ws = workspace();
    let added = ws.add_remote_model("kimi-k2", "https://api.smartloop.ai", caps(&["general", "coding"]), Some(131072), true);

    assert!(added.is_remote());
    assert!(added.downloaded);
    assert!(!added.is_active);
    assert_eq!(added.url.as_deref(), Some("https://api.smartloop.ai"));
    assert_eq!(added.context_length, Some(131072));
}

#[test]
fn test_re_adding_refreshes_catalog_details() {
    let mut ws = workspace();
    ws.add_remote_model("kimi-k2", "https://old.example", caps(&["general"]), None, false);
    let again = ws.add_remote_model("kimi-k2", "https://api.smartloop.ai", caps(&["coding"]), Some(8192), true);

    assert_eq!(ws.models.len(), 1);
    assert_eq!(again.url.as_deref(), Some("https://api.smartloop.ai"));
    assert_eq!(again.capabilities, caps(&["coding"]));
    // Re-adding keeps the user's enabled choice.
    assert!(!again.enabled);
}

#[test]
fn test_a_remote_model_has_no_weights_path() {
    let mut ws = workspace();
    let added = ws.add_remote_model("kimi-k2", "https://api.smartloop.ai", vec![], None, true);

    // An empty model_id must not resolve to the models directory itself.
    assert_eq!(added.model_dir(), "");
    assert_eq!(added.model_path(), "");
    assert_eq!(added.bin_dir(), "");
    assert_eq!(added.gguf_path(), "");
}

#[test]
fn test_remote_capabilities_come_from_the_catalog() {
    let model = WorkspaceModel {
        model_name: "deepseek-coder-remote".to_string(),
        kind: ModelKind::Remote,
        capabilities: caps(&["general"]),
        ..Default::default()
    };
    // Name inference would add "coding"; the catalog's list wins.
    assert_eq!(model.effective_capabilities(), caps(&["general"]));
}

#[test]
fn test_remote_models_are_not_auto_routed() {
    let mut ws = workspace();
    ws.add_remote_model("kimi-k2", "https://api.smartloop.ai", caps(&["general", "coding"]), None, true);
    assert!(ws.ranked_models_for("coding", None).is_empty());
}

#[test]
fn test_a_local_model_persists_without_remote_fields() {
    let model = WorkspaceModel { model_name: "gemma3-4b".into(), model_id: "google/gemma-3-4b-it".into(), ..Default::default() };
    let value = serde_json::to_value(&model).unwrap();
    assert!(value.get("type").is_none());
    assert!(value.get("url").is_none());

    let remote = WorkspaceModel { kind: ModelKind::Remote, url: Some("https://api.smartloop.ai".into()), ..model };
    let value = serde_json::to_value(&remote).unwrap();
    assert_eq!(value["type"], "remote");
    let back: WorkspaceModel = serde_json::from_value(value).unwrap();
    assert!(back.is_remote());
}
