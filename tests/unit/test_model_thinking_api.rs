//! Port of `tests/unit/test_model_thinking_api.py`.
//!
//! The picker reads and writes the thinking setting through /v1/models.
//!
//! The toggle has to be visible per model (so the UI knows which models can
//! reason) and settable (so switching it on sticks) — and switching it on has
//! to reach a model that is already loaded, since the always-resident
//! orchestrator would otherwise never pick the change up.

use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use smartloop::api::routers::models::{update_workspace_model, UpdateWorkspaceModelRequest, WorkspaceModelHost};
use smartloop::workspace::workspace::{Project, SharedWorkspace, Workspace, WorkspaceModel};

fn workspace() -> SharedWorkspace {
    let mut ws = Workspace::new("ws1");
    ws.models = vec![
        WorkspaceModel {
            model_name: "qwen3.5-2b".to_string(),
            model_id: "Qwen/Qwen3.5-2B".to_string(),
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
    ];
    ws.projects = vec![Project {
        id: "p1".to_string(),
        project_dir: "/tmp/p1".to_string(),
        current: true,
        name: Some("p1".to_string()),
        ..Default::default()
    }];
    Arc::new(RwLock::new(ws))
}

/// The `MagicMock` model service: hands out the workspace and records the
/// calls the route makes on it.
struct Service {
    workspace: SharedWorkspace,
    unloaded: Mutex<Vec<String>>,
    thinking_applied: Mutex<Vec<String>>,
}

impl Service {
    fn new(workspace: SharedWorkspace) -> Self {
        Self { workspace, unloaded: Mutex::new(Vec::new()), thinking_applied: Mutex::new(Vec::new()) }
    }
}

impl WorkspaceModelHost for Service {
    fn ensure_workspace(&self) -> anyhow::Result<SharedWorkspace> {
        Ok(Arc::clone(&self.workspace))
    }

    fn unload_model_by_name(&self, model_name: &str) {
        self.unloaded.lock().push(model_name.to_string());
    }

    fn apply_thinking_setting(&self, model_name: &str) {
        self.thinking_applied.lock().push(model_name.to_string());
    }
}

/// Storing the preference would leave the picker showing a toggle that
/// silently does nothing.
///
/// (The Python `saves` fixture patched `Workspace.save` so nothing reached
/// disk; the rejection returns before any save, so there is nothing to stub.)
#[tokio::test]
async fn test_a_model_that_cannot_reason_is_rejected() {
    let ws = workspace();
    let svc = Service::new(Arc::clone(&ws));

    let exc = update_workspace_model(
        "gemma3-4b",
        UpdateWorkspaceModelRequest { thinking: Some(true), ..Default::default() },
        &svc,
    )
    .await
    .expect_err("thinking on a model that cannot reason must be refused");

    assert_eq!(exc.status_code, 400);
    assert_eq!(ws.read().get_model(Some("gemma3-4b")).unwrap().thinking, None);
    // Nothing reached the resident instance either.
    assert!(svc.thinking_applied.lock().is_empty());
    assert!(svc.unloaded.lock().is_empty());
}

/// Capabilities set here win over the catalog's and the model's name, for
/// a local or a remote model; an empty list takes the override off; a
/// name the router can't route is refused.
#[tokio::test]
async fn capabilities_can_be_set_here_over_the_catalog() {
    let workspace = workspace();
    let service = Service::new(workspace.clone());
    let update = |caps: Vec<&str>| {
        UpdateWorkspaceModelRequest { capabilities: Some(caps.into_iter().map(str::to_string).collect()), ..Default::default() }
    };
    let before = workspace.read().get_model(Some("gemma3-4b")).unwrap().effective_capabilities();
    assert!(!before.iter().any(|c| c == "vision"), "{before:?}");

    // "general" and "coding" are what older clients said about every text
    // model: they still parse, and declare nothing.
    let reply = update_workspace_model("gemma3-4b", update(vec!["vision", "general", "Coding"]), &service).await.unwrap();
    assert_eq!(reply["capabilities"], serde_json::json!(["vision"]));
    assert_eq!(reply["capabilities_overridden"], serde_json::json!(true));
    assert_eq!(workspace.read().get_model(Some("gemma3-4b")).unwrap().effective_capabilities(), ["vision"]);

    let refused = update_workspace_model("gemma3-4b", update(vec!["poetry"]), &service).await.unwrap_err();
    assert_eq!(refused.status_code, 400);
    assert_eq!(workspace.read().get_model(Some("gemma3-4b")).unwrap().effective_capabilities(), ["vision"]);

    let reply = update_workspace_model("gemma3-4b", update(vec![]), &service).await.unwrap();
    assert_eq!(reply["capabilities_overridden"], serde_json::json!(false));
    assert_eq!(workspace.read().get_model(Some("gemma3-4b")).unwrap().effective_capabilities(), before);
}
