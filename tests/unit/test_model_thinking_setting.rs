//! Port of `tests/unit/test_model_thinking_setting.py`.
//!
//! Reasoning is a per-model setting, resolved like `enabled`.
//!
//! Thinking belongs to the model, not to a request: a model the user switches
//! it on for in the picker reasons on every call, so the preference is stored
//! on the workspace model (and overridable per project) and resolved through
//! the same project -> workspace -> default cascade the `enabled` flag uses.

use smartloop::workspace::workspace::{Project, Workspace, WorkspaceModel};

/// Workspace with one reasoning-capable model and one without.
fn make_workspace() -> Workspace {
    Workspace::new("ws1")
        .with_models(vec![
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
        ])
        .with_projects(vec![Project {
            id: "p1".to_string(),
            project_dir: "/tmp/p1".to_string(),
            current: true,
            name: Some("p1".to_string()),
            ..Default::default()
        }])
}

mod support {
    use super::*;

    #[test]
    fn test_an_unsupported_model_resolves_to_none() {
        // None means "bind nothing" — gemma3's template never reads
        // enable_thinking, so a stored preference would do nothing.
        let ws = make_workspace();
        assert_eq!(ws.effective_model_thinking("gemma3-4b", None), None);
    }
}
