//! A model the user deselects must stop serving, even while it is resident.
//!
//! Under supervision the selection is written by the project's child, while
//! the weights live in the model host (the supervisor).  The child's unload on
//! deselect runs where nothing is loaded, so the host's large and vision slots
//! kept handing the deselected model out until restart.  The host now checks a
//! cached model against the bound project's selection before reusing it.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::llms::base_llm::BaseLLM;
use smartloop::workspace::workspace::ProjectModel;

use crate::shared::model_service_b::{FakeLLM, default_model, project, same, service_with_workspace, ws_model};

const DESELECTED: &str = "gemma3-4b";
const KEPT: &str = "chat-model";
const ALSO_KEPT: &str = "other-chat-model";

/// Three selected models (so two remain after a deselect and nothing is pinned), with `DESELECTED` resident in
/// both the large and vision slots.
fn service() -> (Arc<ModelService>, Arc<dyn BaseLLM>) {
    let svc = service_with_workspace(
        "/tmp/slp-deselect-test",
        vec![project("p1", "/tmp/slp-deselect-test/p1", true)],
        vec![
            ws_model(&default_model(), true, true),
            ws_model(DESELECTED, true, true),
            ws_model(KEPT, true, true),
            ws_model(ALSO_KEPT, true, true),
        ],
    );
    *svc.llm.write() = Some(FakeLLM::new("bootstrap_llm").dyn_llm());
    let resident = FakeLLM::new("resident").dyn_llm();
    *svc.large_llm.write() = Some(resident.clone());
    *svc.large_model_name.write() = Some(DESELECTED.to_string());
    *svc.vision_llm.write() = Some(resident.clone());
    *svc.vision_model_name.write() = Some(DESELECTED.to_string());
    svc.large_model_attempted.store(true, Ordering::SeqCst);

    let mut hooks = ModelServiceHooks::default();
    // Nothing can be loaded: whatever comes back must be what was cached.
    hooks.find_model_ggufs = Some(Arc::new(|_dir: &str| (None, None)));
    svc.set_hooks(hooks);
    (svc, resident)
}

/// What the project's child writes when the user unticks the model.
fn deselect_in_project(svc: &ModelService) {
    let ws = svc.workspace_handle().expect("workspace");
    let mut ws = ws.write();
    let project = ws.project_mut("p1").expect("project");
    project.model_settings.push(ProjectModel {
        model_name: DESELECTED.to_string(),
        enabled: false,
        ..ProjectModel::default()
    });
}

#[test]
fn test_selected_large_model_is_reused() {
    let (svc, resident) = service();
    let llm = svc.get_large_llm().expect("cached large model");
    assert!(same(&llm, &resident));
}

#[test]
fn test_deselected_large_model_is_not_reused() {
    let (svc, resident) = service();
    deselect_in_project(&svc);

    let llm = svc.get_large_llm();

    assert!(!llm.is_some_and(|l| same(&l, &resident)));
    assert!(svc.large_llm.read().is_none());
    assert!(svc.large_model_name.read().is_none());
}

#[test]
fn test_deselected_vision_model_is_not_reused() {
    let (svc, resident) = service();
    deselect_in_project(&svc);

    let llm = svc.get_vision_llm(None);

    assert!(!llm.is_some_and(|l| same(&l, &resident)));
    assert!(svc.vision_llm.read().is_none());
}
