//! A vision model that does not fit releases the resident capability models
//! and re-checks, instead of waiting for memory that is never freed.
//!
//! Capability models stay loaded across responses, so on a tight device
//! (e.g. Qwen 3.5 2B resident, a vision-only Qwen2-VL 2B selected for OCR)
//! "will retry on the next document" never succeeded.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serial_test::serial;

use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::workspace::{Workspace, WorkspaceModel};

use crate::shared::model_service_a::{attach_workspace, bare_service, capture_logs};

/// A service with a vision-only model selected whose fit depends on whether
/// eviction ran; returns whether it did and the vision-path logs.
fn attempt(eviction_makes_room: bool) -> (bool, String) {
    let svc: Arc<ModelService> = bare_service();
    let mut ws = Workspace::new("ws");
    ws.models = vec![WorkspaceModel {
        model_name: "qwen2-vl-2b".to_string(),
        model_id: "Qwen/Qwen2-VL-2B-Instruct".to_string(),
        enabled: true,
        downloaded: true,
        ..Default::default()
    }];
    attach_workspace(&svc, ws);

    let evicted = Arc::new(AtomicBool::new(false));
    let (on_evict, on_fit) = (evicted.clone(), evicted.clone());
    svc.set_hooks(ModelServiceHooks {
        // Nonexistent paths: the load itself fails, after the fit decision.
        find_model_ggufs: Some(Arc::new(|_| {
            (Some("/nonexistent/qwen2-vl.gguf".to_string()), Some("/nonexistent/mmproj.gguf".to_string()))
        })),
        evict_capability_models: Some(Arc::new(move |_| {
            on_evict.store(true, Ordering::SeqCst);
            true
        })),
        large_model_fits: Some(Arc::new(move |_, _, _, _| {
            eviction_makes_room && on_fit.load(Ordering::SeqCst)
        })),
        ..Default::default()
    });

    let (_llm, text) = capture_logs(tracing::Level::INFO, || svc.get_vision_llm(None));
    (evicted.load(Ordering::SeqCst), text)
}

#[test]
#[serial]
fn test_capability_models_are_released_to_fit_the_vision_model() {
    let (evicted, text) = attempt(true);

    assert!(evicted);
    assert!(text.contains("[vision] Freed capability models to fit qwen2-vl-2b"), "{text}");
    assert!(!text.contains("won't fit"), "{text}");
}

#[test]
#[serial]
fn test_it_is_still_skipped_when_releasing_does_not_make_room() {
    let (evicted, text) = attempt(false);

    assert!(evicted);
    assert!(text.contains("qwen2-vl-2b skipped — won't fit"), "{text}");
}
