//! Port of `tests/unit/test_capability_model_reuses_resident.py`.
//!
//! A capability step must reuse a resident model, not load a second copy.
//!
//! The orchestrator model declares GENERAL, so a step routed to that capability
//! used to walk straight past the resident instance and build its own llama.cpp
//! context over the *same* GGUF — the weights and the KV cache of a model
//! already sitting in memory, duplicated.  On a shared-memory GPU that second
//! copy is what tips the device over its allocation limit.
//!
//! `get_llm_for_capability` now returns the resident instance for any
//! candidate that is already loaded — the orchestrator model or the on-demand
//! large one.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use smartloop::api::services::model_service::{CachedLlm, ModelService, ModelServiceHooks};
use smartloop::llms::base_llm::BaseLLM;
use smartloop::types::ANSWER_TIER;
use smartloop::workspace::{Workspace, WorkspaceModel};

use crate::shared::model_service_a::{FakeLLM, attach_workspace, bare_service, same};

fn ws_model(name: &str) -> WorkspaceModel {
    WorkspaceModel { model_name: name.to_string(), model_id: name.to_string(), ..Default::default() }
}

/// A ModelService carrying only what the capability path reads.
fn svc() -> Arc<ModelService> {
    let s = bare_service();
    *s.llm.write() = Some(FakeLLM::named("resident_orchestrator"));
    *s.model_name.write() = Some("sl-mini".to_string());
    *s.large_llm.write() = None;
    *s.large_model_name.write() = None;
    attach_workspace(&s, Workspace::new("ws"));
    s
}

/// Run the real `get_llm_for_capability` over `ranked` candidates, returning
/// `(llm, loaded)` — `loaded` is true if a GGUF was opened.
fn resolve(svc: &Arc<ModelService>, capability: &str, ranked: Vec<WorkspaceModel>) -> (Option<Arc<dyn BaseLLM>>, bool) {
    let loaded = Arc::new(AtomicBool::new(false));
    let loaded_in_hook = loaded.clone();
    svc.set_hooks(ModelServiceHooks {
        ranked_models_for: Some(Arc::new(move |_| ranked.clone())),
        find_model_ggufs: Some(Arc::new(|_| (Some("/models/x.gguf".to_string()), None))),
        large_model_fits: Some(Arc::new(|_, _, _, _| true)),
        load_llama_instance: Some(Arc::new(move |_, name: &str, _, _| {
            loaded_in_hook.store(true, Ordering::SeqCst);
            Ok(FakeLLM::named(name))
        })),
        ..Default::default()
    });
    let llm = svc.get_llm_for_capability(capability);
    (llm, loaded.load(Ordering::SeqCst))
}

/// The regression: the answer tier resolves to the model already answering
/// as the router, so it must hand back that instance.
#[test]
fn test_the_orchestrator_model_is_not_loaded_twice() {
    let svc = svc();
    let (llm, loaded) = resolve(&svc, ANSWER_TIER, vec![ws_model("sl-mini")]);

    assert!(same(&llm.expect("resident"), &svc.llm.read().clone().unwrap()));
    assert!(!loaded);
}

/// A workspace written before the rename records the same model as
/// `slp-orchestra-mini`; it is not a different model.
#[test]
fn test_a_legacy_name_still_matches_the_resident_model() {
    let svc = svc();
    let (llm, loaded) = resolve(&svc, ANSWER_TIER, vec![ws_model("slp-orchestra-mini")]);

    assert!(same(&llm.expect("resident"), &svc.llm.read().clone().unwrap()));
    assert!(!loaded);
}

/// The on-demand large model is resident outside the capability cache, so it
/// is the same trap.
#[test]
fn test_the_resident_large_model_is_reused_too() {
    let svc = svc();
    let large = FakeLLM::named("resident_large");
    *svc.large_llm.write() = Some(large.clone());
    *svc.large_model_name.write() = Some("sl-4b".to_string());

    let (llm, loaded) = resolve(&svc, "reasoning", vec![ws_model("sl-4b")]);

    assert!(same(&llm.expect("resident"), &large));
    assert!(!loaded);
}

/// Reuse must not turn into a refusal to load anything: a genuinely different
/// capability model is loaded and cached as before.
#[test]
fn test_a_model_that_is_not_resident_is_still_loaded() {
    let svc = svc();
    let (llm, loaded) = resolve(&svc, "reasoning", vec![ws_model("coder-7b")]);

    assert!(loaded);
    let llm = llm.expect("a loaded model");
    assert!(!same(&llm, &svc.llm.read().clone().unwrap()));
    match svc.cap_llm("coder-7b") {
        Some(CachedLlm::Loaded(cached)) => assert!(same(&cached, &llm)),
        other => panic!("expected the loaded model to be cached, got {other:?}"),
    }
}
