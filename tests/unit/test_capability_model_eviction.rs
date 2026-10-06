//! Port of `tests/unit/test_capability_model_eviction.py`.
//!
//! Regression tests for `evict_capability_models` and the default-model guard.
//!
//! Pressing Esc mid-stream aborts the SSE generator, whose cleanup calls
//! `unload_auxiliary_models()` -> `evict_capability_models()`.  When the
//! capability ranking had picked the *default* orchestrator model, its
//! `_cap_llm_<default>` slot holds a separate Llama instance for the same
//! weights, so `free_llm`'s identity check against the resident model does not
//! cover it.  Closing it there frees the native context a live stream is still
//! decoding on — a use-after-free that crashes the whole server process.

use std::sync::Arc;

use parking_lot::Mutex;

use smartloop::api::services::model_service::{CachedLlm, GPU_FAULTED, ModelService, ModelServiceHooks};
use smartloop::llms::base_llm::BaseLLM;

use crate::shared::model_service_a::{FakeLLM, bare_service, same};

/// A ModelService with only the state the eviction path touches, and the
/// models `free_llm` was asked to release.
fn svc() -> (Arc<ModelService>, Arc<Mutex<Vec<Arc<dyn BaseLLM>>>>) {
    let s = bare_service();
    *s.llm.write() = Some(FakeLLM::named("resident_orchestrator"));
    *s.model_name.write() = Some("slp-orchestra-mini".to_string());
    *s.large_llm.write() = None;
    *s.large_model_name.write() = None;
    s.large_model_attempted.store(false, std::sync::atomic::Ordering::SeqCst);
    // The on-demand vision model is evicted here too, so the next step's
    // capability model can fit (see test_vision_evicted_for_capability).
    *s.vision_llm.write() = None;
    *s.vision_model_name.write() = None;
    s.vision_unavailable.store(false, std::sync::atomic::Ordering::SeqCst);

    let freed: Arc<Mutex<Vec<Arc<dyn BaseLLM>>>> = Arc::default();
    let record = freed.clone();
    s.set_hooks(ModelServiceHooks {
        cleanup_gpu_cache: Some(Arc::new(|| {})),
        free_llm: Some(Arc::new(move |llm: &Arc<dyn BaseLLM>| record.lock().push(llm.clone()))),
        ..Default::default()
    });
    (s, freed)
}

fn slot(svc: &ModelService, name: &str) -> Option<Arc<dyn BaseLLM>> {
    svc.cap_llm(name).and_then(|s| s.llm().cloned())
}

/// The default model's own capability slot must survive eviction.
#[test]
fn test_default_model_capability_slot_is_not_evicted() {
    let (svc, freed) = svc();
    let default_slot = FakeLLM::named("cap_llm_default");
    svc.set_cap_llm("slp_orchestra_mini", None); // unrelated attr
    svc.set_cap_llm("slp-orchestra-mini", Some(CachedLlm::Loaded(default_slot.clone())));

    let was_freed = svc.evict_capability_models(None);

    assert!(!was_freed);
    assert!(freed.lock().is_empty());
    assert!(same(&slot(&svc, "slp-orchestra-mini").expect("kept"), &default_slot));
}

/// Other capability models are still dropped to reclaim VRAM.
#[test]
fn test_non_default_capability_models_are_still_evicted() {
    let (svc, freed) = svc();
    let default_slot = FakeLLM::named("cap_llm_default");
    let other_slot = FakeLLM::named("cap_llm_other");
    svc.set_cap_llm("slp-orchestra-mini", Some(CachedLlm::Loaded(default_slot.clone())));
    svc.set_cap_llm("big-doc-model", Some(CachedLlm::Loaded(other_slot.clone())));

    let was_freed = svc.evict_capability_models(None);

    assert!(was_freed);
    let freed = freed.lock();
    assert_eq!(freed.len(), 1);
    assert!(same(&freed[0], &other_slot));
    assert!(slot(&svc, "big-doc-model").is_none());
    assert!(same(&slot(&svc, "slp-orchestra-mini").expect("kept"), &default_slot));
}

/// An explicitly kept model is preserved alongside the default.
#[test]
fn test_keep_model_name_still_honoured() {
    let (svc, freed) = svc();
    let keep_slot = FakeLLM::named("cap_llm_keep");
    let drop_slot = FakeLLM::named("cap_llm_drop");
    svc.set_cap_llm("keep-me", Some(CachedLlm::Loaded(keep_slot.clone())));
    svc.set_cap_llm("drop-me", Some(CachedLlm::Loaded(drop_slot.clone())));

    let was_freed = svc.evict_capability_models(Some("keep-me"));

    assert!(was_freed);
    let freed = freed.lock();
    assert_eq!(freed.len(), 1);
    assert!(same(&freed[0], &drop_slot));
    assert!(same(&slot(&svc, "keep-me").expect("kept"), &keep_slot));
}

/// A faulted slot is a latch, not a live model — never freed.
#[test]
fn test_gpu_faulted_sentinel_is_left_in_place() {
    let (svc, freed) = svc();
    svc.set_cap_llm("faulted", Some(GPU_FAULTED));

    let was_freed = svc.evict_capability_models(None);

    assert!(!was_freed);
    assert!(freed.lock().is_empty());
    assert!(svc.cap_llm("faulted").expect("latched").is_gpu_faulted());
}

/// Before a default is resolved, eviction still works normally.
#[test]
fn test_no_default_model_name_does_not_crash() {
    let (svc, freed) = svc();
    *svc.model_name.write() = None;
    let some_slot = FakeLLM::named("cap_llm_some");
    svc.set_cap_llm("some-model", Some(CachedLlm::Loaded(some_slot.clone())));

    let was_freed = svc.evict_capability_models(None);

    assert!(was_freed);
    let freed = freed.lock();
    assert_eq!(freed.len(), 1);
    assert!(same(&freed[0], &some_slot));
}
