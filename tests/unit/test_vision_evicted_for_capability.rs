//! Port of `tests/unit/test_vision_evicted_for_capability.py`.
//!
//! The vision model must give up its VRAM to the next step's capability model.
//!
//! A document request with an image runs OCR first, which loads the multimodal
//! model (e.g. qwen2-vl-2b) into `vision_llm`.  Extraction runs next and needs
//! the 'dp' model.  `evict_capability_models` used to free only the capability
//! slots and the large model, so the vision model stayed resident, the 'dp'
//! model failed its fit guard against the VRAM it held, and extraction fell
//! through to the bootstrap router.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;

use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::llms::base_llm::BaseLLM;

use crate::shared::model_service_b::{FakeLLM, same};

struct Fixture {
    svc: Arc<ModelService>,
    freed: Arc<Mutex<Vec<Arc<dyn BaseLLM>>>>,
}

fn svc() -> Fixture {
    let svc = ModelService::new_bare();
    *svc.llm.write() = Some(FakeLLM::new("resident_orchestrator").dyn_llm());
    *svc.model_name.write() = Some("sl-mini".into());
    *svc.large_llm.write() = None;
    *svc.large_model_name.write() = None;
    svc.large_model_attempted.store(false, Ordering::SeqCst);
    *svc.vision_llm.write() = None;
    *svc.vision_model_name.write() = None;
    svc.vision_unavailable.store(false, Ordering::SeqCst);

    let freed: Arc<Mutex<Vec<Arc<dyn BaseLLM>>>> = Arc::new(Mutex::new(Vec::new()));
    let record = freed.clone();
    let mut hooks = ModelServiceHooks::default();
    hooks.cleanup_gpu_cache = Some(Arc::new(|| {}));
    hooks.free_llm = Some(Arc::new(move |llm: &Arc<dyn BaseLLM>| record.lock().push(llm.clone())));
    svc.set_hooks(hooks);
    Fixture { svc, freed }
}

fn load_vision(svc: &ModelService, name: &str) -> Arc<dyn BaseLLM> {
    let vision = FakeLLM::new(&format!("vision_{name}")).dyn_llm();
    *svc.vision_llm.write() = Some(vision.clone());
    *svc.vision_model_name.write() = Some(name.to_string());
    vision
}

#[test]
fn test_vision_model_is_evicted_to_make_room() {
    let f = svc();
    let vision = load_vision(&f.svc, "qwen2-vl-2b");

    let freed = f.svc.evict_capability_models(Some("gemma4-e2b"));

    assert!(freed);
    let calls = f.freed.lock();
    assert_eq!(calls.len(), 1);
    assert!(same(&calls[0], &vision));
    assert!(f.svc.vision_llm.read().is_none());
}

/// A later page still needing OCR must be able to load it again.
#[test]
fn test_eviction_rearms_lazy_vision_reload() {
    let f = svc();
    load_vision(&f.svc, "qwen2-vl-2b");
    f.svc.vision_unavailable.store(true, Ordering::SeqCst);

    f.svc.evict_capability_models(Some("gemma4-e2b"));

    assert!(!f.svc.vision_unavailable.load(Ordering::SeqCst));
    assert!(f.svc.vision_model_name.read().is_none());
}

#[test]
fn test_vision_model_survives_when_it_is_the_model_being_loaded() {
    let f = svc();
    let vision = load_vision(&f.svc, "qwen2-vl-2b");

    let freed = f.svc.evict_capability_models(Some("qwen2-vl-2b"));

    assert!(!freed);
    assert!(f.freed.lock().is_empty());
    assert!(f.svc.vision_llm.read().as_ref().is_some_and(|l| same(l, &vision)));
}

#[test]
fn test_no_vision_model_loaded_is_a_no_op() {
    let f = svc();

    let freed = f.svc.evict_capability_models(Some("gemma4-e2b"));

    assert!(!freed);
    assert!(f.freed.lock().is_empty());
}
