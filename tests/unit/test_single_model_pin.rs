//! Port of `tests/unit/test_single_model_pin.py`.
//!
//! When the user selects exactly one model, that model runs the whole request.
//!
//! Capability routing exists to spread a request across several selected
//! models.  With a single selection there is nothing to route between, so
//! every path that would otherwise fall back to the bootstrap orchestrator
//! model resolves to the selected model instead.  Two or more selections keep
//! today's routing untouched.
//!
//! The pin assumes the selected model can hold a text conversation.  One that
//! declares only `vision` (qwen2-vl-2b) is pinned to the vision steps it was
//! chosen for and nothing else.
//!
//! The Python tests mocked `_load_workspace_llm` and `_detect_large_model`.
//! Here the workspace is real and the GGUF lookup / fit guard / llama.cpp load
//! go through the service's seams: a load is recorded by model name, a model
//! with "no GGUF" stands in for one that cannot be loaded, and large-model
//! detection is observed through the `large_model_attempted` latch it sets.
//! "chat-model" (an unsupported name) declares only `general`; "gemma3-4b"
//! stands in for the Python "big-doc-model" (`general` + `dp`).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;

use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::llms::base_llm::BaseLLM;
use smartloop::types::ANSWER_TIER;
use smartloop::workspace::WorkspaceModel;

use crate::shared::model_service_b::{FakeLLM, default_model, project, same, service_with_workspace, ws_model};

const BIG_DOC_MODEL: &str = "gemma3-4b";

struct Pin {
    svc: Arc<ModelService>,
    bootstrap: Arc<dyn BaseLLM>,
    /// Model names the llama.cpp load seam was asked for, in order.
    loads: Arc<Mutex<Vec<String>>>,
}

impl Pin {
    fn last_load(&self) -> Option<String> {
        self.loads.lock().last().cloned()
    }

    fn detection_ran(&self) -> bool {
        self.svc.large_model_attempted.load(Ordering::SeqCst)
    }

    fn is_bootstrap(&self, llm: &Option<Arc<dyn BaseLLM>>) -> bool {
        llm.as_ref().is_some_and(|l| same(l, &self.bootstrap))
    }
}

fn model(name: &str, enabled: bool, downloaded: bool) -> WorkspaceModel {
    ws_model(name, enabled, downloaded)
}

/// The model name a load was for: the argument that is not the GGUF path.
fn model_name_of(a: &str, b: &str) -> String {
    if a.ends_with(".gguf") { b.to_string() } else { a.to_string() }
}

/// A service whose workspace holds `models` plus the bootstrap model.
/// `loadable=false` stands in for a model with no GGUF on disk.
fn service_with(models: Vec<WorkspaceModel>, loadable: bool) -> Pin {
    let mut all = vec![model(&default_model(), true, true)];
    all.extend(models);
    let svc = service_with_workspace("/tmp/slp-pin-test", vec![project("p1", "/tmp/slp-pin-test/p1", true)], all);
    let bootstrap = FakeLLM::new("bootstrap_llm").dyn_llm();
    *svc.llm.write() = Some(bootstrap.clone());
    *svc.large_llm.write() = None;
    *svc.large_model_name.write() = None;
    svc.large_model_attempted.store(false, Ordering::SeqCst);

    let loads: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let seen = loads.clone();
    let mut hooks = ModelServiceHooks::default();
    hooks.find_model_ggufs = Some(Arc::new(move |model_dir: &str| {
        if loadable { (Some(format!("{model_dir}/model.gguf")), None) } else { (None, None) }
    }));
    hooks.large_model_fits = Some(Arc::new(|_gguf: &str, _cap: i64, _label: &str, _reserve: Option<f64>| true));
    hooks.load_llama_instance = Some(Arc::new(move |a: &str, b: &str, _cap: i64, _thinking: Option<bool>| {
        let name = model_name_of(a, b);
        seen.lock().push(name.clone());
        let llm: Arc<dyn BaseLLM> = FakeLLM::new(&format!("llm_{name}"));
        Ok(llm)
    }));
    svc.set_hooks(hooks);
    Pin { svc, bootstrap, loads }
}

fn service(models: Vec<WorkspaceModel>) -> Pin {
    service_with(models, true)
}

// ── which selection counts ────────────────────────────────────────────────

#[test]
fn test_single_selection_is_pinned() {
    let pin = service(vec![model(BIG_DOC_MODEL, true, true)]);
    assert_eq!(pin.svc.pinned_model(None).map(|m| m.model_name).as_deref(), Some(BIG_DOC_MODEL));
}

/// It is always enabled and hidden from the picker, so it never blocks a pin.
#[test]
fn test_bootstrap_model_does_not_count_as_a_selection() {
    let pin = service(vec![model(BIG_DOC_MODEL, true, true)]);
    let names: Vec<String> = pin.svc.selected_models().into_iter().map(|m| m.model_name).collect();
    assert_eq!(names, vec![BIG_DOC_MODEL]);
}

#[test]
fn test_two_selections_are_not_pinned() {
    let pin = service(vec![model(BIG_DOC_MODEL, true, true), model("chat-model", true, true)]);
    assert!(pin.svc.pinned_model(None).is_none());
}

#[test]
fn test_disabled_model_is_not_a_selection() {
    let pin = service(vec![model(BIG_DOC_MODEL, true, true), model("chat-model", false, true)]);
    assert_eq!(pin.svc.pinned_model(None).map(|m| m.model_name).as_deref(), Some(BIG_DOC_MODEL));
}

#[test]
fn test_undownloaded_model_is_not_a_selection() {
    let pin = service(vec![model(BIG_DOC_MODEL, true, true), model("chat-model", true, false)]);
    assert_eq!(pin.svc.pinned_model(None).map(|m| m.model_name).as_deref(), Some(BIG_DOC_MODEL));
}

#[test]
fn test_no_selection_is_not_pinned() {
    let pin = service(vec![]);
    assert!(pin.svc.pinned_model(None).is_none());
}

// ── what the pin replaces ─────────────────────────────────────────────────

/// The point of the fix: no fallback to the model the user did not pick.
#[test]
fn test_pinned_model_serves_a_capability_it_does_not_declare() {
    let pin = service(vec![model("chat-model", true, true)]);

    let llm = pin.svc.get_llm_for_capability("reasoning");

    assert!(llm.is_some());
    assert_eq!(pin.last_load().as_deref(), Some("chat-model"));
}

#[test]
fn test_unloadable_pinned_model_falls_back_rather_than_failing() {
    let pin = service_with(vec![model("chat-model", true, true)], false);

    assert!(pin.svc.get_llm_for_capability("reasoning").is_none());
}

/// Even a small selection: detection only promotes bigger models.
#[test]
fn test_pin_is_the_large_agent_model() {
    let pin = service(vec![model("chat-model", true, true)]);

    let llm = pin.svc.get_large_llm();

    assert!(llm.is_some());
    assert!(!pin.detection_ran());
}

#[test]
fn test_pin_writes_the_response_when_no_capability_is_tagged() {
    let pin = service(vec![model("chat-model", true, true)]);

    assert!(!pin.is_bootstrap(&pin.svc.llm_for_response(None)));
}

// ── two or more selections keep today's routing ───────────────────────────

#[test]
fn test_multi_selection_still_routes_by_capability() {
    let pin = service(vec![model(BIG_DOC_MODEL, true, true), model("chat-model", true, true)]);

    pin.svc.get_llm_for_capability(ANSWER_TIER);

    assert_eq!(pin.last_load().as_deref(), Some(BIG_DOC_MODEL));
}

#[test]
fn test_multi_selection_with_no_capable_model_still_falls_back() {
    let pin = service(vec![model(BIG_DOC_MODEL, true, true), model("chat-model", true, true)]);

    assert!(pin.svc.get_llm_for_capability("vision").is_none());
}

#[test]
fn test_multi_selection_response_still_uses_the_bootstrap_llm_untagged() {
    let pin = service(vec![model(BIG_DOC_MODEL, true, true), model("chat-model", true, true)]);

    assert!(pin.is_bootstrap(&pin.svc.llm_for_response(None)));
}

#[test]
fn test_multi_selection_large_model_detection_still_runs() {
    let pin = service(vec![model(BIG_DOC_MODEL, true, true), model("chat-model", true, true)]);

    pin.svc.get_large_llm();

    assert!(pin.detection_ran());
}

// ── a pin that cannot be loaded falls back to the default model ───────────

/// One model selected, but it has no GGUF / doesn't fit / faulted.
fn broken_pin() -> Pin {
    service_with(vec![model("chat-model", true, true)], false)
}

#[test]
fn test_failed_pin_leaves_the_capability_caller_to_fall_back() {
    assert!(broken_pin().svc.get_llm_for_capability("reasoning").is_none());
}

/// The pinned model is the only candidate detection could find.
#[test]
fn test_failed_pin_does_not_retry_through_large_model_detection() {
    let pin = broken_pin();
    assert!(pin.svc.get_large_llm().is_none());
    assert!(!pin.detection_ran());
}

#[test]
fn test_failed_pin_answers_on_the_default_model_untagged() {
    let pin = broken_pin();
    assert!(pin.is_bootstrap(&pin.svc.llm_for_response(None)));
}

#[test]
fn test_failed_pin_answers_on_the_default_model_with_a_capability() {
    let pin = broken_pin();
    assert!(pin.is_bootstrap(&pin.svc.llm_for_response(Some("reasoning"))));
}

// ── a vision-only selection is pinned only to vision work ─────────────────

/// One model selected, and it declares nothing but `vision`.
fn vision_only() -> Pin {
    service(vec![model("qwen2-vl-2b", true, true)])
}

/// It is what the user picked it for, so the image step is still its own.
#[test]
fn test_a_vision_only_selection_is_pinned_for_a_vision_step() {
    let pin = vision_only();
    assert_eq!(pin.svc.pinned_model(Some("vision")).map(|m| m.model_name).as_deref(), Some("qwen2-vl-2b"));
}

/// The regression: a vision-only model pinned to the answer tier, or to a
/// capability it doesn't declare, hands text work to a model that cannot do it.
#[test]
fn test_a_vision_only_selection_is_not_pinned_for_a_text_step() {
    let pin = vision_only();
    assert!(pin.svc.pinned_model(Some(ANSWER_TIER)).is_none());
    assert!(pin.svc.pinned_model(Some("reasoning")).is_none());
}

/// The planner, the skill matcher and the relevance judge ask with no
/// capability at all — they are text nodes and must not land on it.
#[test]
fn test_a_vision_only_selection_is_not_pinned_without_a_capability() {
    let pin = vision_only();
    assert!(pin.svc.pinned_model(None).is_none());
    assert!(pin.svc.pinned_llm(None).is_none());
}

/// No selected model declares 'dp', so the caller gets None and answers on
/// the default model rather than on the vision-only pin.
#[test]
fn test_a_text_step_falls_back_to_the_default_model() {
    let pin = vision_only();
    assert!(pin.svc.get_llm_for_capability("reasoning").is_none());
    assert!(pin.loads.lock().is_empty());
}

#[test]
fn test_a_vision_step_still_loads_the_selection() {
    let pin = vision_only();

    let llm = pin.svc.get_llm_for_capability("vision");

    assert!(llm.is_some());
    assert_eq!(pin.last_load().as_deref(), Some("qwen2-vl-2b"));
}

/// Nothing selected declares 'dp', so the chain runs out at the default — it
/// does not stop at the pin on the way.
#[test]
fn test_the_response_to_a_text_step_is_written_by_the_default_model() {
    let pin = vision_only();
    assert!(pin.is_bootstrap(&pin.svc.llm_for_response(None)));
    assert!(pin.is_bootstrap(&pin.svc.llm_for_response(Some("reasoning"))));
}

/// The answer tier resolves to the bootstrap model (resident, so nothing is
/// loaded), never to the vision-only selection.
#[test]
fn test_an_answer_tier_step_resolves_to_the_bootstrap_model_not_the_selection() {
    let pin = vision_only();

    let llm = pin.svc.llm_for_response(Some(ANSWER_TIER));

    assert!(pin.is_bootstrap(&llm));
    assert!(!pin.loads.lock().iter().any(|name| name == "qwen2-vl-2b"));
}

#[test]
fn test_the_response_to_a_vision_step_is_written_by_the_selection() {
    let pin = vision_only();
    assert!(!pin.is_bootstrap(&pin.svc.llm_for_response(Some("vision"))));
}

/// Agent/synthesis work is text, so detection runs instead of the pin.
#[test]
fn test_the_vision_only_selection_is_not_the_large_agent_model() {
    let pin = vision_only();

    pin.svc.get_large_llm();

    assert!(pin.detection_ran());
}

/// has_capability_model gates the "Loading model for X…" event: promising a
/// load the vision-only pin will not do is the dishonesty it guards against.
#[test]
fn test_a_text_capability_is_not_announced_as_available() {
    let pin = vision_only();
    assert!(!pin.svc.has_capability_model("reasoning"));
    assert!(pin.svc.has_capability_model("vision"));
}

/// The pin is unchanged for any text model.
#[test]
fn test_a_text_selection_still_serves_every_capability() {
    let pin = service(vec![model("chat-model", true, true)]);

    assert_eq!(pin.svc.pinned_model(Some("vision")).map(|m| m.model_name).as_deref(), Some("chat-model"));
    assert_eq!(pin.svc.pinned_model(Some("reasoning")).map(|m| m.model_name).as_deref(), Some("chat-model"));
    assert_eq!(pin.svc.pinned_model(None).map(|m| m.model_name).as_deref(), Some("chat-model"));
}
