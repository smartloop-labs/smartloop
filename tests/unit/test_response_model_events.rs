//! Port of `tests/unit/test_response_model_events.py`.
//!
//! The model that writes the answer reports itself, like every other one.
//!
//! The response model is resolved by the streaming layer *after*
//! orchestration finishes, so it never went through ModelRouter and surfaced
//! nothing at all: a cold load sat between the last orchestration event and
//! the first token, as dead air exactly where the answer was expected to start.

use std::sync::Arc;

use parking_lot::Mutex;

use smartloop::agents::orchestration::model_router::{CapabilitySkipReason, ProgressCallback};
use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::llms::base_llm::BaseLLM;
use smartloop::types::ModelCapability;
use smartloop::workspace::WorkspaceModel;

use crate::shared::model_service_b::{FakeLLM, default_model, project, same, service_with_workspace, ws_model};

const GENERAL: &str = "general";

fn llm(name: &str) -> Arc<dyn BaseLLM> {
    FakeLLM::named(name).dyn_llm()
}

/// What `has_capability_model` answers, by workspace state: two selected
/// general models (no pin) when a capability model is available, no models
/// at all when it is not, and a single selection when the request is pinned.
#[derive(Clone, Copy, PartialEq)]
enum Selection {
    Available,
    Unavailable,
    Pinned,
}

struct Options {
    capability_llm: Option<Arc<dyn BaseLLM>>,
    selection: Selection,
    pinned: bool,
    pinned_llm: Option<Arc<dyn BaseLLM>>,
    large_llm: Option<Arc<dyn BaseLLM>>,
    skip: Option<CapabilitySkipReason>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            capability_llm: None,
            selection: Selection::Available,
            pinned: false,
            pinned_llm: None,
            large_llm: None,
            skip: None,
        }
    }
}

fn service(options: Options) -> Arc<ModelService> {
    let models: Vec<WorkspaceModel> = match options.selection {
        Selection::Available => vec![
            ws_model(&default_model(), true, true),
            ws_model("gemma3-4b", true, true),
            ws_model("ministral3-3b", true, true),
        ],
        // Nothing registered at all: not even the bootstrap model declares it.
        Selection::Unavailable => vec![],
        Selection::Pinned => vec![ws_model(&default_model(), true, true), ws_model("chat-model", true, true)],
    };
    let svc = service_with_workspace("/tmp", vec![project("p1", "/tmp/p1", true)], models);
    *svc.llm.write() = Some(llm("Slp Orchestra Mini"));
    *svc.response_router_instance.lock() = None;
    *svc.capability_skip.lock() = options.skip;

    let mut hooks = ModelServiceHooks::default();
    let capability_llm = options.capability_llm.clone();
    hooks.get_llm_for_capability = Some(Arc::new(move |_cap: &str| capability_llm.clone()));
    let large_llm = options.large_llm.clone();
    hooks.get_large_llm = Some(Arc::new(move || large_llm.clone()));
    let pinned = options.pinned;
    hooks.pinned_model = Some(Arc::new(move |_cap: Option<&str>| {
        pinned.then(|| ws_model("chat-model", true, true))
    }));
    let pinned_llm = options.pinned_llm.clone();
    hooks.pinned_llm = Some(Arc::new(move |_cap: Option<&str>| pinned_llm.clone()));
    svc.set_hooks(hooks);
    svc
}

type Sink = Arc<Mutex<Vec<(String, String, String)>>>;

fn events(svc: &ModelService) -> Sink {
    let sink: Sink = Arc::new(Mutex::new(Vec::new()));
    let recorder = sink.clone();
    let callback: ProgressCallback = Arc::new(move |step: &str, status: &str, message: &str| {
        recorder.lock().push((step.to_string(), status.to_string(), message.to_string()));
    });
    svc.begin_response_request(Some(callback));
    sink
}

fn messages(sink: &Sink) -> Vec<String> {
    sink.lock().iter().map(|(_, _, message)| message.clone()).collect()
}

// ---------------------------------------------------- the load is announced first

#[test]
fn test_a_pending_capability_load_warns_before_it_blocks() {
    let svc = service(Options { selection: Selection::Available, ..Options::default() });
    let sink = events(&svc);

    svc.announce_response_model(Some(GENERAL));

    assert_eq!(messages(&sink), vec!["Loading model for the response…"]);
}

#[test]
fn test_nothing_is_announced_when_no_model_provides_it() {
    let svc = service(Options { selection: Selection::Unavailable, ..Options::default() });
    let sink = events(&svc);

    svc.announce_response_model(Some(GENERAL));

    assert!(messages(&sink).is_empty());
}

/// A pin loads whatever the request asked for, capability or not.
#[test]
fn test_a_pinned_model_is_announced_even_without_a_capability() {
    let svc = service(Options { selection: Selection::Pinned, pinned: true, ..Options::default() });
    let sink = events(&svc);

    svc.announce_response_model(None);

    assert_eq!(messages(&sink), vec!["Loading model for the response…"]);
}

#[test]
fn test_plain_chat_announces_nothing() {
    let svc = service(Options { selection: Selection::Unavailable, ..Options::default() });
    let sink = events(&svc);

    svc.announce_response_model(None);

    assert!(messages(&sink).is_empty());
}

// ------------------------------------------------------- the chosen model is named

#[test]
fn test_a_capability_model_is_named() {
    let svc = service(Options { capability_llm: Some(llm("Gemma 3n E2B")), ..Options::default() });
    let sink = events(&svc);

    svc.llm_for_response(Some(GENERAL));

    assert_eq!(messages(&sink), vec!["Selected Gemma 3n E2B for the response"]);
}

#[test]
fn test_the_large_model_fallback_is_named() {
    let svc = service(Options { capability_llm: None, large_llm: Some(llm("Ministral 3B")), ..Options::default() });
    let sink = events(&svc);

    svc.llm_for_response(Some(GENERAL));

    assert_eq!(messages(&sink), vec!["Selected Ministral 3B for the response"]);
}

#[test]
fn test_a_pinned_model_is_named() {
    let svc = service(Options { pinned: true, pinned_llm: Some(llm("Gemma 3n E2B")), ..Options::default() });
    let sink = events(&svc);

    svc.llm_for_response(None);

    assert_eq!(messages(&sink), vec!["Selected Gemma 3n E2B for the response"]);
}

#[test]
fn test_the_capability_tag_never_reaches_the_user() {
    let svc = service(Options { capability_llm: Some(llm("Gemma 3n E2B")), ..Options::default() });
    let sink = events(&svc);

    svc.llm_for_response(Some(GENERAL));

    assert!(!messages(&sink).iter().any(|m| m.to_lowercase().contains("general")));
}

// --------------------------------------------------------- falling back says why

#[test]
fn test_memory_is_named() {
    let svc = service(Options {
        capability_llm: None,
        skip: Some(CapabilitySkipReason::new("gemma4-e2b", "memory")),
        ..Options::default()
    });
    let sink = events(&svc);

    svc.llm_for_response(Some(GENERAL));

    assert_eq!(messages(&sink), vec!["Selected sl-mini — not enough memory for gemma4-e2b"]);
}

#[test]
fn test_a_failed_pin_also_falls_back_with_a_reason() {
    let svc = service(Options {
        pinned: true,
        pinned_llm: None,
        skip: Some(CapabilitySkipReason::new("gemma4-e2b", "memory")),
        ..Options::default()
    });
    let sink = events(&svc);

    svc.llm_for_response(Some(GENERAL));

    assert!(messages(&sink)[0].contains("not enough memory"));
}

/// Nothing loaded and nothing overridden — there is nothing to report.
#[test]
fn test_plain_chat_on_the_resident_model_says_nothing() {
    let svc = service(Options::default());
    let sink = events(&svc);

    svc.llm_for_response(None);

    assert!(messages(&sink).is_empty());
}

// ------------------------------------------------------------ per-request state

#[test]
fn test_each_request_reports_its_model_again() {
    let svc = service(Options { capability_llm: Some(llm("Gemma 3n E2B")), ..Options::default() });

    let first = events(&svc);
    svc.llm_for_response(Some(GENERAL));
    let second = events(&svc);
    svc.llm_for_response(Some(GENERAL));

    assert_eq!(messages(&first), vec!["Selected Gemma 3n E2B for the response"]);
    assert_eq!(messages(&second), vec!["Selected Gemma 3n E2B for the response"]);
}

/// stream_completion resolves the model again after sizing the window.
#[test]
fn test_one_request_reports_once() {
    let svc = service(Options { capability_llm: Some(llm("Gemma 3n E2B")), ..Options::default() });
    let sink = events(&svc);

    svc.llm_for_response(Some(GENERAL));
    svc.llm_for_response(Some(GENERAL));

    assert_eq!(messages(&sink), vec!["Selected Gemma 3n E2B for the response"]);
}

#[test]
fn test_the_router_tracks_a_model_switch() {
    let svc = service(Options { capability_llm: None, ..Options::default() });
    events(&svc);
    let switched = llm("Ministral 3B");
    *svc.llm.write() = Some(switched.clone());

    svc.llm_for_response(Some(GENERAL));

    assert!(same(&svc.response_router().orchestrator_llm, &switched));
}

/// A log line naming `ModelCapability::General` is one nobody can grep for.
#[test]
fn test_capability_names_are_logged_by_value_not_member() {
    assert_eq!(ModelService::cap_name(ModelCapability::General.value()), "general");
    assert_eq!(ModelService::cap_name("dp"), "dp");
    assert_eq!(ModelService::cap_name_opt(None), "None");
}
