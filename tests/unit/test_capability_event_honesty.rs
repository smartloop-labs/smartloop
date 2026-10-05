//! Port of `tests/unit/test_capability_event_honesty.py`.
//!
//! Model-selection events must describe what actually happened.
//!
//! "Loading model for reasoning…" fires *before* the provider is consulted —
//! that is the point of it, since the load it precedes is slow.  But on a
//! workspace where no selected model declares the capability there is nothing
//! to load, so the event promised a load that never happened and the step then
//! fell back to the bootstrap model, which was in turn announced as "Selected
//! sl-mini for reasoning" — naming a capability nothing had been selected for.

use std::sync::Arc;

use smartloop::agents::mcp::explorer::Explorer;
use smartloop::agents::orchestration::model_router::{ModelRouter, ModelRouterOptions};
use smartloop::agents::orchestration::skill_processor::SkillProcessor;
use smartloop::agents::planner::ModelTier;
use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::llms::base_llm::BaseLLM;
use smartloop::types::ModelCapability;
use smartloop::workspace::{Project, Workspace, WorkspaceModel};

use crate::shared::model_service_a::{FakeLLM, ProgressRecorder, attach_workspace, bare_service, same};

const REASONING: &str = "reasoning";

fn llm(name: &str) -> Arc<dyn BaseLLM> {
    FakeLLM::with_metadata(name)
}

fn router(capability_llm: Option<Arc<dyn BaseLLM>>, available: bool) -> (ModelRouter, ProgressRecorder) {
    let router = ModelRouter::new(
        llm("Slp Orchestra Mini"),
        llm("Slp Orchestra Mini"),
        ModelRouterOptions {
            capability_llm_provider: Some(Arc::new(move |_cap: &str| Ok(capability_llm.clone()))),
            capability_available: Some(Arc::new(move |_cap: &str| Ok(available))),
            ..Default::default()
        },
    );
    let recorder = ProgressRecorder::new();
    router.begin_request(Some(recorder.callback()));
    (router, recorder)
}

#[test]
fn test_no_load_is_announced_when_nothing_provides_the_capability() {
    let (router, rec) = router(None, false);

    router.llm_for_tier(ModelTier::Large, Some(REASONING), None);

    assert!(!rec.messages().iter().any(|m| m.contains("Loading model for reasoning")));
}

/// The user should know which model answered, just not a false capability.
#[test]
fn test_the_fallback_model_is_still_named() {
    let (router, rec) = router(None, false);

    router.llm_for_tier(ModelTier::Large, Some(REASONING), None);

    assert_eq!(rec.messages(), vec!["Selected sl-mini"]);
}

#[test]
fn test_the_fallback_does_not_claim_the_capability() {
    let (router, rec) = router(None, false);

    router.llm_for_tier(ModelTier::Large, Some(REASONING), None);

    assert!(!rec.messages().iter().any(|m| m.contains("for reasoning")));
}

#[test]
fn test_a_real_capability_model_is_announced_as_before() {
    let cap_llm = llm("Gemma 3n E2B");
    let (router, rec) = router(Some(cap_llm.clone()), true);

    let resolved = router.llm_for_tier(ModelTier::Large, Some(REASONING), None);

    assert!(same(&resolved, &cap_llm));
    assert_eq!(
        rec.messages(),
        vec!["Loading model for reasoning…", "Selected Gemma 3n E2B for reasoning"]
    );
}

/// Availability is declared, not guaranteed — a fit failure still falls back.
#[test]
fn test_an_available_capability_that_fails_to_load_still_warns_first() {
    let (router, rec) = router(None, true);

    router.llm_for_tier(ModelTier::Large, Some(REASONING), None);

    assert_eq!(rec.messages(), vec!["Loading model for reasoning…", "Selected sl-mini"]);
}

#[test]
fn test_no_predicate_wired_in_keeps_the_previous_behaviour() {
    let router = ModelRouter::new(
        llm("Slp Orchestra Mini"),
        llm("Slp Orchestra Mini"),
        ModelRouterOptions {
            capability_llm_provider: Some(Arc::new(|_cap: &str| Ok(None))),
            ..Default::default()
        },
    );
    let rec = ProgressRecorder::new();
    router.begin_request(Some(rec.callback()));

    router.llm_for_tier(ModelTier::Large, Some(REASONING), None);

    assert!(rec.messages().iter().any(|m| m == "Loading model for reasoning…"));
}

#[test]
fn test_a_broken_predicate_does_not_suppress_the_event() {
    let router = ModelRouter::new(
        llm("Slp Orchestra Mini"),
        llm("Slp Orchestra Mini"),
        ModelRouterOptions {
            capability_llm_provider: Some(Arc::new(|_cap: &str| Ok(None))),
            capability_available: Some(Arc::new(|_cap: &str| Err(anyhow::anyhow!("boom")))),
            ..Default::default()
        },
    );
    let rec = ProgressRecorder::new();
    router.begin_request(Some(rec.callback()));

    router.llm_for_tier(ModelTier::Large, Some(REASONING), None);

    assert!(rec.messages().iter().any(|m| m == "Loading model for reasoning…"));
}

/// A fan-out resolves the model per document; the user needs one line.
#[test]
fn test_each_event_is_reported_once_per_request() {
    let (router, rec) = router(None, false);

    router.llm_for_tier(ModelTier::Large, Some(REASONING), None);
    router.llm_for_tier(ModelTier::Large, Some(REASONING), None);

    assert_eq!(rec.messages(), vec!["Selected sl-mini"]);
}

// ── TestHasCapabilityModel ────────────────────────────────────────────────

fn has_capability_service(ranked: Vec<WorkspaceModel>, pinned: Option<WorkspaceModel>) -> Arc<ModelService> {
    let svc = bare_service();
    let mut workspace = Workspace::new("ws");
    workspace.projects = vec![Project { id: "p1".to_string(), current: true, ..Default::default() }];
    attach_workspace(&svc, workspace);
    svc.set_hooks(ModelServiceHooks {
        ranked_models_for: Some(Arc::new(move |_| ranked.clone())),
        pinned_model: Some(Arc::new(move |_| pinned.clone())),
        ..Default::default()
    });
    svc
}

#[test]
fn test_true_when_a_model_declares_it() {
    assert!(has_capability_service(vec![WorkspaceModel::default()], None).has_capability_model(REASONING));
}

#[test]
fn test_false_when_none_declares_it() {
    assert!(!has_capability_service(vec![], None).has_capability_model(REASONING));
}

#[test]
fn test_a_pinned_model_serves_every_capability() {
    let svc = has_capability_service(vec![], Some(WorkspaceModel::default()));
    assert!(svc.has_capability_model(REASONING));
}

#[test]
fn test_false_without_a_workspace() {
    let svc = bare_service();
    *svc.model.write() = None;
    assert!(!svc.has_capability_model(REASONING));
}

// ── the event names the work, not the routing tag ─────────────────────────
//
// Synthesis routes on the 'reasoning' capability so a capable model writes the
// answer.  But the user is watching the final answer being written, and
// "Loading Model For Reasoning…" under "Synthesizing the final answer…" reads
// as a second, unrelated step.

#[test]
fn test_loading_names_the_purpose() {
    let (router, rec) = router(Some(llm("Gemma 3n E2B")), true);

    router.llm_for_tier(ModelTier::Large, Some(REASONING), Some("the final answer"));

    assert_eq!(
        rec.messages(),
        vec!["Loading model for the final answer…", "Selected Gemma 3n E2B for the final answer"]
    );
}

#[test]
fn test_the_capability_tag_never_reaches_the_user() {
    let (router, rec) = router(Some(llm("Gemma 3n E2B")), true);

    router.llm_for_tier(ModelTier::Large, Some(REASONING), Some("the final answer"));

    assert!(!rec.messages().iter().any(|m| {
        let lower = m.to_lowercase();
        ["reasoning", "skills"].iter().any(|tag| lower.contains(tag))
    }));
}

#[test]
fn test_the_fallback_still_drops_the_capability_claim() {
    let (router, rec) = router(None, true);

    router.llm_for_tier(ModelTier::Large, Some(REASONING), Some("the final answer"));

    assert_eq!(rec.messages(), vec!["Loading model for the final answer…", "Selected sl-mini"]);
}

/// Dedupe is per purpose, not per capability — they are separate jobs.
#[test]
fn test_two_purposes_on_one_capability_each_get_a_line() {
    let (router, rec) = router(Some(llm("Gemma 3n E2B")), true);

    router.llm_for_tier(ModelTier::Large, Some(REASONING), None);
    router.llm_for_tier(ModelTier::Large, Some(REASONING), Some("the final answer"));

    assert_eq!(
        rec.messages(),
        vec![
            "Loading model for reasoning…",
            "Selected Gemma 3n E2B for reasoning",
            "Loading model for the final answer…",
            "Selected Gemma 3n E2B for the final answer",
        ]
    );
}

#[test]
fn test_the_same_purpose_is_still_reported_once() {
    let (router, rec) = router(Some(llm("Gemma 3n E2B")), true);

    for _ in 0..2 {
        router.llm_for_tier(ModelTier::Large, Some(REASONING), Some("the final answer"));
    }

    assert_eq!(
        rec.messages(),
        vec!["Loading model for the final answer…", "Selected Gemma 3n E2B for the final answer"]
    );
}

#[test]
fn test_no_purpose_keeps_the_capability_label() {
    let (router, rec) = router(Some(llm("Gemma 3n E2B")), true);

    router.llm_for_tier(ModelTier::Large, Some(ModelCapability::Dp.value()), None);

    assert_eq!(
        rec.messages(),
        vec!["Loading model for document processing…", "Selected Gemma 3n E2B for document processing"]
    );
}

/// End to end through `SkillProcessor::synthesize` — the reported case.
#[test]
fn test_synthesis_reports_the_final_answer_not_skills() {
    let (router, rec) = router(Some(llm("Gemma 3n E2B")), true);
    let router = Arc::new(router);
    let explorer = Arc::new(Explorer::new(FakeLLM::named("explorer"), None, None, false));
    let processor = SkillProcessor::new(router, explorer);

    processor.synthesize("what is this", "step output", "skill body");

    let messages = rec.messages();
    assert!(!messages.iter().any(|m| {
        let lower = m.to_lowercase();
        ["reasoning", "skills"].iter().any(|tag| lower.contains(tag))
    }));
    assert!(messages.iter().any(|m| m == "Loading model for the final answer…"));
}
