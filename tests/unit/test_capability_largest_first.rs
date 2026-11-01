//! A routed step runs on the largest selected model that declares its
//! capability.
//!
//! When two selected models can both do the step, the bigger one gives the
//! better answer.  A smaller capable model is used only when the larger ones
//! cannot be loaded (no GGUF, faulted, or it does not fit).
//!
//! The Python test faked `_rank` on mocks.  A workspace model's rank comes
//! from its model spec, so the walk uses real built-in models whose sizes
//! give the same big/mid/small spread (gemma4-e4b 4B, ministral3-3b 3B,
//! gemma4-e2b 2B); the tie-break test ranks plain candidates through the
//! [`RankedModel`] seam the sort is written against.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use smartloop::api::services::model_service::{ModelService, ModelServiceHooks, RankedModel};
use smartloop::llms::base_llm::BaseLLM;
use smartloop::llms::cloud_llm::local_only_scope;
use smartloop::model::spec::MODEL_SPECS;
use smartloop::types::ModelKind;
use smartloop::workspace::{Workspace, WorkspaceModel};

use crate::shared::model_service_a::{FakeLLM, attach_workspace, bare_service, same};

const BIG: &str = "gemma4-e4b";
const MID: &str = "ministral3-3b";
const SMALL: &str = "gemma4-e2b";

fn ws_model(name: &str) -> WorkspaceModel {
    WorkspaceModel { model_name: name.to_string(), model_id: name.to_string(), ..Default::default() }
}

/// A candidate with an explicit rank (Python's `_ws_model(name, size, version, caps)`).
struct Candidate {
    name: &'static str,
    size: f64,
    version: i64,
    caps: Vec<String>,
}

fn candidate(name: &'static str, size: f64, version: i64, caps: &[&str]) -> Candidate {
    Candidate { name, size, version, caps: caps.iter().map(|c| c.to_string()).collect() }
}

impl RankedModel for Candidate {
    fn model_rank(&self) -> (f64, i64) {
        (self.size, self.version)
    }

    fn model_capabilities(&self) -> Vec<String> {
        self.caps.clone()
    }
}

/// How a fake candidate loads: as things stand (resident, or fits), or only
/// once prior-step models are evicted.
#[derive(Clone, Copy, PartialEq)]
enum Fit {
    AsIs,
    AfterEviction,
}

/// The candidates tried, in order (with whether eviction was allowed), and
/// what was returned.
fn walk(
    ranked: Vec<WorkspaceModel>,
    loads: HashMap<&'static str, (Arc<dyn BaseLLM>, Fit)>,
) -> (Vec<(String, bool)>, Option<Arc<dyn BaseLLM>>) {
    let tried: Arc<Mutex<Vec<(String, bool)>>> = Arc::default();
    let svc = bare_service();
    attach_workspace(&svc, Workspace::new("ws"));

    let tried_in_hook = tried.clone();
    let loads: HashMap<String, (Arc<dyn BaseLLM>, Fit)> =
        loads.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    svc.set_hooks(ModelServiceHooks {
        pinned_model: Some(Arc::new(|_| None)),
        ranked_models_for: Some(Arc::new(move |_| ranked.clone())),
        load_workspace_llm: Some(Arc::new(move |ws_model: &WorkspaceModel, _reason: &str, _reserve, evict| {
            tried_in_hook.lock().push((ws_model.model_name.clone(), evict));
            match loads.get(&ws_model.model_name) {
                Some((llm, Fit::AsIs)) => Some(llm.clone()),
                Some((llm, Fit::AfterEviction)) if evict => Some(llm.clone()),
                _ => None,
            }
        })),
        ..Default::default()
    });
    let llm = svc.get_llm_for_capability("reasoning");
    let tried = tried.lock().clone();
    (tried, llm)
}

fn attempt(name: &str, evict: bool) -> (String, bool) {
    (name.to_string(), evict)
}

#[test]
fn test_the_largest_capable_model_is_used() {
    let big = FakeLLM::named("big_llm");
    let mut loads = HashMap::new();
    loads.insert(BIG, (big.clone(), Fit::AsIs));
    loads.insert(MID, (FakeLLM::named("mid"), Fit::AsIs));
    loads.insert(SMALL, (FakeLLM::named("small"), Fit::AsIs));
    let (tried, llm) = walk(vec![ws_model(SMALL), ws_model(MID), ws_model(BIG)], loads);
    assert_eq!(tried, vec![attempt(BIG, false)]);
    assert!(same(&llm.expect("a model"), &big));
}

#[test]
fn test_a_smaller_one_is_used_when_the_larger_cannot_load() {
    let mid = FakeLLM::named("mid_llm");
    let mut loads = HashMap::new();
    loads.insert(MID, (mid.clone(), Fit::AsIs));
    let (tried, llm) = walk(vec![ws_model(SMALL), ws_model(MID), ws_model(BIG)], loads);
    assert_eq!(tried, vec![attempt(BIG, false), attempt(MID, false)]);
    assert!(same(&llm.expect("a model"), &mid));
}

#[test]
fn test_ministral_3b_outranks_qwen3_5_2b() {
    let ministral = FakeLLM::named("ministral");
    let mut loads = HashMap::new();
    loads.insert("ministral3-3b", (ministral.clone(), Fit::AsIs));
    loads.insert("qwen3.5-2b", (FakeLLM::named("qwen"), Fit::AsIs));
    let (tried, llm) = walk(vec![ws_model("qwen3.5-2b"), ws_model("ministral3-3b")], loads);
    assert_eq!(tried, vec![attempt("ministral3-3b", false)]);
    assert!(same(&llm.expect("a model"), &ministral));
}

/// The regression: Qwen 3.5 2B was resident, Ministral 3B did not fit, and
/// evicting Qwen to try Ministral left the step on the bootstrap model.  A
/// bigger model that does not fit as things stand must not cost the
/// resident one.
#[test]
fn test_a_resident_smaller_model_is_kept_over_a_bigger_one_that_does_not_fit() {
    let qwen = FakeLLM::named("qwen");
    let mut loads = HashMap::new();
    loads.insert("ministral3-3b", (FakeLLM::named("ministral"), Fit::AfterEviction));
    loads.insert("qwen3.5-2b", (qwen.clone(), Fit::AsIs));
    let (tried, llm) = walk(vec![ws_model("qwen3.5-2b"), ws_model("ministral3-3b")], loads);
    assert_eq!(tried, vec![attempt("ministral3-3b", false), attempt("qwen3.5-2b", false)]);
    assert!(same(&llm.expect("a model"), &qwen));
}

#[test]
fn test_eviction_is_tried_only_when_nothing_fits_as_things_stand() {
    let ministral = FakeLLM::named("ministral");
    let mut loads = HashMap::new();
    loads.insert("ministral3-3b", (ministral.clone(), Fit::AfterEviction));
    let (tried, llm) = walk(vec![ws_model("qwen3.5-2b"), ws_model("ministral3-3b")], loads);
    assert_eq!(
        tried,
        vec![attempt("ministral3-3b", false), attempt("qwen3.5-2b", false), attempt("ministral3-3b", true)]
    );
    assert!(same(&llm.expect("a model"), &ministral));
}

#[test]
fn test_size_ties_go_to_broader_coverage_then_the_newer_version() {
    let ordered = ModelService::ranked_candidates(
        vec![
            candidate("old", 4.0, 1, &["reasoning"]),
            candidate("narrow-new", 4.0, 3, &["reasoning"]),
            candidate("broad", 4.0, 2, &["reasoning", "vision"]),
            candidate("new", 4.0, 2, &["reasoning"]),
        ],
        &["reasoning", "vision"],
        true,
    );
    let names: Vec<&str> = ordered.iter().map(|m| m.name).collect();
    assert_eq!(names, vec!["broad", "narrow-new", "new", "old"]);
}

fn remote_model(name: &str) -> WorkspaceModel {
    WorkspaceModel { model_name: name.to_string(), kind: ModelKind::Remote, ..Default::default() }
}

/// A local model that fits only once earlier steps' models are evicted loses
/// to a remote one: the remote model answers before anything is evicted.
#[test]
fn test_a_remote_model_answers_before_a_local_one_is_evicted_for() {
    let kimi = FakeLLM::named("kimi");
    let mut loads = HashMap::new();
    loads.insert(BIG, (FakeLLM::named("big"), Fit::AfterEviction));
    loads.insert("kimi-k2", (kimi.clone(), Fit::AsIs));
    let (tried, llm) = walk(vec![ws_model(BIG), remote_model("kimi-k2")], loads);
    assert_eq!(tried, vec![attempt(BIG, false), attempt("kimi-k2", false)]);
    assert!(same(&llm.expect("a model"), &kimi));
}

/// Decisions resolve local-only: the remote model is not a candidate, so the
/// local one is loaded, evicting if it must.
#[test]
fn test_a_local_only_resolution_skips_remote_models() {
    let big = FakeLLM::named("big");
    let mut loads = HashMap::new();
    loads.insert(BIG, (big.clone(), Fit::AfterEviction));
    loads.insert("kimi-k2", (FakeLLM::named("kimi"), Fit::AsIs));
    let (tried, llm) = {
        let _scope = local_only_scope(true);
        walk(vec![ws_model(BIG), remote_model("kimi-k2")], loads)
    };
    assert_eq!(tried, vec![attempt(BIG, false), attempt(BIG, true)]);
    assert!(same(&llm.expect("a model"), &big));
}

/// Publish a mixture-of-experts remote model's sizes to the (process-wide)
/// spec registry, under a name only these tests use.
fn sized_remote(name: &str, total_b: f64, active_b: f64) -> WorkspaceModel {
    let serde_json::Value::Object(row) = serde_json::json!({
        "name": name, "type": "remote", "capabilities": ["reasoning"],
        "param_size_b": total_b, "active_param_size_b": active_b,
    }) else {
        unreachable!()
    };
    MODEL_SPECS.hydrate(&[row]);
    remote_model(name)
}

/// With two remote models selected, the one with more active parameters
/// does the step — not whichever registered first, and not by total size.
#[test]
fn test_the_remote_model_with_more_active_params_is_used() {
    let light = sized_remote("walk-moe-light", 900.0, 23.0);
    let heavy = sized_remote("walk-moe-heavy", 753.0, 40.0);
    let heavy_llm = FakeLLM::named("heavy");
    let mut loads = HashMap::new();
    loads.insert("walk-moe-light", (FakeLLM::named("light"), Fit::AsIs));
    loads.insert("walk-moe-heavy", (heavy_llm.clone(), Fit::AsIs));
    let (tried, llm) = walk(vec![light, heavy], loads);
    assert_eq!(tried, vec![attempt("walk-moe-heavy", false)]);
    assert!(same(&llm.expect("a model"), &heavy_llm));
}

/// A remote model well beyond a local one's size does the step: quality
/// first, even though the local model fits.
#[test]
fn test_a_much_larger_remote_model_ranks_before_a_local_one() {
    let remote = sized_remote("walk-moe-huge", 753.0, 40.0);
    let huge = FakeLLM::named("huge");
    let mut loads = HashMap::new();
    loads.insert(BIG, (FakeLLM::named("big"), Fit::AsIs));
    loads.insert("walk-moe-huge", (huge.clone(), Fit::AsIs));
    let (tried, llm) = walk(vec![ws_model(BIG), remote], loads);
    assert_eq!(tried, vec![attempt("walk-moe-huge", false)]);
    assert!(same(&llm.expect("a model"), &huge));
}

/// Between similar models — the remote one within the local margin — the
/// local one goes first: a 4B on device over a 4.5B-active cloud model.
#[test]
fn test_a_similar_remote_model_ranks_after_a_local_one() {
    let remote = sized_remote("walk-moe-similar", 30.0, 4.5);
    let big = FakeLLM::named("big");
    let mut loads = HashMap::new();
    loads.insert(BIG, (big.clone(), Fit::AsIs));
    loads.insert("walk-moe-similar", (FakeLLM::named("similar"), Fit::AsIs));
    let (tried, llm) = walk(vec![remote, ws_model(BIG)], loads);
    assert_eq!(tried, vec![attempt(BIG, false)]);
    assert!(same(&llm.expect("a model"), &big));
}

/// The bootstrap model is the last resort: a remote model that declares the
/// capability does the step even though the bootstrap model fits.
#[test]
fn test_the_bootstrap_model_ranks_after_remote_models() {
    let bootstrap = smartloop::api::config::api_settings().default_model;
    let kimi = FakeLLM::named("kimi");
    let mut loads = HashMap::new();
    loads.insert("kimi-k2", (kimi.clone(), Fit::AsIs));
    let mut ranked = vec![ws_model(&bootstrap), remote_model("kimi-k2")];
    ranked[0].model_name = bootstrap.clone();
    let (tried, llm) = walk(ranked, loads);
    assert_eq!(tried, vec![attempt("kimi-k2", false)]);
    assert!(same(&llm.expect("a model"), &kimi));
}
