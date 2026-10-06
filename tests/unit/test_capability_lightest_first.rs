//! Port of `tests/unit/test_capability_lightest_first.py`.
//!
//! A routed step runs on the lightest model that declares its capability.
//!
//! Every candidate can do the step, so the smallest is enough — and it loads
//! sooner, decodes faster and leaves the most room for the next step.  A
//! larger capable model is used only when the smaller ones cannot be loaded.
//!
//! The Python test faked `_rank` on mocks.  A workspace model's rank comes
//! from its model class, so the walk uses real built-in models whose sizes
//! give the same big/mid/small spread (gemma4-e4b 4B, ministral3-3b 3B,
//! gemma4-e2b 2B); the tie-break test ranks plain candidates through the
//! [`RankedModel`] seam the sort is written against.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use smartloop::api::services::model_service::{ModelService, ModelServiceHooks, RankedModel};
use smartloop::llms::base_llm::BaseLLM;
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

/// The candidates tried, in order, and what was returned.
fn walk(
    ranked: Vec<WorkspaceModel>,
    loads: HashMap<&'static str, Arc<dyn BaseLLM>>,
) -> (Vec<String>, Option<Arc<dyn BaseLLM>>) {
    let tried: Arc<Mutex<Vec<String>>> = Arc::default();
    let svc = bare_service();
    attach_workspace(&svc, Workspace::new("ws"));

    let tried_in_hook = tried.clone();
    let loads: HashMap<String, Arc<dyn BaseLLM>> = loads.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    svc.set_hooks(ModelServiceHooks {
        pinned_model: Some(Arc::new(|_| None)),
        ranked_models_for: Some(Arc::new(move |_| ranked.clone())),
        load_workspace_llm: Some(Arc::new(move |ws_model: &WorkspaceModel, _reason: &str, _reserve| {
            tried_in_hook.lock().push(ws_model.model_name.clone());
            loads.get(&ws_model.model_name).cloned()
        })),
        ..Default::default()
    });
    let llm = svc.get_llm_for_capability("reasoning");
    let tried = tried.lock().clone();
    (tried, llm)
}

#[test]
fn test_the_smallest_capable_model_is_used() {
    let small = FakeLLM::named("small_llm");
    let mut loads = HashMap::new();
    loads.insert(BIG, FakeLLM::named("big"));
    loads.insert(MID, FakeLLM::named("mid"));
    loads.insert(SMALL, small.clone());
    let (tried, llm) = walk(vec![ws_model(BIG), ws_model(MID), ws_model(SMALL)], loads);
    assert_eq!(tried, vec![SMALL.to_string()]);
    assert!(same(&llm.expect("a model"), &small));
}

#[test]
fn test_a_larger_one_is_used_when_the_smaller_cannot_load() {
    let mid = FakeLLM::named("mid_llm");
    let mut loads = HashMap::new();
    loads.insert(MID, mid.clone());
    let (tried, llm) = walk(vec![ws_model(BIG), ws_model(MID), ws_model(SMALL)], loads);
    assert_eq!(tried, vec![SMALL.to_string(), MID.to_string()]);
    assert!(same(&llm.expect("a model"), &mid));
}

#[test]
fn test_size_ties_go_to_broader_coverage_then_the_newer_version() {
    let ordered = ModelService::lightest_first(
        vec![
            candidate("old", 4.0, 1, &["reasoning"]),
            candidate("narrow-new", 4.0, 3, &["reasoning"]),
            candidate("broad", 4.0, 2, &["reasoning", "general"]),
            candidate("new", 4.0, 2, &["reasoning"]),
        ],
        &["reasoning", "general"],
    );
    let names: Vec<&str> = ordered.iter().map(|m| m.name).collect();
    assert_eq!(names, vec!["broad", "narrow-new", "new", "old"]);
}
