//! Port of `tests/unit/test_resolution_model_loads_like_vision.py`.
//!
//! Tool resolution loads its model on the same terms as image OCR.
//!
//! MCPAgent's resolve call is the one step whose judgement is worth a
//! reasoning block: it picks the tool *and* fills its arguments in a single
//! pass, and it runs with `thinking=true` whatever the picker's per-model
//! setting says (the monologue is stripped before anything reads the JSON).
//! Which model runs it is capability routing's job —
//! `ModelCapability::Thinking` — exactly as `Vision` decides which model reads
//! an image.
//!
//! The fit guard is where the two used to differ.  `get_vision_llm` waives the
//! headroom kept for the main model (`reserve_gb=0.0`), so an image step gets
//! its model; the thinking capability kept the reserve, so on a memory-tight
//! device the tool call quietly resolved on the small router model instead —
//! the outcome the routing exists to avoid.  There is nothing for the reserve
//! to protect here: the alternative to loading is not a smaller reasoning
//! model, it is no reasoning at all.
//!
//! The Python test patched `_ranked_candidates` / `_pinned_model` to hand the
//! resolver its candidates; here the same candidates come from a workspace
//! whose selection produces them (two selected models → no pin, one → pinned),
//! and the GGUF lookup, the fit guard and the factory are the service's seams.

use std::any::Any;
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationStream};
use smartloop::model::model_manager::Model;
use smartloop::types::ModelCapability;
use smartloop::workspace::workspace::{Workspace, WorkspaceModel};

/// A loaded model stand-in (`MagicMock`).
struct FakeLlm(&'static str);

impl BaseLLM for FakeLlm {
    fn project_id(&self) -> &str {
        ""
    }
    fn get_num_tokens(&self, text: &str) -> usize {
        text.len() / 4
    }
    fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        messages
    }
    fn get_max_tokens(&self) -> usize {
        4096
    }
    fn call(&self, _prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
        Ok(String::new())
    }
    fn stream<'a>(&'a self, _prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
        Box::new(std::iter::empty())
    }
    fn cancelled(&self) -> Option<CancelEvent> {
        None
    }
    fn set_cancelled(&self, _event: Option<CancelEvent>) {}
    fn llm_type(&self) -> &str {
        self.0
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `_ws_model(name)`: a selected (enabled + downloaded) workspace model.
fn ws_model(name: &str) -> WorkspaceModel {
    WorkspaceModel {
        model_name: name.to_string(),
        model_id: name.to_string(),
        downloaded: true,
        enabled: true,
        ..WorkspaceModel::default()
    }
}

/// The `svc` fixture: a bare service with the resident orchestrator loaded,
/// no large model, and a workspace (no current project) selecting `selected`.
fn svc(selected: &[&str]) -> Arc<ModelService> {
    let s = ModelService::new_bare();
    let resident: Arc<dyn BaseLLM> = Arc::new(FakeLlm("resident_orchestrator"));
    *s.llm.write() = Some(resident);
    *s.model_name.write() = Some("sl-mini".to_string());
    *s.large_llm.write() = None;
    *s.large_model_name.write() = None;

    let mut workspace = Workspace::new("ws");
    workspace.models_dir = "/models".to_string();
    workspace.models = selected.iter().map(|n| ws_model(n)).collect();
    let model = Model::new("m", Arc::new(RwLock::new(workspace)), None);
    *s.model.write() = Some(Arc::new(model));
    s
}

/// What the seams saw during one resolution.
#[derive(Default)]
struct Recorded {
    /// `reserve_gb` of every fit-guard call.
    fits: Mutex<Vec<Option<f64>>>,
    /// `(gguf_path, model_name)` of every factory call.
    factory: Mutex<Vec<(String, String)>>,
}

/// Run the real resolution and report how the fit guard and factory were called.
fn resolve(
    svc: &ModelService,
    capability: &str,
    fits_result: bool,
) -> (Option<Arc<dyn BaseLLM>>, Arc<dyn BaseLLM>, Arc<Recorded>) {
    let recorded = Arc::new(Recorded::default());
    let loaded: Arc<dyn BaseLLM> = Arc::new(FakeLlm("factory_return_value"));

    let fits_rec = recorded.clone();
    let factory_rec = recorded.clone();
    let factory_llm = loaded.clone();
    svc.set_hooks(ModelServiceHooks {
        find_model_ggufs: Some(Arc::new(|_dir: &str| (Some("/models/x.gguf".to_string()), None))),
        large_model_fits: Some(Arc::new(move |_path: &str, _cap: i64, _label: &str, reserve_gb: Option<f64>| {
            fits_rec.fits.lock().push(reserve_gb);
            fits_result
        })),
        load_llama_instance: Some(Arc::new(
            move |path: &str, model_name: &str, _cap: i64, _thinking: Option<bool>| {
                factory_rec.factory.lock().push((path.to_string(), model_name.to_string()));
                Ok(factory_llm.clone())
            },
        )),
        ..ModelServiceHooks::default()
    });

    let llm = svc.get_llm_for_capability(capability);
    (llm, loaded, recorded)
}

fn same_llm(a: &Arc<dyn BaseLLM>, b: &Arc<dyn BaseLLM>) -> bool {
    std::ptr::addr_eq(Arc::as_ptr(a), Arc::as_ptr(b))
}

mod the_reasoning_model_loads_without_the_reserve {
    use super::*;

    /// Two models selected: the one declaring THINKING runs the step.
    #[test]
    fn test_a_selected_thinking_model_is_loaded_for_the_resolve_step() {
        let svc = svc(&["qwen3.5-2b", "gemma3-4b"]);
        let (llm, factory_return, recorded) = resolve(&svc, ModelCapability::Thinking.value(), true);

        let llm = llm.expect("the thinking model was loaded");
        assert!(same_llm(&llm, &factory_return));
        let calls = recorded.factory.lock();
        assert_eq!(calls.last().expect("factory called").1, "qwen3.5-2b");
    }

    #[test]
    fn test_the_fit_guard_waives_the_reserve() {
        let svc = svc(&["qwen3.5-2b", "gemma3-4b"]);
        let (_, _, recorded) = resolve(&svc, ModelCapability::Thinking.value(), true);

        assert_eq!(recorded.fits.lock().last().copied(), Some(Some(0.0)));
    }

    /// One selection is pinned to every capability it can serve, and the
    /// resolve step is one of them.
    #[test]
    fn test_the_sole_selected_model_gets_the_same_treatment() {
        let svc = svc(&["qwen3.5-2b"]);
        let (_, _, recorded) = resolve(&svc, ModelCapability::Thinking.value(), true);

        assert_eq!(recorded.fits.lock().last().copied(), Some(Some(0.0)));
    }

    /// Waiving it is specific to the step that has no fallback worth having —
    /// a skills or dp model that doesn't fit still steps down.
    #[test]
    fn test_every_other_capability_keeps_the_reserve() {
        let svc = svc(&["qwen3.5-2b", "gemma3-4b"]);
        let (_, _, recorded) = resolve(&svc, "reasoning", true);

        assert_eq!(recorded.fits.lock().last().copied(), Some(None));
    }

    /// Waived is not unconditional: no free memory is no free memory, and the
    /// step lands back on the router model rather than OOMing.
    #[test]
    fn test_a_reasoning_model_that_still_does_not_fit_falls_back() {
        let svc = svc(&["qwen3.5-2b", "gemma3-4b"]);
        let (llm, _, recorded) = resolve(&svc, ModelCapability::Thinking.value(), false);

        assert!(llm.is_none());
        assert!(recorded.factory.lock().is_empty());
    }
}
