//! Port of `tests/unit/test_thinking_binding.py`.
//!
//! `enable_thinking` reaches the model's chat template, and vision opts out.
//!
//! `create_chat_completion` has no `**kwargs` to forward template variables
//! through, so the flag is bound onto the resolved chat handler at load time —
//! that is the only hook a hybrid-reasoning model's template reads, and it
//! covers the call sites that reach `create_chat_completion` directly (vision
//! Q&A, OCR) as well as the LLM wrapper.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{Map, Value, json};

use smartloop::llms::llama_cpp::{
    ChatCompletionOutput, ChatCompletionRequest, ChatHandler, ChatHandlerRequest, LlamaCpp, LlamaModel, LlamaState,
    LlamaToken, THINKING_MIN_BUDGET,
};
use smartloop::llms::llm_factory::LlmFactory;
use smartloop::model::spec::MODEL_SPECS;
use smartloop::types::ModelCapability;
use smartloop::workspace::workspace::WorkspaceModel;

/// Just enough of `Llama` for handler resolution.
struct FakeLlama {
    state: LlamaState,
}

impl FakeLlama {
    fn new(handler: Option<ChatHandler>) -> Self {
        Self { state: LlamaState::new(handler, Some("chat_template.default".to_string())) }
    }
}

impl LlamaModel for FakeLlama {
    fn state(&self) -> &LlamaState {
        &self.state
    }

    fn n_ctx(&self) -> usize {
        4096
    }

    fn tokenize(&self, text: &[u8]) -> anyhow::Result<Vec<LlamaToken>> {
        Ok(vec![1; text.len() / 4 + 1])
    }

    fn detokenize(&self, tokens: &[LlamaToken]) -> anyhow::Result<Vec<u8>> {
        Ok(vec![b'x'; tokens.len()])
    }

    fn create_chat_completion(&self, request: ChatCompletionRequest) -> anyhow::Result<ChatCompletionOutput<'_>> {
        let handler = smartloop::llms::llama::resolve_chat_handler(self)?;
        handler(ChatHandlerRequest { llama: Some(self), request, template_kwargs: Map::new() })
    }
}

type Calls = Arc<Mutex<Vec<Map<String, Value>>>>;

/// A handler that records the template kwargs of every call.
fn handler() -> (ChatHandler, Calls) {
    let calls: Calls = Arc::new(Mutex::new(Vec::new()));
    let recorded = calls.clone();
    let handler: ChatHandler = Arc::new(move |req: ChatHandlerRequest<'_>| {
        recorded.lock().push(req.template_kwargs.clone());
        Ok(ChatCompletionOutput::Response(json!({"choices": []})))
    });
    (handler, calls)
}

/// `llm.chat_handler(llama=llm, messages=[], **kwargs)`.
fn call_handler(llm: &dyn LlamaModel, kwargs: Map<String, Value>) {
    let handler = llm.state().chat_handler().expect("a chat handler is set");
    handler(ChatHandlerRequest { llama: Some(llm), request: ChatCompletionRequest::default(), template_kwargs: kwargs })
        .expect("handler call succeeds");
}

/// `calls[-1]["enable_thinking"]`.
fn last_flag(calls: &Calls) -> Option<Value> {
    calls.lock().last().expect("a recorded call").get("enable_thinking").cloned()
}

/// `LlamaCpp`'s budget guard over a fake model.
fn wrapper(llm: &Arc<FakeLlama>) -> LlamaCpp {
    let model: Arc<dyn LlamaModel> = llm.clone();
    LlamaCpp::new(model, "p", false, None)
}

#[test]
fn test_bound_flag_is_passed_to_every_completion() {
    let (handler, calls) = handler();
    let llm = FakeLlama::new(Some(handler));

    LlmFactory::bind_thinking(&llm, true);
    call_handler(&llm, Map::new());

    assert_eq!(calls.lock()[0].get("enable_thinking"), Some(&Value::Bool(true)));
}

#[test]
fn test_thinking_can_be_bound_off() {
    let (handler, calls) = handler();
    let llm = FakeLlama::new(Some(handler));

    LlmFactory::bind_thinking(&llm, false);
    call_handler(&llm, Map::new());

    assert_eq!(calls.lock()[0].get("enable_thinking"), Some(&Value::Bool(false)));
}

#[test]
fn test_a_caller_can_still_override_the_bound_default() {
    let (handler, calls) = handler();
    let llm = FakeLlama::new(Some(handler));

    LlmFactory::bind_thinking(&llm, true);
    let mut kwargs = Map::new();
    kwargs.insert("enable_thinking".into(), Value::Bool(false));
    call_handler(&llm, kwargs);

    assert_eq!(calls.lock()[0].get("enable_thinking"), Some(&Value::Bool(false)));
}

#[test]
fn test_a_short_output_budget_suppresses_reasoning() {
    // Decision calls (8-256 tokens for a yes/no) would spend the whole budget
    // inside <think> and return a truncated thought instead of an answer.
    let (handler, calls) = handler();
    let llm = Arc::new(FakeLlama::new(Some(handler)));
    LlmFactory::bind_thinking(&*llm, true);

    let wrapper = wrapper(&llm);

    {
        let _guard = wrapper.thinking_for_budget(Some(16));
        call_handler(&*llm, Map::new());
    }
    assert_eq!(last_flag(&calls), Some(Value::Bool(false)));

    // ...and the model's default is back afterwards.
    {
        let _guard = wrapper.thinking_for_budget(None);
        call_handler(&*llm, Map::new());
    }
    assert_eq!(last_flag(&calls), Some(Value::Bool(true)));
}

#[test]
fn test_a_prompt_that_fills_the_context_suppresses_reasoning() {
    // No explicit cap, but a 38k-token prompt left only a sliver of headroom:
    // the guard sees the effective budget, not just what the caller asked for.
    let (handler, calls) = handler();
    let llm = Arc::new(FakeLlama::new(Some(handler)));
    LlmFactory::bind_thinking(&*llm, true);

    let wrapper = wrapper(&llm);

    {
        let _guard = wrapper.thinking_for_budget(Some(200));
        call_handler(&*llm, Map::new());
    }
    assert_eq!(last_flag(&calls), Some(Value::Bool(false)));
}

#[test]
fn test_a_full_output_budget_keeps_reasoning() {
    let (handler, calls) = handler();
    let llm = Arc::new(FakeLlama::new(Some(handler)));
    LlmFactory::bind_thinking(&*llm, true);

    let wrapper = wrapper(&llm);

    {
        let _guard = wrapper.thinking_for_budget(Some(THINKING_MIN_BUDGET * 8));
        call_handler(&*llm, Map::new());
    }
    assert_eq!(last_flag(&calls), Some(Value::Bool(true)));
}

#[test]
fn test_a_non_reasoning_model_is_left_alone() {
    // Nothing was bound, so the budget guard must not invent a flag.
    let (handler, calls) = handler();
    let llm = Arc::new(FakeLlama::new(Some(handler)));

    let wrapper = wrapper(&llm);

    {
        let _guard = wrapper.thinking_for_budget(Some(16));
        call_handler(&*llm, Map::new());
    }
    assert!(!calls.lock().last().unwrap().contains_key("enable_thinking"));
}

#[test]
fn test_binding_failure_leaves_the_model_usable() {
    // A llama-cpp that resolves handlers differently must not fail the load —
    // the model just keeps its chat template's own default.
    let llm = FakeLlama { state: LlamaState::without_handler_registry(None, Some("unknown".to_string())) };
    LlmFactory::bind_thinking(&llm, true);
    assert_eq!(llm.state().thinking(), None);
}

#[test]
fn test_a_model_without_the_capability_is_never_bound() {
    assert_eq!(LlmFactory::thinking_for(Some("gemma3-4b"), None), None);
    assert_eq!(LlmFactory::thinking_for(None, None), None);
}

#[test]
fn test_unknown_workspace_model_is_left_alone() {
    assert_eq!(LlmFactory::thinking_for(Some("some-user-added-model"), None), None);
}

#[test]
fn test_the_bootstrap_model_answers_the_capability_question() {
    // It declares nothing, and the answer is no.
    assert!(!LlmFactory::supports_thinking(Some("sl-mini")));
    assert_eq!(LlmFactory::thinking_for(Some("sl-mini"), None), None);
}

#[test]
fn test_sl_mini_declares_nothing_like_the_base_default() {

    // The base default has to match what capability inference already gave a
    // 1B text model, or adding it would silently re-scope the orchestrator.
    let sl_mini = WorkspaceModel {
        model_name: "sl-mini".to_string(),
        model_id: "smartloop-ai/slp-orchestra-mini".to_string(),
        ..WorkspaceModel::default()
    };
    assert!(sl_mini.effective_capabilities().is_empty());
    assert!(ModelCapability::text_model(&sl_mini.effective_capabilities()));
}

#[test]
fn test_every_supported_model_resolves() {
    // Guards against a spec that answers the capability question but not the
    // default one (or vice versa) breaking loads for that model.
    for name in MODEL_SPECS.keys() {
        let resolved = LlmFactory::thinking_for(Some(&name), None);
        assert!(matches!(resolved, Some(true) | Some(false) | None), "{name}");
    }
}

#[test]
fn test_nothing_can_force_reasoning_onto_a_model_without_it() {
    // gemma3-4b's template never reads enable_thinking, so honoring the flag
    // would just bind a no-op wrapper and mislead the caller.
    assert_eq!(LlmFactory::thinking_for(Some("gemma3-4b"), Some(true)), None);
}
