//! Port of `tests/unit/test_thinking_is_for_the_tool_call.py`.
//!
//! Reasoning is spent on the tool call, and nowhere else.
//!
//! A `<think>` block is how a tool call is course-corrected into the right
//! tool with the right arguments: the model reads the catalog, works out which
//! call the request means and what to pass it, and the monologue is stripped
//! before the JSON is parsed. By the time the answer is written that decision
//! is made and its results are already in the context — so reasoning there
//! buys nothing and costs the stream, because the filter withholds every token
//! of a thought the reader will never see.
//!
//! So the flag travels with the request rather than living on the model: the
//! resolve call asks for it, the answer asks against it, and neither depends
//! on what the other left behind.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use ndarray::Array2;
use parking_lot::Mutex;
use serial_test::serial;

use smartloop::llms::base_llm::{
    BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream,
};
use smartloop::llms::llama_cpp::{
    ChatCompletionOutput, ChatCompletionRequest, LlamaCpp, LlamaModel, LlamaState, LlamaToken,
};
use smartloop::model_host::client::{ModelClient, TextStream};
use smartloop::model_host::remote_llm::RemoteLLM;

// ------------------------------------------------------------------- fakes

/// Stands in for the `Llama` the reasoning flag is bound onto.
struct Flag {
    state: LlamaState,
}

impl Flag {
    /// A model with `_slp_thinking` bound to `enabled`, or — with `None` — a
    /// model that has no flag at all (Python's bare `object()`).
    fn new(enabled: Option<bool>) -> Arc<Self> {
        let state = LlamaState::default();
        state.set_thinking(enabled);
        Arc::new(Self { state })
    }
}

impl LlamaModel for Flag {
    fn state(&self) -> &LlamaState {
        &self.state
    }

    fn n_ctx(&self) -> usize {
        8192
    }

    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    fn tokenize(&self, text: &[u8]) -> anyhow::Result<Vec<LlamaToken>> {
        Ok((0..text.len() as i32).map(|t| t as LlamaToken).collect())
    }

    fn detokenize(&self, _tokens: &[LlamaToken]) -> anyhow::Result<Vec<u8>> {
        Ok(Vec::new())
    }

    fn create_chat_completion(&self, _request: ChatCompletionRequest) -> anyhow::Result<ChatCompletionOutput<'_>> {
        anyhow::bail!("not a real model")
    }
}

fn wrapper(enabled: bool) -> (LlamaCpp, Arc<Flag>) {
    let flag = Flag::new(Some(enabled));
    (LlamaCpp::new(flag.clone(), "", false, None), flag)
}

/// A `MagicMock` LLM: records how it was asked, answers what it was told to.
struct RecordingLLM {
    reply: String,
    chunks: Vec<String>,
    supports_thinking: bool,
    label: String,
    invoked: Mutex<Vec<(Option<usize>, Option<bool>)>>,
    streamed: Mutex<Vec<CallOptions>>,
}

impl RecordingLLM {
    fn new(reply: &str) -> Arc<Self> {
        Arc::new(Self {
            reply: reply.to_string(),
            chunks: vec!["hi".to_string()],
            supports_thinking: true,
            label: "test-model".to_string(),
            invoked: Mutex::new(Vec::new()),
            streamed: Mutex::new(Vec::new()),
        })
    }

    fn last_invoke_thinking(&self) -> Option<bool> {
        self.invoked.lock().last().expect("invoke was called").1
    }
}

impl BaseLLM for RecordingLLM {
    fn project_id(&self) -> &str {
        ""
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        text.len().div_ceil(4)
    }

    fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        messages
    }

    fn get_max_tokens(&self) -> usize {
        8192
    }

    fn call(&self, _prompt: &str, _stop: Option<&[String]>, options: CallOptions) -> anyhow::Result<String> {
        self.invoked.lock().push((options.max_tokens, options.thinking));
        Ok(self.reply.clone())
    }

    fn stream<'a>(&'a self, _prompt: &str, options: CallOptions) -> GenerationStream<'a> {
        self.streamed.lock().push(options);
        Box::new(self.chunks.clone().into_iter().map(|text| GenerationChunk { text }))
    }

    fn invoke(&self, _prompt: &str, max_tokens: Option<usize>, thinking: Option<bool>) -> String {
        self.invoked.lock().push((max_tokens, thinking));
        self.reply.clone()
    }

    fn supports_thinking(&self) -> bool {
        self.supports_thinking
    }

    fn model_label(&self) -> String {
        self.label.clone()
    }

    fn cancelled(&self) -> Option<CancelEvent> {
        None
    }

    fn set_cancelled(&self, _event: Option<CancelEvent>) {}

    fn llm_type(&self) -> &str {
        "recording"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A `MagicMock` model client: records the keyword arguments it was called
/// with.
#[derive(Default)]
struct RecordingClient {
    generated: Mutex<Vec<(Option<usize>, Option<bool>)>>,
    streamed: Mutex<Vec<Option<bool>>>,
}

impl ModelClient for RecordingClient {
    fn generate(
        &self,
        _prompt: &str,
        _model_name: Option<&str>,
        _capability: Option<&str>,
        max_tokens: Option<usize>,
        thinking: Option<bool>,
    ) -> anyhow::Result<String> {
        self.generated.lock().push((max_tokens, thinking));
        Ok("answered".to_string())
    }

    fn stream<'a>(
        &'a self,
        _prompt: &str,
        _model_name: Option<&str>,
        _capability: Option<&str>,
        _cancelled: Option<CancelEvent>,
        thinking: Option<bool>,
    ) -> anyhow::Result<TextStream<'a>> {
        self.streamed.lock().push(thinking);
        Ok(Box::new(vec![Ok("a".to_string()), Ok("b".to_string())].into_iter()))
    }

    fn encode_texts(&self, _texts: &[String], _base_dir: Option<&str>) -> anyhow::Result<Option<Array2<f32>>> {
        Ok(None)
    }

    fn is_loaded(&self) -> anyhow::Result<bool> {
        Ok(true)
    }

    fn model_name(&self) -> anyhow::Result<Option<String>> {
        Ok(None)
    }

    fn model_label(&self, _capability: Option<&str>) -> anyhow::Result<String> {
        Ok(String::new())
    }

    fn count_tokens(&self, _text: &str) -> anyhow::Result<usize> {
        Ok(0)
    }

    fn max_context_tokens(&self) -> anyhow::Result<usize> {
        Ok(0)
    }

    fn response_max_context_tokens(&self, _capability: Option<&str>) -> anyhow::Result<usize> {
        Ok(0)
    }

    fn has_capability_model(&self, _capability: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    fn idle_seconds(&self) -> anyhow::Result<f64> {
        Ok(0.0)
    }

    fn load_model(&self) -> anyhow::Result<Option<String>> {
        Ok(None)
    }

    fn unload_auxiliary_models(&self, _include_embedding: bool) -> anyhow::Result<()> {
        Ok(())
    }

    fn unload_embedding_model(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn unload_model_by_name(&self, _model_name: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ----------------------------------------------- TestTheFlagIsScopedToOneCall

#[test]
fn test_a_call_that_asks_to_reason_does_not_leave_it_on() {
    let (llm, flag) = wrapper(false);

    {
        let _guard = llm.thinking_override(Some(true));
        assert_eq!(flag.state().thinking(), Some(true));
    }

    assert_eq!(flag.state().thinking(), Some(false));
}

/// The picker's setting survives an answer that turned it down.
#[test]
fn test_a_call_that_asks_not_to_reason_does_not_leave_it_off() {
    let (llm, flag) = wrapper(true);

    {
        let _guard = llm.thinking_override(Some(false));
        assert_eq!(flag.state().thinking(), Some(false));
    }

    assert_eq!(flag.state().thinking(), Some(true));
}

#[test]
fn test_the_setting_is_restored_when_the_call_raises() {
    let (llm, flag) = wrapper(false);

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = llm.thinking_override(Some(true));
        panic!("oom");
    }));
    assert!(outcome.is_err());

    assert_eq!(flag.state().thinking(), Some(false));
}

#[test]
fn test_asking_for_nothing_changes_nothing() {
    let (llm, flag) = wrapper(true);

    {
        let _guard = llm.thinking_override(None);
        assert_eq!(flag.state().thinking(), Some(true));
    }

    assert_eq!(flag.state().thinking(), Some(true));
}

#[test]
fn test_a_model_that_cannot_reason_is_left_alone() {
    let flag = Flag::new(None);
    let llm = LlamaCpp::new(flag.clone(), "", false, None);

    {
        let _guard = llm.thinking_override(Some(true)); // no flag to set, and no panic either
    }

    assert_eq!(flag.state().thinking(), None);
}

// ------------------------------------------------- TestReasoningIsOptInPerCall
//
// A call that does not ask to reason does not reason.  Classification,
// planning, per-document extraction and synthesis all leave room to think and
// none of them use the thought; inheriting the model's loaded flag made every
// one of those calls slower with nothing to show for it.

#[test]
fn test_a_call_that_says_nothing_does_not_reason() {
    assert!(!LlamaCpp::requested_thinking(&CallOptions::default()));
}

/// What a call carries over the model host socket when nobody asked.
#[test]
fn test_an_explicit_none_is_not_an_opt_in() {
    assert!(!LlamaCpp::requested_thinking(&CallOptions { thinking: None, ..Default::default() }));
}

#[test]
fn test_asking_for_it_is_honoured() {
    assert!(LlamaCpp::requested_thinking(&CallOptions { thinking: Some(true), ..Default::default() }));
}

#[test]
fn test_the_loaded_flag_does_not_decide_a_call() {
    let (llm, flag) = wrapper(true); // a model loaded with reasoning on

    {
        let _guard = llm.thinking_override(Some(LlamaCpp::requested_thinking(&CallOptions::default())));
        assert_eq!(flag.state().thinking(), Some(false));
    }

    // Restored, so the model is left as it was found.
    assert_eq!(flag.state().thinking(), Some(true));
}

// ---------------------------------------------------------- TestWhoAsksForWhat

#[test]
#[serial]
fn test_the_resolve_call_asks_to_reason() {
    use smartloop::agents::mcp::mcp_agent::{clear_resolution_cache, MCPAgent};
    use smartloop::workspace::workspace::MCPServer;

    clear_resolution_cache();
    let llm = RecordingLLM::new(r#"{"tool": 1, "arguments": {"q": "pitch"}}"#);
    let server: MCPServer = serde_json::from_value(serde_json::json!({
        "id": "s1",
        "name": "notion",
        "enabled": true,
        "system": false,
        "tools": [{
            "name": "notion-search",
            "description": "Search",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string", "description": "query"}},
                "required": ["q"],
            },
        }],
    }))
    .unwrap();

    MCPAgent::new(llm.clone()).resolve("outline the pitch", &[server], None, None);
    clear_resolution_cache();

    assert_eq!(llm.last_invoke_thinking(), Some(true));
}

/// Pinned at the call site, since this is the one the user waits on.
#[test]
fn test_the_answer_asks_not_to() {
    let src = include_str!("../../src/api/routers/chat.rs");

    // The streamed answer passes the cancel event and then `Some(false)` as
    // the thinking argument (`cancelled=cancelled, thinking=False,` in the
    // Python source).
    let streamed = regex::Regex::new(r"cancelled[^;]{0,120}?,\s*Some\(false\)\s*,?\s*\)").unwrap();
    assert!(streamed.is_match(src), "the streamed answer must ask for no reasoning");
    // …and the non-streaming answer, which shares the model.
    assert!(src.matches("Some(false)").count() >= 3);
}

// ------------------------------------------------- TestTheFlagReachesTheModelHost
//
// A project agent holds a socket, not the weights, so the flag has to travel
// with the request rather than being set on a local model.

#[test]
fn test_generate_carries_it() {
    let client = Arc::new(RecordingClient::default());
    let llm = RemoteLLM::new(client.clone(), "p", None);

    llm.invoke("hello", Some(1024), Some(true));

    assert_eq!(*client.generated.lock(), vec![(Some(1024), Some(true))]);
}

#[test]
fn test_stream_carries_it() {
    let client = Arc::new(RecordingClient::default());
    let llm = RemoteLLM::new(client.clone(), "p", None);

    let chunks: Vec<GenerationChunk> =
        llm.stream("hello", CallOptions { thinking: Some(false), ..Default::default() }).collect();

    assert_eq!(chunks.len(), 2);
    assert_eq!(*client.streamed.lock(), vec![Some(false)]);
}

#[tokio::test]
#[serial]
async fn test_the_host_serves_what_the_request_asked_for() {
    use smartloop::model_host::host::ModelHostServicer;
    use smartloop::model_host::pb;
    use smartloop::model_host::pb::model_host_server::ModelHost as _;

    let served = Arc::new(RecordingClient::default());
    let request = pb::GenerateRequest {
        prompt: "hi".into(),
        thinking: Some(true),
        max_tokens: Some(1024),
        ..Default::default()
    };
    ModelHostServicer::new(served.clone())
        .generate(tonic::Request::new(request))
        .await
        .unwrap();

    assert_eq!(*served.generated.lock(), vec![(Some(1024), Some(true))]);
}

// ---------------------------------------------------- TestTheServiceForwardsIt

fn service_with(llm: Arc<RecordingLLM>) -> Arc<smartloop::api::services::model_service::ModelService> {
    let svc = smartloop::api::services::model_service::ModelService::new_bare();
    *svc.llm.write() = Some(llm);
    svc
}

#[test]
fn test_a_streamed_answer_passes_the_flag_down() {
    let llm = RecordingLLM::new("unused");
    let svc = service_with(llm.clone());

    let chunks: Vec<_> = svc
        .stream_completion("prompt", None, None, None, Some(false))
        .unwrap()
        .collect();

    assert!(!chunks.is_empty());
    assert_eq!(llm.streamed.lock().last().expect("stream was called").thinking, Some(false));
}

#[test]
fn test_a_non_streamed_answer_passes_it_too() {
    let llm = RecordingLLM::new("answer");
    let svc = service_with(llm.clone());

    svc.generate_completion("prompt", None, None, None, Some(false)).unwrap();

    assert_eq!(llm.last_invoke_thinking(), Some(false));
}
