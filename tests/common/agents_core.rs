//! ===== agents_core shared fixtures =====
//!
//! Test doubles for the orchestrator tests (unit `agents_core`): a
//! programmable LLM standing in for Python's `MagicMock()` model, a log
//! capture standing in for pytest's `caplog`, and builders for the MCP
//! servers / requests the orchestrator tests share.

use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Map, Value};

use smartloop::agents::agent::SharedLLM;
use smartloop::agents::nodes::{AgentNode, NodeCategory, NodeKind};
use smartloop::agents::orchestrator::{SLPOrchestrator, SLPOrchestratorOptions};
use smartloop::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream};
use smartloop::workspace::workspace::{MCPServer, MCPServerType, MCPToolDefinition, RemoteMCPServerConfig};

/// What the fake answers for a prompt: `Ok(reply)`, or `Err(message)` for a
/// failing model (Python `side_effect = RuntimeError(message)`).
pub type Responder = Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>;

/// A programmable model (Python `MagicMock()` with `invoke`,
/// `get_max_tokens` and `get_num_tokens` set).
///
/// Every prompt is recorded.  A failing reply surfaces from `call` as an
/// error and from `invoke` the way the real wrappers report one — as
/// `"Error generating response: <message>"` — since `BaseLLM::invoke` never
/// fails.
pub struct FakeLlm {
    responder: Mutex<Responder>,
    /// When set, a failing reply makes `invoke` raise (panic) instead of
    /// returning the wrappers' error string — `MagicMock(side_effect=...)`
    /// replacing `invoke` itself.
    raises: AtomicBool,
    calls: Mutex<Vec<String>>,
    max_tokens: AtomicUsize,
    num_tokens: AtomicUsize,
    thinking: AtomicBool,
    label: Mutex<String>,
    cancelled: Mutex<Option<CancelEvent>>,
}

impl FakeLlm {
    /// A model answering `reply` to everything (4096-token window, every
    /// text 100 tokens long).
    pub fn new(reply: &str) -> Arc<Self> {
        let reply = reply.to_string();
        Arc::new(Self {
            responder: Mutex::new(Arc::new(move |_: &str| Ok(reply.clone()))),
            raises: AtomicBool::new(false),
            calls: Mutex::new(Vec::new()),
            max_tokens: AtomicUsize::new(4096),
            num_tokens: AtomicUsize::new(100),
            thinking: AtomicBool::new(false),
            label: Mutex::new(String::new()),
            cancelled: Mutex::new(None),
        })
    }

    /// Same model, as the trait object the production code takes.
    pub fn shared(self: &Arc<Self>) -> Arc<dyn BaseLLM> {
        Arc::clone(self) as Arc<dyn BaseLLM>
    }

    /// Same model, as a [`SharedLLM`].
    pub fn shared_llm(self: &Arc<Self>) -> SharedLLM {
        Arc::clone(self) as SharedLLM
    }

    /// `llm.invoke.return_value = reply`.
    pub fn set_reply(&self, reply: &str) {
        let reply = reply.to_string();
        *self.responder.lock() = Arc::new(move |_: &str| Ok(reply.clone()));
        self.raises.store(false, Ordering::SeqCst);
    }

    /// `llm.invoke.side_effect = RuntimeError(message)`.
    pub fn set_failure(&self, message: &str) {
        let message = message.to_string();
        *self.responder.lock() = Arc::new(move |_: &str| Err(message.clone()));
        self.raises.store(false, Ordering::SeqCst);
    }

    /// `llm.invoke = MagicMock(side_effect=RuntimeError(message))`: `invoke`
    /// itself raises (panics) rather than reporting the failure as the
    /// wrappers' `"Error generating response: ..."` string.
    pub fn set_raising(&self, message: &str) {
        let message = message.to_string();
        *self.responder.lock() = Arc::new(move |_: &str| Err(message.clone()));
        self.raises.store(true, Ordering::SeqCst);
    }

    /// `llm.invoke.side_effect = f`.
    pub fn set_responder(&self, f: impl Fn(&str) -> Result<String, String> + Send + Sync + 'static) {
        *self.responder.lock() = Arc::new(f);
        self.raises.store(false, Ordering::SeqCst);
    }

    /// `get_max_tokens.return_value = n`.
    pub fn set_max_tokens(&self, n: usize) {
        self.max_tokens.store(n, Ordering::SeqCst);
    }

    /// Whether the model reads as reasoning-capable.
    pub fn set_supports_thinking(&self, on: bool) {
        self.thinking.store(on, Ordering::SeqCst);
    }

    /// The model's user-facing label.
    pub fn set_label(&self, label: &str) {
        *self.label.lock() = label.to_string();
    }

    /// Every prompt the model was asked, in order.
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().clone()
    }

    /// `llm.invoke.call_count`.
    pub fn call_count(&self) -> usize {
        self.calls.lock().len()
    }

    /// `llm.invoke.reset_mock()`.
    pub fn reset_calls(&self) {
        self.calls.lock().clear();
    }

    fn answer(&self, prompt: &str) -> Result<String, String> {
        self.calls.lock().push(prompt.to_string());
        let responder = self.responder.lock().clone();
        responder(prompt)
    }
}

impl BaseLLM for FakeLlm {
    fn project_id(&self) -> &str {
        "test-project"
    }
    fn get_num_tokens(&self, _text: &str) -> usize {
        self.num_tokens.load(Ordering::SeqCst)
    }
    fn format_prompt(&self, prompt: &str, system_message: Option<&str>) -> Vec<ChatMessage> {
        vec![ChatMessage::new("system", system_message.unwrap_or("")), ChatMessage::new("user", prompt)]
    }
    fn create_messages_from_prompt(&self, prompt: &str) -> Vec<ChatMessage> {
        vec![ChatMessage::new("user", prompt)]
    }
    fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        messages
    }
    fn get_max_tokens(&self) -> usize {
        self.max_tokens.load(Ordering::SeqCst)
    }
    fn call(&self, prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
        self.answer(prompt).map_err(|e| anyhow::anyhow!(e))
    }
    fn stream<'a>(&'a self, prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
        let text = match self.answer(prompt) {
            Ok(text) => text,
            Err(e) => format!("Error generating response: {e}"),
        };
        Box::new(std::iter::once(GenerationChunk { text }))
    }
    fn invoke(&self, prompt: &str, _max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
        // A mock's `invoke` returns its value verbatim.
        match self.answer(prompt) {
            Ok(text) => text,
            Err(e) if self.raises.load(Ordering::SeqCst) => panic!("{e}"),
            Err(e) => format!("Error generating response: {e}"),
        }
    }
    fn supports_thinking(&self) -> bool {
        self.thinking.load(Ordering::SeqCst)
    }
    fn model_label(&self) -> String {
        self.label.lock().clone()
    }
    fn cancelled(&self) -> Option<CancelEvent> {
        self.cancelled.lock().clone()
    }
    fn set_cancelled(&self, event: Option<CancelEvent>) {
        *self.cancelled.lock() = event;
    }
    fn llm_type(&self) -> &str {
        "mock"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `SLPOrchestrator(orchestrator_llm=llm)`.
pub fn orchestrator_over(llm: &Arc<FakeLlm>) -> SLPOrchestrator {
    SLPOrchestrator::new(llm.shared(), SLPOrchestratorOptions::default())
}

/// A model answering `reply`, with an 8192-token window — the
/// `_make_orchestrator()` model most of the orchestrator tests share.
pub fn llm_8k(reply: &str) -> Arc<FakeLlm> {
    let llm = FakeLlm::new(reply);
    llm.set_max_tokens(8192);
    llm
}

/// Python's class name for a node (`type(n).__name__`).
pub fn class_name(node: &AgentNode) -> &'static str {
    match (node.category, node.kind) {
        (_, NodeKind::Enrich) => "EnrichNode",
        (_, NodeKind::SkillMatch) => "SkillMatchNode",
        (_, NodeKind::Collect) => "CollectNode",
        (_, NodeKind::Route) => "RouteNode",
        (NodeCategory::LLMDecision, _) => "LLMDecisionNode",
        (NodeCategory::Transform, _) => "TransformNode",
        (NodeCategory::Search, _) => "SearchNode",
        (NodeCategory::ToolCall, _) => "ToolCallNode",
        (NodeCategory::Base, _) => "AgentNode",
    }
}

/// A JSON object from a `json!` literal.
pub fn obj(v: Value) -> Map<String, Value> {
    v.as_object().cloned().expect("a JSON object")
}

/// A tool definition with an object input schema.
pub fn tool(name: &str, description: &str, properties: Value, required: &[&str]) -> MCPToolDefinition {
    MCPToolDefinition {
        name: name.to_string(),
        description: Some(description.to_string()),
        input_schema: Some(obj(json!({
            "type": "object",
            "properties": properties,
            "required": required,
        }))),
    }
}

/// A remote MCP server exposing `tools`.
pub fn remote_server(id: &str, name: &str, tools: Vec<MCPToolDefinition>) -> MCPServer {
    MCPServer {
        id: id.to_string(),
        name: name.to_string(),
        server_type: MCPServerType::Remote,
        remote_config: Some(RemoteMCPServerConfig {
            server_url: "https://mcp.example.com".to_string(),
            ..Default::default()
        }),
        enabled: true,
        tools,
        ..Default::default()
    }
}

/// The remote "fetch" server of the e2e tests (`_make_fetch_server`).
pub fn fetch_server(id: &str, name: &str) -> MCPServer {
    remote_server(
        id,
        name,
        vec![tool(
            "fetch",
            "Fetch a URL and return its content as markdown",
            json!({"url": {"type": "string", "description": "URL to fetch"}}),
            &["url"],
        )],
    )
}

/// A buffer the log capture writes into.
#[derive(Clone, Default)]
pub struct LogBuffer(pub Arc<Mutex<Vec<u8>>>);

impl LogBuffer {
    /// Everything logged so far (pytest `caplog.text`).
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock()).to_string()
    }
}

impl std::io::Write for LogBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Run `f` with INFO-and-above logs on this thread captured (pytest
/// `caplog.at_level(logging.INFO)`); returns `f`'s result and the log text.
pub fn capture_logs<R>(f: impl FnOnce() -> R) -> (R, String) {
    let buffer = LogBuffer::default();
    let writer = buffer.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let result = tracing::subscriber::with_default(subscriber, f);
    (result, buffer.text())
}

/// The hosted web search, stubbed at its seams for a whole test: a workspace
/// on disk whose current project has web search on, a signed-in account whose
/// status endpoint says yes, a search endpoint answering `links`, and every
/// [`smartloop::agents::fetch_agent::FetchAgent`] fetch answering `page`.
/// Everything is restored on drop; tests using it must be `#[serial]`.
pub struct WebSearchStub {
    _tmp: tempfile::TempDir,
}

struct StubSearchClient(Vec<String>);

impl smartloop::search::providers::WebSearchClient for StubSearchClient {
    fn web_search(
        &self,
        _query: &str,
        _count: usize,
    ) -> Result<Vec<String>, smartloop::search::providers::WebSearchCallError> {
        Ok(self.0.clone())
    }
    fn web_search_status(&self) -> Result<bool, smartloop::search::providers::WebSearchCallError> {
        Ok(true)
    }
}

impl WebSearchStub {
    /// Install the stubs.
    pub fn new(links: &[&str], page: &str) -> Self {
        use smartloop::search::providers;
        use smartloop::workspace::workspace::Workspace;

        let tmp = tempfile::tempdir().expect("temp dir");
        let dir = tmp.path().to_string_lossy().to_string();
        let cfg = format!("{dir}/config.json");
        std::fs::write(
            &cfg,
            serde_json::to_string(&json!({
                "id": "m",
                "workspace": {
                    "id": "ws", "workspace_dir": dir, "models_dir": dir, "models": [],
                    "projects": [{"id": "p1", "project_dir": dir, "current": true,
                                  "name": "p1", "web_search_enabled": true}],
                },
            }))
            .expect("config json"),
        )
        .expect("write config");
        Workspace::reset();
        Workspace::set_config_file(Some(cfg));

        let links: Vec<String> = links.iter().map(|s| s.to_string()).collect();
        providers::reset_status_cache();
        providers::set_credentials_check(Some(Arc::new(|| true)));
        providers::set_platform_client_factory(Some(Arc::new(move || {
            Box::new(StubSearchClient(links.clone())) as Box<dyn providers::WebSearchClient>
        })));

        let page = page.to_string();
        smartloop::agents::fetch_agent::set_fetch_override(Some(Arc::new(move |_url, max_length, start, _raw| {
            page.chars().skip(start).take(max_length).collect()
        })));
        Self { _tmp: tmp }
    }
}

impl Drop for WebSearchStub {
    fn drop(&mut self) {
        use smartloop::search::providers;
        use smartloop::workspace::workspace::Workspace;

        smartloop::agents::fetch_agent::set_fetch_override(None);
        providers::set_credentials_check(None);
        providers::set_platform_client_factory(None);
        providers::reset_status_cache();
        Workspace::set_config_file(None);
        Workspace::reset();
    }
}

/// A [`smartloop::mcp::MCPToolCallResult`] carrying `text`.
pub fn text_result(text: &str) -> smartloop::mcp::MCPToolCallResult {
    smartloop::mcp::MCPToolCallResult {
        success: true,
        content: Some(vec![json!({"type": "text", "text": text})]),
        is_error: false,
        error: None,
        auth_expired: false,
    }
}

/// A failed [`smartloop::mcp::MCPToolCallResult`].
pub fn failed_result(error: &str) -> smartloop::mcp::MCPToolCallResult {
    smartloop::mcp::MCPToolCallResult {
        success: false,
        content: None,
        is_error: false,
        error: Some(error.to_string()),
        auth_expired: false,
    }
}

/// Collects what the router reports (`models.emit(step, status, message)`
/// calls): install with `models().begin_request(Some(sink.callback()))`.
#[derive(Clone, Default)]
pub struct EmitSink(pub Arc<Mutex<Vec<(String, String, String)>>>);

impl EmitSink {
    /// The progress callback appending to this sink.
    pub fn callback(&self) -> smartloop::agents::orchestration::model_router::ProgressCallback {
        let events = Arc::clone(&self.0);
        Arc::new(move |step: &str, status: &str, message: &str| {
            events.lock().push((step.to_string(), status.to_string(), message.to_string()));
        })
    }

    /// The reported messages (`[c.args[2] for c in models.emit.call_args_list]`).
    pub fn messages(&self) -> Vec<String> {
        self.0.lock().iter().map(|(_, _, m)| m.clone()).collect()
    }
}
