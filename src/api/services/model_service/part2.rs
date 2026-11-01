//! Second half of `smartloop/api/services/model_service.py` (from
//! `_llm_for_response` onward): response-model routing and generation,
//! documents, projects, model unloading, MCP server registration and
//! provisioning, orchestration, auto-write, assets and vision Q&A.
//!
//! The `ModelService` struct, its fields and the first half of its methods
//! live in the parent module (`super`).

use super::*;

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::agents::agent::SharedLLM;
use crate::agents::hybrid_executor::LLMExecutor;
use crate::agents::mcp::argument_extractor::MCPArgumentExtractor;
use crate::agents::mcp::mcp_tool_caller::{MCPToolCaller, PersistFn};
use crate::agents::mcp::result_mapper::MCPResultMapper;
use crate::agents::mcp::server_summarizer::MCPServerSummarizer;
use crate::agents::orchestration::model_router::ProgressCallback;
use crate::agents::orchestrator::{ExecuteOptions, SLPOrchestrator, SLPOrchestratorOptions};
use crate::llms::base_llm::{BatchPrompt, CallOptions, CancelEvent, ChatMessage, GenerationChunk, GenerationStream};
use crate::llms::cloud_llm::{local_only_scope, CloudLLM};
use crate::model_host::remote_llm::RemoteLLM;
use crate::llms::llama::ChatCompletionRequest;
use crate::mcp::account::probe_account;
use crate::mcp::client::{
    MCPAuthExpiredError, MCPClient, MCPClientError, MCPConnectResult, MCPDiscoveryError, MCPServerInfo,
    MCPToolCallResult,
};
use crate::mcp::definition_loader::{USER_DEFINITIONS_DIR, reload_definitions};
use crate::mcp::definitions::{definition_for, split_command_line};
use crate::mcp::local_client::LocalMCPClient;
use crate::mcp::read_only_calls::ReadOnlyToolCaller;
use crate::model::downloader::SLPArtifactDownloader;
use crate::model::downloader::slp_downloader::ProgressCallback as DownloadProgressCallback;
use crate::model::slp_model::{DocumentProgressCallback, DocumentSource};
use crate::parsers::base_parser::ParserError;
use crate::parsers::parser_factory::ParserFactory;
use crate::paths::{SANDBOX_HOME_MARKER, expand_home};
use crate::tasks::base_task::{TaskLogger, TaskOptions, TaskStatus};
use crate::tasks::{BaseTask, DownloaderTask, DocumentsManifest, IndexerTask, MCPTask};
use crate::utils::logging_setup::get_logger;
use crate::workspace::context::{own_project, projects_delegated, unbind_project};
use crate::workspace::Document;
use crate::workspace::workspace::{MCPAuthType, MCPOAuthCredentials, SharedWorkspace};

/// The Python logger name of this module.
const LOG: &str = "smartloop.api.services.model_service";

/// Callback fired with the name of the vision model chosen for a request.
pub type ModelSelectedCallback = Arc<dyn Fn(&str) + Send + Sync>;

/// `orchestrate` result:
/// `(context, matched, error, citations, skill_instruction, mutation)`.
///
/// * `context` — the formatted tool output;
/// * `matched` — true when the orchestrator produced content;
/// * `error` — a user-facing error message (or `None` on success);
/// * `citations` — source document dicts (or `None`);
/// * `skill_instruction` — the matched skill content (or `None`);
/// * `mutation` — the router should stream `context` verbatim instead of
///   running it through another LLM.
pub type OrchestrateOutcome = (
    Option<String>,
    bool,
    Option<String>,
    Option<Vec<Value>>,
    Option<String>,
    bool,
);

/// Signals that an image-attachment request wants to *use* indexed documents
/// (cross-validate / answer with them), not just read the image. When none of
/// these appear, an attached image is treated as the sole source and the
/// retrieved documents are discarded so they can't mis-attribute its contents.
pub static IMAGE_CROSS_REF_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r"(?i)\b(compare|comparison|match(?:es|ing)?|validate|verif(?:y|ies|ication)|",
        r"cross[-\s]?(?:check|validat\w*|referenc\w*)|reconcile|consistent|discrepanc\w*|",
        r"recommend|suggest|against (?:our|the|my)|based on (?:our|the|my)|",
        r"using (?:our|the|my)|according to (?:our|the|my)|",
        r"(?:in|from|with) (?:our|the|my|these|those) (?:record|document|doc|file|data|report)s?)\b",
    ))
    .expect("valid IMAGE_CROSS_REF_RE")
});

/// Inline write intent in a user message ("create", "write", "export", ...).
pub static WRITE_INTENT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(create|generate|write|save|export|produce|make|build|draft)\b")
        .expect("valid WRITE_INTENT_RE")
});

/// Matches bare "write" / "save" / "write it" / "save that" commands.
pub static BARE_WRITE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^\s*(write|save|write\s+(it|that|this)|save\s+(it|that|this))\s*[.!]?\s*$")
        .expect("valid BARE_WRITE_RE")
});

/// Conversational preamble ("Sure, here's ...") stripped before auto-write.
pub static PREAMBLE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r"(?i)^(?:(?:sure|okay|of course|absolutely|certainly|here(?:'s| is| are| you go))",
        r"[^\n]*\n+)+",
    ))
    .expect("valid PREAMBLE_RE")
});

/// Conversational postamble ("Let me know ...") stripped before auto-write.
pub static POSTAMBLE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r"(?i)\n+(?:(?:i hope|let me know|feel free|enjoy|bon appétit|happy|hope this|",
        r"if you (?:have|need|want)|don't hesitate|",
        r"would you like|do you want|shall i|",
        r"you (?:can|could|may|might) (?:also|adjust|modify|change|customize|tweak)|",
        r"this (?:can be|could be|is just|is a) (?:adjusted|modified|customized|starting)|",
        r"note:|please note|disclaimer|tips?:)[^\n]*\n*)+\s*$",
    ))
    .expect("valid POSTAMBLE_RE")
});

/// System-message steering for direct vision Q&A.
///
/// Sent as the system message.  (An earlier version folded this into the
/// *user* message on the theory that a system message made Qwen2-VL emit EOS
/// immediately; that theory was wrong — the chat handler always injects a
/// default system message when the caller supplies none — so supplying our
/// own simply replaces that default with useful steering.)
pub const VISION_QA_GUIDANCE: &str = "Answer using only what is actually visible in the image. If the answer \
is not visible, say so plainly instead of guessing. Do not invent \
people, text, or details that are not there.";

/// Perception prompt for the orchestrator's image path; `{request}` is
/// replaced with the user's message.  Asks the vision model only to describe
/// what it sees, scoped to the request, because the perception is fed back
/// into the text pipeline which does the actual answering.
pub const VISION_PERCEPTION_PROMPT: &str = "Describe what is visible in this image, focusing on the details that \
are relevant to the following request. Be concrete and specific about \
colors, objects, and any visible text. Describe only what is visible; \
do not answer the request or make recommendations.\n\nRequest: {request}";

// --------------------------------------------------------------- MCP regexes

static MCP2_NO_MODULE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"no module named ['"]?mcp(\.|['" ])"#).expect("valid regex"));
static MCP2_CANNOT_IMPORT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"cannot import name .* from ['"]?mcp(\.|['" ])"#).expect("valid regex"));
static MCP2_NO_ATTRIBUTE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"module ['"]?mcp['"]? has no attribute"#).expect("valid regex"));
static MCP2_DECORATOR_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r"'(?:server|fastmcp)' object has no attribute ",
        r"'(?:list_tools|call_tool|list_resources|read_resource|",
        r"list_prompts|get_prompt)'",
    ))
    .expect("valid regex")
});
static UV_HINT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"Use `uv(?:x| tool run) (.*?)` instead").expect("valid regex"));

static FRIENDLY_SDK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r#"no module named ['"]?mcp[.'"]|mcp\.server\.fastmcp|mcp\.shared|"#,
        r#"cannot import name .* from ['"]?mcp"#,
    ))
    .expect("valid regex")
});
static FRIENDLY_NOT_FOUND_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r"not found in the package registry|no solution found when resolving|",
        r"no matching distribution|could not find a version",
    ))
    .expect("valid regex")
});
static FRIENDLY_NO_UVX_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"command not found|uvx: not found|no such file or directory[^\n]*uvx|executable .* not found")
        .expect("valid regex")
});
static FRIENDLY_CRASH_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"modulenotfounderror|importerror|traceback \(most recent call last\)|process exited")
        .expect("valid regex")
});
/// `[Errno 1] Operation not permitted: '/path'` — the EPERM the macOS App
/// Sandbox answers an exec outside the app bundle with.
static EPERM_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\[Errno 1\] Operation not permitted: '([^']*)'").expect("valid regex"));
static MCP_DOC_NAME_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^a-z0-9]+").expect("valid regex"));
static SAFE_FILENAME_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^\w\-]").expect("valid regex"));

// ------------------------------------------------------------------ helpers

/// Python `ValueError(msg)` raised by the service.
fn value_err(msg: impl Into<String>) -> anyhow::Error {
    ModelServiceError::Value(msg.into()).into()
}

/// Python `RuntimeError(msg)` raised by the service.
fn runtime_err(msg: impl Into<String>) -> anyhow::Error {
    ModelServiceError::Runtime(msg.into()).into()
}

/// Python truthiness of an optional string.
fn truthy(value: &Option<String>) -> bool {
    value.as_deref().is_some_and(|s| !s.is_empty())
}

/// A non-empty id, or `None` (Python's `if project_id:`).
fn non_empty(value: Option<&str>) -> Option<&str> {
    value.filter(|s| !s.is_empty())
}

/// Identity comparison of two LLM handles (Python `is`).
fn same_llm(a: &Arc<dyn BaseLLM>, b: &Arc<dyn BaseLLM>) -> bool {
    std::ptr::addr_eq(Arc::as_ptr(a), Arc::as_ptr(b))
}

/// `a is b` where `b` may be `None`.
fn is_llm(a: &Arc<dyn BaseLLM>, b: Option<&Arc<dyn BaseLLM>>) -> bool {
    b.is_some_and(|b| same_llm(a, b))
}

/// `llm`, resolving on device only when it is a model-host handle; a model
/// loaded in this process is returned as is.
fn local_only_handle(llm: Arc<dyn BaseLLM>) -> Arc<dyn BaseLLM> {
    match llm.as_any().downcast_ref::<RemoteLLM>() {
        Some(remote) if !remote.is_local_only() => Arc::new(remote.local_only_copy()),
        _ => llm,
    }
}

/// `_cap_name(capability)` for an optional capability (`None` → `"None"`).
fn cap_label(capability: Option<&str>) -> String {
    capability.map(ModelService::cap_name).unwrap_or_else(|| "None".to_string())
}

/// The text of a caught panic payload.
fn panic_message(panic: &Box<dyn Any + Send>) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Seconds since the UNIX epoch (`time.time()`).
fn now_secs() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// The first `n` characters of `s`.
fn prefix_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Python's `f"{n:,}"`.
fn fmt_thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// `os.path.abspath`: absolute (against the current directory) and lexically
/// normalised, without resolving symlinks.
fn abspath(path: &str) -> String {
    let p = Path::new(path);
    let joined = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().map(|cwd| cwd.join(p)).unwrap_or_else(|_| p.to_path_buf())
    };
    let mut out = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out.to_string_lossy().into_owned()
}

/// Python `json.dumps(value)`: default separators and `ensure_ascii=True`.
fn python_json_dumps(value: &Value) -> String {
    fn write_str(s: &str, out: &mut String) {
        out.push('"');
        for ch in s.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{08}' => out.push_str("\\b"),
                '\u{0c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                    let mut buf = [0u16; 2];
                    for unit in c.encode_utf16(&mut buf) {
                        out.push_str(&format!("\\u{:04x}", unit));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
    fn write(value: &Value, out: &mut String) {
        match value {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => out.push_str(&n.to_string()),
            Value::String(s) => write_str(s, out),
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write(item, out);
                }
                out.push(']');
            }
            Value::Object(map) => {
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_str(k, out);
                    out.push_str(": ");
                    write(v, out);
                }
                out.push('}');
            }
        }
    }
    let mut out = String::new();
    write(value, &mut out);
    out
}

/// A serialization-ready `DocumentInfo` as a JSON value (`model_dump()`).
fn info_value(info: &DocumentInfo) -> Value {
    serde_json::to_value(info).unwrap_or(Value::Null)
}

/// `DocumentInfo(id=did, status="unknown")`.
fn unknown_document(doc_id: &str) -> DocumentInfo {
    DocumentInfo {
        id: Some(doc_id.to_string()),
        path: None,
        status: "unknown".to_string(),
        summary: None,
        filename: None,
    }
}

/// The `server_info` dict recorded from a discovery handshake.
fn server_info_dict(si: &MCPServerInfo) -> Map<String, Value> {
    let mut map = Map::new();
    map.insert("name".into(), json!(si.name));
    map.insert("title".into(), json!(si.title));
    map.insert("version".into(), json!(si.version));
    if truthy(&si.description) {
        map.insert("description".into(), json!(si.description));
    }
    if let Some(icons) = si.icons.as_ref().filter(|icons| !icons.is_empty()) {
        map.insert("icons".into(), Value::Array(icons.clone()));
    }
    map
}

/// Record a successful discovery on `server`: its tools, and the server's
/// self-description (also filling a blank description).
fn apply_discovery(server: &mut MCPServer, result: &MCPConnectResult) {
    server.tools = result.tools.clone();
    if let Some(si) = result.server_info.as_ref() {
        if !truthy(&server.description) && truthy(&si.description) {
            server.description = si.description.clone();
        }
        server.server_info = Some(server_info_dict(si));
    }
}

/// Copy everything a provisioning pass may have changed onto the live record.
fn merge_provisioned(live: &mut MCPServer, server: &MCPServer) {
    live.name = server.name.clone();
    live.pkg = server.pkg.clone();
    live.description = server.description.clone();
    live.summary = server.summary.clone();
    live.tags = server.tags.clone();
    live.config = server.config.clone();
    live.tools = server.tools.clone();
    live.server_info = server.server_info.clone();
    live.oauth_credentials = server.oauth_credentials.clone();
    live.status = server.status;
    live.last_error = server.last_error.clone();
    live.account = server.account.clone();
}

/// How a discovery pass failed (Python's `except` ladder).
enum DiscoveryFailure {
    /// `MCPAuthExpiredError`: already marked and explained on the record.
    AuthExpired,
    /// `MCPDiscoveryError`: the message is already user-facing.
    Discovery(String),
    /// Anything else: a raw error to be made friendly.
    Other(String),
}

fn classify_discovery_error(err: &anyhow::Error) -> DiscoveryFailure {
    if err.downcast_ref::<MCPAuthExpiredError>().is_some() {
        return DiscoveryFailure::AuthExpired;
    }
    if let Some(e) = err.downcast_ref::<MCPDiscoveryError>() {
        return DiscoveryFailure::Discovery(e.0.clone());
    }
    if let Some(e) = err.downcast_ref::<MCPClientError>() {
        return match e {
            MCPClientError::AuthExpired(_) => DiscoveryFailure::AuthExpired,
            MCPClientError::Discovery(d) => DiscoveryFailure::Discovery(d.0.clone()),
            other => DiscoveryFailure::Other(other.to_string()),
        };
    }
    DiscoveryFailure::Other(err.to_string())
}

/// True when `$HOME` (or `SLP_HOME`) sits inside the macOS App Sandbox
/// container.
fn under_app_sandbox() -> bool {
    let in_container = |value: Option<String>| {
        value.is_some_and(|v| v.find(SANDBOX_HOME_MARKER).is_some_and(|index| index > 0))
    };
    in_container(std::env::var("HOME").ok()) || in_container(std::env::var("SLP_HOME").ok())
}

/// An LLM that cannot generate: the `arg_extractor`/`result_mapper`-less
/// `MCPToolCaller(None, None, ...)` the transport helpers are built with.
struct TransportOnlyLLM {
    cancelled: parking_lot::Mutex<Option<CancelEvent>>,
}

impl TransportOnlyLLM {
    fn new() -> Self {
        Self { cancelled: parking_lot::Mutex::new(None) }
    }
}

impl BaseLLM for TransportOnlyLLM {
    fn project_id(&self) -> &str {
        ""
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }

    fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        messages
    }

    fn get_max_tokens(&self) -> usize {
        DEFAULT_MAX_CONTEXT_TOKENS
    }

    fn call(&self, _prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
        anyhow::bail!("no model behind a transport-only MCP tool caller")
    }

    fn stream<'a>(&'a self, _prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
        Box::new(std::iter::empty())
    }

    fn invoke(&self, _prompt: &str, _max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
        "No response generated".to_string()
    }

    fn cancelled(&self) -> Option<CancelEvent> {
        self.cancelled.lock().clone()
    }

    fn set_cancelled(&self, event: Option<CancelEvent>) {
        *self.cancelled.lock() = event;
    }

    fn llm_type(&self) -> &str {
        "transport-only"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `MCPToolCaller(None, None, persist=...)` as the read-only caller the
/// account probe takes.
struct PersistingToolCaller(MCPToolCaller);

impl ReadOnlyToolCaller for PersistingToolCaller {
    fn call_with_refresh(
        &self,
        server: &mut MCPServer,
        tool_name: &str,
        arguments: &Map<String, Value>,
    ) -> anyhow::Result<(MCPToolCallResult, bool)> {
        // A transport that panics is the Rust analogue of one that raised.
        catch_unwind(AssertUnwindSafe(|| self.0.call_with_refresh(server, tool_name, Some(arguments), None)))
            .map_err(|panic| anyhow::anyhow!(panic_message(&panic)))
    }
}

/// RAII guard for `ModelService::inference_slot` (Python `@contextmanager
/// _inference_slot`): holds one inference permit and counts the model as in
/// use; on drop it decrements the active count and stamps the last-inference
/// time read by `idle_seconds`.
pub struct InferenceSlot<'a> {
    pub(crate) service: &'a ModelService,
}

impl Drop for InferenceSlot<'_> {
    fn drop(&mut self) {
        {
            let mut state = self.service.inference_state.lock();
            state.active = state.active.saturating_sub(1);
            state.last_inference_at = now_secs();
        }
        self.service.inference_lock.release();
    }
}

// ------------------------------------------------------------ streaming

/// What the generation thread sends back.
enum StreamMsg {
    Chunk(GenerationChunk),
    /// The model raised (panicked) mid-generation.
    Failed(String),
    Done,
}

/// Run `llm.stream(prompt)` on its own thread, handing chunks over one at a
/// time.  A `GenerationStream` borrows its model and is drained on the thread
/// that started it, so the thread owns the handle; the rendezvous channel keeps
/// generation in step with the consumer, and a dropped consumer stops it at the
/// next chunk.
fn spawn_llm_stream(
    llm: Arc<dyn BaseLLM>,
    prompt: String,
    options: CallOptions,
    cancelled: Option<CancelEvent>,
) -> Receiver<StreamMsg> {
    let (tx, rx): (SyncSender<StreamMsg>, Receiver<StreamMsg>) = sync_channel(0);
    let job = crate::workspace::context::bind(move || {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            for chunk in llm.stream(&prompt, options) {
                if cancelled.as_ref().is_some_and(|c| c.load(Ordering::SeqCst)) {
                    break;
                }
                if tx.send(StreamMsg::Chunk(chunk)).is_err() {
                    return;
                }
            }
        }));
        let last = match outcome {
            Ok(()) => StreamMsg::Done,
            Err(panic) => StreamMsg::Failed(panic_message(&panic)),
        };
        let _ = tx.send(last);
    });
    let spawned = std::thread::Builder::new().name("slp-stream".into()).spawn(job);
    if let Err(e) = spawned {
        let (tx, rx) = sync_channel(1);
        let _ = tx.send(StreamMsg::Failed(format!("could not start generation: {e}")));
        return rx;
    }
    rx
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StreamPhase {
    NotStarted,
    Primary,
    Fallback,
    Finished,
}

/// The generator behind [`ModelService::stream_completion`].
struct CompletionStream<'a> {
    service: &'a ModelService,
    llm: Arc<dyn BaseLLM>,
    fallback: Option<Arc<dyn BaseLLM>>,
    prompt: String,
    thinking: Option<bool>,
    cancelled: Option<CancelEvent>,
    slot: Option<InferenceSlot<'a>>,
    rx: Option<Receiver<StreamMsg>>,
    emitted: usize,
    phase: StreamPhase,
}

impl CompletionStream<'_> {
    fn is_cancelled(&self) -> bool {
        self.cancelled.as_ref().is_some_and(|c| c.load(Ordering::SeqCst))
    }

    /// The generator's `finally`: release the slot and clear cancellation.
    fn finish(&mut self) {
        if self.phase == StreamPhase::Finished {
            return;
        }
        self.phase = StreamPhase::Finished;
        self.rx = None;
        if let Some(fallback) = self.fallback.take() {
            fallback.set_cancelled(None);
        }
        self.llm.set_cancelled(None);
        self.slot = None;
    }
}

impl Iterator for CompletionStream<'_> {
    type Item = anyhow::Result<GenerationChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.phase == StreamPhase::Finished {
            return None;
        }
        if self.phase == StreamPhase::NotStarted {
            // Lock around inference - llama-cpp is not thread-safe.  A cloud
            // model holds no context, so it doesn't queue behind local ones.
            if !ModelService::is_cloud_llm(self.llm.as_ref()) {
                self.slot = Some(self.service.inference_slot());
            }
            // Propagate cancellation event so the LLM can stop mid-inference.
            self.llm.set_cancelled(self.cancelled.clone());
            self.rx = Some(spawn_llm_stream(
                self.llm.clone(),
                self.prompt.clone(),
                CallOptions { max_tokens: None, thinking: self.thinking },
                self.cancelled.clone(),
            ));
            self.phase = StreamPhase::Primary;
        }
        loop {
            let msg = match self.rx.as_ref().map(|rx| rx.recv()) {
                Some(Ok(msg)) => msg,
                _ => StreamMsg::Done,
            };
            match msg {
                StreamMsg::Chunk(chunk) => {
                    if self.is_cancelled() {
                        self.finish();
                        return None;
                    }
                    if self.phase == StreamPhase::Primary {
                        self.emitted += 1;
                    }
                    return Some(Ok(chunk));
                }
                StreamMsg::Done => {
                    self.finish();
                    return None;
                }
                StreamMsg::Failed(ex) => {
                    // A cloud model's failure is a message for the user (sign
                    // in, plan, busy), not a GPU fault: no eviction, and no
                    // quiet answer from a different model.
                    let fallback = if self.phase == StreamPhase::Primary
                        && !ModelService::is_cloud_llm(self.llm.as_ref())
                    {
                        self.service.default_fallback_llm(&self.llm)
                    } else {
                        None
                    };
                    // A larger response model can OOM the GPU once generation
                    // allocates its compute buffers.  Re-route to the default
                    // router model, which is small enough to fit — but only
                    // when nothing has been streamed yet, since a mid-stream
                    // restart would duplicate the emitted text.
                    match fallback {
                        Some(fallback) if self.emitted == 0 => {
                            tracing::warn!(
                                target: LOG,
                                "[response] Response model failed during generation ({ex}) — falling back to the default model"
                            );
                            self.service.evict_response_llm(&self.llm);
                            self.llm.set_cancelled(None);
                            fallback.set_cancelled(self.cancelled.clone());
                            self.rx = Some(spawn_llm_stream(
                                fallback.clone(),
                                self.prompt.clone(),
                                CallOptions::default(),
                                self.cancelled.clone(),
                            ));
                            self.fallback = Some(fallback);
                            self.phase = StreamPhase::Fallback;
                        }
                        _ => {
                            self.finish();
                            return Some(Err(anyhow::anyhow!(ex)));
                        }
                    }
                }
            }
        }
    }
}

impl Drop for CompletionStream<'_> {
    fn drop(&mut self) {
        if self.phase != StreamPhase::NotStarted {
            self.finish();
        }
    }
}

impl ModelService {
    // ---------------------------------------------------------------- seams

    /// Read one test seam without holding the lock across the call.
    fn hook<T>(&self, pick: impl FnOnce(&ModelServiceHooks) -> Option<T>) -> Option<T> {
        let hooks = self.hooks.read();
        pick(&hooks)
    }

    fn load_model_seam(&self) -> anyhow::Result<Arc<dyn BaseLLM>> {
        match self.hook(|h| h.load_model.clone()) {
            Some(load) => load(),
            None => self.load_model(),
        }
    }

    fn pinned_model_seam(&self, capability: Option<&str>) -> Option<WorkspaceModel> {
        match self.hook(|h| h.pinned_model.clone()) {
            Some(pinned) => pinned(capability),
            None => self.pinned_model(capability),
        }
    }

    fn pinned_llm_seam(&self, capability: Option<&str>) -> Option<Arc<dyn BaseLLM>> {
        match self.hook(|h| h.pinned_llm.clone()) {
            Some(pinned) => pinned(capability),
            None => self.pinned_llm(capability),
        }
    }

    fn llm_for_capability_seam(&self, capability: &str) -> Option<Arc<dyn BaseLLM>> {
        match self.hook(|h| h.get_llm_for_capability.clone()) {
            Some(resolve) => resolve(capability),
            None => self.get_llm_for_capability(capability),
        }
    }

    fn large_llm_seam(&self) -> Option<Arc<dyn BaseLLM>> {
        match self.hook(|h| h.get_large_llm.clone()) {
            Some(resolve) => resolve(),
            None => self.get_large_llm(),
        }
    }

    fn setup_project_tools_seam(&self, project_id: &str) -> anyhow::Result<()> {
        match self.hook(|h| h.setup_project_tools.clone()) {
            Some(setup) => setup(project_id),
            None => self.setup_project_tools(project_id),
        }
    }

    /// `_get_project(project_id)`, honouring the `get_project` seam.
    fn project_by_id(&self, project_id: &str) -> anyhow::Result<Project> {
        match self.hook(|h| h.get_project.clone()) {
            Some(get) => get(project_id),
            None => self.get_project(project_id),
        }
    }

    // --------------------------------------------------------------- helpers

    /// `self._model`, or `RuntimeError("Model not loaded")`.
    fn model_or_err(&self) -> anyhow::Result<Arc<Model>> {
        self.model.read().clone().ok_or_else(|| runtime_err("Model not loaded"))
    }

    /// `self._model.save()`, honouring the `save_model` seam.
    fn save_model(&self) -> anyhow::Result<()> {
        self.persist_model()
    }

    /// A snapshot of `self._model.workspace.current_project`.
    fn current_project_snapshot(&self) -> Option<Project> {
        let model = self.model.read().clone()?;
        let workspace = model.workspace.read();
        workspace.current_project().cloned()
    }

    /// Re-read a project from the workspace this service holds.
    fn live_project(&self, project_id: &str) -> Option<Project> {
        let model = self.model.read().clone()?;
        let workspace = model.workspace.read();
        workspace.project(project_id).cloned()
    }

    /// Apply `f` to the project as held by the workspace; `None` when this
    /// workspace does not hold it.
    fn with_project_mut<R>(&self, project_id: &str, f: impl FnOnce(&mut Project) -> R) -> anyhow::Result<Option<R>> {
        let model = self.model_or_err()?;
        let mut workspace = model.workspace.write();
        Ok(workspace.project_mut(project_id).map(f))
    }

    /// Apply `f` to the live project and persist it as one step.
    ///
    /// A chat request opens with `workspace.reload()`, which brings every
    /// record back to what is on disk.  Mutating under one lock and saving
    /// under another left a gap for that reload to land in: the mutation was
    /// reverted in memory and the save that followed wrote the reverted state
    /// out, so the change was lost on disk too.  Holding the write lock across
    /// the save closes the gap.  (A `save_model` seam cannot run under the
    /// lock — it may read the workspace — so it keeps the two-step form.)
    fn with_project_mut_saved<R>(
        &self,
        project_id: &str,
        f: impl FnOnce(&mut Project) -> R,
    ) -> anyhow::Result<Option<R>> {
        if self.hook(|h| h.save_model.clone()).is_some() {
            let result = self.with_project_mut(project_id, f)?;
            self.save_model()?;
            return Ok(result);
        }
        let model = self.model_or_err()?;
        let mut workspace = model.workspace.write();
        let result = workspace.project_mut(project_id).map(f);
        workspace.save()?;
        Ok(result)
    }

    /// Find a project by id, or the current one, raising `missing` when there
    /// is none.  `initialize_first` mirrors call sites that resolve the
    /// workspace even when an id is given.
    fn resolve_project(&self, project_id: Option<&str>, missing: &str, initialize_first: bool) -> anyhow::Result<Project> {
        if initialize_first {
            self.initialize_workspace()?;
        }
        match non_empty(project_id) {
            Some(pid) => self.project_by_id(pid),
            None => {
                if !initialize_first {
                    self.initialize_workspace()?;
                }
                self.current_project_snapshot().ok_or_else(|| value_err(missing))
            }
        }
    }

    /// The logger the background tasks report to.
    fn task_logger() -> TaskLogger {
        Arc::new(get_logger(LOG))
    }

    /// The resident orchestrator model (`self._llm`).
    fn resident_llm(&self) -> Option<Arc<dyn BaseLLM>> {
        self.llm.read().clone()
    }

    // ------------------------------------------------------ response routing

    /// `_cap_name(capability)` for an optional capability: the value, or
    /// `"None"` when no capability was asked for (as Python logged it).
    pub fn cap_name_opt(capability: Option<&str>) -> String {
        cap_label(capability)
    }

    /// Return the LLM to use for the final response.
    ///
    /// With exactly one model selected, that model writes the response —
    /// including when `capability` is `None`, which otherwise answers on the
    /// bootstrap orchestrator LLM (a vision-only selection writes only the
    /// response to a vision-tagged step).  When it cannot be loaded the default
    /// model answers instead: the rest of the chain below would only rediscover
    /// the same unavailable model, and an answer on the default beats no answer
    /// at all.
    ///
    /// Fallback chain when `capability` is set: capability-tagged model →
    /// largest downloaded model → default orchestrator LLM.  Small models
    /// struggle to synthesize long tool-result contexts, so any larger model is
    /// preferred when one is available.
    pub fn llm_for_response(&self, capability: Option<&str>) -> Option<Arc<dyn BaseLLM>> {
        if let Some(resolve) = self.hook(|h| h.llm_for_response.clone()) {
            return resolve(capability);
        }
        if let Some(pinned_model) = self.pinned_model_seam(capability) {
            if let Some(pinned) = self.pinned_llm_seam(capability) {
                // DEBUG, not INFO: this resolution runs for *every* generation
                // the process serves, not only the final answer.  Which model
                // wrote the answer is reported once, by `report_response_model`.
                tracing::debug!(
                    target: LOG,
                    "[response] Using the sole selected model (capability='{}')",
                    cap_label(capability)
                );
                self.report_response_model(capability, Some(&pinned));
                return Some(pinned);
            }
            tracing::warn!(
                target: LOG,
                "[response] Sole selected model {} unavailable — answering on the default model",
                pinned_model.model_name
            );
            let default = self.resident_llm();
            self.report_response_model(capability, default.as_ref());
            return default;
        }

        if let Some(cap) = capability.filter(|c| !c.is_empty()) {
            // DEBUG for the same reason as above: a project agent's remote
            // handle asks for the capability's context window, whether it
            // reasons and how to label it before generating anything.  The
            // outcome is reported once, where it belongs.
            let cap_llm = self.llm_for_capability_seam(cap);
            if let Some(cap_llm) = cap_llm
                && !is_llm(&cap_llm, self.resident_llm().as_ref())
            {
                tracing::debug!(target: LOG, "[response] Using capability='{}' LLM", Self::cap_name(cap));
                self.report_response_model(capability, Some(&cap_llm));
                return Some(cap_llm);
            }
            let large = self.large_llm_seam();
            if let Some(large) = large
                && !is_llm(&large, self.resident_llm().as_ref())
            {
                tracing::debug!(
                    target: LOG,
                    "[response] Capability='{}' unavailable — falling back to large LLM",
                    Self::cap_name(cap)
                );
                self.report_response_model(capability, Some(&large));
                return Some(large);
            }
            tracing::debug!(
                target: LOG,
                "[response] Capability='{}' unavailable and no larger model — using default LLM",
                Self::cap_name(cap)
            );
        }
        let default = self.resident_llm();
        self.report_response_model(capability, default.as_ref());
        default
    }

    /// The default router LLM to fall back to when a larger response model
    /// fails during generation, or `None` when the failed model already IS the
    /// default (there's nothing smaller/lighter to retry on).
    pub fn default_fallback_llm(&self, failed_llm: &Arc<dyn BaseLLM>) -> Option<Arc<dyn BaseLLM>> {
        let default = self.resident_llm();
        if is_llm(failed_llm, default.as_ref()) {
            return None;
        }
        default
    }

    /// Stop routing responses to a model that faulted during generation.
    ///
    /// A larger model can pass the load-time fit guard yet still exhaust GPU
    /// memory once generation allocates its compute buffers (e.g. Vulkan
    /// `ErrorOutOfDeviceMemory`).  Drop it from the response caches so later
    /// requests route straight to the default model instead of re-loading a
    /// model that won't fit the GPU.
    pub fn evict_response_llm(&self, llm: &Arc<dyn BaseLLM>) {
        let large = self.large_llm.read().clone();
        if is_llm(llm, large.as_ref()) {
            tracing::warn!(
                target: LOG,
                "[response] Large model {} faulted on GPU — evicting; routing responses to the default model",
                self.large_model_name.read().clone().unwrap_or_else(|| "None".into())
            );
            *self.large_llm.write() = None;
            *self.large_model_name.write() = None;
            // Latch so get_large_llm doesn't re-attempt the load this session.
            self.large_model_attempted.store(true, Ordering::SeqCst);
            return;
        }
        let mut slots = self.cap_llms.lock();
        for (name, slot) in slots.iter_mut() {
            if slot.llm().is_some_and(|cached| same_llm(cached, llm)) {
                tracing::warn!(
                    target: LOG,
                    "[response] Capability model {name} faulted on GPU — evicting; routing responses to the default model"
                );
                *slot = GPU_FAULTED;
                return;
            }
        }
    }

    /// Stream completion from the response LLM, holding an inference slot.
    ///
    /// Uses the inference lock because llama.cpp does not support concurrent
    /// inference on one context.  `model_name` is unused (caller
    /// compatibility).  `cancelled` is checked between chunks.  When the
    /// response model fails before emitting anything the request is retried on
    /// the default model.  Errors with
    /// `"Model not loaded. Call load_model() first."` when no model is loaded.
    pub fn stream_completion<'a>(
        &'a self,
        prompt: &str,
        model_name: Option<&str>,
        cancelled: Option<CancelEvent>,
        capability: Option<&str>,
        thinking: Option<bool>,
    ) -> anyhow::Result<Box<dyn Iterator<Item = anyhow::Result<GenerationChunk>> + 'a>> {
        let _ = model_name;
        let llm = self
            .llm_for_response(capability)
            .ok_or_else(|| runtime_err("Model not loaded. Call load_model() first."))?;

        tracing::info!(
            target: LOG,
            "[models] RESPONSE → {} (capability={})",
            Self::model_label(llm.as_ref()),
            cap_label(capability)
        );

        Ok(Box::new(CompletionStream {
            service: self,
            llm,
            fallback: None,
            prompt: prompt.to_string(),
            thinking,
            cancelled,
            slot: None,
            rx: None,
            emitted: 0,
            phase: StreamPhase::NotStarted,
        }))
    }

    /// Generate a non-streaming completion on the response LLM, falling back
    /// to the default model when a larger one fails.
    ///
    /// `max_tokens` and `thinking` belong to the request, and travel with it
    /// from a project agent through the model host.
    pub fn generate_completion(
        &self,
        prompt: &str,
        model_name: Option<&str>,
        capability: Option<&str>,
        max_tokens: Option<usize>,
        thinking: Option<bool>,
    ) -> anyhow::Result<String> {
        let _ = model_name;
        let llm = self
            .llm_for_response(capability)
            .ok_or_else(|| runtime_err("Model not loaded. Call load_model() first."))?;

        if ModelService::is_cloud_llm(llm.as_ref()) {
            // `invoke` would turn a refusal into answer text; keep it an error.
            return llm.call(prompt, None, CallOptions { max_tokens, thinking }).map(|text| text.trim().to_string());
        }
        let _slot = self.inference_slot();
        match catch_unwind(AssertUnwindSafe(|| llm.invoke(prompt, max_tokens, thinking))) {
            Ok(text) => Ok(text),
            Err(panic) => {
                let ex = panic_message(&panic);
                // Same GPU-OOM fallback as stream_completion: retry the
                // non-streaming response on the default router model.
                let Some(fallback) = self.default_fallback_llm(&llm) else {
                    return Err(anyhow::anyhow!(ex));
                };
                tracing::warn!(
                    target: LOG,
                    "[response] Response model failed during generation ({ex}) — falling back to the default model"
                );
                self.evict_response_llm(&llm);
                catch_unwind(AssertUnwindSafe(|| fallback.invoke(prompt, None, None)))
                    .map_err(|panic| anyhow::anyhow!(panic_message(&panic)))
            }
        }
    }

    /// [`Self::generate_completion`] for several independent prompts,
    /// generated together on the device (see `BaseLLM::invoke_batch`).
    ///
    /// Holds one inference slot for the whole batch: the batch is one job on
    /// the device, however many sequences it carries.
    pub fn generate_batch_completion(
        &self,
        prompts: &[BatchPrompt],
        model_name: Option<&str>,
        capability: Option<&str>,
        cancelled: Option<CancelEvent>,
    ) -> anyhow::Result<Vec<String>> {
        let _ = model_name;
        let llm = self
            .llm_for_response(capability)
            .ok_or_else(|| runtime_err("Model not loaded. Call load_model() first."))?;

        let _slot = self.inference_slot();
        match catch_unwind(AssertUnwindSafe(|| llm.invoke_batch(prompts, cancelled.clone()))) {
            Ok(texts) => Ok(texts),
            Err(panic) => {
                let ex = panic_message(&panic);
                // Same GPU-OOM fallback as generate_completion.
                let Some(fallback) = self.default_fallback_llm(&llm) else {
                    return Err(anyhow::anyhow!(ex));
                };
                tracing::warn!(
                    target: LOG,
                    "[response] Response model failed during batch generation ({ex}) — falling back to the default model"
                );
                self.evict_response_llm(&llm);
                catch_unwind(AssertUnwindSafe(|| fallback.invoke_batch(prompts, cancelled)))
                    .map_err(|panic| anyhow::anyhow!(panic_message(&panic)))
            }
        }
    }

    // --------------------------------------------------------------- documents

    /// Add a document source (file path or URL) to a project (by ID or
    /// current).  Returns `{"added": [...]}`.
    ///
    /// `project_id` is *bound* for the call rather than written to the
    /// `current` flag: the model resolves the project through
    /// `Workspace::current_project`, which reads the binding before the flag.
    pub fn add_document(
        &self,
        source: &str,
        project_id: Option<&str>,
        progress_callback: Option<DocumentProgressCallback>,
    ) -> anyhow::Result<Value> {
        self.initialize_workspace()?;
        let _binding = use_project(project_id);
        let project = match non_empty(project_id) {
            Some(pid) => Some(self.project_by_id(pid)?),
            None => self.current_project_snapshot(),
        };
        let project = project.ok_or_else(|| value_err("No active project"))?;
        let model = self.model_or_err()?;
        let new_docs = model.add_source(DocumentSource::from(source), progress_callback)?;
        let added: Vec<Value> =
            new_docs.iter().map(|doc| info_value(&doc.to_info(Some(project.project_dir.as_str())))).collect();
        Ok(json!({ "added": added }))
    }

    /// Register a document with PENDING status, enqueue it on the indexer and
    /// return immediately (`{"status": "pending"|"skipped", ...}`).
    pub fn add_document_async(
        &self,
        source: &str,
        project_id: Option<&str>,
        cwd: Option<&str>,
        generate_filename: bool,
    ) -> anyhow::Result<Value> {
        self.initialize_workspace()?;

        // Bound, not flipped: see `add_document`.
        let _binding = use_project(project_id);
        let project = match non_empty(project_id) {
            Some(pid) => Some(self.project_by_id(pid)?),
            None => self.current_project_snapshot(),
        };
        let project = project.ok_or_else(|| value_err("No active project"))?;

        // Register the document with PENDING status (no heavy processing).
        let model = self.model_or_err()?;
        let slp_model = model.slp_model.clone().ok_or_else(|| runtime_err("Model not loaded"))?;
        let new_docs = slp_model.add_document(DocumentSource::from(source), None, cwd, generate_filename)?;

        if new_docs.is_empty() {
            return Ok(json!({
                "status": "skipped",
                "message": "Document(s) already exist in the project",
                "documents": [],
            }));
        }

        // Enqueue BEFORE saving config so that when the watcher detects the
        // config change the queue already contains the document IDs.
        let indexer = self.task_queue.read().get_task_as::<IndexerTask>("indexer");
        if let Some(indexer) = indexer {
            indexer.enqueue(&new_docs);
        }

        // Persist config so the watcher also sees the change as a fallback.
        model.save()?;

        let documents: Vec<Value> =
            new_docs.iter().map(|doc| info_value(&doc.to_info(Some(project.project_dir.as_str())))).collect();
        Ok(json!({
            "status": "pending",
            "message": format!("{} document(s) queued for processing", new_docs.len()),
            "documents": documents,
        }))
    }

    /// Drop an uploaded file into the project's documents inbox folder.
    ///
    /// The indexer's scan loop picks it up, registers it and starts processing
    /// automatically.  Returns immediately with `status=accepted`.  `cwd` is
    /// accepted for API compatibility and ignored: uploads always land in the
    /// project's own documents folder.
    pub fn upload_document_to_inbox(
        &self,
        project_id: &str,
        filename: &str,
        file_bytes: &[u8],
        cwd: Option<&str>,
    ) -> anyhow::Result<Value> {
        let _ = cwd;
        let project = self.project_by_id(project_id)?;
        let inbox_dir = PathBuf::from(Self::project_documents_dir(&project));
        std::fs::create_dir_all(&inbox_dir)?;

        let safe_name = Path::new(filename)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| "upload".to_string());
        let mut dest = inbox_dir.join(&safe_name);
        if dest.exists() {
            let stem = dest.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            let suffix = dest.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
            let mut counter = 1;
            while dest.exists() {
                dest = inbox_dir.join(format!("{stem}_{counter}{suffix}"));
                counter += 1;
            }
        }
        std::fs::write(&dest, file_bytes)?;
        let dest_name = dest.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        tracing::info!(target: LOG, "[upload] Dropped {dest_name} into inbox for project {project_id}");
        Ok(json!({
            "status": "accepted",
            "filename": dest_name,
            "path": dest.to_string_lossy(),
        }))
    }

    /// Fresh status for each document ID, re-reading the workspace from disk
    /// so the caller gets the latest state written by the indexer thread;
    /// unknown IDs get status `"unknown"`.
    pub fn get_document_statuses(
        &self,
        doc_ids: &[String],
        project_id: Option<&str>,
    ) -> anyhow::Result<Vec<DocumentInfo>> {
        let Some(workspace) = Workspace::load() else {
            return Ok(doc_ids.iter().map(|id| unknown_document(id)).collect());
        };
        let project = match non_empty(project_id) {
            Some(pid) => workspace.projects.iter().find(|p| p.id == pid),
            None => workspace.current_project(),
        };
        let Some(project) = project else {
            return Ok(doc_ids.iter().map(|id| unknown_document(id)).collect());
        };
        let doc_map: HashMap<&str, &Document> = project.documents.iter().map(|d| (d.id.as_str(), d)).collect();
        Ok(doc_ids
            .iter()
            .map(|id| match doc_map.get(id.as_str()) {
                Some(doc) => doc.to_info(Some(project.project_dir.as_str())),
                None => unknown_document(id),
            })
            .collect())
    }

    // -------------------------------------------------------------- Task Queue

    /// Register (once) and return the downloader task.
    pub fn ensure_downloader_task(
        &self,
        model_id: &str,
        progress_callback: Option<DownloadProgressCallback>,
    ) -> anyhow::Result<Arc<DownloaderTask>> {
        let queue = self.task_queue.read().clone();
        if let Some(existing) = queue.get_task_as::<DownloaderTask>("downloader") {
            return Ok(existing);
        }

        let workspace: SharedWorkspace = match self.model.read().clone() {
            Some(model) => model.workspace.clone(),
            None => self.ensure_workspace(None)?,
        };

        // Not gated on credentials.  The platform serves free models to
        // anonymous callers and is the only authority on the rest, so a
        // logged-out user downloading a free model belongs here.
        let downloader = SLPArtifactDownloader::new(workspace.clone(), Self::task_logger(), None);

        let weak = self.weak_self.clone();
        let on_download_complete: crate::tasks::base_task::StatusCallback =
            Arc::new(move |_task: &str, status: TaskStatus, _message: Option<&str>| {
                if status == TaskStatus::Completed
                    && let Some(service) = weak.upgrade()
                {
                    // Re-arm large-model detection so the just-downloaded
                    // model is considered on the next orchestration request.
                    service.large_model_attempted.store(false, Ordering::SeqCst);
                }
            });

        let task = Arc::new(DownloaderTask::new(
            workspace,
            Arc::new(downloader),
            Self::task_logger(),
            model_id,
            progress_callback,
            false,
            TaskOptions { max_retries: None, on_status_change: Some(on_download_complete) },
        ));
        queue.register(task.clone() as Arc<dyn BaseTask>);
        Ok(task)
    }

    /// Set up all background tasks for the workspace (indexer, MCP) and start the auto tasks.  Called during API server
    /// startup.
    pub fn setup_tasks(&self, workspace: Option<SharedWorkspace>) -> anyhow::Result<()> {
        let workspace: SharedWorkspace = match workspace {
            Some(workspace) => workspace,
            None => match self.model.read().clone() {
                Some(model) => model.workspace.clone(),
                None => self.ensure_workspace(None)?,
            },
        };
        let queue = self.task_queue.read().clone();

        // Document conversion uses MarkItDown-style converters (no model
        // pre-download).  OCR / image understanding is handled on demand by
        // the vision-capable model loaded in-process via get_vision_llm().

        // Indexer task (auto — watches for pending documents).
        //
        // Not in a process that has delegated its projects.  Each child runs
        // an IndexerTask pinned to its own project; one here would own no
        // project, fall back to the `current` flag, and index whatever a child
        // just wrote a second time.  Nothing is lost by skipping it: a
        // delegating save keeps only the workspace record, so the document
        // state this task exists to persist would be discarded anyway.
        let indexer = if !projects_delegated() {
            let indexer = IndexerTask::new(workspace.clone(), Self::task_logger(), None, None, TaskOptions::default());
            queue.register(indexer.clone() as Arc<dyn BaseTask>);
            Some(indexer)
        } else {
            None
        };

        // MCP task (auto — owns background provisioning of newly registered
        // servers).
        let weak = self.weak_self.clone();
        let provisioner: crate::tasks::mcp_task::Provisioner = Arc::new(move |project_id: &str, server_id: &str| {
            let service = weak.upgrade().ok_or_else(|| anyhow::anyhow!("model service is gone"))?;
            service.provision_mcp_server(project_id, server_id)
        });
        let weak = self.weak_self.clone();
        let pending_provider: crate::tasks::mcp_task::PendingProvider = Arc::new(move || {
            let service = weak.upgrade().ok_or_else(|| anyhow::anyhow!("model service is gone"))?;
            service.pending_mcp_servers()
        });
        let mcp = MCPTask::new(Self::task_logger(), Some(provisioner), Some(pending_provider), TaskOptions::default());
        queue.register(mcp.clone() as Arc<dyn BaseTask>);

        // Build / train tasks live in the optional trainer package, which this
        // build does not ship (`BuildTask is None`), so neither is registered.

        // Start auto tasks (indexer watches for changes).
        queue.start_auto_tasks();

        // Start the indexer watcher thread.
        if let Some(indexer) = indexer {
            indexer.start_watcher();
        }

        // Start the MCP loop: it adopts any server left pending by a previous
        // run.
        mcp.start_loop();

        tracing::info!(target: LOG, "All tasks initialized and auto-tasks started");
        Ok(())
    }

    /// Gracefully shut down all tasks.  Called during API server shutdown.
    pub fn shutdown_tasks(&self) {
        let queue = self.task_queue.read().clone();
        queue.shutdown();
        self.reset_llm_executor();
    }

    /// Delete a document by ID from a project (by ID or current).
    ///
    /// Removes it from the vector store, deletes the parquet and source files,
    /// removes it from the manifest and the project's document list, and saves.
    pub fn delete_document(&self, document_id: &str, project_id: Option<&str>) -> anyhow::Result<Value> {
        let project = match non_empty(project_id) {
            Some(pid) => self.project_by_id(pid)?,
            None => {
                if self.model.read().is_none() {
                    return Err(runtime_err("Model not loaded"));
                }
                self.current_project_snapshot().ok_or_else(|| value_err("No current project found"))?
            }
        };

        // Find the document.
        let doc = project
            .documents
            .iter()
            .find(|d| d.id == document_id)
            .cloned()
            .ok_or_else(|| value_err(format!("Document not found: {document_id}")))?;

        let model = self.model_or_err()?;
        let workspace_dir = model.workspace.read().workspace_dir.clone();

        // 1. Delete from vector store.
        let store = StoreFactory::create_store(&project.project_dir, "documents", Some(&workspace_dir), None)?;
        store.delete_document(document_id)?;

        // 2. Delete parquet file.
        let parquet_path = Path::new(&project.project_dir).join("datasets").join(format!("{document_id}.parquet"));
        if parquet_path.exists() {
            std::fs::remove_file(&parquet_path)?;
            tracing::info!(target: LOG, "Deleted parquet: {}", parquet_path.display());
        }

        // 3. Delete the original file from the user-visible documents folder.
        if !doc.path.is_empty() && Path::new(&doc.path).exists() {
            std::fs::remove_file(&doc.path)?;
            tracing::info!(target: LOG, "Deleted document file: {}", doc.path);
        }

        // 4. Remove from the JSONL manifest.
        let manifest = DocumentsManifest::new(Path::new(&project.project_dir).join("documents.jsonl"));
        manifest.remove(document_id);

        // 5. Remove from project documents list.
        self.with_project_mut(&project.id, |p| p.documents.retain(|d| d.id != document_id))?;

        // 6. Save workspace.
        model.save()?;

        Ok(json!({"id": document_id, "deleted": true, "message": "Document deleted successfully"}))
    }

    /// List all documents in the current project.
    pub fn get_documents(&self) -> anyhow::Result<Vec<Value>> {
        if self.model.read().is_none() {
            return Err(runtime_err("Model not loaded"));
        }
        let Some(project) = self.current_project_snapshot() else {
            return Ok(Vec::new());
        };
        Ok(project.documents.iter().map(|doc| info_value(&doc.to_info(Some(project.project_dir.as_str())))).collect())
    }

    /// List documents for a project.
    ///
    /// Reads the project's `documents/` folder as the source of truth for
    /// which files exist, then enriches each entry with status from the
    /// workspace config.  Files dropped into the folder but not yet picked up
    /// by the indexer appear with status=pending and no id; registered
    /// documents whose source is not in the folder (URLs, external paths)
    /// follow.
    pub fn get_documents_for_project(&self, project_id: &str) -> anyhow::Result<Vec<Value>> {
        let project = self.project_by_id(project_id)?;
        let inbox_dir = PathBuf::from(Self::project_documents_dir(&project));

        // Index registered docs by their original inbox path, and by original
        // filename since the indexer may move `doc.path` to a converted copy.
        let mut registered_by_path: HashMap<String, &Document> = HashMap::new();
        let mut registered_by_name: HashMap<String, &Document> = HashMap::new();
        for doc in &project.documents {
            registered_by_path.insert(doc.path.clone(), doc);
            let lookup_name = doc.name.clone().filter(|n| !n.is_empty()).unwrap_or_else(|| {
                Path::new(&doc.path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
            });
            registered_by_name.insert(lookup_name, doc);
        }

        let mut results: Vec<Value> = Vec::new();
        let mut seen_ids: HashSet<String> = HashSet::new();

        if inbox_dir.is_dir() {
            let mut entries: Vec<PathBuf> =
                std::fs::read_dir(&inbox_dir)?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
            entries.sort();
            for file_path in entries {
                if !file_path.is_file() {
                    continue;
                }
                let path_str = file_path.to_string_lossy().into_owned();
                let file_name = file_path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                let doc = registered_by_path.get(&path_str).or_else(|| registered_by_name.get(&file_name));
                match doc {
                    Some(doc) => {
                        seen_ids.insert(doc.id.clone());
                        results.push(info_value(&doc.to_info(Some(project.project_dir.as_str()))));
                    }
                    None => {
                        // In the inbox but not yet registered by the indexer.
                        results.push(json!({
                            "id": null,
                            "path": path_str,
                            "filename": file_name,
                            "status": "pending",
                            "summary": null,
                        }));
                    }
                }
            }
        }

        for doc in &project.documents {
            if !seen_ids.contains(&doc.id) {
                results.push(info_value(&doc.to_info(Some(project.project_dir.as_str()))));
            }
        }
        Ok(results)
    }

    // ---------------------------------------------------------------- projects

    /// Create a new project with the given name and description.
    ///
    /// The project uses the workspace's active model; model download is
    /// handled separately via `/v1/init`.
    pub fn create_project(
        &self,
        name: &str,
        description: Option<&str>,
        system: bool,
        skills: Option<Vec<Skill>>,
        cwd: Option<&str>,
        client: Option<&str>,
    ) -> anyhow::Result<Value> {
        self.ensure_model()?;
        let model = self.model_or_err()?;
        let workspace = model.workspace.clone();

        let project_id = Uuid::new_v4().to_string();
        let workspace_dir = workspace.read().workspace_dir.clone();
        let project_dir = Path::new(&workspace_dir).join(&project_id).to_string_lossy().into_owned();
        std::fs::create_dir_all(&project_dir)?;

        {
            let mut ws = workspace.write();
            // Unset current on all existing projects.
            for p in ws.projects.iter_mut() {
                p.current = false;
            }
            let mut project = Project {
                id: project_id.clone(),
                project_dir: project_dir.clone(),
                current: true,
                name: Some(name.to_string()),
                description: description.map(str::to_string),
                client: client.map(str::to_string),
                system,
                skills: skills.clone().unwrap_or_default(),
                ..Project::default()
            };
            // Seed the project's model settings from the workspace
            // (app-default) model registry.
            ws.sync_detached_project_models(&mut project);
            ws.projects.push(project);
        }
        // No child owns this project yet — it has never been served — so this
        // process writes its first `project.json` itself.
        {
            let _claim = own_project(Some(&project_id));
            model.save()?;
        }

        // A caller-supplied `cwd` is recorded for reference only — nothing is
        // ever located through it.  Skills, documents, MCP docs and the
        // workspace index all live under `project_dir`.
        let (skills_dir, documents_dir) = self
            .with_project_mut(&project_id, |p| {
                if let Some(cwd) = non_empty(cwd) {
                    p.cwd = Some(abspath(&expand_home(cwd)));
                }
                (p.skills_dir(), p.documents_dir())
            })?
            .ok_or_else(|| value_err(format!("Project not found: {project_id}")))?;
        std::fs::create_dir_all(&skills_dir)?;
        std::fs::create_dir_all(&documents_dir)?;
        {
            let _claim = own_project(Some(&project_id));
            model.save()?;
        }
        tracing::info!(target: LOG, "[create_project] State dir: {project_dir}");

        // Drop any slot left over from a project that reused this id.
        self.drop_caches(Some(&project_id));

        // Initialize the index store so the schema is ready before the first
        // query.
        match StoreFactory::create_store(&project_dir, "documents", Some(&workspace_dir), None) {
            Ok(_store) => tracing::info!(target: LOG, "Initialized vector store for project {project_id}"),
            Err(e) => tracing::warn!(target: LOG, "Could not initialize vector store: {e}"),
        }

        // Claimed for the same reason the two saves above are: installing the
        // system MCP server persists it, and a delegating supervisor writes
        // no project file without a claim.
        {
            let _claim = own_project(Some(&project_id));
            self.setup_project_tools_seam(&project_id)?;
        }

        let project = self
            .live_project(&project_id)
            .ok_or_else(|| value_err(format!("Project not found: {project_id}")))?;

        // Write explicitly provided skills to disk.
        if let Some(skills) = skills.as_ref() {
            for skill in skills {
                Self::write_skill_md(&project, skill)?;
            }
        }

        Workspace::reset();

        Ok(self.project_to_dict(&project))
    }

    /// Update an existing project (label, description, skills on disk,
    /// web-search toggle, bound folder).
    ///
    /// Renaming only changes the label: skills and documents live under
    /// `project_dir` (keyed by id), so nothing moves on disk.  `cwd` binds
    /// the user's folder, already resolved by the caller; `""` unbinds it.
    pub fn update_project(
        &self,
        project_id: &str,
        name: Option<&str>,
        description: Option<&str>,
        skills: Option<Vec<Skill>>,
        web_search_enabled: Option<bool>,
        cwd: Option<&str>,
    ) -> anyhow::Result<Value> {
        self.ensure_model()?;
        let updated = self.with_project_mut(project_id, |project| {
            if let Some(name) = non_empty(name)
                && project.name.as_deref() != Some(name)
            {
                project.name = Some(name.to_string());
            }
            if let Some(description) = description {
                project.description = Some(description.to_string());
            }
            if let Some(enabled) = web_search_enabled {
                project.web_search_enabled = enabled;
            }
            if let Some(cwd) = cwd {
                project.cwd = non_empty(Some(cwd)).map(str::to_string);
            }
            project.clone()
        })?;
        let project = updated.ok_or_else(|| value_err(format!("Project not found: {project_id}")))?;

        if let Some(skills) = skills.as_ref() {
            // Write skills to disk rather than config.json.
            for skill in skills {
                Self::write_skill_md(&project, skill)?;
            }
        }
        if let Some(enabled) = web_search_enabled {
            tracing::info!(target: LOG, "[web-search] project {} enabled={}", project.id, py_bool(enabled));
        }

        self.save_model()?;
        Ok(self.project_to_dict(&project))
    }

    /// Convert a Project to a dict for API responses.
    pub fn project_to_dict(&self, p: &Project) -> Value {
        let skills = if truthy(&p.name) { Self::read_skills_from_disk(p) } else { Vec::new() };
        json!({
            "id": p.id,
            "name": p.name,
            "cwd": p.work_dir().unwrap_or_default(),
            "description": p.description,
            "model_name": self.model_name.read().clone(),
            "current": p.current,
            "system": p.system,
            "skills": skills.iter().map(|s| serde_json::to_value(s).unwrap_or(Value::Null)).collect::<Vec<_>>(),
            "mcp": p.mcp_servers.iter().map(Self::mcp_server_to_dict).collect::<Vec<_>>(),
            "documents": p.documents.iter().map(|doc| info_value(&doc.to_info(Some(p.project_dir.as_str())))).collect::<Vec<_>>(),
            "web_search_enabled": p.web_search_enabled,
        })
    }

    /// List all projects in the workspace.
    pub fn list_projects(&self) -> anyhow::Result<Vec<Value>> {
        let Some(ws) = Workspace::load() else {
            return Ok(Vec::new());
        };
        Ok(ws.projects.iter().map(|p| self.project_to_dict(p)).collect())
    }

    /// Delete a project (system projects cannot be deleted).
    ///
    /// Runs in the process that owns the workspace record.  The user's own
    /// `cwd` folder holds no framework state and is never touched.
    pub fn delete_project(&self, project_id: &str) -> anyhow::Result<Value> {
        self.ensure_model()?;
        let model = self.model_or_err()?;
        let workspace = model.workspace.clone();

        let mut project = workspace.read().project(project_id).cloned();
        if project.is_none() {
            // The cached singleton can predate a project created by the CLI or
            // by an import working on its own detached instance.
            workspace.write().reload();
            project = workspace.read().project(project_id).cloned();
        }
        let project = project.ok_or_else(|| value_err(format!("Project not found: {project_id}")))?;

        if project.system {
            return Err(value_err("System projects cannot be deleted"));
        }

        let was_current = project.current;

        // Remove internal project directory.
        if !project.project_dir.is_empty() && Path::new(&project.project_dir).exists() {
            std::fs::remove_dir_all(&project.project_dir)?;
            tracing::info!(target: LOG, "Deleted project directory: {}", project.project_dir);
        }

        // The request arrived bound to this project, and from the next line
        // the project does not exist — resolve as if none were named.
        {
            let _unbound = unbind_project();
            {
                let mut ws = workspace.write();
                ws.projects.retain(|p| p.id != project_id);
                // If the deleted project was current, the first remaining
                // project becomes current.
                if was_current && let Some(first) = ws.projects.first_mut() {
                    first.current = true;
                }
            }

            // Release the deleted project's store handles.
            self.drop_caches(Some(project_id));

            // Claim the project being removed: under supervision this process
            // writes only the workspace record, and the claim says this one
            // file is ours to have stopped keeping.
            {
                let _claim = own_project(Some(project_id));
                model.save()?;
            }

            // Drop the singleton so the next init re-reads what was written.
            Workspace::reset();
        }

        Ok(json!({"id": project_id, "deleted": true, "message": "Project deleted successfully"}))
    }

    /// Record which project the UI has selected.
    ///
    /// Not how a request chooses its project — that is bound per request — so
    /// this cannot move the project out from under a running chat.  What
    /// remains is the selection a client with no `project_id` gets.
    pub fn set_current_project(&self, project_id: &str) -> anyhow::Result<()> {
        self.ensure_model()?;
        let changed = {
            let model = self.model_or_err()?;
            let mut ws = model.workspace.write();
            let target = ws
                .project(project_id)
                .map(|p| p.current)
                .ok_or_else(|| value_err(format!("Project not found: {project_id}")))?;
            // Already selected — nothing to persist.
            if target {
                false
            } else {
                for p in ws.projects.iter_mut() {
                    p.current = p.id == project_id;
                }
                true
            }
        };
        // Store handles are keyed by project, so a selection change no longer
        // invalidates anything.
        if changed {
            self.save_model()?;
        }
        Ok(())
    }

    // ---------------------------------------------------------- model lifecycle

    /// Explicitly close the native llama.cpp context so VRAM is freed
    /// immediately.  Never closes the always-resident orchestrator model.
    pub fn free_llm(&self, llm: Arc<dyn BaseLLM>) {
        if let Some(free) = self.hook(|h| h.free_llm.clone()) {
            free(&llm);
            return;
        }
        if is_llm(&llm, self.resident_llm().as_ref()) {
            return;
        }
        if let Some(model) = llm.llama_model() {
            let _ = catch_unwind(AssertUnwindSafe(|| model.close()));
        }
    }

    /// Record that `model_name`'s weights are present in the workspace, and
    /// persist it.
    ///
    /// `workspace` must be passed by any caller holding unsaved edits (one
    /// that registered the model with `add_model`): on a cold process the
    /// resolved workspace is a fresh instance read off disk without them.
    pub fn mark_model_downloaded(&self, model_name: &str, workspace: Option<SharedWorkspace>) -> anyhow::Result<()> {
        let workspace = match workspace {
            Some(workspace) => workspace,
            None => match self.model.read().clone() {
                Some(model) => model.workspace.clone(),
                None => self.ensure_workspace(None)?,
            },
        };
        {
            let mut ws = workspace.write();
            if let Some(m) = ws.get_model_mut(Some(model_name)) {
                m.downloaded = true;
            }
        }
        workspace.read().save()?;
        Ok(())
    }

    /// Release cached GPU memory after unloading models.
    ///
    /// Python ran `gc.collect()` and emptied torch's CUDA cache here.  Native
    /// memory is released deterministically when the last handle drops, and
    /// there is no torch allocator in this build, so only the test seam does
    /// anything.
    pub fn cleanup_gpu_cache(&self) {
        if let Some(cleanup) = self.hook(|h| h.cleanup_gpu_cache.clone()) {
            cleanup();
        }
    }

    /// Unload the current model to free resources.
    pub fn unload_model(&self) {
        let Some(llm) = self.resident_llm() else {
            return;
        };
        let project_name = self.get_current_project_name().unwrap_or_else(|| "default".to_string());
        tracing::info!(
            target: LOG,
            "Unloading model: {} (project: {project_name})",
            self.model_name.read().clone().unwrap_or_else(|| "None".into())
        );
        self.free_llm(llm);
        *self.llm.write() = None;
        *self.model_name.write() = None;
        let decision = self.decision_llm.read().clone();
        if let Some(decision) = decision {
            self.free_llm(decision);
        }
        *self.decision_llm.write() = None;
        *self.decision_model_name.write() = None;

        self.reset_llm_executor();
        self.cleanup_gpu_cache();
    }

    /// Release the on-demand vision/OCR model and return its VRAM.
    ///
    /// Only the orchestrator model stays resident; the vision model is loaded
    /// on demand and freed once a stage completes.  Clearing it re-arms lazy
    /// reload, and `vision_unavailable` is reset so a fresh attempt
    /// re-evaluates fit rather than staying latched off.
    pub fn unload_vision_model(&self) {
        if self.vision_llm.read().is_none() {
            return;
        }
        let _slot = self.inference_slot();
        let Some(vision) = self.vision_llm.read().clone() else {
            return;
        };
        tracing::info!(
            target: LOG,
            "[vision] Unloading vision model {} — freeing VRAM",
            self.vision_model_name.read().clone().filter(|n| !n.is_empty()).unwrap_or_else(|| "?".into())
        );
        self.free_llm(vision);
        *self.vision_llm.write() = None;
        *self.vision_model_name.write() = None;
        self.vision_unavailable.store(false, Ordering::SeqCst);
        self.cleanup_gpu_cache();
    }

    /// Release the on-demand large agent/skills model and return its VRAM.
    pub fn unload_large_model(&self) {
        if self.large_llm.read().is_none() {
            return;
        }
        let _slot = self.inference_slot();
        let Some(large) = self.large_llm.read().clone() else {
            return;
        };
        tracing::info!(target: LOG, "[large-model] Unloading large agent model — freeing VRAM");
        self.free_llm(large);
        *self.large_llm.write() = None;
        self.large_model_attempted.store(false, Ordering::SeqCst);
        self.cleanup_gpu_cache();
    }

    /// Free resident per-capability models to reclaim VRAM between steps.
    ///
    /// Drops every cached capability model except `keep_model_name`, plus the
    /// on-demand large model and the vision/OCR model, then reclaims the GPU.
    /// The default orchestrator model is never evicted, whether reached via
    /// `llm` or via its own capability slot: closing that slot's separate
    /// instance would tear down a context a live stream may still be decoding
    /// on.  Idempotent; returns true if anything was freed.  Not held under
    /// the inference lock: it runs on the load path, before the requesting
    /// step infers.
    pub fn evict_capability_models(&self, keep_model_name: Option<&str>) -> bool {
        let mut freed = false;
        let keep = non_empty(keep_model_name);
        let default_name = self.model_name.read().clone().filter(|n| !n.is_empty());
        let victims: Vec<(String, Arc<dyn BaseLLM>)> = {
            let slots = self.cap_llms.lock();
            slots
                .iter()
                .filter(|(name, _)| Some(name.as_str()) != keep && Some(name) != default_name.as_ref())
                .filter_map(|(name, slot)| slot.llm().map(|llm| (name.clone(), llm.clone())))
                .collect()
        };
        for (name, cached) in victims {
            tracing::info!(target: LOG, "[mem] Evicting prior-step capability model {name} to free VRAM");
            self.free_llm(cached);
            self.cap_llms.lock().retain(|(slot_name, _)| slot_name != &name);
            freed = true;
        }

        let large = self.large_llm.read().clone();
        if let Some(large) = large
            && self.large_model_name.read().as_deref() != keep_model_name
        {
            tracing::info!(target: LOG, "[mem] Evicting on-demand large model to free VRAM");
            self.free_llm(large);
            *self.large_llm.write() = None;
            *self.large_model_name.write() = None;
            self.large_model_attempted.store(false, Ordering::SeqCst);
            freed = true;
        }

        // The vision/OCR model is on-demand and per-step exactly like the
        // capability models: a document request that ran OCR first must not
        // leave it resident for the next step's model to fail its fit guard
        // against.  get_vision_llm reloads lazily if another page needs it.
        if self.vision_llm.read().is_some() && self.vision_model_name.read().as_deref() != keep_model_name {
            tracing::info!(
                target: LOG,
                "[mem] Evicting prior-step vision model {} to free VRAM",
                self.vision_model_name.read().clone().filter(|n| !n.is_empty()).unwrap_or_else(|| "?".into())
            );
            self.unload_vision_model();
            freed = true;
        }
        if freed {
            self.cleanup_gpu_cache();
        }
        freed
    }

    /// Release the embedding model and free its VRAM.
    pub fn unload_embedding_model(&self) {
        // Cleanup must never fail a response.
        if let Err(panic) = catch_unwind(crate::db::embeddings::unload) {
            tracing::warn!(target: LOG, "[models] embedding unload failed: {}", panic_message(&panic));
        }
    }

    /// Release the on-demand auxiliary models (vision, large, embedding),
    /// keeping the orchestrator and the capability models resident.
    ///
    /// Called at the end of each response cycle.  The embedding model goes
    /// with them by default; `include_embedding=false` is for the caller that
    /// still needs it for citation verification after generation ends.  Each
    /// unload is guarded on its own so one failure still returns the other
    /// models' VRAM.
    pub fn unload_auxiliary_models(&self, include_embedding: bool) {
        // Capability models are not released here but once idle (see
        // `release_capability_models_when_idle`).  One that loaded passed the
        // fit guard and stays resident across a conversation, so the next
        // message reuses it instead of reloading the GGUF — and instead of
        // re-routing from scratch, which on a memory-tight device swapped
        // two selected models in and out on every step.  A step that needs
        // a model that does not fit still evicts them (see
        // `get_llm_for_capability`), and deselecting a model unloads it.
        let mut unloads: Vec<(&str, Box<dyn Fn() + '_>)> = vec![
            ("vision", Box::new(|| self.unload_vision_model())),
            ("large", Box::new(|| self.unload_large_model())),
        ];
        if include_embedding {
            unloads.push(("embedding", Box::new(|| self.unload_embedding_model())));
        }
        for (name, unload) in unloads {
            if let Err(panic) = catch_unwind(AssertUnwindSafe(|| unload())) {
                tracing::warn!(target: LOG, "[models] {name} unload failed: {}", panic_message(&panic));
            }
        }
        self.release_capability_models_when_idle();
    }

    /// Unload the capability models once nothing has run for
    /// `SLP_MODEL_KEEP_ALIVE` — the way Ollama's `keep_alive` works.
    ///
    /// Called as each response completes.  `0` releases them now; a negative
    /// keep-alive never does.  Otherwise a timer waits out the idle period;
    /// any later response completing starts a fresh one, so only the timer
    /// from the most recent response acts.  Unloading drops the cached
    /// handles only: a generation still holding one keeps that model alive
    /// until it finishes, so a timer can never pull a model out from under a
    /// running step.
    pub fn release_capability_models_when_idle(&self) {
        let keep_alive = AppSettings::new().model_keep_alive;
        if keep_alive < 0 {
            return;
        }
        if keep_alive == 0 {
            self.release_idle_capability_models(0.0);
            return;
        }
        let epoch = self.idle_unload_epoch.fetch_add(1, Ordering::SeqCst) + 1;
        let Some(service) = self.weak_self.upgrade() else {
            return;
        };
        let service = Arc::downgrade(&service);
        let keep_alive = keep_alive as f64;
        let spawned = std::thread::Builder::new().name("idle-unload".into()).spawn(move || {
            let mut wait = keep_alive;
            loop {
                std::thread::sleep(std::time::Duration::from_secs_f64(wait.max(1.0)));
                let Some(service) = service.upgrade() else {
                    return;
                };
                if service.idle_unload_epoch.load(Ordering::SeqCst) != epoch {
                    return;
                }
                if service.release_idle_capability_models(keep_alive) {
                    return;
                }
                // Something ran in the meantime: wait out the rest of the
                // idle period from when it finished.
                wait = keep_alive - service.idle_seconds().min(keep_alive);
            }
        });
        if let Err(e) = spawned {
            tracing::warn!(target: LOG, "[models] could not start the idle-unload timer: {e}");
        }
    }

    /// Unload the capability models when nothing has run for `keep_alive`
    /// seconds; true when the idle period has passed (whether or not anything
    /// was loaded).
    pub fn release_idle_capability_models(&self, keep_alive: f64) -> bool {
        let idle = self.idle_seconds();
        if idle < keep_alive || (keep_alive > 0.0 && idle == 0.0) {
            return false;
        }
        if self.evict_capability_models(None) {
            tracing::info!(target: LOG, "[models] Unloaded capability models after {:.0}s idle", idle.min(keep_alive));
        }
        true
    }

    /// Free VRAM held by a specific model, e.g. when it's deselected.
    ///
    /// Clears every slot backed by `model_name` — the on-demand large/vision
    /// caches and any per-capability cache — then reclaims the GPU.
    /// Idempotent; returns true if anything was unloaded.
    pub fn unload_model_by_name(&self, model_name: &str) -> bool {
        let _slot = self.inference_slot();
        self.unload_model_by_name_unlocked(model_name)
    }

    /// [`Self::unload_model_by_name`] without taking an inference slot.
    ///
    /// For callers that already hold every permit — re-acquiring here would
    /// deadlock against the caller.
    pub fn unload_model_by_name_unlocked(&self, model_name: &str) -> bool {
        let mut unloaded = false;

        let large = self.large_llm.read().clone();
        if let Some(large) = large
            && self.large_model_name.read().as_deref() == Some(model_name)
        {
            tracing::info!(target: LOG, "[unload] Releasing deselected large model {model_name}");
            self.free_llm(large);
            *self.large_llm.write() = None;
            *self.large_model_name.write() = None;
            self.large_model_attempted.store(false, Ordering::SeqCst);
            unloaded = true;
        }

        let vision = self.vision_llm.read().clone();
        if let Some(vision) = vision
            && self.vision_model_name.read().as_deref() == Some(model_name)
        {
            tracing::info!(target: LOG, "[unload] Releasing deselected vision model {model_name}");
            self.free_llm(vision);
            *self.vision_llm.write() = None;
            *self.vision_model_name.write() = None;
            self.vision_unavailable.store(false, Ordering::SeqCst);
            unloaded = true;
        }

        let cached = {
            let slots = self.cap_llms.lock();
            slots.iter().find(|(name, _)| name == model_name).and_then(|(_, slot)| slot.llm().cloned())
        };
        if let Some(cached) = cached {
            tracing::info!(target: LOG, "[unload] Releasing deselected capability model {model_name}");
            self.free_llm(cached);
            self.cap_llms.lock().retain(|(name, _)| name != model_name);
            unloaded = true;
        }

        if unloaded {
            self.cleanup_gpu_cache();
        }
        unloaded
    }

    // ------------------------------------------------------------- MCP servers

    /// Register a remote (HTTP) MCP server to a project (by ID or current)
    /// and queue its provisioning.
    #[allow(clippy::too_many_arguments)]
    pub fn add_remote_mcp_server(
        &self,
        name: &str,
        server_url: &str,
        auth_type: &str,
        tools: Option<Vec<Value>>,
        token: Option<&str>,
        token_endpoint: Option<&str>,
        authorization_endpoint: Option<&str>,
        resource: Option<&str>,
        client_id: Option<&str>,
        client_secret: Option<&str>,
        access_token: Option<&str>,
        refresh_token: Option<&str>,
        expires_at: Option<f64>,
        token_type: &str,
        scope: Option<&str>,
        project_id: Option<&str>,
        server_info: Option<Map<String, Value>>,
        description: Option<&str>,
        tags: Option<Vec<String>>,
        catalog_slug: Option<&str>,
        pkg: Option<&str>,
    ) -> anyhow::Result<Value> {
        let project = self.resolve_project(project_id, "No active project", false)?;

        let mcp_auth = match auth_type {
            "oauth" => MCPAuthType::Oauth,
            "token_url" => MCPAuthType::TokenUrl,
            "token" => MCPAuthType::Token,
            _ => MCPAuthType::None,
        };
        let oauth_credentials = non_empty(client_id).map(|client_id| MCPOAuthCredentials {
            client_id: client_id.to_string(),
            client_secret: client_secret.map(str::to_string),
            access_token: access_token.map(str::to_string),
            refresh_token: refresh_token.map(str::to_string),
            expires_at,
            token_type: token_type.to_string(),
            scope: scope.map(str::to_string),
        });

        let mut tool_defs = Vec::new();
        for t in tools.unwrap_or_default() {
            let name = t
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("tool definition is missing 'name'"))?;
            tool_defs.push(MCPToolDefinition {
                name: name.to_string(),
                description: t.get("description").and_then(Value::as_str).map(str::to_string),
                input_schema: t.get("input_schema").and_then(Value::as_object).cloned(),
            });
        }

        // Registration persists the connection and returns.  Filling in any
        // missing tools and generating metadata with the LLM both happen on
        // the MCP task (see `provision_mcp_server`), so no client waits on the
        // network or the model to add a connection.
        let server = MCPServer {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            pkg: pkg.map(str::to_string),
            description: description.map(str::to_string),
            tags: tags.unwrap_or_default(),
            catalog_slug: catalog_slug.map(str::to_string),
            server_type: MCPServerType::Remote,
            remote_config: Some(RemoteMCPServerConfig {
                server_url: server_url.to_string(),
                auth_type: mcp_auth,
                token: token.map(str::to_string),
                token_endpoint: token_endpoint.map(str::to_string),
                authorization_endpoint: authorization_endpoint.map(str::to_string),
                resource: resource.map(str::to_string),
            }),
            oauth_credentials,
            tools: tool_defs,
            server_info,
            status: MCPServerStatus::Pending,
            ..MCPServer::default()
        };
        self.with_project_mut_saved(&project.id, |p| p.mcp_servers.push(server.clone()))?
            .ok_or_else(|| value_err(format!("Project not found: {}", project.id)))?;
        self.queue_mcp_provisioning(&project, &server);

        Ok(Self::mcp_server_to_dict(&server))
    }

    /// True when the discovery failure is the MCP SDK 2.x breakage.
    ///
    /// `mcp` 2.0.0 removed `mcp.server.fastmcp` and dropped several top-level
    /// exports; packages with an unbounded `mcp>=1.x` requirement die at
    /// startup.  `mcp_types` is deliberately not matched — it is a 2.x-only
    /// dependency and pinning below 2 would be wrong for it.
    pub fn mcp2_incompat_error(msg: &str) -> bool {
        let low = msg.to_lowercase();
        MCP2_NO_MODULE_RE.is_match(&low)
            || low.contains("mcp.server.fastmcp")
            || low.contains("mcp.shared")
            || MCP2_CANNOT_IMPORT_RE.is_match(&low)
            || MCP2_NO_ATTRIBUTE_RE.is_match(&low)
            // 2.x also kept the low-level Server class and removed the
            // decorators every 1.x server hangs its handlers off, so a
            // package can import cleanly and then die registering its tools.
            || MCP2_DECORATOR_RE.is_match(&low)
    }

    /// Return `args` with `--with mcp<2` injected for uv/uvx launchers, pinning
    /// the ephemeral environment to the MCP SDK 1.x line.  `None` when not
    /// applicable (non-uv launcher, or a `--with` flag is already present so
    /// we never re-pin or loop forever).
    pub fn pin_mcp1x_args(command: &str, args: &[String]) -> Option<Vec<String>> {
        if command != "uv" && command != "uvx" {
            return None;
        }
        if args.iter().any(|a| a == "--with" || a.starts_with("--with=")) {
            return None;
        }
        let pin = vec!["--with".to_string(), "mcp<2".to_string()];
        if command == "uvx" {
            return Some(pin.into_iter().chain(args.iter().cloned()).collect());
        }
        if args.len() >= 2 && args[0] == "tool" && args[1] == "run" {
            return Some(args[..2].iter().cloned().chain(pin).chain(args[2..].iter().cloned()).collect());
        }
        if args.first().is_some_and(|a| a == "run") {
            return Some(args[..1].iter().cloned().chain(pin).chain(args[1..].iter().cloned()).collect());
        }
        None
    }

    /// Register a local (stdio) MCP server to a project (by ID or current).
    ///
    /// Follows the standard MCP stdio config: the caller supplies the
    /// `command` and `args`, and they are persisted and launched exactly as
    /// given — resolved to a full path at launch, never rewritten.
    #[allow(clippy::too_many_arguments)]
    pub fn add_local_mcp_server(
        &self,
        command: &str,
        args: Option<Vec<String>>,
        name: Option<&str>,
        env: Option<HashMap<String, String>>,
        cwd: Option<&str>,
        description: Option<&str>,
        tags: Option<Vec<String>>,
        catalog_slug: Option<&str>,
        project_id: Option<&str>,
    ) -> anyhow::Result<Value> {
        if let Some(register) = self.hook(|h| h.add_local_mcp_server.clone()) {
            return register(command, project_id);
        }
        if command.is_empty() {
            return Err(value_err("'command' is required to register a local MCP server"));
        }

        let mut command = command.to_string();
        let mut args = args.unwrap_or_default();

        // If the caller passed the full command as a single string (e.g.
        // "uvx yahoo-finance-mcp"), split it into command + args.
        if args.is_empty() && command.contains(' ') {
            let (cmd, rest) = split_command_line(&command).map_err(|e| value_err(e.to_string()))?;
            command = cmd.unwrap_or_default();
            args = rest;
        }

        // A launcher with nothing to launch cannot start.  A command that is a
        // path to a program is complete on its own.
        if args.is_empty() && !(Path::new(&command).is_absolute() || command.contains(std::path::MAIN_SEPARATOR)) {
            return Err(value_err(format!(
                "'args' is required to register a local MCP server (command='{command}'). \
                 Provide the package or module name, e.g. args=['yahoo-finance-mcp'] \
                 for uvx, args=['-y', '@scope/pkg'] for npx, or args=['-m', 'module']."
            )));
        }

        let name = match name {
            Some(name) => name.to_string(),
            None => {
                // Derive a display name from the package/script — the first
                // non-flag arg — falling back to the command itself.
                let pkg_arg = args.iter().find(|a| !a.starts_with('-')).cloned().unwrap_or_else(|| command.clone());
                pkg_arg.rsplit('/').next().unwrap_or_default().to_string()
            }
        };

        let project = self.resolve_project(project_id, "No current project found", false)?;

        let server_id: String = Uuid::new_v4().to_string().chars().take(8).collect();
        let resolved_cwd = non_empty(cwd).map(expand_home);
        let resolved_env = env.unwrap_or_default();

        // Registration persists the connection and returns immediately: tool
        // discovery launches a process and enrichment calls the LLM, so both
        // run on the MCP task.  The server stays `pending` until that pass
        // records `ready` or `failed` with a reason clients can show.
        let server = MCPServer {
            id: server_id,
            name: name.clone(),
            pkg: Some(name),
            description: description.map(str::to_string),
            tags: tags.unwrap_or_default(),
            catalog_slug: catalog_slug.map(str::to_string),
            config: Some(MCPServerConfig { command, args, env: resolved_env, cwd: resolved_cwd }),
            status: MCPServerStatus::Pending,
            ..MCPServer::default()
        };
        self.with_project_mut_saved(&project.id, |p| p.mcp_servers.push(server.clone()))?
            .ok_or_else(|| value_err(format!("Project not found: {}", project.id)))?;
        self.queue_mcp_provisioning(&project, &server);

        Ok(Self::mcp_server_to_dict(&server))
    }

    /// Launch, hand-shake and list tools on one local server (or the
    /// `local_mcp_discovery` seam standing in for the launch).
    fn run_local_discovery(
        &self,
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
        cwd: Option<&str>,
    ) -> Result<MCPConnectResult, MCPClientError> {
        if let Some(discover) = self.hook(|h| h.local_mcp_discovery.clone()) {
            return discover(command, args, env, cwd);
        }
        let mut client = LocalMCPClient::new(command, args.to_vec()).with_env(Some(env.clone())).with_cwd(cwd);
        let result = client.enter().and_then(|()| client.connect_and_discover_tools());
        client.exit();
        result
    }

    /// Launch a local MCP server and record the tools it advertises.
    ///
    /// Runs on the MCP task, never on the registration request.  Mutates
    /// `server` in place — `tools`, `server_info`, and, when a retry fixed the
    /// launch, `config`/`name`/`pkg`.  Errors with [`MCPDiscoveryError`]
    /// carrying a human-readable reason when the tools cannot be enumerated.
    pub fn discover_local_mcp_tools(&self, server: &mut MCPServer) -> anyhow::Result<()> {
        let Some(config) = server.config.clone() else {
            return Err(MCPDiscoveryError("This connection has no launch command.".into()).into());
        };
        let command = config.command.clone();
        let args = config.args.clone();
        let resolved_env = config.env.clone();
        let resolved_cwd = config.cwd.clone();
        let mut name = server.name.clone();

        // A server whose tools cannot be enumerated is left registered but
        // marked failed with the reason — the connection the user typed is
        // not silently discarded.
        let mut discovery_error: Option<String> = None;
        let mut result: Option<MCPConnectResult>;
        match self.run_local_discovery(&command, &args, &resolved_env, resolved_cwd.as_deref()) {
            Ok(r) => result = Some(r),
            Err(MCPClientError::Launch(exc)) => {
                let exc_text = exc.0.clone();
                // Candidate fixes, tried in order until one succeeds:
                // 1. `uvx` / `uv tool run` print "Use `uvx --from <pkg> <exec>`
                //    instead" when package and executable names differ — parse
                //    the suggested args and retry under our launcher.
                // 2. MCP SDK 2.x removed `mcp.server.fastmcp`; pin the
                //    ephemeral env to `mcp<2` (see `pin_mcp1x_args`).
                let mut fixes: Vec<Vec<String>> = Vec::new();
                let suggested: Option<Vec<String>> = UV_HINT_RE
                    .captures(&exc_text)
                    .map(|c| c[1].split_whitespace().map(str::to_string).collect());
                if let Some(suggested) = suggested.as_ref() {
                    // Preserve the `tool run` prefix when launching via the
                    // real `uv` binary; for bare `uvx` there is no prefix.
                    let prefix: Vec<String> =
                        if command == "uv" && args.len() >= 2 && args[0] == "tool" && args[1] == "run" {
                            args[..2].to_vec()
                        } else {
                            Vec::new()
                        };
                    fixes.push(prefix.into_iter().chain(suggested.iter().cloned()).collect());
                }
                if Self::mcp2_incompat_error(&exc_text)
                    && let Some(pinned) = Self::pin_mcp1x_args(&command, &args)
                {
                    tracing::info!(
                        target: LOG,
                        "Retrying {command} with --with mcp<2 (MCP SDK 2.x incompatibility; original: {args:?})"
                    );
                    fixes.push(pinned);
                }
                result = None;
                discovery_error = Some(exc_text);
                let mut index = 0;
                while index < fixes.len() {
                    let fixed_args = fixes[index].clone();
                    index += 1;
                    match self.run_local_discovery(&command, &fixed_args, &resolved_env, resolved_cwd.as_deref()) {
                        Err(retry_exc) => {
                            result = None;
                            let retry_text = retry_exc.to_string();
                            discovery_error = Some(retry_text.clone());
                            // The retry can fail differently from the original
                            // (the hint fixed the executable, then the server
                            // died importing 2.x symbols).  `pin_mcp1x_args`
                            // returns None once `--with` is present, so this
                            // never re-pins or loops forever.
                            if Self::mcp2_incompat_error(&retry_text)
                                && let Some(pinned) = Self::pin_mcp1x_args(&command, &fixed_args)
                            {
                                tracing::info!(
                                    target: LOG,
                                    "Retrying {command} with --with mcp<2 (MCP SDK 2.x incompatibility; original: {fixed_args:?})"
                                );
                                fixes.push(pinned);
                            }
                            continue;
                        }
                        Ok(r) => {
                            if !r.success {
                                discovery_error =
                                    Some(r.error.clone().filter(|e| !e.is_empty()).unwrap_or_else(|| {
                                        "tool discovery failed".to_string()
                                    }));
                                result = Some(r);
                                continue;
                            }
                            result = Some(r);
                        }
                    }
                    server.config = Some(MCPServerConfig {
                        command: command.clone(),
                        args: fixed_args.clone(),
                        env: resolved_env.clone(),
                        cwd: resolved_cwd.clone(),
                    });
                    if let Some(suggested) = suggested.as_ref().filter(|s| !s.is_empty()) {
                        let fixed_pkg = if let Some(from_idx) = suggested.iter().position(|a| a == "--from") {
                            suggested.get(from_idx + 1).cloned()
                        } else {
                            suggested.iter().find(|a| !a.starts_with('-')).cloned()
                        };
                        if let Some(fixed_pkg) = fixed_pkg.filter(|p| !p.is_empty()) {
                            server.name = fixed_pkg.rsplit('/').next().unwrap_or_default().to_string();
                            server.pkg = Some(server.name.clone());
                            name = server.name.clone();
                        }
                    }
                    discovery_error = None;
                    break;
                }
            }
            Err(other) => {
                result = None;
                discovery_error = Some(other.to_string());
            }
        }

        match result.as_ref() {
            Some(r) if r.success => {
                apply_discovery(server, r);
                tracing::info!(target: LOG, "Discovered {} tool(s) from local MCP server '{name}'", r.tools.len());
            }
            Some(r) => {
                discovery_error =
                    Some(r.error.clone().filter(|e| !e.is_empty()).unwrap_or_else(|| "tool discovery failed".into()));
            }
            None => {}
        }

        if let Some(discovery_error) = discovery_error {
            // Keep the full raw error (often a multi-line traceback) in the
            // logs, but hand the client a short, human-readable reason.
            tracing::warn!(target: LOG, "Could not discover tools from local MCP server '{name}': {discovery_error}");
            return Err(MCPDiscoveryError(Self::friendly_discovery_error(&discovery_error)).into());
        }
        Ok(())
    }

    // ------------------------------------------------- background provisioning

    /// The registered MCP background task, or `None` before `setup_tasks()`.
    pub fn mcp_task(&self) -> Option<Arc<MCPTask>> {
        let queue = self.task_queue.read().clone();
        queue.get_task_as::<MCPTask>("mcp")
    }

    /// Hand a freshly registered server to the MCP task.
    ///
    /// Falls back to provisioning on a thread when the task isn't registered
    /// (CLI use, tests): the point is that the caller doesn't wait.
    pub fn queue_mcp_provisioning(&self, project: &Project, server: &MCPServer) {
        if let Some(queue) = self.hook(|h| h.queue_mcp_provisioning.clone()) {
            queue(project, server);
            return;
        }
        if let Some(task) = self.mcp_task() {
            task.enqueue(&project.id, &server.id);
            return;
        }
        let Some(service) = self.weak_self.upgrade() else {
            return;
        };
        let project_id = project.id.clone();
        let server_id = server.id.clone();
        let spawned = std::thread::Builder::new().name(format!("mcp-provision-{server_id}")).spawn(move || {
            if let Err(e) = service.provision_mcp_server(&project_id, &server_id) {
                tracing::warn!(target: LOG, "[mcp-provision] {server_id} failed: {e}");
            }
        });
        if let Err(e) = spawned {
            tracing::warn!(target: LOG, "[mcp-provision] could not start provisioning thread: {e}");
        }
    }

    /// `(project_id, server_id)` for PENDING servers this process may
    /// provision.
    ///
    /// Only ever this process's own projects: a per-project child adopts its
    /// own project and no other; a supervisor that delegated its projects
    /// adopts nothing — the child is the tier that can persist the answer.
    pub fn pending_mcp_servers(&self) -> anyhow::Result<Vec<(String, String)>> {
        self.initialize_workspace()?;
        if projects_delegated() {
            return Ok(Vec::new());
        }
        let pinned = pinned_project_id();
        let model = self.model_or_err()?;
        let workspace = model.workspace.read();
        let mut pending = Vec::new();
        for project in &workspace.projects {
            if pinned.as_deref().is_some_and(|pinned| project.id != pinned) {
                continue;
            }
            for server in &project.mcp_servers {
                if server.system {
                    continue;
                }
                if server.status == MCPServerStatus::Pending {
                    pending.push((project.id.clone(), server.id.clone()));
                }
            }
        }
        Ok(pending)
    }

    /// Write a provisioning pass's result back onto the live record (unless
    /// the server was deleted meanwhile) and save.
    fn store_provisioned(&self, project_id: &str, server: &MCPServer) -> anyhow::Result<()> {
        self.with_project_mut_saved(project_id, |p| {
            if let Some(live) = p.mcp_servers.iter_mut().find(|s| s.id == server.id) {
                merge_provisioned(live, server);
            }
        })?;
        Ok(())
    }

    /// Discover and enrich one registered server; returns
    /// `{"status": "ready"|"failed"|"skipped", ...}`.
    ///
    /// The slow half of registration, run off the request path: discover the
    /// tools when registration couldn't supply them, then enrich blank
    /// metadata via the LLM.  Records `ready` or `failed` + `last_error`, so
    /// the state a client polls always reflects a finished pass.
    pub fn provision_mcp_server(&self, project_id: &str, server_id: &str) -> anyhow::Result<Value> {
        let project = match self.project_by_id(project_id) {
            Ok(project) => project,
            Err(_) => {
                tracing::warn!(target: LOG, "[mcp-provision] Project {project_id} is gone — skipping");
                return Ok(json!({"status": "skipped", "reason": "project not found"}));
            }
        };

        let mut server = project.mcp_servers.iter().find(|s| s.id == server_id).cloned();
        if server.is_none() {
            // Registered by another process, or deleted before we got to it.
            if let Ok(model) = self.model_or_err() {
                model.workspace.write().reload();
            }
            let project = self.project_by_id(project_id)?;
            server = project.mcp_servers.iter().find(|s| s.id == server_id).cloned();
        }
        let Some(mut server) = server else {
            tracing::info!(target: LOG, "[mcp-provision] Server {server_id} is gone — skipping");
            return Ok(json!({"status": "skipped", "reason": "server not found"}));
        };

        tracing::info!(
            target: LOG,
            "[mcp-provision] Provisioning '{}' ({}, {} tool(s) known)",
            server.name,
            server.server_type.as_str(),
            server.tools.len()
        );

        let discovered = server.tools.is_empty();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            if discovered { self.discover_mcp_tools_seam(&mut server) } else { Ok(()) }
        }));
        let failure = match outcome {
            Ok(Ok(())) => None,
            Ok(Err(e)) => Some(classify_discovery_error(&e)),
            Err(panic) => Some(DiscoveryFailure::Other(panic_message(&panic))),
        };
        if let Some(failure) = failure {
            match failure {
                DiscoveryFailure::AuthExpired => {
                    // Already marked and explained by the refresh attempt;
                    // saving is still ours.
                    self.store_provisioned(project_id, &server)?;
                    tracing::warn!(target: LOG, "[mcp-provision] '{}' can no longer authenticate", server.name);
                }
                DiscoveryFailure::Discovery(message) => {
                    server.status = MCPServerStatus::Failed;
                    server.last_error = Some(message);
                    self.store_provisioned(project_id, &server)?;
                    tracing::warn!(
                        target: LOG,
                        "[mcp-provision] '{}' failed: {}",
                        server.name,
                        server.last_error.clone().unwrap_or_default()
                    );
                }
                DiscoveryFailure::Other(raw) => {
                    server.status = MCPServerStatus::Failed;
                    server.last_error = Some(Self::friendly_discovery_error(&raw));
                    self.store_provisioned(project_id, &server)?;
                    tracing::warn!(target: LOG, "[mcp-provision] '{}' failed: {raw}", server.name);
                }
            }
            return Ok(json!({"status": "failed", "error": server.last_error}));
        }

        // The tools are the catalog a chat resolves against, and they are
        // known now.  Publish them before the enrichment below: that is one or
        // two LLM calls on a model host the user's own chat competes for, and
        // a prompt sent in the meantime would otherwise reload the workspace,
        // find the connection with no tools, and leave it out of the turn.
        // The status stays `pending` until the pass is finished.
        if discovered && !server.tools.is_empty() {
            self.store_provisioned(project_id, &server)?;
            tracing::info!(
                target: LOG,
                "[mcp-provision] '{}' — {} tool(s) available, enriching metadata",
                server.name,
                server.tools.len()
            );
        }

        // Metadata for the UI and the tool router — an LLM call, which is why
        // it is here and not on the registration request.
        self.enrich_mcp_server_seam(&mut server);

        server.status = MCPServerStatus::Ready;
        server.last_error = None;
        self.store_provisioned(project_id, &server)?;

        self.probe_mcp_account(&mut server);

        tracing::info!(
            target: LOG,
            "[mcp-provision] '{}' ready — {} tool(s) discovered",
            server.name,
            server.tools.len()
        );
        Ok(json!({"status": "ready", "tools": server.tools.len()}))
    }

    /// `_discover_mcp_tools(server)` as `_provision_mcp_server` calls it,
    /// honouring the `discover_mcp_tools` seam.
    fn discover_mcp_tools_seam(&self, server: &mut MCPServer) -> anyhow::Result<()> {
        match self.hook(|h| h.discover_mcp_tools.clone()) {
            Some(discover) => discover(server),
            None => self.discover_mcp_tools(server),
        }
    }

    /// `_enrich_mcp_server(server)` as `_provision_mcp_server` calls it,
    /// honouring the `enrich_mcp_server` seam.
    fn enrich_mcp_server_seam(&self, server: &mut MCPServer) {
        match self.hook(|h| h.enrich_mcp_server.clone()) {
            Some(enrich) => enrich(server),
            None => self.enrich_mcp_server(server),
        }
    }

    /// The hook a tool caller uses to write a rotated token back.
    fn persist_hook(&self) -> PersistFn {
        let weak = self.weak_self.clone();
        Arc::new(move |server: &MCPServer| match weak.upgrade() {
            Some(service) => service.persist_mcp_server(server),
            None => Ok(()),
        })
    }

    /// `MCPToolCaller(None, None, persist=...)`.
    fn transport_tool_caller(&self) -> MCPToolCaller {
        let llm: Arc<dyn BaseLLM> = Arc::new(TransportOnlyLLM::new());
        MCPToolCaller::new(
            Arc::new(MCPArgumentExtractor::new(llm)),
            Arc::new(parking_lot::Mutex::new(MCPResultMapper::new())),
            Some(self.persist_hook()),
        )
    }

    /// Record which account `server` is linked as, when it can say.
    ///
    /// One read-only call on the provisioning pass; best-effort.  It is also
    /// the call most likely to renew an expired token, so its caller carries
    /// the persistence hook and a reported rotation is saved here as well —
    /// a rotation nothing writes would be discarded by the next reload,
    /// leaving a refresh token the provider has already retired.
    pub fn probe_mcp_account(&self, server: &mut MCPServer) {
        let probed = catch_unwind(AssertUnwindSafe(|| {
            let caller: Arc<dyn ReadOnlyToolCaller> = Arc::new(PersistingToolCaller(self.transport_tool_caller()));
            match self.hook(|h| h.probe_account.clone()) {
                Some(probe) => probe(&mut *server, Some(caller)),
                None => probe_account(&mut *server, Some(caller)),
            }
        }));
        let (account, credentials_changed) = match probed {
            Ok(outcome) => outcome,
            Err(panic) => {
                tracing::debug!(target: LOG, "[mcp-account] probe failed for '{}': {}", server.name, panic_message(&panic));
                return;
            }
        };

        let mut changed = credentials_changed;
        if truthy(&account) && account != server.account {
            server.account = account;
            changed = true;
        }
        if changed {
            // The hook has already written a rotation; this covers a caller
            // built without one.
            if let Err(e) = self.persist_mcp_server(server) {
                tracing::warn!(target: LOG, "[mcp-account] could not save '{}': {e}", server.name);
            }
        }
    }

    /// Save `server`'s state onto the workspace this service holds now.
    ///
    /// Re-finds the server by id so the write lands on the instance that will
    /// be serialized; a server deleted while the pass ran is left gone rather
    /// than resurrected.
    pub fn persist_mcp_server(&self, server: &MCPServer) -> anyhow::Result<()> {
        let model = self.model_or_err()?;
        {
            let mut workspace = model.workspace.write();
            let live = workspace.projects.iter_mut().flat_map(|p| p.mcp_servers.iter_mut()).find(|s| s.id == server.id);
            if let Some(live) = live {
                live.oauth_credentials = server.oauth_credentials.clone();
                live.account = server.account.clone();
                live.status = server.status;
                live.last_error = server.last_error.clone();
            }
        }
        self.persist_model()
    }

    /// Fill in a server's tools, launching or connecting as needed.  Errors
    /// with [`MCPDiscoveryError`] / [`MCPAuthExpiredError`].
    pub fn discover_mcp_tools(&self, server: &mut MCPServer) -> anyhow::Result<()> {
        if server.server_type == MCPServerType::Local {
            return self.discover_local_mcp_tools(server);
        }

        let server_url = server.remote_config.as_ref().map(|rc| rc.server_url.clone()).filter(|u| !u.is_empty());
        let Some(server_url) = server_url else {
            return Err(MCPDiscoveryError("This connection has no server URL.".into()).into());
        };

        // Discovery runs on the stored credentials, so it ages out exactly as a
        // tool call does.  The caller owns reading the token, renewing it and
        // writing the rotation back.
        let caller = self.transport_tool_caller();
        let token = MCPToolCaller::get_access_token(server);

        let remote_discovery = self.hook(|h| h.remote_mcp_discovery.clone());
        let connect = |access_token: Option<&str>| {
            if let Some(discover) = remote_discovery.as_ref() {
                return discover(&server_url, access_token);
            }
            let mut client = MCPClient::new();
            client.enter();
            let result = client.connect_and_discover_tools(&server_url, access_token, "Bearer");
            client.exit();
            result
        };

        let mut result = connect(token.as_deref());
        if !result.success && result.auth_expired {
            match caller.refresh_access_token(server, token.as_deref()) {
                Some(new_token) => result = connect(Some(&new_token)),
                None => {
                    if server.status == MCPServerStatus::Failed {
                        // The refresh diagnosed a retired grant and has
                        // already said so on the record.
                        return Err(MCPAuthExpiredError(server.last_error.clone().unwrap_or_default()).into());
                    }
                }
            }
        }

        if !result.success {
            let raw = result.error.clone().filter(|e| !e.is_empty()).unwrap_or_else(|| "tool discovery failed".into());
            return Err(MCPDiscoveryError(Self::friendly_discovery_error(&raw)).into());
        }
        apply_discovery(server, &result);
        tracing::info!(
            target: LOG,
            "Discovered {} tool(s) from remote MCP server '{}'",
            result.tools.len(),
            server.name
        );
        Ok(())
    }

    // ---------------------------------------------------------- busy / idle

    /// Hold an inference slot, and record that the model is in use until the
    /// returned guard drops.  Background work reads the count through
    /// [`Self::idle_seconds`].
    pub fn inference_slot(&self) -> InferenceSlot<'_> {
        self.inference_lock.acquire();
        self.inference_state.lock().active += 1;
        InferenceSlot { service: self }
    }

    /// Seconds since the model last finished answering; `0.0` while it is,
    /// `f64::INFINITY` when nothing has run this session.
    pub fn idle_seconds(&self) -> f64 {
        let state = self.inference_state.lock();
        if state.active > 0 {
            return 0.0;
        }
        if state.last_inference_at == 0.0 {
            // Nothing has run this session — idle since startup.
            return f64::INFINITY;
        }
        (now_secs() - state.last_inference_at).max(0.0)
    }

    /// Re-run provisioning for a server (a failed discovery, or a refresh);
    /// returns as soon as it is queued.
    pub fn retry_mcp_server(&self, server_id: &str, project_id: Option<&str>) -> anyhow::Result<Value> {
        let project = self.resolve_project(project_id, "No current project found", true)?;
        if !project.mcp_servers.iter().any(|s| s.id == server_id) {
            return Err(value_err(format!("MCP server not found: {server_id}")));
        }

        // Drop the cached catalog too: provisioning only discovers when a
        // server has no tools, so keeping them made "refresh" a no-op that
        // declared a dead connection READY without a single network call.
        let updated = self
            .with_project_mut(&project.id, |p| {
                p.mcp_servers.iter_mut().find(|s| s.id == server_id).map(|server| {
                    server.status = MCPServerStatus::Pending;
                    server.last_error = None;
                    server.tools = Vec::new();
                    server.clone()
                })
            })?
            .flatten();
        let server = match updated {
            Some(server) => server,
            None => {
                let mut server = project
                    .mcp_servers
                    .iter()
                    .find(|s| s.id == server_id)
                    .cloned()
                    .ok_or_else(|| value_err(format!("MCP server not found: {server_id}")))?;
                server.status = MCPServerStatus::Pending;
                server.last_error = None;
                server.tools = Vec::new();
                server
            }
        };
        self.save_model()?;
        self.queue_mcp_provisioning(&project, &server);
        Ok(Self::mcp_server_to_dict(&server))
    }

    /// Whether a connection is ready to answer with, and why not.
    ///
    /// `in_progress` is the flag to drive a spinner off; `tools` is the count
    /// discovery recorded (0 on a ready server means it advertised none).
    pub fn mcp_provisioning_state(server: &MCPServer) -> Value {
        let state = match server.status {
            MCPServerStatus::Failed => "failed",
            MCPServerStatus::Pending => "provisioning",
            MCPServerStatus::Ready => "ready",
        };
        json!({
            "state": state,
            "in_progress": state == "provisioning",
            "tools": server.tools.len(),
            "error": server.last_error,
        })
    }

    /// The server dict with its `indexing` (provisioning) block.
    fn mcp_server_with_status(server: &MCPServer) -> Value {
        let mut dict = Self::mcp_server_to_dict(server);
        if let Value::Object(map) = &mut dict {
            map.insert("indexing".into(), Self::mcp_provisioning_state(server));
        }
        dict
    }

    /// Every non-system server in a project, each with its provisioning
    /// state, so watching a fresh connection costs one request per poll.
    pub fn list_mcp_servers_with_status(&self, project_id: &str) -> anyhow::Result<Vec<Value>> {
        let project = self.project_by_id(project_id)?;
        Ok(project.mcp_servers.iter().filter(|s| !s.system).map(Self::mcp_server_with_status).collect())
    }

    /// One server, with the provisioning state a client polls it for.
    pub fn mcp_server_status(&self, server_id: &str, project_id: Option<&str>) -> anyhow::Result<Value> {
        let project = self.resolve_project(project_id, "No current project found", true)?;
        let server = project
            .mcp_servers
            .iter()
            .find(|s| s.id == server_id)
            .ok_or_else(|| value_err(format!("MCP server not found: {server_id}")))?;
        Ok(Self::mcp_server_with_status(server))
    }

    /// Diagnostic: how many MCP tools a project has available, as discovery
    /// recorded them.
    pub fn count_mcp_tools(&self, project_id: Option<&str>) -> anyhow::Result<Value> {
        let project = self.resolve_project(project_id, "No current project found", false)?;
        let servers: Vec<&MCPServer> = project.mcp_servers.iter().filter(|s| !s.system).collect();
        let mut by_server = Map::new();
        for s in &servers {
            by_server.insert(s.name.clone(), json!(s.tools.len()));
        }
        Ok(json!({
            "project": project.name,
            "servers": servers.len(),
            "tools": servers.iter().map(|s| s.tools.len()).sum::<usize>(),
            "by_server": by_server,
        }))
    }

    /// Enrich a hand-added MCP server via the local LLM when fields are blank:
    /// a short `description` and `tags` for the UI, and a `summary` the tool
    /// selector routes on.  Best-effort — never fails.
    ///
    /// Curated catalog servers arrive with description and tags and skip the
    /// LLM entirely.  Always the small router model: registering a server must
    /// never load (or compete for) a bigger one.
    pub fn enrich_mcp_server(&self, server: &mut MCPServer) {
        let outcome = catch_unwind(AssertUnwindSafe(|| -> anyhow::Result<()> {
            if server.tools.is_empty() {
                return Ok(());
            }
            // Curated server — nothing to generate.  A resolved catalog_slug is
            // authoritative; description + tags both present means there is
            // nothing left to write.  The catalog has no summary field, so the
            // description doubles as the selector's capability text.
            if truthy(&server.catalog_slug) || (truthy(&server.description) && !server.tags.is_empty()) {
                if !truthy(&server.summary) {
                    server.summary = server.description.clone();
                }
                return Ok(());
            }
            if self.resident_llm().is_none() {
                self.load_model_seam()?;
            }
            // The router model, even when a prior chat request left a bigger
            // one resident: these prompts are short and low-stakes.
            let Some(llm) = self.resident_llm() else {
                return Ok(());
            };
            let summarizer = MCPServerSummarizer::new(llm);
            let _slot = self.inference_slot();
            // Description + tags (UI) — one JSON call when either is missing.
            if !truthy(&server.description) || server.tags.is_empty() {
                let meta = summarizer.generate_metadata(server);
                if !truthy(&server.description)
                    && let Some(description) = meta.get("description").and_then(Value::as_str).filter(|d| !d.is_empty())
                {
                    server.description = Some(description.to_string());
                }
                if server.tags.is_empty()
                    && let Some(tags) = meta.get("tags").and_then(Value::as_array).filter(|t| !t.is_empty())
                {
                    server.tags = tags
                        .iter()
                        .take(3)
                        .map(|t| t.as_str().map(str::to_string).unwrap_or_else(|| t.to_string()))
                        .collect();
                }
            }
            // Summary (tool selection) — a separate free-text call so a JSON
            // flake above doesn't cost the summary; falls back to the
            // description so the selector always has capability text.
            if !truthy(&server.summary) {
                let summary = summarizer.invoke(server).filter(|s| !s.is_empty());
                server.summary = summary.or_else(|| server.description.clone().filter(|d| !d.is_empty()));
            }
            Ok(())
        }));
        let failure = match outcome {
            Ok(Ok(())) => None,
            Ok(Err(e)) => Some(e.to_string()),
            Err(panic) => Some(panic_message(&panic)),
        };
        if let Some(e) = failure {
            tracing::warn!(target: LOG, "MCP enrichment failed for '{}': {e}", server.name);
        }
    }

    /// Map a raw MCP discovery failure (often a uvx/npx traceback) to a short,
    /// human-readable message; falls back to the first line of the raw error.
    pub fn friendly_discovery_error(raw: &str) -> String {
        let msg = raw.trim();
        if msg.is_empty() {
            return "Couldn't start this connection.".to_string();
        }
        // Inside the macOS App Sandbox the path resolves correctly but the
        // sandbox refuses to exec any program outside the app bundle with
        // EPERM.  No retry can walk that back, so say what it is.  A Developer
        // ID build (no container) keeps the raw line: there the denial is a
        // genuine permissions problem.
        if under_app_sandbox()
            && let Some(caps) = EPERM_RE.captures(msg)
        {
            return format!(
                "The App Store build can't run '{}': the macOS App Sandbox only lets it start \
                 programs bundled with the app. To use local MCP servers, install the build from \
                 smartloop.ai.",
                &caps[1]
            );
        }
        let low = msg.to_lowercase();
        // Package deps unpinned/out of date vs the current MCP SDK.
        if FRIENDLY_SDK_RE.is_match(&low) {
            return "This package isn't compatible with the current MCP SDK — its \
                    dependencies are out of date. Try another package, or ask its \
                    maintainer to update it."
                .to_string();
        }
        // Package name doesn't resolve to anything published.
        if FRIENDLY_NOT_FOUND_RE.is_match(&low) {
            return "Package not found. Double-check the name — it must be a published \
                    package (PyPI for uvx, npm for npx)."
                .to_string();
        }
        // uv / uvx (or npx) missing from PATH.
        if FRIENDLY_NO_UVX_RE.is_match(&low) {
            return "Couldn't run uvx. Make sure uv is installed and on your PATH.".to_string();
        }
        // Any other import/startup crash from the package process.
        if FRIENDLY_CRASH_RE.is_match(&low) {
            return "This package failed to start (a dependency or startup error). \
                    It may be broken or incompatible."
                .to_string();
        }
        let first = msg.split('\n').next().unwrap_or_default().trim();
        if first.chars().count() > 180 { format!("{}…", prefix_chars(first, 180)) } else { first.to_string() }
    }

    // --------------------------------------------------------- web search (pro)

    /// `{"project_id", "enabled"}` web-search config for a project.
    pub fn get_web_search_config(&self, project_id: &str) -> anyhow::Result<Value> {
        let project = self.project_by_id(project_id)?;
        Ok(json!({"project_id": project.id, "enabled": project.web_search_enabled}))
    }

    /// Enable/disable web search for a project.
    pub fn set_web_search_enabled(&self, project_id: &str, enabled: bool) -> anyhow::Result<Value> {
        let project = self.project_by_id(project_id)?;
        self.with_project_mut(&project.id, |p| p.web_search_enabled = enabled)?;
        self.save_model()?;
        tracing::info!(target: LOG, "[web-search] project {} enabled={}", project.id, py_bool(enabled));
        Ok(json!({"project_id": project.id, "enabled": enabled}))
    }

    /// Write MCP tool description to disk for workspace indexer discovery;
    /// true on success, false when it already exists or the write failed.
    pub fn write_mcp_doc(&self, project: &Project, server: &MCPServer, content: &str) -> bool {
        let outcome = (|| -> std::io::Result<bool> {
            // Framework state, written where the workspace indexer scans.
            let base_dir = PathBuf::from(&project.project_dir);
            std::fs::create_dir_all(&base_dir)?;

            let lowered = server.name.to_lowercase();
            let safe_name = MCP_DOC_NAME_RE.replace_all(&lowered, "-");
            let filename = format!("mcp-{}.md", safe_name.trim_matches('-'));
            let doc_path = base_dir.join(&filename);

            if doc_path.exists() {
                tracing::info!(target: LOG, "Workspace doc '{filename}' already exists, skipping");
                return Ok(false);
            }
            std::fs::write(&doc_path, content)?;
            tracing::info!(target: LOG, "Workspace doc '{filename}' created for MCP server '{}'", server.name);
            Ok(true)
        })();
        match outcome {
            Ok(written) => written,
            Err(e) => {
                tracing::warn!(target: LOG, "Failed to write workspace doc for MCP server '{}': {e}", server.name);
                false
            }
        }
    }

    /// Convert an MCPServer to a dict for API responses.
    pub fn mcp_server_to_dict(s: &MCPServer) -> Value {
        let mut result = Map::new();
        result.insert("id".into(), json!(s.id));
        result.insert("name".into(), json!(s.name));
        result.insert("description".into(), json!(s.description));
        result.insert("summary".into(), json!(s.summary));
        result.insert("tags".into(), json!(s.tags));
        result.insert("catalog_slug".into(), json!(s.catalog_slug));
        result.insert("server_type".into(), json!(s.server_type.as_str()));
        result.insert("enabled".into(), json!(s.enabled));
        result.insert("system".into(), json!(s.system));
        result.insert(
            "tools".into(),
            Value::Array(s.tools.iter().map(|t| serde_json::to_value(t).unwrap_or(Value::Null)).collect()),
        );
        // Background provisioning state — clients poll this instead of
        // blocking on discovery at registration time.
        result.insert("status".into(), json!(s.status.as_str()));
        result.insert("last_error".into(), json!(s.last_error));
        result.insert("account".into(), json!(s.account));
        if let Some(config) = s.config.as_ref() {
            result.insert("command".into(), json!(config.command));
            result.insert("args".into(), json!(config.args));
            if truthy(&config.cwd) {
                result.insert("cwd".into(), json!(config.cwd));
            }
        }
        if let Some(remote) = s.remote_config.as_ref() {
            result.insert("server_url".into(), json!(remote.server_url));
        }
        if let Some(info) = s.server_info.as_ref().filter(|info| !info.is_empty()) {
            result.insert("server_info".into(), Value::Object(info.clone()));
        }
        Value::Object(result)
    }

    /// MCP servers of the current project.
    pub fn list_mcp_servers(&self) -> anyhow::Result<Vec<Value>> {
        self.initialize_workspace()?;
        let Some(project) = self.current_project_snapshot() else {
            return Ok(Vec::new());
        };
        Ok(project.mcp_servers.iter().filter(|s| !s.system).map(Self::mcp_server_to_dict).collect())
    }

    /// Remove an MCP server (and its definition file) from a project (by ID
    /// or current).
    pub fn delete_mcp_server(&self, server_id: &str, project_id: Option<&str>) -> anyhow::Result<Value> {
        let mut project = self.resolve_project(project_id, "No current project found", false)?;

        let mut server = project.mcp_servers.iter().find(|s| s.id == server_id).cloned();
        if server.is_none() {
            // The server may have been registered by another process since
            // the last workspace load.  Reload from disk and retry once.
            let model = self.model_or_err()?;
            model.workspace.write().reload();
            let target = non_empty(project_id).map(str::to_string).unwrap_or_else(|| project.id.clone());
            if let Some(reloaded) = model.workspace.read().project(&target).cloned() {
                project = reloaded;
            }
            server = project.mcp_servers.iter().find(|s| s.id == server_id).cloned();
        }
        let server = server.ok_or_else(|| value_err(format!("MCP server not found: {server_id}")))?;

        self.with_project_mut(&project.id, |p| p.mcp_servers.retain(|s| s.id != server_id))?;
        self.save_model()?;
        self.remove_mcp_definition_file(&server);

        Ok(json!({"id": server_id, "deleted": true, "message": "MCP server deleted successfully"}))
    }

    /// Delete the dropped-in file that defined `server`, if it was one.
    ///
    /// Without this a deleted connection leaves its file behind, and re-adding
    /// a server by the same name silently picks up the old one's command, env
    /// and cwd.  Only files under `mcp.d` are touched, and only when no
    /// remaining server matches the same definition.
    pub fn remove_mcp_definition_file(&self, server: &MCPServer) {
        let outcome = catch_unwind(AssertUnwindSafe(|| -> anyhow::Result<()> {
            let Some(definition) = definition_for(server) else {
                return Ok(());
            };
            let key = definition.key();
            let model = self.model_or_err()?;
            {
                let workspace = model.workspace.read();
                for project in &workspace.projects {
                    for other in &project.mcp_servers {
                        if other.id == server.id {
                            continue;
                        }
                        if definition_for(other).is_some_and(|m| m.key() == key) {
                            return Ok(());
                        }
                    }
                }
            }
            let path = Path::new(&settings().home_dir()).join(USER_DEFINITIONS_DIR).join(format!("{key}.toml"));
            if path.is_file() {
                std::fs::remove_file(&path)?;
                reload_definitions();
                tracing::info!(target: LOG, "[mcp-definition] Removed {}", path.display());
            }
            Ok(())
        }));
        let failure = match outcome {
            Ok(Ok(())) => None,
            Ok(Err(e)) => Some(e.to_string()),
            Err(panic) => Some(panic_message(&panic)),
        };
        if let Some(e) = failure {
            tracing::debug!(target: LOG, "[mcp-definition] Could not remove the definition file: {e}");
        }
    }

    // ------------------------------------------------------------ orchestration

    /// The shared LLMExecutor (an I/O thread pool), created lazily.
    pub fn get_llm_executor(&self) -> Arc<LLMExecutor> {
        let mut slot = self.llm_executor.lock();
        slot.get_or_insert_with(|| Arc::new(LLMExecutor::new(6))).clone()
    }

    /// Shut down the shared executor so it's rebuilt for the new model.
    pub fn reset_llm_executor(&self) {
        let executor = self.llm_executor.lock().take();
        if let Some(executor) = executor {
            let _ = catch_unwind(AssertUnwindSafe(|| executor.shutdown(false)));
        }
    }

    /// Fire a progress/SSE event, swallowing any callback panic.  A no-op
    /// without a callback; a failing callback must never break orchestration.
    pub fn emit_progress(progress_callback: Option<&ProgressCallback>, step: &str, status: &str, message: &str) {
        let Some(callback) = progress_callback else {
            return;
        };
        if catch_unwind(AssertUnwindSafe(|| callback(step, status, message))).is_err() {
            tracing::debug!(target: LOG, "[progress] callback failed");
        }
    }

    /// Full orchestrator entry point for a user message: classification,
    /// context enrichment, DAG execution, capability routing and — when the
    /// plan calls for it — MCP tool calls.  See [`OrchestrateOutcome`].
    ///
    /// `conversation_history` is the recent `[{"role", "content"}]` turns
    /// passed to the context explorer.
    #[allow(clippy::too_many_arguments)]
    pub fn orchestrate(
        &self,
        user_message: &str,
        skills: Option<Vec<Skill>>,
        cwd: Option<&str>,
        conversation_history: Option<Vec<Value>>,
        has_attachments: bool,
        images: Option<Vec<(Vec<u8>, String)>>,
        progress_callback: Option<ProgressCallback>,
        cancelled: Option<CancelEvent>,
        requested_skill: Option<&str>,
        current_date: Option<&str>,
    ) -> OrchestrateOutcome {
        let none: OrchestrateOutcome = (None, false, None, None, None, false);
        tracing::info!(target: LOG, "[orchestrator] entered orchestrate: {:?}", prefix_chars(user_message, 120));

        // A missing handle is recoverable when the weights are the host's —
        // building one is a constructor call, not a load.  The alternative is
        // quietly degrading to raw generation, losing skills, tools and
        // planning.
        if self.resident_llm().is_none() && self.remote_device().is_some() {
            *self.llm.write() = Some(self.remote_llm(None));
            let mut model_name = self.model_name.write();
            if model_name.as_deref().is_none_or(str::is_empty) {
                *model_name = Some(api_settings().default_model);
            }
            tracing::info!(target: LOG, "[orchestrator] bound to the model host");
        }

        let model = self.model.read().clone();
        let main_llm = self.resident_llm();
        let (Some(model), Some(main_llm)) = (model, main_llm) else {
            tracing::warn!(
                target: LOG,
                "[orchestrator] BYPASSED — model/llm not loaded (model={}, llm={}). Falling back to raw stream_completion.",
                py_bool(self.model.read().is_some()),
                py_bool(self.resident_llm().is_some())
            );
            return none;
        };

        let mut user_message = user_message.to_string();

        // Image attachment → a vision-capable model first *perceives* the
        // image (a concrete description scoped to the user's request) and that
        // perception is folded into the message, so the rest of the pipeline
        // plans and answers grounded in what the image shows.  With no project
        // to orchestrate over, the perception is streamed back directly.
        let mut image_perception: Option<String> = None;
        let mut image_sole_source = false;
        if let Some(images) = images.as_ref().filter(|images| !images.is_empty()) {
            // Surface the vision model on the SSE stream, naming the model
            // actually picked, right before the (heavy) load.
            let progress = progress_callback.clone();
            let on_vision_model: ModelSelectedCallback = Arc::new(move |name: &str| {
                Self::emit_progress(progress.as_ref(), "model", "processing", &format!("Processing image with {name}"));
            });
            let perceived = catch_unwind(AssertUnwindSafe(|| {
                self.vision_perceive(&user_message, images, 512, Some(on_vision_model))
            }));
            match perceived {
                Ok(Ok(perception)) => image_perception = Some(perception),
                Ok(Err(e)) => tracing::warn!(
                    target: LOG,
                    "[orchestrator] vision perception failed ({e}) — continuing without image context"
                ),
                Err(panic) => tracing::warn!(
                    target: LOG,
                    "[orchestrator] vision perception failed ({}) — continuing without image context",
                    panic_message(&panic)
                ),
            }
            // The vision model has done its one job for this request; free its
            // VRAM for the downstream synthesis now.
            self.unload_vision_model();
            match image_perception.as_deref().filter(|p| !p.is_empty()) {
                Some(perception) => {
                    tracing::info!(target: LOG, "[orchestrator] image perception: {:?}", prefix_chars(perception, 200));
                    // A bare extract/read of the attached image discards the
                    // retrieved docs (they only cause mis-attribution); a request
                    // to compare, validate, or answer using indexed content
                    // keeps them for cross-validation.
                    image_sole_source = !IMAGE_CROSS_REF_RE.is_match(&user_message);
                    tracing::info!(target: LOG, "[orchestrator] image sole-source={}", py_bool(image_sole_source));
                    user_message = format!("{user_message}\n\n[Image context: {perception}]");
                }
                None => {
                    image_perception = None;
                    tracing::warn!(target: LOG, "[orchestrator] vision perception empty — continuing without image context");
                }
            }
        }

        // The perception is folded into `user_message` for planning, but the
        // final response prompt is rebuilt from the original messages plus the
        // returned context — so the description must ride along here, phrased
        // as an instruction: the answering model is text-only and would
        // otherwise reply that it cannot view images.
        let with_image_context = |ctx: Option<String>| -> Option<String> {
            let Some(perception) = image_perception.as_deref() else {
                return ctx;
            };
            let authority = if image_sole_source {
                "the SOLE source for answering this request"
            } else {
                "an authoritative source for this request"
            };
            let block = format!(
                "The user attached an image. A vision model has already analyzed \
                 it — treat the description below as the actual contents of that \
                 image and {authority}. Answer the user's request directly; do \
                 NOT say you cannot see or view the image.\n\n\
                 Image contents: {perception}"
            );
            match ctx.filter(|c| !c.is_empty()) {
                None => Some(block),
                // Cross-validation intent — the image is authoritative but the
                // retrieved documents may corroborate or help answer it.
                Some(ctx) if !image_sole_source => Some(format!("{block}\n\n{ctx}")),
                // Pure extract/read — the retrieved documents are background
                // only, or the answering model mis-attributes the image's own
                // fields to them.
                Some(ctx) => Some(format!(
                    "{block}\n\n---\n\n\
                     The material below was retrieved from indexed documents as \
                     background only. Do NOT answer from it, cite it, or attribute \
                     the image's contents to any document — the image above is the \
                     sole source for this request.\n\n\
                     {ctx}"
                )),
            }
        };

        // Reload workspace from disk to pick up changes made by background
        // indexers or project setup since the last request.
        model.workspace.write().reload();

        let Some(current_project) = model.workspace.read().current_project().cloned() else {
            if let Some(perception) = image_perception.clone() {
                tracing::info!(target: LOG, "[orchestrator] no current project — streaming image perception directly.");
                return (Some(perception), true, None, None, None, true);
            }
            tracing::warn!(target: LOG, "[orchestrator] BYPASSED — no current project. Falling back to raw stream_completion.");
            return none;
        };

        // Migrate legacy REMOTE system servers to LOCAL on first access.
        if let Err(e) = self.migrate_servers_to_local(&current_project.id) {
            tracing::warn!(target: LOG, "[orchestrator] system server migration failed: {e}");
        }
        let current_project = model.workspace.read().project(&current_project.id).cloned().unwrap_or(current_project);

        // Enabled servers that have tools (both local and remote).
        let eligible_servers: Vec<MCPServer> =
            current_project.mcp_servers.iter().filter(|s| s.enabled && !s.tools.is_empty()).cloned().collect();

        // Always read skills from disk — the single source of truth.  The
        // `skills` parameter is a fallback for backward compatibility.
        let disk_skills = current_project.get_skills();
        let all_skills = if disk_skills.is_empty() { skills.unwrap_or_default() } else { disk_skills };
        let skill_contents: Option<Vec<String>> = if all_skills.is_empty() {
            None
        } else {
            Some(
                all_skills
                    .iter()
                    .map(|s| {
                        let mut header = format!("---\nname: {}\n", s.name);
                        if let Some(description) = s.description.as_deref().filter(|d| !d.is_empty()) {
                            header.push_str(&format!("description: {description}\n"));
                        }
                        header.push_str("---\n\n");
                        header + &s.content
                    })
                    .collect(),
            )
        };

        // Point the orchestrator at the framework's own per-project dir so
        // tools and scans stay self-contained.
        let orchestrator_cwd = current_project.project_dir.clone();

        // With exactly one model selected, it runs the routing and decision
        // nodes too.  Asked without a capability, so a vision-only selection
        // declines these text-only nodes and they stay on the default model.
        // Orchestration runs on device: a remote pin writes what the user
        // reads, so the routing and decision models resolve local-only.
        let pinned_llm = {
            let _scope = local_only_scope(true);
            self.pinned_llm_seam(None)
        };
        if let Some(pinned) = pinned_llm.as_ref() {
            tracing::info!(
                target: LOG,
                "[models] Single selected model pinned for the whole request: {}",
                Self::model_label(pinned.as_ref())
            );
        }

        let run = catch_unwind(AssertUnwindSafe(|| -> anyhow::Result<OrchestrateOutcome> {
            let service = self.arc();
            // In a project child these are handles to the model host, which
            // resolves them per call; mark them local-only so it does too.
            let main_llm = local_only_handle(main_llm.clone());
            let decision_llm = self.decision_llm.read().clone().map(local_only_handle);
            let orchestrator_llm: Arc<dyn BaseLLM> = pinned_llm.clone().unwrap_or_else(|| main_llm.clone());
            let agent_llm: Arc<dyn BaseLLM> =
                pinned_llm.clone().or_else(|| decision_llm.clone()).unwrap_or_else(|| main_llm.clone());

            // The providers are built through `provider0`/`provider1` and the
            // `ProviderOutput` conversions so their return type is taken from
            // the options field they are stored in.
            let mut options = SLPOrchestratorOptions::default();
            options.agent_llm = Some(ProviderOutput::from_value(agent_llm));
            options.large_llm = None;
            // Not local-only: these write step output the user reads.  The
            // router resolves its decision nodes inside a local-only scope.
            options.large_llm_provider = Some(provider0({
                let service = service.clone();
                move || ProviderOutput::from_value(service.large_llm_seam())
            }));
            options.capability_llm_provider = Some(provider1({
                let service = service.clone();
                move |capability: &str| ProviderOutput::from_value(service.llm_for_capability_seam(capability))
            }));
            options.capability_available = Some(provider1({
                let service = service.clone();
                move |capability: &str| ProviderOutput::from_value(service.has_capability_model(capability))
            }));
            options.capability_skip_reason = Some(provider0({
                let service = service.clone();
                move || ProviderOutput::from_value(service.last_capability_skip())
            }));
            options.model_path = self.model_gguf_path.read().clone();
            options.project_id = self.get_project_id();
            options.answers_remotely = self.answers_remotely();
            options.executor = Some(self.get_llm_executor());
            // Gathered context answers on the `general` response model (see
            // the chat router), so it is sized against that model's window.
            options.answer_window = Some(Arc::new({
                let service = service.clone();
                move || service.get_response_max_context_tokens(Some(ModelCapability::General.value()))
            }));
            let orchestrator = SLPOrchestrator::new(orchestrator_llm, options);

            // A tool call that renews an expired OAuth token mutates the
            // server; this is what writes it back.  The orchestrator has no
            // workspace of its own — the service owns it.
            orchestrator.tool_caller.set_persist(Some(self.persist_hook()));

            let available_capabilities: Vec<String> = {
                let workspace = model.workspace.read();
                let project = workspace.current_project();
                let mut caps: Vec<String> = Vec::new();
                let mut snapshot: Vec<String> = Vec::new();
                for m in &workspace.models {
                    let enabled = workspace.effective_model_enabled(&m.model_name, project);
                    let downloaded = workspace.effective_model_downloaded(&m.model_name);
                    let effective = m.effective_capabilities();
                    snapshot.push(format!(
                        "({:?}, {}, {}, {:?})",
                        m.model_name,
                        py_bool(enabled),
                        py_bool(downloaded),
                        effective
                    ));
                    if enabled && downloaded {
                        for cap in effective {
                            if !caps.contains(&cap) {
                                caps.push(cap);
                            }
                        }
                    }
                }
                tracing::info!(
                    target: LOG,
                    "[caps] models=[{}] available_capabilities={:?}",
                    snapshot.join(", "),
                    caps
                );
                caps
            };

            // Propagate the cancellation event onto the orchestrator's primary
            // LLMs so short planning steps can be interrupted between tokens.
            let mut cancel_llms: Vec<Arc<dyn BaseLLM>> = Vec::new();
            for llm in [Some(main_llm.clone()), decision_llm.clone(), pinned_llm.clone()].into_iter().flatten() {
                if !cancel_llms.iter().any(|seen| same_llm(seen, &llm)) {
                    cancel_llms.push(llm);
                }
            }
            struct CancelReset<'c> {
                llms: &'c [Arc<dyn BaseLLM>],
                armed: bool,
            }
            impl Drop for CancelReset<'_> {
                fn drop(&mut self) {
                    if self.armed {
                        for llm in self.llms {
                            llm.set_cancelled(None);
                        }
                    }
                }
            }
            if let Some(cancelled) = cancelled.as_ref() {
                for llm in &cancel_llms {
                    llm.set_cancelled(Some(cancelled.clone()));
                }
            }
            let result = {
                let _reset = CancelReset { llms: &cancel_llms, armed: cancelled.is_some() };
                orchestrator.execute(
                    &user_message,
                    &eligible_servers,
                    ExecuteOptions {
                        skills: skill_contents.clone(),
                        cwd: cwd.map(str::to_string),
                        document_context: None,
                        has_attachments,
                        project_dir: Some(orchestrator_cwd.clone()),
                        model_service: Some(service.clone()),
                        conversation_history: conversation_history.as_ref().map(|h| history_maps(h)),
                        available_capabilities: if available_capabilities.is_empty() {
                            None
                        } else {
                            Some(available_capabilities.clone())
                        },
                        progress_callback: progress_callback.clone(),
                        requested_skill: requested_skill.map(str::to_string),
                        current_date: current_date.map(str::to_string),
                        cancelled: cancelled.clone(),
                    },
                )
            };

            // Save workspace in case tokens were refreshed during the call.
            let _ = catch_unwind(AssertUnwindSafe(|| model.save()));

            if !result.handled {
                // No tool handled it, but a perceived image's description must
                // still reach the final response as context.
                if image_perception.is_some() {
                    return Ok((with_image_context(None), true, None, None, None, false));
                }
                return Ok(none.clone());
            }

            if let Some(error) = result.error.as_ref().filter(|e| !e.is_empty()) {
                let tool = result.tool_name.clone().unwrap_or_else(|| "None".into());
                tracing::warn!(target: LOG, "Tool call failed ({tool}): {error}");
                return Ok((None, true, Some(format!("Tool '{tool}' failed: {error}")), None, None, false));
            }

            if let Some(content) = result.content.as_ref().filter(|c| !c.is_empty()) {
                tracing::info!(
                    target: LOG,
                    "Tool '{}' returned {} chars of context",
                    result.tool_name.clone().unwrap_or_else(|| "None".into()),
                    content.chars().count()
                );
                // Drop document citations when the image is the sole source —
                // citing retrieved docs would mis-attribute its contents.
                let citations = if image_sole_source { None } else { result.citations.clone() };
                return Ok((
                    with_image_context(Some(content.clone())),
                    true,
                    None,
                    citations,
                    result.skill_instruction.clone(),
                    result.mutation,
                ));
            }

            Ok((with_image_context(None), true, None, None, result.skill_instruction.clone(), false))
        }));

        let failure = match run {
            Ok(Ok(outcome)) => return outcome,
            Ok(Err(e)) => e.to_string(),
            Err(panic) => panic_message(&panic),
        };
        tracing::warn!(target: LOG, "Tool call attempt failed: {failure}");
        (None, true, Some(format!("Tool call failed: {failure}")), None, None, false)
    }

    // --------------------------------------------------------------- auto-write

    /// Write LLM output to `<cwd>/.out/<name>.md` when the user asked to
    /// create/generate/write.
    ///
    /// Two modes: a bare "write"/"save" command writes the last assistant
    /// response from session history (refine back and forth, then save); an
    /// inline write intent writes the current `llm_response`.  Returns the
    /// written path, or `None` when no write was needed.
    pub fn maybe_write_output(
        &self,
        user_message: &str,
        llm_response: &str,
        cwd: Option<&str>,
        session_id: Option<&str>,
    ) -> Option<String> {
        let cwd = non_empty(cwd)?;
        let llm = self.resident_llm()?;

        let cwd = expand_home(cwd);
        let mut content_to_write = llm_response.to_string();

        if BARE_WRITE_RE.is_match(user_message) {
            // Bare "write" command — write the last assistant response.
            if let Some(session_id) = non_empty(session_id) {
                match self.get_last_assistant_response(session_id) {
                    Some(last_response) => {
                        content_to_write = last_response;
                        tracing::info!(
                            target: LOG,
                            "[auto-write] Bare write command — using last assistant response ({} chars)",
                            content_to_write.chars().count()
                        );
                    }
                    None => {
                        tracing::info!(target: LOG, "[auto-write] Bare write command but no prior assistant response found");
                        return None;
                    }
                }
            } else if llm_response.is_empty() {
                return None;
            }
        } else if llm_response.is_empty() || !WRITE_INTENT_RE.is_match(user_message) {
            return None;
        }

        if content_to_write.is_empty() {
            return None;
        }
        let mut content_to_write = Self::extract_document_content(&content_to_write);
        if content_to_write.is_empty() {
            return None;
        }

        // Ask the LLM only for the filename — write the actual content.
        let prompt = format!(
            "User request: {user_message}\n\n\
             Pick a short descriptive filename (without extension) for saving this output. \
             Examples: \"project_summary\", \"api_notes\", \"readme_draft\".\n\n\
             Reply with ONLY the filename, nothing else."
        );
        let outcome = catch_unwind(AssertUnwindSafe(|| -> anyhow::Result<Option<String>> {
            let raw = llm.invoke(&prompt, None, None);
            let file_name = raw.trim().trim_matches(|c| c == '"' || c == '\'' || c == '`').to_string();
            if file_name.is_empty() {
                return Ok(None);
            }

            let safe_name = SAFE_FILENAME_RE.replace_all(&file_name, "_").trim_matches('_').to_string();
            let out_dir = Path::new(&cwd).join(".out");
            std::fs::create_dir_all(&out_dir)?;
            let mut full_path = out_dir.join(format!("{safe_name}.md"));

            // If the file already exists, append a counter.
            let mut counter = 1;
            while full_path.exists() {
                full_path = out_dir.join(format!("{safe_name}_{counter}.md"));
                counter += 1;
            }

            if !content_to_write.ends_with('\n') {
                content_to_write.push('\n');
            }
            content_to_write.push_str("\nCo-Authored By: Smartloop AI\n");

            std::fs::write(&full_path, &content_to_write)?;
            let full_path = full_path.to_string_lossy().into_owned();
            tracing::info!(
                target: LOG,
                "[auto-write] Saved to {full_path} ({} chars)",
                content_to_write.chars().count()
            );
            Ok(Some(full_path))
        }));
        match outcome {
            Ok(Ok(path)) => path,
            Ok(Err(e)) => {
                tracing::warn!(target: LOG, "[auto-write] Failed: {e}");
                None
            }
            Err(panic) => {
                tracing::warn!(target: LOG, "[auto-write] Failed: {}", panic_message(&panic));
                None
            }
        }
    }

    /// The last assistant message from session history.
    pub fn get_last_assistant_response(&self, session_id: &str) -> Option<String> {
        let history = self.get_session_history(session_id);
        history.iter().rev().find(|m| m.role == "assistant" && !m.content.is_empty()).map(|m| m.content.clone())
    }

    /// Strip conversational preamble/postamble, keeping only the document body.
    pub fn extract_document_content(text: &str) -> String {
        let lines: Vec<&str> = text.split('\n').collect();

        // If the response contains a markdown heading, start from the first.
        let first_heading = lines.iter().position(|line| line.starts_with('#'));

        let mut text = text.to_string();
        match first_heading {
            Some(index) if index > 0 => {
                let preamble = lines[..index].join("\n");
                if preamble.trim().chars().count() < 200 {
                    text = lines[index..].join("\n");
                } else {
                    text = PREAMBLE_RE.replace_all(&text, "").into_owned();
                }
            }
            _ => {
                text = PREAMBLE_RE.replace_all(&text, "").into_owned();
            }
        }

        let text = POSTAMBLE_RE.replace_all(&text, "");
        text.trim().to_string()
    }

    // ------------------------------------------------------------------- assets

    /// Return (and create if needed) the directory used for asset storage.
    pub fn assets_dir(&self) -> anyhow::Result<PathBuf> {
        let path = Path::new(&settings().home_dir()).join("assets");
        std::fs::create_dir_all(&path)?;
        Ok(path)
    }

    /// Save an uploaded file as an asset, converting to markdown when
    /// possible (the same parser pipeline as document upload).  Returns
    /// `{asset_id, filename, content_type, size, markdown, image}`.
    pub fn upload_asset(&self, file_bytes: &[u8], filename: &str, content_type: Option<&str>) -> anyhow::Result<Value> {
        let asset_id = Uuid::new_v4().to_string();
        let ext = Path::new(filename)
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
            .unwrap_or_default();

        let content_type = match non_empty(content_type) {
            Some(ct) => ct.to_string(),
            None => mime_guess::from_path(filename)
                .first()
                .map(|m| m.essence_str().to_string())
                .unwrap_or_else(|| "application/octet-stream".to_string()),
        };

        let asset_dir = self.assets_dir()?.join(&asset_id);
        std::fs::create_dir_all(&asset_dir)?;

        let mut markdown_content: Option<String> = None;
        let mut is_markdown = false;
        let is_image = content_type.starts_with("image/");

        if is_image {
            // Images are NOT pre-processed to text: OCR-at-upload throws away
            // the visual content.  Persist the original pixels and let a
            // vision-capable model read them at query time with the user's
            // actual question (see `vision_answer` / `orchestrate`).
            let ext = if ext.is_empty() { ".img".to_string() } else { ext.clone() };
            std::fs::write(asset_dir.join(format!("original{ext}")), file_bytes)?;
        } else {
            // Write the original file to a temp location for parsing.
            let mut tmp = tempfile::Builder::new().suffix(&ext).tempfile()?;
            std::io::Write::write_all(&mut tmp, file_bytes)?;
            let tmp_path = tmp.path().to_string_lossy().into_owned();

            let parsed = catch_unwind(AssertUnwindSafe(|| -> anyhow::Result<()> {
                let mime_type = content_type.clone();
                let llm = self.load_model_seam()?;

                // get_parser fails with a ValueError for unsupported mime
                // types — treat as raw storage.
                let parser = match ParserFactory::new(Some(llm.clone() as SharedLLM)).get_parser(&mime_type, &tmp_path) {
                    Ok(parser) => Some(parser),
                    Err(ParserError::Value(_)) => None,
                    Err(e) => return Err(e.into()),
                };

                if let Some(parser) = parser {
                    let markdown = parser.parse()?;
                    markdown_content = Some(markdown.clone());
                    is_markdown = true;

                    // Enforce the token limit using the loaded LLM's tokenizer.
                    // A sole selected cloud model answers with the whole
                    // attachment, so its context window is the limit, not
                    // the local orchestrator's.
                    let max_tokens = self
                        .pinned_model_seam(None)
                        .filter(|ws_model| ws_model.is_remote())
                        .and_then(|ws_model| ws_model.context_length())
                        .map(|context_length| context_length as usize)
                        .unwrap_or_else(|| llm.get_max_tokens());
                    let estimated_tokens = llm.get_num_tokens(&markdown);
                    if estimated_tokens > max_tokens {
                        return Err(value_err(format!(
                            "Attachment too large: ~{} tokens estimated (maximum is {}). \
                             Please split the document into smaller sections.",
                            fmt_thousands(estimated_tokens),
                            fmt_thousands(max_tokens)
                        )));
                    }
                }
                Ok(())
            }));
            drop(tmp);
            match parsed {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    let is_value_error = matches!(e.downcast_ref::<ModelServiceError>(), Some(ModelServiceError::Value(_)))
                        || matches!(e.downcast_ref::<ParserError>(), Some(ParserError::Value(_)));
                    if is_value_error {
                        return Err(e);
                    }
                    tracing::error!(target: LOG, "Parser conversion failed for {filename}: {e}; storing raw bytes");
                }
                Err(panic) => {
                    tracing::error!(
                        target: LOG,
                        "Parser conversion failed for {filename}: {}; storing raw bytes",
                        panic_message(&panic)
                    );
                }
            }
        }

        // Persist metadata.
        let metadata = json!({
            "asset_id": asset_id,
            "filename": filename,
            "content_type": content_type,
            "size": file_bytes.len(),
            "markdown": is_markdown,
            "image": is_image,
        });
        std::fs::write(asset_dir.join("metadata.json"), python_json_dumps(&metadata))?;

        // Persist content.
        if let Some(markdown) = markdown_content.as_ref() {
            std::fs::write(asset_dir.join("content.md"), markdown)?;
        }

        tracing::info!(target: LOG, "Asset stored: {asset_id} ({filename}, markdown={})", py_bool(is_markdown));
        Ok(metadata)
    }

    /// Retrieve asset metadata and content (`content`: the markdown, or null).
    pub fn get_asset(&self, asset_id: &str) -> anyhow::Result<Value> {
        let asset_dir = self.assets_dir()?.join(asset_id);
        let meta_path = asset_dir.join("metadata.json");
        if !meta_path.exists() {
            return Err(value_err(format!("Asset not found: {asset_id}")));
        }
        let mut metadata: Value = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;

        let content_path = asset_dir.join("content.md");
        let content = if content_path.exists() { Value::String(std::fs::read_to_string(&content_path)?) } else { Value::Null };
        if let Value::Object(map) = &mut metadata {
            map.insert("content".into(), content);
        }
        Ok(metadata)
    }

    /// Delete an asset and all its stored files.
    pub fn delete_asset(&self, asset_id: &str) -> anyhow::Result<Value> {
        let asset_dir = self.assets_dir()?.join(asset_id);
        if !asset_dir.exists() {
            return Err(value_err(format!("Asset not found: {asset_id}")));
        }
        std::fs::remove_dir_all(&asset_dir)?;
        tracing::info!(target: LOG, "Asset deleted: {asset_id}");
        Ok(json!({"asset_id": asset_id, "deleted": true, "message": "Asset deleted successfully"}))
    }

    /// A combined markdown string for a list of asset IDs, injected into the
    /// chat prompt.
    pub fn resolve_attachment_context(&self, asset_ids: &[String]) -> String {
        let mut parts = Vec::new();
        for aid in asset_ids {
            match self.get_asset(aid) {
                Ok(asset) => {
                    let content = asset.get("content").and_then(Value::as_str).unwrap_or_default();
                    if !content.is_empty() {
                        let filename = asset.get("filename").and_then(Value::as_str).unwrap_or(aid);
                        parts.push(format!("ATTACHMENT [{filename}]:\n{content}"));
                    }
                }
                Err(_) => tracing::warn!(target: LOG, "Attachment asset not found: {aid}"),
            }
        }
        parts.join("\n\n---\n\n")
    }

    /// `(bytes, mime_type)` for the image assets among `asset_ids`; document
    /// assets are ignored so the caller can route image turns to the vision
    /// model.
    pub fn get_asset_images(&self, asset_ids: &[String]) -> Vec<(Vec<u8>, String)> {
        let mut images = Vec::new();
        let Ok(assets_dir) = self.assets_dir() else {
            return images;
        };
        for aid in asset_ids {
            let asset_dir = assets_dir.join(aid);
            let meta_path = asset_dir.join("metadata.json");
            if !meta_path.exists() {
                tracing::warn!(target: LOG, "Attachment asset not found: {aid}");
                continue;
            }
            let Some(meta) = std::fs::read_to_string(&meta_path)
                .ok()
                .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            else {
                continue;
            };
            let content_type = meta
                .get("content_type")
                .and_then(Value::as_str)
                .filter(|ct| !ct.is_empty())
                .unwrap_or("application/octet-stream")
                .to_string();
            let flagged_image = meta.get("image").is_some_and(|v| match v {
                Value::Bool(b) => *b,
                Value::Null => false,
                Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
                Value::String(s) => !s.is_empty(),
                Value::Array(a) => !a.is_empty(),
                Value::Object(o) => !o.is_empty(),
            });
            if !(flagged_image || content_type.starts_with("image/")) {
                continue;
            }
            let mut originals: Vec<PathBuf> = std::fs::read_dir(&asset_dir)
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| {
                            p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("original."))
                        })
                        .collect()
                })
                .unwrap_or_default();
            originals.sort();
            if let Some(original) = originals.first()
                && let Ok(bytes) = std::fs::read(original)
            {
                images.push((bytes, content_type));
            }
        }
        images
    }

    /// A concrete description of `images` scoped to `user_message` — the
    /// orchestrator's image entry point.  Reuses `vision_answer`'s inference
    /// path (and its GPU-clip recovery) with a description-only prompt.
    pub fn vision_perceive(
        &self,
        user_message: &str,
        images: &[(Vec<u8>, String)],
        max_tokens: usize,
        on_model_selected: Option<ModelSelectedCallback>,
    ) -> anyhow::Result<String> {
        let prompt = VISION_PERCEPTION_PROMPT.replace("{request}", user_message);
        self.vision_answer(&prompt, images, max_tokens, on_model_selected)
    }

    /// One non-streaming chat completion on the vision model, under a slot.
    fn ask_vision(&self, vision_llm: &Arc<dyn BaseLLM>, messages: &[Value], max_tokens: usize) -> anyhow::Result<Value> {
        // A cloud model takes the images over HTTP: no local slot to hold.
        if let Some(cloud) = vision_llm.as_any().downcast_ref::<CloudLLM>() {
            return cloud.create_chat_completion(messages.to_vec(), Some(max_tokens), 0.0);
        }
        let _slot = self.inference_slot();
        let model = vision_llm.llama_model().ok_or_else(|| anyhow::anyhow!("vision model has no local weights"))?;
        let output = model.create_chat_completion(ChatCompletionRequest {
            messages: messages.to_vec(),
            max_tokens: Some(max_tokens),
            temperature: 0.0,
            stream: false,
            ..ChatCompletionRequest::default()
        })?;
        Ok(output.into_response())
    }

    /// A vision-model answer to `prompt` about `images`.
    ///
    /// The pixels and the user's own prompt go straight to the multimodal
    /// model (non-streaming — the proven OCR path); the anti-hallucination
    /// steering goes in the system message.  Errors with
    /// `"No vision-capable model available"` when none can be loaded.
    pub fn vision_answer(
        &self,
        prompt: &str,
        images: &[(Vec<u8>, String)],
        max_tokens: usize,
        on_model_selected: Option<ModelSelectedCallback>,
    ) -> anyhow::Result<String> {
        let selected: Option<&dyn Fn(&str)> = on_model_selected.as_ref().map(|f| f.as_ref() as &dyn Fn(&str));
        let llm = self.get_vision_answer_llm(selected).ok_or_else(|| runtime_err("No vision-capable model available"))?;

        let mut content: Vec<Value> = vec![json!({"type": "text", "text": prompt})];
        for (data, mime) in images {
            let b64 = base64::engine::general_purpose::STANDARD.encode(data);
            content.push(json!({
                "type": "image_url",
                "image_url": {"url": format!("data:{mime};base64,{b64}")},
            }));
        }
        let messages = vec![
            json!({"role": "system", "content": VISION_QA_GUIDANCE}),
            json!({"role": "user", "content": content}),
        ];

        let first = catch_unwind(AssertUnwindSafe(|| self.ask_vision(&llm, &messages, max_tokens)))
            .unwrap_or_else(|panic| Err(anyhow::anyhow!(panic_message(&panic))));
        let result = match first {
            Ok(result) => result,
            Err(e) => {
                // The GPU clip encode of a large image can exhaust device
                // memory inside native code.  Reload once with the CLIP
                // encoder on CPU — the LLM itself stays on GPU — and retry
                // instead of degrading to the blind text pipeline.  A cloud
                // model has no local encoder to move.
                if Self::is_cloud_llm(llm.as_ref()) || self.vision_clip_cpu.load(Ordering::SeqCst) {
                    return Err(e);
                }
                tracing::warn!(
                    target: LOG,
                    "[vision-answer] encode failed with GPU clip ({e}) — reloading with CLIP on CPU and retrying"
                );
                self.vision_clip_cpu.store(true, Ordering::SeqCst);
                self.unload_vision_model();
                let Some(llm) = self.get_vision_llm(None) else {
                    return Err(e);
                };
                catch_unwind(AssertUnwindSafe(|| self.ask_vision(&llm, &messages, max_tokens)))
                    .unwrap_or_else(|panic| Err(anyhow::anyhow!(panic_message(&panic))))?
            }
        };

        let choice0 = result
            .get("choices")
            .and_then(Value::as_array)
            .filter(|choices| !choices.is_empty())
            .map(|choices| choices[0].clone())
            .unwrap_or_else(|| json!({}));
        let answer = choice0
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        // This path calls the chat completion directly, so it does its own
        // <think> filtering — a reasoning model must not leak its monologue.
        let answer = strip_thinking(&answer);
        // Qwen2-VL sometimes prefixes its reply with a stray "label:"; drop it.
        let answer = answer.trim_start_matches(':').trim().to_string();
        let finish = match choice0.get("finish_reason") {
            Some(Value::String(s)) => s.clone(),
            _ => "None".to_string(),
        };
        tracing::info!(
            target: LOG,
            "[vision-answer] {} char(s), finish={finish}, preview={:?}",
            answer.chars().count(),
            prefix_chars(&answer, 120)
        );
        Ok(answer)
    }
}

/// Python's `str(bool)` for log lines.
fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

/// `[{"role": ..., "content": ...}]` turns as string maps.
fn history_maps(history: &[Value]) -> Vec<HashMap<String, String>> {
    history
        .iter()
        .filter_map(Value::as_object)
        .map(|turn| {
            turn.iter()
                .map(|(k, v)| (k.clone(), v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())))
                .collect()
        })
        .collect()
}

/// A provider closure taking no argument, typed by where it is stored.
fn provider0<R>(f: impl Fn() -> R + Send + Sync + 'static) -> Arc<dyn Fn() -> R + Send + Sync> {
    Arc::new(f)
}

/// A provider closure taking a capability, typed by where it is stored.
fn provider1<R>(f: impl Fn(&str) -> R + Send + Sync + 'static) -> Arc<dyn Fn(&str) -> R + Send + Sync> {
    Arc::new(f)
}

/// Converts what the service answers into the shape an orchestrator option
/// carries: the bare value, or the value wrapped in `anyhow::Result` (the
/// provider aliases of the model router), with LLM handles as either trait
/// object spelling.
trait ProviderOutput<T> {
    fn from_value(value: T) -> Self;
}

impl ProviderOutput<Arc<dyn BaseLLM>> for Arc<dyn BaseLLM> {
    fn from_value(value: Arc<dyn BaseLLM>) -> Self {
        value
    }
}

impl ProviderOutput<Arc<dyn BaseLLM>> for SharedLLM {
    fn from_value(value: Arc<dyn BaseLLM>) -> Self {
        value
    }
}

impl ProviderOutput<Option<Arc<dyn BaseLLM>>> for Option<Arc<dyn BaseLLM>> {
    fn from_value(value: Option<Arc<dyn BaseLLM>>) -> Self {
        value
    }
}

impl ProviderOutput<Option<Arc<dyn BaseLLM>>> for Option<SharedLLM> {
    fn from_value(value: Option<Arc<dyn BaseLLM>>) -> Self {
        value.map(|llm| -> SharedLLM { llm })
    }
}

impl ProviderOutput<Option<Arc<dyn BaseLLM>>> for anyhow::Result<Option<Arc<dyn BaseLLM>>> {
    fn from_value(value: Option<Arc<dyn BaseLLM>>) -> Self {
        Ok(value)
    }
}

impl ProviderOutput<Option<Arc<dyn BaseLLM>>> for anyhow::Result<Option<SharedLLM>> {
    fn from_value(value: Option<Arc<dyn BaseLLM>>) -> Self {
        Ok(value.map(|llm| -> SharedLLM { llm }))
    }
}

impl ProviderOutput<bool> for bool {
    fn from_value(value: bool) -> Self {
        value
    }
}

impl ProviderOutput<bool> for anyhow::Result<bool> {
    fn from_value(value: bool) -> Self {
        Ok(value)
    }
}

impl ProviderOutput<Option<CapabilitySkipReason>> for Option<CapabilitySkipReason> {
    fn from_value(value: Option<CapabilitySkipReason>) -> Self {
        value
    }
}

impl ProviderOutput<Option<CapabilitySkipReason>> for anyhow::Result<Option<CapabilitySkipReason>> {
    fn from_value(value: Option<CapabilitySkipReason>) -> Self {
        Ok(value)
    }
}

impl ProviderOutput<Option<CapabilitySkipReason>> for Option<(String, String, Option<String>)> {
    fn from_value(value: Option<CapabilitySkipReason>) -> Self {
        value.map(|r| (r.model_name, r.cause, r.detail))
    }
}
