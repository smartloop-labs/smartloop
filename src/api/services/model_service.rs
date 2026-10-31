//! Port of `smartloop/api/services/model_service.py`.
//!
//! Singleton model service for managing LLM lifecycle.  This file holds the
//! module-level helpers, [`ProjectCaches`], the [`ModelService`] struct with all
//! of its state, and every method the Python defined before
//! `_llm_for_response`; the remaining methods live in [`part2`] as a second
//! `impl ModelService` block.

// `part2` glob-imports this module (`use super::*`), so the imports below are
// shared with it even where this half does not use them.
#![allow(unused_imports, clippy::too_many_arguments, clippy::type_complexity)]

#[allow(dead_code)]
mod part2;

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use indexmap::IndexMap;
use once_cell::sync::{Lazy, OnceCell};
use parking_lot::{Condvar, Mutex, RwLock};
use regex::Regex;
use serde_json::{Map, Value, json};

use crate::agents::hybrid_executor::LLMExecutor;
use crate::agents::orchestration::model_router::{
    CapabilitySkipReason, ModelRouter, ModelRouterOptions, ProgressCallback,
};
use crate::agents::planner::{FETCH_SERVER_ID, SYSTEM_TOOLS_SERVER_ID, SYSTEM_TOOLS_SERVER_NAME};
use crate::api::config::api_settings;
use crate::config::app_settings::AppSettings;
use crate::config::model_config::ModelConfig;
use crate::constants::{canonical_model_name, display_model_label};
use crate::conversation_store::{ConversationStore, MessageInput};
use crate::db::feedback_indexer::FeedbackIndexer;
use crate::db::store::{ChunkResult, VectorStore};
use crate::db::store_factory::StoreFactory;
use crate::db::workspace_indexer::{WORKSPACE_COLLECTION, WorkspaceIndexer};
use crate::feedback_store::{ConversationTurn, FeedbackEntry, FeedbackStore, Rating};
use crate::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationStream};
use crate::llms::cloud_llm::CloudLLM;
use crate::llms::llm_factory::{LlamaInstanceOptions, LlmFactory};
use crate::llms::thinking::strip_thinking;
use crate::mcp::client::{MCPClientError, MCPConnectResult};
use crate::mcp::read_only_calls::ReadOnlyToolCaller;
use crate::model::model_manager::Model;
use crate::model::slp_model::TrainProgressCallback;
use crate::model::spec::MODEL_SPECS;
use crate::model_factory::ModelFactory;
use crate::model_host::client::{LocalModelService, ModelClient, ModelHostError, get_model_client};
use crate::model_host::remote_llm::RemoteLLM;
use crate::tasks::TaskQueue;
use crate::types::ModelCapability;
use crate::utils::log_utils::LoggerLike;
use crate::utils::normalize_path;
use crate::workspace::workspace::SharedWorkspace;
use crate::workspace::{
    DocumentInfo, DocumentStatus, MCPServer, MCPServerConfig, MCPServerStatus, MCPServerType,
    MCPToolDefinition, Project, RemoteMCPServerConfig, Skill, Workspace, WorkspaceModel,
    pinned_project_id, use_project,
};

/// Logger name of the Python module (`logging.getLogger(__name__)`).
pub const LOG_TARGET: &str = "smartloop.api.services.model_service";

/// Module-level settings (`_settings = AppSettings()`).
pub fn settings() -> &'static AppSettings {
    static SETTINGS: Lazy<AppSettings> = Lazy::new(AppSettings::new);
    &SETTINGS
}

/// How to describe an `llm_large_n_ctx_cap` value in a log line.
///
/// `0` is the sentinel for *no cap* — the factory auto-sizes the context from
/// live available memory — but printing it raw read as a context window of
/// zero tokens, which is both alarming and wrong.
pub fn n_ctx_desc(cap: i64) -> String {
    if cap > 0 { format!("n_ctx={cap}") } else { "an auto-sized context".to_string() }
}

/// Default context window when no model is loaded or it cannot be queried.
pub const DEFAULT_MAX_CONTEXT_TOKENS: usize = 8192;

// Shared vector-retrieval selection criteria — applied identically to
// document and workspace search so both rank and select content the same
// way (broad recall, quality floor, relevance cutoff) instead of each path
// using its own divergent thresholds.

/// Broad initial recall before selection (shared document/workspace criteria).
pub const RECALL_MAX_DISTANCE: f64 = 0.8;
/// Quality floor: reject a weak best match.
pub const MIN_BEST_DISTANCE: f64 = 0.55;
/// Keep chunks within this margin of the best.
pub const DISTANCE_CUTOFF_MARGIN: f64 = 0.15;
/// How many matched chunks a citation carries for the router to verify against.
///
/// Closest-first, so this keeps the passages the document was actually
/// retrieved for and drops its long tail of weaker hits — which is both the
/// cheaper and the stricter thing to check the finished answer against.
pub const MAX_CITATION_CHUNKS: usize = 5;

/// Documents fetched per parallel batch during vector-search expansion.
const DOC_BATCH_SIZE: usize = 4;

/// Apply the shared relevance-selection criteria to ranked chunks.
///
/// `chunks` must be ordered closest-first (ascending distance).  Returns the
/// kept chunks, or `[]` when the best match fails the quality floor.  Shared by
/// document and workspace retrieval so both apply the same thresholds.
pub fn select_relevant_chunks(chunks: Vec<ChunkResult>, label: &str) -> Vec<ChunkResult> {
    let Some(first) = chunks.first() else {
        return Vec::new();
    };
    let best = first.distance;
    if best > MIN_BEST_DISTANCE {
        tracing::info!(
            target: LOG_TARGET,
            "[{label}] best match distance {best:.3} exceeds quality floor {MIN_BEST_DISTANCE} — no relevant matches"
        );
        return Vec::new();
    }
    let cutoff = best + DISTANCE_CUTOFF_MARGIN;
    let total = chunks.len();
    let kept: Vec<ChunkResult> = chunks.into_iter().filter(|c| c.distance <= cutoff).collect();
    if kept.len() < total {
        tracing::info!(
            target: LOG_TARGET,
            "[{label}] cutoff {cutoff:.3} (best={best:.3}+{DISTANCE_CUTOFF_MARGIN}) trimmed {} chunk(s)",
            total - kept.len()
        );
    }
    kept
}

/// One cached per-capability model slot (Python's dynamic `_cap_llm_<name>`
/// attributes).  `GpuFaulted` is the `_GPU_FAULTED` sentinel: a larger response
/// model that passed the load-time fit guard but exhausted GPU memory during
/// generation, stored in place of the LLM so later requests skip straight to
/// the default router model instead of re-loading a model that won't fit.
#[derive(Clone)]
pub enum CachedLlm {
    /// A resident model instance.
    Loaded(Arc<dyn BaseLLM>),
    /// The GPU-fault eviction latch.
    GpuFaulted,
}

impl CachedLlm {
    /// True for the `_GPU_FAULTED` sentinel.
    pub fn is_gpu_faulted(&self) -> bool {
        matches!(self, CachedLlm::GpuFaulted)
    }

    /// The loaded instance, if this slot holds one.
    pub fn llm(&self) -> Option<&Arc<dyn BaseLLM>> {
        match self {
            CachedLlm::Loaded(llm) => Some(llm),
            CachedLlm::GpuFaulted => None,
        }
    }
}

impl std::fmt::Debug for CachedLlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CachedLlm::Loaded(llm) => write!(f, "Loaded({})", llm.llm_type()),
            CachedLlm::GpuFaulted => write!(f, "GpuFaulted"),
        }
    }
}

/// The `_GPU_FAULTED` cache sentinel.
pub const GPU_FAULTED: CachedLlm = CachedLlm::GpuFaulted;

/// Python `ValueError` / `RuntimeError` raised by the service, so API callers
/// can map them to the same status codes by downcasting the `anyhow::Error`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelServiceError {
    /// `ValueError(msg)`.
    #[error("{0}")]
    Value(String),
    /// `RuntimeError(msg)`.
    #[error("{0}")]
    Runtime(String),
}

impl ModelServiceError {
    /// `ValueError(msg)` as an `anyhow::Error`.
    pub fn value(msg: impl Into<String>) -> anyhow::Error {
        anyhow::Error::new(ModelServiceError::Value(msg.into()))
    }

    /// `RuntimeError(msg)` as an `anyhow::Error`.
    pub fn runtime(msg: impl Into<String>) -> anyhow::Error {
        anyhow::Error::new(ModelServiceError::Runtime(msg.into()))
    }
}

/// True when two LLM handles are the same instance (Python `is`).
///
/// Compares the data pointers only: two `Arc<dyn BaseLLM>` to one object can
/// carry different vtable pointers.
pub fn same_llm(a: &Arc<dyn BaseLLM>, b: &Arc<dyn BaseLLM>) -> bool {
    std::ptr::eq(Arc::as_ptr(a) as *const (), Arc::as_ptr(b) as *const ())
}

/// The command a built-in `-m smartloop.*` MCP server is launched with
/// (`sys.executable`): `SLP_EXE` when set, else this binary.
fn sys_executable() -> String {
    if let Ok(exe) = std::env::var("SLP_EXE") {
        if !exe.is_empty() {
            return exe;
        }
    }
    std::env::current_exe()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "slp".to_string())
}

/// Python's truthiness for an optional string: `None` and `""` are falsy.
fn truthy(s: &Option<String>) -> Option<&str> {
    s.as_deref().filter(|v| !v.is_empty())
}

/// Python's truthiness for a JSON value.
fn value_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python's `repr` of a sorted list of strings (`['a', 'b']`).
fn py_list_repr(items: &[String]) -> String {
    let inner: Vec<String> = items.iter().map(|s| format!("'{s}'")).collect();
    format!("[{}]", inner.join(", "))
}

/// `chunk.document_name or chunk.document_id`.
fn chunk_key(chunk: &ChunkResult) -> String {
    match chunk.document_name.as_deref() {
        Some(name) if !name.is_empty() => name.to_string(),
        _ => chunk.document_id.clone(),
    }
}

/// Sort page numbers the way Python's `sorted()` orders homogeneous values.
fn sort_values(values: &mut [Value]) {
    values.sort_by(|a, b| match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
        _ => a.to_string().cmp(&b.to_string()),
    });
}

/// The logger handed to the task queue (`TaskQueue(logger)`), writing to this
/// module's log target.
struct ServiceLogger;

impl LoggerLike for ServiceLogger {
    fn info(&self, msg: &str) {
        tracing::info!(target: LOG_TARGET, "{msg}");
    }

    fn warning(&self, msg: &str) {
        tracing::warn!(target: LOG_TARGET, "{msg}");
    }

    fn debug(&self, msg: &str) {
        tracing::debug!(target: LOG_TARGET, "{msg}");
    }

    fn error(&self, msg: &str) {
        tracing::error!(target: LOG_TARGET, "{msg}");
    }
}

/// Stand-in for Python's `None` where a [`ModelRouter`] needs a resident model
/// before one is loaded.  The response router only uses its orchestrator slot
/// for identity, so this is never asked to generate; if it is, it fails the
/// way an unloaded service does.
struct UnloadedLLM;

impl BaseLLM for UnloadedLLM {
    fn project_id(&self) -> &str {
        ""
    }

    fn get_num_tokens(&self, _text: &str) -> usize {
        0
    }

    fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        messages
    }

    fn get_max_tokens(&self) -> usize {
        DEFAULT_MAX_CONTEXT_TOKENS
    }

    fn call(&self, _prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
        Err(ModelServiceError::runtime("Model not loaded. Call load_model() first."))
    }

    fn stream<'a>(&'a self, _prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
        Box::new(std::iter::empty())
    }

    fn invoke(&self, _prompt: &str, _max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
        "Error generating response: Model not loaded. Call load_model() first.".to_string()
    }

    fn cancelled(&self) -> Option<CancelEvent> {
        None
    }

    fn set_cancelled(&self, _event: Option<CancelEvent>) {}

    fn llm_type(&self) -> &str {
        "NoneType"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The shared [`UnloadedLLM`] instance, so router identity checks are stable.
static UNLOADED_LLM: Lazy<Arc<dyn BaseLLM>> = Lazy::new(|| Arc::new(UnloadedLLM));

/// The surface the candidate-ranking helpers read off a model: its selection
/// rank (`_rank`: size, then version) and its declared capabilities.
pub trait RankedModel {
    /// `(param_size_b, model_version)`.
    fn model_rank(&self) -> (f64, i64);
    /// Declared capability values (`effective_capabilities`).
    fn model_capabilities(&self) -> Vec<String>;
}

/// How much larger a remote model must be before it outranks a local one
/// for the same capability: a local model counts as this many times its
/// active size, so a remote model up to ~25% larger still loses to it.
/// Within that margin the models are similar and the one on device wins.
pub const LOCAL_SIZE_MARGIN: f64 = 1.25;

impl RankedModel for WorkspaceModel {
    fn model_rank(&self) -> (f64, i64) {
        self.rank()
    }

    fn model_capabilities(&self) -> Vec<String> {
        self.effective_capabilities()
    }
}

/// `len(ref_caps & set(caps))`.
fn cap_overlap<S: AsRef<str>>(ref_caps: &[S], caps: &[String]) -> usize {
    let wanted: HashSet<&str> = ref_caps.iter().map(|c| c.as_ref()).collect();
    let have: HashSet<&str> = caps.iter().map(|c| c.as_str()).collect();
    wanted.intersection(&have).count()
}

/// The per-project store handles a request resolves against.
///
/// One slot per project so two concurrent chats on different projects never
/// evict each other's store (reopening SQLite each turn, or answering one
/// project's query from the other's index).
#[derive(Default, Clone)]
pub struct ProjectCaches {
    pub workspace_indexer: Option<Arc<WorkspaceIndexer>>,
    pub feedback_indexer: Option<Arc<FeedbackIndexer>>,
    pub doc_store: Option<Arc<dyn VectorStore>>,
}

/// A counting semaphore sized to `n_parallel` (Python `threading.Semaphore`).
pub struct InferenceSemaphore {
    permits: Mutex<usize>,
    cond: Condvar,
}

impl InferenceSemaphore {
    /// A semaphore with `permits` slots.
    pub fn new(permits: usize) -> Self {
        Self { permits: Mutex::new(permits), cond: Condvar::new() }
    }

    /// Block until a slot is free and take it.
    pub fn acquire(&self) {
        let mut permits = self.permits.lock();
        while *permits == 0 {
            self.cond.wait(&mut permits);
        }
        *permits -= 1;
    }

    /// Return a slot.
    pub fn release(&self) {
        let mut permits = self.permits.lock();
        *permits += 1;
        self.cond.notify_one();
    }
}

/// Inference bookkeeping guarded by `_inference_state_lock`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct InferenceState {
    /// `_inference_active`: generations currently holding a slot.
    pub active: usize,
    /// `_last_inference_at`: UNIX seconds the last generation finished (0.0 = never).
    pub last_inference_at: f64,
}

/// `(command, args, env, cwd) -> connect result` seam for local MCP discovery.
pub type LocalMcpDiscovery = Arc<
    dyn Fn(&str, &[String], &HashMap<String, String>, Option<&str>) -> Result<MCPConnectResult, MCPClientError>
        + Send
        + Sync,
>;

/// `probe_account(server, tool_caller) -> (account, credentials_changed)` seam.
pub type ProbeAccount =
    Arc<dyn Fn(&mut MCPServer, Option<Arc<dyn ReadOnlyToolCaller>>) -> (Option<String>, bool) + Send + Sync>;

/// `(project_dir, base_dir) -> FeedbackIndexer` constructor seam.
pub type FeedbackIndexerFactory =
    Arc<dyn Fn(&str, Option<&str>) -> anyhow::Result<Arc<FeedbackIndexer>> + Send + Sync>;

/// Overridable seams standing in for the Python tests' monkeypatching of
/// methods on the instance.  Each `None` runs the real implementation.
#[derive(Clone, Default)]
pub struct ModelServiceHooks {
    /// Replaces `_free_llm(llm)`.
    pub free_llm: Option<Arc<dyn Fn(&Arc<dyn BaseLLM>) + Send + Sync>>,
    /// Replaces `_cleanup_gpu_cache()`.
    pub cleanup_gpu_cache: Option<Arc<dyn Fn() + Send + Sync>>,
    /// Replaces `_device_free_total_gb()`.
    pub device_free_total_gb: Option<Arc<dyn Fn() -> Option<(f64, f64)> + Send + Sync>>,
    /// Replaces `_large_model_fits(gguf_path, cap, label, reserve_gb)`.
    pub large_model_fits: Option<Arc<dyn Fn(&str, i64, &str, Option<f64>) -> bool + Send + Sync>>,
    /// Replaces `_find_model_ggufs(model_dir)`.
    pub find_model_ggufs: Option<Arc<dyn Fn(&str) -> (Option<String>, Option<String>) + Send + Sync>>,
    /// Replaces `LlmFactory.get_llama_cpp_instance(gguf_path, ..., max_n_ctx=cap,
    /// thinking=..., model_name=...)` for capability/large loads.
    pub load_llama_instance: Option<
        Arc<dyn Fn(&str, &str, i64, Option<bool>) -> anyhow::Result<Arc<dyn BaseLLM>> + Send + Sync>,
    >,
    /// Replaces `generate_completion(prompt)` for the query-rewrite helpers.
    pub generate_completion: Option<Arc<dyn Fn(&str) -> anyhow::Result<String> + Send + Sync>>,
    /// Replaces `_get_project(project_id)`.
    pub get_project: Option<Arc<dyn Fn(&str) -> anyhow::Result<Project> + Send + Sync>>,
    /// Replaces `_ensure_workspace(model_config)` (the API tests'
    /// `model_service._ensure_workspace.side_effect = ...`).
    pub ensure_workspace:
        Option<Arc<dyn Fn(Option<&ModelConfig>) -> anyhow::Result<Arc<RwLock<Workspace>>> + Send + Sync>>,
    /// Replaces `load_model()` (the API tests' `model_service.load_model =
    /// MagicMock(side_effect=...)`).
    pub load_model: Option<Arc<dyn Fn() -> anyhow::Result<Arc<dyn BaseLLM>> + Send + Sync>>,
    /// Replaces `add_local_mcp_server(...)`; receives `(command, project_id)`.
    pub add_local_mcp_server: Option<Arc<dyn Fn(&str, Option<&str>) -> anyhow::Result<Value> + Send + Sync>>,
    /// Replaces `_setup_project_tools(project)` where `create_project` calls it
    /// (receives the project id).
    pub setup_project_tools: Option<Arc<dyn Fn(&str) -> anyhow::Result<()> + Send + Sync>>,
    /// Replaces `_llm_for_response(capability)`.
    pub llm_for_response: Option<Arc<dyn Fn(Option<&str>) -> Option<Arc<dyn BaseLLM>> + Send + Sync>>,
    /// Replaces `self._model.save()` wherever the service persists the
    /// workspace (tests assert on whether it was called).
    pub save_model: Option<Arc<dyn Fn() -> anyhow::Result<()> + Send + Sync>>,
    /// Replaces `workspace.ranked_models_for(capability, current_project)`.
    pub ranked_models_for: Option<Arc<dyn Fn(&str) -> Vec<WorkspaceModel> + Send + Sync>>,
    /// Replaces `_pinned_model(capability)`.
    pub pinned_model: Option<Arc<dyn Fn(Option<&str>) -> Option<WorkspaceModel> + Send + Sync>>,
    /// Replaces `_pinned_llm(capability)`.
    pub pinned_llm: Option<Arc<dyn Fn(Option<&str>) -> Option<Arc<dyn BaseLLM>> + Send + Sync>>,
    /// Replaces `_load_workspace_llm(ws_model, reason, reserve_gb)`; the last
    /// argument says whether the load may evict prior-step models to fit.
    pub load_workspace_llm: Option<
        Arc<dyn Fn(&WorkspaceModel, &str, Option<f64>, bool) -> Option<Arc<dyn BaseLLM>> + Send + Sync>,
    >,
    /// Replaces `_get_llm_for_capability(capability)`.
    pub get_llm_for_capability: Option<Arc<dyn Fn(&str) -> Option<Arc<dyn BaseLLM>> + Send + Sync>>,
    /// Replaces `has_capability_model(capability)`.
    pub has_capability_model: Option<Arc<dyn Fn(&str) -> bool + Send + Sync>>,
    /// Replaces `last_capability_skip()`.
    pub last_capability_skip: Option<Arc<dyn Fn() -> Option<CapabilitySkipReason> + Send + Sync>>,
    /// Replaces `_get_large_llm()`.
    pub get_large_llm: Option<Arc<dyn Fn() -> Option<Arc<dyn BaseLLM>> + Send + Sync>>,
    /// Replaces `_evict_capability_models(keep_model_name)`.
    pub evict_capability_models: Option<Arc<dyn Fn(Option<&str>) -> bool + Send + Sync>>,
    /// Replaces the `FeedbackIndexer(project_dir=..., base_dir=...)` constructor.
    pub feedback_indexer_factory: Option<FeedbackIndexerFactory>,

    // ---- MCP provisioning seams, consulted by the methods in `part2` ----
    /// Replaces `_queue_mcp_provisioning(project, server)`.
    pub queue_mcp_provisioning: Option<Arc<dyn Fn(&Project, &MCPServer) + Send + Sync>>,
    /// Replaces `_discover_mcp_tools(server)` where `_provision_mcp_server`
    /// calls it.
    pub discover_mcp_tools: Option<Arc<dyn Fn(&mut MCPServer) -> anyhow::Result<()> + Send + Sync>>,
    /// Replaces `_enrich_mcp_server(server)` where `_provision_mcp_server`
    /// calls it.
    pub enrich_mcp_server: Option<Arc<dyn Fn(&mut MCPServer) + Send + Sync>>,
    /// Replaces the `LocalMCPClient(command, args=..., env=..., cwd=...)`
    /// launch + `connect_and_discover_tools()` of local discovery; receives
    /// `(command, args, env, cwd)`.
    pub local_mcp_discovery: Option<LocalMcpDiscovery>,
    /// Replaces the `MCPClient().connect_and_discover_tools(server_url,
    /// access_token=...)` of remote discovery; receives `(server_url,
    /// access_token)`.
    pub remote_mcp_discovery: Option<Arc<dyn Fn(&str, Option<&str>) -> MCPConnectResult + Send + Sync>>,
    /// Replaces `smartloop.mcp.account.probe_account(server, tool_caller=...)`
    /// where `_probe_mcp_account` calls it.
    pub probe_account: Option<ProbeAccount>,
}

/// The process-wide singleton (`ModelService._instance`).
static INSTANCE: OnceCell<Arc<ModelService>> = OnceCell::new();

impl ModelService {
    /// The singleton if it has been created (`ModelService._instance`),
    /// without creating it.  Added by the `db` unit for
    /// `embeddings::device_idle_seconds`.
    pub fn existing_instance() -> Option<Arc<ModelService>> {
        INSTANCE.get().cloned()
    }
}

/// Singleton service to manage model lifecycle and inference.
///
/// All state uses interior mutability so the service is shared as
/// `Arc<ModelService>` across request handlers, DAG workers and background
/// tasks, exactly as the Python singleton was shared across threads.
pub struct ModelService {
    /// Back-reference so `&self` methods can hand out `Arc<Self>` (the model
    /// client and the response router need an owning handle).
    weak_self: Weak<ModelService>,

    /// `_model_client`: how this process reaches the generative model.
    pub model_client_slot: Mutex<Option<Arc<dyn ModelClient>>>,

    /// `_llm`: the always-resident orchestrator model.
    pub llm: RwLock<Option<Arc<dyn BaseLLM>>>,
    /// `_model`: the Model object (gives access to the workspace).
    pub model: RwLock<Option<Arc<Model>>>,
    /// `_model_name`.
    pub model_name: RwLock<Option<String>>,
    /// `_model_gguf_path`: GGUF of the orchestrator model, for worker processes.
    pub model_gguf_path: RwLock<Option<String>>,
    /// `_decision_llm`: smaller LLM for classify / match_skills nodes.
    pub decision_llm: RwLock<Option<Arc<dyn BaseLLM>>>,
    /// `_decision_model_name`.
    pub decision_model_name: RwLock<Option<String>>,
    /// `_large_llm`: the on-demand agent model.
    pub large_llm: RwLock<Option<Arc<dyn BaseLLM>>>,
    /// `_large_model_name`.
    pub large_model_name: RwLock<Option<String>>,
    /// `_vision_llm`: multimodal vision model for document OCR.
    pub vision_llm: RwLock<Option<Arc<dyn BaseLLM>>>,
    /// `_vision_model_name`.
    pub vision_model_name: RwLock<Option<String>>,
    /// `_vision_unavailable`.
    pub vision_unavailable: AtomicBool,
    /// `_vision_clip_cpu`: latched after a GPU clip encode crashes.
    pub vision_clip_cpu: AtomicBool,
    /// `_large_model_attempted`: the large model has been considered.
    pub large_model_attempted: AtomicBool,
    /// `_cap_llm_<model_name>` slots, in insertion order.
    pub cap_llms: Mutex<Vec<(String, CachedLlm)>>,
    /// `_capability_skip`: why the most recent capability model could not load.
    pub capability_skip: Mutex<Option<CapabilitySkipReason>>,
    /// `_last_fit_shortfall`: `(needed_gb, free_gb)` from the last failed fit check.
    pub last_fit_shortfall: Mutex<Option<(f64, f64)>>,
    /// `_llm_executor`: shared, lazily created LLM executor.
    pub llm_executor: Mutex<Option<Arc<LLMExecutor>>>,
    /// `_inference_lock`: semaphore sized to `n_parallel`.
    pub inference_lock: InferenceSemaphore,
    /// `_inference_state_lock` + `_inference_active` + `_last_inference_at`.
    pub inference_state: Mutex<InferenceState>,
    /// Bumped each time a response completes; an idle-unload timer only acts
    /// while the epoch it was started under is still current.
    pub idle_unload_epoch: AtomicUsize,
    /// `_sessions`: session_id -> chat history (in-memory cache).
    pub sessions: Mutex<HashMap<String, Vec<MessageInput>>>,
    /// `_conversation_store`.
    pub conversation_store: RwLock<Option<Arc<ConversationStore>>>,
    /// `_task_queue`.
    pub task_queue: RwLock<Arc<TaskQueue>>,
    /// `_project_caches` (guarded by `_project_caches_lock`).
    pub project_caches: Mutex<HashMap<String, Arc<Mutex<ProjectCaches>>>>,
    /// `_response_router_instance`.
    pub response_router_instance: Mutex<Option<Arc<ModelRouter>>>,
    /// The progress sink installed by the current request's
    /// `begin_response_request`, re-armed on a router rebuilt mid-request.
    pub response_progress_callback: Mutex<Option<ProgressCallback>>,
    /// Test seams (see [`ModelServiceHooks`]).
    pub hooks: RwLock<ModelServiceHooks>,
}

impl ModelService {
    /// Built-in tool skill names never exposed by `list_skills`.
    pub const SYSTEM_SKILL_NAMES: &'static [&'static str] =
        &["file-read", "file-write", "bash", "grep", "explore"];

    /// Legacy fetch skill directories swept from disk.
    pub const STALE_FETCH_SKILLS: &'static [&'static str] = &["fetch-mcp", "fetch-url"];

    /// Purpose label for response-model events.
    pub const RESPONSE_PURPOSE: &'static str = "the response";

    /// The singleton (`ModelService()`): created on first call, then shared.
    pub fn instance() -> Arc<ModelService> {
        INSTANCE.get_or_init(Self::new_detached).clone()
    }

    /// Build an instance with every field empty, around the given store,
    /// task queue and inference width.
    fn build(
        conversation_store: Option<Arc<ConversationStore>>,
        task_queue: Arc<TaskQueue>,
        n_parallel: usize,
    ) -> Arc<ModelService> {
        Arc::new_cyclic(|weak| ModelService {
            weak_self: weak.clone(),
            model_client_slot: Mutex::new(None),
            llm: RwLock::new(None),
            model: RwLock::new(None),
            model_name: RwLock::new(None),
            model_gguf_path: RwLock::new(None),
            decision_llm: RwLock::new(None),
            decision_model_name: RwLock::new(None),
            large_llm: RwLock::new(None),
            large_model_name: RwLock::new(None),
            vision_llm: RwLock::new(None),
            vision_model_name: RwLock::new(None),
            vision_unavailable: AtomicBool::new(false),
            vision_clip_cpu: AtomicBool::new(false),
            large_model_attempted: AtomicBool::new(false),
            cap_llms: Mutex::new(Vec::new()),
            capability_skip: Mutex::new(None),
            last_fit_shortfall: Mutex::new(None),
            llm_executor: Mutex::new(None),
            inference_lock: InferenceSemaphore::new(n_parallel),
            inference_state: Mutex::new(InferenceState::default()),
            idle_unload_epoch: AtomicUsize::new(0),
            sessions: Mutex::new(HashMap::new()),
            conversation_store: RwLock::new(conversation_store),
            task_queue: RwLock::new(task_queue),
            project_caches: Mutex::new(HashMap::new()),
            response_router_instance: Mutex::new(None),
            response_progress_callback: Mutex::new(None),
            hooks: RwLock::new(ModelServiceHooks::default()),
        })
    }

    /// A fresh, fully initialised instance that is *not* the singleton.
    pub fn new_detached() -> Arc<ModelService> {
        let s = settings();
        // Use a semaphore sized to n_parallel so agents can run concurrently
        // when the model is loaded with n_parallel > 1 (continuous batching).
        // Always taken through `inference_slot`, which is what makes "is the
        // model answering someone right now" an answerable question —
        // background work waits for a quiet moment rather than competing
        // with a reply the user is watching stream.
        let n_parallel = s.n_parallel.max(1) as usize;
        let store = match ConversationStore::new(&s.home_dir()) {
            Ok(store) => Some(Arc::new(store)),
            Err(e) => {
                tracing::error!(target: LOG_TARGET, "Could not open the conversation store: {e}");
                None
            }
        };
        let queue = Arc::new(TaskQueue::new(Arc::new(ServiceLogger)));
        Self::build(store, queue, n_parallel)
    }

    /// An instance with every field at its empty default and no conversation
    /// store — the Rust analogue of `object.__new__(ModelService)` in tests.
    pub fn new_bare() -> Arc<ModelService> {
        let queue = Arc::new(TaskQueue::new(Arc::new(ServiceLogger)));
        Self::build(None, queue, 1)
    }

    /// An owning handle to this instance.
    pub fn arc(&self) -> Arc<ModelService> {
        self.weak_self.upgrade().expect("ModelService is always owned by an Arc")
    }

    /// Replace the test seams.
    pub fn set_hooks(&self, hooks: ModelServiceHooks) {
        *self.hooks.write() = hooks;
    }

    /// A snapshot of the current test seams.
    fn hooks(&self) -> ModelServiceHooks {
        self.hooks.read().clone()
    }

    /// The `_SYSTEM_TOOL_DEFS` registered on the System Tools MCP server.
    pub fn system_tool_defs() -> Vec<MCPToolDefinition> {
        fn def(name: &str, description: &str, schema: Value) -> MCPToolDefinition {
            MCPToolDefinition {
                name: name.to_string(),
                description: Some(description.to_string()),
                input_schema: schema.as_object().cloned(),
            }
        }
        vec![
            def(
                "bash",
                "Execute a shell command (git, echo, etc.) on the host machine",
                json!({"type": "object", "properties": {"command": {"type": "string", "description": "Shell command to execute"}}, "required": ["command"]}),
            ),
            def(
                "file-read",
                "Read a file and return its content",
                json!({"type": "object", "properties": {"path": {"type": "string", "description": "File path to read"}}, "required": ["path"]}),
            ),
            def(
                "file-write",
                "Write content to a file in the working directory",
                json!({"type": "object", "properties": {"path": {"type": "string", "description": "File path to write"}, "content": {"type": "string", "description": "Content to write"}}, "required": ["path", "content"]}),
            ),
            def(
                "grep",
                "Recursively search files in the working directory for a text pattern or regex",
                json!({"type": "object", "properties": {"pattern": {"type": "string", "description": "Text or regex to search for"}, "glob": {"type": "string", "description": "Optional filename filter, e.g. '*.py'"}}, "required": ["pattern"]}),
            ),
            def(
                "explore",
                "Explore the local project directory, read files, and produce an LLM-enriched summary of the codebase",
                json!({"type": "object", "properties": {"query": {"type": "string", "description": "What to look for in the codebase"}}, "required": ["query"]}),
            ),
        ]
    }

    /// The `_DEFAULT_SKILLS` written to every project.
    pub fn default_skills() -> Vec<Skill> {
        fn skill(name: &str, description: &str, content: &str) -> Skill {
            Skill {
                id: name.to_string(),
                name: name.to_string(),
                description: Some(description.to_string()),
                content: content.to_string(),
                system: false,
            }
        }
        vec![
            skill(
                "file-read",
                "Use when the user wants to read, view, or inspect the contents of a file.",
                "read the file at the given path and return its content",
            ),
            skill(
                "file-write",
                "Use when the user wants to write, save, or create a file with specific content.",
                "write content to a file at the given path",
            ),
            skill(
                "bash",
                "Use when the user wants to run a shell command, execute a script, or perform system operations.",
                "execute the shell command on the host machine",
            ),
        ]
    }

    // ------------------------------------------------------------ shared state

    /// The `Model` object, if one is initialised (`self._model`).
    pub fn model_handle(&self) -> Option<Arc<Model>> {
        self.model.read().clone()
    }

    /// The workspace the initialised `Model` serves from (`self._model.workspace`).
    pub fn workspace_handle(&self) -> Option<SharedWorkspace> {
        self.model_handle().map(|m| m.workspace.clone())
    }

    /// `self._model.save()` (or the `save_model` seam).
    pub fn persist_model(&self) -> anyhow::Result<()> {
        if let Some(hook) = self.hooks().save_model {
            return hook();
        }
        match self.model_handle() {
            Some(model) => model.save(),
            None => Err(ModelServiceError::runtime("Model not loaded")),
        }
    }

    /// The current project of the live workspace, as a snapshot.
    fn current_project_copy(&self) -> Option<Project> {
        let ws = self.workspace_handle()?;
        let guard = ws.read();
        guard.current_project().cloned()
    }

    /// Resolve `project_id` the way [`ModelService::get_project`] does
    /// (re-reading the workspace once on a miss, sweeping the legacy fetch
    /// server), then run `f` on the live record in the workspace.
    ///
    /// The Python service mutated the object `_get_project` returned; here the
    /// project lives behind the workspace lock, so mutations go through this.
    pub fn update_project_record<R>(&self, project_id: &str, f: impl FnOnce(&mut Project) -> R) -> anyhow::Result<R> {
        self.get_project(project_id)?;
        let ws = self
            .workspace_handle()
            .ok_or_else(|| ModelServiceError::runtime("Model not loaded"))?;
        let mut guard = ws.write();
        let project = guard
            .project_mut(project_id)
            .ok_or_else(|| ModelServiceError::value(format!("Project not found: {project_id}")))?;
        Ok(f(project))
    }

    // ------------------------------------------------------------ model client

    /// How this process reaches the generative model: in-process by default,
    /// or the model host named by `SLP_MODEL_HOST`.  Resolved once.
    ///
    /// In-process the same `stream_completion` / `generate_completion` calls
    /// sit behind one interface, so the seam costs a method call and nothing
    /// else.  Resolved once: the transport is a property of how this process
    /// was started, and re-reading the environment per call would let it
    /// change underneath an in-flight generation.
    pub fn model_client(&self) -> Result<Arc<dyn ModelClient>, ModelHostError> {
        let mut slot = self.model_client_slot.lock();
        if let Some(client) = slot.as_ref() {
            return Ok(client.clone());
        }
        let service: Arc<dyn LocalModelService> = self.arc();
        let client = get_model_client(Some(service))?;
        *slot = Some(client.clone());
        Ok(client)
    }

    /// Swap in a model client (tests), or clear it so it is re-resolved.
    pub fn set_model_client(&self, client: Option<Arc<dyn ModelClient>>) {
        *self.model_client_slot.lock() = client;
    }

    /// The model client when the weights live in another process, else `None`.
    ///
    /// In-process — the supervisor, and the `--no-supervised` fallback — this
    /// is `None` and every path behaves exactly as it always has.  In a
    /// per-project agent it is the host, and the service builds handles to it
    /// rather than loading a second copy of the weights into the same device.
    pub fn remote_device(&self) -> Option<Arc<dyn ModelClient>> {
        match self.model_client() {
            Ok(client) if client.as_remote().is_some() => Some(client),
            Ok(_) => None,
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "Could not resolve the model client: {e}");
                None
            }
        }
    }

    /// A handle to the host's model for `capability`.
    pub fn remote_llm(&self, capability: Option<&str>) -> Arc<dyn BaseLLM> {
        let client = self
            .remote_device()
            .or_else(|| self.model_client().ok())
            .expect("a remote model handle needs a model client");
        Arc::new(RemoteLLM::new(
            client,
            self.get_project_id().unwrap_or_default(),
            capability.map(str::to_string),
        ))
    }

    // ------------------------------------------------------- per-project caches

    /// Return (creating if needed) the cache slot for `project_id`.
    pub fn caches(&self, project_id: &str) -> Arc<Mutex<ProjectCaches>> {
        let mut caches = self.project_caches.lock();
        caches
            .entry(project_id.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(ProjectCaches::default())))
            .clone()
    }

    /// Discard cached stores for `project_id`, or for every project.
    ///
    /// Called when a project's on-disk state changes underneath the handles
    /// (create, delete, UI switch), so the next read reopens rather than
    /// querying an index that no longer describes the directory.
    pub fn drop_caches(&self, project_id: Option<&str>) {
        let mut caches = self.project_caches.lock();
        match project_id {
            None => caches.clear(),
            Some(id) => {
                caches.remove(id);
            }
        }
    }

    // ------------------------------------------------------- per-model cache slots

    /// The `_cap_llm_<model_name>` slot, if set.
    pub fn cap_llm(&self, model_name: &str) -> Option<CachedLlm> {
        self.cap_llms
            .lock()
            .iter()
            .find(|(name, _)| name == model_name)
            .map(|(_, slot)| slot.clone())
    }

    /// Set (`Some`) or clear (`None`) the `_cap_llm_<model_name>` slot.
    pub fn set_cap_llm(&self, model_name: &str, value: Option<CachedLlm>) {
        let mut slots = self.cap_llms.lock();
        match value {
            Some(slot) => {
                if let Some(entry) = slots.iter_mut().find(|(name, _)| name == model_name) {
                    entry.1 = slot;
                } else {
                    slots.push((model_name.to_string(), slot));
                }
            }
            None => slots.retain(|(name, _)| name != model_name),
        }
    }

    // ---------------------------------------------------------------- sessions

    /// Load a session from the conversation store (empty without one).
    fn load_session(&self, session_id: &str) -> Vec<MessageInput> {
        let store = self.conversation_store.read().clone();
        match store {
            Some(store) => store.load(session_id).unwrap_or_else(|e| {
                tracing::warn!(target: LOG_TARGET, "Could not load session {session_id}: {e}");
                Vec::new()
            }),
            None => Vec::new(),
        }
    }

    /// Get chat history for a session, loading from disk on first access.
    pub fn get_session_history(&self, session_id: &str) -> Vec<MessageInput> {
        if let Some(history) = self.sessions.lock().get(session_id) {
            return history.clone();
        }
        let loaded = self.load_session(session_id);
        self.sessions
            .lock()
            .entry(session_id.to_string())
            .or_insert(loaded)
            .clone()
    }

    /// Return the most recent conversation turns as `{"role","content"}` maps,
    /// with strict user/assistant alternation, trimmed to `max_turns` pairs
    /// (Python default 4) and starting with a user message.
    pub fn get_relevant_history(
        &self,
        session_id: &str,
        user_query: &str,
        max_turns: usize,
    ) -> Vec<HashMap<String, String>> {
        let _ = user_query;
        let history = self.get_session_history(session_id);

        // Enforce strict alternation: collapse consecutive same-role
        // messages and ensure the sequence starts with user.
        let mut alternating: Vec<HashMap<String, String>> = Vec::new();
        for m in history.iter().filter(|m| !m.content.is_empty()) {
            let same_role = alternating
                .last()
                .map(|last| last.get("role").map(String::as_str) == Some(m.role.as_str()))
                .unwrap_or(false);
            if same_role {
                // Merge into the previous message of the same role
                let last = alternating.last_mut().expect("checked non-empty");
                let content = last.entry("content".to_string()).or_default();
                content.push_str("\n\n");
                content.push_str(&m.content);
            } else {
                let mut msg = HashMap::new();
                msg.insert("role".to_string(), m.role.clone());
                msg.insert("content".to_string(), m.content.clone());
                alternating.push(msg);
            }
        }

        // Trim to last max_turns pairs (user+assistant = 2 messages each)
        if alternating.len() > max_turns * 2 {
            alternating = alternating.split_off(alternating.len() - max_turns * 2);
        }

        // Must start with a user message for proper alternation
        if alternating
            .first()
            .map(|first| first.get("role").map(String::as_str) != Some("user"))
            .unwrap_or(false)
        {
            alternating.remove(0);
        }

        alternating
    }

    /// Append a message to session history, persist it, and track the session
    /// on the current project.
    pub fn append_to_session(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
        citations: Option<Vec<Value>>,
        message_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let needs_load = !self.sessions.lock().contains_key(session_id);
        if needs_load {
            let loaded = self.load_session(session_id);
            self.sessions.lock().entry(session_id.to_string()).or_insert(loaded);
        }
        let message = MessageInput {
            role: role.to_string(),
            content: content.to_string(),
            message_id: message_id.map(str::to_string),
            rating: None,
            citations,
        };
        self.sessions
            .lock()
            .entry(session_id.to_string())
            .or_default()
            .push(message.clone());
        let store = self.conversation_store.read().clone();
        if let Some(store) = store {
            store.append(session_id, &message)?;
        }

        // Track session on the current project
        if let Some(ws) = self.workspace_handle() {
            let added = {
                let mut guard = ws.write();
                match guard.current_project_mut() {
                    Some(project) if !project.sessions.iter().any(|s| s == session_id) => {
                        project.sessions.push(session_id.to_string());
                        true
                    }
                    _ => false,
                }
            };
            if added {
                self.persist_model()?;
            }
        }
        Ok(())
    }

    // --------------------------------------------------------------- workspace

    /// The `ModelConfig` for the default model built from the API settings.
    fn default_model_config(default: &str) -> ModelConfig {
        let api = api_settings();
        ModelConfig {
            model_name: default.to_string(),
            hf_token: Some(api.hf_token.clone()),
            developer_token: api.developer_token.clone(),
            ..Default::default()
        }
    }

    /// Ensure the default orchestrator model is initialized.
    pub fn ensure_model(&self) -> anyhow::Result<()> {
        let default = api_settings().default_model;

        if self.model.read().is_some() && self.model_name.read().as_deref() == Some(default.as_str()) {
            return Ok(());
        }

        let workspace = self.ensure_workspace(Some(Self::default_model_config(&default)))?;

        {
            let mut ws = workspace.write();
            if ws.get_model(Some(&default)).is_none() {
                let Some(spec) = MODEL_SPECS.get(&default) else {
                    return Err(ModelServiceError::value(format!(
                        "Default model {default} not found in workspace"
                    )));
                };
                ws.add_model(
                    &default,
                    &spec.model_id,
                    Some(Self::default_model_config(&default)),
                    true,
                );
            }
        }

        let model_config = Self::default_model_config(&default);

        tracing::info!(target: LOG_TARGET, "[_ensure_model] loading default model {default}");
        let model = ModelFactory::get_model(&model_config)?;
        *self.model.write() = Some(Arc::new(model));
        *self.model_name.write() = Some(default);
        Ok(())
    }

    /// Return the workspace, creating it if necessary.
    ///
    /// If a model pipeline has already been initialized its workspace is
    /// returned directly.  Otherwise the workspace is loaded from disk,
    /// falling back to `Workspace::init` when no workspace exists yet.
    pub fn ensure_workspace(&self, model_config: Option<ModelConfig>) -> anyhow::Result<Arc<RwLock<Workspace>>> {
        if let Some(hook) = self.hooks().ensure_workspace {
            return hook(model_config.as_ref());
        }
        if let Some(workspace) = self.workspace_handle() {
            if pinned_project_id().is_some() {
                // A child holds the workspace it loaded at spawn, and the
                // model registry inside it is written by the process that owns
                // config.json — so a model downloaded since is invisible here
                // until the child restarts.  Gated on the file's mtime, and
                // only in a child: the owner must not re-read its own record
                // out from under mutations it has not saved yet (/v1/init
                // registers a model well before it persists one).
                workspace.write().refresh_models();
            }
            return Ok(workspace);
        }

        if let Some(workspace) = Workspace::load() {
            return Ok(Arc::new(RwLock::new(workspace)));
        }

        let model_config =
            model_config.unwrap_or_else(|| Self::default_model_config(&api_settings().default_model));
        Ok(Workspace::init(&model_config))
    }

    /// The workspace instance this process actually serves from (never a
    /// detached copy).  Use anywhere a project is created or adopted.
    ///
    /// Narrower than [`ModelService::ensure_workspace`], which falls back to
    /// `Workspace::load()` — a *detached* instance, fresh per call — when no
    /// model is initialised yet.  Whatever a caller adds to a detached
    /// instance is invisible to this process's own reads, and the next save
    /// from the instance it does serve writes a project list without it.
    pub fn live_workspace(&self) -> anyhow::Result<Arc<RwLock<Workspace>>> {
        self.ensure_model()?;
        self.workspace_handle()
            .ok_or_else(|| ModelServiceError::runtime("Model not loaded"))
    }

    /// Pick up a child's write to `project_id`'s own file, if there is one.
    ///
    /// The model host runs in this process and answers for whichever project
    /// asked, but a project's own state is written by the child pinned to it.
    pub fn refresh_project_state(&self, project_id: Option<&str>) -> bool {
        let Some(project_id) = project_id.filter(|p| !p.is_empty()) else {
            return false;
        };
        let Some(ws) = self.workspace_handle() else {
            return false;
        };
        ws.write().refresh_project(project_id)
    }

    /// Ensure the workspace is loaded, raising if it cannot be resolved.
    pub fn initialize_workspace(&self) -> anyhow::Result<()> {
        self.ensure_model()?;
        if self.model.read().is_none() {
            return Err(ModelServiceError::runtime("Model not loaded"));
        }
        Ok(())
    }

    /// Find a project by ID (re-reading the workspace from disk once on a
    /// miss), raising `ValueError("Project not found: …")`.  Returns a snapshot.
    pub fn get_project(&self, project_id: &str) -> anyhow::Result<Project> {
        if let Some(hook) = self.hooks().get_project {
            return hook(project_id);
        }
        self.initialize_workspace()?;
        let ws = self
            .workspace_handle()
            .ok_or_else(|| ModelServiceError::runtime("Model not loaded"))?;
        let found = ws.read().project(project_id).is_some();
        if !found {
            // The cached workspace singleton can be stale when another process
            // (e.g. the CLI/TUI) created the project and wrote config.json
            // after this server loaded. Re-read from disk once before giving up.
            ws.write().reload();
        }
        if ws.read().project(project_id).is_none() {
            return Err(ModelServiceError::value(format!("Project not found: {project_id}")));
        }
        self.sweep_legacy_fetch_server(project_id);
        let project = ws.read().project(project_id).cloned();
        project.ok_or_else(|| ModelServiceError::value(format!("Project not found: {project_id}")))
    }

    /// Drop the obsolete built-in `__fetch__` MCP server from a project.
    ///
    /// URL fetching moved to the in-process FetchAgent, but projects created
    /// before that change still carry a persisted `__fetch__` entry that shows
    /// up in MCP listings.  Sweep on every project access so they self-heal.
    /// Idempotent — only writes when an entry is actually removed.
    pub fn sweep_legacy_fetch_server(&self, project_id: &str) {
        let Some(ws) = self.workspace_handle() else {
            return;
        };
        let removed = {
            let mut guard = ws.write();
            match guard.project_mut(project_id) {
                Some(project) if project.mcp_servers.iter().any(|s| s.id == FETCH_SERVER_ID) => {
                    project.mcp_servers.retain(|s| s.id != FETCH_SERVER_ID);
                    true
                }
                _ => false,
            }
        };
        if !removed {
            return;
        }
        match self.persist_model() {
            Ok(()) => {
                tracing::info!(target: LOG_TARGET, "Swept legacy fetch server from project {project_id}");
            }
            Err(e) => {
                // Never let a persistence hiccup break project lookup.
                tracing::warn!(
                    target: LOG_TARGET,
                    "Failed to persist legacy fetch sweep for {project_id}: {e}"
                );
            }
        }
    }

    // ------------------------------------------------------------------ models

    /// Load and cache the orchestrator LLM for the current project.
    ///
    /// The routing/main model is always the small base model
    /// (`api_settings.default_model`), independent of the workspace's active
    /// selection.  A larger model is loaded lazily as the on-demand large
    /// agent model only when a plan step needs it.
    pub fn load_model(&self) -> anyhow::Result<Arc<dyn BaseLLM>> {
        if let Some(hook) = self.hooks.read().load_model.clone() {
            return hook();
        }
        let workspace = self.ensure_workspace(None)?;
        let target_model = api_settings().default_model;
        tracing::info!(target: LOG_TARGET, "[load_model] target={target_model} (orchestrator model)");

        // Return cached model if already loaded
        let current_llm = self.llm.read().clone();
        let current_name = self.model_name.read().clone();
        if let Some(llm) = &current_llm {
            if current_name.as_deref() == Some(target_model.as_str()) {
                tracing::info!(target: LOG_TARGET, "Using cached model: {target_model}");
                return Ok(llm.clone());
            }
        }

        // If a different model is loaded, unload it first
        if current_llm.is_some() && current_name.as_deref() != Some(target_model.as_str()) {
            tracing::info!(
                target: LOG_TARGET,
                "Switching model from {} to {target_model}",
                current_name.as_deref().unwrap_or("None")
            );
            *self.llm.write() = None;
            *self.model.write() = None;
            *self.model_name.write() = None;
            *self.decision_llm.write() = None;
            *self.decision_model_name.write() = None;
            Workspace::reset();
        }

        self.ensure_model()?;

        let project_name = self.get_current_project_name().unwrap_or_else(|| "default".to_string());
        tracing::info!(target: LOG_TARGET, "Loading model: {target_model} (project: {project_name})");

        // Require an active project before attempting to load
        let current_project = workspace.read().current_project().cloned();
        let Some(current_project) = current_project else {
            return Err(ModelServiceError::value(
                "No active project. Please create a project first.",
            ));
        };
        self.setup_project_tools(&current_project.id)?;

        if self.remote_device().is_some() {
            // The device is the model host's.  Every role the orchestrator
            // asks for — main, decision, large, capability — becomes a handle
            // to it; the host does the selection, because it is the only side
            // that can see what fits.
            let llm = self.remote_llm(None);
            *self.llm.write() = Some(llm.clone());
            *self.decision_llm.write() = None;
            *self.model_name.write() = Some(target_model.clone());
            tracing::info!(
                target: LOG_TARGET,
                "[load_model] using the model host for {target_model} (project: {project_name})"
            );
            return Ok(llm);
        }

        let model = self
            .model_handle()
            .ok_or_else(|| ModelServiceError::runtime("Model not loaded"))?;
        let llm = model.get_instance()?;
        *self.llm.write() = Some(llm.clone());
        *self.model_name.write() = Some(target_model.clone());

        // Cache the GGUF path so the orchestrator can pass it to LLM worker
        // processes (each process needs to load its own model instance).
        // Resolve the orchestrator model (target_model) explicitly rather than
        // the workspace-active model, so the worker pool copies the small
        // orchestrator model — not a larger active model that's meant to load
        // lazily on demand.
        let gguf_path = {
            let ws = model.workspace.read();
            let model_id = ws.get_model(Some(&target_model)).map(|m| m.model_id).unwrap_or_default();
            ws.resolve_gguf_path(&model_id)
        };
        *self.model_gguf_path.write() = Some(gguf_path);

        // Rebuild the shared LLM worker pool so its processes load this model.
        self.reset_llm_executor();

        tracing::info!(
            target: LOG_TARGET,
            "Model loaded successfully: {target_model} (project: {project_name})"
        );

        // Auto-load decision model if configured
        if let Some(decision_model) = truthy(&settings().decision_model) {
            if self.decision_llm.read().is_none() {
                self.load_decision_model(decision_model);
            }
        }

        // The larger agent model is loaded lazily on first agent use (see
        // get_large_llm) rather than eagerly here.  Reset the per-model attempt
        // flag so the next agent request re-evaluates against the newly active
        // model.
        *self.large_llm.write() = None;
        *self.large_model_name.write() = None;
        self.large_model_attempted.store(false, Ordering::SeqCst);
        // Vision (multimodal) model is also loaded lazily on first OCR use.
        *self.vision_llm.write() = None;
        *self.vision_model_name.write() = None;
        self.vision_unavailable.store(false, Ordering::SeqCst);
        // Clear any capability-model GPU-fault evictions so a fresh model load
        // re-evaluates them against the newly active configuration.
        self.cap_llms.lock().retain(|(_, slot)| !slot.is_gpu_faulted());

        self.warmup_embeddings();
        self.warmup_inference();

        Ok(llm)
    }

    /// Load a secondary small model for decision nodes.
    ///
    /// Called automatically during [`ModelService::load_model`] when
    /// `SLP_DECISION_MODEL` is set.  The decision model shares the same
    /// inference semaphore as the main model.
    pub fn load_decision_model(&self, model_name: &str) {
        if self.decision_llm.read().is_some() && self.decision_model_name.read().as_deref() == Some(model_name) {
            return;
        }
        tracing::info!(target: LOG_TARGET, "[decision-model] Loading decision model: {model_name}");
        let loaded = ModelFactory::get_model(&ModelConfig::new(model_name)).and_then(|m| m.get_instance());
        match loaded {
            Ok(llm) => {
                *self.decision_llm.write() = Some(llm);
                *self.decision_model_name.write() = Some(model_name.to_string());
                tracing::info!(target: LOG_TARGET, "[decision-model] Decision model loaded: {model_name}");
            }
            Err(e) => {
                tracing::warn!(
                    target: LOG_TARGET,
                    "[decision-model] Failed to load '{model_name}', falling back to main model: {e}"
                );
                *self.decision_llm.write() = None;
                *self.decision_model_name.write() = None;
            }
        }
    }

    /// Return the large (agent) model, loading it lazily on first use.
    ///
    /// Attempted at most once per active model (`large_model_attempted`): if
    /// it loads it's cached; if it's absent or skipped for fit we return `None`
    /// and the caller falls back to the main/decision model.
    ///
    /// With exactly one model selected that model IS the agent model, even
    /// when it does not out-rank the bootstrap router — unless it is a
    /// vision-only selection, which cannot run agent work at all and so leaves
    /// detection to find a text model.  A pin that fails to load returns
    /// `None` rather than falling through to detection: the pinned model is
    /// the only candidate detection could find.
    pub fn get_large_llm(&self) -> Option<Arc<dyn BaseLLM>> {
        if let Some(hook) = self.hooks().get_large_llm {
            return hook();
        }
        if self.remote_device().is_some() {
            return Some(self.remote_llm(Some(ModelCapability::Reasoning.value())));
        }
        if self.pinned_model(None).is_some() {
            return self.pinned_llm(None);
        }
        let cached = self.large_llm.read().clone();
        if let Some(llm) = cached {
            let name = self.large_model_name.read().clone().unwrap_or_default();
            if self.still_selected(&name) {
                return Some(llm);
            }
            tracing::info!(
                target: LOG_TARGET,
                "[large-model] {name} is no longer selected for this project — releasing it"
            );
            // Clears the slot and the attempted latch, so detection below
            // picks again from what is selected now.
            self.unload_model_by_name_unlocked(&name);
        }
        if self.large_model_attempted.load(Ordering::SeqCst) {
            return None;
        }
        self.large_model_attempted.store(true, Ordering::SeqCst);
        self.detect_large_model();
        self.large_llm.read().clone()
    }

    /// Return `(main_gguf, mmproj_gguf)` found directly in `model_dir`.
    ///
    /// The mmproj / CLIP projector ships as a separate `*mmproj*.gguf` (or
    /// `*clip*.gguf`) alongside the model GGUF; it must not be mistaken for
    /// the main weights, so it is matched by name and excluded from the main
    /// pick.
    pub fn find_model_ggufs(model_dir: &str) -> (Option<String>, Option<String>) {
        let mut main_gguf = None;
        let mut mmproj_gguf = None;
        let dir = Path::new(model_dir);
        if let Some(entries) = dir.is_dir().then(|| std::fs::read_dir(dir).ok()).flatten() {
            let mut names: Vec<String> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            for f in names {
                if !f.ends_with(".gguf") {
                    continue;
                }
                let lower = f.to_lowercase();
                let full = dir.join(&f).to_string_lossy().into_owned();
                if lower.contains("mmproj") || lower.contains("clip") {
                    if mmproj_gguf.is_none() {
                        mmproj_gguf = Some(full);
                    }
                } else if main_gguf.is_none() {
                    main_gguf = Some(full);
                }
            }
        }
        (main_gguf, mmproj_gguf)
    }

    /// [`ModelService::find_model_ggufs`] through the test seam.
    fn find_ggufs(&self, model_dir: &str) -> (Option<String>, Option<String>) {
        match self.hooks().find_model_ggufs {
            Some(hook) => hook(model_dir),
            None => Self::find_model_ggufs(model_dir),
        }
    }

    /// Load a text model GGUF through the factory (or the test seam).
    fn load_llama(
        &self,
        gguf_path: &str,
        model_name: &str,
        cap: i64,
        thinking: Option<bool>,
    ) -> anyhow::Result<Arc<dyn BaseLLM>> {
        if let Some(hook) = self.hooks().load_llama_instance {
            return hook(gguf_path, model_name, cap, thinking);
        }
        let project_id = self.get_project_id().unwrap_or_default();
        let llm = LlmFactory::get_llama_cpp_instance(
            gguf_path,
            &project_id,
            None,
            LlamaInstanceOptions {
                max_n_ctx: (cap >= 0).then_some(cap as usize),
                thinking,
                model_name: Some(model_name.to_string()),
                ..Default::default()
            },
        )?;
        Ok(Arc::new(llm))
    }

    /// Enabled, downloaded models declaring vision: the local ones
    /// (`remote == false`) or the remote ones, in workspace order.
    fn vision_candidates(&self, remote: bool) -> Vec<WorkspaceModel> {
        let Some(workspace) = self.workspace_handle() else {
            return Vec::new();
        };
        let ws = workspace.read();
        let project = ws.current_project();
        ws.models
            .iter()
            .filter(|m| {
                m.is_remote() == remote
                    && ws.effective_model_enabled(&m.model_name, project)
                    && ws.effective_model_downloaded(&m.model_name)
                    && m.effective_capabilities().iter().any(|c| c == ModelCapability::Vision.value())
            })
            .cloned()
            .collect()
    }

    /// The best selected remote vision model (largest first), unless this
    /// thread resolves on device only.
    fn best_remote_vision_model(&self) -> Option<WorkspaceModel> {
        if crate::llms::cloud_llm::local_only() {
            return None;
        }
        Self::ranked_candidates(self.vision_candidates(true), &[ModelCapability::Vision.value()], true)
            .into_iter()
            .next()
    }

    /// The vision model for a question about the user's images.
    ///
    /// Quality first, local among similar models — the rule every capability
    /// step follows (see [`LOCAL_SIZE_MARGIN`]): a remote vision model well
    /// beyond the largest local one answers; otherwise the local path
    /// ([`Self::get_vision_llm`]) does, and still falls back to the remote
    /// model when nothing local can run.
    pub fn get_vision_answer_llm(&self, on_model_selected: Option<&dyn Fn(&str)>) -> Option<Arc<dyn BaseLLM>> {
        if !AppSettings::new().vision_enabled {
            return None;
        }
        if let Some(remote) = self.best_remote_vision_model() {
            let best_local = self.vision_candidates(false).iter().map(|m| m.rank().0).fold(0.0, f64::max);
            if remote.rank().0 > best_local * LOCAL_SIZE_MARGIN {
                tracing::info!(
                    target: LOG_TARGET,
                    "[vision] {} answers the image question — larger than any local vision model",
                    remote.model_name
                );
                if let Some(cb) = on_model_selected {
                    cb(&remote.model_name);
                }
                return Some(self.cloud_llm(&remote));
            }
        }
        self.get_vision_llm(on_model_selected)
    }

    /// Return a loaded *multimodal* vision LLM, or `None` when unavailable.
    /// `on_model_selected(model_name)` fires right before the chosen model loads.
    ///
    /// Local vision models are considered smallest-first, but a model is
    /// only usable for OCR when its mmproj / CLIP projector GGUF is present,
    /// the llama.cpp build has a chat handler for its architecture, and it
    /// fits in available memory.  Unrunnable candidates are skipped.  Local
    /// first: OCR transcribes every page of every indexed document, so it
    /// stays on device and a remote vision model reads the pages only when
    /// no local one can run.
    pub fn get_vision_llm(&self, on_model_selected: Option<&dyn Fn(&str)>) -> Option<Arc<dyn BaseLLM>> {
        let app = AppSettings::new();
        if !app.vision_enabled {
            return None;
        }
        let cached = self.vision_llm.read().clone();
        if let Some(llm) = cached {
            let name = self.vision_model_name.read().clone().filter(|n| !n.is_empty());
            if name.as_deref().is_none_or(|n| self.still_selected(n)) {
                if let (Some(cb), Some(name)) = (on_model_selected, name) {
                    cb(&name);
                }
                return Some(llm);
            }
            let name = name.unwrap_or_default();
            tracing::info!(
                target: LOG_TARGET,
                "[vision] {name} is no longer selected for this project — releasing it"
            );
            self.unload_model_by_name_unlocked(&name);
        }
        let workspace = self.workspace_handle()?;

        let candidates = self.vision_candidates(false);
        // Smallest first: lowest latency OCR; same-size ties broken by cap
        // overlap then version.
        let candidates = Self::ranked_candidates(candidates, &[ModelCapability::Vision.value()], false);

        // OCR is a latency-tolerant batch task, so give the vision model a
        // small dedicated context and waive the agent-model reserve.
        let cap = app.vision_n_ctx;
        // Track whether any candidate was rejected for a *transient* reason
        // (not enough free VRAM right now, or a load error) versus a
        // *permanent* one (no mmproj, no handler).  A transient failure must
        // NOT latch `vision_unavailable` — free VRAM recovers once the
        // indexing worker copies are released, so the next document should
        // retry rather than be stuck on text-only for the process lifetime.
        let mut transient_failure = false;
        for ws_model in candidates {
            let model_dir = workspace.read().model_dir_for(&ws_model.model_id);
            let (gguf_path, mmproj_path) = self.find_ggufs(&model_dir);
            let (Some(gguf_path), Some(mmproj_path)) = (gguf_path.clone(), mmproj_path) else {
                tracing::info!(
                    target: LOG_TARGET,
                    "[vision] {} skipped — missing {}",
                    ws_model.model_name,
                    if gguf_path.is_none() { "weights GGUF" } else { "mmproj/CLIP GGUF" }
                );
                continue;
            };
            // The handler the model's spec declares alongside its VISION
            // capability (falls back to GGUF-arch sniffing when absent).
            let vision_handler = MODEL_SPECS.get(&ws_model.model_name).and_then(|spec| spec.vision_chat_handler.clone());
            if !LlmFactory::vision_handler_supported(&gguf_path, vision_handler.as_deref()) {
                tracing::warn!(
                    target: LOG_TARGET,
                    "[vision] {} skipped — installed llama-cpp has no vision handler (declared={}, arch={})",
                    ws_model.model_name,
                    vision_handler.as_deref().unwrap_or("None"),
                    LlmFactory::read_gguf_arch(&gguf_path).as_deref().unwrap_or("None")
                );
                continue;
            }
            // Capability models stay resident across responses, so on a tight
            // device they can hold the memory the vision copy needs for good —
            // waiting for "the next document" would never free it.  Release
            // them and re-check, as a text step does; they reload on the next
            // step that asks for them.  The vision copy is its own instance
            // (with the mmproj), so a resident text copy of the same model
            // cannot stand in for it.
            if !self.large_model_fits(&gguf_path, cap, &ws_model.model_name, Some(0.0)) {
                if !(self.evict_capability_models_hooked(None)
                    && self.large_model_fits(&gguf_path, cap, &ws_model.model_name, Some(0.0)))
                {
                    tracing::warn!(
                        target: LOG_TARGET,
                        "[vision] {} skipped — won't fit in available memory right now (will retry on the next document)",
                        ws_model.model_name
                    );
                    transient_failure = true;
                    continue;
                }
                tracing::info!(
                    target: LOG_TARGET,
                    "[vision] Freed capability models to fit {}",
                    ws_model.model_name
                );
            }
            let clip_on_cpu = self.clip_on_cpu(&gguf_path, cap);
            // Announce the chosen model before the (heavy) load so callers can
            // surface it immediately.
            if let Some(cb) = on_model_selected {
                cb(&ws_model.model_name);
            }
            let loaded = LlmFactory::get_llama_cpp_instance(
                &gguf_path,
                &self.get_project_id().unwrap_or_default(),
                None,
                LlamaInstanceOptions {
                    max_n_ctx: (cap >= 0).then_some(cap as usize),
                    mmproj_path: Some(mmproj_path),
                    clip_on_cpu,
                    vision_handler: vision_handler.clone(),
                    model_name: Some(ws_model.model_name.clone()),
                    // Reasoning is off for the vision copy whatever the model
                    // declares: transcribing a page and describing an image
                    // are perception, not deliberation, and the small
                    // vision_n_ctx has to hold several thousand image tokens.
                    thinking: Some(false),
                    ..Default::default()
                },
            );
            match loaded {
                Ok(llm) => {
                    let llm: Arc<dyn BaseLLM> = Arc::new(llm);
                    *self.vision_llm.write() = Some(llm.clone());
                    *self.vision_model_name.write() = Some(ws_model.model_name.clone());
                    tracing::info!(
                        target: LOG_TARGET,
                        "[vision] Loaded {} (+mmproj) for image OCR",
                        ws_model.model_name
                    );
                    return Some(llm);
                }
                Err(e) => {
                    tracing::warn!(target: LOG_TARGET, "[vision] Failed to load {}: {e}", ws_model.model_name);
                    transient_failure = true;
                }
            }
        }

        if let Some(remote) = self.best_remote_vision_model() {
            tracing::info!(target: LOG_TARGET, "[vision] No local vision model can run — using {}", remote.model_name);
            if let Some(cb) = on_model_selected {
                cb(&remote.model_name);
            }
            return Some(self.cloud_llm(&remote));
        }

        if transient_failure {
            tracing::warn!(
                target: LOG_TARGET,
                "[vision] No vision model could be loaded right now — text-only for this document; will retry later"
            );
        } else {
            tracing::warn!(target: LOG_TARGET, "[vision] No runnable vision model available — image OCR disabled");
        }
        None
    }

    /// Sort candidates by `(size, cap-overlap, version)`; `descending` walks
    /// largest → smallest.
    ///
    /// `ref_caps` is the reference capability set used to score overlap so
    /// that same-size models with broader coverage rank above narrower ones.
    pub fn ranked_candidates<M: RankedModel, S: AsRef<str>>(
        candidates: Vec<M>,
        ref_caps: &[S],
        descending: bool,
    ) -> Vec<M> {
        let mut keyed: Vec<((f64, usize, i64), M)> = candidates
            .into_iter()
            .map(|m| {
                let (size, version) = m.model_rank();
                ((size, cap_overlap(ref_caps, &m.model_capabilities()), version), m)
            })
            .collect();
        let cmp = |a: &(f64, usize, i64), b: &(f64, usize, i64)| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
                .then(a.2.cmp(&b.2))
        };
        if descending {
            keyed.sort_by(|a, b| cmp(&b.0, &a.0));
        } else {
            keyed.sort_by(|a, b| cmp(&a.0, &b.0));
        }
        keyed.into_iter().map(|(_, m)| m).collect()
    }

    /// The models the user has chosen — enabled *and* downloaded, excluding
    /// the bootstrap/orchestrator model.
    ///
    /// The bootstrap model is always enabled, filtered out of the model list
    /// the user picks from and cannot be configured per project, so it is
    /// infrastructure rather than a selection — counting it would mean the
    /// user could never get down to a single selected model.  Resolution is
    /// project-scoped.
    pub fn selected_models(&self) -> Vec<WorkspaceModel> {
        let Some(workspace) = self.workspace_handle() else {
            return Vec::new();
        };
        let ws = workspace.read();
        let project = ws.current_project();
        let bootstrap = canonical_model_name(&api_settings().default_model);
        ws.models
            .iter()
            .filter(|m| {
                canonical_model_name(&m.model_name) != bootstrap
                    && ws.effective_model_enabled(&m.model_name, project)
                    && ws.effective_model_downloaded(&m.model_name)
            })
            .cloned()
            .collect()
    }

    /// True while `model_name` is enabled and downloaded for the bound
    /// project; true for an empty name or when there is no workspace.
    ///
    /// The large and vision slots are process-wide, but selection is per
    /// project and is written by that project's child.  Deselecting there
    /// cannot release what this process (the model host) holds, so a cached
    /// model is checked against the selection each time it is reused.
    fn still_selected(&self, model_name: &str) -> bool {
        if model_name.is_empty() {
            return true;
        }
        let Some(workspace) = self.workspace_handle() else {
            return true;
        };
        let ws = workspace.read();
        ws.effective_model_enabled(model_name, ws.current_project()) && ws.effective_model_downloaded(model_name)
    }

    /// The sole selected model, or `None` when it cannot serve the step.
    ///
    /// With two or more models selected, capability routing chooses per
    /// step.  With exactly one the choice is the user's, so that model runs
    /// everything — except a selection that cannot hold a text conversation
    /// at all (no `general`), which is pinned only to the steps whose
    /// capability it declares.
    pub fn pinned_model(&self, capability: Option<&str>) -> Option<WorkspaceModel> {
        if let Some(hook) = self.hooks().pinned_model {
            return hook(capability);
        }
        let mut selected = self.selected_models();
        if selected.len() != 1 {
            return None;
        }
        let ws_model = selected.remove(0);
        if ws_model.is_remote() && crate::llms::cloud_llm::local_only() {
            // Orchestration stays on device; the remote pin is for the answer.
            return None;
        }

        let caps: HashSet<String> = ws_model
            .effective_capabilities()
            .iter()
            .map(|c| Self::cap_name(c).to_lowercase())
            .collect();
        if caps.contains(ModelCapability::General.value()) {
            return Some(ws_model);
        }

        let cap = capability.map(|c| Self::cap_name(c).to_lowercase()).unwrap_or_default();
        if !cap.is_empty() && caps.contains(&cap) {
            return Some(ws_model);
        }

        let mut sorted: Vec<String> = caps.into_iter().collect();
        sorted.sort();
        tracing::info!(
            target: LOG_TARGET,
            "[models] Sole selected model {} declares {} but not 'general' — {} runs on the default model",
            ws_model.model_name,
            if sorted.is_empty() { "no capability".to_string() } else { py_list_repr(&sorted) },
            if cap.is_empty() { "this step".to_string() } else { format!("capability '{cap}'") }
        );
        None
    }

    /// Loaded instance of the pinned model, or `None`.
    ///
    /// `None` means the user has not pinned the selection to one model, that
    /// the pin does not cover `capability`, or that the pinned model could not
    /// be loaded — callers then take their existing fallback path.
    pub fn pinned_llm(&self, capability: Option<&str>) -> Option<Arc<dyn BaseLLM>> {
        if let Some(hook) = self.hooks().pinned_llm {
            return hook(capability);
        }
        let ws_model = self.pinned_model(capability)?;
        if ws_model.is_remote() {
            // Answered over HTTP from whichever process asks; there is
            // nothing for the model host to hold.
            return Some(self.cloud_llm(&ws_model));
        }
        if self.remote_device().is_some() {
            // The pin says *which* model; putting it in memory is the host's
            // business.  Loading the GGUF here would be a second copy of the
            // weights in the same device — and a fit check made by a process
            // that cannot see the device.
            return Some(self.remote_llm(capability));
        }
        self.load_workspace_llm(&ws_model, "pinned", None)
    }

    /// `(model_name, cause, detail)` for the last capability model that could
    /// not be loaded, or `None` when the last resolution succeeded.
    ///
    /// Lets the caller explain a fallback rather than silently answering on a
    /// smaller model.  Advisory only — a race between concurrent fan-out
    /// threads costs at most a wrong model name in one status line.
    pub fn last_capability_skip(&self) -> Option<CapabilitySkipReason> {
        if let Some(hook) = self.hooks().last_capability_skip {
            return hook();
        }
        self.capability_skip.lock().clone()
    }

    /// Record (or clear) the capability skip reason.
    fn set_capability_skip(&self, model_name: &str, cause: &str, detail: Option<String>) {
        *self.capability_skip.lock() = Some(CapabilitySkipReason {
            model_name: model_name.to_string(),
            cause: cause.to_string(),
            detail,
        });
    }

    /// `"needs ~4.9GB, 4.4GB free"` from the last failed fit check.
    ///
    /// The numbers are the actionable part: they say how much to free.
    pub fn fit_shortfall_detail(&self) -> Option<String> {
        let (needed_gb, free_gb) = (*self.last_fit_shortfall.lock())?;
        Some(format!("needs ~{needed_gb:.1}GB, {free_gb:.1}GB free"))
    }

    /// `_evict_capability_models(keep)` through the test seam.
    fn evict_capability_models_hooked(&self, keep_model_name: Option<&str>) -> bool {
        match self.hooks().evict_capability_models {
            Some(hook) => hook(keep_model_name),
            None => self.evict_capability_models(keep_model_name),
        }
    }

    /// Return a resident (or freshly loaded) instance of one workspace model,
    /// or `None` when it has no GGUF, faulted, or won't fit.
    ///
    /// Shared by capability routing and the pinned-model path: consults the
    /// per-model cache, honours a GPU-fault eviction, applies the fit guard
    /// (evicting prior-step models to make room) and loads.  `reason` names
    /// the caller in the logs.  `reserve_gb` overrides the headroom the fit
    /// guard keeps for the main model.
    pub fn load_workspace_llm(
        &self,
        ws_model: &WorkspaceModel,
        reason: &str,
        reserve_gb: Option<f64>,
    ) -> Option<Arc<dyn BaseLLM>> {
        self.load_workspace_llm_with(ws_model, reason, reserve_gb, true)
    }

    /// [`Self::load_workspace_llm`], with `evict` saying whether a model
    /// that does not fit may release prior steps' capability models (and the
    /// on-demand large model) and retry.
    ///
    /// Without eviction the call only returns a model that is already
    /// resident or fits as things stand, so it never costs a model that is
    /// already serving.
    pub fn load_workspace_llm_with(
        &self,
        ws_model: &WorkspaceModel,
        reason: &str,
        reserve_gb: Option<f64>,
        evict: bool,
    ) -> Option<Arc<dyn BaseLLM>> {
        if let Some(hook) = self.hooks().load_workspace_llm {
            return hook(ws_model, reason, reserve_gb, evict);
        }
        if ws_model.is_remote() {
            // No weights, so no cache slot, fit guard or eviction.
            *self.capability_skip.lock() = None;
            return Some(self.cloud_llm(ws_model));
        }
        let Some(workspace) = self.workspace_handle() else {
            tracing::warn!(
                target: LOG_TARGET,
                "[capability-model] Failed to load {} ({reason}): no workspace loaded",
                ws_model.model_name
            );
            self.set_capability_skip(&ws_model.model_name, "error", None);
            return None;
        };
        match self.cap_llm(&ws_model.model_name) {
            Some(CachedLlm::GpuFaulted) => {
                // Evicted after a runtime GPU OOM — don't reload it.  That is
                // a memory failure too, just one discovered during generation.
                self.set_capability_skip(&ws_model.model_name, "memory", None);
                return None;
            }
            Some(CachedLlm::Loaded(llm)) => {
                // Resident and already passed the fit guard when loaded.
                *self.capability_skip.lock() = None;
                return Some(llm);
            }
            None => {}
        }

        // A step routed to the orchestrator model (or the on-demand large one)
        // must reuse the resident instance rather than build a second
        // llama.cpp context over the same GGUF.  Duplicating the weights and
        // KV cache of a model already in memory is what trips a shared-memory
        // GPU over its allocation limit:
        //
        //     ggml_vulkan: Failed to allocate pinned memory
        //     (Requested buffer size exceeds device buffer size limit)
        //
        // Canonicalize both sides so a legacy workspace name still resolves to
        // the resident model.
        let canonical = canonical_model_name(&ws_model.model_name);
        let resident_orchestrator = self.llm.read().clone();
        if let Some(resident) = resident_orchestrator {
            let name = self.model_name.read().as_deref().map(canonical_model_name);
            if name.as_deref() == Some(canonical.as_str()) {
                *self.capability_skip.lock() = None;
                return Some(resident);
            }
        }
        let resident_large = self.large_llm.read().clone();
        if let Some(resident) = resident_large {
            let name = self.large_model_name.read().as_deref().map(canonical_model_name);
            if name.as_deref() == Some(canonical.as_str()) {
                *self.capability_skip.lock() = None;
                return Some(resident);
            }
        }

        let cap = AppSettings::new().llm_large_n_ctx_cap;
        let model_dir = workspace.read().model_dir_for(&ws_model.model_id);
        // find_model_ggufs excludes the mmproj/CLIP GGUF from the main pick.
        // This path is text-only capability routing; image OCR goes through
        // get_vision_llm(), which loads the multimodal handler.
        let (gguf_path, _mmproj) = self.find_ggufs(&model_dir);
        let Some(gguf_path) = gguf_path else {
            tracing::warn!(
                target: LOG_TARGET,
                "[capability-model] No GGUF found for {} at {model_dir} ({reason})",
                ws_model.model_name
            );
            self.set_capability_skip(&ws_model.model_name, "missing", None);
            return None;
        };

        // Fit guard: skip a model that won't fit alongside the main model.
        // First eagerly release prior steps' capability models (and the
        // on-demand large model) and retry — that frees enough VRAM to run
        // this model on a memory-tight device.
        if !self.large_model_fits(&gguf_path, cap, &ws_model.model_name, reserve_gb) {
            if !(evict
                && self.evict_capability_models_hooked(Some(&ws_model.model_name))
                && self.large_model_fits(&gguf_path, cap, &ws_model.model_name, reserve_gb))
            {
                tracing::warn!(
                    target: LOG_TARGET,
                    "[capability-model] {} doesn't fit alongside the main model at {} ({reason})",
                    ws_model.model_name,
                    n_ctx_desc(cap)
                );
                let detail = self.fit_shortfall_detail();
                self.set_capability_skip(&ws_model.model_name, "memory", detail);
                return None;
            }
            tracing::info!(
                target: LOG_TARGET,
                "[capability-model] Freed prior-step models to fit {} ({reason})",
                ws_model.model_name
            );
        }
        let thinking = self.thinking_for(&ws_model.model_name);
        match self.load_llama(&gguf_path, &ws_model.model_name, cap, thinking) {
            Ok(llm) => {
                self.set_cap_llm(&ws_model.model_name, Some(CachedLlm::Loaded(llm.clone())));
                tracing::info!(
                    target: LOG_TARGET,
                    "[capability-model] Loaded {} ({reason})",
                    ws_model.model_name
                );
                *self.capability_skip.lock() = None;
                Some(llm)
            }
            Err(e) => {
                tracing::warn!(
                    target: LOG_TARGET,
                    "[capability-model] Failed to load {} ({reason}): {e}",
                    ws_model.model_name
                );
                self.set_capability_skip(&ws_model.model_name, "error", None);
                None
            }
        }
    }

    /// A [`CloudLLM`] for a remote workspace model.  Built per call: it holds
    /// no weights, and a fresh instance keeps one request's cancellation
    /// event off another's.
    pub fn cloud_llm(&self, ws_model: &WorkspaceModel) -> Arc<dyn BaseLLM> {
        let project_id = self
            .workspace_handle()
            .and_then(|ws| ws.read().current_project().map(|p| p.id.clone()))
            .unwrap_or_default();
        let url = ws_model.url().unwrap_or_else(|| AppSettings::new().slp_api_base);
        Arc::new(
            CloudLLM::new(&project_id, &ws_model.model_name, &url, ws_model.context_length())
                .with_reasoning_budget(ws_model.reasoning_budget_tokens()),
        )
    }

    /// Whether `llm` answers over HTTP rather than from local weights.
    pub fn is_cloud_llm(llm: &dyn BaseLLM) -> bool {
        llm.as_any().is::<CloudLLM>()
    }

    /// Whether `model_name` should reason, per the picker's setting.
    ///
    /// Reasoning is a property of the model, not of a request: it is stored
    /// per model (and per project) like `enabled` and set from the model
    /// picker.  `None` means the model does not support it.
    pub fn thinking_for(&self, model_name: &str) -> Option<bool> {
        match self.ensure_workspace(None) {
            Ok(workspace) => {
                let ws = workspace.read();
                ws.effective_model_thinking(model_name, ws.current_project())
            }
            Err(_) => LlmFactory::thinking_for(Some(model_name), None),
        }
    }

    /// Push a just-changed reasoning setting onto a resident model.  Returns
    /// true when a resident instance was updated.
    ///
    /// The setting otherwise only takes effect on the next load, which for an
    /// always-resident model is never.  Re-flagging is a single write that the
    /// bound chat handler reads on its next call — no reload, no VRAM churn.
    pub fn apply_thinking_setting(&self, model_name: &str) -> bool {
        let Some(value) = self.thinking_for(model_name) else {
            return false;
        };
        let mut updated = false;
        for (name, llm) in self.loaded_thinking_models() {
            if name != model_name {
                continue;
            }
            match llm.llama_model() {
                Some(model) => {
                    model.state().set_thinking(Some(value));
                    updated = true;
                }
                // Never fail a settings change over a hint.
                None => {
                    tracing::debug!(
                        target: LOG_TARGET,
                        "[thinking] could not re-flag {name}: no local llama.cpp model"
                    );
                }
            }
        }
        if updated {
            tracing::info!(
                target: LOG_TARGET,
                "[thinking] {model_name} → enable_thinking={}",
                if value { "True" } else { "False" }
            );
        }
        updated
    }

    /// `(model_name, llm)` for every resident model that can reason.
    fn loaded_thinking_models(&self) -> Vec<(String, Arc<dyn BaseLLM>)> {
        let mut pairs: Vec<(String, Arc<dyn BaseLLM>)> = Vec::new();
        let main_name = self.model_name.read().clone().filter(|n| !n.is_empty());
        if let (Some(name), Some(llm)) = (main_name, self.llm.read().clone()) {
            pairs.push((name, llm));
        }
        let large_name = self.large_model_name.read().clone().filter(|n| !n.is_empty());
        if let (Some(name), Some(llm)) = (large_name, self.large_llm.read().clone()) {
            pairs.push((name, llm));
        }
        for (name, slot) in self.cap_llms.lock().iter() {
            if let CachedLlm::Loaded(llm) = slot {
                pairs.push((name.clone(), llm.clone()));
            }
        }
        pairs
            .into_iter()
            .filter(|(name, _)| LlmFactory::supports_thinking(Some(name)))
            .collect()
    }

    /// The capability's value, for logs and messages.
    pub fn cap_name(capability: &str) -> String {
        capability.to_string()
    }

    /// Whether the model that would answer for `capability` can reason.
    ///
    /// Asked by a project agent through the model host, which is why it lives
    /// here rather than being read off an LLM handle.
    pub fn supports_thinking(&self, capability: Option<&str>) -> bool {
        self.llm_for_response(capability)
            .map(|llm| llm.supports_thinking())
            .unwrap_or(false)
    }

    /// `workspace.ranked_models_for(capability, current_project)` through the
    /// test seam, without remote models inside a local-only scope.
    fn ranked_models_for(&self, capability: &str) -> Vec<WorkspaceModel> {
        let mut ranked = match self.hooks().ranked_models_for {
            Some(hook) => hook(capability),
            None => {
                let Some(workspace) = self.workspace_handle() else {
                    return Vec::new();
                };
                let ws = workspace.read();
                ws.ranked_models_for(capability, ws.current_project())
            }
        };
        if crate::llms::cloud_llm::local_only() {
            ranked.retain(|m| !m.is_remote());
        }
        ranked
    }

    /// True when some selected model declares `capability` (loads nothing).
    ///
    /// The orchestrator uses it to decide whether to announce a capability
    /// model load at all: the "Loading model for X…" event fires *before* the
    /// slow load, so without this check it promised a load on workspaces
    /// where no selected model provides X.
    pub fn has_capability_model(&self, capability: &str) -> bool {
        if let Some(hook) = self.hooks().has_capability_model {
            return hook(capability);
        }
        if self.workspace_handle().is_none() {
            return false;
        }
        // A pinned model serves every capability it can — declared or not for
        // a general-capable selection, only its own for a specialist one.
        if self.pinned_model(Some(capability)).is_some() {
            return true;
        }
        !self.ranked_models_for(capability).is_empty()
    }

    /// Return a loaded LLM matching the requested capability.
    ///
    /// When the user has selected exactly one model, that model is returned
    /// for every capability it can serve.  Otherwise walks the capability's
    /// models largest-first and returns the first that is already resident
    /// or fits in memory — evicting prior-step capability models to make room
    /// when needed.  Returns `None` only when no capable model fits.
    pub fn get_llm_for_capability(&self, capability: &str) -> Option<Arc<dyn BaseLLM>> {
        if let Some(hook) = self.hooks().get_llm_for_capability {
            return hook(capability);
        }
        if self.remote_device().is_some() {
            // The host owns model selection: it can see the device, and a
            // child cannot.
            return Some(self.remote_llm(Some(capability)));
        }

        if self.workspace_handle().is_none() {
            tracing::warn!(
                target: LOG_TARGET,
                "[capability-model] No workspace loaded — cannot resolve capability '{capability}'"
            );
            return None;
        }

        // Reasoning is the tool-resolution step's model, the way the vision
        // model is the image step's: it loads on the same terms as vision —
        // without the headroom the fit guard keeps for the main model.
        let reserve_gb = if Self::cap_name(capability) == ModelCapability::Thinking.value() {
            Some(0.0)
        } else {
            None
        };

        // One selected model runs every step, whatever capability the step
        // asked for (a selection without 'general' covers only its own).
        if let Some(pinned) = self.pinned_model(Some(capability)) {
            let llm = self.load_workspace_llm(
                &pinned,
                &format!("pinned, capability '{}'", Self::cap_name(capability)),
                reserve_gb,
            );
            if llm.is_none() {
                tracing::warn!(
                    target: LOG_TARGET,
                    "[capability-model] Sole selected model {} unavailable for '{capability}' — falling back to the default model",
                    pinned.model_name
                );
            }
            return llm;
        }

        // Candidate selection is project-scoped (per-project model settings
        // override the workspace defaults).  Walked largest-first: when two
        // selected models both declare the capability, the bigger one does
        // the step, and a smaller one runs it only when the bigger cannot be
        // loaded.  Size ties go to the local model, then broader coverage,
        // then the newer version.
        //
        // Quality first, local among similar models: a bigger model (by
        // active size) does the step whether it runs here or in the cloud,
        // but a local model within LOCAL_SIZE_MARGIN of a remote one wins —
        // a 3B on device over the same 3B remote, but a 23B-active cloud
        // model over a 2B local one.  Every candidate declares the step's
        // capability, so size is what separates them.  A local model that
        // does not fit is passed over for the next one down.  The bootstrap
        // model goes last, since any other selected model does the step
        // better.
        let mut ranked = Self::ranked_candidates(self.ranked_models_for(capability), &[capability], true);
        let bootstrap = canonical_model_name(&api_settings().default_model);
        let quality = |m: &WorkspaceModel| {
            let size = m.rank().0;
            if m.is_remote() { size } else { size * LOCAL_SIZE_MARGIN }
        };
        ranked.sort_by(|a, b| {
            let is_bootstrap = |m: &WorkspaceModel| canonical_model_name(&m.model_name) == bootstrap;
            is_bootstrap(a)
                .cmp(&is_bootstrap(b))
                .then(quality(b).partial_cmp(&quality(a)).unwrap_or(std::cmp::Ordering::Equal))
                .then(a.is_remote().cmp(&b.is_remote()))
        });
        if ranked.is_empty() {
            tracing::warn!(
                target: LOG_TARGET,
                "[capability-model] No enabled+downloaded model declares capability '{capability}'"
            );
            return None;
        }
        tracing::info!(
            target: LOG_TARGET,
            "[capability-model] '{capability}' candidates in order: {} (project {}, local_only={})",
            ranked.iter().map(|m| m.model_name.as_str()).collect::<Vec<_>>().join(", "),
            self.get_project_id().unwrap_or_default(),
            crate::llms::cloud_llm::local_only()
        );

        // Two passes.  The first takes the largest model that is already
        // resident or fits as things stand, without evicting anything: a
        // bigger model that does not fit must not cost the smaller capable
        // model already serving — evicting it and then finding the bigger one
        // still does not fit leaves the step on the tiny default.  Only when
        // nothing is resident or fits does the second pass evict prior-step
        // models to make room.  A remote model always resolves, so with one
        // selected the first pass ends on it and nothing is evicted.
        let reason = format!("capability '{}'", Self::cap_name(capability));
        for evict in [false, true] {
            for ws_model in &ranked {
                // Move down to the next capable model when this one has no
                // GGUF, faulted, or doesn't fit — rather than the tiny default.
                let llm = self.load_workspace_llm_with(ws_model, &reason, reserve_gb, evict);
                if llm.is_some() {
                    return llm;
                }
            }
        }

        tracing::info!(
            target: LOG_TARGET,
            "[capability-model] No model declaring capability '{capability}' fits in memory — falling back to the default/large model"
        );
        None
    }

    /// Return `(free_gb, total_gb)` for the active accelerator, or `None`
    /// (only Apple Silicon unified memory reports a fit budget).
    ///
    /// VRAM is not probed: for a discrete GPU the Vulkan backend manages
    /// device memory, so llama.cpp handles offload/fit there.
    pub fn device_free_total_gb(&self) -> Option<(f64, f64)> {
        if let Some(hook) = self.hooks().device_free_total_gb {
            return hook();
        }
        if crate::utils::device_utils::is_mps() {
            return Some(crate::utils::device_utils::available_memory_gb());
        }
        None
    }

    /// True if a large model's capped footprint fits in free memory.
    ///
    /// Estimates the model's footprint (weights + KV cache @ `cap` +
    /// per-process overhead) and compares it against free device memory after
    /// a reserve (`reserve_gb`, defaulting to `llm_large_fit_reserve_gb`).
    /// Always true on CPU or when the device can't be queried.
    pub fn large_model_fits(&self, gguf_path: &str, cap: i64, label: &str, reserve_gb: Option<f64>) -> bool {
        if let Some(hook) = self.hooks().large_model_fits {
            return hook(gguf_path, cap, label, reserve_gb);
        }
        let Some((free_gb, total_gb)) = self.device_free_total_gb() else {
            return true;
        };
        let model_gb = match std::fs::metadata(gguf_path) {
            Ok(meta) => meta.len() as f64 / 1024f64.powi(3),
            Err(_) => return true,
        };
        let reserve_gb = reserve_gb.unwrap_or_else(|| AppSettings::new().llm_large_fit_reserve_gb);
        // 0 = no cap — the factory auto-sizes the model's context from live
        // available memory, so the KV cache fits by construction.  Only the
        // bare weights + overhead footprint can veto here; estimate no KV
        // cache.  (Estimating it at the device bound instead over-counts and
        // rejects every capable model on a tight box.)
        let est_ctx = if cap > 0 { cap } else { 0 };
        let kv_gb = 0.5 * (est_ctx as f64 / 4096.0);
        let footprint_gb = model_gb + kv_gb + 0.7;
        if footprint_gb + reserve_gb > free_gb {
            // Only suggest the cap when there is a cap to lower.  Uncapped,
            // the KV cache is already estimated at zero and the window
            // auto-sizes, so lowering the setting cannot move the figure.
            let remedy = if cap > 0 {
                "Lower SLP_LLM_LARGE_N_CTX_CAP or free memory to enable it."
            } else {
                "Free memory to enable it — SLP_LLM_LARGE_N_CTX_CAP is unset, so the context window already auto-sizes and this is the weights alone."
            };
            let name = if label.is_empty() {
                Path::new(gguf_path)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            } else {
                label.to_string()
            };
            tracing::warn!(
                target: LOG_TARGET,
                "[large-model] {name} needs ~{footprint_gb:.1}GB (+{reserve_gb:.1} reserve) but only {free_gb:.1}GB free of {total_gb:.1}GB — falling back to the default model. {remedy}"
            );
            // Carry the figures out for the progress event.  The reserve is
            // folded into the requirement because that is the threshold that
            // actually gated the load.
            *self.last_fit_shortfall.lock() = Some((footprint_gb + reserve_gb, free_gb));
            return false;
        }
        *self.last_fit_shortfall.lock() = None;
        true
    }

    /// Decide whether the vision CLIP/mmproj encoder must run on CPU.
    ///
    /// GPU is preferred (faster), but the projector + per-image activations
    /// OOM the native clip code — an uncatchable crash — when VRAM is too
    /// tight.  `vision_clip_device` forces the choice ("gpu"/"cpu"); in
    /// "auto" the projector stays on GPU only when free VRAM comfortably
    /// exceeds the weights + KV + projector/activation headroom.
    pub fn clip_on_cpu(&self, gguf_path: &str, cap: i64) -> bool {
        let device = AppSettings::new().vision_clip_device;
        let pref = if device.is_empty() { "auto".to_string() } else { device.to_lowercase() };
        if pref == "gpu" {
            return false;
        }
        if pref == "cpu" || self.vision_clip_cpu.load(Ordering::SeqCst) {
            return true;
        }
        let Some((free_gb, _)) = self.device_free_total_gb() else {
            // VRAM unknown.  Follow the build: when llama.cpp supports GPU
            // offload the projector stays on GPU with the model (Vulkan
            // included); a CPU-only build runs it on CPU.
            return !crate::llms::llama::llama_supports_gpu_offload();
        };
        let model_gb = match std::fs::metadata(gguf_path) {
            Ok(meta) => meta.len() as f64 / 1024f64.powi(3),
            Err(_) => return true,
        };
        // Projector weights (~0.8GB) + per-image encode activation headroom.
        // flash_attn is CUDA-only; without it the clip encode of a large image
        // allocates a ~6GB compute buffer instead of ~3GB, so non-CUDA
        // backends need correspondingly more free VRAM.
        let act_gb = if crate::llms::llm_factory::llama_cpp_has_cuda_backend() { 1.8 } else { 6.0 };
        let need_gb = model_gb + 0.5 * (cap as f64 / 4096.0) + act_gb;
        if free_gb < need_gb {
            tracing::info!(
                target: LOG_TARGET,
                "[vision] CLIP on CPU — only {free_gb:.1}GB free, GPU clip needs ~{need_gb:.1}GB (set SLP_VISION_CLIP_DEVICE=gpu to force, or free VRAM)"
            );
            return true;
        }
        false
    }

    /// Load the largest downloaded model for agent/skill tasks, if it fits.
    ///
    /// Skips the load when the model's capped footprint won't fit in free
    /// VRAM (after a reserve for the main model + worker pool), so loading the
    /// large model never OOM-crashes the process or starves the workers.  The
    /// active (smaller) model stays as the routing LLM.
    pub fn detect_large_model(&self) {
        if self.large_llm.read().is_some() {
            return;
        }
        if let Err(e) = self.detect_large_model_inner() {
            tracing::warn!(target: LOG_TARGET, "[large-model] Failed to load larger model: {e}");
            *self.large_llm.write() = None;
            *self.large_model_name.write() = None;
        }
    }

    fn detect_large_model_inner(&self) -> anyhow::Result<()> {
        let workspace = self
            .workspace_handle()
            .ok_or_else(|| ModelServiceError::runtime("Model not loaded"))?;
        let router_name = self.model_name.read().clone();
        // Pick the agent/large model from any general-capable (text) model —
        // filtering by these excludes vision-only models without requiring a
        // specific REASONING tag.  Walk largest-first; when the biggest
        // doesn't fit, fall to the next lower tier rather than the tiny router.
        let text_caps = [
            ModelCapability::General.value(),
            ModelCapability::Reasoning.value(),
            ModelCapability::Dp.value(),
        ];
        let (orchestrator_size, orchestrator_rank, agent_models, models_dir) = {
            let ws = workspace.read();
            let project = ws.current_project();
            // Compare against the orchestrator model (the loaded small model),
            // not the workspace-active model.
            let orchestrator = ws.get_model(router_name.as_deref());
            let size = orchestrator.as_ref().map(|m| m.param_size_b()).unwrap_or(0.0);
            let rank = orchestrator.as_ref().map(|m| m.rank()).unwrap_or((0.0, 0));
            let agents: Vec<WorkspaceModel> = ws
                .models
                .iter()
                .filter(|m| {
                    ws.effective_model_enabled(&m.model_name, project)
                        && ws.effective_model_downloaded(&m.model_name)
                        && Some(m.model_name.as_str()) != router_name.as_deref()
                        && !m.is_remote()
                        && m.effective_capabilities().iter().any(|c| text_caps.contains(&c.as_str()))
                })
                .cloned()
                .collect();
            (size, rank, agents, ws.models_dir.clone())
        };
        if agent_models.is_empty() {
            tracing::info!(
                target: LOG_TARGET,
                "[large-model] No general-capable model downloaded — using the router model for agent/synthesis tasks"
            );
            return Ok(());
        }
        // Rank largest→smallest; same-size models with more cap overlap win.
        let agent_models = Self::ranked_candidates(agent_models, &text_caps, true);

        let cap = AppSettings::new().llm_large_n_ctx_cap;

        for mut largest in agent_models {
            largest.models_dir = models_dir.clone();
            if let Some(m) = workspace.write().get_model_mut(Some(&largest.model_name)) {
                m.models_dir = models_dir.clone();
            }
            // Sorted largest-first: once a candidate no longer out-ranks the
            // router, none of the remaining (smaller) ones will either.
            if largest.rank() <= orchestrator_rank {
                break;
            }

            let model_dir = workspace.read().model_dir_for(&largest.model_id);
            // Exclude the mmproj/CLIP GGUF — it is not loadable as a main
            // model; loading it as weights crashes the native loader.
            let (gguf_path, _mmproj) = self.find_ggufs(&model_dir);
            let Some(gguf_path) = gguf_path else {
                tracing::info!(
                    target: LOG_TARGET,
                    "[large-model] GGUF not found for {}, trying next-smaller",
                    largest.model_name
                );
                continue;
            };

            // Fit guard: skip a model whose capped footprint won't fit in free
            // VRAM after the reserve.  When it doesn't fit, promote down to
            // the next-largest model instead of dropping straight to the router.
            if !self.large_model_fits(&gguf_path, cap, &largest.model_name, None) {
                tracing::info!(
                    target: LOG_TARGET,
                    "[large-model] {} doesn't fit — trying next-smaller model",
                    largest.model_name
                );
                continue;
            }

            tracing::info!(
                target: LOG_TARGET,
                "[large-model] Loading larger model on demand: {} ({:.0}B) > router {} ({:.0}B), n_ctx<={} — {}",
                largest.model_name,
                largest.param_size_b(),
                router_name.as_deref().filter(|n| !n.is_empty()).unwrap_or("?"),
                orchestrator_size,
                cap,
                gguf_path
            );
            let thinking = self.thinking_for(&largest.model_name);
            let llm = self.load_llama(&gguf_path, &largest.model_name, cap, thinking)?;
            *self.large_llm.write() = Some(llm);
            *self.large_model_name.write() = Some(largest.model_name.clone());
            tracing::info!(
                target: LOG_TARGET,
                "[large-model] Loaded {} for agent/skill tasks",
                largest.model_name
            );
            return Ok(());
        }
        Ok(())
    }

    /// Pre-load the embedding model in a background thread.
    fn warmup_embeddings(&self) {
        let base_dir = self.workspace_handle().map(|ws| ws.read().workspace_dir.clone());
        let spawned = std::thread::Builder::new()
            .name("embedding-warmup".to_string())
            .spawn(move || {
                let result = std::panic::catch_unwind(|| crate::db::embeddings::warmup(base_dir.as_deref()));
                if let Err(panic) = result {
                    let msg = panic
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "unknown error".to_string());
                    tracing::warn!(target: LOG_TARGET, "[embeddings] Warmup failed: {msg}");
                }
            });
        if let Err(e) = spawned {
            tracing::warn!(target: LOG_TARGET, "[embeddings] Warmup failed: {e}");
        }
    }

    /// Run a trivial inference to prime the KV cache after model load.
    fn warmup_inference(&self) {
        let Some(llm) = self.llm.read().clone() else {
            return;
        };
        // `invoke` reports a generation error in its return value rather than
        // failing, so a completed call is a completed warmup.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            llm.invoke("<system>Reply OK</system><user>hi</user>", None, None)
        }));
        match result {
            Ok(_) => tracing::info!(target: LOG_TARGET, "[warmup] Inference warmup completed"),
            Err(panic) => {
                let msg = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "unknown error".to_string());
                tracing::warn!(target: LOG_TARGET, "[warmup] Inference warmup failed: {msg}");
            }
        }
    }

    /// The decision LLM, or `None` to fall back to the main model.
    pub fn decision_llm(&self) -> Option<Arc<dyn BaseLLM>> {
        self.decision_llm.read().clone()
    }

    // ----------------------------------------------------------- search/feedback

    /// Search the workspace-files collection with the shared selection criteria.
    ///
    /// The query is enriched from history for reference/keyword emphasis,
    /// recalled broadly, then filtered by the shared quality-floor +
    /// relevance-cutoff helper.  Indexing is done in the background by the
    /// indexer task, so this only queries the store.
    pub fn search_workspace(
        &self,
        query: &str,
        work_dir: &str,
        n_results: usize,
        conversation_history: Option<&[HashMap<String, String>]>,
    ) -> Vec<ChunkResult> {
        let Some(workspace) = self.workspace_handle() else {
            return Vec::new();
        };
        if work_dir.is_empty() || !Path::new(work_dir).is_dir() {
            return Vec::new();
        }
        let (current_project, workspace_dir) = {
            let ws = workspace.read();
            (ws.current_project().cloned(), ws.workspace_dir.clone())
        };
        let Some(current_project) = current_project else {
            return Vec::new();
        };
        let cache = self.caches(&current_project.id);
        let indexer = {
            let mut slot = cache.lock();
            match &slot.workspace_indexer {
                Some(indexer) => Ok(indexer.clone()),
                None => match WorkspaceIndexer::new(&current_project.project_dir, Some(&workspace_dir)) {
                    Ok(indexer) => {
                        let indexer = Arc::new(indexer);
                        slot.workspace_indexer = Some(indexer.clone());
                        Ok(indexer)
                    }
                    Err(e) => Err(e),
                },
            }
        };
        let indexer = match indexer {
            Ok(indexer) => indexer,
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "[workspace-index] search_workspace failed: {e}");
                cache.lock().workspace_indexer = None;
                return Vec::new();
            }
        };

        // Same emphasis step as documents: resolve references / keywords
        // against history so follow-ups are grounded.
        let search_query = match conversation_history {
            Some(history) if !history.is_empty() => self.build_history_search_query(query, history),
            _ => query.to_string(),
        };

        let recall = n_results.max(20);
        let mut chunks = indexer.search(&search_query, recall, RECALL_MAX_DISTANCE);
        // Raw-query fallback, mirroring the document path.
        if chunks.is_empty() && search_query != query {
            chunks = indexer.search(query, recall, RECALL_MAX_DISTANCE);
        }
        select_relevant_chunks(chunks, "workspace-search")
    }

    /// The feedback index for `project`, or for whatever is bound.
    ///
    /// `project` is explicit wherever the caller has already resolved one.
    /// Resolving it through `current_project` was wrong on any request that
    /// names its project in the *body*: nothing binds those, so the index was
    /// built for the project the UI had selected and a thumbs-up on one
    /// project was embedded into another's.
    pub fn ensure_feedback_indexer(&self, project: Option<&Project>) -> anyhow::Result<Arc<FeedbackIndexer>> {
        let workspace = self
            .workspace_handle()
            .ok_or_else(|| ModelServiceError::runtime("Model not loaded"))?;
        let resolved: Project = match project {
            Some(p) => p.clone(),
            None => workspace
                .read()
                .current_project()
                .cloned()
                .ok_or_else(|| ModelServiceError::value("No active project"))?,
        };
        let workspace_dir = workspace.read().workspace_dir.clone();
        let cache = self.caches(&resolved.id);
        let mut slot = cache.lock();
        if let Some(indexer) = &slot.feedback_indexer {
            return Ok(indexer.clone());
        }
        let indexer = match self.hooks().feedback_indexer_factory {
            Some(factory) => factory(&resolved.project_dir, Some(&workspace_dir))?,
            None => Arc::new(FeedbackIndexer::new(&resolved.project_dir, Some(&workspace_dir))?),
        };
        slot.feedback_indexer = Some(indexer.clone());
        Ok(indexer)
    }

    /// Record a thumbs up/down on an assistant message; returns the feedback
    /// summary dict (`feedback_id`, `session_id`, `message_id`, `rating`, `indexed`).
    pub fn submit_feedback(
        &self,
        session_id: &str,
        message_id: &str,
        project_id: &str,
        rating: &str,
    ) -> anyhow::Result<Value> {
        self.ensure_model()?;

        let history = self.get_session_history(session_id);
        if history.is_empty() {
            return Err(ModelServiceError::value(format!("Session not found: {session_id}")));
        }

        let Some(target_idx) = history
            .iter()
            .position(|m| m.message_id.as_deref() == Some(message_id))
        else {
            return Err(ModelServiceError::value(format!("Message not found: {message_id}")));
        };

        let target = &history[target_idx];
        if target.role != "assistant" {
            return Err(ModelServiceError::value(format!(
                "Message {message_id} is not an assistant message"
            )));
        }

        let question = history[..target_idx]
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or_default();

        let answer = target.content.clone();
        let conversation: Vec<ConversationTurn> = history[..=target_idx]
            .iter()
            .map(|m| ConversationTurn { role: m.role.clone(), content: m.content.clone() })
            .collect();

        let selected_project = self.get_project(project_id)?;

        let feedback_store = FeedbackStore::new(&selected_project.project_dir);
        let indexer = self.ensure_feedback_indexer(Some(&selected_project))?;

        // Update rating on the conversation message
        let updated = {
            let mut sessions = self.sessions.lock();
            let entry = sessions.entry(session_id.to_string()).or_insert_with(|| history.clone());
            if let Some(m) = entry.get_mut(target_idx) {
                m.rating = Some(rating.to_string());
            }
            entry.clone()
        };
        let store = self.conversation_store.read().clone();
        if let Some(store) = store {
            store.save(session_id, &updated)?;
        }

        if let Some(existing) = feedback_store.find(session_id, message_id)? {
            indexer.remove_feedback(&existing.feedback_id);
            feedback_store.remove(&existing.feedback_id)?;
        }

        if rating == "up" {
            let entry = FeedbackEntry::create(
                session_id,
                message_id,
                Rating::Up,
                &question,
                &answer,
                Some(conversation),
            );
            feedback_store.append(&entry)?;
            let indexed = !question.is_empty() && !answer.is_empty();
            if indexed {
                indexer.index_feedback(&entry)?;
            }

            Ok(json!({
                "feedback_id": entry.feedback_id,
                "session_id": session_id,
                "message_id": message_id,
                "rating": "up",
                "indexed": indexed,
            }))
        } else {
            let entry = FeedbackEntry::create(
                session_id,
                message_id,
                Rating::Down,
                &question,
                &answer,
                Some(conversation),
            );
            feedback_store.append(&entry)?;

            Ok(json!({
                "feedback_id": entry.feedback_id,
                "session_id": session_id,
                "message_id": message_id,
                "rating": "down",
                "indexed": false,
            }))
        }
    }

    /// `message_id -> rating` for one session in a project.
    pub fn get_feedback_ratings(&self, session_id: &str, project_id: &str) -> anyhow::Result<HashMap<String, String>> {
        self.ensure_model()?;

        let project = self.get_project(project_id)?;

        let feedback_store = FeedbackStore::new(&project.project_dir);

        Ok(feedback_store
            .load()?
            .into_iter()
            .filter(|e| e.session_id == session_id)
            .map(|e| {
                let rating = match e.rating {
                    Rating::Up => "up",
                    Rating::Down => "down",
                };
                (e.message_id, rating.to_string())
            })
            .collect())
    }

    /// Delete feedback for a specific message and remove it from the index.
    pub fn delete_feedback_ratings(
        &self,
        session_id: &str,
        message_id: &str,
        project_id: Option<&str>,
    ) -> anyhow::Result<()> {
        self.ensure_model()?;

        let project = match project_id.filter(|p| !p.is_empty()) {
            Some(project_id) => self.get_project(project_id)?,
            None => return Err(ModelServiceError::value("No active project")),
        };

        let feedback_store = FeedbackStore::new(&project.project_dir);
        if let Some(existing) = feedback_store.find(session_id, message_id)? {
            let indexer = self.ensure_feedback_indexer(Some(&project))?;
            indexer.remove_feedback(&existing.feedback_id);
            feedback_store.remove(&existing.feedback_id)?;
        }

        // Clear the rating on the in-memory session message
        let history = self.get_session_history(session_id);
        let updated = {
            let mut sessions = self.sessions.lock();
            let entry = sessions.entry(session_id.to_string()).or_insert(history);
            if let Some(m) = entry.iter_mut().find(|m| m.message_id.as_deref() == Some(message_id)) {
                m.rating = None;
            }
            entry.clone()
        };
        let store = self.conversation_store.read().clone();
        if let Some(store) = store {
            store.save(session_id, &updated)?;
        }
        Ok(())
    }

    /// Search stored feedback Q&A using the shared relevance criteria.
    ///
    /// Feedback was the one retrieval path with no quality floor: a single
    /// stored Q&A on an unrelated topic matched every question, poisoning the
    /// context and suppressing web search.  Recall broadly and then apply the
    /// same floor/cutoff as documents and workspace files.
    pub fn search_feedback(&self, query: &str, n_results: usize) -> Vec<ChunkResult> {
        let Some(workspace) = self.workspace_handle() else {
            return Vec::new();
        };
        match self.ensure_feedback_indexer(None) {
            Ok(indexer) => {
                let chunks = indexer.search(query, n_results.max(20), RECALL_MAX_DISTANCE);
                let mut kept = select_relevant_chunks(chunks, "feedback-search");
                kept.truncate(n_results);
                kept
            }
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "[feedback-index] search_feedback failed: {e}");
                let project_id = workspace.read().current_project().map(|p| p.id.clone());
                if let Some(project_id) = project_id {
                    self.caches(&project_id).lock().feedback_indexer = None;
                }
                Vec::new()
            }
        }
    }

    // ------------------------------------------------------------ build/train

    /// Build the GGUF model (no download step).
    ///
    /// Unloads the inference LLM to free memory, ensures the workspace/Model
    /// object is initialized, then converts to GGUF.  All artifacts must
    /// already be on disk.
    pub fn build_model(
        &self,
        model_name: Option<&str>,
        quantize: Option<&str>,
        developer_token: Option<&str>,
        progress_callback: Option<TrainProgressCallback>,
    ) -> anyhow::Result<()> {
        let _ = (model_name, developer_token);
        self.unload_model();
        self.ensure_model()?;

        let model = self
            .model_handle()
            .ok_or_else(|| ModelServiceError::runtime("Model not loaded"))?;
        if model.workspace.read().current_project().is_none() {
            return Err(ModelServiceError::value(
                "No active project. Please create a project first.",
            ));
        }

        if let Some(quantize) = quantize.filter(|q| !q.is_empty()) {
            let mut ws = model.workspace.write();
            if let Some(config) = ws.get_model_mut(None).and_then(|m| m.config.as_mut()) {
                config.quantize = quantize.to_string();
            }
        }

        tracing::info!(target: LOG_TARGET, "Starting model build…");
        let result = model.build(progress_callback);
        if result.is_ok() {
            tracing::info!(target: LOG_TARGET, "Model build completed.");
        }
        *self.llm.write() = None;
        *self.model_name.write() = None;
        *self.decision_llm.write() = None;
        *self.decision_model_name.write() = None;
        result
    }

    /// Run the full training pipeline.
    ///
    /// Unloads the inference LLM, initialises the workspace, trains, then
    /// resets entirely so a fresh `load_model()` picks up the new weights.
    pub fn train_model(
        &self,
        model_name: Option<&str>,
        progress_callback: Option<TrainProgressCallback>,
    ) -> anyhow::Result<()> {
        let _ = model_name;
        // 1. Free inference model from memory first
        self.unload_model();

        // 2. Initialise workspace (no LLM loaded)
        self.ensure_model()?;

        tracing::info!(target: LOG_TARGET, "Starting model training…");
        let result = match self.model_handle() {
            Some(model) => model.train(progress_callback).map(|()| {
                tracing::info!(target: LOG_TARGET, "Model training completed.");
                // Persist updated document statuses (PENDING → PROCESSED) to
                // disk so the auto-train check in unload_model() does not
                // re-fire.
                if let Err(e) = self.persist_model() {
                    tracing::warn!(target: LOG_TARGET, "Could not save model after training: {e}");
                }
            }),
            None => Err(ModelServiceError::runtime("Model not loaded")),
        };
        // Reset entirely so next load_model() re-initialises with new weights
        *self.llm.write() = None;
        *self.model.write() = None;
        *self.model_name.write() = None;
        *self.decision_llm.write() = None;
        *self.decision_model_name.write() = None;
        Workspace::reset();
        result
    }

    /// Get the currently loaded LLM instance.
    pub fn get_llm(&self) -> Option<Arc<dyn BaseLLM>> {
        self.llm.read().clone()
    }

    /// Get the current project name.
    pub fn get_current_project_name(&self) -> Option<String> {
        self.current_project_copy().and_then(|p| p.name)
    }

    /// Get the current project ID.
    pub fn get_project_id(&self) -> Option<String> {
        self.current_project_copy().map(|p| p.id)
    }

    // ------------------------------------------------------------ project tools

    /// The launch config of the built-in System Tools server.
    fn system_tools_config() -> MCPServerConfig {
        MCPServerConfig {
            command: sys_executable(),
            args: vec!["-m".to_string(), "smartloop.api.mcp.system_server".to_string()],
            ..Default::default()
        }
    }

    /// Register (or refresh) the System Tools MCP server on a project.
    pub fn install_system_tools_server(&self, project_id: &str) -> anyhow::Result<()> {
        let defs = Self::system_tool_defs();
        let wanted: Vec<String> = defs.iter().map(|t| t.name.clone()).collect();
        // Match on the stable id, not the display name: records saved under
        // an earlier name (e.g. "Smartloop") would otherwise never be found,
        // so their tool list stays stale and a duplicate server with the same
        // id gets appended.
        enum Outcome {
            Refreshed,
            Current,
            Registered,
        }
        let outcome = self.update_project_record(project_id, |project| {
            if let Some(existing) = project.mcp_servers.iter_mut().find(|s| s.id == SYSTEM_TOOLS_SERVER_ID) {
                // Refresh the advertised name/tool list so projects created
                // before a rename or before a tool was added (e.g. grep) pick
                // up the changes on next load.
                let current: Vec<String> = existing.tools.iter().map(|t| t.name.clone()).collect();
                if current != wanted || existing.name != SYSTEM_TOOLS_SERVER_NAME {
                    existing.name = SYSTEM_TOOLS_SERVER_NAME.to_string();
                    existing.tools = defs.clone();
                    return Outcome::Refreshed;
                }
                return Outcome::Current;
            }
            project.mcp_servers.push(MCPServer {
                id: SYSTEM_TOOLS_SERVER_ID.to_string(),
                name: SYSTEM_TOOLS_SERVER_NAME.to_string(),
                description: Some("Built-in tools for file operations, search, and shell commands".to_string()),
                server_type: MCPServerType::Local,
                config: Some(Self::system_tools_config()),
                enabled: true,
                system: true,
                tools: defs.clone(),
                ..Default::default()
            });
            Outcome::Registered
        })?;
        match outcome {
            Outcome::Refreshed => {
                self.persist_model()?;
                tracing::info!(
                    target: LOG_TARGET,
                    "Refreshed System Tools server tools: {}",
                    py_list_repr(&wanted)
                );
            }
            Outcome::Current => {
                tracing::info!(target: LOG_TARGET, "System Tools server already registered");
            }
            Outcome::Registered => {
                self.persist_model()?;
                tracing::info!(target: LOG_TARGET, "Registered System Tools server on project {project_id}");
            }
        }
        Ok(())
    }

    /// Remove the legacy fetch MCP server — fetching is now the FetchAgent.
    ///
    /// URL fetching moved from an MCP server (`__fetch__`) to an in-process
    /// agent resolved as a plan step, so any previously installed fetch
    /// server is dropped from the project.
    pub fn remove_fetch_server(&self, project_id: &str) -> anyhow::Result<()> {
        let (removed, project) = self.update_project_record(project_id, |project| {
            let before = project.mcp_servers.len();
            project.mcp_servers.retain(|s| s.id != FETCH_SERVER_ID);
            (project.mcp_servers.len() != before, project.clone())
        })?;
        if removed {
            self.persist_model()?;
            tracing::info!(target: LOG_TARGET, "Removed legacy fetch server from project {project_id}");
        }
        self.remove_stale_fetch_skills(&project);
        Ok(())
    }

    /// Delete leftover fetch skill directories from disk.
    ///
    /// Projects created before the FetchAgent change may still have
    /// `fetch-mcp`/`fetch-url` SKILL.md directories on disk.  They reference
    /// the removed `__fetch__` server, so drop them.
    fn remove_stale_fetch_skills(&self, project: &Project) {
        let skills_dir = Self::skills_dir(project);
        for name in Self::STALE_FETCH_SKILLS {
            let skill_dir = Path::new(&skills_dir).join(name);
            if skill_dir.is_dir() {
                let _ = std::fs::remove_dir_all(&skill_dir);
                tracing::info!(
                    target: LOG_TARGET,
                    "Removed stale fetch skill '{name}' from project {}",
                    project.id
                );
            }
        }
    }

    /// Migrate legacy REMOTE system servers to LOCAL (stdio subprocess) and fix
    /// empty launcher args.
    pub fn migrate_servers_to_local(&self, project_id: &str) -> anyhow::Result<()> {
        // command → how that launcher takes a package name.
        fn launcher_args(command: &str, pkg: &str) -> Option<Vec<String>> {
            let args: Vec<&str> = match command {
                "uvx" => vec!["--from", pkg, pkg],
                "uv" => vec!["tool", "run", "--from", pkg, pkg],
                "npx" => vec!["-y", pkg],
                "bunx" => vec![pkg],
                "pipx" => vec!["run", pkg],
                "deno" => vec!["run", "-A", pkg],
                _ => return None,
            };
            Some(args.into_iter().map(str::to_string).collect())
        }

        let migrated = self.update_project_record(project_id, |project| {
            let mut migrated = false;
            for server in project.mcp_servers.iter_mut() {
                if server.id == SYSTEM_TOOLS_SERVER_ID && server.server_type == MCPServerType::Remote {
                    server.server_type = MCPServerType::Local;
                    server.config = Some(Self::system_tools_config());
                    server.remote_config = None;
                    migrated = true;
                    tracing::info!(
                        target: LOG_TARGET,
                        "Migrated system tools server to LOCAL on project {}",
                        project.id
                    );
                }
                // Fix stale user-registered local servers saved with empty
                // args (registered before args validation was enforced),
                // recovering from the saved pkg or server name.  How a package
                // name is passed is the launcher's own convention — handing
                // uv's `--from <pkg> <pkg>` to npx or deno produces a command
                // that cannot start.
                if server.server_type == MCPServerType::Local {
                    let pkg_name = truthy(&server.pkg).unwrap_or(&server.name).to_string();
                    let server_name = server.name.clone();
                    let fixed = server
                        .config
                        .as_mut()
                        .filter(|config| config.args.is_empty())
                        .and_then(|config| launcher_args(&config.command, &pkg_name).map(|args| (config, args)));
                    if let Some((config, args)) = fixed {
                        config.args = args;
                        migrated = true;
                        tracing::info!(
                            target: LOG_TARGET,
                            "Fixed empty args for local MCP server '{server_name}' → {} {}",
                            config.command,
                            py_list_repr(&config.args)
                        );
                    }
                }
            }
            migrated
        })?;
        if migrated {
            self.persist_model()?;
        }
        Ok(())
    }

    /// True for built-in tool skills and `__`-prefixed internal skills.
    pub fn is_system_skill(name: &str) -> bool {
        !name.is_empty() && (Self::SYSTEM_SKILL_NAMES.contains(&name) || name.starts_with("__"))
    }

    /// Install system MCP servers and register default skills.
    pub fn setup_project_tools(&self, project_id: &str) -> anyhow::Result<()> {
        self.migrate_servers_to_local(project_id)?;
        self.install_system_tools_server(project_id)?;
        self.remove_fetch_server(project_id)?;
        // Re-fetch: install_* may have replaced the model via ensure_model(),
        // so an earlier project snapshot can be stale.
        let project = self.get_project(project_id)?;
        self.install_default_skills(&project)
    }

    /// Write default skill SKILL.md files if they don't already exist.
    fn install_default_skills(&self, project: &Project) -> anyhow::Result<()> {
        let existing: HashSet<String> = Self::read_skills_from_disk(project).into_iter().map(|s| s.name).collect();
        for skill in Self::default_skills() {
            if !existing.contains(&skill.name) {
                Self::write_skill_md(project, &skill)?;
                tracing::info!(
                    target: LOG_TARGET,
                    "Installed default skill '{}' on project {}",
                    skill.name,
                    project.id
                );
            }
        }
        Ok(())
    }

    /// Return the system project (general_chat), if it exists.
    pub fn get_system_project(&self) -> Option<Project> {
        let ws = self.workspace_handle()?;
        let guard = ws.read();
        guard.projects.iter().find(|p| p.system).cloned()
    }

    // ------------------------------------------------------------------ skills

    /// Get project skills for the current project (read from disk).
    pub fn get_project_skills(&self) -> Vec<Skill> {
        if self.model.read().is_none() {
            if let Err(e) = self.ensure_model() {
                tracing::warn!(target: LOG_TARGET, "Could not get project skills: {e}");
                return Vec::new();
            }
        }
        if self.model.read().is_none() {
            return Vec::new();
        }
        match self.current_project_copy() {
            Some(project) if truthy(&project.name).is_some() => project.get_skills(),
            _ => Vec::new(),
        }
    }

    /// Add a skill to a project and persist as SKILL.md; returns the skill dict.
    pub fn add_skill(
        &self,
        project_id: &str,
        content: &str,
        name: Option<&str>,
        description: Option<&str>,
    ) -> anyhow::Result<Value> {
        static NON_SLUG: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^a-z0-9]+").expect("valid regex"));

        let project = self.get_project(project_id)?;

        let (body, meta) = Self::parse_skill_content(content);
        let meta_str = |key: &str| meta.get(key).and_then(Value::as_str).map(str::to_string);
        let final_description = description
            .filter(|d| !d.is_empty())
            .map(str::to_string)
            .or_else(|| meta_str("description"));

        // Skills are authored directly (indexed and matched by user intent);
        // nothing is LLM-generated here.  Persist the provided content/metadata.
        let resolved_name = name
            .filter(|n| !n.is_empty())
            .map(str::to_string)
            .or_else(|| meta_str("name"))
            .filter(|n| !n.is_empty());
        let raw_name = resolved_name.unwrap_or_else(|| Self::extract_skill_name(&body));
        let slug = NON_SLUG.replace_all(&raw_name.to_lowercase(), "-").trim_matches('-').to_string();
        let sanitized_name = if slug.is_empty() {
            format!("skill-{}", &uuid::Uuid::new_v4().to_string()[..8])
        } else {
            slug
        };

        let mut skill = Skill {
            id: sanitized_name.clone(),
            name: sanitized_name,
            description: final_description,
            content: body,
            system: false,
        };

        Self::write_skill_md(&project, &skill)?;
        skill.content = Self::render_skill_content(&skill);
        Ok(serde_json::to_value(&skill)?)
    }

    /// Split raw SKILL.md text into `(body, frontmatter_fields)`.
    ///
    /// If the text starts with `---` frontmatter, the metadata fields (name,
    /// description, tools) are extracted and the body is returned without the
    /// fences.  Otherwise the full text is returned as-is.
    pub fn parse_skill_content(raw: &str) -> (String, Map<String, Value>) {
        let mut meta = Map::new();
        let mut body = raw.to_string();
        if raw.starts_with("---") {
            let parts: Vec<&str> = raw.splitn(3, "---").collect();
            if parts.len() >= 3 {
                body = parts[2].trim().to_string();
                for line in parts[1].trim().lines() {
                    if let Some(rest) = line.strip_prefix("name:") {
                        meta.insert("name".to_string(), Value::String(rest.trim().to_string()));
                    } else if let Some(rest) = line.strip_prefix("description:") {
                        meta.insert("description".to_string(), Value::String(rest.trim().to_string()));
                    } else if let Some(rest) = line.strip_prefix("tools:") {
                        let tools_str = rest.trim().trim_matches(|c| c == '[' || c == ']');
                        let tools: Vec<Value> = tools_str
                            .split(',')
                            .map(str::trim)
                            .filter(|t| !t.is_empty())
                            .map(|t| Value::String(t.to_string()))
                            .collect();
                        meta.insert("tools".to_string(), Value::Array(tools));
                    }
                }
            }
        }
        (body, meta)
    }

    /// Update a skill on disk from full SKILL.md text; returns the skill dict.
    ///
    /// `content` is the full SKILL.md text including YAML frontmatter (name,
    /// description, tools).
    pub fn update_skill(&self, project_id: &str, skill_id: &str, content: &str) -> anyhow::Result<Value> {
        let project = self.get_project(project_id)?;
        if truthy(&project.name).is_none() {
            return Err(ModelServiceError::value(format!("Skill not found: {skill_id}")));
        }

        // Find existing skill so we can detect renames.
        let Some(old_skill) = Self::read_skills_from_disk(&project)
            .into_iter()
            .find(|s| s.name == skill_id || s.id == skill_id)
        else {
            return Err(ModelServiceError::value(format!("Skill not found: {skill_id}")));
        };

        let old_name = old_skill.name.clone();

        // Parse frontmatter from the incoming content.
        let (body, meta) = Self::parse_skill_content(content);
        let new_name = meta
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| old_name.clone());

        let mut skill = Skill {
            id: new_name.clone(),
            name: new_name.clone(),
            description: match meta.get("description").and_then(Value::as_str) {
                Some(d) => Some(d.to_string()),
                None => old_skill.description.clone(),
            },
            content: body,
            system: false,
        };

        // If name changed, remove the old directory.
        if new_name != old_name {
            let old_dir = Path::new(&Self::skills_dir(&project)).join(&old_name);
            if old_dir.is_dir() {
                std::fs::remove_dir_all(&old_dir)?;
            }
        }

        Self::write_skill_md(&project, &skill)?;

        skill.content = Self::render_skill_content(&skill);
        Ok(serde_json::to_value(&skill)?)
    }

    /// Extract a 1-2 word skill name from content as a fallback.
    ///
    /// Looks for markdown headings or the first meaningful line.
    pub fn extract_skill_name(content: &str) -> String {
        static HEADING: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)^#{1,6}\s+(.+)").expect("valid regex"));

        let converter = htmd::HtmlToMarkdown::builder()
            .skip_tags(vec!["img", "script", "style"])
            .build();
        let text = converter.convert(content).unwrap_or_else(|_| content.to_string());
        if let Some(caps) = HEADING.captures(&text) {
            let words: Vec<&str> = caps[1].split_whitespace().take(2).collect();
            return words.join(" ");
        }
        for line in text.lines() {
            let stripped = line.trim();
            if !stripped.is_empty() && stripped.chars().count() > 3 && !stripped.starts_with("---") {
                let words: Vec<&str> = stripped.split_whitespace().take(2).collect();
                return words.join(" ");
            }
        }
        "skill".to_string()
    }

    /// Return the project's documents directory (under `project_dir`).
    pub fn project_documents_dir(project: &Project) -> String {
        normalize_path(&project.documents_dir())
    }

    /// Render a Skill as full SKILL.md text with YAML frontmatter.
    pub fn render_skill_content(skill: &Skill) -> String {
        let mut lines = vec!["---".to_string(), format!("name: {}", skill.name)];
        if let Some(description) = truthy(&skill.description) {
            lines.push(format!("description: {description}"));
        }
        lines.push("---".to_string());
        // Strip any existing frontmatter from content before prepending.
        let (body, _) = Self::parse_skill_content(&skill.content);
        format!("{}\n\n{}", lines.join("\n"), body)
    }

    /// Write a SKILL.md file with YAML frontmatter for the given skill.
    pub fn write_skill_md(project: &Project, skill: &Skill) -> std::io::Result<()> {
        let skill_dir = Path::new(&Self::skills_dir(project)).join(&skill.name);
        std::fs::create_dir_all(&skill_dir)?;
        std::fs::write(skill_dir.join("SKILL.md"), Self::render_skill_content(skill))
    }

    /// Return the skills directory for a project (under `project_dir`).
    pub fn skills_dir(project: &Project) -> String {
        normalize_path(&project.skills_dir())
    }

    /// Read all skills from the project skills directory.
    ///
    /// Each subdirectory containing a SKILL.md with YAML frontmatter is parsed
    /// into a `Skill`.
    pub fn read_skills_from_disk(project: &Project) -> Vec<Skill> {
        let skills_dir = Self::skills_dir(project);
        let dir = Path::new(&skills_dir);
        if !dir.is_dir() {
            return Vec::new();
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut names: Vec<String> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();

        let mut skills = Vec::new();
        for entry in names {
            let entry_path = dir.join(&entry);
            if !entry_path.is_dir() {
                continue;
            }
            let skill_path = entry_path.join("SKILL.md");
            if !skill_path.is_file() {
                continue;
            }

            let raw = match std::fs::read_to_string(&skill_path) {
                Ok(raw) => raw,
                Err(e) => {
                    tracing::warn!(
                        target: LOG_TARGET,
                        "Could not read skill {}: {e}",
                        skill_path.display()
                    );
                    continue;
                }
            };

            let mut name = entry.clone();
            let mut description = None;

            if raw.starts_with("---") {
                let parts: Vec<&str> = raw.splitn(3, "---").collect();
                if parts.len() >= 3 {
                    for line in parts[1].trim().lines() {
                        if let Some(rest) = line.strip_prefix("name:") {
                            name = rest.trim().to_string();
                        } else if let Some(rest) = line.strip_prefix("description:") {
                            description = Some(rest.trim().to_string());
                        }
                    }
                }
            }

            skills.push(Skill {
                id: name.clone(),
                name,
                description,
                content: raw,
                system: false,
            });
        }

        skills
    }

    /// Get a single skill by name from disk.
    pub fn get_skill(&self, project_id: &str, skill_id: &str) -> anyhow::Result<Value> {
        let project = self.get_project(project_id)?;
        for skill in Self::read_skills_from_disk(&project) {
            if skill.name == skill_id || skill.id == skill_id {
                return Ok(serde_json::to_value(&skill)?);
            }
        }
        Err(ModelServiceError::value(format!("Skill not found: {skill_id}")))
    }

    /// List a project's user-created skills, read from disk.
    ///
    /// Built-in tool skills and `__`-prefixed internal skills back the
    /// orchestrator's routing but are never exposed here.
    pub fn list_skills(&self, project_id: &str) -> anyhow::Result<Vec<Value>> {
        let project = self.get_project(project_id)?;
        Self::read_skills_from_disk(&project)
            .into_iter()
            .filter(|s| !Self::is_system_skill(&s.name))
            .map(|s| serde_json::to_value(&s).map_err(anyhow::Error::from))
            .collect()
    }

    /// Delete a skill by removing its SKILL.md directory from disk.
    pub fn delete_skill(&self, project_id: &str, skill_id: &str) -> anyhow::Result<Value> {
        let project = self.get_project(project_id)?;

        // Verify it exists on disk
        let Some(skill) = Self::read_skills_from_disk(&project)
            .into_iter()
            .find(|s| s.name == skill_id || s.id == skill_id)
        else {
            return Err(ModelServiceError::value(format!("Skill not found: {skill_id}")));
        };

        let skill_dir = Path::new(&Self::skills_dir(&project)).join(&skill.name);
        if skill_dir.is_dir() {
            std::fs::remove_dir_all(&skill_dir)?;
            tracing::info!(target: LOG_TARGET, "Deleted skill directory: {}", skill_dir.display());
        }

        Ok(json!({"id": skill_id, "deleted": true, "message": "Skill deleted successfully"}))
    }

    // ---------------------------------------------------------------- context

    /// Build a project info summary for context injection.
    pub fn get_project_info(&self) -> Option<String> {
        let project = self.current_project_copy()?;

        let mut lines = Vec::new();
        lines.push(format!(
            "Project: {}",
            truthy(&project.name).unwrap_or(&project.id)
        ));
        if let Some(description) = truthy(&project.description) {
            lines.push(format!("Description: {description}"));
        }

        // Documents summary — only list indexed documents, omit status
        let docs: Vec<_> = project
            .documents
            .iter()
            .filter(|d| d.status == DocumentStatus::Indexed)
            .collect();
        if !docs.is_empty() {
            lines.push(format!("\nDocuments ({}):", docs.len()));
            for doc in docs {
                let doc_name = Path::new(&doc.path)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                lines.push(format!("  - {doc_name}"));
            }
        }

        // MCP servers & tools summary
        let servers: Vec<_> = project.mcp_servers.iter().filter(|s| s.enabled && !s.system).collect();
        lines.push(format!("\nMCP Servers ({}):", servers.len()));
        for srv in servers {
            let tool_names: Vec<&str> = srv.tools.iter().map(|t| t.name.as_str()).collect();
            let desc = truthy(&srv.description).map(|d| format!(" - {d}")).unwrap_or_default();
            lines.push(format!("  - {}{desc}", srv.name));
            if !tool_names.is_empty() {
                lines.push(format!("    Tools: {}", tool_names.join(", ")));
            }
        }

        Some(lines.join("\n"))
    }

    /// Vector search on indexed documents; the joined context string or `None`.
    ///
    /// `None` when there are no documents, no relevant results, or the results
    /// don't address the user's intent — allowing the caller to fall back to
    /// local explore or the model's trained knowledge.
    pub fn get_document_context(&self, query: Option<&str>) -> Option<String> {
        self.get_document_context_with_citations(query, None, None)
            .map(|(context, _)| context)
    }

    /// Core vector search → `(doc_groups, citations)` or `None`.  `doc_groups`
    /// is an ordered `(document name, combined source text)` list.
    ///
    /// Shared by [`ModelService::get_document_context_with_citations`] (which
    /// joins the groups into one string) and
    /// [`ModelService::get_document_groups`] (which exposes them per document
    /// so the orchestrator can fan out).  Summaries are stored as vector
    /// content at index time; the original text comes from `source_content`.
    pub fn vector_search_grouped(
        &self,
        query: Option<&str>,
        skill_instruction: Option<&str>,
        conversation_history: Option<&[HashMap<String, String>]>,
    ) -> Option<(Vec<(String, String)>, Vec<Value>)> {
        let _ = skill_instruction;
        let (Some(workspace), Some(query)) = (self.workspace_handle(), query.filter(|q| !q.is_empty())) else {
            tracing::info!(target: LOG_TARGET, "[vector-search] Skipped — no model or query");
            return None;
        };

        match self.vector_search_grouped_inner(&workspace, query, conversation_history) {
            Ok(result) => result,
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "Vector search failed: {e:?}");
                None
            }
        }
    }

    fn vector_search_grouped_inner(
        &self,
        workspace: &SharedWorkspace,
        query: &str,
        conversation_history: Option<&[HashMap<String, String>]>,
    ) -> anyhow::Result<Option<(Vec<(String, String)>, Vec<Value>)>> {
        let (current_project, workspace_dir) = {
            let ws = workspace.read();
            (ws.current_project().cloned(), ws.workspace_dir.clone())
        };
        let Some(current_project) = current_project else {
            tracing::info!(target: LOG_TARGET, "[vector-search] Skipped — no active project");
            return Ok(None);
        };

        // Reuse a single store per project to avoid SQLite lock conflicts
        // with the background indexer.
        let cache = self.caches(&current_project.id);
        let store = {
            let mut slot = cache.lock();
            match &slot.doc_store {
                Some(store) => store.clone(),
                None => {
                    let store = StoreFactory::create_store(
                        &current_project.project_dir,
                        "documents",
                        Some(&workspace_dir),
                        None,
                    )?;
                    slot.doc_store = Some(store.clone());
                    store
                }
            }
        };

        let search_query = match conversation_history {
            Some(history) if !history.is_empty() => self.build_history_search_query(query, history),
            _ => query.to_string(),
        };

        // Semantic discovery: find which documents are relevant.  The goal is
        // document-level recall — with multi-chunk documents a small
        // n_results can be dominated by chunks from 1-2 documents, hiding the
        // rest.  Use a generous limit; the expansion step deduplicates by
        // document_id anyway.
        let indexed_doc_count = current_project
            .documents
            .iter()
            .filter(|d| matches!(d.status, DocumentStatus::Indexed | DocumentStatus::Pending))
            .count();
        let n_results = (indexed_doc_count * 5).max(20);

        tracing::info!(
            target: LOG_TARGET,
            "[vector-search] Querying {indexed_doc_count} indexed document(s) with: {}",
            search_query.chars().take(120).collect::<String>()
        );

        let mut discovery = store.query_with_metadata(&search_query, Some(n_results), 0.8, None)?;

        // Fallback: if the enriched query missed, retry with just the raw
        // user query as a safety net.
        if discovery.is_empty() && search_query != query {
            tracing::info!(
                target: LOG_TARGET,
                "[vector-search] Enriched query returned no results, retrying with raw query: {}",
                query.chars().take(120).collect::<String>()
            );
            discovery = store.query_with_metadata(query, Some(n_results), 0.8, None)?;
        }

        if discovery.is_empty() {
            tracing::info!(target: LOG_TARGET, "[vector-search] No chunks found");
            return Ok(None);
        }

        // Apply the shared relevance-selection criteria (quality floor +
        // relevance cutoff).  Without the floor, weak matches populate
        // vector_context and force the orchestrator onto the EXPLORE path,
        // wrapping a conversational prompt in DOCUMENT CONTEXT scaffolding
        // pointing at irrelevant docs.
        let discovery = select_relevant_chunks(discovery, "vector-search");
        if discovery.is_empty() {
            return Ok(None);
        }

        // Log distances to help diagnose relevance issues
        let unique_docs: HashSet<String> = discovery.iter().map(chunk_key).collect();
        let dist_summary = discovery
            .iter()
            .take(6)
            .map(|r| format!("{}={:.3}", chunk_key(r), r.distance))
            .collect::<Vec<_>>()
            .join(", ");
        tracing::info!(
            target: LOG_TARGET,
            "[vector-search] Discovery returned {} chunk(s) from {} document(s) — distances: [{dist_summary}]",
            discovery.len(),
            unique_docs.len()
        );

        // Best (lowest) distance per document id, so everything is ordered by
        // relevance.
        let mut doc_id_distance: HashMap<String, f64> = HashMap::new();
        for r in &discovery {
            if !r.document_id.is_empty() {
                let entry = doc_id_distance.entry(r.document_id.clone()).or_insert(r.distance);
                if r.distance < *entry {
                    *entry = r.distance;
                }
            }
        }

        // Expansion: fetch ALL chunks from every matched document.  Nothing
        // is dropped — all documents are passed through.  Documents are
        // fetched in parallel batches ordered by similarity so the most
        // relevant documents are ready first.
        let mut matched_doc_ids: Vec<String> = Vec::new();
        for r in &discovery {
            if !r.document_id.is_empty() && !matched_doc_ids.contains(&r.document_id) {
                matched_doc_ids.push(r.document_id.clone());
            }
        }
        // Sort doc IDs by similarity (closest first).
        matched_doc_ids.sort_by(|a, b| {
            let da = doc_id_distance.get(a).copied().unwrap_or(f64::INFINITY);
            let db = doc_id_distance.get(b).copied().unwrap_or(f64::INFINITY);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut chunk_results: Vec<ChunkResult> = Vec::new();
        for batch in matched_doc_ids.chunks(DOC_BATCH_SIZE) {
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::scope(|scope| {
                for doc_id in batch {
                    let tx = tx.clone();
                    let store = store.clone();
                    scope.spawn(move || {
                        let _ = tx.send(store.get_by_document_ids(std::slice::from_ref(doc_id)));
                    });
                }
            });
            drop(tx);
            // Completion order, like `as_completed`.
            for result in rx {
                chunk_results.extend(result?);
            }
        }

        if chunk_results.is_empty() {
            chunk_results = discovery.clone();
        }

        if chunk_results.is_empty() {
            return Ok(None);
        }

        // Group source text by document, ordered by relevance.  Both the
        // concatenated-context and per-document (fan-out) callers build their
        // output from these groups.
        let mut grouped: IndexMap<String, Vec<String>> = IndexMap::new();
        for r in &chunk_results {
            let content = match r.source_content.as_deref() {
                Some(s) if !s.is_empty() => s.to_string(),
                _ => r.text.clone(),
            };
            grouped.entry(chunk_key(r)).or_default().push(content);
        }
        let doc_groups: Vec<(String, String)> = grouped
            .into_iter()
            .map(|(name, texts)| (name, texts.join("\n")))
            .collect();

        // Build citations directly from chunk metadata.  Deduplicate by
        // document, accumulate page numbers from all matched chunks, ordered
        // by similarity.
        let mut seen_docs: IndexMap<String, Map<String, Value>> = IndexMap::new();
        for chunk in &discovery {
            if chunk.document_id.is_empty() {
                continue;
            }
            let page = chunk.metadata.get("page_number").cloned().filter(value_truthy);
            if !seen_docs.contains_key(&chunk.document_id) {
                let mut citation = Map::new();
                citation.insert("document_id".to_string(), Value::String(chunk.document_id.clone()));
                let document_name = match chunk.document_name.as_deref() {
                    Some(n) if !n.is_empty() => Value::String(n.to_string()),
                    _ => chunk.metadata.get("document_name").cloned().unwrap_or(Value::Null),
                };
                citation.insert("document_name".to_string(), document_name);
                // Carry the exact text this document contributed to the
                // prompt, privately (`_` keys are stripped before a citation
                // reaches a client).  Being retrieved is not evidence a
                // document was used — nearest neighbour always returns
                // something — so the router checks the finished answer
                // against this before crediting the source.
                let key = chunk_key(chunk);
                let extract = doc_groups
                    .iter()
                    .find(|(name, _)| *name == key)
                    .map(|(_, extract)| extract)
                    .filter(|extract| !extract.is_empty());
                if let Some(extract) = extract {
                    citation.insert("_extract".to_string(), Value::String(extract.clone()));
                }
                citation.insert("_chunks".to_string(), Value::Array(Vec::new()));
                citation.insert("_vectors".to_string(), Value::Array(Vec::new()));
                if let Some(page) = &page {
                    citation.insert("page_numbers".to_string(), Value::Array(vec![page.clone()]));
                }
                seen_docs.insert(chunk.document_id.clone(), citation);
            } else if let Some(page) = &page {
                let existing = seen_docs.get_mut(&chunk.document_id).expect("present");
                let mut pages = existing
                    .get("page_numbers")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                if !pages.contains(page) {
                    pages.push(page.clone());
                    sort_values(&mut pages);
                    existing.insert("page_numbers".to_string(), Value::Array(pages));
                }
            }

            // The chunks that actually matched, closest first — the passages
            // this document was retrieved *for*.  `_extract` above is the
            // whole document, and checking a finished answer against a whole
            // document credits almost anything.  These are what the router
            // scores instead.
            //
            // Their vectors ride along too (the raw index BLOB, as an array of
            // bytes; `null` when the store has none), so the router can check
            // them against the answer without re-encoding any source text.
            let existing = seen_docs.get_mut(&chunk.document_id).expect("present");
            let chunk_text = chunk.text.trim();
            let n_chunks = existing.get("_chunks").and_then(Value::as_array).map(Vec::len).unwrap_or(0);
            if !chunk_text.is_empty() && n_chunks < MAX_CITATION_CHUNKS {
                if let Some(Value::Array(chunks)) = existing.get_mut("_chunks") {
                    chunks.push(Value::String(chunk_text.to_string()));
                }
                let vector = match &chunk.embedding {
                    Some(blob) => Value::Array(blob.iter().map(|b| Value::from(*b)).collect()),
                    None => Value::Null,
                };
                if let Some(Value::Array(vectors)) = existing.get_mut("_vectors") {
                    vectors.push(vector);
                }
            }
        }

        // seen_docs preserves discovery (similarity) order.
        let all_citations: Vec<Value> = seen_docs.into_values().map(Value::Object).collect();
        Ok(Some((doc_groups, all_citations)))
    }

    /// Vector search → `(context_str, citations_list)` or `None`.
    ///
    /// Joins all matched documents into a single context string.
    pub fn get_document_context_with_citations(
        &self,
        query: Option<&str>,
        skill_instruction: Option<&str>,
        conversation_history: Option<&[HashMap<String, String>]>,
    ) -> Option<(String, Vec<Value>)> {
        let (doc_groups, citations) = self.vector_search_grouped(query, skill_instruction, conversation_history)?;
        let context = doc_groups
            .iter()
            .map(|(name, content)| format!("=== Document: {name} ===\n{content}"))
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");
        Some((context, citations))
    }

    /// Vector search → `(documents, citations)` for fan-out, where each
    /// document is `{"document_name", "content"}`.
    ///
    /// One entry per matched document, ordered by relevance — so the
    /// orchestrator can process each document (or batch) in its own DAG node
    /// and reduce them into a single synthesized result.
    pub fn get_document_groups(
        &self,
        query: Option<&str>,
        skill_instruction: Option<&str>,
        conversation_history: Option<&[HashMap<String, String>]>,
    ) -> Option<(Vec<Value>, Vec<Value>)> {
        let (doc_groups, citations) = self.vector_search_grouped(query, skill_instruction, conversation_history)?;
        let documents = doc_groups
            .into_iter()
            .map(|(name, content)| json!({"document_name": name, "content": content}))
            .collect();
        Some((documents, citations))
    }

    /// `generate_completion(prompt)` through the test seam.
    fn rewrite_completion(&self, prompt: &str) -> anyhow::Result<String> {
        match self.hooks().generate_completion {
            Some(hook) => hook(prompt),
            None => self.generate_completion(prompt, None, None, None, None),
        }
    }

    /// `result.strip().strip('"\'')`.
    fn clean_rewrite(result: &str) -> String {
        result.trim().trim_matches(|c| c == '"' || c == '\'').to_string()
    }

    /// Use the LLM to synthesize a vector-search query from the user message
    /// and the matched skill instruction.
    ///
    /// The generated query targets the *data* the skill needs rather than the
    /// user's high-level request.
    pub fn build_skill_search_query(&self, user_query: &str, skill_instruction: &str) -> String {
        let prompt = format!(
            "Given the user's request and a skill instruction, generate a short \
search query (max 15 words) that will find the relevant document \
content in a vector database. Focus on the data and entities the \
skill needs, not the action the user asked for.\n\n\
User request: {user_query}\n\
Skill instruction: {skill_instruction}\n\n\
Search query:"
        );
        match self.rewrite_completion(&prompt) {
            Ok(result) => {
                let result = Self::clean_rewrite(&result);
                if !result.is_empty() {
                    tracing::info!(target: LOG_TARGET, "[skill-search] Rewrote query: {user_query:?} → {result:?}");
                    return result;
                }
            }
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "[skill-search] Query rewrite failed: {e}");
            }
        }
        user_query.to_string()
    }

    /// Resolve references in the search query using conversation history.
    ///
    /// Whenever the session has prior turns, history is consulted to resolve
    /// references (pronouns, omitted subjects, definite-article follow-ups) in
    /// the latest message.  The rewrite prompt returns a self-contained
    /// message unchanged, so there is no semantic gate here.  The history is
    /// expected to be pre-filtered for relevance by the caller.
    pub fn build_history_search_query(
        &self,
        user_query: &str,
        conversation_history: &[HashMap<String, String>],
    ) -> String {
        if conversation_history.len() <= 1 {
            return user_query.to_string();
        }

        let history_text = conversation_history
            .iter()
            .filter_map(|turn| {
                let content = turn.get("content").filter(|c| !c.is_empty())?;
                let role = turn.get("role").map(String::as_str).unwrap_or_default();
                Some(format!("{role}: {}", content.chars().take(200).collect::<String>()))
            })
            .collect::<Vec<_>>()
            .join("\n");
        if history_text.trim().is_empty() {
            return user_query.to_string();
        }

        let prompt = format!(
            "Rewrite the user's latest message into a short document-search \
query (max 15 words).\n\
- If the latest message is already self-contained, return it \
UNCHANGED.\n\
- Use the conversation history ONLY to resolve references in the \
latest message (pronouns like 'it'/'that'/'those', or an omitted \
subject).\n\
- Do NOT add names, entities, or topics from the history that the \
latest message does not refer to. The latest message defines what \
to search for; history is only background context.\n\n\
Conversation history:\n{history_text}\n\n\
Latest message: {user_query}\n\n\
Search query:"
        );
        match self.rewrite_completion(&prompt) {
            Ok(result) => {
                let result = Self::clean_rewrite(&result);
                if !result.is_empty() {
                    tracing::info!(
                        target: LOG_TARGET,
                        "[history-search] Rewrote query: {user_query:?} → {result:?}"
                    );
                    return result;
                }
            }
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "[history-search] Query rewrite failed: {e}");
            }
        }
        user_query.to_string()
    }

    // ------------------------------------------------------------ token sizing

    /// Count tokens in `text` with the loaded LLM's tokenizer (0 when none).
    pub fn count_tokens(&self, text: &str) -> usize {
        if let Some(remote) = self.remote_device() {
            // A child has no tokenizer of its own; the host's is the only one
            // that matches the model the prompt will actually be run on.
            return remote.count_tokens(text).unwrap_or(0);
        }
        match self.llm.read().clone() {
            Some(llm) => llm.get_num_tokens(text),
            None => 0,
        }
    }

    /// The model's maximum context window size in tokens.
    pub fn get_max_context_tokens(&self) -> usize {
        if let Some(remote) = self.remote_device() {
            return match remote.max_context_tokens() {
                Ok(0) | Err(_) => DEFAULT_MAX_CONTEXT_TOKENS,
                Ok(n) => n,
            };
        }
        match self.llm.read().clone() {
            Some(llm) => llm.get_max_tokens(),
            None => DEFAULT_MAX_CONTEXT_TOKENS,
        }
    }

    /// Max context window of the model that will synthesize the final response.
    ///
    /// The default model's window is often larger than the capability model
    /// that actually consumes the prompt.  Sizing the response context
    /// against it lets web/document context overflow the response model.
    pub fn get_response_max_context_tokens(&self, capability: Option<&str>) -> usize {
        match self.llm_for_response(capability) {
            Some(llm) => llm.get_max_tokens(),
            None => self.get_max_context_tokens(),
        }
    }

    /// Whether this service can serve a request right now.
    ///
    /// With the device in another process, that is two questions and both
    /// have to hold: the host must have weights, *and* this process must hold
    /// a handle to them.
    pub fn is_loaded(&self) -> bool {
        if let Some(remote) = self.remote_device() {
            if self.llm.read().is_none() {
                return false;
            }
            return remote.is_loaded().unwrap_or(false);
        }
        self.llm.read().is_some()
    }

    /// Human-readable name for a loaded LLM, from its GGUF metadata.
    ///
    /// Falls back to the class name when metadata is unavailable so the
    /// response log always names *something* concrete.  The metadata name is
    /// mapped to the product's own so logs and events name the model the user
    /// sees in the picker.
    pub fn model_label(llm: &dyn BaseLLM) -> String {
        // A handle to a model in another process carries the label its owner
        // reports; there is no local GGUF to read metadata from.
        let explicit = llm.model_label();
        if !explicit.is_empty() {
            return explicit;
        }
        if let Some(model) = llm.llama_model() {
            let meta = model.metadata();
            let name = display_model_label(meta.get("general.name").map(String::as_str).unwrap_or(""));
            let size = meta.get("general.size_label").cloned().unwrap_or_default();
            if !name.is_empty() {
                return if size.is_empty() { name } else { format!("{name} ({size})") };
            }
        }
        // Python: `type(llm).__name__`.
        llm.class_name().to_string()
    }

    /// Name the model that will actually write the response.
    ///
    /// Deliberately the *same* resolution the response itself goes through —
    /// `llm_for_response` — and not a cheaper approximation: this label goes
    /// on the progress event a user reads, so it has to name the model that
    /// answers.
    pub fn response_model_label(&self, capability: Option<&str>) -> String {
        match self.llm_for_response(capability) {
            Some(llm) => Self::model_label(llm.as_ref()),
            // `type(None).__name__`.
            None => "NoneType".to_string(),
        }
    }

    /// Whether the final answer comes from a remote model.
    ///
    /// Resolved the way the answer is (`general`), so a selection mixing
    /// local and remote models says remote only when no local model can
    /// answer.  No remote selection, no resolution.
    pub fn answers_remotely(&self) -> bool {
        let remote: Vec<String> = self
            .selected_models()
            .into_iter()
            .filter(|m| m.is_remote())
            .map(|m| m.model_name)
            .collect();
        if remote.is_empty() {
            return false;
        }
        let label = self.response_model_label(Some(ModelCapability::General.value()));
        remote.contains(&label)
    }

    /// Reports response-model selection, the same way the orchestrator does.
    ///
    /// The response model is resolved after orchestration finishes, by the
    /// streaming layer, so without this a cold load landed between the last
    /// orchestration event and the first token as dead air.  Routing it
    /// through [`ModelRouter`] means it inherits the same rules as every other
    /// resolution.  Held on the service and re-armed per request by
    /// [`ModelService::begin_response_request`].
    ///
    /// The resident model can change between requests (a model switch reloads
    /// it), so the router always points at the current one: when it moved,
    /// the router is rebuilt and this request's progress sink re-installed.
    pub fn response_router(&self) -> Arc<ModelRouter> {
        let llm = self.llm.read().clone().unwrap_or_else(|| UNLOADED_LLM.clone());
        let mut slot = self.response_router_instance.lock();
        if let Some(router) = slot.as_ref() {
            if same_llm(&router.orchestrator_llm, &llm) && same_llm(&router.agent_llm, &llm) {
                return router.clone();
            }
        }
        let rebuilt = slot.is_some();
        let weak = self.weak_self.clone();
        let provider_weak = weak.clone();
        let available_weak = weak.clone();
        let skip_weak = weak;
        let router = Arc::new(ModelRouter::new(
            llm.clone(),
            llm,
            ModelRouterOptions {
                capability_llm_provider: Some(Arc::new(move |cap: &str| {
                    Ok(provider_weak.upgrade().and_then(|svc| svc.get_llm_for_capability(cap)))
                })),
                capability_available: Some(Arc::new(move |cap: &str| {
                    Ok(available_weak.upgrade().map(|svc| svc.has_capability_model(cap)).unwrap_or(false))
                })),
                capability_skip_reason: Some(Arc::new(move || {
                    Ok(skip_weak.upgrade().and_then(|svc| svc.last_capability_skip()))
                })),
                ..Default::default()
            },
        ));
        if rebuilt {
            let callback = self.response_progress_callback.lock().clone();
            router.begin_request(callback);
        }
        *slot = Some(router.clone());
        router
    }

    /// Install this request's progress sink for response-model events.
    ///
    /// Mirrors [`ModelRouter::begin_request`], which the orchestrator calls
    /// for its own phase.  Clearing per request also resets the once-per-
    /// request dedupe, so a second question reports its model again.
    pub fn begin_response_request(&self, progress_callback: Option<ProgressCallback>) {
        *self.response_progress_callback.lock() = progress_callback.clone();
        self.response_router().begin_request(progress_callback);
    }

    /// Announce a pending response-model load *before* it blocks.
    ///
    /// Resolution happens inside a synchronous call the streaming generator
    /// cannot interleave with, so the warning has to be emitted ahead of it.
    /// Only when something could actually load.
    pub fn announce_response_model(&self, capability: Option<&str>) {
        let pinned = self.pinned_model(capability).is_some();
        let available = || capability.map(|c| self.has_capability_model(c)).unwrap_or(false);
        if pinned || available() {
            self.response_router()
                .emit_model_loading(capability, Some(Self::RESPONSE_PURPOSE));
        }
    }

    /// Name the model that will write the answer.
    ///
    /// Silent when the always-resident router model answers and no capability
    /// was asked for — nothing was loaded and nothing was overridden.  When a
    /// capability *was* asked for and we still landed on the router model,
    /// that is a fallback and says why.
    pub fn report_response_model(&self, capability: Option<&str>, llm: Option<&Arc<dyn BaseLLM>>) {
        let Some(llm) = llm else {
            return;
        };
        let resident = self.llm.read().clone();
        let is_resident = resident.as_ref().map(|r| same_llm(r, llm)).unwrap_or(false);
        if !is_resident {
            self.response_router()
                .emit_model_selected(capability, llm.as_ref(), Some(Self::RESPONSE_PURPOSE));
        } else if capability.is_some() {
            self.response_router()
                .emit_model_fallback(capability, llm.as_ref(), Some(Self::RESPONSE_PURPOSE));
        }
    }
}
