//! Port of `smartloop/api/services/model_service.py`.
//!
//! Singleton model service for managing LLM lifecycle.  This file holds the
//! module-level helpers, [`ProjectCaches`], the [`ModelService`] struct with all
//! of its state, and every method the Python defined before
//! `_llm_for_response`; the remaining methods live in [`part2`] as a second
//! `impl ModelService` block.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

mod part2;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use once_cell::sync::{Lazy, OnceCell};
use parking_lot::{Condvar, Mutex, RwLock};
use serde_json::{Map, Value, json};

use crate::agents::hybrid_executor::LLMExecutor;
use crate::agents::orchestration::model_router::{CapabilitySkipReason, ModelRouter, ProgressCallback};
use crate::agents::planner::{FETCH_SERVER_ID, SYSTEM_TOOLS_SERVER_ID, SYSTEM_TOOLS_SERVER_NAME};
use crate::api::config::api_settings;
use crate::config::app_settings::AppSettings;
use crate::config::model_config::ModelConfig;
use crate::config::SUPPORTED_MODELS;
use crate::constants::{canonical_model_name, display_model_label};
use crate::conversation_store::{ConversationStore, MessageInput};
use crate::db::feedback_indexer::FeedbackIndexer;
use crate::db::store::{ChunkResult, VectorStore};
use crate::db::store_factory::StoreFactory;
use crate::db::workspace_indexer::{WORKSPACE_COLLECTION, WorkspaceIndexer};
use crate::feedback_store::{ConversationTurn, FeedbackEntry, FeedbackStore};
use crate::llms::base_llm::BaseLLM;
use crate::llms::thinking::strip_thinking;
use crate::model::model_manager::Model;
use crate::model::slp_model::TrainProgressCallback;
use crate::model_host::client::{ModelClient, ModelHostError};
use crate::tasks::TaskQueue;
use crate::types::ModelCapability;
use crate::utils::normalize_path;
use crate::workspace::{
    DocumentInfo, DocumentStatus, MCPServer, MCPServerConfig, MCPServerStatus, MCPServerType,
    MCPToolDefinition, Project, RemoteMCPServerConfig, Skill, Workspace, WorkspaceModel,
    pinned_project_id, use_project,
};

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

/// Broad initial recall before selection (shared document/workspace criteria).
pub const RECALL_MAX_DISTANCE: f64 = 0.8;
/// Quality floor: reject a weak best match.
pub const MIN_BEST_DISTANCE: f64 = 0.55;
/// Keep chunks within this margin of the best.
pub const DISTANCE_CUTOFF_MARGIN: f64 = 0.15;
/// How many matched chunks a citation carries for the router to verify against.
pub const MAX_CITATION_CHUNKS: usize = 5;

/// Apply the shared relevance-selection criteria to ranked chunks.
///
/// `chunks` must be ordered closest-first (ascending distance).  Returns the
/// kept chunks, or `[]` when the best match fails the quality floor.  Shared by
/// document and workspace retrieval so both apply the same thresholds.
pub fn select_relevant_chunks(chunks: Vec<ChunkResult>, label: &str) -> Vec<ChunkResult> {
    let _ = (chunks, label);
    todo!()
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
        todo!()
    }

    /// Return a slot.
    pub fn release(&self) {
        todo!()
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

/// Overridable seams standing in for the Python tests' monkeypatching of
/// methods on the instance.  Each `None` runs the real implementation.
#[derive(Clone, Default)]
pub struct ModelServiceHooks {
    // ---- model_service_b seams (consulted by the methods in `part2`) ----
    /// Replaces `_pinned_model(capability)` where `part2` calls it.
    pub pinned_model: Option<Arc<dyn Fn(Option<&str>) -> Option<WorkspaceModel> + Send + Sync>>,
    /// Replaces `_pinned_llm(capability)` where `part2` calls it.
    pub pinned_llm: Option<Arc<dyn Fn(Option<&str>) -> Option<Arc<dyn BaseLLM>> + Send + Sync>>,
    /// Replaces `_get_llm_for_capability(capability)` where `part2` calls it.
    pub get_llm_for_capability: Option<Arc<dyn Fn(&str) -> Option<Arc<dyn BaseLLM>> + Send + Sync>>,
    /// Replaces `_get_large_llm()` where `part2` calls it.
    pub get_large_llm: Option<Arc<dyn Fn() -> Option<Arc<dyn BaseLLM>> + Send + Sync>>,
    /// Replaces `_llm_for_response(capability)`.
    pub llm_for_response: Option<Arc<dyn Fn(Option<&str>) -> Option<Arc<dyn BaseLLM>> + Send + Sync>>,
    /// Replaces `_setup_project_tools(project)` where `create_project` calls it
    /// (receives the project id).
    pub setup_project_tools: Option<Arc<dyn Fn(&str) -> anyhow::Result<()> + Send + Sync>>,
    // ---- end model_service_b seams ----
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
    /// Replaces `load_model()` (the API tests' `model_service.load_model =
    /// MagicMock(side_effect=...)`).
    pub load_model: Option<Arc<dyn Fn() -> anyhow::Result<Arc<dyn BaseLLM>> + Send + Sync>>,
    /// Replaces `add_local_mcp_server(...)`; receives `(command, project_id)`.
    pub add_local_mcp_server: Option<Arc<dyn Fn(&str, Option<&str>) -> anyhow::Result<Value> + Send + Sync>>,
}

/// The process-wide singleton (`ModelService._instance`).
static INSTANCE: OnceCell<Arc<ModelService>> = OnceCell::new();

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

    /// A fresh, fully initialised instance that is *not* the singleton.
    pub fn new_detached() -> Arc<ModelService> {
        todo!()
    }

    /// An instance with every field at its empty default and no conversation
    /// store — the Rust analogue of `object.__new__(ModelService)` in tests.
    pub fn new_bare() -> Arc<ModelService> {
        todo!()
    }

    /// An owning handle to this instance.
    pub fn arc(&self) -> Arc<ModelService> {
        todo!()
    }

    /// Replace the test seams.
    pub fn set_hooks(&self, hooks: ModelServiceHooks) {
        *self.hooks.write() = hooks;
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

    // ------------------------------------------------------------ model client

    /// How this process reaches the generative model: in-process by default,
    /// or the model host named by `SLP_MODEL_HOST`.  Resolved once.
    pub fn model_client(&self) -> Result<Arc<dyn ModelClient>, ModelHostError> {
        todo!()
    }

    /// Swap in a model client (tests), or clear it so it is re-resolved.
    pub fn set_model_client(&self, client: Option<Arc<dyn ModelClient>>) {
        *self.model_client_slot.lock() = client;
    }

    /// The model client when the weights live in another process, else `None`.
    pub fn remote_device(&self) -> Option<Arc<dyn ModelClient>> {
        todo!()
    }

    /// A handle to the host's model for `capability`.
    pub fn remote_llm(&self, capability: Option<&str>) -> Arc<dyn BaseLLM> {
        let _ = capability;
        todo!()
    }

    // ------------------------------------------------------- per-project caches

    /// Return (creating if needed) the cache slot for `project_id`.
    pub fn caches(&self, project_id: &str) -> Arc<Mutex<ProjectCaches>> {
        let _ = project_id;
        todo!()
    }

    /// Discard cached stores for `project_id`, or for every project.
    pub fn drop_caches(&self, project_id: Option<&str>) {
        let _ = project_id;
        todo!()
    }

    // ------------------------------------------------------- per-model cache slots

    /// The `_cap_llm_<model_name>` slot, if set.
    pub fn cap_llm(&self, model_name: &str) -> Option<CachedLlm> {
        let _ = model_name;
        todo!()
    }

    /// Set (`Some`) or clear (`None`) the `_cap_llm_<model_name>` slot.
    pub fn set_cap_llm(&self, model_name: &str, value: Option<CachedLlm>) {
        let _ = (model_name, value);
        todo!()
    }

    // ---------------------------------------------------------------- sessions

    /// Get chat history for a session, loading from disk on first access.
    pub fn get_session_history(&self, session_id: &str) -> Vec<MessageInput> {
        let _ = session_id;
        todo!()
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
        let _ = (session_id, user_query, max_turns);
        todo!()
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
        let _ = (session_id, role, content, citations, message_id);
        todo!()
    }

    // --------------------------------------------------------------- workspace

    /// Ensure the default orchestrator model is initialized.
    pub fn ensure_model(&self) -> anyhow::Result<()> {
        todo!()
    }

    /// Return the workspace, creating it if necessary.
    pub fn ensure_workspace(&self, model_config: Option<ModelConfig>) -> anyhow::Result<Arc<RwLock<Workspace>>> {
        let _ = model_config;
        todo!()
    }

    /// The workspace instance this process actually serves from (never a
    /// detached copy).  Use anywhere a project is created or adopted.
    pub fn live_workspace(&self) -> anyhow::Result<Arc<RwLock<Workspace>>> {
        todo!()
    }

    /// Pick up a child's write to `project_id`'s own file, if there is one.
    pub fn refresh_project_state(&self, project_id: Option<&str>) -> bool {
        let _ = project_id;
        todo!()
    }

    /// Ensure the workspace is loaded, raising if it cannot be resolved.
    pub fn initialize_workspace(&self) -> anyhow::Result<()> {
        todo!()
    }

    /// Find a project by ID (re-reading the workspace from disk once on a
    /// miss), raising `ValueError("Project not found: …")`.  Returns a snapshot.
    pub fn get_project(&self, project_id: &str) -> anyhow::Result<Project> {
        let _ = project_id;
        todo!()
    }

    /// Drop the obsolete built-in `__fetch__` MCP server from a project.
    fn sweep_legacy_fetch_server(&self, project_id: &str) {
        let _ = project_id;
        todo!()
    }

    // ------------------------------------------------------------------ models

    /// Load and cache the orchestrator LLM for the current project.
    pub fn load_model(&self) -> anyhow::Result<Arc<dyn BaseLLM>> {
        todo!()
    }

    /// Load a secondary small model for decision nodes.
    pub fn load_decision_model(&self, model_name: &str) {
        let _ = model_name;
        todo!()
    }

    /// Return the large (agent) model, loading it lazily on first use.
    pub fn get_large_llm(&self) -> Option<Arc<dyn BaseLLM>> {
        todo!()
    }

    /// Return `(main_gguf, mmproj_gguf)` found directly in `model_dir`.
    pub fn find_model_ggufs(model_dir: &str) -> (Option<String>, Option<String>) {
        let _ = model_dir;
        todo!()
    }

    /// Return a loaded *multimodal* vision LLM, or `None` when unavailable.
    /// `on_model_selected(model_name)` fires right before the chosen model loads.
    pub fn get_vision_llm(&self, on_model_selected: Option<&dyn Fn(&str)>) -> Option<Arc<dyn BaseLLM>> {
        let _ = on_model_selected;
        todo!()
    }

    /// Sort candidates by `(size, cap-overlap, version)`; `descending` walks
    /// largest → smallest.
    pub fn ranked_candidates(
        candidates: Vec<WorkspaceModel>,
        ref_caps: &[String],
        descending: bool,
    ) -> Vec<WorkspaceModel> {
        let _ = (candidates, ref_caps, descending);
        todo!()
    }

    /// Sort candidates smallest-first for a routed step; size ties go to
    /// broader capability coverage, then the newer version.
    pub fn lightest_first(candidates: Vec<WorkspaceModel>, ref_caps: &[String]) -> Vec<WorkspaceModel> {
        let _ = (candidates, ref_caps);
        todo!()
    }

    /// The models the user has chosen — enabled *and* downloaded, excluding
    /// the bootstrap/orchestrator model.
    pub fn selected_models(&self) -> Vec<WorkspaceModel> {
        todo!()
    }

    /// The sole selected model, or `None` when it cannot serve the step.
    pub fn pinned_model(&self, capability: Option<&str>) -> Option<WorkspaceModel> {
        let _ = capability;
        todo!()
    }

    /// Loaded instance of the pinned model, or `None`.
    pub fn pinned_llm(&self, capability: Option<&str>) -> Option<Arc<dyn BaseLLM>> {
        let _ = capability;
        todo!()
    }

    /// `(model_name, cause, detail)` for the last capability model that could
    /// not be loaded, or `None` when the last resolution succeeded.
    pub fn last_capability_skip(&self) -> Option<CapabilitySkipReason> {
        todo!()
    }

    /// `"needs ~4.9GB, 4.4GB free"` from the last failed fit check.
    pub fn fit_shortfall_detail(&self) -> Option<String> {
        todo!()
    }

    /// Return a resident (or freshly loaded) instance of one workspace model,
    /// or `None` when it has no GGUF, faulted, or won't fit.
    pub fn load_workspace_llm(
        &self,
        ws_model: &WorkspaceModel,
        reason: &str,
        reserve_gb: Option<f64>,
    ) -> Option<Arc<dyn BaseLLM>> {
        let _ = (ws_model, reason, reserve_gb);
        todo!()
    }

    /// Whether `model_name` should reason, per the picker's setting.
    pub fn thinking_for(&self, model_name: &str) -> Option<bool> {
        let _ = model_name;
        todo!()
    }

    /// Push a just-changed reasoning setting onto a resident model.  Returns
    /// true when a resident instance was updated.
    pub fn apply_thinking_setting(&self, model_name: &str) -> bool {
        let _ = model_name;
        todo!()
    }

    /// `(model_name, llm)` for every resident model that can reason.
    fn loaded_thinking_models(&self) -> Vec<(String, Arc<dyn BaseLLM>)> {
        todo!()
    }

    /// The capability's value, for logs and messages.
    pub fn cap_name(capability: &str) -> String {
        capability.to_string()
    }

    /// Whether the model that would answer for `capability` can reason.
    pub fn supports_thinking(&self, capability: Option<&str>) -> bool {
        let _ = capability;
        todo!()
    }

    /// True when some selected model declares `capability` (loads nothing).
    pub fn has_capability_model(&self, capability: &str) -> bool {
        let _ = capability;
        todo!()
    }

    /// Return a loaded LLM matching the requested capability.
    pub fn get_llm_for_capability(&self, capability: &str) -> Option<Arc<dyn BaseLLM>> {
        let _ = capability;
        todo!()
    }

    /// Return `(free_gb, total_gb)` for the active accelerator, or `None`
    /// (only Apple Silicon unified memory reports a fit budget).
    pub fn device_free_total_gb(&self) -> Option<(f64, f64)> {
        todo!()
    }

    /// True if a large model's capped footprint fits in free memory.
    pub fn large_model_fits(&self, gguf_path: &str, cap: i64, label: &str, reserve_gb: Option<f64>) -> bool {
        let _ = (gguf_path, cap, label, reserve_gb);
        todo!()
    }

    /// Decide whether the vision CLIP/mmproj encoder must run on CPU.
    pub fn clip_on_cpu(&self, gguf_path: &str, cap: i64) -> bool {
        let _ = (gguf_path, cap);
        todo!()
    }

    /// Load the largest downloaded model for agent/skill tasks, if it fits.
    pub fn detect_large_model(&self) {
        todo!()
    }

    /// Pre-load the embedding model in a background thread.
    fn warmup_embeddings(&self) {
        todo!()
    }

    /// Run a trivial inference to prime the KV cache after model load.
    fn warmup_inference(&self) {
        todo!()
    }

    /// The decision LLM, or `None` to fall back to the main model.
    pub fn decision_llm(&self) -> Option<Arc<dyn BaseLLM>> {
        self.decision_llm.read().clone()
    }

    // ----------------------------------------------------------- search/feedback

    /// Search the workspace-files collection with the shared selection criteria.
    pub fn search_workspace(
        &self,
        query: &str,
        work_dir: &str,
        n_results: usize,
        conversation_history: Option<&[HashMap<String, String>]>,
    ) -> Vec<ChunkResult> {
        let _ = (query, work_dir, n_results, conversation_history);
        todo!()
    }

    /// The feedback index for `project`, or for whatever is bound.
    pub fn ensure_feedback_indexer(&self, project: Option<&Project>) -> anyhow::Result<Arc<FeedbackIndexer>> {
        let _ = project;
        todo!()
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
        let _ = (session_id, message_id, project_id, rating);
        todo!()
    }

    /// `message_id -> rating` for one session in a project.
    pub fn get_feedback_ratings(&self, session_id: &str, project_id: &str) -> anyhow::Result<HashMap<String, String>> {
        let _ = (session_id, project_id);
        todo!()
    }

    /// Delete feedback for a specific message and remove it from the index.
    pub fn delete_feedback_ratings(
        &self,
        session_id: &str,
        message_id: &str,
        project_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let _ = (session_id, message_id, project_id);
        todo!()
    }

    /// Search stored feedback Q&A using the shared relevance criteria.
    pub fn search_feedback(&self, query: &str, n_results: usize) -> Vec<ChunkResult> {
        let _ = (query, n_results);
        todo!()
    }

    // ------------------------------------------------------------ build/train

    /// Build the GGUF model (no download step).
    pub fn build_model(
        &self,
        model_name: Option<&str>,
        quantize: Option<&str>,
        developer_token: Option<&str>,
        progress_callback: Option<TrainProgressCallback>,
    ) -> anyhow::Result<()> {
        let _ = (model_name, quantize, developer_token, progress_callback);
        todo!()
    }

    /// Run the full training pipeline.
    pub fn train_model(
        &self,
        model_name: Option<&str>,
        progress_callback: Option<TrainProgressCallback>,
    ) -> anyhow::Result<()> {
        let _ = (model_name, progress_callback);
        todo!()
    }

    /// Get the currently loaded LLM instance.
    pub fn get_llm(&self) -> Option<Arc<dyn BaseLLM>> {
        self.llm.read().clone()
    }

    /// Get the current project name.
    pub fn get_current_project_name(&self) -> Option<String> {
        todo!()
    }

    /// Get the current project ID.
    pub fn get_project_id(&self) -> Option<String> {
        todo!()
    }

    // ------------------------------------------------------------ project tools

    /// Register (or refresh) the System Tools MCP server on a project.
    pub fn install_system_tools_server(&self, project_id: &str) -> anyhow::Result<()> {
        let _ = project_id;
        todo!()
    }

    /// Remove the legacy fetch MCP server — fetching is now the FetchAgent.
    pub fn remove_fetch_server(&self, project_id: &str) -> anyhow::Result<()> {
        let _ = project_id;
        todo!()
    }

    /// Delete leftover fetch skill directories from disk.
    fn remove_stale_fetch_skills(&self, project: &Project) {
        let _ = project;
        todo!()
    }

    /// Migrate legacy REMOTE system servers to LOCAL (stdio subprocess) and fix
    /// empty launcher args.
    pub fn migrate_servers_to_local(&self, project_id: &str) -> anyhow::Result<()> {
        let _ = project_id;
        todo!()
    }

    /// True for built-in tool skills and `__`-prefixed internal skills.
    pub fn is_system_skill(name: &str) -> bool {
        !name.is_empty() && (Self::SYSTEM_SKILL_NAMES.contains(&name) || name.starts_with("__"))
    }

    /// Install system MCP servers and register default skills.
    pub fn setup_project_tools(&self, project_id: &str) -> anyhow::Result<()> {
        let _ = project_id;
        todo!()
    }

    /// Write default skill SKILL.md files if they don't already exist.
    fn install_default_skills(&self, project: &Project) -> anyhow::Result<()> {
        let _ = project;
        todo!()
    }

    /// Return the system project (general_chat), if it exists.
    pub fn get_system_project(&self) -> Option<Project> {
        todo!()
    }

    // ------------------------------------------------------------------ skills

    /// Get project skills for the current project (read from disk).
    pub fn get_project_skills(&self) -> Vec<Skill> {
        todo!()
    }

    /// Add a skill to a project and persist as SKILL.md; returns the skill dict.
    pub fn add_skill(
        &self,
        project_id: &str,
        content: &str,
        name: Option<&str>,
        description: Option<&str>,
    ) -> anyhow::Result<Value> {
        let _ = (project_id, content, name, description);
        todo!()
    }

    /// Split raw SKILL.md text into `(body, frontmatter_fields)`.
    pub fn parse_skill_content(raw: &str) -> (String, Map<String, Value>) {
        let _ = raw;
        todo!()
    }

    /// Update a skill on disk from full SKILL.md text; returns the skill dict.
    pub fn update_skill(&self, project_id: &str, skill_id: &str, content: &str) -> anyhow::Result<Value> {
        let _ = (project_id, skill_id, content);
        todo!()
    }

    /// Extract a 1-2 word skill name from content as a fallback.
    pub fn extract_skill_name(content: &str) -> String {
        let _ = content;
        todo!()
    }

    /// Return the project's documents directory (under `project_dir`).
    pub fn project_documents_dir(project: &Project) -> String {
        let _ = project;
        todo!()
    }

    /// Render a Skill as full SKILL.md text with YAML frontmatter.
    pub fn render_skill_content(skill: &Skill) -> String {
        let _ = skill;
        todo!()
    }

    /// Write a SKILL.md file with YAML frontmatter for the given skill.
    pub fn write_skill_md(project: &Project, skill: &Skill) -> std::io::Result<()> {
        let _ = (project, skill);
        todo!()
    }

    /// Return the skills directory for a project (under `project_dir`).
    pub fn skills_dir(project: &Project) -> String {
        let _ = project;
        todo!()
    }

    /// Read all skills from the project skills directory.
    pub fn read_skills_from_disk(project: &Project) -> Vec<Skill> {
        let _ = project;
        todo!()
    }

    /// Get a single skill by name from disk.
    pub fn get_skill(&self, project_id: &str, skill_id: &str) -> anyhow::Result<Value> {
        let _ = (project_id, skill_id);
        todo!()
    }

    /// List a project's user-created skills, read from disk.
    pub fn list_skills(&self, project_id: &str) -> anyhow::Result<Vec<Value>> {
        let _ = project_id;
        todo!()
    }

    /// Delete a skill by removing its SKILL.md directory from disk.
    pub fn delete_skill(&self, project_id: &str, skill_id: &str) -> anyhow::Result<Value> {
        let _ = (project_id, skill_id);
        todo!()
    }

    // ---------------------------------------------------------------- context

    /// Build a project info summary for context injection.
    pub fn get_project_info(&self) -> Option<String> {
        todo!()
    }

    /// Vector search on indexed documents; the joined context string or `None`.
    pub fn get_document_context(&self, query: Option<&str>) -> Option<String> {
        let _ = query;
        todo!()
    }

    /// Core vector search → `(doc_groups, citations)` or `None`.  `doc_groups`
    /// is an ordered `(document name, combined source text)` list.
    pub fn vector_search_grouped(
        &self,
        query: Option<&str>,
        skill_instruction: Option<&str>,
        conversation_history: Option<&[HashMap<String, String>]>,
    ) -> Option<(Vec<(String, String)>, Vec<Value>)> {
        let _ = (query, skill_instruction, conversation_history);
        todo!()
    }

    /// Vector search → `(context_str, citations_list)` or `None`.
    pub fn get_document_context_with_citations(
        &self,
        query: Option<&str>,
        skill_instruction: Option<&str>,
        conversation_history: Option<&[HashMap<String, String>]>,
    ) -> Option<(String, Vec<Value>)> {
        let _ = (query, skill_instruction, conversation_history);
        todo!()
    }

    /// Vector search → `(documents, citations)` for fan-out, where each
    /// document is `{"document_name", "content"}`.
    pub fn get_document_groups(
        &self,
        query: Option<&str>,
        skill_instruction: Option<&str>,
        conversation_history: Option<&[HashMap<String, String>]>,
    ) -> Option<(Vec<Value>, Vec<Value>)> {
        let _ = (query, skill_instruction, conversation_history);
        todo!()
    }

    /// Use the LLM to synthesize a vector-search query from the user message
    /// and the matched skill instruction.
    pub fn build_skill_search_query(&self, user_query: &str, skill_instruction: &str) -> String {
        let _ = (user_query, skill_instruction);
        todo!()
    }

    /// Resolve references in the search query using conversation history.
    pub fn build_history_search_query(
        &self,
        user_query: &str,
        conversation_history: &[HashMap<String, String>],
    ) -> String {
        let _ = (user_query, conversation_history);
        todo!()
    }

    // ------------------------------------------------------------ token sizing

    /// Count tokens in `text` with the loaded LLM's tokenizer (0 when none).
    pub fn count_tokens(&self, text: &str) -> usize {
        let _ = text;
        todo!()
    }

    /// The model's maximum context window size in tokens.
    pub fn get_max_context_tokens(&self) -> usize {
        todo!()
    }

    /// Max context window of the model that will synthesize the final response.
    pub fn get_response_max_context_tokens(&self, capability: Option<&str>) -> usize {
        let _ = capability;
        todo!()
    }

    /// Whether this service can serve a request right now.
    pub fn is_loaded(&self) -> bool {
        todo!()
    }

    /// Human-readable name for a loaded LLM, from its GGUF metadata.
    pub fn model_label(llm: &dyn BaseLLM) -> String {
        let _ = llm;
        todo!()
    }

    /// Name the model that will actually write the response.
    pub fn response_model_label(&self, capability: Option<&str>) -> String {
        let _ = capability;
        todo!()
    }

    /// Reports response-model selection, the same way the orchestrator does.
    pub fn response_router(&self) -> Arc<ModelRouter> {
        todo!()
    }

    /// Install this request's progress sink for response-model events.
    pub fn begin_response_request(&self, progress_callback: Option<ProgressCallback>) {
        let _ = progress_callback;
        todo!()
    }

    /// Announce a pending response-model load *before* it blocks.
    pub fn announce_response_model(&self, capability: Option<&str>) {
        let _ = capability;
        todo!()
    }

    /// Name the model that will write the answer.
    pub fn report_response_model(&self, capability: Option<&str>, llm: Option<&Arc<dyn BaseLLM>>) {
        let _ = (capability, llm);
        todo!()
    }
}
