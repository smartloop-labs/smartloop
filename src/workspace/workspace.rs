//! Port of `smartloop/workspace/workspace.py`.
//!
//! The workspace model: registered models, projects, MCP servers, and their
//! on-disk persistence (`config.json` workspace record plus one
//! `<project_dir>/project.json` per project).
//!
//! # Ownership model in Rust
//!
//! The process-wide workspace singleton is a [`SharedWorkspace`]
//! (`Arc<RwLock<Workspace>>`), returned by [`Workspace::init`] and
//! [`Workspace::get_instance`].  Records are plain owned values inside it;
//! background passes that would have held a Python object across a reload hold
//! the record's id instead and re-find it under the lock.  [`Workspace::reload`]
//! and [`Workspace::refresh_project`] still update records *in place* (merge by
//! id) rather than replacing the lists wholesale.
//!
//! Lookups that Python returned as a live (mutable) object come in two forms:
//! a read form returning an owned snapshot (e.g. [`Workspace::get_model`]) and
//! a `_mut` form returning `&mut` into the workspace (e.g.
//! [`Workspace::get_model_mut`]).  Per-project mutators take the project id so
//! they can borrow the workspace mutably.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

use crate::config::{AppSettings, ModelConfig, SUPPORTED_MODELS};
use crate::constants::canonical_model_name;
use crate::paths::expand_home;
use crate::types::ModelCapability;
use crate::workspace::context::{
    active_project_id, owned_project_id, pinned_project_id, projects_delegated,
};
pub use crate::workspace::types::{Document, DocumentInfo, DocumentStatus, Skill};

/// Logger name used by the Python module.
const LOG_TARGET: &str = "smartloop.workspace.workspace";

/// The process-wide shared workspace handle.
pub type SharedWorkspace = Arc<RwLock<Workspace>>;

/// Filename of a project's own state, inside its `project_dir`.
pub const PROJECT_STATE_FILE: &str = "project.json";

/// Schema version stamped into `project.json`, so a future layout change can
/// recognise what it is reading rather than guessing from the shape.
pub const PROJECT_STATE_VERSION: u32 = 1;

/// Fields `config.json` stays authoritative for even when `project.json` also
/// carries them.  Selection is a property of the workspace — which project the
/// UI is looking at — not of any one project, and only the supervisor writes it.
pub const WORKSPACE_OWNED_PROJECT_FIELDS: &[&str] = &["current"];

/// The thin per-project record kept in `config.json`.  Enough to list and
/// route projects without opening any of them.
pub const PROJECT_INDEX_FIELDS: &[&str] =
    &["id", "name", "description", "project_dir", "current", "system"];

/// Import-time `AppSettings().home_dir` (Python's module-level `_settings`).
/// Overridable for tests via [`set_settings_home_dir`].
static SETTINGS_HOME_DIR: Lazy<RwLock<Option<String>>> = Lazy::new(|| RwLock::new(None));

/// Assignment override for `Workspace.config_file` (tests patch it).
static CONFIG_FILE_OVERRIDE: Lazy<RwLock<Option<String>>> = Lazy::new(|| RwLock::new(None));

/// The cached singleton (`Workspace._instance`), guarded by `Workspace._lock`.
static INSTANCE: Lazy<Mutex<Option<SharedWorkspace>>> = Lazy::new(|| Mutex::new(None));

/// Serializes `save()`: background tasks each persist from their own thread.
static SAVE_LOCK: Mutex<()> = Mutex::new(());

/// Override the import-time settings home (Python tests patch
/// `workspace._settings`).  `None` restores `AppSettings::new().home_dir`.
pub fn set_settings_home_dir(home_dir: Option<String>) {
    todo!()
}

/// The framework home, resolved now rather than at import.
///
/// `SLP_HOME` (expanded) when set, else the import-time settings home.
fn home_dir() -> String {
    todo!()
}

/// True when the workspace should be kept as a single `config.json`.
///
/// The rollback path for the split: `SLP_LEGACY_CONFIG` in `1/true/yes`.
pub fn legacy_layout() -> bool {
    todo!()
}

/// Write `data` to `path` so a reader sees the old file or the new one.
///
/// Writes a sibling `<path>.tmp` (4-space indented JSON), fsyncs it, then
/// renames over the target.
pub fn atomic_write_json(path: &str, data: &Value) -> std::io::Result<()> {
    todo!()
}

/// Where a project's own state lives.
pub fn project_state_path(project_dir: &str) -> String {
    todo!()
}

/// Return a project's persisted record (the `"project"` object), or `None`.
///
/// Never fails: a missing, empty or corrupt file yields `None`.
pub fn read_project_state(project_dir: &str) -> Option<Map<String, Value>> {
    todo!()
}

/// A record that can be reconciled by id across a reload (Python's
/// `_carry_fields` / `_merge_records_by_id` over pydantic models).
pub trait MergeById: Sized {
    /// The record's id, or `None` when it has none (then lists of it are taken
    /// wholesale).
    fn record_id(&self) -> Option<&str>;

    /// Copy every field of `source` onto `self`, merging nested id-keyed
    /// lists in place.
    fn carry_fields(&mut self, source: Self);
}

/// Reconcile two lists of id-keyed records, keeping the records we hold.
///
/// The result is in `fresh` order — disk decides what exists and in what
/// order.  A record that is still there keeps its existing value with its
/// fields brought up to date; one that is new is taken as it came; one that
/// has gone is dropped.  If no held record has an id, `fresh` is taken
/// wholesale.
pub fn merge_records_by_id<T: MergeById>(current: &mut Vec<T>, fresh: Vec<T>) {
    todo!()
}

/// True when a GGUF is a vision mmproj / CLIP projector (not main weights).
pub fn is_mmproj_gguf(filename: &str) -> bool {
    todo!()
}

fn default_true() -> bool {
    true
}

fn default_token_type() -> String {
    "Bearer".to_string()
}

/// Deserialize a model name through `canonical_model_name` (the pydantic
/// `_canonicalize_model_name` field validator).
fn deserialize_canonical_model_name<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = String::deserialize(deserializer)?;
    Ok(canonical_model_name(&raw))
}

/// Tool definition discovered from an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MCPToolDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<Map<String, Value>>,
}

/// Configuration for launching a local MCP server process.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MCPServerConfig {
    /// e.g. "npx", "uvx"
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Working directory; supports `~/`.
    #[serde(default)]
    pub cwd: Option<String>,
}

/// Where an MCP server runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MCPServerType {
    #[default]
    Local,
    Remote,
}

impl MCPServerType {
    /// The wire value.
    pub fn as_str(&self) -> &'static str {
        match self {
            MCPServerType::Local => "local",
            MCPServerType::Remote => "remote",
        }
    }
}

/// How a remote MCP server authenticates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MCPAuthType {
    Oauth,
    TokenUrl,
    Token,
    #[default]
    None,
}

impl MCPAuthType {
    /// The wire value.
    pub fn as_str(&self) -> &'static str {
        match self {
            MCPAuthType::Oauth => "oauth",
            MCPAuthType::TokenUrl => "token_url",
            MCPAuthType::Token => "token",
            MCPAuthType::None => "none",
        }
    }
}

/// Configuration for a remote MCP server.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct RemoteMCPServerConfig {
    pub server_url: String,
    #[serde(default)]
    pub auth_type: MCPAuthType,
    #[serde(default)]
    pub token_endpoint: Option<String>,
    #[serde(default)]
    pub authorization_endpoint: Option<String>,
    #[serde(default)]
    pub token: Option<String>,
    /// RFC 8707 resource indicator negotiated at connect time. Persisted
    /// because refresh requests must carry the same indicator as the original
    /// grant.
    #[serde(default)]
    pub resource: Option<String>,
}

/// OAuth credentials for a remote MCP server.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MCPOAuthCredentials {
    pub client_id: String,
    #[serde(default)]
    pub client_secret: Option<String>,
    #[serde(default)]
    pub access_token: Option<String>,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub expires_at: Option<f64>,
    #[serde(default = "default_token_type")]
    pub token_type: String,
    /// What the grant actually came back with (RFC 6749 §5.1), which need not
    /// be what was asked for.
    #[serde(default)]
    pub scope: Option<String>,
}

impl Default for MCPOAuthCredentials {
    fn default() -> Self {
        Self {
            client_id: String::new(),
            client_secret: None,
            access_token: None,
            refresh_token: None,
            expires_at: None,
            token_type: default_token_type(),
            scope: None,
        }
    }
}

/// Provisioning state of a registered MCP server.
///
/// Registration only persists the connection; discovery and metadata
/// enrichment run in the background, so a server is `Pending` until that pass
/// finishes.  `Ready` is the default so servers written before this field
/// existed are not re-queued on upgrade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MCPServerStatus {
    Pending,
    #[default]
    Ready,
    Failed,
}

impl MCPServerStatus {
    /// The wire value.
    pub fn as_str(&self) -> &'static str {
        match self {
            MCPServerStatus::Pending => "pending",
            MCPServerStatus::Ready => "ready",
            MCPServerStatus::Failed => "failed",
        }
    }
}

/// An MCP server registered to a project, persisted in the project's state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MCPServer {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub pkg: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Slug of the curated catalog entry this server was installed from.
    #[serde(default)]
    pub catalog_slug: Option<String>,
    #[serde(default)]
    pub config: Option<MCPServerConfig>,
    #[serde(default)]
    pub server_type: MCPServerType,
    #[serde(default)]
    pub remote_config: Option<RemoteMCPServerConfig>,
    #[serde(default)]
    pub oauth_credentials: Option<MCPOAuthCredentials>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub system: bool,
    #[serde(default)]
    pub tools: Vec<MCPToolDefinition>,
    #[serde(default)]
    pub server_info: Option<Map<String, Value>>,
    /// Background provisioning state (tool discovery, then metadata
    /// enrichment).
    #[serde(default)]
    pub status: MCPServerStatus,
    /// Human-readable reason the last provisioning pass failed, if it did.
    #[serde(default)]
    pub last_error: Option<String>,
    /// Which account this connection is authenticated as, probed at
    /// provisioning time; `None` when the server ships no identity tool.
    #[serde(default)]
    pub account: Option<String>,
}

impl Default for MCPServer {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            pkg: None,
            description: None,
            summary: None,
            tags: Vec::new(),
            catalog_slug: None,
            config: None,
            server_type: MCPServerType::Local,
            remote_config: None,
            oauth_credentials: None,
            enabled: true,
            system: false,
            tools: Vec::new(),
            server_info: None,
            status: MCPServerStatus::Ready,
            last_error: None,
            account: None,
        }
    }
}

/// A model's settings scoped to a single project.
///
/// Seeded from the workspace defaults when the project is created (see
/// [`Workspace::sync_project_models`]), then diverges independently per
/// project.  Download state deliberately lives only on [`WorkspaceModel`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProjectModel {
    #[serde(deserialize_with = "deserialize_canonical_model_name")]
    pub model_name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub is_active: bool,
    /// Reason before answering.  `None` = follow the workspace default.
    #[serde(default)]
    pub thinking: Option<bool>,
    #[serde(default)]
    pub settings: Map<String, Value>,
}

impl ProjectModel {
    /// A new entry for `model_name` (canonicalized), with defaults
    /// (enabled, inactive, no thinking override, empty settings).
    pub fn new(model_name: &str) -> Self {
        todo!()
    }
}

impl Default for ProjectModel {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            enabled: true,
            is_active: false,
            thinking: None,
            settings: Map::new(),
        }
    }
}

/// A model registered in the workspace.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkspaceModel {
    /// Short name e.g. "sl-mini" (canonicalized on load).
    #[serde(deserialize_with = "deserialize_canonical_model_name")]
    pub model_name: String,
    /// HF id e.g. "smartloop-ai/slp-orchestra-mini".
    pub model_id: String,
    #[serde(default)]
    pub downloaded: bool,
    #[serde(default)]
    pub is_active: bool,
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Reason before answering; `None` = the model class's own default.
    #[serde(default)]
    pub thinking: Option<bool>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub config: Option<ModelConfig>,
    /// Runtime-only; never persisted (`Field(exclude=True)`).
    #[serde(skip)]
    pub models_dir: String,
}

impl Default for WorkspaceModel {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            model_id: String::new(),
            downloaded: false,
            is_active: false,
            enabled: true,
            thinking: None,
            capabilities: Vec::new(),
            config: None,
            models_dir: String::new(),
        }
    }
}

impl WorkspaceModel {
    /// `<models_dir>/<model_id>`, or `""` when `models_dir` is unset.
    pub fn model_dir(&self) -> String {
        todo!()
    }

    /// `<models_dir>/<model_id>/<basename(model_id)>.gguf`, or `""`.
    pub fn model_path(&self) -> String {
        todo!()
    }

    /// Directory where the built GGUF is written for this model.
    pub fn bin_dir(&self) -> String {
        todo!()
    }

    /// GGUF file path under the model's bin directory.
    pub fn gguf_path(&self) -> String {
        todo!()
    }

    /// Parameter count in billions, taken from the model class (0 if unknown).
    pub fn param_size_b(&self) -> f64 {
        todo!()
    }

    /// Model family version declared by the model class (0 if unknown).
    pub fn model_version(&self) -> i64 {
        todo!()
    }

    /// Selection rank: larger size wins first, then newer model version.
    pub fn rank(&self) -> (f64, i64) {
        todo!()
    }

    /// Capabilities declared by the model class — read-only.  Models without a
    /// `SUPPORTED_MODELS` entry fall back to metadata-inferred capabilities.
    pub fn effective_capabilities(&self) -> Vec<String> {
        todo!()
    }

    /// Infer capabilities from the model name/id and size.
    fn infer_capabilities(&self) -> Vec<String> {
        todo!()
    }
}

/// A project: its documents, skills, MCP servers and per-project model
/// settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Project {
    pub id: String,
    pub project_dir: String,
    pub current: bool,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub system: bool,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub documents: Vec<Document>,
    #[serde(default)]
    pub skills: Vec<Skill>,
    #[serde(default)]
    pub mcp_servers: Vec<MCPServer>,
    #[serde(default)]
    pub adapters: BTreeSet<String>,
    #[serde(default)]
    pub sessions: Vec<String>,
    /// Pro/premium feature: when enabled, the orchestrator runs the web
    /// search agent for latest-info queries.  On by default; only an explicit
    /// `false` on disk turns it off.
    #[serde(default = "default_true")]
    pub web_search_enabled: bool,
    #[serde(default)]
    pub model_settings: Vec<ProjectModel>,
}

impl Default for Project {
    fn default() -> Self {
        Self {
            id: String::new(),
            project_dir: String::new(),
            current: false,
            name: None,
            description: None,
            system: false,
            cwd: None,
            documents: Vec::new(),
            skills: Vec::new(),
            mcp_servers: Vec::new(),
            adapters: BTreeSet::new(),
            sessions: Vec::new(),
            web_search_enabled: true,
            model_settings: Vec::new(),
        }
    }
}

impl Project {
    /// Where this project's SKILL.md files live (`<project_dir>/skills`).
    ///
    /// Anchored to `project_dir`, never to a client-supplied path.
    pub fn skills_dir(&self) -> String {
        todo!()
    }

    /// Where this project's ingested documents live (`<project_dir>/documents`).
    pub fn documents_dir(&self) -> String {
        todo!()
    }

    /// The client-supplied folder this project was opened on, or `None`.
    ///
    /// Reported back on the API for reference only — nothing is located
    /// through it.
    pub fn work_dir(&self) -> Option<String> {
        todo!()
    }

    /// Read skills from disk (SKILL.md files), the single source of truth.
    pub fn get_skills(&self) -> Vec<Skill> {
        todo!()
    }

    /// This project's own `project.json`, inside its `project_dir`.
    pub fn state_file(&self) -> String {
        todo!()
    }

    /// The thin record `config.json` keeps for this project
    /// (only [`PROJECT_INDEX_FIELDS`]).
    pub fn to_index(&self) -> Map<String, Value> {
        todo!()
    }

    /// Write this project's full record to its own directory as
    /// `{"version": PROJECT_STATE_VERSION, "project": {...}}`.  No-op when
    /// `project_dir` is empty.
    pub fn save_state(&self) -> std::io::Result<()> {
        todo!()
    }

    /// Per-project directory where the built GGUF is written.
    pub fn model_build_dir(&self) -> String {
        todo!()
    }

    /// Per-project GGUF file path (produced by build).
    pub fn model_gguf_path(&self, model_id: &str) -> String {
        todo!()
    }
}

impl MergeById for Document {
    fn record_id(&self) -> Option<&str> {
        Some(&self.id)
    }
    fn carry_fields(&mut self, source: Self) {
        todo!()
    }
}

impl MergeById for Skill {
    fn record_id(&self) -> Option<&str> {
        Some(&self.id)
    }
    fn carry_fields(&mut self, source: Self) {
        todo!()
    }
}

impl MergeById for MCPServer {
    fn record_id(&self) -> Option<&str> {
        Some(&self.id)
    }
    fn carry_fields(&mut self, source: Self) {
        todo!()
    }
}

impl MergeById for Project {
    fn record_id(&self) -> Option<&str> {
        Some(&self.id)
    }
    fn carry_fields(&mut self, source: Self) {
        todo!()
    }
}

/// Errors raised by workspace operations (Python `ValueError`s).
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum WorkspaceError {
    /// `set_model_enabled` on an unregistered model.
    #[error("Model not found: {0}")]
    ModelNotFound(String),
}

/// The workspace: registered models and projects, persisted under
/// `<home_dir>/config.json` plus per-project `project.json` files.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Workspace {
    pub id: String,
    #[serde(default)]
    pub projects: Vec<Project>,
    #[serde(default)]
    pub models: Vec<WorkspaceModel>,
    #[serde(default)]
    pub adapters: Vec<Value>,
    #[serde(default)]
    pub models_dir: String,
    #[serde(default)]
    pub workspace_dir: String,
    #[serde(default)]
    pub merge_model_dir: Option<String>,

    /// Last `project.json` mtime (ns) this instance has read, per project.
    #[serde(skip)]
    state_mtimes: HashMap<String, i128>,

    /// Last `config.json` mtime (ns) this instance has read its models from.
    #[serde(skip)]
    models_mtime: Option<i128>,
}

impl Workspace {
    /// A workspace with `id` and every other field defaulted.
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into(), ..Default::default() }
    }

    /// `Workspace.config_file`: the assignment override if set, else
    /// `SLP_CONFIG_FILE` (expanded), else `<home_dir>/config.json`.  Resolved
    /// on every access.
    pub fn config_file() -> String {
        todo!()
    }

    /// Override `Workspace.config_file` (tests patch it); `None` clears it.
    pub fn set_config_file(path: Option<String>) {
        todo!()
    }

    /// Return the currently active model (snapshot), or `None`.
    pub fn active_model(&self) -> Option<WorkspaceModel> {
        todo!()
    }

    /// Mark the given model as active, deactivating all others.
    pub fn set_active_model(&mut self, model_name: &str) {
        todo!()
    }

    /// Enable or disable a model.  Errors if not found.
    pub fn set_model_enabled(&mut self, model_name: &str, enabled: bool) -> Result<(), WorkspaceError> {
        todo!()
    }

    // ------------------------------------------------------------- per-project

    /// True for the always-on bootstrap/orchestrator model
    /// (`AppSettings::new().default_model`), which gets no per-project entry.
    pub fn is_bootstrap_model(&self, model_name: &str) -> bool {
        todo!()
    }

    /// Seed the project's `model_settings` from the workspace defaults.
    ///
    /// Adds a [`ProjectModel`] for every registered workspace model the project
    /// doesn't already have, copying the workspace-level enabled/active state;
    /// prunes any bootstrap-model entry.  Returns false when no project has
    /// `project_id`.
    pub fn sync_project_models(&mut self, project_id: &str) -> bool {
        todo!()
    }

    /// [`Workspace::sync_project_models`] for a project not (yet) held by this
    /// workspace.
    pub fn sync_detached_project_models(&self, project: &mut Project) {
        todo!()
    }

    /// A project's `ProjectModel` entry for `model_name`, or `None`.  No
    /// fallback to the workspace default.
    pub fn project_model<'a>(&self, project: Option<&'a Project>, model_name: &str) -> Option<&'a ProjectModel> {
        todo!()
    }

    /// Whether `model_name` is enabled for `project` (or the workspace default).
    ///
    /// Resolution: project override → workspace entry → true.
    pub fn effective_model_enabled(&self, model_name: &str, project: Option<&Project>) -> bool {
        todo!()
    }

    /// Whether `model_name` declares `ModelCapability::Thinking`.
    pub fn supports_thinking(&self, model_name: &str) -> bool {
        todo!()
    }

    /// Whether `model_name` should reason before answering.
    ///
    /// Resolution: project override → workspace entry → `slp_thinking_enabled`
    /// → the model class's default (via `LlmFactory::thinking_for`).  `None`
    /// for a model that does not declare thinking.
    pub fn effective_model_thinking(&self, model_name: &str, project: Option<&Project>) -> Option<bool> {
        todo!()
    }

    /// Set `model_name`'s reasoning preference on the project; `None` clears
    /// the override.  Returns `None` without writing for the bootstrap model
    /// (or an unknown project).
    pub fn set_project_model_thinking(
        &mut self,
        project_id: &str,
        model_name: &str,
        thinking: Option<bool>,
    ) -> Option<ProjectModel> {
        todo!()
    }

    /// Whether `model_name`'s weights are present in the workspace
    /// (workspace-wide, no project argument).
    pub fn effective_model_downloaded(&self, model_name: &str) -> bool {
        todo!()
    }

    /// Whether `model_name` is the active selection for `project`.
    pub fn effective_model_active(&self, model_name: &str, project: Option<&Project>) -> bool {
        todo!()
    }

    /// Freeze what `model_name` currently resolves to, per project, so a later
    /// change to the workspace default only reaches projects created after.
    ///
    /// Does nothing in a pinned child or a supervisor that has delegated its
    /// projects (warns), nor for the bootstrap model.
    pub fn pin_model_for_existing_projects(&mut self, model_name: &str) {
        todo!()
    }

    /// Make `model_name` an app-default for projects created from now on
    /// (called on download).  Pins existing projects first.  One-way.
    pub fn promote_model_to_workspace_default(&mut self, model_name: &str) {
        todo!()
    }

    /// Set `model_name`'s enabled state on the project, upserting an entry.
    /// `None` for the bootstrap model (or an unknown project).
    pub fn set_project_model_enabled(
        &mut self,
        project_id: &str,
        model_name: &str,
        enabled: bool,
    ) -> Option<ProjectModel> {
        todo!()
    }

    /// Set `model_name` as active on the project, deactivating the others.
    /// `None` for the bootstrap model (or an unknown project).
    pub fn set_project_model_active(
        &mut self,
        project_id: &str,
        model_name: &str,
        active: bool,
    ) -> Option<ProjectModel> {
        todo!()
    }

    /// Look up a registered model by its short name (alias-aware); `None`
    /// means the default model.  Returns a snapshot with `models_dir` set.
    pub fn get_model(&self, model_name: Option<&str>) -> Option<WorkspaceModel> {
        todo!()
    }

    /// Mutable form of [`Workspace::get_model`]; also stamps the record's
    /// `models_dir` from the workspace.
    pub fn get_model_mut(&mut self, model_name: Option<&str>) -> Option<&mut WorkspaceModel> {
        todo!()
    }

    /// Register a model without changing the active model; merges `config`
    /// into an existing record.  Stored under the canonical name.  Returns a
    /// snapshot of the (new or existing) record.
    pub fn add_model(
        &mut self,
        model_name: &str,
        model_id: &str,
        config: Option<ModelConfig>,
        enabled: bool,
    ) -> WorkspaceModel {
        todo!()
    }

    /// The best enabled, downloaded model for `project` (largest, then newest).
    pub fn largest_downloaded_model(&self, project: Option<&Project>) -> Option<WorkspaceModel> {
        todo!()
    }

    /// Enabled, downloaded models declaring `capability`, best-first.
    pub fn ranked_models_for(&self, capability: &str, project: Option<&Project>) -> Vec<WorkspaceModel> {
        todo!()
    }

    /// The single best model for a capability, or the largest downloaded model
    /// when none declares it.
    pub fn best_model_for(&self, capability: &str, project: Option<&Project>) -> Option<WorkspaceModel> {
        todo!()
    }

    /// The workspace-level directory for a given `model_id`.
    pub fn model_dir_for(&self, model_id: &str) -> String {
        todo!()
    }

    /// The workspace-level GGUF file path for a given `model_id`.
    pub fn model_path_for(&self, model_id: &str) -> String {
        todo!()
    }

    /// Cascade: quantize-matched bin GGUF → first bin GGUF → exact workspace
    /// GGUF → first main GGUF in model dir → exact path.
    pub fn resolve_gguf_path(&self, model_id: &str) -> String {
        todo!()
    }

    /// The `config.json` document, with `projects` as its project list:
    /// `{"id": <active model_id or "">, "workspace": {...}}`.
    fn workspace_record(&self, projects: Vec<Value>) -> Value {
        todo!()
    }

    /// Persist workspace state, each project into its own file.
    ///
    /// Serialized and atomic.  A pinned child writes only its own
    /// `project.json`; a supervisor that delegated projects writes the
    /// workspace record plus only an explicitly owned project; any other
    /// process writes the record and every project's file.
    /// `SLP_LEGACY_CONFIG=1` restores the single-document format.
    pub fn save(&self) -> std::io::Result<()> {
        todo!()
    }

    /// The projects whose own file this unpinned process may write.
    fn writable_projects(&self) -> Vec<&Project> {
        todo!()
    }

    /// Fold each project's own `project.json` into the raw workspace record
    /// before deserialization.
    fn hydrate_projects(ws_data: Map<String, Value>) -> Map<String, Value> {
        todo!()
    }

    /// A fresh workspace with a new uuid and its models dir created.
    fn create_new_workspace(config: &ModelConfig) -> Workspace {
        todo!()
    }

    /// Load the workspace singleton from `config.json` on disk.
    ///
    /// Returns the cached instance if already loaded; creates a new workspace
    /// when the file is missing, empty or invalid.  Merges `config` into the
    /// requested model's record.
    pub fn init(config: &ModelConfig) -> SharedWorkspace {
        todo!()
    }

    /// Re-anchor every persisted absolute path onto the current home.
    fn rebase_paths(workspace: &mut Workspace) {
        todo!()
    }

    /// Load the workspace directly from `config.json` on disk, without
    /// touching the singleton.  `None` when missing, empty, or invalid.
    pub fn load() -> Option<Workspace> {
        todo!()
    }

    /// Re-read one project's own file if someone has written it since
    /// (mtime-gated).  Updates the project in place, keeping workspace-owned
    /// fields.  Returns true when something was re-read.
    pub fn refresh_project(&mut self, project_id: &str) -> bool {
        todo!()
    }

    /// Re-read the registered models if the workspace record has moved on
    /// (mtime-gated).  Returns true when something was re-read.
    pub fn refresh_models(&mut self) -> bool {
        todo!()
    }

    /// Re-read project state from disk into this instance, updating projects
    /// and their records in place (merge by id).
    pub fn reload(&mut self) {
        todo!()
    }

    /// Clear the cached singleton.
    pub fn reset() {
        todo!()
    }

    /// Return the current singleton instance, or `None`.
    pub fn get_instance() -> Option<SharedWorkspace> {
        todo!()
    }

    /// The project this unit of work belongs to: the bound project (see
    /// [`crate::workspace::context`]) if this workspace holds it (warning
    /// otherwise), else the project flagged `current`.
    pub fn current_project(&self) -> Option<&Project> {
        todo!()
    }

    /// Mutable form of [`Workspace::current_project`].
    pub fn current_project_mut(&mut self) -> Option<&mut Project> {
        todo!()
    }

    /// Find a project by id.
    pub fn project(&self, project_id: &str) -> Option<&Project> {
        self.projects.iter().find(|p| p.id == project_id)
    }

    /// Find a project by id, mutably.
    pub fn project_mut(&mut self, project_id: &str) -> Option<&mut Project> {
        self.projects.iter_mut().find(|p| p.id == project_id)
    }
}
