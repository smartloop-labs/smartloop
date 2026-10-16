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

use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value, json};

use crate::config::{AppSettings, ModelConfig, SUPPORTED_MODELS};
use crate::constants::canonical_model_name;
use crate::llms::llm_factory::LlmFactory;
use crate::paths::{expand_home, real_home};
use crate::types::{ModelCapability, ModelKind};
use crate::workspace::context::{
    active_project_id, owned_project_id, pinned_project_id, projects_delegated,
};
use crate::workspace::types::path_name;
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
/// route projects without opening any of them; everything else lives in the
/// project's own file.
pub const PROJECT_INDEX_FIELDS: &[&str] =
    &["id", "name", "description", "project_dir", "current", "system"];

/// Computed/runtime keys stripped from a persisted workspace record before it
/// is validated (and from the record before it is written).
const RUNTIME_WORKSPACE_KEYS: &[&str] = &["model_dir", "model_path", "current_project", "model_id"];

/// Override of the import-time settings home (Python tests patch
/// `workspace._settings`).
static SETTINGS_HOME_DIR: Lazy<RwLock<Option<String>>> = Lazy::new(|| RwLock::new(None));

/// Import-time `AppSettings()` (Python's module-level `_settings`): pins the
/// `SLP_HOME` / `.env` values seen when the workspace module was first used.
static IMPORT_SETTINGS: Lazy<Option<AppSettings>> = Lazy::new(|| AppSettings::try_new().ok());

/// Assignment override for `Workspace.config_file` (tests patch it).
static CONFIG_FILE_OVERRIDE: Lazy<RwLock<Option<String>>> = Lazy::new(|| RwLock::new(None));

/// The cached singleton (`Workspace._instance`), guarded by `Workspace._lock`.
static INSTANCE: Lazy<Mutex<Option<SharedWorkspace>>> = Lazy::new(|| Mutex::new(None));

/// Serializes `save()`: background tasks (document indexing, MCP
/// provisioning) each persist from their own thread.
static SAVE_LOCK: Mutex<()> = Mutex::new(());

/// Override the import-time settings home (Python tests patch
/// `workspace._settings`).  `None` restores `AppSettings::new().home_dir`.
pub fn set_settings_home_dir(home_dir: Option<String>) {
    *SETTINGS_HOME_DIR.write() = home_dir;
}

/// The framework home, resolved now rather than at import.
///
/// The import-time settings pin `SLP_HOME` to whatever it was then.  A
/// per-project agent child is spawned with its own environment, so the home
/// has to be re-read at the moment it is needed: `SLP_HOME` (expanded) when
/// set, else the import-time settings home.
fn home_dir() -> String {
    if let Ok(home) = std::env::var("SLP_HOME")
        && !home.is_empty()
    {
        return expand_home(&home);
    }
    if let Some(home) = SETTINGS_HOME_DIR.read().clone() {
        return home;
    }
    match IMPORT_SETTINGS.as_ref() {
        Some(settings) => settings.home_dir(),
        None => join(&real_home(), ".smartloop"),
    }
}

/// True when the workspace should be kept as a single `config.json`.
///
/// The rollback path for the split: set `SLP_LEGACY_CONFIG=1` and the
/// framework reads and writes the monolith again, ignoring any `project.json`
/// already written beside it.  Nothing has to be undone by hand — the
/// pre-migration backup is still there, and the split files are simply not
/// consulted.
pub fn legacy_layout() -> bool {
    let value = std::env::var("SLP_LEGACY_CONFIG").unwrap_or_default();
    matches!(value.trim().to_lowercase().as_str(), "1" | "true" | "yes")
}

/// `os.path.join(a, b)` for two string paths.
pub(crate) fn join(base: &str, part: &str) -> String {
    Path::new(base).join(part).to_string_lossy().into_owned()
}

/// `os.path.basename`: everything after the last `/`.
pub(crate) fn basename(path: &str) -> String {
    match path.rfind(std::path::MAIN_SEPARATOR) {
        Some(index) => path[index + 1..].to_string(),
        None => path.to_string(),
    }
}

/// `os.stat(path).st_mtime_ns`, or `None` when the file cannot be stat-ed.
fn mtime_ns(path: &str) -> Option<i128> {
    let modified = std::fs::metadata(path).ok()?.modified().ok()?;
    Some(match modified.duration_since(std::time::UNIX_EPOCH) {
        Ok(after) => after.as_nanos() as i128,
        Err(before) => -(before.duration().as_nanos() as i128),
    })
}

/// A JSON formatter producing exactly what Python's
/// `json.dump(data, fh, indent=4)` writes: 4-space indentation, `", "`-free
/// separators and `ensure_ascii` escaping of every non-ASCII character.
struct PythonJsonFormatter {
    inner: serde_json::ser::PrettyFormatter<'static>,
}

impl PythonJsonFormatter {
    fn new() -> Self {
        Self { inner: serde_json::ser::PrettyFormatter::with_indent(b"    ") }
    }
}

impl serde_json::ser::Formatter for PythonJsonFormatter {
    fn begin_array<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        self.inner.begin_array(writer)
    }

    fn end_array<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        self.inner.end_array(writer)
    }

    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        self.inner.begin_array_value(writer, first)
    }

    fn end_array_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        self.inner.end_array_value(writer)
    }

    fn begin_object<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        self.inner.begin_object(writer)
    }

    fn end_object<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        self.inner.end_object(writer)
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        self.inner.begin_object_key(writer, first)
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        self.inner.begin_object_value(writer)
    }

    fn end_object_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        self.inner.end_object_value(writer)
    }

    fn write_string_fragment<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> std::io::Result<()> {
        // `ensure_ascii=True`: anything outside printable ASCII is written as
        // a (lowercase) `\uXXXX` escape, astral characters as surrogate pairs.
        let mut start = 0;
        for (index, ch) in fragment.char_indices() {
            if (' '..='~').contains(&ch) {
                continue;
            }
            writer.write_all(fragment[start..index].as_bytes())?;
            let mut units = [0u16; 2];
            for unit in ch.encode_utf16(&mut units) {
                write!(writer, "\\u{:04x}", unit)?;
            }
            start = index + ch.len_utf8();
        }
        writer.write_all(fragment[start..].as_bytes())
    }
}

/// Serialize `data` the way Python's `json.dump(data, fh, indent=4)` does.
pub fn python_json_dumps(data: &Value) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, PythonJsonFormatter::new());
    data.serialize(&mut serializer).map_err(std::io::Error::other)?;
    Ok(out)
}

/// Write `data` to `path` so a reader sees the old file or the new one.
///
/// Same contract the workspace config has always had: write a sibling
/// `<path>.tmp` (4-space indented JSON), flush it to the platter, then rename
/// over the target.
pub fn atomic_write_json(path: &str, data: &Value) -> std::io::Result<()> {
    let parent = match Path::new(path).parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => Path::new(".").to_path_buf(),
    };
    std::fs::create_dir_all(&parent)?;
    let tmp_file = format!("{path}.tmp");
    {
        let mut fh = std::fs::File::create(&tmp_file)?;
        fh.write_all(&python_json_dumps(data)?)?;
        fh.flush()?;
        fh.sync_all()?;
    }
    std::fs::rename(&tmp_file, path)
}

/// Where a project's own state lives.
pub fn project_state_path(project_dir: &str) -> String {
    join(project_dir, PROJECT_STATE_FILE)
}

/// Return a project's persisted record (the `"project"` object), or `None`.
///
/// Never fails: a project whose file is missing, empty or corrupt falls back
/// to the thin record in `config.json` rather than failing the whole load and
/// taking every other project down with it.
pub fn read_project_state(project_dir: &str) -> Option<Map<String, Value>> {
    let path = project_state_path(project_dir);
    let content = std::fs::read_to_string(&path).ok()?;
    let content = content.trim();
    if content.is_empty() {
        return None;
    }
    let data: Value = match serde_json::from_str(content) {
        Ok(data) => data,
        Err(_) => {
            tracing::warn!(target: LOG_TARGET, "[project-state] {} is not valid JSON; ignoring it", path);
            return None;
        }
    };
    match data {
        Value::Object(mut data) => match data.remove("project") {
            Some(Value::Object(project)) => Some(project),
            _ => None,
        },
        _ => None,
    }
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
///
/// Identity is the whole point.  Background passes act on individual records:
/// the provisioning pass on the [`MCPServer`] whose token it is rotating, the
/// indexer on the [`Document`] it is marking indexed.  When the ids and their
/// order are unchanged the records are updated where they lie, so nothing
/// moves under them.
pub fn merge_records_by_id<T: MergeById>(current: &mut Vec<T>, fresh: Vec<T>) {
    if !current.iter().any(|record| record.record_id().is_some()) {
        *current = fresh;
        return;
    }

    let same_shape = current.len() == fresh.len()
        && current.iter().zip(fresh.iter()).all(|(held, incoming)| {
            held.record_id().is_some() && held.record_id() == incoming.record_id()
        });
    if same_shape {
        for (held, incoming) in current.iter_mut().zip(fresh) {
            held.carry_fields(incoming);
        }
        return;
    }

    let mut held: HashMap<String, T> = HashMap::new();
    for record in current.drain(..) {
        if let Some(id) = record.record_id().map(str::to_string) {
            held.insert(id, record);
        }
    }
    let mut merged = Vec::with_capacity(fresh.len());
    for record in fresh {
        let existing = record.record_id().and_then(|id| held.remove(id));
        match existing {
            Some(mut existing) => {
                existing.carry_fields(record);
                merged.push(existing);
            }
            None => merged.push(record),
        }
    }
    *current = merged;
}

/// True when a GGUF is a vision mmproj / CLIP projector (not main weights).
pub fn is_mmproj_gguf(filename: &str) -> bool {
    let name = basename(filename).to_lowercase();
    name.contains("mmproj") || name.contains("clip")
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

/// Deserialize `null` as the type's default — the `x or []` the Python
/// `Workspace.__init__` applies to its list arguments.
fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
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
    /// be what was asked for: a server may narrow the set and say so here.
    /// Persisted because it is the only record of what this connection may do.
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
/// existed — already discovered — are not re-queued on upgrade.
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
    /// Slug of the curated catalog entry this server was installed from, when
    /// it was.  Provenance: it is what marks metadata as ours rather than
    /// generated.
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
    /// enrichment).  Registration returns as soon as the connection is
    /// persisted, so clients poll this instead of blocking on the handshake.
    #[serde(default)]
    pub status: MCPServerStatus,
    /// Human-readable reason the last provisioning pass failed, if it did.
    #[serde(default)]
    pub last_error: Option<String>,
    /// Which account this connection is authenticated as, probed at
    /// provisioning time.  It is what distinguishes two connections to the
    /// same product; `None` when the server ships no identity tool.
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
/// Seeded from the workspace defaults (the app-default template) when the
/// project is created (see [`Workspace::sync_project_models`]), then diverges
/// independently per project.  `settings` is a catch-all map for
/// project-level inference defaults (temperature, top_p, top_k, …).
///
/// Download state deliberately lives only on [`WorkspaceModel`]: weights are
/// downloaded and removed for the whole workspace, so a per-project copy would
/// be a cache of a global fact with nothing to invalidate it.  See
/// [`Workspace::effective_model_downloaded`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProjectModel {
    /// Canonicalized on load, so a project whose `model_settings` were written
    /// before a rename still resolves against the workspace registry.
    #[serde(deserialize_with = "deserialize_canonical_model_name")]
    pub model_name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub is_active: bool,
    /// Reason before answering.  `None` = follow the workspace default (and
    /// below that the model class's).  Only meaningful for a model declaring
    /// `ModelCapability::Thinking`; see [`Workspace::effective_model_thinking`].
    #[serde(default)]
    pub thinking: Option<bool>,
    #[serde(default)]
    pub settings: Map<String, Value>,
}

impl ProjectModel {
    /// A new entry for `model_name` (canonicalized), with defaults
    /// (enabled, inactive, no thinking override, empty settings).
    pub fn new(model_name: &str) -> Self {
        Self { model_name: canonical_model_name(model_name), ..Default::default() }
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
    /// Short name e.g. "sl-mini".  Canonicalized on every load, so a workspace
    /// created before a rename reports the current name and picks up the new
    /// one on the next save.  `model_id` is deliberately left alone — it names
    /// the on-disk model directory.
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
    /// Reason before answering; `None` = the model class's own default.  Set
    /// from the model picker for models whose capabilities include "thinking".
    #[serde(default)]
    pub thinking: Option<bool>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub config: Option<ModelConfig>,
    /// Remote models run behind `url` and have no weights on disk.
    #[serde(rename = "type", default, skip_serializing_if = "ModelKind::is_local")]
    pub kind: ModelKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
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
            kind: ModelKind::Local,
            url: None,
            context_length: None,
            models_dir: String::new(),
        }
    }
}

impl WorkspaceModel {
    /// Whether this model runs behind the platform API rather than on device.
    pub fn is_remote(&self) -> bool {
        self.kind == ModelKind::Remote
    }

    /// Whether there is no on-disk location to name: `models_dir` unset, or a
    /// remote model (whose empty `model_id` would otherwise resolve to the
    /// whole models directory).
    fn has_no_weights_path(&self) -> bool {
        self.models_dir.is_empty() || self.is_remote() || self.model_id.is_empty()
    }

    /// `<models_dir>/<model_id>`, or `""` when there are no weights to locate.
    pub fn model_dir(&self) -> String {
        if self.has_no_weights_path() {
            return String::new();
        }
        join(&self.models_dir, &self.model_id)
    }

    /// `<models_dir>/<model_id>/<basename(model_id)>.gguf`, or `""`.
    pub fn model_path(&self) -> String {
        if self.has_no_weights_path() {
            return String::new();
        }
        join(
            &join(&self.models_dir, &self.model_id),
            &format!("{}.gguf", path_name(&self.model_id)),
        )
    }

    /// Directory where the built GGUF is written for this model.
    pub fn bin_dir(&self) -> String {
        if self.has_no_weights_path() {
            return String::new();
        }
        join(&self.model_dir(), "bin")
    }

    /// GGUF file path under the model's bin directory.
    pub fn gguf_path(&self) -> String {
        if self.has_no_weights_path() {
            return String::new();
        }
        join(&self.bin_dir(), &format!("{}.gguf", path_name(&self.model_id)))
    }

    /// Parameter count in billions, taken from the model class (the
    /// authoritative source).  0 for models without a `SUPPORTED_MODELS`
    /// entry or an explicit size.
    pub fn param_size_b(&self) -> f64 {
        match SUPPORTED_MODELS.get(&self.model_name) {
            Some(model_cls) => (model_cls.param_size_in_b)(),
            None => 0.0,
        }
    }

    /// Model family version declared by the model class (0 if unknown).
    ///
    /// Selection prefers a higher version over a lower one only between two
    /// models of the same size (see [`WorkspaceModel::rank`]).
    pub fn model_version(&self) -> i64 {
        match SUPPORTED_MODELS.get(&self.model_name) {
            Some(model_cls) => (model_cls.model_version)(),
            None => 0,
        }
    }

    /// Selection rank: larger size wins first, then newer model version.
    ///
    /// Size is the primary key so the most capable (largest) model is always
    /// preferred; the model version only breaks a tie between two models of
    /// the same parameter count.
    pub fn rank(&self) -> (f64, i64) {
        (self.param_size_b(), self.model_version())
    }

    /// Capabilities declared by the model class — read-only.
    ///
    /// The stored `capabilities` field is ignored so class defaults stay
    /// authoritative and updates to any model's capabilities take effect
    /// immediately.  A remote model has no class, so its stored list (from
    /// the catalog) is used.  Other models without a `SUPPORTED_MODELS` entry
    /// fall back to metadata-inferred capabilities.
    pub fn effective_capabilities(&self) -> Vec<String> {
        if self.is_remote() {
            return self.capabilities.clone();
        }
        match SUPPORTED_MODELS.get(&self.model_name) {
            Some(model_cls) => (model_cls.default_capabilities)()
                .into_iter()
                .map(|c| c.value().to_string())
                .collect(),
            None => self.infer_capabilities(),
        }
    }

    /// Infer capabilities from the model name/id and size.
    fn infer_capabilities(&self) -> Vec<String> {
        let combined = format!("{} {}", self.model_name, self.model_id).to_lowercase();
        let mut caps: Vec<String> = Vec::new();
        let size = self.param_size_b();

        caps.push(ModelCapability::General.value().to_string());

        if ["code", "coder", "starcoder", "deepseek-coder"].iter().any(|kw| combined.contains(kw)) {
            caps.push(ModelCapability::Coding.value().to_string());
        }

        // Vision is inferred only from an explicit vision marker in the name
        // (e.g. Qwen2.5-VL).  gemma-3-4b is NOT inferred as vision: llama-cpp
        // has no Gemma3 vision handler, so tagging it vision only produces a
        // model that's considered for OCR and then skipped every time.
        if ["vision", "vlm", "vl"].iter().any(|kw| combined.contains(kw)) {
            caps.push(ModelCapability::Vision.value().to_string());
        }

        if size >= 3.0 {
            // GENERAL already added above; larger models additionally handle
            // reasoning (skill matching, MCP tool resolution, planning) and
            // document processing.
            caps.push(ModelCapability::Reasoning.value().to_string());
            caps.push(ModelCapability::Dp.value().to_string());
        }

        caps
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
    /// search agent for latest-info queries.  On by default, so new projects —
    /// and configs written before this flag existed — start enabled; only an
    /// explicit `false` on disk turns it off.  Searching still no-ops without
    /// a configured provider/token.
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
    /// Anchored to `project_dir` — i.e. under the framework's own home — never
    /// to a client-supplied path.  `project_dir` derives from
    /// `AppSettings.home_dir`, which is `SLP_HOME` else `~/.smartloop` in
    /// every build, sandboxed or not.  A client-supplied absolute path would
    /// not follow it, leaving the app reading one workspace and writing
    /// another.
    pub fn skills_dir(&self) -> String {
        join(&self.project_dir, "skills")
    }

    /// Where this project's ingested documents live (`<project_dir>/documents`).
    pub fn documents_dir(&self) -> String {
        join(&self.project_dir, "documents")
    }

    /// The client-supplied folder this project was opened on, or `None`.
    ///
    /// Reported back on the API for reference only — **nothing is located
    /// through it**.  Skills, documents, MCP docs and the workspace index all
    /// hang off `project_dir` so the framework stays self-contained.
    pub fn work_dir(&self) -> Option<String> {
        match self.cwd.as_deref() {
            Some(cwd) if !cwd.is_empty() => Some(expand_home(cwd)),
            _ => None,
        }
    }

    /// Read skills from disk (SKILL.md files), the single source of truth.
    pub fn get_skills(&self) -> Vec<Skill> {
        let skills_dir = self.skills_dir();
        if !Path::new(&skills_dir).is_dir() {
            return Vec::new();
        }
        let Ok(entries) = std::fs::read_dir(&skills_dir) else {
            return Vec::new();
        };
        let mut names: Vec<String> = entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();

        let mut result = Vec::new();
        for entry in names {
            let entry_path = join(&skills_dir, &entry);
            if !Path::new(&entry_path).is_dir() {
                continue;
            }
            let skill_path = join(&entry_path, "SKILL.md");
            if !Path::new(&skill_path).is_file() {
                continue;
            }
            let Ok(raw) = std::fs::read_to_string(&skill_path) else {
                continue;
            };
            // Text-mode reads translate line endings, as Python's `open` does.
            let raw = raw.replace("\r\n", "\n").replace('\r', "\n");
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
            result.push(Skill {
                id: name.clone(),
                name,
                description,
                content: raw,
                system: false,
            });
        }
        result
    }

    // ------------------------------------------------------------- persistence

    /// This project's own `project.json`, inside its `project_dir`.
    pub fn state_file(&self) -> String {
        project_state_path(&self.project_dir)
    }

    /// The thin record `config.json` keeps for this project
    /// (only [`PROJECT_INDEX_FIELDS`]).
    ///
    /// Enough to list projects and route a request to one without opening any
    /// of them.  Documents, skills, MCP servers and model settings are
    /// deliberately absent: they are the bulk of the old monolith, and the
    /// reason two processes could not both hold a workspace.
    pub fn to_index(&self) -> Map<String, Value> {
        let dumped = match serde_json::to_value(self) {
            Ok(Value::Object(map)) => map,
            _ => Map::new(),
        };
        let mut index = Map::new();
        for field in PROJECT_INDEX_FIELDS {
            if let Some(value) = dumped.get(*field) {
                index.insert((*field).to_string(), value.clone());
            }
        }
        index
    }

    /// Write this project's full record to its own directory as
    /// `{"version": PROJECT_STATE_VERSION, "project": {...}}`.  No-op when
    /// `project_dir` is empty.
    ///
    /// The whole record, not just the fields `config.json` leaves out, so the
    /// directory is self-describing: a `project_dir` can be copied to another
    /// workspace and read back without the file it came from.
    pub fn save_state(&self) -> std::io::Result<()> {
        if self.project_dir.is_empty() {
            return Ok(());
        }
        let project = serde_json::to_value(self).map_err(std::io::Error::other)?;
        atomic_write_json(
            &self.state_file(),
            &json!({"version": PROJECT_STATE_VERSION, "project": project}),
        )
    }

    /// Per-project directory where the built GGUF is written.
    pub fn model_build_dir(&self) -> String {
        join(&self.project_dir, "model")
    }

    /// Per-project GGUF file path (produced by build).
    pub fn model_gguf_path(&self, model_id: &str) -> String {
        join(&join(&self.project_dir, "model"), &format!("{}.gguf", path_name(model_id)))
    }

    /// Copy `source` onto this project, merging the id-keyed record lists
    /// (documents, skills, MCP servers) in place.  With `keep_identity`, the
    /// fields that are not the project's own to change — `id`, `project_dir`
    /// and the workspace-owned selection — are left as they are.
    fn carry_project_fields(&mut self, source: Project, keep_identity: bool) {
        let Project {
            id,
            project_dir,
            current,
            name,
            description,
            system,
            cwd,
            documents,
            skills,
            mcp_servers,
            adapters,
            sessions,
            web_search_enabled,
            model_settings,
        } = source;
        if !keep_identity {
            self.id = id;
            self.project_dir = project_dir;
            self.current = current;
        }
        self.name = name;
        self.description = description;
        self.system = system;
        self.cwd = cwd;
        merge_records_by_id(&mut self.documents, documents);
        merge_records_by_id(&mut self.skills, skills);
        merge_records_by_id(&mut self.mcp_servers, mcp_servers);
        self.adapters = adapters;
        // Plain strings and id-less records are not reconcilable by id and
        // are taken wholesale.
        self.sessions = sessions;
        self.web_search_enabled = web_search_enabled;
        self.model_settings = model_settings;
    }
}

impl MergeById for Document {
    fn record_id(&self) -> Option<&str> {
        Some(&self.id)
    }
    fn carry_fields(&mut self, source: Self) {
        // No nested id-keyed lists: every field is simply brought up to date.
        *self = source;
    }
}

impl MergeById for Skill {
    fn record_id(&self) -> Option<&str> {
        Some(&self.id)
    }
    fn carry_fields(&mut self, source: Self) {
        *self = source;
    }
}

impl MergeById for MCPServer {
    fn record_id(&self) -> Option<&str> {
        Some(&self.id)
    }
    fn carry_fields(&mut self, source: Self) {
        // `tags` (strings) and `tools` (records without an id) are taken
        // wholesale, exactly as `_carry_field` does for them.
        *self = source;
    }
}

impl MergeById for Project {
    fn record_id(&self) -> Option<&str> {
        Some(&self.id)
    }
    fn carry_fields(&mut self, source: Self) {
        self.carry_project_fields(source, false);
    }
}

/// Errors raised by workspace operations (Python `ValueError`s).
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum WorkspaceError {
    /// `set_model_enabled` on an unregistered model.
    #[error("Model not found: {0}")]
    ModelNotFound(String),
}

/// The canonical name of the always-on bootstrap model
/// (`AppSettings().default_model`).
fn bootstrap_model_name() -> String {
    canonical_model_name(&AppSettings::new().default_model)
}

/// Merge `incoming` into `existing` the way `ModelConfig.model_validate` over
/// `existing.model_dump()` updated with the non-`None` fields of
/// `incoming.model_dump()` does.
fn merge_model_config(existing: &ModelConfig, incoming: &ModelConfig) -> ModelConfig {
    let mut merged = match serde_json::to_value(existing) {
        Ok(Value::Object(map)) => map,
        _ => Map::new(),
    };
    if let Ok(Value::Object(fresh)) = serde_json::to_value(incoming) {
        for (key, value) in fresh {
            if !value.is_null() {
                merged.insert(key, value);
            }
        }
    }
    serde_json::from_value(Value::Object(merged)).unwrap_or_else(|_| incoming.clone())
}

/// Seed `project.model_settings` from `models` (see
/// [`Workspace::sync_project_models`]).
fn sync_models_into(models: &[WorkspaceModel], project: &mut Project) {
    let bootstrap = bootstrap_model_name();
    let is_bootstrap = |name: &str| canonical_model_name(name) == bootstrap;

    project.model_settings.retain(|m| !is_bootstrap(&m.model_name));
    let existing: HashSet<String> =
        project.model_settings.iter().map(|m| canonical_model_name(&m.model_name)).collect();
    for wm in models {
        if existing.contains(&canonical_model_name(&wm.model_name)) || is_bootstrap(&wm.model_name) {
            continue;
        }
        project.model_settings.push(ProjectModel {
            enabled: wm.enabled,
            is_active: wm.is_active,
            ..ProjectModel::new(&wm.model_name)
        });
    }
}

/// The workspace: registered models and projects, persisted under
/// `<home_dir>/config.json` plus per-project `project.json` files.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Workspace {
    pub id: String,
    /// `projects or []`: an explicit `null` loads as an empty list.
    #[serde(default, deserialize_with = "null_as_default")]
    pub projects: Vec<Project>,
    #[serde(default, deserialize_with = "null_as_default")]
    pub models: Vec<WorkspaceModel>,
    #[serde(default, deserialize_with = "null_as_default")]
    pub adapters: Vec<Value>,
    #[serde(default)]
    pub models_dir: String,
    #[serde(default)]
    pub workspace_dir: String,
    #[serde(default)]
    pub merge_model_dir: Option<String>,

    /// Last `project.json` mtime (ns) this instance has read, per project.
    /// Per *instance*, not shared: two instances can hold different vintages
    /// of the same project, and a map shared between them would let one skip
    /// a re-read the other had already accounted for.
    #[serde(skip)]
    state_mtimes: HashMap<String, i128>,

    /// Last `config.json` mtime (ns) this instance has read its models from.
    /// Per instance for the same reason as `state_mtimes`.
    #[serde(skip)]
    models_mtime: Option<i128>,
}

impl Workspace {
    /// A workspace with `id` and every other field defaulted.
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into(), ..Default::default() }
    }

    /// Builder: set `projects`.
    pub fn with_projects(mut self, projects: Vec<Project>) -> Self {
        self.projects = projects;
        self
    }

    /// Builder: set `models`.
    pub fn with_models(mut self, models: Vec<WorkspaceModel>) -> Self {
        self.models = models;
        self
    }

    /// Builder: set `workspace_dir`.
    pub fn with_workspace_dir(mut self, workspace_dir: impl Into<String>) -> Self {
        self.workspace_dir = workspace_dir.into();
        self
    }

    /// Builder: set `models_dir`.
    pub fn with_models_dir(mut self, models_dir: impl Into<String>) -> Self {
        self.models_dir = models_dir.into();
        self
    }

    /// `Workspace.config_file`: the assignment override if set, else
    /// `SLP_CONFIG_FILE` (expanded), else `<home_dir>/config.json`.  Resolved
    /// on every access rather than at import, so a child pointed at a
    /// different home finds its own.
    pub fn config_file() -> String {
        if let Some(path) = CONFIG_FILE_OVERRIDE.read().clone() {
            return path;
        }
        if let Ok(path) = std::env::var("SLP_CONFIG_FILE")
            && !path.is_empty()
        {
            return expand_home(&path);
        }
        join(&home_dir(), "config.json")
    }

    /// Override `Workspace.config_file` (tests patch it); `None` clears it.
    pub fn set_config_file(path: Option<String>) {
        *CONFIG_FILE_OVERRIDE.write() = path;
    }

    /// Return the currently active model (snapshot), or `None`.
    pub fn active_model(&self) -> Option<WorkspaceModel> {
        self.get_model(None)
    }

    /// Mark the given model as active, deactivating all others.
    pub fn set_active_model(&mut self, model_name: &str) {
        let target = canonical_model_name(model_name);
        for m in &mut self.models {
            m.is_active = canonical_model_name(&m.model_name) == target;
        }
    }

    /// Enable or disable a model.  Errors if not found.
    pub fn set_model_enabled(&mut self, model_name: &str, enabled: bool) -> Result<(), WorkspaceError> {
        let target = canonical_model_name(model_name);
        for m in &mut self.models {
            if canonical_model_name(&m.model_name) == target {
                m.enabled = enabled;
                return Ok(());
            }
        }
        Err(WorkspaceError::ModelNotFound(model_name.to_string()))
    }

    // ------------------------------------------------------------- per-project

    /// True for the always-on bootstrap/orchestrator model
    /// (`AppSettings::new().default_model`), which gets no per-project entry.
    ///
    /// That model is always enabled, can't be turned off, and is filtered out
    /// of every models list response, so it deliberately gets no per-project
    /// entry: a per-project `enabled=false` would shadow its always-on
    /// workspace state everywhere selection resolves through
    /// [`Workspace::effective_model_enabled`].  With no entry it keeps falling
    /// through to the workspace record, which is the real answer.
    pub fn is_bootstrap_model(&self, model_name: &str) -> bool {
        canonical_model_name(model_name) == bootstrap_model_name()
    }

    /// Seed the project's `model_settings` from the workspace defaults.
    ///
    /// Adds a [`ProjectModel`] for every registered workspace model the project
    /// doesn't already have, copying the workspace-level (app-default)
    /// enabled/active state.  Existing entries are left untouched so
    /// per-project overrides survive.  Called at project creation and lazily
    /// before reads/writes so models added to the catalog after a project was
    /// created show up too.
    ///
    /// The bootstrap model is excluded (see [`Workspace::is_bootstrap_model`]),
    /// and any entry a previous build already wrote for it is pruned here so
    /// existing workspaces heal on next access.  Returns false when no project
    /// has `project_id`.
    pub fn sync_project_models(&mut self, project_id: &str) -> bool {
        let Workspace { projects, models, .. } = self;
        match projects.iter_mut().find(|p| p.id == project_id) {
            Some(project) => {
                sync_models_into(models, project);
                true
            }
            None => false,
        }
    }

    /// [`Workspace::sync_project_models`] for a project not (yet) held by this
    /// workspace.
    pub fn sync_detached_project_models(&self, project: &mut Project) {
        sync_models_into(&self.models, project);
    }

    /// A project's `ProjectModel` entry for `model_name`, or `None`.
    ///
    /// No fallback to the workspace default here — callers that want the
    /// merged view use the `effective_*` accessors.
    pub fn project_model<'a>(&self, project: Option<&'a Project>, model_name: &str) -> Option<&'a ProjectModel> {
        let project = project?;
        let target = canonical_model_name(model_name);
        project.model_settings.iter().find(|m| canonical_model_name(&m.model_name) == target)
    }

    /// Whether `model_name` is enabled for `project` (or the workspace default).
    ///
    /// Resolution: project override → workspace (app-default) entry → true.
    pub fn effective_model_enabled(&self, model_name: &str, project: Option<&Project>) -> bool {
        if let Some(pm) = self.project_model(project, model_name) {
            return pm.enabled;
        }
        self.get_model(Some(model_name)).map(|ws| ws.enabled).unwrap_or(true)
    }

    /// Whether `model_name` declares `ModelCapability::Thinking`.
    ///
    /// Read straight off the resolved capabilities, so it tracks the model
    /// class rather than anything persisted.
    pub fn supports_thinking(&self, model_name: &str) -> bool {
        let caps = self
            .get_model(Some(model_name))
            .map(|ws| ws.effective_capabilities())
            .unwrap_or_default();
        let thinking = ModelCapability::Thinking.value();
        caps.iter().any(|c| c.to_lowercase() == thinking)
    }

    /// Whether `model_name` should reason before answering.
    ///
    /// Resolution: project override → workspace (app-default) entry →
    /// `slp_thinking_enabled` → the model class's `thinking_default()` (via
    /// `LlmFactory::thinking_for`).
    ///
    /// Returns `None` for a model that does not declare thinking, which means
    /// "bind nothing": its chat template never reads `enable_thinking`, so a
    /// stored preference for it would be a setting that silently does nothing.
    pub fn effective_model_thinking(&self, model_name: &str, project: Option<&Project>) -> Option<bool> {
        if !self.supports_thinking(model_name) {
            return None;
        }
        if let Some(thinking) = self.project_model(project, model_name).and_then(|pm| pm.thinking) {
            return Some(thinking);
        }
        if let Some(thinking) = self.get_model(Some(model_name)).and_then(|ws| ws.thinking) {
            return Some(thinking);
        }
        LlmFactory::thinking_for(Some(model_name), None)
    }

    /// Set `model_name`'s reasoning preference on the project; `None` clears
    /// the override so the model falls back to the workspace default.
    /// Returns `None` without writing anything for the bootstrap model, which
    /// has no per-project entry (or for an unknown project).
    pub fn set_project_model_thinking(
        &mut self,
        project_id: &str,
        model_name: &str,
        thinking: Option<bool>,
    ) -> Option<ProjectModel> {
        if self.is_bootstrap_model(model_name) {
            return None;
        }
        if !self.sync_project_models(project_id) {
            return None;
        }
        let target = canonical_model_name(model_name);
        let project = self.project_mut(project_id)?;
        match project.model_settings.iter_mut().find(|m| canonical_model_name(&m.model_name) == target) {
            Some(pm) => {
                pm.thinking = thinking;
                Some(pm.clone())
            }
            None => {
                let pm = ProjectModel { thinking, ..ProjectModel::new(model_name) };
                project.model_settings.push(pm.clone());
                Some(pm)
            }
        }
    }

    /// Whether `model_name`'s weights are present in the workspace.
    ///
    /// Downloads are workspace-wide — the weights live in one shared
    /// `models_dir` and are added or removed for every project at once — so
    /// this takes no project argument.
    pub fn effective_model_downloaded(&self, model_name: &str) -> bool {
        self.get_model(Some(model_name)).map(|ws| ws.downloaded).unwrap_or(false)
    }

    /// Whether `model_name` is the active selection for `project`.
    pub fn effective_model_active(&self, model_name: &str, project: Option<&Project>) -> bool {
        if let Some(pm) = self.project_model(project, model_name) {
            return pm.is_active;
        }
        self.get_model(Some(model_name)).map(|ws| ws.is_active).unwrap_or(false)
    }

    /// Freeze what `model_name` currently resolves to, per project.
    ///
    /// A project with no `ProjectModel` entry reads through to the workspace
    /// app-default, so changing that default would retroactively change the
    /// project.  Writing today's answer down first means the new default only
    /// reaches projects created after.  Only this model is pinned: a blanket
    /// sync would also detach every unrelated model from future defaults.
    ///
    /// Does nothing in a per-project agent child: writing every *sibling*
    /// project's settings is precisely what a pinned process may not do — its
    /// `save()` writes one file.  Nor in a supervisor that has delegated its
    /// projects: there `save()` writes no project file at all, so the entries
    /// appended here would be discarded just as silently.  Each project
    /// instead pins itself lazily through `sync_project_models` on its next
    /// project-route read or write, where the child that owns the file is the
    /// one doing the writing.
    pub fn pin_model_for_existing_projects(&mut self, model_name: &str) {
        if let Some(pinned) = pinned_project_id() {
            tracing::warn!(
                target: LOG_TARGET,
                "[workspace] not pinning {} across projects: this process owns only project {}",
                model_name,
                pinned
            );
            return;
        }
        if projects_delegated() {
            tracing::warn!(
                target: LOG_TARGET,
                "[workspace] not pinning {} across projects: their state belongs to the per-project agents",
                model_name
            );
            return;
        }
        if self.is_bootstrap_model(model_name) {
            return; // holds no per-project entry by design
        }
        let additions: Vec<(usize, ProjectModel)> = self
            .projects
            .iter()
            .enumerate()
            .filter(|(_, project)| self.project_model(Some(project), model_name).is_none())
            .map(|(index, project)| {
                (
                    index,
                    ProjectModel {
                        enabled: self.effective_model_enabled(model_name, Some(project)),
                        is_active: self.effective_model_active(model_name, Some(project)),
                        thinking: self.effective_model_thinking(model_name, Some(project)),
                        ..ProjectModel::new(model_name)
                    },
                )
            })
            .collect();
        for (index, pm) in additions {
            self.projects[index].model_settings.push(pm);
        }
    }

    /// Make `model_name` an app-default for projects created from now on.
    ///
    /// Called when a model is *downloaded* (`/v1/init`), not when one is
    /// switched on: fetching a model is the act that says it belongs to this
    /// workspace.  One-way — nothing demotes a model.  Existing projects are
    /// pinned first and keep whatever they resolve to today.
    pub fn promote_model_to_workspace_default(&mut self, model_name: &str) {
        if self.is_bootstrap_model(model_name) {
            return; // always enabled, not a configurable default
        }
        if let Some(pinned) = pinned_project_id() {
            // `enabled` lives on the workspace record, which a pinned process
            // never writes.  Promoting here would look like it worked and be
            // gone on the next load — along with the pin that keeps it from
            // leaking into projects that never asked for it.
            tracing::warn!(
                target: LOG_TARGET,
                "[workspace] not promoting {} to a workspace default: this process owns only project {}",
                model_name,
                pinned
            );
            return;
        }
        match self.get_model(Some(model_name)) {
            Some(ws) if !ws.enabled => {}
            _ => return, // nothing to leak, skip the pin
        }
        self.pin_model_for_existing_projects(model_name);
        if let Some(ws) = self.get_model_mut(Some(model_name)) {
            ws.enabled = true;
        }
    }

    /// Set `model_name`'s enabled state on the project, upserting an entry.
    /// `None` for the bootstrap model (or an unknown project).
    pub fn set_project_model_enabled(
        &mut self,
        project_id: &str,
        model_name: &str,
        enabled: bool,
    ) -> Option<ProjectModel> {
        if self.is_bootstrap_model(model_name) {
            return None;
        }
        if !self.sync_project_models(project_id) {
            return None;
        }
        let target = canonical_model_name(model_name);
        let project = self.project_mut(project_id)?;
        match project.model_settings.iter_mut().find(|m| canonical_model_name(&m.model_name) == target) {
            Some(pm) => {
                pm.enabled = enabled;
                Some(pm.clone())
            }
            None => {
                let pm = ProjectModel { enabled, ..ProjectModel::new(model_name) };
                project.model_settings.push(pm.clone());
                Some(pm)
            }
        }
    }

    /// Set `model_name` as active on the project, deactivating the others.
    /// `None` for the bootstrap model (or an unknown project).
    pub fn set_project_model_active(
        &mut self,
        project_id: &str,
        model_name: &str,
        active: bool,
    ) -> Option<ProjectModel> {
        if self.is_bootstrap_model(model_name) {
            return None;
        }
        if !self.sync_project_models(project_id) {
            return None;
        }
        let target = canonical_model_name(model_name);
        let project = self.project_mut(project_id)?;
        let index = match project
            .model_settings
            .iter()
            .position(|m| canonical_model_name(&m.model_name) == target)
        {
            Some(index) => index,
            None => {
                project
                    .model_settings
                    .push(ProjectModel { is_active: active, ..ProjectModel::new(model_name) });
                project.model_settings.len() - 1
            }
        };
        if active {
            for (i, other) in project.model_settings.iter_mut().enumerate() {
                other.is_active = i == index;
            }
        } else {
            project.model_settings[index].is_active = false;
        }
        Some(project.model_settings[index].clone())
    }

    /// Index of the registered model `model_name` resolves to (alias-aware);
    /// `None` means the default model.
    fn find_model_index(&self, model_name: Option<&str>) -> Option<usize> {
        let model_name = match model_name {
            Some(name) => name.to_string(),
            None => AppSettings::new().default_model,
        };
        let target = canonical_model_name(&model_name);
        let target_cls = SUPPORTED_MODELS.get(&target);

        self.models.iter().position(|m| {
            // Compare by class where both sides are known models, so two names
            // for the same model still match; otherwise fall back to the name,
            // which keeps unknown (user-added) models from colliding with each
            // other on a shared `None` class.
            if canonical_model_name(&m.model_name) == target {
                return true;
            }
            match target_cls {
                Some(target_cls) => SUPPORTED_MODELS.get(&m.model_name) == Some(target_cls),
                None => false,
            }
        })
    }

    /// Look up a registered model by its short name (alias-aware); `None`
    /// means the default model.  Returns a snapshot with `models_dir` set.
    ///
    /// Matching is alias-aware, so a record written under a model's old name
    /// (e.g. a workspace from before `slp-orchestra-mini` became `sl-mini`) is
    /// found by its current name and vice versa.
    pub fn get_model(&self, model_name: Option<&str>) -> Option<WorkspaceModel> {
        let index = self.find_model_index(model_name)?;
        let mut model = self.models[index].clone();
        model.models_dir = self.models_dir.clone();
        Some(model)
    }

    /// Mutable form of [`Workspace::get_model`]; also stamps the record's
    /// `models_dir` from the workspace.
    pub fn get_model_mut(&mut self, model_name: Option<&str>) -> Option<&mut WorkspaceModel> {
        let index = self.find_model_index(model_name)?;
        let models_dir = self.models_dir.clone();
        let model = &mut self.models[index];
        model.models_dir = models_dir;
        Some(model)
    }

    /// Register a model without changing the active model; merges `config`
    /// into an existing record.  Returns a snapshot of the (new or existing)
    /// record.
    ///
    /// The record is stored under the canonical name, so a legacy name coming
    /// from the cloud catalog is not written back into `config.json`.
    /// `enabled` seeds the app-default that `sync_project_models` copies into
    /// every project, so a caller registering a model on one project's behalf
    /// should pass `false` and enable it on that project itself.
    pub fn add_model(
        &mut self,
        model_name: &str,
        model_id: &str,
        config: Option<ModelConfig>,
        enabled: bool,
    ) -> WorkspaceModel {
        let model_name = canonical_model_name(model_name);
        if let Some(existing) = self.get_model_mut(Some(&model_name)) {
            if let Some(config) = config {
                existing.config = Some(match &existing.config {
                    Some(current) => merge_model_config(current, &config),
                    None => config,
                });
            }
            return existing.clone();
        }
        let has_active = self.models.iter().any(|m| m.is_active);
        // The bootstrap model gets no per-project entry, so its always-on state
        // resolves through this record — never register it disabled.
        let enabled = if self.is_bootstrap_model(&model_name) { true } else { enabled };
        let entry = WorkspaceModel {
            model_name,
            model_id: model_id.to_string(),
            config,
            enabled,
            is_active: !has_active,
            models_dir: self.models_dir.clone(),
            ..Default::default()
        };
        self.models.push(entry.clone());
        entry
    }

    /// Register a remote model, or refresh the catalog details of one already
    /// registered.  It is never the active (bootstrap) model and is
    /// `downloaded` from the start, since there is nothing to fetch.
    pub fn add_remote_model(
        &mut self,
        model_name: &str,
        url: &str,
        capabilities: Vec<String>,
        context_length: Option<u32>,
        enabled: bool,
    ) -> WorkspaceModel {
        let model_name = canonical_model_name(model_name);
        if let Some(existing) = self.get_model_mut(Some(&model_name)) {
            existing.kind = ModelKind::Remote;
            existing.url = Some(url.to_string());
            existing.capabilities = capabilities;
            existing.context_length = context_length;
            existing.downloaded = true;
            existing.is_active = false;
            return existing.clone();
        }
        let entry = WorkspaceModel {
            model_name,
            downloaded: true,
            enabled,
            capabilities,
            kind: ModelKind::Remote,
            url: Some(url.to_string()),
            context_length,
            models_dir: self.models_dir.clone(),
            ..Default::default()
        };
        self.models.push(entry.clone());
        entry
    }

    /// Sort `candidates` best-first by [`WorkspaceModel::rank`] (stable, so
    /// equal ranks keep registration order) and stamp their `models_dir`.
    fn rank_candidates(&self, mut candidates: Vec<WorkspaceModel>) -> Vec<WorkspaceModel> {
        candidates.sort_by(|a, b| {
            b.rank().partial_cmp(&a.rank()).unwrap_or(std::cmp::Ordering::Equal)
        });
        for m in &mut candidates {
            m.models_dir = self.models_dir.clone();
        }
        candidates
    }

    /// The best enabled, downloaded model for `project`, or `None`.
    ///
    /// Ranks by parameter count first (the largest model wins), then by model
    /// version as a tie-breaker between same-size models.  When `project` is
    /// given, its per-project model settings override the workspace defaults.
    pub fn largest_downloaded_model(&self, project: Option<&Project>) -> Option<WorkspaceModel> {
        let candidates: Vec<WorkspaceModel> = self
            .models
            .iter()
            .filter(|m| {
                self.effective_model_enabled(&m.model_name, project)
                    && self.effective_model_downloaded(&m.model_name)
                    && m.param_size_b() > 0.0
            })
            .cloned()
            .collect();
        self.rank_candidates(candidates).into_iter().next()
    }

    /// Enabled, downloaded models declaring `capability`, best-first.
    ///
    /// Remote models are included; callers that route place them after
    /// every local model (see `ModelService::get_llm_for_capability`).
    ///
    /// Callers walk this list down to a smaller same-capability model when
    /// the largest doesn't fit in memory, rather than dropping straight to the
    /// tiny default.  Empty when no downloaded model declares the capability.
    /// Per-project settings (`project`) override the workspace defaults.
    pub fn ranked_models_for(&self, capability: &str, project: Option<&Project>) -> Vec<WorkspaceModel> {
        let cap = capability.to_lowercase();
        let candidates: Vec<WorkspaceModel> = self
            .models
            .iter()
            .filter(|m| {
                self.effective_model_enabled(&m.model_name, project)
                    && self.effective_model_downloaded(&m.model_name)
                    && (m.param_size_b() > 0.0 || m.is_remote())
                    && m.effective_capabilities().iter().any(|c| c.to_lowercase() == cap)
            })
            .cloned()
            .collect();
        self.rank_candidates(candidates)
    }

    /// The single best model for a capability — the largest (by size, then
    /// newer version) that declares it — or the largest downloaded model when
    /// none declares it.
    ///
    /// Memory is NOT considered here; callers that must fit the model in VRAM
    /// use [`Workspace::ranked_models_for`] and walk down to a smaller one.
    pub fn best_model_for(&self, capability: &str, project: Option<&Project>) -> Option<WorkspaceModel> {
        let ranked = self.ranked_models_for(capability, project);
        if let Some(best) = ranked.into_iter().next() {
            return Some(best);
        }
        self.largest_downloaded_model(project)
    }

    /// The workspace-level directory for a given `model_id`.
    pub fn model_dir_for(&self, model_id: &str) -> String {
        join(&self.models_dir, model_id)
    }

    /// The workspace-level GGUF file path for a given `model_id`.
    pub fn model_path_for(&self, model_id: &str) -> String {
        join(&join(&self.models_dir, model_id), &format!("{}.gguf", path_name(model_id)))
    }

    /// Cascade: quantize-matched bin GGUF → first bin GGUF → exact workspace
    /// GGUF → first main GGUF in model dir → exact path.
    pub fn resolve_gguf_path(&self, model_id: &str) -> String {
        // Honor the requested model_id for the bin_dir cascade (fall back to
        // the active model) so resolving a non-active model — e.g. the small
        // orchestrator model while a larger model is active — returns that
        // model's GGUF, not the active one's.
        let ws_model = match self.models.iter().find(|m| m.model_id == model_id) {
            Some(m) => {
                let mut m = m.clone();
                m.models_dir = self.models_dir.clone();
                Some(m)
            }
            None => self.get_model(None),
        };
        if let Some(ws_model) = ws_model {
            let bin_dir = ws_model.bin_dir();
            if !bin_dir.is_empty() && Path::new(&bin_dir).is_dir() {
                let quantize = ws_model
                    .config
                    .as_ref()
                    .map(|c| c.quantize.to_uppercase())
                    .unwrap_or_default();
                let gguf_files: Vec<String> = std::fs::read_dir(&bin_dir)
                    .map(|entries| {
                        entries
                            .filter_map(|e| e.ok())
                            .map(|e| e.file_name().to_string_lossy().into_owned())
                            .filter(|f| f.ends_with(".gguf") && !is_mmproj_gguf(f))
                            .collect()
                    })
                    .unwrap_or_default();
                if !quantize.is_empty() {
                    let suffix = format!("-{quantize}.GGUF");
                    if let Some(f) = gguf_files.iter().find(|f| f.to_uppercase().ends_with(&suffix)) {
                        return join(&bin_dir, f);
                    }
                }
                if let Some(f) = gguf_files.first() {
                    return join(&bin_dir, f);
                }
            }
        }
        let exact = self.model_path_for(model_id);
        if Path::new(&exact).exists() {
            return exact;
        }
        // Fallback: find any main .gguf in the model directory.  The mmproj /
        // CLIP projector GGUF is excluded — it is not a loadable main model.
        let model_dir = self.model_dir_for(model_id);
        if Path::new(&model_dir).is_dir()
            && let Ok(entries) = std::fs::read_dir(&model_dir)
        {
            for entry in entries.filter_map(|e| e.ok()) {
                let f = entry.file_name().to_string_lossy().into_owned();
                if f.ends_with(".gguf") && !is_mmproj_gguf(&f) {
                    return join(&model_dir, &f);
                }
            }
        }
        exact
    }

    /// The `config.json` document, with `projects` as its project list:
    /// `{"id": <active model_id or "">, "workspace": {...}}`.
    fn workspace_record(&self, projects: Vec<Value>) -> Value {
        // Python's `model_dump` evaluates the computed `current_project`
        // field, which is what warns about a stale binding at save time.
        let _ = self.current_project();
        let mut ws_data = match serde_json::to_value(self) {
            Ok(Value::Object(map)) => map,
            _ => Map::new(),
        };
        for key in RUNTIME_WORKSPACE_KEYS {
            ws_data.shift_remove(*key);
        }
        ws_data.insert("projects".to_string(), Value::Array(projects));
        let active_id = self.active_model().map(|m| m.model_id).unwrap_or_default();
        json!({"id": active_id, "workspace": ws_data})
    }

    /// Persist workspace state, each project into its own file.
    ///
    /// Serialized and atomic.  Several background tasks write the workspace
    /// from their own threads — document indexing, MCP provisioning — so every
    /// file is snapshotted under the lock and swapped in with a rename, and a
    /// reader (or a crash) sees either the old file or the new one.
    ///
    /// What gets written depends on what this process owns:
    ///
    /// * a per-project agent child (pinned) writes **only** its own
    ///   `project.json` — the workspace record belongs to the supervisor, and a
    ///   child writing it would clobber its siblings;
    /// * the supervisor, which has delegated every project to a child, writes
    ///   the workspace record and *nothing else* — unless it has explicitly
    ///   claimed one project with `own_project`, which is how create, import
    ///   and delete still write a file no child owns yet;
    /// * any other unpinned process — the single server, the CLI — writes the
    ///   workspace record and every project's file.
    ///
    /// `SLP_LEGACY_CONFIG=1` restores the single-document format.
    pub fn save(&self) -> std::io::Result<()> {
        let _guard = SAVE_LOCK.lock();
        if legacy_layout() {
            let projects = match serde_json::to_value(&self.projects).map_err(std::io::Error::other)? {
                Value::Array(projects) => projects,
                _ => Vec::new(),
            };
            return atomic_write_json(&Self::config_file(), &self.workspace_record(projects));
        }

        if let Some(pinned) = pinned_project_id().filter(|p| !p.is_empty()) {
            match self.projects.iter().find(|p| p.id == pinned) {
                Some(project) => project.save_state()?,
                None => {
                    tracing::warn!(
                        target: LOG_TARGET,
                        "[workspace] pinned to project {}, which is not in this workspace — nothing saved",
                        pinned
                    );
                }
            }
            return Ok(());
        }

        for project in self.writable_projects() {
            project.save_state()?;
        }
        let index: Vec<Value> = self.projects.iter().map(|p| Value::Object(p.to_index())).collect();
        atomic_write_json(&Self::config_file(), &self.workspace_record(index))
    }

    /// The projects whose own file this unpinned process may write.
    ///
    /// Every one of them, unless the projects have been delegated to
    /// per-project children — then only the one explicitly claimed, if any.
    fn writable_projects(&self) -> Vec<&Project> {
        if !projects_delegated() {
            return self.projects.iter().collect();
        }
        match owned_project_id().filter(|p| !p.is_empty()) {
            None => Vec::new(),
            Some(owned) => self.projects.iter().filter(|p| p.id == owned).collect(),
        }
    }

    /// Fold each project's own `project.json` into the raw workspace record
    /// before deserialization.
    ///
    /// Called on the raw document before validation, so a project is built
    /// once from its complete state.  `project_dir` is derived here from the
    /// current home, never from what was persisted, because the file has to
    /// be found before the project object that would say where it is exists.
    /// A project with no file of its own keeps whatever `config.json` holds
    /// for it, so a half-migrated workspace still loads completely.
    fn hydrate_projects(mut ws_data: Map<String, Value>) -> Map<String, Value> {
        if legacy_layout() {
            return ws_data;
        }

        let workspace_id = ws_data.get("id").and_then(Value::as_str).unwrap_or("").to_string();
        let workspace_dir = join(&home_dir(), &workspace_id);
        let entries = match ws_data.get("projects") {
            Some(Value::Array(entries)) => entries.clone(),
            _ => Vec::new(),
        };
        let mut hydrated = Vec::with_capacity(entries.len());
        for entry in entries {
            let project_id = entry
                .as_object()
                .and_then(|e| e.get("id"))
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .map(str::to_string);
            let Some(project_id) = project_id else {
                hydrated.push(entry);
                continue;
            };
            let Some(state) = read_project_state(&join(&workspace_dir, &project_id)).filter(|s| !s.is_empty())
            else {
                hydrated.push(entry);
                continue;
            };
            // The project's own file wins for its own state; the workspace
            // record stays authoritative for the selection.
            let entry_map = entry.as_object().cloned().unwrap_or_default();
            let mut merged = entry_map.clone();
            for (key, value) in state {
                merged.insert(key, value);
            }
            for field in WORKSPACE_OWNED_PROJECT_FIELDS {
                if let Some(value) = entry_map.get(*field) {
                    merged.insert((*field).to_string(), value.clone());
                }
            }
            hydrated.push(Value::Object(merged));
        }
        // `ws_data["projects"] = hydrated`, where a missing or null list
        // hydrates to an empty one; anything else is left for validation to
        // reject.
        if matches!(ws_data.get("projects"), None | Some(Value::Null) | Some(Value::Array(_))) {
            ws_data.insert("projects".to_string(), Value::Array(hydrated));
        }
        ws_data
    }

    /// Parse a `config.json` document into a workspace, re-anchored on the
    /// current home.  `Err` carries the reason it could not be used.
    fn parse_record(content: &str) -> Result<Workspace, String> {
        let data: Value = serde_json::from_str(content).map_err(|e| e.to_string())?;
        let data = data.as_object().ok_or("config.json is not a JSON object")?;
        let mut ws_data = match data.get("workspace") {
            None => Map::new(),
            Some(Value::Object(ws_data)) => ws_data.clone(),
            Some(_) => return Err("the workspace record is not a JSON object".to_string()),
        };
        // Strip computed/runtime fields that aren't part of the model.
        for key in RUNTIME_WORKSPACE_KEYS {
            ws_data.shift_remove(*key);
        }
        let ws_data = Self::hydrate_projects(ws_data);

        let mut workspace: Workspace =
            serde_json::from_value(Value::Object(ws_data)).map_err(|e| e.to_string())?;

        // Re-resolve paths against the current home.
        workspace.workspace_dir = join(&home_dir(), &workspace.id);
        workspace.models_dir = join(&workspace.workspace_dir, "models");
        Self::rebase_paths(&mut workspace);
        Ok(workspace)
    }

    /// A fresh workspace with a new uuid and its models dir created.
    fn create_new_workspace(_config: &ModelConfig) -> Workspace {
        let workspace_id = uuid::Uuid::new_v4().to_string();
        let workspace_dir = join(&home_dir(), &workspace_id);
        let models_dir = join(&workspace_dir, "models");

        if let Err(err) = std::fs::create_dir_all(&models_dir) {
            tracing::warn!(target: LOG_TARGET, "Workspace.init: could not create {}: {}", models_dir, err);
        }

        Workspace::new(workspace_id).with_workspace_dir(workspace_dir).with_models_dir(models_dir)
    }

    /// Load the workspace singleton from `config.json` on disk.
    ///
    /// Returns the cached instance if already loaded; creates a new workspace
    /// when the file is missing, empty or invalid.  Merges `config` into the
    /// requested model's record (without changing the active model).
    pub fn init(config: &ModelConfig) -> SharedWorkspace {
        let mut instance = INSTANCE.lock();
        if let Some(existing) = instance.as_ref() {
            return Arc::clone(existing);
        }

        let config_file = Self::config_file();
        let workspace = if !Path::new(&config_file).exists() {
            Self::create_new_workspace(config)
        } else {
            match std::fs::read_to_string(&config_file) {
                Err(err) => {
                    tracing::warn!(
                        target: LOG_TARGET,
                        "Workspace.init: failed to load config.json, creating new workspace: {}",
                        err
                    );
                    Self::create_new_workspace(config)
                }
                Ok(content) if content.trim().is_empty() => Self::create_new_workspace(config),
                Ok(content) => match Self::parse_record(content.trim()) {
                    Ok(mut workspace) => {
                        // Merge config into the requested WorkspaceModel
                        // (without changing active).
                        if let Some(ws_model) = workspace.get_model_mut(Some(&config.model_name)) {
                            ws_model.config = Some(match &ws_model.config {
                                Some(current) => merge_model_config(current, config),
                                None => config.clone(),
                            });
                        }
                        workspace
                    }
                    Err(err) => {
                        tracing::warn!(
                            target: LOG_TARGET,
                            "Workspace.init: failed to load config.json, creating new workspace: {}",
                            err
                        );
                        Self::create_new_workspace(config)
                    }
                },
            }
        };

        let shared = Arc::new(RwLock::new(workspace));
        *instance = Some(Arc::clone(&shared));
        shared
    }

    /// Re-anchor every persisted absolute path onto the current home.
    ///
    /// The framework must be self-contained: copy `<home_dir>` somewhere else
    /// and it has to run as-is.  `config.json` stores absolute paths that were
    /// correct when written, so nothing persisted is trusted as a location:
    ///
    /// * `project_dir` — `<workspace_dir>/<project_id>` by construction;
    /// * `document.path` — repointed at the project's own `documents/` folder
    ///   whenever the file is actually there.
    ///
    /// `project.cwd` is deliberately left alone: it is the user's own folder,
    /// not framework state.
    fn rebase_paths(workspace: &mut Workspace) {
        let workspace_dir = workspace.workspace_dir.clone();
        for project in &mut workspace.projects {
            if !project.id.is_empty() {
                project.project_dir = join(&workspace_dir, &project.id);
            }
            let docs_dir = project.documents_dir();
            for doc in &mut project.documents {
                if doc.path.is_empty() {
                    continue;
                }
                // The framework's own copy always wins, even when the stored
                // path still resolves.  After a migration both exist, and
                // leaving the pointer on the old user-space copy would keep
                // the install dependent on a directory outside <home_dir>.
                let file_name = match doc.name.as_deref() {
                    Some(name) if !name.is_empty() => name.to_string(),
                    _ => basename(&doc.path),
                };
                let candidate = join(&docs_dir, &file_name);
                if Path::new(&candidate).is_file() {
                    doc.path = candidate;
                }
            }
        }
    }

    /// Load the workspace directly from `config.json` on disk, without
    /// touching the singleton or merging any model config.  `None` when the
    /// file is missing, empty, or invalid.
    pub fn load() -> Option<Workspace> {
        let config_file = Self::config_file();
        if !Path::new(&config_file).exists() {
            return None;
        }
        let content = std::fs::read_to_string(&config_file).ok()?;
        let content = content.trim();
        if content.is_empty() {
            return None;
        }
        Self::parse_record(content).ok()
    }

    /// Re-read one project's own file if someone has written it since.
    ///
    /// Under supervision a project's own state belongs to the child pinned to
    /// it, and this process holds a copy taken when it first learned the
    /// project existed.  Narrower than [`Workspace::reload`]: called on the
    /// host's request path, where the bound project is the only one that
    /// matters.  Gated on the file's mtime, so the usual cost is one `stat`.
    ///
    /// The workspace record stays authoritative for the fields it owns (see
    /// [`WORKSPACE_OWNED_PROJECT_FIELDS`]), and the existing [`Project`] is
    /// updated in place rather than replaced, down to the records inside it.
    /// Returns true when something was re-read.
    pub fn refresh_project(&mut self, project_id: &str) -> bool {
        if project_id.is_empty() {
            return false;
        }
        let Some(index) = self.projects.iter().position(|p| p.id == project_id) else {
            return false;
        };
        let project_dir = self.projects[index].project_dir.clone();
        if project_dir.is_empty() {
            return false;
        }
        let Some(mtime) = mtime_ns(&project_state_path(&project_dir)) else {
            return false;
        };
        if self.state_mtimes.get(project_id) == Some(&mtime) {
            return false;
        }
        let state = read_project_state(&project_dir);
        // Record the mtime even for an unreadable file: a project.json that is
        // missing or corrupt is not going to become readable by being stat-ed
        // once per frame, and read_project_state has already logged it.
        self.state_mtimes.insert(project_id.to_string(), mtime);
        let Some(mut state) = state.filter(|s| !s.is_empty()) else {
            return false;
        };
        state.insert("id".to_string(), Value::String(self.projects[index].id.clone()));
        state.insert("project_dir".to_string(), Value::String(project_dir));
        let fresh: Project = match serde_json::from_value(Value::Object(state)) {
            Ok(fresh) => fresh,
            Err(err) => {
                tracing::warn!(
                    target: LOG_TARGET,
                    "[project-state] could not read project {}: {}",
                    project_id,
                    err
                );
                return false;
            }
        };
        self.projects[index].carry_project_fields(fresh, true);
        true
    }

    /// Re-read the registered models if the workspace record has moved on.
    ///
    /// The mirror of [`Workspace::refresh_project`]: it carries the workspace
    /// owner's write *down* to a child.  Weights are workspace-wide — one
    /// shared `models_dir`, one registry in `config.json` — so a model
    /// downloaded while a child is running would otherwise be invisible to
    /// it.  Only the model registry is taken: adopting a sibling's project
    /// record here would undo the split.  Returns true when something was
    /// re-read.
    pub fn refresh_models(&mut self) -> bool {
        let config_file = Self::config_file();
        let Some(mtime) = mtime_ns(&config_file) else {
            return false;
        };
        if self.models_mtime == Some(mtime) {
            return false;
        }
        // Record it either way: an unreadable or malformed config will not
        // become readable by being stat-ed again on the next request.
        self.models_mtime = Some(mtime);
        let Ok(content) = std::fs::read_to_string(&config_file) else {
            return false;
        };
        let content = content.trim();
        if content.is_empty() {
            return false;
        }
        let Ok(data) = serde_json::from_str::<Value>(content) else {
            return false;
        };
        let Some(models) = data
            .as_object()
            .and_then(|d| d.get("workspace"))
            .and_then(Value::as_object)
            .and_then(|ws| ws.get("models"))
            .and_then(Value::as_array)
        else {
            return false;
        };
        let fresh: Result<Vec<WorkspaceModel>, _> =
            models.iter().map(|m| serde_json::from_value::<WorkspaceModel>(m.clone())).collect();
        let Ok(mut fresh) = fresh else {
            tracing::warn!(
                target: LOG_TARGET,
                "[workspace] could not read the model registry from {}; keeping the one we have",
                config_file
            );
            return false;
        };
        for model in &mut fresh {
            // Same re-anchoring `load` does: nothing persisted is trusted as
            // a location, so the weights are found under *this* home.
            model.models_dir = self.models_dir.clone();
        }
        self.models = fresh;
        true
    }

    /// Re-read project state from disk into this instance, updating projects
    /// and their records in place (merge by id).
    ///
    /// What this does *not* do is make an unsaved mutation survive a reload
    /// that lands on top of it — the fields still come from disk.  A pass that
    /// mutates and then saves has to treat that pair as one step.
    pub fn reload(&mut self) {
        let Some(fresh) = Workspace::load() else {
            return;
        };
        merge_records_by_id(&mut self.projects, fresh.projects);
    }

    /// Clear the cached singleton (useful after model switch or tests).
    pub fn reset() {
        *INSTANCE.lock() = None;
    }

    /// Return the current singleton instance, or `None`.
    pub fn get_instance() -> Option<SharedWorkspace> {
        INSTANCE.lock().clone()
    }

    /// Index of the project [`Workspace::current_project`] resolves to.
    fn current_project_index(&self) -> Option<usize> {
        if let Some(bound_id) = active_project_id().filter(|p| !p.is_empty()) {
            if let Some(index) = self.projects.iter().position(|p| p.id == bound_id) {
                return Some(index);
            }
            // Bound to a project this workspace doesn't have.  Falling back to
            // the flag would silently answer for the wrong project, so say so —
            // but don't fail: a stale binding must not take down the request.
            tracing::warn!(
                target: LOG_TARGET,
                "[project-context] bound project {} is not in this workspace; falling back to the selected project",
                bound_id
            );
        }
        self.projects.iter().position(|p| p.current)
    }

    /// The project this unit of work belongs to.
    ///
    /// Resolution is scoped before it is global.  A request that named a
    /// `project_id` binds it for its own lifetime (and a per-project agent
    /// child pins one for the whole process), so concurrent chats on
    /// different projects each resolve their own.  The `current` flag remains
    /// the fallback, and means only "the project the UI has selected".
    pub fn current_project(&self) -> Option<&Project> {
        self.current_project_index().map(|index| &self.projects[index])
    }

    /// Mutable form of [`Workspace::current_project`].
    pub fn current_project_mut(&mut self) -> Option<&mut Project> {
        let index = self.current_project_index()?;
        Some(&mut self.projects[index])
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
