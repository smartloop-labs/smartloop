//! Port of `smartloop/api/services/storage_service.py`.
//!
//! Disk-storage reporting and cleanup for the SLP home directory.
//!
//! Reports what `<home_dir>` is spending disk on and reclaims the parts that
//! are safe to lose.  The framework owns that state, so it does the work here
//! and the client just asks.
//!
//! Two rules shape everything below:
//!
//! * **Sizing and deleting read the same list.**  Each `scan_*` helper returns
//!   candidate rows and both [`compute_usage`] and [`run_cleanup`] work from
//!   them, so what the UI offers to free is exactly what gets freed.
//! * **A path is deletable only if it matches a known shape.**  Validation is
//!   positive (this is a `models/<org>/<repo>` directory) rather than a list of
//!   things to avoid, and every candidate is additionally checked to resolve
//!   inside `<home_dir>` and not to be a symlink.
//!
//! Models are kept when the workspace says they are in use — `enabled` (the
//! picker's flag), `is_active`, the default orchestrator, or resident in
//! `ModelService` — plus the bundled `smartloop-ai` and `embeddings` trees,
//! which the framework cannot run without.
//!
//! One case overrides all of that: a directory holding only a `.part` sidecar
//! is a download that died, and none of those reasons can be true of bytes
//! that never finished arriving.  Such a tree is removable, because the next
//! download deletes it anyway — see [`weights_state`].
//!
//! Except when the download has not died.  A transfer in progress leaves
//! exactly the same thing on disk, so that override is gated on
//! `inflight::is_downloading`, which answers the one question the directory
//! cannot: is something still writing here.  Without it the sweep deletes a
//! running download out from under the downloader.

#![allow(unused_imports)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use regex::Regex;
use serde_json::{Map, Value};

use crate::config::app_settings::AppSettings;
use crate::model::downloader::inflight;
use crate::utils::logging_setup::{truncate_log_file, DEFAULT_BACKUP_COUNT};
use crate::workspace::workspace::{is_mmproj_gguf, SharedWorkspace, Workspace};

/// Logger name used by the Python module (tests capture it).
pub const LOG_TARGET: &str = "smartloop.api.services.storage_service";

/// Model trees the framework needs to function: the bundled orchestrator and
/// the embedding model RAG depends on.  Matched on the org directory so an
/// unregistered leftover under them is protected too.
pub const BUNDLED_ORGS: &[&str] = &["smartloop-ai", "embeddings"];

/// How a model's bytes are presented, keyed by the keep reason the scan
/// already worked out.  `bundled` is its own group (`system`): infrastructure
/// rather than a choice someone made.  Everything else that is held is the
/// user's own footprint (`in_use`).
pub const MODEL_GROUPS: &[(&str, &str)] = &[
    ("bundled", "system"),
    ("downloading", "in_use"),
    ("loaded", "in_use"),
    ("default", "in_use"),
    ("active", "in_use"),
    ("enabled", "in_use"),
];

/// Installed framework payloads sit in `<home>/<x.y.z>/` next to everything else.
pub static VERSION_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\d+\.\d+\.\d+$").unwrap());

/// Regenerable caches.  Written by the desktop app / installer rather than by
/// us, but they live under our home and nothing else prunes them.
pub const CACHE_DIRS: &[&str] = &["cache", "templates"];

/// How long a project directory must sit untouched before the sweep will take
/// it.  An orphan is litter left by a bug; waiting a week costs nothing.
pub const ORPHAN_MIN_AGE_S: u64 = 7 * 24 * 60 * 60;

/// Never removed, whatever category is being swept: credentials, the
/// workspace record itself, liveness markers, install markers, and the
/// desktop app's own DB.
pub const NEVER_DELETE: &[&str] = &[
    "config.json", ".credentials", ".key", ".installed-version", ".archive-etag",
    ".binary-hash", "server.pid", "server.port", "version", "installed",
    "studio.db", "workspace.db", "token_usage.jsonl",
];

/// Windows can refuse to unlink a GGUF a worker still has mapped, so give the
/// mapping a moment to drop.
pub const DELETE_ATTEMPTS: u32 = 3;
/// Pause between delete attempts, in seconds.
pub const DELETE_BACKOFF_S: f64 = 0.25;

/// How long to wait for in-flight generation to finish before giving up on a
/// model sweep.  Deleting weights out from under a decoding context is a
/// native use-after-free, so this is a hard precondition.
pub const DRAIN_TIMEOUT_S: f64 = 30.0;

/// A candidate row / report object (Python `dict`), serialized verbatim.
pub type Row = Map<String, Value>;

/// Errors surfaced to the router: `ValueError` -> 400, `RuntimeError` -> 503.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StorageError {
    /// Python `ValueError`: a path or request that fails validation.
    #[error("{0}")]
    ValueError(String),
    /// Python `RuntimeError`: the service is busy or the workspace unavailable.
    #[error("{0}")]
    RuntimeError(String),
}

/// A counting semaphore guarding inference (`ModelService._inference_lock`).
pub trait InferencePermits: Send + Sync {
    /// Try to take one permit, waiting at most `timeout`.  True on success.
    fn acquire(&self, timeout: Duration) -> bool;
    /// Hand one permit back.
    fn release(&self);
}

/// What the storage service needs from `ModelService` (duck-typed in Python;
/// tests substitute fakes).
pub trait StorageModelService: Send + Sync {
    /// The live workspace singleton (`ModelService._ensure_workspace`).
    /// `Ok(None)` means "not initialized".
    fn ensure_workspace(&self) -> anyhow::Result<Option<SharedWorkspace>>;

    /// Re-read one project's own state file (`refresh_project_state`).
    fn refresh_project_state(&self, project_id: &str) -> anyhow::Result<bool> {
        let _ = project_id;
        Ok(false)
    }

    /// Model names holding a loaded context: the named slots
    /// (`model_name`, `decision_model_name`, `large_model_name`,
    /// `vision_model_name`) plus every live, non-faulted capability cache.
    fn resident_model_names(&self) -> HashSet<String> {
        HashSet::new()
    }

    /// `TaskQueue.get_status(name)` for a background task, if a queue exists.
    fn task_status(&self, name: &str) -> Option<Value> {
        let _ = name;
        None
    }

    /// The inference semaphore, if any.
    fn inference_lock(&self) -> Option<Arc<dyn InferencePermits>> {
        None
    }

    /// Unload a model without taking an inference permit
    /// (`_unload_model_by_name_unlocked`).
    fn unload_model_by_name_unlocked(&self, name: &str) -> anyhow::Result<()>;

    /// Tear down the shared LLM executor (`_reset_llm_executor`).
    fn reset_llm_executor(&self) {}
}

/// Which categories a [`run_cleanup`] call sweeps.  Nothing is deleted unless
/// its category is explicitly requested.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CleanupOptions {
    pub models: bool,
    pub model_ids: Option<Vec<String>>,
    pub versions: bool,
    pub logs: bool,
    pub cache: bool,
    pub orphan_projects: bool,
    pub dry_run: bool,
}

/// Replacement for [`remove`] (tests patch `_remove`).
pub type RemoveHook = Arc<dyn Fn(&str, bool) -> (bool, Option<String>) + Send + Sync>;
/// Replacement for `inflight::is_downloading` (tests patch `is_downloading`).
pub type IsDownloadingHook = Arc<dyn Fn(&str) -> bool + Send + Sync>;
/// Replacement for the directory removal (tests patch `shutil.rmtree`).
pub type RmtreeHook = Arc<dyn Fn(&str) -> std::io::Result<()> + Send + Sync>;

/// Module-level settings (Python `_settings`); tests override it.
static SETTINGS: Lazy<RwLock<Option<AppSettings>>> = Lazy::new(|| RwLock::new(None));
static REMOVE_HOOK: Lazy<RwLock<Option<RemoveHook>>> = Lazy::new(|| RwLock::new(None));
static IS_DOWNLOADING_HOOK: Lazy<RwLock<Option<IsDownloadingHook>>> = Lazy::new(|| RwLock::new(None));
static RMTREE_HOOK: Lazy<RwLock<Option<RmtreeHook>>> = Lazy::new(|| RwLock::new(None));

/// Override the module settings (tests patch `storage_service._settings`).
/// `None` restores `AppSettings::new()`.
pub fn set_settings(settings: Option<AppSettings>) {
    *SETTINGS.write() = settings;
}

/// Override [`remove`] for the sweep loop; `None` restores the real one.
pub fn set_remove_hook(hook: Option<RemoveHook>) {
    *REMOVE_HOOK.write() = hook;
}

/// Override the download-registry lookup; `None` restores the real one.
pub fn set_is_downloading_hook(hook: Option<IsDownloadingHook>) {
    *IS_DOWNLOADING_HOOK.write() = hook;
}

/// Override recursive directory removal; `None` restores `remove_dir_all`.
pub fn set_rmtree_hook(hook: Option<RmtreeHook>) {
    *RMTREE_HOOK.write() = hook;
}

// ── Paths and sizes ─────────────────────────────────────────────────────

/// The SLP home directory, resolved.
pub fn home() -> String {
    todo!()
}

/// Return `(bytes, files)` for a file or directory tree.  Symlinks are neither
/// followed nor counted; missing paths are 0.
pub fn walk_size(path: &str) -> (u64, u64) {
    let _ = path;
    todo!()
}

/// Immediate subdirectory names of `path`, skipping symlinks.  Sorted.
fn dirs_in(path: &str, skip_hidden: bool) -> Vec<String> {
    let _ = (path, skip_hidden);
    todo!()
}

/// Build a candidate row with its size filled in; `extra` is merged on top.
fn row(path: &str, removable: bool, keep_reason: Option<&str>, extra: Row) -> Row {
    let _ = (path, removable, keep_reason, extra);
    todo!()
}

// ── Workspace and model state ───────────────────────────────────────────

/// Return the live workspace, refreshing every project's state first.
/// `RuntimeError` when unavailable.
fn load_workspace(model_service: Option<&dyn StorageModelService>) -> Result<SharedWorkspace, StorageError> {
    let _ = model_service;
    todo!()
}

/// Model names currently holding a loaded context in `ModelService`.
fn resident_model_names(model_service: Option<&dyn StorageModelService>) -> HashSet<String> {
    let _ = model_service;
    todo!()
}

/// Map `model_id` -> why it must be kept, for every protected model.
fn keep_model_ids(workspace: &Workspace, model_service: Option<&dyn StorageModelService>) -> HashMap<String, String> {
    let _ = (workspace, model_service);
    todo!()
}

/// Which usage group a model's bytes belong to (`unused` when nothing keeps
/// it; unknown reasons count as `in_use`).
pub fn model_group(keep_reason: Option<&str>) -> &'static str {
    let _ = keep_reason;
    todo!()
}

// ── Scanners ────────────────────────────────────────────────────────────

/// Return `(has_weights, has_part)` for a model directory (and its `bin/`).
pub fn weights_state(model_dir: &str) -> (bool, bool) {
    let _ = model_dir;
    todo!()
}

/// One row per `models/<org>/<repo>` directory present on disk.
pub fn scan_models(workspace: &Workspace, model_service: Option<&dyn StorageModelService>) -> Vec<Row> {
    let _ = (workspace, model_service);
    todo!()
}

/// Version directory names that belong to the running install.
pub fn running_versions() -> HashSet<String> {
    todo!()
}

/// Installed framework payloads other than the running one.
pub fn scan_versions() -> Vec<Row> {
    todo!()
}

/// The server log and its rotated backups.
pub fn scan_logs() -> Vec<Row> {
    todo!()
}

/// Contents of the regenerable cache directories (the dirs themselves stay).
pub fn scan_cache() -> Vec<Row> {
    todo!()
}

/// Project directories under the workspace that `config.json` doesn't know about.
pub fn scan_orphan_projects(workspace: &Workspace) -> Vec<Row> {
    let _ = workspace;
    todo!()
}

/// Session transcripts, reported but never deleted.
pub fn scan_conversations(workspace: &Workspace) -> Vec<Row> {
    let _ = workspace;
    todo!()
}

// ── Validation ──────────────────────────────────────────────────────────

/// Resolve `path` and require it to sit strictly inside the home directory.
fn assert_inside_home(path: &str) -> Result<String, StorageError> {
    let _ = path;
    todo!()
}

/// Require `path` to match the expected shape for its category
/// (`models`, `versions`, `logs`, `cache`, `orphan_projects`).  Returns the
/// resolved path.
pub fn validate(path: &str, shape: &str, workspace: &Workspace) -> Result<String, StorageError> {
    let _ = (path, shape, workspace);
    todo!()
}

// ── Removal ─────────────────────────────────────────────────────────────

/// Delete (or empty, when `truncate`) `path`, retrying briefly.
/// Returns `(ok, error)`.
pub fn remove(path: &str, truncate: bool) -> (bool, Option<String>) {
    let _ = (path, truncate);
    todo!()
}

/// Remove `models/<org>` directories left empty by a model deletion.
fn prune_empty_orgs(workspace: &Workspace) {
    let _ = workspace;
    todo!()
}

/// `RuntimeError` while the indexer is writing to these trees.
fn refuse_if_busy(model_service: Option<&dyn StorageModelService>, needs_indexer: bool) -> Result<(), StorageError> {
    let _ = (model_service, needs_indexer);
    todo!()
}

/// Acquire every inference permit so no generation is in flight.  Returns the
/// number held; `RuntimeError` on timeout.
fn drain_inference(model_service: Option<&dyn StorageModelService>) -> Result<usize, StorageError> {
    let _ = model_service;
    todo!()
}

/// Release `held` inference permits.
fn release_inference(model_service: Option<&dyn StorageModelService>, held: usize) {
    let _ = (model_service, held);
    todo!()
}

// ── Public API ──────────────────────────────────────────────────────────

/// Roll candidate rows up into a category report.
fn category(rows: Vec<Row>, total_path: Option<&str>, extra: Row) -> Row {
    let _ = (rows, total_path, extra);
    todo!()
}

/// The models report, split into `in_use` / `system` / `reclaimable` /
/// `overhead` bands that partition `bytes`.
fn models_category(workspace: &Workspace, model_service: Option<&dyn StorageModelService>) -> Row {
    let _ = (workspace, model_service);
    todo!()
}

/// Report what the home directory is using and what can be reclaimed.
pub fn compute_usage(model_service: Option<&dyn StorageModelService>) -> Result<Row, StorageError> {
    let _ = model_service;
    todo!()
}

/// Resolve the requested categories into validated `(category, row)` work.
fn plan_cleanup(
    workspace: &Workspace,
    model_service: Option<&dyn StorageModelService>,
    opts: &CleanupOptions,
) -> Result<Vec<(String, Row)>, StorageError> {
    let _ = (workspace, model_service, opts);
    todo!()
}

/// Reclaim disk space in the selected categories.
///
/// Model deletion additionally unloads the model and rewrites its workspace
/// flags, so the picker stops offering a model whose weights are gone.
pub fn run_cleanup(model_service: Option<&dyn StorageModelService>, opts: CleanupOptions) -> Result<Row, StorageError> {
    let _ = (model_service, opts);
    todo!()
}

/// Remove what a cancelled download left behind, and clear its flags.
///
/// Deletes `models/<org>/<repo>` outright.  Call only once the transfer's
/// threads have stopped.
pub fn discard_partial_download(
    model_service: Option<&dyn StorageModelService>,
    model_id: &str,
) -> Result<Row, StorageError> {
    let _ = (model_service, model_id);
    todo!()
}

/// Flip the workspace flags for models whose weights are gone, and persist.
/// Returns the names of the models marked.
fn mark_models_removed(workspace: &mut Workspace, deleted_ids: &HashSet<String>) -> Vec<String> {
    let _ = (workspace, deleted_ids);
    todo!()
}

/// Build the cleanup result, including a fresh usage report.
fn report(
    model_service: Option<&dyn StorageModelService>,
    deleted: Vec<Row>,
    failed: Vec<Row>,
    models_marked: Vec<String>,
    dry_run: bool,
) -> Result<Row, StorageError> {
    let _ = (model_service, deleted, failed, models_marked, dry_run);
    todo!()
}
