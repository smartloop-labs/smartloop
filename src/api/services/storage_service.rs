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

use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use regex::Regex;
use serde_json::{Map, Value};

use crate::api::services::model_service::ModelService;
use crate::config::app_settings::AppSettings;
use crate::model::downloader::inflight;
use crate::utils::logging_setup::{truncate_log_file, DEFAULT_BACKUP_COUNT};
use crate::workspace::workspace::{is_mmproj_gguf, Project, SharedWorkspace, Workspace, WorkspaceModel};

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

/// Errors surfaced to the router: `ValueError` -> 400, `RuntimeError` -> 503,
/// anything else (`Other`) -> 500.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StorageError {
    /// Python `ValueError`: a path or request that fails validation.
    #[error("{0}")]
    ValueError(String),
    /// Python `RuntimeError`: the service is busy or the workspace unavailable.
    #[error("{0}")]
    RuntimeError(String),
    /// Any other failure (e.g. the workspace record could not be written).
    #[error("{0}")]
    Other(String),
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
/// Replacement for [`discard_partial_download`] (tests patch
/// `storage_service.discard_partial_download`); receives the model id.
pub type DiscardHook = Arc<dyn Fn(&str) -> Result<Row, StorageError> + Send + Sync>;

/// Module-level settings (Python `_settings`); tests override it.
static SETTINGS: Lazy<RwLock<Option<AppSettings>>> = Lazy::new(|| RwLock::new(None));
static REMOVE_HOOK: Lazy<RwLock<Option<RemoveHook>>> = Lazy::new(|| RwLock::new(None));
static IS_DOWNLOADING_HOOK: Lazy<RwLock<Option<IsDownloadingHook>>> = Lazy::new(|| RwLock::new(None));
static RMTREE_HOOK: Lazy<RwLock<Option<RmtreeHook>>> = Lazy::new(|| RwLock::new(None));
static DISCARD_HOOK: Lazy<RwLock<Option<DiscardHook>>> = Lazy::new(|| RwLock::new(None));

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

/// Override [`discard_partial_download`] as a whole; `None` restores the real
/// one.
pub fn set_discard_hook(hook: Option<DiscardHook>) {
    *DISCARD_HOOK.write() = hook;
}


/// The module settings: the override when set, else fresh `AppSettings`.
fn settings() -> AppSettings {
    if let Some(settings) = SETTINGS.read().as_ref() {
        return settings.clone();
    }
    AppSettings::new()
}

/// `inflight.is_downloading`, or the test override.
fn is_downloading(path: &str) -> bool {
    let hook = IS_DOWNLOADING_HOOK.read().clone();
    match hook {
        Some(hook) => hook(path),
        None => inflight::is_downloading(path),
    }
}

/// `_remove` as the sweep sees it: the test override when set.
fn remove_dispatch(path: &str, truncate: bool) -> (bool, Option<String>) {
    let hook = REMOVE_HOOK.read().clone();
    match hook {
        Some(hook) => hook(path, truncate),
        None => remove(path, truncate),
    }
}

/// `shutil.rmtree`, or the test override.
fn rmtree(path: &str) -> std::io::Result<()> {
    let hook = RMTREE_HOOK.read().clone();
    match hook {
        Some(hook) => hook(path),
        None => std::fs::remove_dir_all(path),
    }
}

// ── Path helpers (Python `os.path` semantics) ───────────────────────────

/// `os.path.realpath` (non-strict): symlinks are resolved wherever a prefix
/// exists, missing components are kept as written, and `..` is applied to the
/// resolved prefix.  Relative paths are anchored at the working directory.
fn realpath(path: &str) -> String {
    let p = Path::new(path);
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")).join(p)
    };
    let mut links = 0usize;
    resolve_into(PathBuf::new(), &abs, &mut links).to_string_lossy().into_owned()
}

/// Walk `rest` onto the already-resolved `base`.
fn resolve_into(mut base: PathBuf, rest: &Path, links: &mut usize) -> PathBuf {
    for comp in rest.components() {
        match comp {
            Component::Prefix(_) | Component::RootDir => base.push(comp.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                base.pop();
            }
            Component::Normal(name) => {
                let candidate = base.join(name);
                match std::fs::symlink_metadata(&candidate) {
                    Ok(meta) if meta.file_type().is_symlink() && *links < 40 => {
                        *links += 1;
                        match std::fs::read_link(&candidate) {
                            Ok(target) => {
                                let anchor = if target.is_absolute() { PathBuf::new() } else { base.clone() };
                                base = resolve_into(anchor, &target, links);
                            }
                            Err(_) => base = candidate,
                        }
                    }
                    _ => base = candidate,
                }
            }
        }
    }
    if base.as_os_str().is_empty() {
        base.push(std::path::MAIN_SEPARATOR_STR);
    }
    base
}

/// `os.path.join(a, b)` for a relative `b`.
fn join(a: &str, b: &str) -> String {
    Path::new(a).join(b).to_string_lossy().into_owned()
}

/// `os.path.dirname`.
fn dirname(path: &str) -> String {
    Path::new(path).parent().map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| {
        if path.starts_with(std::path::MAIN_SEPARATOR) {
            std::path::MAIN_SEPARATOR_STR.to_string()
        } else {
            String::new()
        }
    })
}

/// `os.path.basename`.
fn basename(path: &str) -> String {
    Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// `os.path.isdir` (follows symlinks).
fn isdir(path: &str) -> bool {
    !path.is_empty() && Path::new(path).is_dir()
}

/// `os.path.isfile` (follows symlinks).
fn isfile(path: &str) -> bool {
    Path::new(path).is_file()
}

/// `os.path.islink`.
fn islink(path: &str) -> bool {
    std::fs::symlink_metadata(path).map(|m| m.file_type().is_symlink()).unwrap_or(false)
}

/// `os.path.relpath(path, start)` for a `path` known to sit under `start`.
fn relpath_under(path: &str, start: &str) -> String {
    match Path::new(path).strip_prefix(start) {
        Ok(rel) if rel.as_os_str().is_empty() => ".".to_string(),
        Ok(rel) => rel.to_string_lossy().into_owned(),
        Err(_) => path.to_string(),
    }
}

/// `uuid.UUID(name)`: true when `name` parses as a UUID.
fn is_uuid(name: &str) -> bool {
    uuid::Uuid::parse_str(name).is_ok()
}

// ── Paths and sizes ─────────────────────────────────────────────────────

/// The SLP home directory, resolved.
pub fn home() -> String {
    realpath(&settings().home_dir())
}

/// Return `(bytes, files)` for a file or directory tree.  Symlinks are neither
/// followed nor counted — a link into the user's own files must not show up as
/// space we can reclaim.  Missing paths are 0.
pub fn walk_size(path: &str) -> (u64, u64) {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(_) => return (0, 0),
    };
    let ft = meta.file_type();
    if ft.is_symlink() {
        return (0, 0);
    }
    if ft.is_file() {
        return (meta.len(), 1);
    }
    if !ft.is_dir() {
        return (0, 0);
    }

    let mut total = 0u64;
    let mut files = 0u64;
    let mut stack = vec![PathBuf::from(path)];
    while let Some(current) = stack.pop() {
        let entries = match std::fs::read_dir(&current) {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let ft = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            if ft.is_symlink() {
                continue;
            }
            if ft.is_dir() {
                stack.push(entry.path());
            } else if ft.is_file() {
                if let Ok(meta) = std::fs::symlink_metadata(entry.path()) {
                    total += meta.len();
                    files += 1;
                }
            }
        }
    }
    (total, files)
}

/// Immediate subdirectory names of `path`, skipping symlinks.  Sorted.
fn dirs_in(path: &str, skip_hidden: bool) -> Vec<String> {
    let entries = match std::fs::read_dir(path) {
        Ok(rd) => rd,
        Err(_) => return Vec::new(),
    };
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_dir() && !ft.is_symlink()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| !(skip_hidden && name.starts_with('.')))
        .collect();
    names.sort();
    names
}

/// Entries of `path` sorted by name (`sorted(os.scandir(path), key=name)`).
fn sorted_entries(path: &str) -> std::io::Result<Vec<std::fs::DirEntry>> {
    let mut entries: Vec<std::fs::DirEntry> = std::fs::read_dir(path)?.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    Ok(entries)
}

/// `DirEntry.is_symlink()`.
fn entry_is_symlink(entry: &std::fs::DirEntry) -> bool {
    entry.file_type().map(|ft| ft.is_symlink()).unwrap_or(false)
}

/// Build a candidate row with its size filled in; `extra` is merged on top.
fn row(path: &str, removable: bool, keep_reason: Option<&str>, extra: Row) -> Row {
    let (size, files) = walk_size(path);
    let mut row = Row::new();
    row.insert("path".into(), Value::String(path.to_string()));
    row.insert("bytes".into(), Value::from(size));
    row.insert("files".into(), Value::from(files));
    row.insert("removable".into(), Value::Bool(removable && keep_reason.is_none()));
    row.insert(
        "keep_reason".into(),
        keep_reason.map(|r| Value::String(r.to_string())).unwrap_or(Value::Null),
    );
    for (k, v) in extra {
        row.insert(k, v);
    }
    row
}

/// `row["bytes"]` as an integer.
fn row_bytes(row: &Row) -> u64 {
    row.get("bytes").and_then(Value::as_u64).unwrap_or(0)
}

/// `row["files"]` as an integer.
fn row_files(row: &Row) -> u64 {
    row.get("files").and_then(Value::as_u64).unwrap_or(0)
}

/// `row["removable"]`.
fn row_removable(row: &Row) -> bool {
    row.get("removable").and_then(Value::as_bool).unwrap_or(false)
}

/// A string field of a row (`None` for null / missing).
fn row_str<'a>(row: &'a Row, key: &str) -> Option<&'a str> {
    row.get(key).and_then(Value::as_str)
}

// ── Workspace and model state ───────────────────────────────────────────

/// Return the live workspace, or `RuntimeError`.
///
/// Always the singleton the rest of the process is using — never a fresh
/// `Workspace::load()` copy, since mutating and saving that would clobber
/// whatever the running server has in memory.
///
/// Per-project state is re-read before returning.  Under supervision a
/// project's own file is written by the child pinned to it, while this process
/// holds the copy it took when the project first appeared and nothing tells it
/// when that copy went stale — the case `Workspace::refresh_project` exists
/// for.  Every keep reason below resolves `enabled`/`is_active` per project
/// (see [`keep_model_ids`]), so without this the sweep decides what to delete
/// from a snapshot: weights a child has just turned off look in use for as
/// long as the copy lasts, and — the direction that actually costs something —
/// weights a child has just turned *on* look collectible.  Mtime-gated, so the
/// usual cost is one `stat` per project.
fn load_workspace(model_service: Option<&dyn StorageModelService>) -> Result<SharedWorkspace, StorageError> {
    let workspace = match model_service {
        Some(service) => service
            .ensure_workspace()
            .map_err(|e| StorageError::RuntimeError(format!("Workspace unavailable: {e}")))?,
        None => Workspace::load().map(|ws| Arc::new(RwLock::new(ws))),
    };

    let workspace = workspace.ok_or_else(|| StorageError::RuntimeError("Workspace not initialized".to_string()))?;

    if let Some(service) = model_service {
        // Ids first: a refresh re-reads a project record, and iterating the
        // list while it is being rewritten is not worth the risk.
        let ids: Vec<String> = workspace.read().projects.iter().map(|p| p.id.clone()).collect();
        for project_id in ids {
            if service.refresh_project_state(&project_id).is_err() {
                // A project whose file cannot be read is not a reason to
                // refuse the whole report; it just keeps the copy we already had.
                tracing::debug!(target: LOG_TARGET, "[storage] Could not refresh project {}", project_id);
            }
        }
    }

    Ok(workspace)
}

/// Model names currently holding a loaded context in `ModelService`.
///
/// Covers the named slots plus the per-capability caches.  A model resident in
/// one of these is being used right now regardless of what the workspace flags
/// say, so its weights must stay on disk.
fn resident_model_names(model_service: Option<&dyn StorageModelService>) -> HashSet<String> {
    match model_service {
        Some(service) => service.resident_model_names(),
        None => HashSet::new(),
    }
}

/// Map `model_id` -> why it must be kept, for every protected model.
fn keep_model_ids(workspace: &Workspace, model_service: Option<&dyn StorageModelService>) -> HashMap<String, String> {
    fn mark(keep: &mut HashMap<String, String>, model_id: &str, reason: &str) {
        if !model_id.is_empty() && !keep.contains_key(model_id) {
            keep.insert(model_id.to_string(), reason.to_string());
        }
    }
    let mut keep: HashMap<String, String> = HashMap::new();

    let mut resident: Vec<String> = resident_model_names(model_service).into_iter().collect();
    resident.sort();
    for name in resident {
        if let Some(entry) = workspace.get_model(Some(&name)) {
            mark(&mut keep, &entry.model_id, "loaded");
        }
    }

    if let Some(default) = workspace.get_model(None) {
        mark(&mut keep, &default.model_id, "default");
    }

    // Resolved per scope, not read off the workspace entry: ``enabled`` and
    // ``is_active`` can be overridden per project, so a model turned off
    // app-wide may still be pinned on in a project that is using it.
    //
    // Real projects only — the workspace entry is deliberately *not* a scope
    // of its own.  It is a template: the defaults a **new** project is seeded
    // from, not a place anything runs.  A model it enables is therefore not
    // thereby in use, and treating it as a scope made the app-default pin
    // weights that every existing project had turned off — reclaimable disk
    // the UI could never offer, because the model picker only writes project
    // overrides.  Nothing is lost by dropping it: a project with no override
    // still reads through to that default (``effective_model_enabled``), so the
    // template keeps its meaning through the projects that inherit it.  With
    // no projects at all there is nothing to read through, so consult the
    // default directly rather than declaring every downloaded model collectible.
    let scopes: Vec<Option<&Project>> = if workspace.projects.is_empty() {
        vec![None]
    } else {
        workspace.projects.iter().map(Some).collect()
    };
    for scope in scopes {
        for m in &workspace.models {
            if workspace.effective_model_active(&m.model_name, scope) {
                mark(&mut keep, &m.model_id, "active");
            }
        }
        for m in &workspace.models {
            // ``enabled`` is what the model picker writes, so this is the
            // user's own selection — the thing the old vendor-directory sweep
            // destroyed.
            if workspace.effective_model_enabled(&m.model_name, scope) {
                mark(&mut keep, &m.model_id, "enabled");
            }
        }
    }

    keep
}

/// Which usage group a model's bytes belong to.
///
/// No keep reason means nothing holds the tree, which is exactly what makes it
/// removable — so `unused` and `reclaimable` describe the same bytes from two
/// directions and must stay in step.  An unrecognised reason counts as in use:
/// a new way to hold a model should show up as the user's footprint, not
/// quietly become something the UI offers to delete.
pub fn model_group(keep_reason: Option<&str>) -> &'static str {
    match keep_reason {
        None => "unused",
        Some(reason) => MODEL_GROUPS
            .iter()
            .find(|(r, _)| *r == reason)
            .map(|(_, group)| *group)
            .unwrap_or("in_use"),
    }
}

// ── Scanners ────────────────────────────────────────────────────────────

/// Return `(has_weights, has_part)` for a model directory.
///
/// `has_weights` means a complete, servable GGUF is present.  An `mmproj`
/// companion alone does not count — a vision projector is not weights, and
/// `resolve_gguf_path` already skips those when looking for something to load.
/// `has_part` means the downloader left an unfinished sidecar: bytes go to
/// `<name>.part` and are renamed onto the final path only once the transfer
/// completes and the MD5 matches, so the suffix is the downloader saying this
/// file never arrived.
///
/// These are the two signals `_gguf_exists` in the init router reads, kept
/// apart here because this caller needs to distinguish "nothing usable
/// arrived" (the tree is dead weight) from "the weights landed but a companion
/// did not" (keep it — the downloader checksums the finished file, skips it,
/// and refetches only the rest).
pub fn weights_state(model_dir: &str) -> (bool, bool) {
    let bin = join(model_dir, "bin");
    let mut listings: Vec<String> = Vec::new();
    for d in [model_dir, bin.as_str()] {
        if !isdir(d) {
            continue;
        }
        if let Ok(rd) = std::fs::read_dir(d) {
            listings.extend(rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()));
        }
    }
    let has_part = listings.iter().any(|f| f.ends_with(".part"));
    let has_weights = listings.iter().any(|f| f.ends_with(".gguf") && !is_mmproj_gguf(f));
    (has_weights, has_part)
}

/// One row per `models/<org>/<repo>` directory present on disk.
pub fn scan_models(workspace: &Workspace, model_service: Option<&dyn StorageModelService>) -> Vec<Row> {
    let models_dir = workspace.models_dir.as_str();
    if models_dir.is_empty() || !isdir(models_dir) {
        return Vec::new();
    }

    let keep = keep_model_ids(workspace, model_service);
    let resident = resident_model_names(model_service);
    let mut by_id: HashMap<&str, &WorkspaceModel> = HashMap::new();
    for m in &workspace.models {
        // ``{m.model_id: m for m in ...}`` keeps the last duplicate.
        by_id.insert(m.model_id.as_str(), m);
    }

    let mut rows = Vec::new();
    // Hidden directories are bookkeeping, not models — the HF cache keeps a
    // ".locks" tree under embeddings/.  Their bytes still show up in the
    // category total, they just aren't offered as something to remove.
    for org in dirs_in(models_dir, true) {
        for repo in dirs_in(&join(models_dir, &org), true) {
            let model_id = format!("{org}/{repo}");
            let entry = by_id.get(model_id.as_str()).copied();
            let path = join(&join(models_dir, &org), &repo);
            let loaded = entry.map(|e| resident.contains(&e.model_name)).unwrap_or(false);

            // An abandoned download: a ``.part`` and nothing servable beside
            // it.  This overrides the keep reasons below, because those all
            // answer "is this model in use?" and none of them can be true of
            // bytes that never finished arriving — ``enabled`` is the picker's
            // flag, set when the user asked for the model, not when it landed.
            // A half-downloaded tree is exactly what the next download click
            // unlinks on its way to starting over, so keeping it reserves disk
            // for something nothing will ever read — and on a full disk that
            // deadlocks the user, since the only offer the UI can make is the
            // download this space is blocking.
            let (has_weights, has_part) = weights_state(&path);
            let partial = has_part && !has_weights;

            let keep_reason: Option<String> = if is_downloading(&path) {
                // Must come first: a download in progress is also a partial
                // one, and on disk the two are the same thing.  The registry
                // carries the difference — a transfer that is still writing —
                // which is the one thing the reasoning above cannot see.
                Some("downloading".to_string())
            } else if BUNDLED_ORGS.contains(&org.as_str()) {
                // Above the partial rule, not below it: the framework cannot
                // run without the orchestrator or the embedder, so their retry
                // is not a thing the user might choose — it is the next thing
                // that happens.  That is what makes the deadlock argument above
                // not apply here.  The retry unlinks the stale ``.part`` itself,
                // so the space comes back either way and the sweep does not
                // need to be the one to take it; deleting the tree only risks
                // racing a writer nothing registered.
                Some("bundled".to_string())
            } else if partial && !loaded {
                // ``loaded`` still wins: a resident context means this really
                // is open right now, whatever the directory looks like, and
                // that is the one state where unlinking is a use-after-free.
                None
            } else {
                keep.get(&model_id).cloned()
            };

            let mut extra = Row::new();
            extra.insert("group".into(), Value::String(model_group(keep_reason.as_deref()).to_string()));
            extra.insert("model_id".into(), Value::String(model_id.clone()));
            extra.insert("partial".into(), Value::Bool(partial));
            extra.insert(
                "model_name".into(),
                entry.map(|e| Value::String(e.model_name.clone())).unwrap_or(Value::Null),
            );
            extra.insert("downloaded".into(), Value::Bool(entry.map(|e| e.downloaded).unwrap_or(false)));
            extra.insert("enabled".into(), Value::Bool(entry.map(|e| e.enabled).unwrap_or(false)));
            extra.insert("is_active".into(), Value::Bool(entry.map(|e| e.is_active).unwrap_or(false)));
            extra.insert("loaded".into(), Value::Bool(loaded));
            // No workspace entry means an aborted or orphaned download.
            extra.insert("registered".into(), Value::Bool(entry.is_some()));

            rows.push(row(&path, true, keep_reason.as_deref(), extra));
        }
    }
    rows
}

/// Version directory names that belong to the running install.
pub fn running_versions() -> HashSet<String> {
    let home = home();
    let mut running = HashSet::new();

    for marker in [".installed-version", "version"] {
        if let Ok(raw) = std::fs::read_to_string(join(&home, marker)) {
            let value = raw.trim();
            if !value.is_empty() {
                running.insert(value.to_string());
            }
        }
    }

    // The binary we are executing is the authority, and the markers can be
    // stale or missing entirely.
    let executable = std::env::current_exe()
        .map(|p| realpath(&p.to_string_lossy()))
        .unwrap_or_else(|_| realpath(""));
    let prefix = format!("{}{}", home, std::path::MAIN_SEPARATOR);
    if executable.starts_with(&prefix) {
        let rel = relpath_under(&executable, &home);
        if let Some(first) = rel.split(std::path::MAIN_SEPARATOR).next() {
            if VERSION_RE.is_match(first) {
                running.insert(first.to_string());
            }
        }
    }

    running
}

/// Installed framework payloads other than the running one.
pub fn scan_versions() -> Vec<Row> {
    let home = home();
    let running = running_versions();
    let mut rows = Vec::new();
    for name in dirs_in(&home, false) {
        if !VERSION_RE.is_match(&name) {
            continue;
        }
        let keep_reason = if running.contains(&name) { Some("running") } else { None };
        let mut extra = Row::new();
        extra.insert("version".into(), Value::String(name.clone()));
        rows.push(row(&join(&home, &name), true, keep_reason, extra));
    }
    rows
}

/// The server log and its rotated backups.
pub fn scan_logs() -> Vec<Row> {
    let home = home();
    let mut rows = Vec::new();

    let log_path = join(&home, "server.log");
    if isfile(&log_path) {
        // Emptied in place, never unlinked — the handler holds it open.
        let mut extra = Row::new();
        extra.insert("action".into(), Value::String("truncate".into()));
        rows.push(row(&log_path, true, None, extra));
    }

    for n in 1..=DEFAULT_BACKUP_COUNT {
        let backup = join(&home, &format!("server.log.{n}"));
        if isfile(&backup) {
            let mut extra = Row::new();
            extra.insert("action".into(), Value::String("delete".into()));
            rows.push(row(&backup, true, None, extra));
        }
    }

    rows
}

/// Contents of the regenerable cache directories (the dirs themselves stay).
pub fn scan_cache() -> Vec<Row> {
    let home = home();
    let mut rows = Vec::new();
    for name in CACHE_DIRS {
        let parent = join(&home, name);
        let entries = match sorted_entries(&parent) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries {
            if entry_is_symlink(&entry) {
                continue;
            }
            let mut extra = Row::new();
            extra.insert("cache".into(), Value::String((*name).to_string()));
            rows.push(row(&entry.path().to_string_lossy(), true, None, extra));
        }
    }
    rows
}

/// Project directories under the workspace that `config.json` doesn't know about.
pub fn scan_orphan_projects(workspace: &Workspace) -> Vec<Row> {
    let workspace_dir = workspace.workspace_dir.as_str();
    if workspace_dir.is_empty() || !isdir(workspace_dir) {
        return Vec::new();
    }

    // The workspace directory is ``<home>/<workspace-id>``, so it can never be
    // the home directory itself — but an empty id makes the join collapse it
    // to exactly that, and realpath then strips the trailing separator so
    // every later check agrees.  What sits directly under home is one
    // directory per *workspace*, all uuid-shaped and none of them this
    // workspace's projects, so the scan below would offer every one of them —
    // on a shared machine, another account's entire workspace.  Refuse rather
    // than guess.
    if realpath(workspace_dir) == home() {
        tracing::warn!(
            target: LOG_TARGET,
            "[storage] Workspace directory resolved to the home directory (workspace id '{}'); skipping the orphan-project scan",
            workspace.id
        );
        return Vec::new();
    }

    let known: HashSet<String> = workspace
        .projects
        .iter()
        .filter(|p| !p.project_dir.is_empty())
        .map(|p| realpath(&p.project_dir))
        .collect();

    let mut rows = Vec::new();
    for name in dirs_in(workspace_dir, false) {
        if name == "models" {
            continue;
        }
        if !is_uuid(&name) {
            // Only ever consider directories shaped like a project id.
            continue;
        }
        let path = join(workspace_dir, &name);
        if known.contains(&realpath(&path)) {
            continue;
        }
        let mtime = match std::fs::metadata(&path).and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let age = match SystemTime::now().duration_since(mtime) {
            Ok(d) => d.as_secs_f64(),
            Err(e) => -e.duration().as_secs_f64(),
        };
        if age < ORPHAN_MIN_AGE_S as f64 {
            // Touched recently, so something is still writing here even though
            // the workspace record does not admit it.  Leave it be.
            continue;
        }
        let mut extra = Row::new();
        extra.insert("project_id".into(), Value::String(name.clone()));
        extra.insert("age_s".into(), Value::from(age.trunc() as i64));
        rows.push(row(&path, true, None, extra));
    }
    rows
}

/// Session transcripts, reported but never deleted.
///
/// A transcript not referenced by `project.sessions` is *not* evidence of an
/// orphan: a session id only lands there when an assistant turn arrives while
/// that project is current, and the desktop app keeps its own conversation
/// list in `studio.db`.  So an untracked JSONL is usually a chat the UI still
/// shows, and deleting it would silently drop that chat's memory.  Deleting
/// these needs an explicit keep-list from the client; until then this category
/// is informational.
pub fn scan_conversations(workspace: &Workspace) -> Vec<Row> {
    let conversations = join(&home(), "conversations");
    if !isdir(&conversations) {
        return Vec::new();
    }

    let tracked: HashSet<&str> =
        workspace.projects.iter().flat_map(|p| p.sessions.iter().map(String::as_str)).collect();

    let entries = match sorted_entries(&conversations) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let mut rows = Vec::new();
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry_is_symlink(&entry) || !name.ends_with(".jsonl") {
            continue;
        }
        let session_id = &name[..name.len() - ".jsonl".len()];
        let mut extra = Row::new();
        extra.insert("session_id".into(), Value::String(session_id.to_string()));
        extra.insert("tracked".into(), Value::Bool(tracked.contains(session_id)));
        rows.push(row(
            &entry.path().to_string_lossy(),
            false,
            Some("not deletable without a client keep-list"),
            extra,
        ));
    }
    rows
}

// ── Validation ──────────────────────────────────────────────────────────

/// Resolve `path` and require it to sit strictly inside the home directory.
fn assert_inside_home(path: &str) -> Result<String, StorageError> {
    let home = home();
    let resolved = realpath(path);
    if resolved == home {
        return Err(StorageError::ValueError(format!(
            "Refusing to act on the home directory itself: {path}"
        )));
    }
    if !Path::new(&resolved).starts_with(&home) {
        return Err(StorageError::ValueError(format!("Path escapes {home}: {path}")));
    }
    if NEVER_DELETE.contains(&basename(&resolved).as_str()) {
        return Err(StorageError::ValueError(format!("Protected path: {path}")));
    }
    if islink(path) {
        return Err(StorageError::ValueError(format!("Refusing to delete through a symlink: {path}")));
    }
    Ok(resolved)
}

/// `server.log.<n>` with `<n>` all digits.
static LOG_BACKUP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^server\.log\.(\d+)$").unwrap());

/// Require `path` to match the expected shape for its category
/// (`models`, `versions`, `logs`, `cache`, `orphan_projects`).  Returns the
/// resolved path.
///
/// Positive validation: each category knows precisely what its paths look
/// like, so anything else — however harmless it seems — is rejected.
pub fn validate(path: &str, shape: &str, workspace: &Workspace) -> Result<String, StorageError> {
    let resolved = assert_inside_home(path)?;
    let home = home();

    match shape {
        "models" => {
            let models_dir = realpath(&workspace.models_dir);
            if dirname(&dirname(&resolved)) != models_dir {
                return Err(StorageError::ValueError(format!("Not a models/<org>/<repo> directory: {path}")));
            }
        }
        "versions" => {
            let name = relpath_under(&resolved, &home);
            if name.contains(std::path::MAIN_SEPARATOR) || !VERSION_RE.is_match(&name) {
                return Err(StorageError::ValueError(format!("Not a version directory: {path}")));
            }
            if running_versions().contains(&name) {
                return Err(StorageError::ValueError(format!("Refusing to delete the running install: {path}")));
            }
        }
        "logs" => {
            let name = relpath_under(&resolved, &home);
            // Compare the index numerically: a character class built from
            // DEFAULT_BACKUP_COUNT would silently mean something else at 10+.
            let kept_backup = LOG_BACKUP_RE
                .captures(&name)
                .and_then(|c| c[1].parse::<u64>().ok())
                .map(|n| (1..=u64::from(DEFAULT_BACKUP_COUNT)).contains(&n))
                .unwrap_or(false);
            if name != "server.log" && !kept_backup {
                return Err(StorageError::ValueError(format!("Not a server log file: {path}")));
            }
        }
        "cache" => {
            let parent_dir = dirname(&resolved);
            let parent = basename(&parent_dir);
            if !CACHE_DIRS.contains(&parent.as_str()) || dirname(&parent_dir) != home {
                return Err(StorageError::ValueError(format!("Not a cache entry: {path}")));
            }
        }
        "orphan_projects" => {
            let workspace_dir = realpath(&workspace.workspace_dir);
            // Belt to the scanner's braces: a workspace dir that collapsed onto
            // home would make every sibling workspace look like a project of
            // this one.
            if workspace_dir == home {
                return Err(StorageError::ValueError(format!(
                    "Workspace directory resolved to the home directory: {path}"
                )));
            }
            if dirname(&resolved) != workspace_dir {
                return Err(StorageError::ValueError(format!("Not a project directory: {path}")));
            }
            if !is_uuid(&basename(&resolved)) {
                return Err(StorageError::ValueError("badly formed hexadecimal UUID string".to_string()));
            }
            for project in &workspace.projects {
                if !project.project_dir.is_empty() && realpath(&project.project_dir) == resolved {
                    return Err(StorageError::ValueError(format!("Refusing to delete a live project: {path}")));
                }
            }
        }
        _ => return Err(StorageError::ValueError(format!("Unknown category: {shape}"))),
    }

    Ok(resolved)
}

// ── Removal ─────────────────────────────────────────────────────────────

/// What `truncate_log_file` reports, as a `Result` whichever shape it has
/// (a bare byte count when failures are swallowed, or an `io::Result`).
trait TruncateOutcome {
    fn into_io_result(self) -> std::io::Result<u64>;
}

impl TruncateOutcome for u64 {
    fn into_io_result(self) -> std::io::Result<u64> {
        Ok(self)
    }
}

impl TruncateOutcome for std::io::Result<u64> {
    fn into_io_result(self) -> std::io::Result<u64> {
        self
    }
}

/// Delete (or empty, when `truncate`) `path`, retrying briefly.
/// Returns `(ok, error)`.
pub fn remove(path: &str, truncate: bool) -> (bool, Option<String>) {
    let mut last_error: Option<String> = None;
    for attempt in 0..DELETE_ATTEMPTS {
        let result: std::io::Result<()> = if truncate {
            truncate_log_file(&dirname(path)).into_io_result().map(|_| ())
        } else if isdir(path) {
            rmtree(path)
        } else {
            std::fs::remove_file(path)
        };
        match result {
            Ok(()) => return (true, None),
            Err(e) => {
                // A GGUF a spawn worker still has mapped can refuse to go on
                // Windows; the mapping usually drops within a beat.
                last_error = Some(e.to_string());
                if attempt < DELETE_ATTEMPTS - 1 {
                    std::thread::sleep(Duration::from_secs_f64(DELETE_BACKOFF_S));
                }
            }
        }
    }
    (false, last_error)
}

/// Remove `models/<org>` directories left empty by a model deletion.
fn prune_empty_orgs(workspace: &Workspace) {
    let models_dir = workspace.models_dir.as_str();
    if models_dir.is_empty() || !isdir(models_dir) {
        return;
    }
    for org in dirs_in(models_dir, false) {
        if BUNDLED_ORGS.contains(&org.as_str()) {
            continue;
        }
        let path = join(models_dir, &org);
        let empty = match std::fs::read_dir(&path) {
            Ok(mut rd) => rd.next().is_none(),
            Err(_) => continue,
        };
        if empty {
            let _ = std::fs::remove_dir(&path);
        }
    }
}

/// `RuntimeError` while the indexer is writing to these trees.
///
/// Downloads are deliberately *not* guarded here.  This used to also check a
/// `"downloader"` task, which nothing has ever registered — every download
/// runs through `/v1/init`, outside the task layer — so the check read as
/// protection while providing none, and the sweep deleted a running download.
/// Replacing it with a registry lookup would be worse than useless in a
/// different way: a global refusal, blocking a logs-only sweep because some
/// unrelated model happens to be downloading.
///
/// Downloads are protected per path instead, where the decision belongs — see
/// `inflight::is_downloading`, consulted by [`scan_models`] and again before
/// each removal in [`run_cleanup`].
fn refuse_if_busy(model_service: Option<&dyn StorageModelService>, needs_indexer: bool) -> Result<(), StorageError> {
    if !needs_indexer {
        return Ok(());
    }
    let Some(service) = model_service else {
        return Ok(());
    };
    let Some(status) = service.task_status("indexer") else {
        return Ok(());
    };
    if status.get("status").and_then(Value::as_str) == Some("running") {
        return Err(StorageError::RuntimeError(
            "Cannot clean up while the indexer task is running".to_string(),
        ));
    }
    Ok(())
}

/// Acquire every inference permit so no generation is in flight.
///
/// Returns the number of permits held, for the caller to release.  Fails with
/// `RuntimeError` on timeout rather than deleting weights a live decode could
/// still be reading.
///
/// Workspace-wide, not merely this process: the model host runs inside the
/// supervisor wrapping this same `ModelService`, so a project agent's
/// generation arrives through the in-process model client and takes this very
/// semaphore.  Draining it here stops every child too.
///
/// The exception is a workspace running without that host
/// (`SLP_MODEL_HOST_DISABLE=1`, or the host having lost its socket), where each
/// child loads its own weights and this drain cannot reach them.  What still
/// protects those is [`keep_model_ids`], which reads `enabled`/`is_active` from
/// the workspace record rather than from memory, so a model any project is
/// using is never a candidate.  Only a model resident in a child yet switched
/// off in every scope falls through.
fn drain_inference(model_service: Option<&dyn StorageModelService>) -> Result<usize, StorageError> {
    let Some(semaphore) = model_service.and_then(|s| s.inference_lock()) else {
        return Ok(0);
    };

    let permits = settings().n_parallel.max(1) as usize;
    let deadline = Instant::now() + Duration::from_secs_f64(DRAIN_TIMEOUT_S);
    let mut held = 0usize;
    for _ in 0..permits {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if !semaphore.acquire(remaining) {
            for _ in 0..held {
                semaphore.release();
            }
            return Err(StorageError::RuntimeError(
                "Model is busy generating; try again in a moment".to_string(),
            ));
        }
        held += 1;
    }
    Ok(held)
}

/// Release `held` inference permits.
fn release_inference(model_service: Option<&dyn StorageModelService>, held: usize) {
    let Some(semaphore) = model_service.and_then(|s| s.inference_lock()) else {
        return;
    };
    for _ in 0..held {
        let released = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| semaphore.release()));
        if released.is_err() {
            break;
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────────

/// Roll candidate rows up into a category report.
fn category(rows: Vec<Row>, total_path: Option<&str>, extra: Row) -> Row {
    let (total, files) = match total_path {
        Some(path) => walk_size(path),
        None => (rows.iter().map(row_bytes).sum(), rows.iter().map(row_files).sum()),
    };
    let reclaimable: u64 = rows.iter().filter(|r| row_removable(r)).map(row_bytes).sum();
    let mut report = Row::new();
    report.insert("bytes".into(), Value::from(total));
    report.insert("files".into(), Value::from(files));
    report.insert("reclaimable".into(), Value::from(reclaimable));
    report.insert("kept".into(), Value::from(total.saturating_sub(reclaimable)));
    report.insert("items".into(), Value::Array(rows.into_iter().map(Value::Object).collect()));
    for (k, v) in extra {
        report.insert(k, v);
    }
    report
}

/// The models report, split into the groups the UI draws as its own bands.
///
/// Reported here rather than derived by the client: which models count as in
/// use is the same judgement [`run_cleanup`] acts on, and a second copy of it
/// in the desktop app is a second copy that can disagree.
///
/// `reclaimable` already carries the `unused` total — a row is removable
/// exactly when nothing keeps it — so it is not repeated under another name.
/// The four numbers partition the category:
///
/// ```text
/// in_use + system + reclaimable + overhead == bytes
/// ```
fn models_category(workspace: &Workspace, model_service: Option<&dyn StorageModelService>) -> Row {
    let rows = scan_models(workspace, model_service);

    let mut by_group: HashMap<String, u64> = HashMap::new();
    for r in &rows {
        let group = row_str(r, "group").unwrap_or_default().to_string();
        *by_group.entry(group).or_insert(0) += row_bytes(r);
    }

    let mut report = category(rows, Some(&workspace.models_dir), Row::new());
    let bytes = report.get("bytes").and_then(Value::as_u64).unwrap_or(0);
    let reclaimable = report.get("reclaimable").and_then(Value::as_u64).unwrap_or(0);
    let in_use = by_group.get("in_use").copied().unwrap_or(0);
    let system = by_group.get("system").copied().unwrap_or(0);

    report.insert("in_use".into(), Value::from(in_use));
    report.insert("system".into(), Value::from(system));
    // Inside the walked tree but holding no row of its own: the hidden
    // ``embeddings/.locks`` bookkeeping and loose files sitting directly under
    // an org directory, which is how the embedding GGUF actually lands.  Named
    // so the client can account for every byte it draws rather than reaching
    // the same figure by subtraction and calling the result "system".
    let overhead = i128::from(bytes) - i128::from(in_use) - i128::from(system) - i128::from(reclaimable);
    report.insert("overhead".into(), Value::from(overhead.max(0) as u64));
    report
}

/// Report what the home directory is using and what can be reclaimed.
pub fn compute_usage(model_service: Option<&dyn StorageModelService>) -> Result<Row, StorageError> {
    let shared = load_workspace(model_service)?;
    let workspace = shared.read();
    let home = home();
    let (total, files) = walk_size(&home);

    let mut categories = Row::new();
    categories.insert("models".into(), Value::Object(models_category(&workspace, model_service)));
    categories.insert("versions".into(), Value::Object(category(scan_versions(), None, Row::new())));
    categories.insert("logs".into(), Value::Object(category(scan_logs(), None, Row::new())));
    categories.insert("cache".into(), Value::Object(category(scan_cache(), None, Row::new())));
    categories.insert(
        "orphan_projects".into(),
        Value::Object(category(scan_orphan_projects(&workspace), None, Row::new())),
    );
    categories.insert(
        "conversations".into(),
        Value::Object(category(scan_conversations(&workspace), None, Row::new())),
    );
    drop(workspace);

    let reclaimable: u64 =
        categories.values().map(|c| c.get("reclaimable").and_then(Value::as_u64).unwrap_or(0)).sum();

    let mut report = Row::new();
    report.insert("home_dir".into(), Value::String(home));
    report.insert("total".into(), Value::from(total));
    report.insert("files".into(), Value::from(files));
    report.insert("reclaimable".into(), Value::from(reclaimable));
    report.insert("kept".into(), Value::from(total.saturating_sub(reclaimable)));
    report.insert("categories".into(), Value::Object(categories));
    Ok(report)
}

/// Resolve the requested categories into validated `(category, row)` work.
fn plan_cleanup(
    workspace: &Workspace,
    model_service: Option<&dyn StorageModelService>,
    opts: &CleanupOptions,
) -> Result<Vec<(String, Row)>, StorageError> {
    let mut planned: Vec<(String, Row)> = Vec::new();

    if opts.models {
        let mut rows: Vec<Row> = scan_models(workspace, model_service).into_iter().filter(row_removable).collect();
        if let Some(model_ids) = &opts.model_ids {
            let wanted: HashSet<&str> = model_ids.iter().map(String::as_str).collect();
            let available: HashSet<&str> = rows.iter().filter_map(|r| row_str(r, "model_id")).collect();
            let mut unknown: Vec<&str> = wanted.difference(&available).copied().collect();
            unknown.sort();
            if !unknown.is_empty() {
                return Err(StorageError::ValueError(format!(
                    "Not removable (unknown, in use, or bundled): {}",
                    unknown.join(", ")
                )));
            }
            rows.retain(|r| row_str(r, "model_id").map(|id| wanted.contains(id)).unwrap_or(false));
        }
        planned.extend(rows.into_iter().map(|r| ("models".to_string(), r)));
    }

    if opts.versions {
        planned.extend(scan_versions().into_iter().filter(row_removable).map(|r| ("versions".to_string(), r)));
    }
    if opts.logs {
        planned.extend(scan_logs().into_iter().filter(row_removable).map(|r| ("logs".to_string(), r)));
    }
    if opts.cache {
        planned.extend(scan_cache().into_iter().filter(row_removable).map(|r| ("cache".to_string(), r)));
    }
    if opts.orphan_projects {
        planned.extend(
            scan_orphan_projects(workspace)
                .into_iter()
                .filter(row_removable)
                .map(|r| ("orphan_projects".to_string(), r)),
        );
    }

    for (category, row) in &planned {
        validate(row_str(row, "path").unwrap_or_default(), category, workspace)?;
    }

    Ok(planned)
}

/// `{"category": ..., "path": ..., "bytes": ...}` for a removed (or planned) row.
fn deleted_item(category: &str, row: &Row) -> Row {
    let mut item = Row::new();
    item.insert("category".into(), Value::String(category.to_string()));
    item.insert("path".into(), row.get("path").cloned().unwrap_or(Value::Null));
    item.insert("bytes".into(), Value::from(row_bytes(row)));
    item
}

/// Reclaim disk space in the selected categories.
///
/// Nothing is deleted unless its category is explicitly requested — there is
/// deliberately no "everything" switch.  Model deletion additionally unloads
/// the model and rewrites its workspace flags, so the picker stops offering a
/// model whose weights are gone.
pub fn run_cleanup(model_service: Option<&dyn StorageModelService>, opts: CleanupOptions) -> Result<Row, StorageError> {
    let shared = load_workspace(model_service)?;
    refuse_if_busy(model_service, opts.orphan_projects)?;

    let planned = {
        let workspace = shared.read();
        plan_cleanup(&workspace, model_service, &opts)?
    };

    if opts.dry_run {
        let deleted = planned.iter().map(|(c, r)| deleted_item(c, r)).collect();
        return report(model_service, deleted, Vec::new(), Vec::new(), true);
    }

    let model_rows: Vec<&Row> = planned.iter().filter(|(c, _)| c == "models").map(|(_, r)| r).collect();
    // Only a model sweep touches files the inference path has open.
    let held = if model_rows.is_empty() { 0 } else { drain_inference(model_service)? };
    let mut deleted: Vec<Row> = Vec::new();
    let mut failed: Vec<Row> = Vec::new();
    let mut models_marked: Vec<String> = Vec::new();

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), StorageError> {
        for row in &model_rows {
            if let Some(name) = row_str(row, "model_name").filter(|n| !n.is_empty()) {
                // The unlocked variant: we hold every inference permit, and
                // the public unload_model_by_name takes one itself, so it
                // would block here forever against our own drain.
                if let Some(service) = model_service {
                    if let Err(e) = service.unload_model_by_name_unlocked(name) {
                        tracing::error!(
                            target: LOG_TARGET,
                            "[storage] Failed to unload {} before deletion: {:?}",
                            name,
                            e
                        );
                    }
                }
            }
        }
        if !model_rows.is_empty() {
            // The shared executor's workers each loaded their own copy of the
            // weights, so it has to go before the files can.
            if let Some(service) = model_service {
                service.reset_llm_executor();
            }
        }

        // Kept as text so the record can be re-emitted below.  A sweep that
        // empties server.log destroys the lines it has already written, so
        // without this the one log that says what was just deleted is the one
        // the deletion erases.
        let mut audit: Vec<String> = Vec::new();
        let mut truncated_log = false;

        for (category, row) in &planned {
            let path = row_str(row, "path").unwrap_or_default();
            // Re-asked here because the scan is a snapshot: a download can
            // start between planning and this line.  The window does not close
            // entirely — that would mean holding the registry's lock across an
            // rmtree of a multi-GB tree, stalling every download start — but a
            // transfer that begins inside it has not written anything yet.
            if category == "models" && is_downloading(path) {
                tracing::info!(
                    target: LOG_TARGET,
                    "[storage] Skipping {}: a download started while cleaning",
                    path
                );
                continue;
            }

            let truncate = row_str(row, "action") == Some("truncate");
            let (ok, error) = remove_dispatch(path, truncate);
            if ok {
                deleted.push(deleted_item(category, row));
                audit.push(format!("Removed {} ({} bytes)", path, row_bytes(row)));
                tracing::info!(target: LOG_TARGET, "[storage] Removed {} ({} bytes)", path, row_bytes(row));
                truncated_log = truncated_log || truncate;
            } else {
                let mut item = Row::new();
                item.insert("category".into(), Value::String(category.clone()));
                item.insert("path".into(), Value::String(path.to_string()));
                item.insert("error".into(), error.clone().map(Value::String).unwrap_or(Value::Null));
                failed.push(item);
                let shown = error.as_deref().unwrap_or("None");
                audit.push(format!("Could not remove {path}: {shown}"));
                tracing::warn!(target: LOG_TARGET, "[storage] Could not remove {}: {}", path, shown);
            }
        }

        if !model_rows.is_empty() {
            let deleted_paths: HashSet<&str> = deleted.iter().filter_map(|d| row_str(d, "path")).collect();
            let deleted_ids: HashSet<String> = model_rows
                .iter()
                .filter(|r| row_str(r, "path").map(|p| deleted_paths.contains(p)).unwrap_or(false))
                .filter_map(|r| row_str(r, "model_id").map(String::from))
                .collect();
            {
                let mut workspace = shared.write();
                models_marked = mark_models_removed(&mut workspace, &deleted_ids)?;
            }
            prune_empty_orgs(&shared.read());
        }

        if truncated_log {
            // Everything logged above is gone from server.log now.  Replaying
            // it is not merely tidier than ordering the log sweep first:
            // handlers are fed through a queue listener, so a record emitted
            // before the truncate can still be sitting in the queue when the
            // stream is reopened.  Position in the sweep cannot make the record
            // reliable; writing it again afterwards can.
            tracing::info!(target: LOG_TARGET, "[storage] Sweep record, replayed after server.log was emptied:");
            for line in &audit {
                tracing::info!(target: LOG_TARGET, "[storage] {}", line);
            }
            if !models_marked.is_empty() {
                tracing::info!(target: LOG_TARGET, "[storage] Marked as removed: {}", models_marked.join(", "));
            }
        }
        Ok(())
    }));

    // ``finally``: every permit goes back, whatever happened above.
    release_inference(model_service, held);
    match outcome {
        Ok(result) => result?,
        Err(panic) => std::panic::resume_unwind(panic),
    }

    report(model_service, deleted, failed, models_marked, false)
}

/// Remove what a cancelled download left behind, and clear its flags.
///
/// Deletes `models/<org>/<repo>` outright rather than picking the `.part`
/// sidecars out of it.  A vision model downloads its weights and its mmproj
/// projector in parallel, so a cancel routinely lands with one finished and
/// one half-written — and a tree holding published weights and no projector is
/// the shape `_gguf_exists` reads as "already available", which would make the
/// model permanently unloadable while the picker showed it ready.  There is no
/// verified-weights-lost hazard in exchange: this runs only when a *running*
/// transfer observed the cancel, so the window between the last rename and the
/// thread unwinding is milliseconds, and a cancel arriving after that finds no
/// registration and never reaches here.
///
/// Deliberately not `run_cleanup(models=true, model_ids=[model_id])`, for
/// four reasons:
///
/// * `plan_cleanup` refuses ("Not removable…") any tree the scan does not call
///   `partial` — which includes a cancel that landed before a single `.part`
///   existed, for a model the caller had enabled;
/// * [`run_cleanup`] re-consults `is_downloading` before each removal and
///   would skip this very directory if the mark had not yet been released;
/// * it drains every inference permit with a 30s deadline, which can refuse
///   outright during an active chat — far too heavy for a button press;
/// * it unloads the model and resets the LLM executor, a large user-visible
///   side effect for bytes nothing ever loaded.
///
/// Call only once the transfer's threads have stopped.  Both of the
/// downloader's worker pools join on exit, so by the time `download()` returns
/// no worker can still hold the file open.
pub fn discard_partial_download(
    model_service: Option<&dyn StorageModelService>,
    model_id: &str,
) -> Result<Row, StorageError> {
    let hook = DISCARD_HOOK.read().clone();
    if let Some(hook) = hook {
        return hook(model_id);
    }

    let shared = load_workspace(model_service)?;
    let models_dir = shared.read().models_dir.clone();
    if models_dir.is_empty() {
        return Err(StorageError::RuntimeError("Workspace has no models directory".to_string()));
    }

    // ``os.path.join(models_dir, *model_id.split("/"))``
    let mut joined = PathBuf::from(&models_dir);
    for part in model_id.split('/') {
        joined.push(part);
    }
    let path = joined.to_string_lossy().into_owned();

    let outcome = |path: &str, removed: bool, freed: u64, files: u64, error: Option<String>, marked: Vec<String>| {
        let mut r = Row::new();
        r.insert("model_id".into(), Value::String(model_id.to_string()));
        r.insert("path".into(), Value::String(path.to_string()));
        r.insert("removed".into(), Value::Bool(removed));
        r.insert("freed".into(), Value::from(freed));
        r.insert("files".into(), Value::from(files));
        r.insert("error".into(), error.map(Value::String).unwrap_or(Value::Null));
        r.insert("models_marked".into(), Value::Array(marked.into_iter().map(Value::String).collect()));
        r
    };

    if !isdir(&path) {
        // Cancelled before a byte landed, or the directory is already gone.
        // Not an error: the caller asked for the bytes to be absent, and they
        // are.  No flags are touched either — ``downloaded`` is set only after
        // a transfer completes, so a download that wrote nothing left nothing
        // to correct.
        return Ok(outcome(&path, false, 0, 0, None, Vec::new()));
    }

    let (size, files) = walk_size(&path);
    let resolved = {
        let workspace = shared.read();
        validate(&path, "models", &workspace)?
    };

    let (ok, error) = remove_dispatch(&resolved, false);
    if !ok {
        tracing::warn!(
            target: LOG_TARGET,
            "[storage] Could not discard {}: {}",
            resolved,
            error.as_deref().unwrap_or("None")
        );
        return Ok(outcome(&resolved, false, 0, 0, error, Vec::new()));
    }

    tracing::info!(target: LOG_TARGET, "[storage] Discarded cancelled download {} ({} bytes)", resolved, size);
    prune_empty_orgs(&shared.read());
    let marked = {
        let mut workspace = shared.write();
        let ids: HashSet<String> = std::iter::once(model_id.to_string()).collect();
        mark_models_removed(&mut workspace, &ids)?
    };
    Ok(outcome(&resolved, true, size, files, None, marked))
}

/// Flip the workspace flags for models whose weights are gone, and persist.
/// Returns the names of the models marked.
///
/// The workspace record is the framework's own source of truth for what is
/// downloaded; leaving it claiming a deleted model would keep offering it in
/// the picker.
fn mark_models_removed(workspace: &mut Workspace, deleted_ids: &HashSet<String>) -> Result<Vec<String>, StorageError> {
    if deleted_ids.is_empty() {
        return Ok(Vec::new());
    }

    let mut marked = Vec::new();
    for m in workspace.models.iter_mut() {
        if deleted_ids.contains(&m.model_id) && (m.downloaded || m.enabled || m.is_active) {
            m.downloaded = false;
            m.enabled = false;
            m.is_active = false;
            marked.push(m.model_name.clone());
        }
    }

    if !marked.is_empty() {
        workspace.save().map_err(|e| StorageError::Other(e.to_string()))?;
        tracing::info!(target: LOG_TARGET, "[storage] Marked as removed: {}", marked.join(", "));
    }
    Ok(marked)
}

/// Build the cleanup result, including a fresh usage report.
fn report(
    model_service: Option<&dyn StorageModelService>,
    deleted: Vec<Row>,
    failed: Vec<Row>,
    models_marked: Vec<String>,
    dry_run: bool,
) -> Result<Row, StorageError> {
    let mut freed_by_category = Row::new();
    for item in &deleted {
        let category = row_str(item, "category").unwrap_or_default().to_string();
        let current = freed_by_category.get(&category).and_then(Value::as_u64).unwrap_or(0);
        freed_by_category.insert(category, Value::from(current + row_bytes(item)));
    }

    let freed: u64 = deleted.iter().map(row_bytes).sum();
    let restart_required = !failed.is_empty();

    let mut result = Row::new();
    result.insert("dry_run".into(), Value::Bool(dry_run));
    // Summed from what was actually removed: a before/after comparison of the
    // tree would drift, since the server keeps writing as we work.
    result.insert("freed".into(), Value::from(freed));
    result.insert("freed_by_category".into(), Value::Object(freed_by_category));
    result.insert("deleted".into(), Value::Array(deleted.into_iter().map(Value::Object).collect()));
    result.insert("failed".into(), Value::Array(failed.into_iter().map(Value::Object).collect()));
    result.insert(
        "models_marked".into(),
        Value::Array(models_marked.into_iter().map(Value::String).collect()),
    );
    // A file still mapped by a worker (Windows) needs the agent restarted
    // before it will go.
    result.insert("restart_required".into(), Value::Bool(restart_required));
    result.insert("usage".into(), Value::Object(compute_usage(model_service)?));
    Ok(result)
}

// ── ModelService adapter ────────────────────────────────────────────────

/// `ModelService._inference_lock` seen through [`InferencePermits`].
///
/// The service's semaphore only offers a blocking `acquire`, so a bounded
/// wait hands the blocking call to a helper thread.  If the deadline passes
/// first the helper is told it was abandoned, and gives its permit straight
/// back the moment it gets one — so a timed-out drain never strands a permit.
struct ServiceInferencePermits {
    service: Arc<ModelService>,
}

impl InferencePermits for ServiceInferencePermits {
    fn acquire(&self, timeout: Duration) -> bool {
        // 0 = waiting, 1 = acquired, 2 = abandoned by the caller.
        let state = Arc::new((parking_lot::Mutex::new(0u8), parking_lot::Condvar::new()));
        let helper_state = Arc::clone(&state);
        let service = Arc::clone(&self.service);
        let spawned = std::thread::Builder::new().name("storage-drain".into()).spawn(move || {
            service.inference_lock.acquire();
            let (lock, cond) = &*helper_state;
            let mut s = lock.lock();
            if *s == 2 {
                drop(s);
                service.inference_lock.release();
            } else {
                *s = 1;
                cond.notify_all();
            }
        });
        if spawned.is_err() {
            return false;
        }

        let (lock, cond) = &*state;
        let mut s = lock.lock();
        let deadline = Instant::now() + timeout;
        while *s == 0 {
            if cond.wait_until(&mut s, deadline).timed_out() {
                break;
            }
        }
        if *s == 1 {
            true
        } else {
            *s = 2;
            false
        }
    }

    fn release(&self) {
        self.service.inference_lock.release();
    }
}

impl StorageModelService for ModelService {
    fn ensure_workspace(&self) -> anyhow::Result<Option<SharedWorkspace>> {
        ModelService::ensure_workspace(self, None).map(Some)
    }

    fn refresh_project_state(&self, project_id: &str) -> anyhow::Result<bool> {
        Ok(ModelService::refresh_project_state(self, Some(project_id)))
    }

    fn resident_model_names(&self) -> HashSet<String> {
        let mut names = HashSet::new();
        for slot in [&self.model_name, &self.decision_model_name, &self.large_model_name, &self.vision_model_name] {
            if let Some(name) = slot.read().as_ref() {
                if !name.is_empty() {
                    names.insert(name.clone());
                }
            }
        }
        // A capability slot holding the faulted sentinel is a latch, not a
        // live context, so it does not pin the files.
        for (name, cached) in self.cap_llms.lock().iter() {
            if !cached.is_gpu_faulted() {
                names.insert(name.clone());
            }
        }
        names
    }

    fn task_status(&self, name: &str) -> Option<Value> {
        let queue = Arc::clone(&*self.task_queue.read());
        Some(queue.get_status(Some(name)))
    }

    fn inference_lock(&self) -> Option<Arc<dyn InferencePermits>> {
        Some(Arc::new(ServiceInferencePermits { service: self.arc() }))
    }

    fn unload_model_by_name_unlocked(&self, name: &str) -> anyhow::Result<()> {
        ModelService::unload_model_by_name_unlocked(self, name);
        Ok(())
    }

    fn reset_llm_executor(&self) {
        ModelService::reset_llm_executor(self);
    }
}
