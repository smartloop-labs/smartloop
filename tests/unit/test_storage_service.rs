//! Port of `tests/unit/test_storage_service.py`.
//!
//! Tests for the storage usage/cleanup service.
//!
//! The safety rails are the point of these tests: a bug here deletes a user's
//! models, credentials, or project data.  So most of what follows asserts what
//! must *not* happen.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use parking_lot::{Condvar, Mutex, RwLock};
use serde_json::{Map, Value};
use serial_test::serial;

use smartloop::api::services::storage_service::{
    self, CleanupOptions, InferencePermits, Row, StorageError, StorageModelService,
};
use smartloop::config::app_settings::AppSettings;
use smartloop::model::downloader::inflight::downloading;
use smartloop::workspace::workspace::{self as ws_mod, ProjectModel, SharedWorkspace};
use smartloop::workspace::{Project, Workspace, WorkspaceModel};

// Mirrors the real config: only the bootstrap orchestrator is is_active, while
// the models the user picked carry enabled=True.  Keying the keep-rule on
// is_active alone would delete the picked ones.  The bootstrap model is
// recorded under its canonical name while its model_id keeps the pre-rename
// spelling — renaming the directory would orphan already-downloaded weights.
const MODELS: &[(&str, &str, bool, bool)] = &[
    ("sl-mini", "smartloop-ai/slp-orchestra-mini", true, true),
    ("gemma4-e4b", "google/gemma-4-e4b-it", true, false),
    ("gemma3-4b", "google/gemma-3-4b-it", false, false),
    ("ministral3-3b", "mistralai/Ministral-3-3B-Instruct-2512", false, false),
];

const REMOVABLE_IDS: &[&str] = &["google/gemma-3-4b-it", "mistralai/Ministral-3-3B-Instruct-2512"];
const CURRENT_VERSION: &str = "1.1.0";
const OLD_VERSION: &str = "1.0.9";

// Files that must survive every category, in every combination.
const PROTECTED_FILES: &[&str] = &[
    "config.json", ".credentials", ".key", ".installed-version", ".binary-hash",
    ".archive-etag", "server.pid", "server.port", "version", "installed",
    "studio.db", "token_usage.jsonl",
];

fn write(path: &Path, content: &str) -> String {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_s(path: &str, content: &str) -> String {
    write(Path::new(path), content)
}

/// Backdate a path's mtime so the orphan sweep treats it as settled.
fn age(path: &str) {
    age_by(path, storage_service::ORPHAN_MIN_AGE_S + 3600);
}

fn age_by(path: &str, seconds: u64) {
    let old = SystemTime::now() - Duration::from_secs(seconds);
    let f = fs::File::open(path).unwrap();
    f.set_times(fs::FileTimes::new().set_accessed(old).set_modified(old)).unwrap();
}

fn join(a: &str, b: &str) -> String {
    Path::new(a).join(b).to_string_lossy().into_owned()
}

fn realpath(p: &str) -> String {
    fs::canonicalize(p).unwrap().to_string_lossy().into_owned()
}

fn exists(p: &str) -> bool {
    fs::symlink_metadata(p).is_ok()
}

fn isdir(p: &str) -> bool {
    Path::new(p).is_dir()
}

fn isfile(p: &str) -> bool {
    Path::new(p).is_file()
}

fn removable_ids_plus_unregistered() -> HashSet<String> {
    let mut set: HashSet<String> = REMOVABLE_IDS.iter().map(|s| s.to_string()).collect();
    set.insert("Qwen/unregistered-model".to_string());
    set
}

// ── Fakes for the duck-typed ModelService ───────────────────────────────

/// `MagicMock()` standing in for `_inference_lock`: `acquire` answers a fixed
/// value and counts its calls.
struct MockPermits {
    result: AtomicBool,
    acquire_calls: AtomicUsize,
}

impl InferencePermits for MockPermits {
    fn acquire(&self, _timeout: Duration) -> bool {
        self.acquire_calls.fetch_add(1, Ordering::SeqCst);
        self.result.load(Ordering::SeqCst)
    }
    fn release(&self) {}
}

/// `threading.Semaphore(n)`.
struct Semaphore {
    permits: Mutex<usize>,
    cond: Condvar,
}

impl Semaphore {
    fn new(n: usize) -> Self {
        Self { permits: Mutex::new(n), cond: Condvar::new() }
    }
}

impl InferencePermits for Semaphore {
    fn acquire(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut permits = self.permits.lock();
        while *permits == 0 {
            if self.cond.wait_until(&mut permits, deadline).timed_out() {
                break;
            }
        }
        if *permits > 0 {
            *permits -= 1;
            true
        } else {
            false
        }
    }
    fn release(&self) {
        *self.permits.lock() += 1;
        self.cond.notify_one();
    }
}

type RefreshFn = Box<dyn Fn(&str) -> anyhow::Result<bool> + Send + Sync>;
type StatusFn = Box<dyn Fn(&str) -> Option<Value> + Send + Sync>;
type UnloadFn = Box<dyn Fn(&str) + Send + Sync>;

/// The mock `ModelService` of the `env` fixture.
struct FakeService {
    /// `_ensure_workspace.return_value`.
    workspace: RwLock<Option<SharedWorkspace>>,
    /// `refresh_project_state.call_args_list`.
    refreshed: Mutex<Vec<String>>,
    /// `refresh_project_state.side_effect`.
    refresh_effect: Mutex<Option<RefreshFn>>,
    /// `_model_name` & co. plus the live `_cap_llm_<name>` slots.
    resident: Mutex<HashSet<String>>,
    /// `_task_queue.get_status.side_effect`.
    task_status: Mutex<StatusFn>,
    /// `_inference_lock`.
    lock: Mutex<Arc<dyn InferencePermits>>,
    mock_permits: Arc<MockPermits>,
    /// `_unload_model_by_name_unlocked.call_args_list`.
    unloaded: Mutex<Vec<String>>,
    /// `_unload_model_by_name_unlocked.side_effect`.
    unload_effect: Mutex<Option<UnloadFn>>,
    /// `_reset_llm_executor` calls.
    reset_calls: AtomicUsize,
    /// `unload_model` / `unload_model_by_name`: the public unloads.  The
    /// storage seam does not even expose them, so nothing can call them.
    public_unload_calls: AtomicUsize,
}

impl FakeService {
    fn new(workspace: SharedWorkspace) -> Self {
        let mock_permits = Arc::new(MockPermits { result: AtomicBool::new(true), acquire_calls: AtomicUsize::new(0) });
        Self {
            workspace: RwLock::new(Some(workspace)),
            refreshed: Mutex::new(Vec::new()),
            refresh_effect: Mutex::new(None),
            resident: Mutex::new(HashSet::new()),
            // Answer only for tasks that are really registered.  A blanket
            // return value let a test name any task and get a reply, which is
            // how two tests came to assert that a "downloader" task guarded
            // downloads when nothing has ever registered one.  Mirrors
            // TaskQueue.get_status.
            task_status: Mutex::new(Box::new(|name: &str| {
                if name == "indexer" {
                    Some(serde_json::json!({"status": "idle"}))
                } else {
                    Some(serde_json::json!({"error": format!("Unknown task: {name}")}))
                }
            })),
            lock: Mutex::new(mock_permits.clone() as Arc<dyn InferencePermits>),
            mock_permits,
            unloaded: Mutex::new(Vec::new()),
            unload_effect: Mutex::new(None),
            reset_calls: AtomicUsize::new(0),
            public_unload_calls: AtomicUsize::new(0),
        }
    }
}

impl StorageModelService for FakeService {
    fn ensure_workspace(&self) -> anyhow::Result<Option<SharedWorkspace>> {
        Ok(self.workspace.read().clone())
    }

    fn refresh_project_state(&self, project_id: &str) -> anyhow::Result<bool> {
        self.refreshed.lock().push(project_id.to_string());
        match self.refresh_effect.lock().as_ref() {
            Some(effect) => effect(project_id),
            None => Ok(false),
        }
    }

    fn resident_model_names(&self) -> HashSet<String> {
        self.resident.lock().clone()
    }

    fn task_status(&self, name: &str) -> Option<Value> {
        (self.task_status.lock())(name)
    }

    fn inference_lock(&self) -> Option<Arc<dyn InferencePermits>> {
        Some(self.lock.lock().clone())
    }

    fn unload_model_by_name_unlocked(&self, name: &str) -> anyhow::Result<()> {
        self.unloaded.lock().push(name.to_string());
        if let Some(effect) = self.unload_effect.lock().as_ref() {
            effect(name);
        }
        Ok(())
    }

    fn reset_llm_executor(&self) {
        self.reset_calls.fetch_add(1, Ordering::SeqCst);
    }
}

// ── Fixtures ────────────────────────────────────────────────────────────

/// The `home` + `env` fixtures: a fake ~/.smartloop with every category
/// populated, the framework pointed at it, and a mock ModelService.
struct Env {
    _tmp: tempfile::TempDir,
    home_dir: String,
    outside: String,
    workspace: SharedWorkspace,
    workspace_dir: String,
    models_dir: String,
    project_id: String,
    project_dir: String,
    orphan_id: String,
    fresh_orphan_id: String,
    config_file: String,
    session_id: String,
    untracked_id: String,
    service: Arc<FakeService>,
    settings: AppSettings,
}

impl Env {
    fn svc(&self) -> Option<&dyn StorageModelService> {
        Some(&*self.service as &dyn StorageModelService)
    }

    fn usage(&self) -> Row {
        storage_service::compute_usage(self.svc()).unwrap()
    }

    fn cleanup(&self, opts: CleanupOptions) -> Row {
        storage_service::run_cleanup(self.svc(), opts).unwrap()
    }

    /// `env["project"].model_settings.append(...)`.
    fn add_project_model(&self, pm: ProjectModel) {
        self.workspace.write().projects[0].model_settings.push(pm);
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        storage_service::set_settings(None);
        storage_service::set_remove_hook(None);
        storage_service::set_is_downloading_hook(None);
        storage_service::set_rmtree_hook(None);
        ws_mod::set_settings_home_dir(None);
        Workspace::set_config_file(None);
    }
}

fn project_model(name: &str, enabled: bool, is_active: bool) -> ProjectModel {
    ProjectModel { model_name: name.to_string(), enabled, is_active, ..Default::default() }
}

/// Build the fixture.  `test_name` prefixes the temp directory the way pytest
/// names `tmp_path` after the test, so error messages carrying a path read
/// the same.
fn env(test_name: &str) -> Env {
    let tmp = tempfile::Builder::new().prefix(test_name).tempdir().unwrap();
    let tmp_path = tmp.path().to_path_buf();
    let home_dir = tmp_path.join("dot_smartloop");
    let outside = tmp_path.join("outside");
    fs::create_dir(&outside).unwrap();
    fs::write(outside.join("precious.txt"), "user data").unwrap();
    fs::create_dir(&home_dir).unwrap();

    let workspace_id = uuid::Uuid::new_v4().to_string();
    let workspace_dir = home_dir.join(&workspace_id);
    let models_dir = workspace_dir.join("models");
    fs::create_dir_all(&models_dir).unwrap();

    // Models on disk: every registered one, plus an unregistered leftover.
    for (_name, model_id, _enabled, _active) in MODELS {
        write(&models_dir.join(model_id).join("weights.gguf"), &"w".repeat(100));
    }
    write(&models_dir.join("Qwen").join("unregistered-model").join("weights.gguf"), &"w".repeat(50));
    // A flat file directly under embeddings/, as the embedding downloader writes.
    write(&models_dir.join("embeddings").join("bge-m3.gguf"), &"e".repeat(40));
    // A symlink pointing out of home: never sized, never followed.
    #[cfg(unix)]
    std::os::unix::fs::symlink(&outside, models_dir.join("google").join("link")).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_dir(&outside, models_dir.join("google").join("link")).unwrap();

    let project_id = uuid::Uuid::new_v4().to_string();
    let project_dir = workspace_dir.join(&project_id);
    write(&project_dir.join("documents").join("readme.txt"), "doc");
    let orphan_id = uuid::Uuid::new_v4().to_string();
    write(&workspace_dir.join(&orphan_id).join("documents").join("stale.txt"), "stale");
    // Backdated past ORPHAN_MIN_AGE_S: the sweep leaves a recently touched
    // directory alone, so an orphan created here and now would never be offered.
    age(&workspace_dir.join(&orphan_id).to_string_lossy());
    // An orphan by shape but touched moments ago — the age guard keeps it.
    let fresh_orphan_id = uuid::Uuid::new_v4().to_string();
    write(&workspace_dir.join(&fresh_orphan_id).join("documents").join("live.txt"), "live");
    // A non-uuid directory is not a project shape and must be left alone.
    write(&workspace_dir.join("not-a-project").join("keep.txt"), "keep");
    for name in ["workspace.db", "workspace.db-wal", "workspace.db-shm"] {
        write(&workspace_dir.join(name), "db");
    }

    for name in PROTECTED_FILES {
        write(&home_dir.join(name), "protected");
    }
    fs::write(home_dir.join(".installed-version"), CURRENT_VERSION).unwrap();
    fs::write(home_dir.join("version"), CURRENT_VERSION).unwrap();

    write(&home_dir.join("server.log"), &"l".repeat(300));
    write(&home_dir.join("server.log.1"), &"l".repeat(500));
    write(&home_dir.join("server.log.2"), &"l".repeat(700));

    write(&home_dir.join(CURRENT_VERSION).join("slp"), "binary");
    write(&home_dir.join(OLD_VERSION).join("slp"), "old binary");

    write(&home_dir.join("cache").join("slp-1.1.0.tar.gz"), &"c".repeat(200));
    write(&home_dir.join("templates").join("project_1.zip"), &"t".repeat(100));

    let session_id = uuid::Uuid::new_v4().to_string();
    let untracked_id = uuid::Uuid::new_v4().to_string();
    write(&home_dir.join("conversations").join(format!("{session_id}.jsonl")), "{\"role\":\"user\"}\n");
    write(&home_dir.join("conversations").join(format!("{untracked_id}.jsonl")), "{\"role\":\"user\"}\n");

    let models_dir_s = models_dir.to_string_lossy().into_owned();
    let project = Project {
        id: project_id.clone(),
        project_dir: project_dir.to_string_lossy().into_owned(),
        current: true,
        name: Some("test-project".to_string()),
        sessions: vec![session_id.clone()],
        ..Default::default()
    };
    let mut workspace = Workspace::new(workspace_id);
    workspace.workspace_dir = workspace_dir.to_string_lossy().into_owned();
    workspace.models_dir = models_dir_s.clone();
    workspace.projects = vec![project];
    workspace.models = MODELS
        .iter()
        .map(|(name, model_id, enabled, active)| WorkspaceModel {
            model_name: name.to_string(),
            model_id: model_id.to_string(),
            downloaded: true,
            enabled: *enabled,
            is_active: *active,
            models_dir: models_dir_s.clone(),
            ..Default::default()
        })
        .collect();

    let config_file = home_dir.join("config.json");
    let mut ws_data = serde_json::to_value(&workspace).unwrap();
    ws_data.as_object_mut().unwrap().remove("current_project");
    fs::write(&config_file, serde_json::to_string_pretty(&serde_json::json!({"id": "", "workspace": ws_data})).unwrap())
        .unwrap();

    // Point the framework at the fake home and hand back a mock ModelService.
    let mut settings = AppSettings::new();
    settings.slp_home = Some(home_dir.to_string_lossy().into_owned());
    storage_service::set_settings(Some(settings.clone()));
    ws_mod::set_settings_home_dir(Some(home_dir.to_string_lossy().into_owned()));
    Workspace::set_config_file(Some(config_file.to_string_lossy().into_owned()));
    // The test binary runs from outside the fake home, so version detection
    // has to fall back to the markers.

    let shared: SharedWorkspace = Arc::new(RwLock::new(workspace));
    // No model resident, no task running, permits always available.
    let service = Arc::new(FakeService::new(shared.clone()));

    Env {
        home_dir: home_dir.to_string_lossy().into_owned(),
        outside: outside.to_string_lossy().into_owned(),
        workspace: shared,
        workspace_dir: workspace_dir.to_string_lossy().into_owned(),
        models_dir: models_dir_s,
        project_id,
        project_dir: project_dir.to_string_lossy().into_owned(),
        orphan_id,
        fresh_orphan_id,
        config_file: config_file.to_string_lossy().into_owned(),
        session_id,
        untracked_id,
        service,
        settings,
        _tmp: tmp,
    }
}

fn items(usage: &Row, category: &str) -> Vec<Map<String, Value>> {
    usage["categories"][category]["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_object().unwrap().clone())
        .collect()
}

fn by_model_id(usage: &Row) -> HashMap<String, Map<String, Value>> {
    items(usage, "models").into_iter().map(|r| (r["model_id"].as_str().unwrap().to_string(), r)).collect()
}

fn removable_set(rows: &HashMap<String, Map<String, Value>>) -> HashSet<String> {
    rows.iter().filter(|(_, r)| r["removable"] == true).map(|(k, _)| k.clone()).collect()
}

fn opts() -> CleanupOptions {
    CleanupOptions::default()
}

fn models_only() -> CleanupOptions {
    CleanupOptions { models: true, ..Default::default() }
}

fn everything() -> CleanupOptions {
    CleanupOptions { models: true, versions: true, logs: true, cache: true, orphan_projects: true, ..Default::default() }
}

fn deleted_paths(result: &Row) -> Vec<String> {
    result["deleted"].as_array().unwrap().iter().map(|d| d["path"].as_str().unwrap().to_string()).collect()
}

fn models_marked(result: &Row) -> Vec<String> {
    result["models_marked"].as_array().unwrap().iter().map(|d| d.as_str().unwrap().to_string()).collect()
}

/// Rewrite `model_id`'s directory to a given set of files.
///
/// `keep` lands as finished files, `part` as unfinished sidecars.  The
/// fixture's own `weights.gguf` is cleared first so each case states the whole
/// directory rather than inheriting a stray complete file.
fn make_partial(models_dir: &str, model_id: &str, keep: &[&str], part: &[&str]) -> String {
    let target = join(models_dir, model_id);
    for entry in fs::read_dir(&target).unwrap() {
        fs::remove_file(entry.unwrap().path()).unwrap();
    }
    for name in keep {
        write_s(&join(&target, name), &"w".repeat(100));
    }
    for name in part {
        write_s(&join(&target, name), &"p".repeat(100));
    }
    target
}

fn make_partial_default(models_dir: &str, model_id: &str) -> String {
    make_partial(models_dir, model_id, &[], &["weights.gguf.part"])
}

// ── Log capture (pytest `caplog`) ───────────────────────────────────────

mod caplog {
    use std::sync::Arc;

    use parking_lot::Mutex;
    use tracing::field::{Field, Visit};
    use tracing::{Event, Level, Subscriber};
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

    struct Capture {
        target: &'static str,
        records: Arc<Mutex<Vec<String>>>,
    }

    struct MessageVisitor(String);

    impl Visit for MessageVisitor {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                self.0 = format!("{value:?}");
            }
        }
    }

    impl<S: Subscriber> Layer<S> for Capture {
        fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
            let meta = event.metadata();
            if meta.target() != self.target || *meta.level() > Level::INFO {
                return;
            }
            let mut visitor = MessageVisitor(String::new());
            event.record(&mut visitor);
            self.records.lock().push(visitor.0);
        }
    }

    /// Run `f` capturing INFO+ records of the `target` logger, in order.
    pub fn at_info<R>(target: &'static str, f: impl FnOnce() -> R) -> (R, Vec<String>) {
        let records = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(Capture { target, records: records.clone() });
        let result = tracing::subscriber::with_default(subscriber, f);
        let messages = records.lock().clone();
        (result, messages)
    }
}

// ── TestComputeUsage ────────────────────────────────────────────────────

#[test]
#[serial]
fn test_totals_match_a_walk_of_the_tree() {
    let env = env("test_totals_match_a_walk_of_the_tree");
    let usage = env.usage();

    let mut expected = 0u64;
    for entry in walkdir::WalkDir::new(&env.home_dir).follow_links(false) {
        let entry = entry.unwrap();
        if entry.file_type().is_file() {
            expected += entry.metadata().unwrap().len();
        }
    }

    assert_eq!(usage["total"], expected);
    assert_eq!(usage["home_dir"], realpath(&env.home_dir).as_str());
}

#[test]
#[serial]
fn test_enabled_model_is_never_removable() {
    let env = env("test_enabled_model_is_never_removable");
    let rows = by_model_id(&env.usage());

    let picked = &rows["google/gemma-4-e4b-it"];
    assert_eq!(picked["removable"], false);
    assert_eq!(picked["keep_reason"], "enabled");
}

#[test]
#[serial]
fn test_bundled_and_active_models_are_kept() {
    let env = env("test_bundled_and_active_models_are_kept");
    let rows = by_model_id(&env.usage());

    assert_eq!(rows["smartloop-ai/slp-orchestra-mini"]["removable"], false);
    assert_eq!(rows["smartloop-ai/slp-orchestra-mini"]["keep_reason"], "bundled");
}

#[test]
#[serial]
fn test_disabled_models_are_removable() {
    let env = env("test_disabled_models_are_removable");
    let rows = by_model_id(&env.usage());

    assert_eq!(removable_set(&rows), removable_ids_plus_unregistered());
}

#[test]
#[serial]
fn test_resident_model_is_kept_even_when_disabled() {
    let env = env("test_resident_model_is_kept_even_when_disabled");
    // `_cap_llm_gemma3_4b = None` never loaded; `_cap_llm_gemma3-4b` is live.
    env.service.resident.lock().insert("gemma3-4b".to_string());

    let rows = by_model_id(&env.usage());

    assert_eq!(rows["google/gemma-3-4b-it"]["removable"], false);
    assert_eq!(rows["google/gemma-3-4b-it"]["keep_reason"], "loaded");
}

#[test]
#[serial]
fn test_project_enabled_model_is_kept_though_disabled_app_wide() {
    let env = env("test_project_enabled_model_is_kept_though_disabled_app_wide");
    // A project override resolves ahead of the workspace entry, so this model
    // is in use even though the app-wide picker has it off.
    env.add_project_model(project_model("gemma3-4b", true, false));

    let rows = by_model_id(&env.usage());

    assert_eq!(rows["google/gemma-3-4b-it"]["removable"], false);
    assert_eq!(rows["google/gemma-3-4b-it"]["keep_reason"], "enabled");
}

#[test]
#[serial]
fn test_project_active_model_is_kept_though_disabled_app_wide() {
    let env = env("test_project_active_model_is_kept_though_disabled_app_wide");
    env.add_project_model(project_model("gemma3-4b", false, true));

    let rows = by_model_id(&env.usage());

    assert_eq!(rows["google/gemma-3-4b-it"]["removable"], false);
    assert_eq!(rows["google/gemma-3-4b-it"]["keep_reason"], "active");
}

#[test]
#[serial]
fn test_model_off_in_every_scope_stays_removable() {
    let env = env("test_model_off_in_every_scope_stays_removable");
    // The guard against over-keeping: a project entry that says off must not
    // protect anything, or cleanup quietly frees nothing.
    env.add_project_model(project_model("gemma3-4b", false, false));

    let rows = by_model_id(&env.usage());

    assert_eq!(rows["google/gemma-3-4b-it"]["removable"], true);
}

#[test]
#[serial]
fn test_a_model_every_project_disabled_is_removable_though_the_default_is_on() {
    // The workspace entry is a template, not a scope.
    //
    // It holds the defaults a *new* project is seeded from, so a model it
    // enables is not thereby in use by anything.  Treating it as a scope of
    // its own made the app-default pin weights every existing project had
    // turned off — and since the model picker only ever writes project
    // overrides, that was disk the UI could never offer to reclaim.
    let env = env("test_a_model_every_project_disabled_is_removable_though_the_default_is_on");
    // gemma4-e4b is enabled on the workspace entry (see MODELS).
    env.add_project_model(project_model("gemma4-e4b", false, false));

    let rows = by_model_id(&env.usage());

    assert_eq!(rows["google/gemma-4-e4b-it"]["removable"], true);
    assert_eq!(rows["google/gemma-4-e4b-it"]["keep_reason"], Value::Null);
}

#[test]
#[serial]
fn test_a_project_without_an_override_still_inherits_the_default() {
    // The other half of the same rule: dropping the workspace scope must not
    // cost the template its meaning.  This project says nothing about
    // gemma4-e4b, so it reads through to the app-default and keeps it.
    let env = env("test_a_project_without_an_override_still_inherits_the_default");
    assert!(env.workspace.read().projects[0].model_settings.is_empty());

    let rows = by_model_id(&env.usage());

    assert_eq!(rows["google/gemma-4-e4b-it"]["removable"], false);
    assert_eq!(rows["google/gemma-4-e4b-it"]["keep_reason"], "enabled");
}

#[test]
#[serial]
fn test_with_no_projects_the_default_is_consulted_directly() {
    // Nothing to read through, so the template is all there is — otherwise a
    // fresh workspace would offer its just-downloaded weights for deletion.
    let env = env("test_with_no_projects_the_default_is_consulted_directly");
    env.workspace.write().projects.clear();

    let rows = by_model_id(&env.usage());

    assert_eq!(rows["google/gemma-4-e4b-it"]["removable"], false);
    assert_eq!(rows["google/gemma-4-e4b-it"]["keep_reason"], "enabled");
}

#[test]
#[serial]
fn test_every_project_is_refreshed_before_deciding() {
    // Under supervision a project's file is written by its own child.
    //
    // This process holds the copy it took when the project first appeared and
    // nothing tells it when that went stale, so the sweep would otherwise
    // decide what to delete from a snapshot.
    let env = env("test_every_project_is_refreshed_before_deciding");
    env.usage();

    let refreshed = env.service.refreshed.lock().clone();
    assert_eq!(refreshed, vec![env.project_id.clone()]);
}

#[test]
#[serial]
fn test_a_childs_write_reaches_the_decision() {
    // The direction that actually costs something: a model a child turned
    // back *on* must stop being offered, or the sweep deletes weights a
    // project is using.
    let env = env("test_a_childs_write_reaches_the_decision");
    // gemma3-4b is off app-wide, so it starts removable.
    let rows = by_model_id(&env.usage());
    assert_eq!(rows["google/gemma-3-4b-it"]["removable"], true);

    // A child enables it and writes its own file; the refresh is what picks
    // that up on this side.
    let shared = env.workspace.clone();
    *env.service.refresh_effect.lock() = Some(Box::new(move |_project_id: &str| {
        shared.write().projects[0].model_settings.push(project_model("gemma3-4b", true, false));
        Ok(true)
    }));

    let rows = by_model_id(&env.usage());

    assert_eq!(rows["google/gemma-3-4b-it"]["removable"], false);
    assert_eq!(rows["google/gemma-3-4b-it"]["keep_reason"], "enabled");
}

#[test]
#[serial]
fn test_an_unreadable_project_does_not_fail_the_report() {
    // Keeps the copy already held rather than refusing the whole sweep.
    let env = env("test_an_unreadable_project_does_not_fail_the_report");
    *env.service.refresh_effect.lock() = Some(Box::new(|_: &str| Err(anyhow::anyhow!("gone"))));

    let usage = env.usage();

    assert!(!usage["categories"]["models"]["items"].as_array().unwrap().is_empty());
}

#[test]
#[serial]
fn test_the_bootstrap_model_resolves_under_its_legacy_name() {
    // config.json written before the rename records slp-orchestra-mini;
    // canonicalization has to keep it resolving to the same entry.
    let env = env("test_the_bootstrap_model_resolves_under_its_legacy_name");
    env.workspace.write().models[0].model_name = "slp-orchestra-mini".to_string();

    let rows = by_model_id(&env.usage());

    assert_eq!(rows["smartloop-ai/slp-orchestra-mini"]["removable"], false);
    assert_eq!(removable_set(&rows), removable_ids_plus_unregistered());
}

#[test]
#[serial]
fn test_unregistered_model_dir_is_removable_and_flagged() {
    let env = env("test_unregistered_model_dir_is_removable_and_flagged");
    let rows = by_model_id(&env.usage());

    let leftover = &rows["Qwen/unregistered-model"];
    assert_eq!(leftover["registered"], false);
    assert_eq!(leftover["removable"], true);
}

#[test]
#[serial]
fn test_symlink_is_not_counted() {
    let env = env("test_symlink_is_not_counted");
    let usage = env.usage();
    let paths: Vec<String> = items(&usage, "models").iter().map(|r| r["path"].as_str().unwrap().to_string()).collect();

    assert!(!paths.iter().any(|p| p.ends_with("link")));
    // The 9-byte file behind the symlink must not appear in any total.
    assert!(usage["categories"]["models"]["bytes"].as_u64().unwrap() < 1000);
}

#[test]
#[serial]
fn test_current_version_is_kept_old_one_is_not() {
    let env = env("test_current_version_is_kept_old_one_is_not");
    let rows: HashMap<String, Map<String, Value>> = items(&env.usage(), "versions")
        .into_iter()
        .map(|r| (r["version"].as_str().unwrap().to_string(), r))
        .collect();

    assert_eq!(rows[CURRENT_VERSION]["removable"], false);
    assert_eq!(rows[CURRENT_VERSION]["keep_reason"], "running");
    assert_eq!(rows[OLD_VERSION]["removable"], true);
}

#[test]
#[serial]
fn test_log_and_backups_are_reported_with_actions() {
    let env = env("test_log_and_backups_are_reported_with_actions");
    let rows: HashMap<String, Map<String, Value>> = items(&env.usage(), "logs")
        .into_iter()
        .map(|r| {
            let name = Path::new(r["path"].as_str().unwrap()).file_name().unwrap().to_string_lossy().into_owned();
            (name, r)
        })
        .collect();

    assert_eq!(rows["server.log"]["action"], "truncate");
    assert_eq!(rows["server.log.1"]["action"], "delete");
    assert_eq!(rows["server.log.2"]["action"], "delete");
}

#[test]
#[serial]
fn test_orphan_project_is_found_live_one_is_not() {
    let env = env("test_orphan_project_is_found_live_one_is_not");
    let rows = items(&env.usage(), "orphan_projects");

    let ids: Vec<&str> = rows.iter().map(|r| r["project_id"].as_str().unwrap()).collect();
    assert_eq!(ids, vec![env.orphan_id.as_str()]);
    assert_eq!(rows[0]["removable"], true);
}

#[test]
#[serial]
fn test_conversations_are_reported_but_never_reclaimable() {
    let env = env("test_conversations_are_reported_but_never_reclaimable");
    let usage = env.usage();
    let category = &usage["categories"]["conversations"];

    assert_eq!(category["reclaimable"], 0);
    assert!(category["bytes"].as_u64().unwrap() > 0);
    let rows = category["items"].as_array().unwrap();
    assert!(rows.iter().all(|r| r["removable"] == false));
    let tracked: HashMap<&str, bool> =
        rows.iter().map(|r| (r["session_id"].as_str().unwrap(), r["tracked"].as_bool().unwrap())).collect();
    assert!(tracked[env.session_id.as_str()]);
    assert!(!tracked[env.untracked_id.as_str()]);
}

#[test]
#[serial]
fn test_raises_when_workspace_is_unavailable() {
    let env = env("test_raises_when_workspace_is_unavailable");
    *env.service.workspace.write() = None;

    let err = storage_service::compute_usage(env.svc()).unwrap_err();
    assert!(matches!(err, StorageError::RuntimeError(_)), "{err:?}");
}

// ── TestModelUsageGroups ────────────────────────────────────────────────
//
// The split the usage panel draws: in use, framework-owned, reclaimable.
//
// The UI used to show only what it could delete, so every model the user had
// turned on vanished into a "System" remainder alongside config.json.  These
// assert the report names those bytes instead.

#[test]
#[serial]
fn test_the_groups_partition_the_category() {
    let env = env("test_the_groups_partition_the_category");
    let usage = env.usage();
    let models = &usage["categories"]["models"];

    let sum = models["in_use"].as_u64().unwrap()
        + models["system"].as_u64().unwrap()
        + models["reclaimable"].as_u64().unwrap()
        + models["overhead"].as_u64().unwrap();
    assert_eq!(sum, models["bytes"].as_u64().unwrap());
}

#[test]
#[serial]
fn test_a_bundled_model_is_framework_not_user_footprint() {
    let env = env("test_a_bundled_model_is_framework_not_user_footprint");
    let usage = env.usage();
    let models = &usage["categories"]["models"];
    let rows = by_model_id(&env.usage());

    let bootstrap = &rows["smartloop-ai/slp-orchestra-mini"];
    // It is the default *and* active, so it would have been held anyway —
    // being bundled has to win, or the orchestrator reads as a choice the user
    // made and shows up under their own usage.
    assert_eq!(bootstrap["group"], "system");
    assert_eq!(models["system"], bootstrap["bytes"]);
}

#[test]
#[serial]
fn test_a_model_the_user_turned_on_counts_as_theirs() {
    let env = env("test_a_model_the_user_turned_on_counts_as_theirs");
    let rows = by_model_id(&env.usage());

    assert_eq!(rows["google/gemma-4-e4b-it"]["group"], "in_use");
}

#[test]
#[serial]
fn test_unused_is_the_same_bytes_as_reclaimable() {
    let env = env("test_unused_is_the_same_bytes_as_reclaimable");
    let usage = env.usage();
    let models = &usage["categories"]["models"];
    let rows = models["items"].as_array().unwrap();

    let unused: u64 = rows.iter().filter(|r| r["group"] == "unused").map(|r| r["bytes"].as_u64().unwrap()).sum();
    assert_eq!(unused, models["reclaimable"].as_u64().unwrap());
    let unused_ids: HashSet<String> = rows
        .iter()
        .filter(|r| r["group"] == "unused")
        .map(|r| r["model_id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(unused_ids, removable_ids_plus_unregistered());
}

#[test]
#[serial]
fn test_overhead_catches_what_has_no_row() {
    let env = env("test_overhead_catches_what_has_no_row");
    let usage = env.usage();
    let models = &usage["categories"]["models"];

    // The embedding GGUF lands as a loose file under embeddings/ rather than
    // in an <org>/<repo> tree, so it is inside the walked total and has no
    // item of its own.  Without overhead those bytes would go unexplained.
    assert_eq!(models["overhead"], 40);
}

#[test]
#[serial]
fn test_an_arriving_download_is_in_use_not_reclaimable() {
    let env = env("test_an_arriving_download_is_in_use_not_reclaimable");
    let live = make_partial_default(&env.models_dir, "google/gemma-3-4b-it");

    let _mark = downloading(&live);
    let rows = by_model_id(&env.usage());

    // Growing bytes are not space anyone can offer to free.
    assert_eq!(rows["google/gemma-3-4b-it"]["group"], "in_use");
}

#[test]
#[serial]
fn test_an_abandoned_download_is_reclaimable() {
    let env = env("test_an_abandoned_download_is_reclaimable");
    make_partial_default(&env.models_dir, "google/gemma-4-e4b-it");

    let rows = by_model_id(&env.usage());

    // Enabled, but the weights never arrived — the partial rule wins, and the
    // group follows it rather than the picker's flag.
    assert_eq!(rows["google/gemma-4-e4b-it"]["group"], "unused");
}

// ── TestOrphanProjectsSettleFirst ───────────────────────────────────────

#[test]
#[serial]
fn test_a_recently_touched_orphan_is_left_alone() {
    let env = env("test_a_recently_touched_orphan_is_left_alone");
    let rows = items(&env.usage(), "orphan_projects");

    let ids: Vec<&str> = rows.iter().map(|r| r["project_id"].as_str().unwrap()).collect();
    assert!(ids.contains(&env.orphan_id.as_str()));
    assert!(
        !ids.contains(&env.fresh_orphan_id.as_str()),
        "a directory written moments ago was offered for deletion"
    );
}

#[test]
#[serial]
fn test_the_sweep_leaves_a_fresh_orphan_on_disk() {
    let env = env("test_the_sweep_leaves_a_fresh_orphan_on_disk");
    env.cleanup(CleanupOptions { orphan_projects: true, ..opts() });

    assert!(isdir(&join(&env.workspace_dir, &env.fresh_orphan_id)));
}

#[test]
#[serial]
fn test_an_orphan_that_has_settled_is_swept() {
    let env = env("test_an_orphan_that_has_settled_is_swept");
    env.cleanup(CleanupOptions { orphan_projects: true, ..opts() });

    assert!(!exists(&join(&env.workspace_dir, &env.orphan_id)));
}

// ── TestASiblingWorkspaceIsNeverAProject ────────────────────────────────
//
// Workspaces sit side by side under home, one directory each.
//
// They are uuid-shaped and hold no projects of *this* workspace, so anything
// that roots the orphan scan at home rather than at the workspace directory
// offers another account's entire workspace for deletion.  An empty workspace
// id does exactly that: os.path.join(home, "") is "<home>/", and realpath
// normalises it back to home so every downstream check agrees.

fn collapsed(env: &Env) -> Workspace {
    let mut ws = env.workspace.write();
    ws.id = String::new();
    ws.workspace_dir = format!("{}{}", env.home_dir, std::path::MAIN_SEPARATOR);
    ws.clone()
}

#[test]
#[serial]
fn test_the_scan_refuses_a_collapsed_workspace_dir() {
    let env = env("test_the_scan_refuses_a_collapsed_workspace_dir");
    assert!(storage_service::scan_orphan_projects(&collapsed(&env)).is_empty());
}

#[test]
#[serial]
fn test_validation_refuses_it_too() {
    let env = env("test_validation_refuses_it_too");
    let ws = collapsed(&env);
    let sibling = join(&env.home_dir, &uuid::Uuid::new_v4().to_string());
    fs::create_dir_all(&sibling).unwrap();

    match storage_service::validate(&sibling, "orphan_projects", &ws) {
        Err(StorageError::ValueError(msg)) => assert!(msg.contains("home directory"), "{msg}"),
        other => panic!("expected ValueError, got {other:?}"),
    }
}

#[test]
#[serial]
fn test_a_sibling_workspace_survives_the_sweep() {
    let env = env("test_a_sibling_workspace_survives_the_sweep");
    let sibling = join(&env.home_dir, &uuid::Uuid::new_v4().to_string());
    fs::create_dir_all(join(&sibling, "models")).unwrap();
    age(&sibling);

    env.cleanup(CleanupOptions { orphan_projects: true, ..opts() });

    assert!(isdir(&sibling), "another workspace was swept");
}

// ── TestCleanup ─────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_nothing_happens_without_a_category() {
    let env = env("test_nothing_happens_without_a_category");
    let result = env.cleanup(opts());

    assert_eq!(result["freed"], 0);
    assert_eq!(result["deleted"], Value::Array(vec![]));
}

#[test]
#[serial]
fn test_dry_run_reports_without_deleting() {
    let env = env("test_dry_run_reports_without_deleting");
    let planned = env.cleanup(CleanupOptions { dry_run: true, ..everything() });
    let real = env.cleanup(everything());

    assert_eq!(planned["dry_run"], true);
    assert_eq!(planned["freed"], real["freed"]);
    let a: HashSet<String> = deleted_paths(&planned).into_iter().collect();
    let b: HashSet<String> = deleted_paths(&real).into_iter().collect();
    assert_eq!(a, b);
}

#[test]
#[serial]
fn test_model_sweep_keeps_what_is_in_use() {
    let env = env("test_model_sweep_keeps_what_is_in_use");
    env.cleanup(models_only());

    let models_dir = &env.models_dir;
    assert!(exists(&join(models_dir, "smartloop-ai/slp-orchestra-mini")));
    assert!(exists(&join(models_dir, "google/gemma-4-e4b-it")));
    assert!(exists(&join(models_dir, "embeddings")));
    assert!(!exists(&join(models_dir, "google/gemma-3-4b-it")));
    assert!(!exists(&join(models_dir, "mistralai")));
}

#[test]
#[serial]
fn test_model_sweep_keeps_a_project_enabled_model() {
    let env = env("test_model_sweep_keeps_a_project_enabled_model");
    env.add_project_model(project_model("gemma3-4b", true, false));

    let result = env.cleanup(models_only());

    assert!(exists(&join(&env.models_dir, "google/gemma-3-4b-it")));
    assert!(!models_marked(&result).contains(&"gemma3-4b".to_string()));
}

#[test]
#[serial]
fn test_model_sweep_prunes_the_emptied_org_dir() {
    let env = env("test_model_sweep_prunes_the_emptied_org_dir");
    env.cleanup(models_only());

    // mistralai held only the deleted model; google still has a kept one.
    assert!(!exists(&join(&env.models_dir, "mistralai")));
    assert!(isdir(&join(&env.models_dir, "google")));
}

#[test]
#[serial]
fn test_model_sweep_updates_workspace_flags_in_memory_and_on_disk() {
    let env = env("test_model_sweep_updates_workspace_flags_in_memory_and_on_disk");
    let result = env.cleanup(models_only());

    let marked: HashSet<String> = models_marked(&result).into_iter().collect();
    assert_eq!(marked, ["gemma3-4b", "ministral3-3b"].iter().map(|s| s.to_string()).collect());
    for m in env.workspace.read().models.iter().filter(|m| REMOVABLE_IDS.contains(&m.model_id.as_str())) {
        assert_eq!((m.downloaded, m.enabled, m.is_active), (false, false, false));
    }

    let on_disk: Value = serde_json::from_str(&fs::read_to_string(&env.config_file).unwrap()).unwrap();
    let flags: HashMap<String, Value> = on_disk["workspace"]["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| (m["model_name"].as_str().unwrap().to_string(), m.clone()))
        .collect();
    assert_eq!(flags["gemma3-4b"]["downloaded"], false);
    assert_eq!(flags["gemma4-e4b"]["downloaded"], true);
}

#[test]
#[serial]
fn test_model_sweep_unloads_each_deleted_model_only() {
    let env = env("test_model_sweep_unloads_each_deleted_model_only");
    env.cleanup(models_only());

    let calls: HashSet<String> = env.service.unloaded.lock().iter().cloned().collect();
    assert_eq!(calls, ["gemma3-4b", "ministral3-3b"].iter().map(|s| s.to_string()).collect());
    // unload_model() would evict the always-resident orchestrator.
    assert_eq!(env.service.public_unload_calls.load(Ordering::SeqCst), 0);
    assert_eq!(env.service.reset_calls.load(Ordering::SeqCst), 1);
}

#[test]
#[serial]
fn test_model_sweep_never_takes_the_public_unload() {
    // unload_model_by_name() acquires an inference slot from the very
    // semaphore the sweep has just drained, and the semaphore is not reentrant
    // — calling it here deadlocks the request forever while holding every
    // permit, which wedges inference too.
    let env = env("test_model_sweep_never_takes_the_public_unload");
    env.cleanup(models_only());

    assert_eq!(env.service.public_unload_calls.load(Ordering::SeqCst), 0);
}

#[test]
#[serial]
fn test_model_sweep_unloads_without_a_free_permit() {
    // The invariant behind the unlocked unload, against a real semaphore: by
    // the time unload runs, no permit is left to take.  Sized as ModelService
    // sizes it, from the same n_parallel the sweep reads when deciding how
    // many permits to drain.
    let env = env("test_model_sweep_unloads_without_a_free_permit");
    let permits = env.settings.n_parallel.max(1) as usize;
    let lock = Arc::new(Semaphore::new(permits));
    *env.service.lock.lock() = lock.clone();
    let acquired: Arc<Mutex<Vec<bool>>> = Arc::new(Mutex::new(Vec::new()));

    let probe = lock.clone();
    let seen = acquired.clone();
    *env.service.unload_effect.lock() = Some(Box::new(move |_name: &str| {
        seen.lock().push(probe.acquire(Duration::from_millis(250)));
    }));

    env.cleanup(models_only());

    let acquired = acquired.lock().clone();
    assert!(!acquired.is_empty() && !acquired.iter().any(|a| *a), "the sweep must hold every permit");
    assert!(!exists(&join(&env.models_dir, "mistralai/Ministral-3-3B-Instruct-2512")));
    // Every permit handed back, so inference is not wedged afterwards.
    assert!((0..permits).all(|_| lock.acquire(Duration::from_millis(250))));
}

#[test]
#[serial]
fn test_model_ids_restricts_the_sweep() {
    let env = env("test_model_ids_restricts_the_sweep");
    env.cleanup(CleanupOptions { model_ids: Some(vec!["google/gemma-3-4b-it".to_string()]), ..models_only() });

    assert!(!exists(&join(&env.models_dir, "google/gemma-3-4b-it")));
    assert!(exists(&join(&env.models_dir, "mistralai")));
}

fn assert_model_id_rejected(test_name: &str, model_id: &str) {
    let env = env(test_name);
    let err = storage_service::run_cleanup(
        env.svc(),
        CleanupOptions { model_ids: Some(vec![model_id.to_string()]), ..models_only() },
    )
    .unwrap_err();
    assert!(matches!(err, StorageError::ValueError(_)), "{model_id}: {err:?}");
}

#[test]
#[serial]
fn test_model_ids_rejects_anything_not_removable_in_use() {
    assert_model_id_rejected("test_model_ids_rejects_anything_not_removable_in_use", "google/gemma-4-e4b-it");
}

#[test]
#[serial]
fn test_model_ids_rejects_anything_not_removable_bundled() {
    assert_model_id_rejected("test_model_ids_rejects_anything_not_removable_bundled", "smartloop-ai/slp-orchestra-mini");
}

#[test]
#[serial]
fn test_model_ids_rejects_anything_not_removable_traversal() {
    assert_model_id_rejected("test_model_ids_rejects_anything_not_removable_traversal", "../../../etc");
}

#[test]
#[serial]
fn test_model_ids_rejects_anything_not_removable_unknown() {
    assert_model_id_rejected("test_model_ids_rejects_anything_not_removable_unknown", "no-such/model");
}

#[test]
#[serial]
fn test_log_cleanup_empties_the_log_and_deletes_backups() {
    let env = env("test_log_cleanup_empties_the_log_and_deletes_backups");
    let log = join(&env.home_dir, "server.log");

    let result = env.cleanup(CleanupOptions { logs: true, ..opts() });

    assert!(isfile(&log), "server.log is emptied, never unlinked");
    assert_eq!(fs::metadata(&log).unwrap().len(), 0);
    assert!(!exists(&format!("{log}.1")));
    assert!(!exists(&format!("{log}.2")));
    assert_eq!(result["freed"], 300 + 500 + 700);
}

#[test]
#[serial]
fn test_log_sweep_replays_its_record_after_emptying_the_log() {
    // Emptying server.log must not erase the record of the same sweep.
    //
    // Deletions are logged as they happen, so a sweep that also truncates the
    // log destroys its own audit trail — the one thing a "where did my model
    // go?" report needs.  The record is therefore written again once the log
    // is empty.
    let env = env("test_log_sweep_replays_its_record_after_emptying_the_log");
    let gemma = join(&join(&env.models_dir, "google"), "gemma-3-4b-it");

    let (_, messages) = caplog::at_info(storage_service::LOG_TARGET, || {
        env.cleanup(CleanupOptions { models: true, logs: true, ..opts() })
    });

    let replay_at = messages.iter().position(|m| m.contains("replayed after server.log"));
    let replay_at = replay_at.expect("no record was replayed after the truncate");

    let survives = &messages[replay_at..];
    assert!(
        survives.iter().any(|m| m.contains(&gemma) && m.contains("Removed")),
        "the model deletion is missing from the record that outlives the truncate"
    );
    assert!(survives.iter().any(|m| m.contains("server.log") && m.contains("Removed")));
    assert!(survives.iter().any(|m| m.contains("gemma3-4b")), "models_marked is part of the record too");
}

#[test]
#[serial]
fn test_sweep_without_logs_does_not_replay() {
    // Nothing truncated the log, so the first record still stands.
    let env = env("test_sweep_without_logs_does_not_replay");
    let (_, messages) = caplog::at_info(storage_service::LOG_TARGET, || env.cleanup(models_only()));

    assert!(!messages.iter().any(|m| m.contains("replayed")));
    assert_eq!(messages.iter().filter(|m| m.contains("Removed") && m.contains("gemma-3-4b-it")).count(), 1);
}

#[test]
#[serial]
fn test_version_cleanup_keeps_the_running_install() {
    let env = env("test_version_cleanup_keeps_the_running_install");
    env.cleanup(CleanupOptions { versions: true, ..opts() });

    assert!(isdir(&join(&env.home_dir, CURRENT_VERSION)));
    assert!(!exists(&join(&env.home_dir, OLD_VERSION)));
}

#[test]
#[serial]
fn test_cache_cleanup_empties_contents_but_keeps_the_dirs() {
    let env = env("test_cache_cleanup_empties_contents_but_keeps_the_dirs");
    env.cleanup(CleanupOptions { cache: true, ..opts() });

    for name in ["cache", "templates"] {
        let path = join(&env.home_dir, name);
        assert!(isdir(&path));
        assert_eq!(fs::read_dir(&path).unwrap().count(), 0);
    }
}

#[test]
#[serial]
fn test_orphan_project_cleanup_leaves_live_project_and_odd_dirs() {
    let env = env("test_orphan_project_cleanup_leaves_live_project_and_odd_dirs");
    env.cleanup(CleanupOptions { orphan_projects: true, ..opts() });

    assert!(!exists(&join(&env.workspace_dir, &env.orphan_id)));
    assert!(isdir(&env.project_dir));
    assert!(isdir(&join(&env.workspace_dir, "not-a-project")));
    assert!(isdir(&join(&env.workspace_dir, "models")));
}

#[test]
#[serial]
fn test_conversations_are_never_deleted() {
    let env = env("test_conversations_are_never_deleted");
    env.cleanup(everything());

    let conversations = join(&env.home_dir, "conversations");
    let mut found: Vec<String> =
        fs::read_dir(&conversations).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    found.sort();
    let mut expected = vec![format!("{}.jsonl", env.session_id), format!("{}.jsonl", env.untracked_id)];
    expected.sort();
    assert_eq!(found, expected);
}

#[test]
#[serial]
fn test_protected_files_survive_every_category() {
    let env = env("test_protected_files_survive_every_category");
    env.cleanup(everything());

    for name in PROTECTED_FILES {
        assert!(isfile(&join(&env.home_dir, name)), "{name}");
    }
    for name in ["workspace.db", "workspace.db-wal", "workspace.db-shm"] {
        assert!(isfile(&join(&env.workspace_dir, name)), "{name}");
    }
    assert!(isfile(&join(&env.outside, "precious.txt")));
}

#[test]
#[serial]
fn test_failure_is_reported_and_asks_for_a_restart() {
    let env = env("test_failure_is_reported_and_asks_for_a_restart");
    storage_service::set_rmtree_hook(Some(Arc::new(|_path: &str| {
        Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "still mapped"))
    })));

    let result = env.cleanup(models_only());

    assert_eq!(result["deleted"], Value::Array(vec![]));
    assert_eq!(result["restart_required"], true);
    let errors: HashSet<String> =
        result["failed"].as_array().unwrap().iter().map(|f| f["error"].as_str().unwrap().to_string()).collect();
    assert_eq!(errors, ["still mapped".to_string()].into_iter().collect());
    // Nothing was marked removed, because nothing was actually removed.
    assert_eq!(result["models_marked"], Value::Array(vec![]));
    assert!(exists(&join(&env.models_dir, "google/gemma-3-4b-it")));
}

#[test]
#[serial]
fn test_refuses_while_the_indexer_is_running() {
    let env = env("test_refuses_while_the_indexer_is_running");
    *env.service.task_status.lock() = Box::new(|_: &str| Some(serde_json::json!({"status": "running"})));

    match storage_service::run_cleanup(env.svc(), CleanupOptions { orphan_projects: true, ..opts() }) {
        Err(StorageError::RuntimeError(msg)) => assert!(msg.contains("indexer"), "{msg}"),
        other => panic!("expected RuntimeError, got {other:?}"),
    }
}

#[test]
#[serial]
fn test_a_running_indexer_does_not_block_a_model_sweep() {
    // The indexer writes project trees, not model weights.
    let env = env("test_a_running_indexer_does_not_block_a_model_sweep");
    *env.service.task_status.lock() = Box::new(|_: &str| Some(serde_json::json!({"status": "running"})));

    env.cleanup(models_only());
}

#[test]
#[serial]
fn test_refuses_when_inference_will_not_drain() {
    let env = env("test_refuses_when_inference_will_not_drain");
    env.service.mock_permits.result.store(false, Ordering::SeqCst);

    match storage_service::run_cleanup(env.svc(), models_only()) {
        Err(StorageError::RuntimeError(msg)) => assert!(msg.contains("busy"), "{msg}"),
        other => panic!("expected RuntimeError, got {other:?}"),
    }
}

#[test]
#[serial]
fn test_log_only_cleanup_does_not_wait_on_inference() {
    let env = env("test_log_only_cleanup_does_not_wait_on_inference");
    env.cleanup(CleanupOptions { logs: true, ..opts() });

    assert_eq!(env.service.mock_permits.acquire_calls.load(Ordering::SeqCst), 0);
}

#[test]
#[serial]
fn test_report_includes_fresh_usage() {
    let env = env("test_report_includes_fresh_usage");
    let result = env.cleanup(CleanupOptions { logs: true, ..opts() });

    assert_eq!(result["usage"]["categories"]["logs"]["reclaimable"], 0);
}

// ── TestAbandonedDownloads ──────────────────────────────────────────────
//
// A directory holding only a `.part` is a download that died.
//
// The keep rules all answer "is this model in use?", and none of them can be
// true of bytes that never arrived — `enabled` in particular is set when the
// user asks for a model, not when it lands.  Protecting such a tree strands
// the space on a model nothing can load, and on a full disk that is a
// deadlock: the only remedy the UI offers is the download this space blocks.
//
// The bundled orgs are the exception, for the reason the deadlock argument
// turns on: there the retry is not a thing the user might choose, so the space
// comes back on its own.  See `test_a_bundled_partial_is_kept_for_its_retry`.

#[test]
#[serial]
fn test_abandoned_partial_is_removable_though_enabled() {
    let env = env("test_abandoned_partial_is_removable_though_enabled");
    make_partial_default(&env.models_dir, "google/gemma-4-e4b-it");

    let rows = by_model_id(&env.usage());

    let dead = &rows["google/gemma-4-e4b-it"];
    assert_eq!(dead["partial"], true);
    assert_eq!(dead["removable"], true);
    assert_eq!(dead["keep_reason"], Value::Null);
}

#[test]
#[serial]
fn test_abandoned_partial_beats_a_project_pin() {
    // The strongest keep reason there is: off app-wide but pinned on by a
    // project.  Still nothing to protect.
    let env = env("test_abandoned_partial_beats_a_project_pin");
    env.add_project_model(project_model("gemma3-4b", true, false));
    make_partial_default(&env.models_dir, "google/gemma-3-4b-it");

    let rows = by_model_id(&env.usage());

    assert_eq!(rows["google/gemma-3-4b-it"]["removable"], true);
}

#[test]
#[serial]
fn test_a_bundled_partial_is_kept_for_its_retry() {
    // The inverse of every other case in this group.  A half-downloaded
    // orchestrator cannot run the framework — which is exactly why it will be
    // downloaded again, and why the retry (which unlinks the stale .part
    // itself) reclaims the space without the sweep.  Deleting it buys nothing
    // and risks racing a writer, so bundled wins.
    let env = env("test_a_bundled_partial_is_kept_for_its_retry");
    make_partial_default(&env.models_dir, "smartloop-ai/slp-orchestra-mini");

    let rows = by_model_id(&env.usage());

    let held = &rows["smartloop-ai/slp-orchestra-mini"];
    assert_eq!(held["removable"], false);
    assert_eq!(held["keep_reason"], "bundled");
    // Still an honest description of the directory; it just isn't offered.
    assert_eq!(held["partial"], true);
}

#[test]
#[serial]
fn test_a_bundled_partial_survives_the_sweep() {
    let env = env("test_a_bundled_partial_survives_the_sweep");
    let held = make_partial_default(&env.models_dir, "smartloop-ai/slp-orchestra-mini");
    let removable = make_partial_default(&env.models_dir, "google/gemma-4-e4b-it");

    let result = env.cleanup(models_only());

    assert!(exists(&held), "the sweep reclaimed bundled weights");
    assert!(!deleted_paths(&result).contains(&held));
    assert!(!exists(&removable), "protecting the bundled tree stopped the rest of the sweep");
}

#[test]
#[serial]
fn test_a_live_embedding_fetch_is_never_offered() {
    // Embedding weights land as loose files directly in `models/embeddings/` —
    // the embedding fetcher writes the GGUF and its `.part` there rather than
    // under an `<org>/<repo>` pair, and it takes no `downloading` mark.
    //
    // Two things keep them: the scan enumerates directories, so loose files
    // never become rows at all, and the org is bundled so anything that *is*
    // a directory under it is held.  The first is an accident of scan depth;
    // this test is what fails loudly if that ever changes.
    let env = env("test_a_live_embedding_fetch_is_never_offered");
    let embeddings = join(&env.models_dir, "embeddings");
    fs::create_dir_all(&embeddings).unwrap();
    write_s(&join(&embeddings, "bge-m3-Q4_K_M.gguf.part"), &"p".repeat(100));

    let usage = env.usage();
    let offered: Vec<String> = items(&usage, "models").iter().map(|r| r["path"].as_str().unwrap().to_string()).collect();
    assert!(!offered.iter().any(|p| p.starts_with(&embeddings)));

    env.cleanup(models_only());

    assert!(
        isfile(&join(&embeddings, "bge-m3-Q4_K_M.gguf.part")),
        "the sweep deleted an in-flight embedding download"
    );
}

#[test]
#[serial]
fn test_weights_that_landed_survive_a_partial_companion() {
    // The interrupted-multi-file case: main weights finished, the vision
    // projector did not.  The downloader checksums the finished file and
    // refetches only the rest, so deleting the tree would throw away real
    // progress — this is the one .part case that must be kept.
    let env = env("test_weights_that_landed_survive_a_partial_companion");
    make_partial(&env.models_dir, "google/gemma-4-e4b-it", &["weights.gguf"], &["mmproj-gemma-4.gguf.part"]);

    let rows = by_model_id(&env.usage());

    let kept = &rows["google/gemma-4-e4b-it"];
    assert_eq!(kept["partial"], false);
    assert_eq!(kept["removable"], false);
    assert_eq!(kept["keep_reason"], "enabled");
}

#[test]
#[serial]
fn test_a_finished_mmproj_alone_is_not_weights() {
    // The mirror image: the projector landed, the weights are still
    // transferring.  A projector is not something that can be served, so this
    // tree is as dead as one holding only .part files.
    let env = env("test_a_finished_mmproj_alone_is_not_weights");
    make_partial(&env.models_dir, "google/gemma-4-e4b-it", &["mmproj-gemma-4.gguf"], &["weights.gguf.part"]);

    let rows = by_model_id(&env.usage());

    assert_eq!(rows["google/gemma-4-e4b-it"]["partial"], true);
    assert_eq!(rows["google/gemma-4-e4b-it"]["removable"], true);
}

#[test]
#[serial]
fn test_a_resident_model_is_kept_even_when_partial() {
    // Shouldn't be reachable — weights resolve before a load — but a resident
    // context is the one state where unlinking is a native use-after-free, so
    // it outranks the partial rule.
    let env = env("test_a_resident_model_is_kept_even_when_partial");
    env.service.resident.lock().insert("gemma3-4b".to_string());
    make_partial_default(&env.models_dir, "google/gemma-3-4b-it");

    let rows = by_model_id(&env.usage());

    assert_eq!(rows["google/gemma-3-4b-it"]["removable"], false);
    assert_eq!(rows["google/gemma-3-4b-it"]["keep_reason"], "loaded");
}

#[test]
#[serial]
fn test_complete_models_are_not_flagged_partial() {
    let env = env("test_complete_models_are_not_flagged_partial");
    let rows = by_model_id(&env.usage());

    assert!(rows.values().all(|r| r["partial"] == false));
}

#[test]
#[serial]
fn test_an_in_flight_download_is_kept_while_the_rest_is_swept() {
    // The race the partial rule opens: those .part files belong to a transfer
    // happening right now, and on disk that is indistinguishable from one that
    // died.  The registry is what tells them apart.
    //
    // Both halves matter.  The live tree survives — and the sweep still runs,
    // reclaiming everything else, because protecting a download must not cost
    // the user the cleanup they asked for.  That is the difference between
    // this and the blanket refusal it replaced.
    let env = env("test_an_in_flight_download_is_kept_while_the_rest_is_swept");
    let live = make_partial_default(&env.models_dir, "google/gemma-4-e4b-it");
    let removable = join(&env.models_dir, "google/gemma-3-4b-it");

    let result = {
        let _mark = downloading(&live);
        env.cleanup(models_only())
    };

    assert!(exists(&live), "the sweep deleted a running download");
    assert!(!deleted_paths(&result).contains(&live));
    assert!(!exists(&removable), "protecting the download stopped the rest of the sweep");
}

#[test]
#[serial]
fn test_a_live_download_reports_why_it_was_kept() {
    let env = env("test_a_live_download_reports_why_it_was_kept");
    let live = make_partial_default(&env.models_dir, "google/gemma-4-e4b-it");

    {
        let _mark = downloading(&live);
        let rows = by_model_id(&env.usage());
        let held = &rows["google/gemma-4-e4b-it"];

        assert_eq!(held["keep_reason"], "downloading");
        assert_eq!(held["removable"], false);
        // Still accurate: it really is a partial tree, it just is not dead.
        assert_eq!(held["partial"], true);
    }

    let after = by_model_id(&env.usage());
    assert_eq!(after["google/gemma-4-e4b-it"]["removable"], true, "the mark outlived the download");
}

#[test]
#[serial]
fn test_a_download_starting_mid_sweep_is_not_deleted() {
    // The scan is a snapshot, so run_cleanup asks again before each removal.
    // Here the download starts after planning, which is the window the
    // re-check exists to cover.
    let env = env("test_a_download_starting_mid_sweep_is_not_deleted");
    let live = make_partial_default(&env.models_dir, "google/gemma-4-e4b-it");
    let started = Arc::new(AtomicBool::new(false));

    let flag = started.clone();
    storage_service::set_remove_hook(Some(Arc::new(move |path: &str, truncate: bool| {
        flag.store(true, Ordering::SeqCst);
        storage_service::remove(path, truncate)
    })));
    let flag = started.clone();
    let live_path = live.clone();
    storage_service::set_is_downloading_hook(Some(Arc::new(move |p: &str| {
        flag.load(Ordering::SeqCst) && p == live_path
    })));

    let result = env.cleanup(models_only());

    assert!(exists(&live));
    assert!(!deleted_paths(&result).contains(&live));
    let failed: Vec<String> =
        result["failed"].as_array().unwrap().iter().map(|f| f["path"].as_str().unwrap().to_string()).collect();
    assert!(!failed.contains(&live));
    assert_eq!(result["restart_required"], false);
    assert!(!models_marked(&result).contains(&"gemma4-e4b".to_string()), "a skipped model was recorded as removed");
}

#[test]
#[serial]
fn test_the_sweep_reclaims_an_abandoned_partial() {
    let env = env("test_the_sweep_reclaims_an_abandoned_partial");
    let dead = make_partial_default(&env.models_dir, "google/gemma-4-e4b-it");

    let result = env.cleanup(models_only());

    assert!(!exists(&dead));
    assert!(deleted_paths(&result).contains(&dead));
}

#[test]
#[serial]
fn test_partial_bytes_are_reported_as_reclaimable() {
    let env = env("test_partial_bytes_are_reported_as_reclaimable");
    make_partial_default(&env.models_dir, "google/gemma-4-e4b-it");

    let usage = env.usage();

    assert!(usage["categories"]["models"]["reclaimable"].as_u64().unwrap() >= 100);
}

// ── TestValidateRefusesBadPaths ─────────────────────────────────────────
//
// The traversal and symlink rails, exercised directly.
//
// Every other test in this file feeds `validate` paths its own scanners
// produced, so the refusals never fire and the rails are asserted only
// indirectly — "the protected files survived".  That is the wrong shape of
// evidence for the guard standing between a POST body and a recursive delete.
// Note in particular that the `"../../../etc"` model_ids case does not reach
// here at all: planning rejects it first as an unknown model id.

/// `pytest.raises(ValueError, match=...)`.
fn assert_value_error(result: Result<String, StorageError>, needle: &str) {
    match result {
        Err(StorageError::ValueError(msg)) => assert!(msg.contains(needle), "{msg:?} does not match {needle:?}"),
        other => panic!("expected ValueError matching {needle:?}, got {other:?}"),
    }
}

#[test]
#[serial]
fn test_home_itself_is_refused() {
    let env = env("test_home_itself_is_refused");
    let ws = env.workspace.read().clone();
    assert_value_error(storage_service::validate(&env.home_dir, "models", &ws), "home directory itself");
}

#[test]
#[serial]
fn test_a_path_outside_home_is_refused() {
    let env = env("test_a_path_outside_home_is_refused");
    let ws = env.workspace.read().clone();
    let escape = join(&env.outside, "precious.txt");
    assert_value_error(storage_service::validate(&escape, "cache", &ws), "escapes");
}

#[test]
#[serial]
fn test_traversal_out_of_the_models_dir_is_refused() {
    let env = env("test_traversal_out_of_the_models_dir_is_refused");
    let ws = env.workspace.read().clone();
    let escape = format!("{0}{1}..{1}..{1}..{1}outside", env.models_dir, std::path::MAIN_SEPARATOR);
    assert_value_error(storage_service::validate(&escape, "models", &ws), "escapes");
}

#[test]
#[serial]
fn test_a_never_delete_basename_is_refused() {
    let env = env("test_a_never_delete_basename_is_refused");
    let ws = env.workspace.read().clone();
    for name in PROTECTED_FILES {
        let target = join(&env.home_dir, name);
        assert_value_error(storage_service::validate(&target, "cache", &ws), "Protected path");
    }
}

#[test]
#[serial]
fn test_a_symlink_is_refused() {
    let env = env("test_a_symlink_is_refused");
    let ws = env.workspace.read().clone();
    let link = join(&join(&env.models_dir, "google"), "link");
    assert!(fs::symlink_metadata(&link).unwrap().file_type().is_symlink(), "fixture no longer builds the symlink");
    assert_value_error(storage_service::validate(&link, "models", &ws), "symlink");
}

#[test]
#[serial]
fn test_a_model_path_at_the_wrong_depth_is_refused() {
    // models/<org> is one level short of a model directory.
    let env = env("test_a_model_path_at_the_wrong_depth_is_refused");
    let ws = env.workspace.read().clone();
    let org_only = join(&env.models_dir, "google");
    assert_value_error(storage_service::validate(&org_only, "models", &ws), "models/<org>/<repo>");
}

#[test]
#[serial]
fn test_the_running_install_is_refused() {
    let env = env("test_the_running_install_is_refused");
    let ws = env.workspace.read().clone();
    let target = join(&env.home_dir, CURRENT_VERSION);
    assert_value_error(storage_service::validate(&target, "versions", &ws), "running install");
}

#[test]
#[serial]
fn test_a_non_version_directory_is_refused() {
    let env = env("test_a_non_version_directory_is_refused");
    let ws = env.workspace.read().clone();
    let target = join(&env.home_dir, "cache");
    assert_value_error(storage_service::validate(&target, "versions", &ws), "Not a version directory");
}

#[test]
#[serial]
fn test_a_backup_index_outside_the_kept_range_is_refused() {
    // DEFAULT_BACKUP_COUNT is 5, and the bound is compared numerically so
    // that 10+ does not silently become "any digit".
    for name in ["server.log.0", "server.log.6", "server.log.x"] {
        let env = env(&format!("test_a_backup_index_outside_the_kept_range_is_refused_{name}"));
        let ws = env.workspace.read().clone();
        let target = join(&env.home_dir, name);
        write_s(&target, "l");
        assert_value_error(storage_service::validate(&target, "logs", &ws), "Not a server log file");
    }
}

#[test]
#[serial]
fn test_a_kept_backup_index_is_accepted() {
    let env = env("test_a_kept_backup_index_is_accepted");
    let ws = env.workspace.read().clone();
    let target = join(&env.home_dir, "server.log.1");
    assert_eq!(storage_service::validate(&target, "logs", &ws).unwrap(), realpath(&target));
}

#[test]
#[serial]
fn test_a_path_outside_the_cache_dirs_is_refused() {
    let env = env("test_a_path_outside_the_cache_dirs_is_refused");
    let ws = env.workspace.read().clone();
    let target = join(&env.home_dir, "conversations");
    assert_value_error(storage_service::validate(&target, "cache", &ws), "Not a cache entry");
}

#[test]
#[serial]
fn test_a_live_project_is_refused() {
    let env = env("test_a_live_project_is_refused");
    let ws = env.workspace.read().clone();
    assert_value_error(storage_service::validate(&env.project_dir, "orphan_projects", &ws), "live project");
}

#[test]
#[serial]
fn test_a_non_uuid_directory_is_not_a_project() {
    let env = env("test_a_non_uuid_directory_is_not_a_project");
    let ws = env.workspace.read().clone();
    let target = join(&env.workspace_dir, "not-a-project");
    let result = storage_service::validate(&target, "orphan_projects", &ws);
    assert!(matches!(result, Err(StorageError::ValueError(_))), "{result:?}");
}

#[test]
#[serial]
fn test_an_unknown_category_is_refused() {
    let env = env("test_an_unknown_category_is_refused");
    let ws = env.workspace.read().clone();
    let target = join(&join(&env.home_dir, "cache"), "slp-1.1.0.tar.gz");
    assert_value_error(storage_service::validate(&target, "conversations", &ws), "Unknown category");
}

// ── TestDiscardingACancelledDownload ────────────────────────────────────
//
// What a stopped download leaves behind, and what removes it.
//
// Reached only from the download thread's own unwinding, so every case here
// assumes the transfer has already stopped.

fn discard(env: &Env, model_id: &str) -> Result<Row, StorageError> {
    storage_service::discard_partial_download(env.svc(), model_id)
}

#[test]
#[serial]
fn test_it_removes_the_tree_and_clears_the_flags() {
    let env = env("test_it_removes_the_tree_and_clears_the_flags");
    let model_id = "google/gemma-3-4b-it";
    let path = make_partial_default(&env.models_dir, model_id);

    let report = discard(&env, model_id).unwrap();

    assert_eq!(report["removed"], true);
    assert!(report["freed"].as_u64().unwrap() > 0);
    assert!(!exists(&path));
    let ws = env.workspace.read();
    let entry = ws.models.iter().find(|m| m.model_id == model_id).unwrap();
    assert!(!entry.downloaded && !entry.enabled && !entry.is_active);
    assert_eq!(report["models_marked"], serde_json::json!([entry.model_name]));
}

#[test]
#[serial]
fn test_it_removes_a_vision_model_that_only_half_arrived() {
    // Weights without their projector is the state this exists to prevent.
    //
    // `_gguf_exists` reads such a directory as "already available", so keeping
    // it would leave the model permanently unloadable while the picker showed
    // it ready.
    let env = env("test_it_removes_a_vision_model_that_only_half_arrived");
    let model_id = "google/gemma-3-4b-it";
    let path = make_partial(&env.models_dir, model_id, &["weights.gguf"], &["mmproj-model.gguf.part"]);

    let report = discard(&env, model_id).unwrap();

    assert_eq!(report["removed"], true);
    assert!(!exists(&path));
}

#[test]
#[serial]
fn test_it_takes_both_part_files_of_a_vision_model() {
    let env = env("test_it_takes_both_part_files_of_a_vision_model");
    let model_id = "google/gemma-3-4b-it";
    let path = make_partial(&env.models_dir, model_id, &[], &["weights.gguf.part", "mmproj-model.gguf.part"]);

    discard(&env, model_id).unwrap();

    assert!(!exists(&path));
}

#[test]
#[serial]
fn test_it_prunes_the_org_directory_it_empties() {
    let env = env("test_it_prunes_the_org_directory_it_empties");
    let model_id = "mistralai/Ministral-3-3B-Instruct-2512";
    make_partial_default(&env.models_dir, model_id);

    discard(&env, model_id).unwrap();

    assert!(!exists(&join(&env.models_dir, "mistralai")));
}

#[test]
#[serial]
fn test_it_leaves_an_org_that_still_holds_a_sibling() {
    let env = env("test_it_leaves_an_org_that_still_holds_a_sibling");
    make_partial_default(&env.models_dir, "google/gemma-3-4b-it");

    discard(&env, "google/gemma-3-4b-it").unwrap();

    // gemma-4-e4b-it is still there, so google/ stays.
    assert!(isdir(&join(&join(&env.models_dir, "google"), "gemma-4-e4b-it")));
}

#[test]
#[serial]
fn test_a_directory_that_never_existed_is_not_an_error() {
    // Cancelled before a byte landed.  The caller asked for the bytes to be
    // absent, and they are.
    let env = env("test_a_directory_that_never_existed_is_not_an_error");
    let report = discard(&env, "google/never-started").unwrap();

    assert_eq!(report["removed"], false);
    assert_eq!(report["error"], Value::Null);
    assert_eq!(report["freed"], 0);
}

#[test]
#[serial]
fn test_it_leaves_every_other_model_alone() {
    let env = env("test_it_leaves_every_other_model_alone");
    make_partial_default(&env.models_dir, "google/gemma-3-4b-it");

    discard(&env, "google/gemma-3-4b-it").unwrap();

    for (_name, model_id, _e, _a) in MODELS {
        if *model_id != "google/gemma-3-4b-it" {
            assert!(isdir(&join(&env.models_dir, model_id)), "{model_id}");
        }
    }
}

#[test]
#[serial]
fn test_it_refuses_a_model_id_that_escapes_the_models_dir() {
    let env = env("test_it_refuses_a_model_id_that_escapes_the_models_dir");
    let result = discard(&env, "../../..");
    assert!(matches!(result, Err(StorageError::ValueError(_))), "{result:?}");
}

#[test]
#[serial]
fn test_it_refuses_a_shape_that_is_not_org_slash_repo() {
    let env = env("test_it_refuses_a_shape_that_is_not_org_slash_repo");
    match discard(&env, "google") {
        Err(StorageError::ValueError(msg)) => assert!(msg.contains("models/<org>/<repo>"), "{msg}"),
        other => panic!("expected ValueError, got {other:?}"),
    }
}

#[test]
#[serial]
fn test_it_refuses_to_follow_a_symlink_out_of_home() {
    // The fixture links models/google/link at a directory outside home.
    //
    // The containment rail is the one that fires, not the symlink rail:
    // `validate` resolves the path first, so a link pointing out is refused as
    // an escape before anyone asks what kind of file it is.  Either way the
    // user's own files are untouched, which is the property that matters.
    let env = env("test_it_refuses_to_follow_a_symlink_out_of_home");
    match discard(&env, "google/link") {
        Err(StorageError::ValueError(msg)) => assert!(msg.contains("escapes"), "{msg}"),
        other => panic!("expected ValueError, got {other:?}"),
    }
    assert!(exists(&join(&env.outside, "precious.txt")));
}

#[test]
#[serial]
fn test_it_does_not_drain_inference() {
    // A button press must not wait 30s on permits, and nothing can have
    // weights mapped that never finished downloading.
    let env = env("test_it_does_not_drain_inference");
    make_partial_default(&env.models_dir, "google/gemma-3-4b-it");

    discard(&env, "google/gemma-3-4b-it").unwrap();

    assert_eq!(env.service.mock_permits.acquire_calls.load(Ordering::SeqCst), 0);
}

#[test]
#[serial]
fn test_it_does_not_unload_or_reset_the_executor() {
    let env = env("test_it_does_not_unload_or_reset_the_executor");
    make_partial_default(&env.models_dir, "google/gemma-3-4b-it");

    discard(&env, "google/gemma-3-4b-it").unwrap();

    assert!(env.service.unloaded.lock().is_empty());
    assert_eq!(env.service.reset_calls.load(Ordering::SeqCst), 0);
}

#[test]
#[serial]
fn test_it_does_not_consult_the_indexer_guard() {
    // `run_cleanup` refuses while the indexer runs; this must not.
    let env = env("test_it_does_not_consult_the_indexer_guard");
    make_partial_default(&env.models_dir, "google/gemma-3-4b-it");
    *env.service.task_status.lock() = Box::new(|name: &str| {
        if name == "indexer" {
            Some(serde_json::json!({"status": "running"}))
        } else {
            Some(serde_json::json!({"error": "unknown"}))
        }
    });

    let report = discard(&env, "google/gemma-3-4b-it").unwrap();

    assert_eq!(report["removed"], true);
}
