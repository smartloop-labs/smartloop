//! Port of `tests/unit/test_indexer_is_project_work.py`.
//!
//! Indexing a project's documents belongs to that project's own agent.
//!
//! `ModelService::setup_tasks` registers an `IndexerTask`.  Given no project of
//! its own it falls back to the `current` flag, and its document scan reads the
//! documents folder off disk rather than draining its own queue — so in a
//! supervisor that has delegated every project to a child it picks up whatever
//! a child just wrote and indexes it a second time.
//!
//! Nothing is lost by skipping it there: a delegating `save()` keeps only the
//! workspace record, so the document state the task exists to persist is
//! discarded anyway while its manifest and vector-store writes land.
//!
//! The Python tests stood mocks in for the task classes.  Here the real tasks
//! run, and the test observes what the mocks recorded: which tasks the queue
//! holds, the loop each one starts (its "waiting" status event), and the
//! queue's own "Auto-starting task" line for `start_auto_tasks()`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use serial_test::serial;

use smartloop::api::services::model_service::ModelService;
use smartloop::tasks::base_task::TaskStatus;
use smartloop::workspace::Workspace;
use smartloop::workspace::context::{delegate_projects, pin_project};

use crate::shared::model_service_a::{ScratchConfig, bare_service, capture_logs};

/// The pin and the delegation flag are process-global; never leak one.
struct ClearProcessScope;

impl Drop for ClearProcessScope {
    fn drop(&mut self) {
        pin_project(None);
        delegate_projects(false);
    }
}

/// A service whose task queue records every status event, plus the record.
struct Harness {
    svc: Arc<ModelService>,
    events: Arc<Mutex<Vec<(String, Option<String>)>>>,
    logs: String,
    _config: ScratchConfig,
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.svc.task_queue.read().shutdown();
    }
}

/// Run the real `setup_tasks` on an empty workspace (no projects: the
/// ChromaDB migration loop has nothing to walk, which keeps this about task
/// registration).
fn run_setup_tasks() -> Harness {
    let config = ScratchConfig::new();
    let svc = bare_service();
    let events: Arc<Mutex<Vec<(String, Option<String>)>>> = Arc::default();
    let record = events.clone();
    svc.task_queue.read().on_status_change(Arc::new(
        move |name: &str, _status: TaskStatus, message: Option<&str>| {
            record.lock().push((name.to_string(), message.map(str::to_string)));
        },
    ));
    let workspace = Arc::new(RwLock::new(Workspace::new("ws")));
    let (result, logs) = capture_logs(tracing::Level::INFO, || svc.setup_tasks(Some(workspace)));
    result.expect("setup_tasks");
    Harness { svc, events, logs, _config: config }
}

impl Harness {
    fn registered(&self, name: &str) -> bool {
        self.svc.task_queue.read().get_task(name).is_some()
    }

    /// Wait for `name`'s event loop to report it is waiting for work.
    fn loop_started(&self, name: &str, message: &str, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if self
                .events
                .lock()
                .iter()
                .any(|(n, m)| n == name && m.as_deref() == Some(message))
            {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn any_event_for(&self, name: &str) -> bool {
        self.events.lock().iter().any(|(n, _)| n == name)
    }
}

const INDEXER_WAITING: &str = "Waiting for documents";
const MCP_WAITING: &str = "Waiting for MCP work";

#[test]
#[serial]
fn test_a_delegating_supervisor_registers_no_indexer() {
    let _scope = ClearProcessScope;
    delegate_projects(true);

    let h = run_setup_tasks();

    assert!(!h.registered("indexer"));
}

/// The watcher is the half that would actually scan the folder.
#[test]
#[serial]
fn test_a_delegating_supervisor_starts_no_watcher() {
    let _scope = ClearProcessScope;
    delegate_projects(true);

    let h = run_setup_tasks();

    assert!(!h.loop_started("indexer", INDEXER_WAITING, Duration::from_millis(300)));
    assert!(!h.any_event_for("indexer"));
}

/// The child is the tier that owns the project and may persist it.
#[test]
#[serial]
fn test_a_pinned_child_still_indexes_its_own_project() {
    let _scope = ClearProcessScope;
    pin_project(Some("alpha"));

    let h = run_setup_tasks();

    assert!(h.registered("indexer"));
    assert!(h.loop_started("indexer", INDEXER_WAITING, Duration::from_secs(5)));
}

/// `SLP_SUPERVISOR=0`: unpinned and undelegated, as it always was.
#[test]
#[serial]
fn test_the_single_server_fallback_still_indexes() {
    let _scope = ClearProcessScope;

    let h = run_setup_tasks();

    assert!(h.registered("indexer"));
    assert!(h.loop_started("indexer", INDEXER_WAITING, Duration::from_secs(5)));
}

/// Only the indexer is project work it cannot do.
///
/// MCPTask stays registered and is made inert by `pending_mcp_servers`, which
/// already returns nothing under delegation — so removing it here would
/// change behaviour this guard is not about.
#[test]
#[serial]
fn test_a_delegating_supervisor_still_runs_the_mcp_task() {
    let _scope = ClearProcessScope;
    delegate_projects(true);

    let h = run_setup_tasks();

    assert!(h.registered("mcp"));
    assert!(h.loop_started("mcp", MCP_WAITING, Duration::from_secs(5)));
}

#[test]
#[serial]
fn test_auto_tasks_still_start_under_delegation() {
    let _scope = ClearProcessScope;
    delegate_projects(true);

    let h = run_setup_tasks();

    // `start_auto_tasks()` ran exactly once: the queue announces each auto
    // task it starts, and the MCP task is always eligible.
    assert_eq!(h.logs.matches("[TaskQueue] Auto-starting task: mcp").count(), 1);
}
