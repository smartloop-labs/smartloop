//! Port of `smartloop/tasks/mcp_task.py`.
//!
//! MCP Task — background provisioning for MCP servers.
//!
//! Registering an MCP server used to do all of its work on the request: launch
//! the process (or connect), enumerate tools, then generate
//! description/tags/summary with the LLM.  That is seconds to tens of seconds
//! of subprocess, network and inference latency in front of a UI click, and an
//! LLM call on the registration path competes with whatever the user is
//! chatting to.
//!
//! So registration persists the connection and returns; this task does the
//! rest out of band: a queued server is discovered and enriched, and its
//! `status` moves `pending` → `ready`/`failed` with `last_error` set, which is
//! what clients poll instead of blocking.  A server left `pending` by an
//! interrupted run (app quit, crash) is adopted on the first loop tick.
//!
//! That is the whole job now.  This task also used to run a periodic content
//! crawl; that is gone — a connection's content is read by calling its tools
//! when a question needs them, so nothing indexes at startup.
//!
//! The work itself belongs to `ModelService` (it owns the workspace, the LLM
//! and the connections); this task owns only the threading, and is handed the
//! callables at setup.

use std::any::Any;
use std::collections::HashSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::tasks::base_task::{
    panic_message, BaseTask, TaskCore, TaskKwargs, TaskLogger, TaskOptions, TaskResult, TaskStatus,
    WorkQueue,
};
use crate::workspace::context::ProjectBoundExecutor;

/// How long the loop blocks on the provisioning queue before looking around.
pub const QUEUE_POLL_SECONDS: Duration = Duration::from_secs(1);

/// How long [`MCPTask`]'s `stop` waits for the loop thread to exit.
const STOP_JOIN_TIMEOUT: Duration = Duration::from_secs(2);

/// `provisioner(project_id, server_id) -> dict`.
pub type Provisioner = Arc<dyn Fn(&str, &str) -> anyhow::Result<Value> + Send + Sync>;

/// `pending_provider() -> [(project_id, server_id), ...]`.
pub type PendingProvider = Arc<dyn Fn() -> anyhow::Result<Vec<(String, String)>> + Send + Sync>;

/// Background MCP server provisioning.
pub struct MCPTask {
    core: TaskCore,
    me: Weak<MCPTask>,
    provisioner: Option<Provisioner>,
    pending_provider: Option<PendingProvider>,
    loop_thread: Mutex<Option<JoinHandle<()>>>,
    queue: WorkQueue<(String, String)>,
    executor: Mutex<Option<Arc<ProjectBoundExecutor>>>,
    inflight: Mutex<HashSet<String>>,
    pending_seeded: AtomicBool,
    /// Makes [`MCPTask::start_loop`] idempotent under concurrent callers.  Two
    /// reach it — `start_auto_tasks()` and `setup_tasks`' `start_watcher()` —
    /// and its guard was a check on a field it then assigned, so both could
    /// pass it before either published a thread.
    start_lock: Mutex<()>,
}

impl MCPTask {
    /// Task name `"mcp"`, `auto=true`.
    pub fn new(
        logger: TaskLogger,
        provisioner: Option<Provisioner>,
        pending_provider: Option<PendingProvider>,
        options: TaskOptions,
    ) -> Arc<Self> {
        Arc::new_cyclic(|me| Self {
            core: TaskCore::new("mcp", logger, true, options),
            me: me.clone(),
            provisioner,
            pending_provider,
            loop_thread: Mutex::new(None),
            queue: WorkQueue::new(),
            executor: Mutex::new(None),
            inflight: Mutex::new(HashSet::new()),
            pending_seeded: AtomicBool::new(false),
            start_lock: Mutex::new(()),
        })
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Queue a server for background provisioning (starts the loop).
    pub fn enqueue(&self, project_id: &str, server_id: &str) {
        self.queue.put((project_id.to_string(), server_id.to_string()));
        self.core.logger.info(&format!(
            "[MCP] Server {server_id} queued for provisioning (project {project_id})"
        ));
        self.start_loop();
    }

    /// Snapshot of MCP background work, surfaced via /health:
    /// `{"queued": N, "provisioning": N}`.
    pub fn mcp_status(&self) -> Value {
        let provisioning = self.inflight.lock().len();
        json!({ "queued": self.queue_size(), "provisioning": provisioning })
    }

    /// Current provisioning queue depth.
    pub fn queue_size(&self) -> usize {
        self.queue.qsize()
    }

    // ------------------------------------------------------------------
    // Event loop
    // ------------------------------------------------------------------

    /// Start the `mcp-loop` thread and `mcp-worker` pool (idempotent).
    pub fn start_loop(&self) {
        let _guard = self.start_lock.lock();
        if self.loop_alive() {
            return;
        }
        let Some(me) = self.me.upgrade() else {
            return;
        };
        self.core.stop_event.clear();
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
        let workers = cpus.clamp(2, 4);
        *self.executor.lock() = Some(Arc::new(ProjectBoundExecutor::new(workers, Some("mcp-worker"))));
        let spawned = std::thread::Builder::new()
            .name("mcp-loop".to_string())
            .spawn(move || me.event_loop());
        match spawned {
            Ok(handle) => {
                *self.loop_thread.lock() = Some(handle);
                self.core.logger.info("[MCP] Event loop started");
            }
            Err(exc) => {
                self.core.logger.error(&format!("[MCP] Event loop error: {exc}"));
            }
        }
    }

    /// Alias matching `IndexerTask`, so `setup_tasks` reads the same for both.
    pub fn start_watcher(&self) {
        self.start_loop()
    }

    fn loop_alive(&self) -> bool {
        self.loop_thread
            .lock()
            .as_ref()
            .is_some_and(|handle| !handle.is_finished())
    }

    fn event_loop(self: Arc<Self>) {
        self.core.set_status(TaskStatus::Pending, Some("Waiting for MCP work"));

        while !self.core.stop_event.is_set() {
            let tick = catch_unwind(AssertUnwindSafe(|| self.tick()));
            if let Err(payload) = tick {
                self.core
                    .logger
                    .error(&format!("[MCP] Event loop error: {}", panic_message(&*payload)));
            }
        }

        if let Some(executor) = self.executor.lock().take() {
            executor.shutdown(false);
        }
        self.core.set_status(TaskStatus::Completed, Some("Event loop stopped"));
    }

    /// One iteration of the loop body.
    fn tick(self: &Arc<Self>) {
        self.seed_pending_once();

        let Some((project_id, server_id)) = self.queue.get(QUEUE_POLL_SECONDS) else {
            let idle = self.inflight.lock().is_empty();
            if idle {
                self.core.set_status(TaskStatus::Pending, Some("Waiting for MCP work"));
            }
            return;
        };
        {
            let mut inflight = self.inflight.lock();
            if inflight.contains(&server_id) {
                return;
            }
            inflight.insert(server_id.clone());
        }
        self.core.set_status(TaskStatus::Running, Some("Provisioning MCP server"));
        let executor = self.executor.lock().clone();
        match executor {
            Some(executor) => {
                let me = self.clone();
                let _ = executor.submit(move || me.run_provision(&project_id, &server_id));
            }
            None => {
                self.inflight.lock().remove(&server_id);
                self.core
                    .logger
                    .error("[MCP] Event loop error: worker pool is shut down");
            }
        }
    }

    /// Queue servers left `pending` by a previous run, once per loop start.
    ///
    /// A registration that was interrupted (app quit, crash) leaves a server
    /// persisted but never discovered.  Nothing else would ever pick it up, so
    /// the first loop iteration adopts them.
    fn seed_pending_once(&self) {
        if self.pending_seeded.load(Ordering::SeqCst) {
            return;
        }
        let Some(provider) = self.pending_provider.as_ref() else {
            return;
        };
        self.pending_seeded.store(true, Ordering::SeqCst);
        let pending = match catch_unwind(AssertUnwindSafe(|| provider())) {
            Ok(Ok(pending)) => pending,
            Ok(Err(exc)) => {
                self.core
                    .logger
                    .warning(&format!("[MCP] Could not list pending servers: {exc}"));
                return;
            }
            Err(payload) => {
                self.core.logger.warning(&format!(
                    "[MCP] Could not list pending servers: {}",
                    panic_message(&*payload)
                ));
                return;
            }
        };
        let count = pending.len();
        for (project_id, server_id) in pending {
            self.queue.put((project_id, server_id));
        }
        if count > 0 {
            self.core
                .logger
                .info(&format!("[MCP] Resuming provisioning for {count} pending server(s)"));
        }
    }

    // ------------------------------------------------------------------
    // Workers
    // ------------------------------------------------------------------

    fn run_provision(&self, project_id: &str, server_id: &str) {
        if let Some(provisioner) = self.provisioner.as_ref() {
            let failure = match catch_unwind(AssertUnwindSafe(|| provisioner(project_id, server_id))) {
                Ok(Ok(_)) => None,
                Ok(Err(exc)) => Some(exc.to_string()),
                Err(payload) => Some(panic_message(&*payload)),
            };
            if let Some(exc) = failure {
                self.core
                    .logger
                    .error(&format!("[MCP] Provisioning {server_id} failed: {exc}"));
            }
        }
        self.inflight.lock().remove(server_id);
    }
}

impl BaseTask for MCPTask {
    fn core(&self) -> &TaskCore {
        &self.core
    }

    /// Batch entry point for `TaskQueue::submit()`: start the loop.
    fn execute(&self, _kwargs: &TaskKwargs) -> anyhow::Result<TaskResult> {
        self.start_loop();
        Ok(TaskResult::ok(json!({ "queued": self.queue_size() })))
    }

    /// Auto-started at setup: the loop itself is the work (it owns the
    /// provisioning queue).
    fn should_run(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn into_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn queue_size(&self) -> Option<usize> {
        Some(MCPTask::queue_size(self))
    }
    fn mcp_status(&self) -> Option<Value> {
        Some(MCPTask::mcp_status(self))
    }

    /// Signal stop and join the loop thread (up to 2s).
    fn stop(&self) {
        self.core.stop_event.set();
        let mut slot = self.loop_thread.lock();
        let Some(handle) = slot.as_ref() else {
            return;
        };
        if handle.thread().id() == std::thread::current().id() {
            return;
        }
        let deadline = Instant::now() + STOP_JOIN_TIMEOUT;
        while !handle.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        if handle.is_finished() {
            if let Some(handle) = slot.take() {
                let _ = handle.join();
            }
        }
    }
}
