//! Port of `smartloop/tasks/mcp_task.py`.
//!
//! MCP Task — background provisioning for MCP servers.
//!
//! Registration persists the connection and returns; this task does the rest
//! out of band: a queued server is discovered and enriched, and its `status`
//! moves `pending` → `ready`/`failed` with `last_error` set, which is what
//! clients poll instead of blocking.  A server left `pending` by an
//! interrupted run (app quit, crash) is adopted on the first loop tick.
//!
//! The work itself belongs to `ModelService` (it owns the workspace, the LLM
//! and the connections); this task owns only the threading, and is handed the
//! callables at setup.

use std::any::Any;
use std::collections::HashSet;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::Value;

use crate::tasks::base_task::{
    BaseTask, TaskCore, TaskKwargs, TaskLogger, TaskOptions, TaskResult, WorkQueue,
};
use crate::workspace::context::ProjectBoundExecutor;

/// How long the loop blocks on the provisioning queue before looking around.
pub const QUEUE_POLL_SECONDS: Duration = Duration::from_secs(1);

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
    /// Makes [`MCPTask::start_loop`] idempotent under concurrent callers
    /// (`start_auto_tasks()` and `setup_tasks`' `start_watcher()`).
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
        let _ = (logger, provisioner, pending_provider, options);
        todo!()
    }

    /// Queue a server for background provisioning (starts the loop).
    pub fn enqueue(&self, project_id: &str, server_id: &str) {
        let _ = (project_id, server_id);
        todo!()
    }

    /// Snapshot of MCP background work, surfaced via /health:
    /// `{"queued": N, "provisioning": N}`.
    pub fn mcp_status(&self) -> Value {
        todo!()
    }

    /// Current provisioning queue depth.
    pub fn queue_size(&self) -> usize {
        todo!()
    }

    /// Start the `mcp-loop` thread and `mcp-worker` pool (idempotent).
    pub fn start_loop(&self) {
        todo!()
    }

    /// Alias matching `IndexerTask`, so `setup_tasks` reads the same for both.
    pub fn start_watcher(&self) {
        self.start_loop()
    }

    fn event_loop(self: Arc<Self>) {
        todo!()
    }

    /// Queue servers left `pending` by a previous run, once per loop start.
    fn seed_pending_once(&self) {
        todo!()
    }

    fn run_provision(&self, project_id: &str, server_id: &str) {
        let _ = (project_id, server_id);
        todo!()
    }
}

impl BaseTask for MCPTask {
    fn core(&self) -> &TaskCore {
        &self.core
    }

    /// Batch entry point for `TaskQueue::submit()`: start the loop.
    fn execute(&self, kwargs: &TaskKwargs) -> anyhow::Result<TaskResult> {
        let _ = kwargs;
        todo!()
    }

    /// Auto-started at setup: the loop itself is the work.
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
        todo!()
    }
}
