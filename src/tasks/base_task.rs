//! Port of `smartloop/tasks/base_task.py`.
//!
//! Base Task for long-running background work.
//!
//! Each task runs in its own isolated thread with retry logic and status
//! tracking. Tasks can use agents to fulfill their work.

use std::any::Any;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use parking_lot::{Condvar, Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::utils::log_utils::LoggerLike;

/// Shared logger handle (Python `logging.Logger`).
pub type TaskLogger = Arc<dyn LoggerLike>;

/// Keyword arguments forwarded from `submit(**kwargs)` to `execute(**kwargs)`.
pub type TaskKwargs = Map<String, Value>;

/// Status-change listener: `(task_name, status, message)`.
///
/// Python passed the task object itself to per-task callbacks and the task
/// name to queue-level callbacks; every consumer only reads the name, so both
/// use this one signature.
pub type StatusCallback = Arc<dyn Fn(&str, TaskStatus, Option<&str>) + Send + Sync>;

/// Status of a task in the task queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    #[default]
    Pending,
    Running,
    Completed,
    Failed,
    Retrying,
}

impl TaskStatus {
    /// The string value (`TaskStatus.X.value`).
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskStatus::Pending => "pending",
            TaskStatus::Running => "running",
            TaskStatus::Completed => "completed",
            TaskStatus::Failed => "failed",
            TaskStatus::Retrying => "retrying",
        }
    }
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Result of a task execution.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TaskResult {
    pub success: bool,
    #[serde(default)]
    pub data: Value,
    #[serde(default)]
    pub error: Option<String>,
}

impl TaskResult {
    pub fn new(success: bool, data: Value, error: Option<String>) -> Self {
        Self { success, data, error }
    }

    /// Successful result carrying `data`.
    pub fn ok(data: Value) -> Self {
        Self::new(true, data, None)
    }

    /// Failed result carrying `error`.
    pub fn failure(error: impl Into<String>) -> Self {
        Self::new(false, Value::Null, Some(error.into()))
    }

    /// `{"success": ..., "data": ..., "error": ...}`.
    pub fn to_dict(&self) -> Value {
        todo!()
    }
}

/// Error type that is never retried (Python `PermissionError`, e.g. a 403
/// from the platform).  Returning it — or an error whose chain contains it or
/// an `io::ErrorKind::PermissionDenied` — from [`BaseTask::execute`] fails the
/// task immediately.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct PermissionError(pub String);

/// True when `err` (or its cause chain) is a non-retryable error
/// (`NON_RETRYABLE_ERRORS = (PermissionError,)`).
pub fn is_non_retryable(err: &anyhow::Error) -> bool {
    let _ = err;
    todo!()
}

/// Constructor kwargs shared by every task (`max_retries`, `on_status_change`).
#[derive(Clone, Default)]
pub struct TaskOptions {
    pub max_retries: Option<u32>,
    pub on_status_change: Option<StatusCallback>,
}

/// Interruptible stop flag (Python `threading.Event`).
#[derive(Default)]
pub struct StopEvent {
    flag: Mutex<bool>,
    cond: Condvar,
}

impl StopEvent {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn set(&self) {
        todo!()
    }
    pub fn clear(&self) {
        todo!()
    }
    pub fn is_set(&self) -> bool {
        todo!()
    }
    /// Block up to `timeout`; returns true if the event was set.
    pub fn wait(&self, timeout: Duration) -> bool {
        let _ = timeout;
        todo!()
    }
}

/// Shared state every task carries (the fields `BaseTask.__init__` sets).
pub struct TaskCore {
    pub name: String,
    pub logger: TaskLogger,
    pub auto: bool,
    pub max_retries: u32,
    on_status_change: RwLock<Option<StatusCallback>>,
    status: RwLock<TaskStatus>,
    thread: Mutex<Option<JoinHandle<()>>>,
    /// Signalled (true) when the worker thread has finished.
    finished: Arc<(Mutex<bool>, Condvar)>,
    lock: Mutex<()>,
    retry_count: AtomicU32,
    last_error: RwLock<Option<String>>,
    result: RwLock<Option<TaskResult>>,
    pub stop_event: Arc<StopEvent>,
}

impl TaskCore {
    pub const MAX_RETRIES: u32 = 3;
    /// Seconds; doubles each attempt.
    pub const RETRY_BACKOFF_BASE: u64 = 2;

    pub fn new(name: impl Into<String>, logger: TaskLogger, auto: bool, options: TaskOptions) -> Self {
        let _ = (name.into(), logger, auto, options);
        let _ = (
            &Self::on_status_change,
            AtomicU32::new(0),
            Mutex::new(()),
            Mutex::<Option<JoinHandle<()>>>::new(None),
            Arc::new((Mutex::new(false), Condvar::new())),
        );
        todo!()
    }

    pub fn status(&self) -> TaskStatus {
        todo!()
    }
    pub fn last_error(&self) -> Option<String> {
        todo!()
    }
    pub fn result(&self) -> Option<TaskResult> {
        todo!()
    }
    pub fn is_running(&self) -> bool {
        todo!()
    }
    pub fn on_status_change(&self) -> Option<StatusCallback> {
        todo!()
    }
    /// Replace the per-task status listener (the queue installs its relay here).
    pub fn set_on_status_change(&self, callback: Option<StatusCallback>) {
        let _ = callback;
        todo!()
    }
    /// `_set_status`: record and notify (listener panics are swallowed).
    pub fn set_status(&self, status: TaskStatus, message: Option<&str>) {
        let _ = (status, message);
        todo!()
    }
    pub fn set_last_error(&self, error: Option<String>) {
        let _ = error;
        todo!()
    }
    pub fn set_result(&self, result: Option<TaskResult>) {
        let _ = result;
        todo!()
    }
}

/// Base trait for long-running isolated tasks.
///
/// Each task:
/// - Runs in its own isolated daemon thread
/// - Has configurable auto-start behaviour
/// - Supports retry with exponential backoff on failure
/// - Emits status updates via an optional callback
/// - Can use agents to fulfill its work
pub trait BaseTask: Send + Sync + 'static {
    /// The task's shared state.
    fn core(&self) -> &TaskCore;

    /// Execute the task's work.  Returning `Err` is treated as failure and
    /// triggers a retry (up to `max_retries`), unless non-retryable.
    fn execute(&self, kwargs: &TaskKwargs) -> anyhow::Result<TaskResult>;

    /// Pre-condition check called before `execute` on auto-triggered tasks.
    fn should_run(&self) -> bool {
        true
    }

    /// Downcasting support (`TaskQueue::get_task_as`).
    fn as_any(&self) -> &dyn Any;
    fn into_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    /// Queue depth, for tasks that have one (`hasattr(task, "queue_size")`).
    fn queue_size(&self) -> Option<usize> {
        None
    }
    /// Background workspace-indexing state (IndexerTask).
    fn workspace_status(&self) -> Option<Value> {
        None
    }
    /// MCP provisioning state (MCPTask).
    fn mcp_status(&self) -> Option<Value> {
        None
    }

    fn name(&self) -> &str {
        &self.core().name
    }
    fn auto(&self) -> bool {
        self.core().auto
    }
    fn status(&self) -> TaskStatus {
        self.core().status()
    }
    fn last_error(&self) -> Option<String> {
        self.core().last_error()
    }
    fn result(&self) -> Option<TaskResult> {
        self.core().result()
    }
    fn is_running(&self) -> bool {
        self.core().is_running()
    }

    /// Submit the task for execution in its own thread.
    fn submit(self: Arc<Self>, kwargs: TaskKwargs) {
        submit_task(self, kwargs)
    }

    /// Signal the task to stop gracefully.
    fn stop(&self) {
        self.core().stop_event.set()
    }

    /// Block until the task thread finishes (`timeout` in seconds).  Returns
    /// true if completed (or never started).
    fn wait(&self, timeout: Option<f64>) -> bool {
        wait_task(self.core(), timeout)
    }
}

/// Implementation of [`BaseTask::submit`]: ignore if running, reset retries,
/// clear the stop event and start `task-{name}` running [`run_with_retries`].
pub fn submit_task<T: BaseTask + ?Sized>(task: Arc<T>, kwargs: TaskKwargs) {
    let _ = (task, kwargs);
    todo!()
}

/// Implementation of [`BaseTask::wait`].
pub fn wait_task(core: &TaskCore, timeout: Option<f64>) -> bool {
    let _ = (core, timeout);
    todo!()
}

/// Execute with retry + exponential backoff (`_run_with_retries`).
pub fn run_with_retries<T: BaseTask + ?Sized>(task: &T, kwargs: &TaskKwargs) {
    let _ = (task, kwargs);
    todo!()
}

/// Unbounded thread-safe FIFO (Python `queue.Queue`).
pub struct WorkQueue<T> {
    items: Mutex<std::collections::VecDeque<T>>,
    cond: Condvar,
}

impl<T> Default for WorkQueue<T> {
    fn default() -> Self {
        Self { items: Mutex::new(std::collections::VecDeque::new()), cond: Condvar::new() }
    }
}

impl<T> WorkQueue<T> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn put(&self, item: T) {
        let _ = (item, &self.items, &self.cond);
        todo!()
    }
    /// Block up to `timeout` for an item; `None` on timeout (`queue.Empty`).
    pub fn get(&self, timeout: Duration) -> Option<T> {
        let _ = timeout;
        todo!()
    }
    /// Non-blocking pop (`get_nowait`).
    pub fn get_nowait(&self) -> Option<T> {
        todo!()
    }
    pub fn qsize(&self) -> usize {
        todo!()
    }
    pub fn is_empty(&self) -> bool {
        todo!()
    }
}
