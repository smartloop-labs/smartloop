//! Port of `smartloop/tasks/base_task.py`.
//!
//! Base Task for long-running background work.
//!
//! Each task runs in its own isolated thread with retry logic and status
//! tracking. Tasks can use agents to fulfill their work.

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::model::downloader::artifact_downloader::DownloadError;
use crate::platform::client::{ModelAccessError, PlatformError};
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
        let mut map = Map::new();
        map.insert("success".to_string(), Value::Bool(self.success));
        map.insert("data".to_string(), self.data.clone());
        map.insert(
            "error".to_string(),
            self.error.clone().map(Value::String).unwrap_or(Value::Null),
        );
        Value::Object(map)
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
///
/// Python checked the exception and its `__cause__`.  The Rust equivalents of
/// `PermissionError` are this module's [`PermissionError`], an `io::Error` of
/// kind `PermissionDenied`, the platform client's permission refusals
/// (`PlatformError::Permission`, `ModelAccessError`) and a download failure
/// wrapping any of those.
pub fn is_non_retryable(err: &anyhow::Error) -> bool {
    err.chain().any(is_permission_error)
}

/// True when `err` itself (not its causes) is a `PermissionError` — Python's
/// `except PermissionError:`.  A download failure wrapping a platform error
/// counts as that error, since the wrapper exists only on the Rust side.
pub fn is_permission_exception(err: &anyhow::Error) -> bool {
    if let Some(DownloadError::Source(inner)) = err.downcast_ref::<DownloadError>() {
        return is_permission_exception(inner);
    }
    is_permission_error(err.as_ref())
}

fn is_permission_error(err: &(dyn std::error::Error + 'static)) -> bool {
    if err.downcast_ref::<PermissionError>().is_some() {
        return true;
    }
    if let Some(io) = err.downcast_ref::<std::io::Error>() {
        return io.kind() == std::io::ErrorKind::PermissionDenied;
    }
    if let Some(platform) = err.downcast_ref::<PlatformError>() {
        return platform.is_permission();
    }
    if err.downcast_ref::<ModelAccessError>().is_some() {
        return true;
    }
    if let Some(download) = err.downcast_ref::<DownloadError>() {
        return match download {
            DownloadError::Io(io) => io.kind() == std::io::ErrorKind::PermissionDenied,
            // `#[error(transparent)]` forwards `source()` past the wrapped
            // error itself, so the wrapped error is inspected explicitly.
            DownloadError::Source(inner) => is_non_retryable(inner),
            _ => false,
        };
    }
    false
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
        let mut flag = self.flag.lock();
        *flag = true;
        self.cond.notify_all();
    }
    pub fn clear(&self) {
        *self.flag.lock() = false;
    }
    pub fn is_set(&self) -> bool {
        *self.flag.lock()
    }
    /// Block up to `timeout`; returns true if the event was set.
    pub fn wait(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut flag = self.flag.lock();
        while !*flag {
            if self.cond.wait_until(&mut flag, deadline).timed_out() {
                break;
            }
        }
        *flag
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
        Self {
            name: name.into(),
            logger,
            auto,
            max_retries: options.max_retries.unwrap_or(Self::MAX_RETRIES),
            on_status_change: RwLock::new(options.on_status_change),
            status: RwLock::new(TaskStatus::Pending),
            thread: Mutex::new(None),
            finished: Arc::new((Mutex::new(false), Condvar::new())),
            lock: Mutex::new(()),
            retry_count: AtomicU32::new(0),
            last_error: RwLock::new(None),
            result: RwLock::new(None),
            stop_event: Arc::new(StopEvent::new()),
        }
    }

    pub fn status(&self) -> TaskStatus {
        *self.status.read()
    }
    pub fn last_error(&self) -> Option<String> {
        self.last_error.read().clone()
    }
    pub fn result(&self) -> Option<TaskResult> {
        self.result.read().clone()
    }
    pub fn is_running(&self) -> bool {
        self.status() == TaskStatus::Running
    }
    pub fn on_status_change(&self) -> Option<StatusCallback> {
        self.on_status_change.read().clone()
    }
    /// Replace the per-task status listener (the queue installs its relay here).
    pub fn set_on_status_change(&self, callback: Option<StatusCallback>) {
        *self.on_status_change.write() = callback;
    }
    /// `_set_status`: record and notify (listener panics are swallowed).
    pub fn set_status(&self, status: TaskStatus, message: Option<&str>) {
        *self.status.write() = status;
        if let Some(callback) = self.on_status_change() {
            let _ = catch_unwind(AssertUnwindSafe(|| callback(&self.name, status, message)));
        }
    }
    pub fn set_last_error(&self, error: Option<String>) {
        *self.last_error.write() = error;
    }
    pub fn set_result(&self, result: Option<TaskResult>) {
        *self.result.write() = result;
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

/// Marks the worker thread finished when dropped, even if the work panicked.
struct FinishedGuard(Arc<(Mutex<bool>, Condvar)>);

impl Drop for FinishedGuard {
    fn drop(&mut self) {
        let (flag, cond) = &*self.0;
        *flag.lock() = true;
        cond.notify_all();
    }
}

/// Implementation of [`BaseTask::submit`]: ignore if running, reset retries,
/// clear the stop event and start `task-{name}` running [`run_with_retries`].
pub fn submit_task<T: BaseTask + ?Sized>(task: Arc<T>, kwargs: TaskKwargs) {
    let core = task.core();
    let _guard = core.lock.lock();
    if core.status() == TaskStatus::Running {
        core.logger.warning(&format!("[{}] Already running — ignoring submit", core.name));
        return;
    }
    core.retry_count.store(0, Ordering::SeqCst);
    core.stop_event.clear();
    start_thread(task.clone(), kwargs);
}

/// `_start_thread`: spawn the `task-{name}` worker running the retry loop.
fn start_thread<T: BaseTask + ?Sized>(task: Arc<T>, kwargs: TaskKwargs) {
    let core = task.core();
    let finished = core.finished.clone();
    *finished.0.lock() = false;
    let worker = task.clone();
    let spawned = std::thread::Builder::new()
        .name(format!("task-{}", core.name))
        .spawn(move || {
            let _done = FinishedGuard(finished);
            run_with_retries(&*worker, &kwargs);
        });
    match spawned {
        Ok(handle) => *core.thread.lock() = Some(handle),
        Err(exc) => {
            // The thread never started, so nothing else would flip the flag.
            *core.finished.0.lock() = true;
            core.finished.1.notify_all();
            core.logger.error(&format!("[{}] Could not start task thread: {exc}", core.name));
        }
    }
}

/// Implementation of [`BaseTask::wait`].
pub fn wait_task(core: &TaskCore, timeout: Option<f64>) -> bool {
    if core.thread.lock().is_none() {
        return true;
    }
    let (flag, cond) = &*core.finished;
    let mut done = flag.lock();
    match timeout {
        None => {
            while !*done {
                cond.wait(&mut done);
            }
        }
        Some(seconds) => {
            let deadline = Instant::now() + Duration::from_secs_f64(seconds.max(0.0));
            while !*done {
                if cond.wait_until(&mut done, deadline).timed_out() {
                    break;
                }
            }
        }
    }
    *done
}

/// Message of a caught panic payload (`str(exc)` for a panicking body).
pub(crate) fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "task panicked".to_string()
    }
}

/// Execute with retry + exponential backoff (`_run_with_retries`).
pub fn run_with_retries<T: BaseTask + ?Sized>(task: &T, kwargs: &TaskKwargs) {
    let core = task.core();
    let name = core.name.as_str();
    while core.retry_count.load(Ordering::SeqCst) <= core.max_retries {
        if core.stop_event.is_set() {
            core.set_status(TaskStatus::Failed, Some("Stopped by user"));
            return;
        }

        let attempt = core.retry_count.load(Ordering::SeqCst) + 1;
        core.set_status(TaskStatus::Running, Some(&format!("Attempt {attempt}")));
        core.logger.info(&format!(
            "[{name}] Starting (attempt {attempt}/{})",
            core.max_retries + 1
        ));

        // A panic inside `execute` is an exception like any other: it fails
        // this attempt instead of killing the thread with the status stuck.
        let outcome = match catch_unwind(AssertUnwindSafe(|| task.execute(kwargs))) {
            Ok(outcome) => outcome,
            Err(payload) => Err(anyhow::anyhow!(panic_message(&*payload))),
        };

        let exc = match outcome {
            Ok(result) => {
                let success = result.success;
                let error = result.error.clone();
                core.set_result(Some(result));
                if success {
                    core.set_status(TaskStatus::Completed, Some("Success"));
                    core.logger.info(&format!("[{name}] Completed successfully"));
                    return;
                }
                anyhow::anyhow!(error
                    .filter(|e| !e.is_empty())
                    .unwrap_or_else(|| "Task returned failure".to_string()))
            }
            Err(exc) => exc,
        };

        let message = exc.to_string();
        core.set_last_error(Some(message.clone()));
        core.logger.error(&format!("[{name}] Failed (attempt {attempt}): {message}"));
        core.logger.debug(&format!("{exc:?}"));

        // Non-retryable errors (e.g. 403 Forbidden) fail immediately
        if is_non_retryable(&exc) {
            core.set_result(Some(TaskResult::failure(message.clone())));
            core.set_status(TaskStatus::Failed, Some(&message));
            core.logger.error(&format!("[{name}] Non-retryable error — failing immediately"));
            return;
        }

        let retry_count = core.retry_count.fetch_add(1, Ordering::SeqCst) + 1;
        if retry_count > core.max_retries {
            core.set_result(Some(TaskResult::failure(message.clone())));
            core.set_status(TaskStatus::Failed, Some(&message));
            core.logger.error(&format!(
                "[{name}] Exhausted all {} attempts",
                core.max_retries + 1
            ));
            return;
        }

        let backoff = TaskCore::RETRY_BACKOFF_BASE.saturating_pow(retry_count);
        core.set_status(TaskStatus::Retrying, Some(&format!("Retrying in {backoff}s")));
        core.logger.info(&format!("[{name}] Retrying in {backoff}s..."));

        // Interruptible sleep
        if core.stop_event.wait(Duration::from_secs(backoff)) {
            core.set_status(TaskStatus::Failed, Some("Stopped during backoff"));
            return;
        }
    }
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
        self.items.lock().push_back(item);
        self.cond.notify_one();
    }
    /// Block up to `timeout` for an item; `None` on timeout (`queue.Empty`).
    pub fn get(&self, timeout: Duration) -> Option<T> {
        let deadline = Instant::now() + timeout;
        let mut items = self.items.lock();
        loop {
            if let Some(item) = items.pop_front() {
                return Some(item);
            }
            if self.cond.wait_until(&mut items, deadline).timed_out() {
                return items.pop_front();
            }
        }
    }
    /// Non-blocking pop (`get_nowait`).
    pub fn get_nowait(&self) -> Option<T> {
        self.items.lock().pop_front()
    }
    pub fn qsize(&self) -> usize {
        self.items.lock().len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.lock().is_empty()
    }
}
