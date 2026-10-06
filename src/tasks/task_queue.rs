//! Port of `smartloop/tasks/task_queue.py`.
//!
//! Task queue — manages submission, scheduling, and lifecycle of Tasks.
//!
//! All tasks are registered at startup and executed in isolated threads.
//! Failed tasks are automatically retried.  The queue provides a unified
//! status view for the API layer.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use indexmap::IndexMap;
use parking_lot::{Mutex, RwLock};
use serde_json::{json, Map, Value};

use crate::tasks::base_task::{BaseTask, StatusCallback, TaskKwargs, TaskLogger, TaskStatus};

/// Central registry and scheduler for Tasks.
///
/// ```ignore
/// let queue = TaskQueue::new(logger);
/// queue.register(downloader_task);
/// queue.start_auto_tasks();          // kicks off tasks with auto=true
/// queue.submit("downloader", Default::default());
/// queue.shutdown();                  // graceful stop
/// ```
pub struct TaskQueue {
    pub logger: TaskLogger,
    tasks: Arc<RwLock<IndexMap<String, Arc<dyn BaseTask>>>>,
    lock: Mutex<()>,
    status_callbacks: Arc<RwLock<Vec<StatusCallback>>>,
}

impl TaskQueue {
    pub fn new(logger: TaskLogger) -> Self {
        Self {
            logger,
            tasks: Arc::new(RwLock::new(IndexMap::new())),
            lock: Mutex::new(()),
            status_callbacks: Arc::new(RwLock::new(Vec::new())),
        }
    }

    // ------------------------------------------------------------------
    // Registration
    // ------------------------------------------------------------------

    /// Register a task.  Overwrites any existing task with the same name.
    pub fn register(&self, task: Arc<dyn BaseTask>) {
        let _guard = self.lock.lock();
        let callbacks = self.status_callbacks.clone();
        let relay: StatusCallback = Arc::new(move |name: &str, status: TaskStatus, message: Option<&str>| {
            Self::on_task_status_change(&callbacks, name, status, message)
        });
        task.core().set_on_status_change(Some(relay));
        let name = task.name().to_string();
        let auto = task.auto();
        self.tasks.write().insert(name.clone(), task);
        self.logger.info(&format!(
            "[TaskQueue] Registered task: {name} (auto={})",
            py_bool(auto)
        ));
    }

    pub fn unregister(&self, name: &str) {
        let _guard = self.lock.lock();
        let task = self.tasks.write().shift_remove(name);
        if let Some(task) = task {
            task.stop();
        }
    }

    // ------------------------------------------------------------------
    // Submission
    // ------------------------------------------------------------------

    /// Submit a task for execution by name.  Returns true if found and submitted.
    pub fn submit(&self, name: &str, kwargs: TaskKwargs) -> bool {
        let Some(task) = self.get_task(name) else {
            self.logger.warning(&format!("[TaskQueue] Unknown task: {name}"));
            return false;
        };
        task.submit(kwargs);
        true
    }

    /// Start all tasks that have `auto=true` and pass their `should_run` check.
    pub fn start_auto_tasks(&self) {
        for task in self.snapshot() {
            if task.auto() && task.should_run() {
                self.logger.info(&format!("[TaskQueue] Auto-starting task: {}", task.name()));
                task.submit(TaskKwargs::new());
            }
        }
    }

    // ------------------------------------------------------------------
    // Status
    // ------------------------------------------------------------------

    /// Return status dict for one (`Some(name)`) or all tasks.
    pub fn get_status(&self, name: Option<&str>) -> Value {
        if let Some(name) = name.filter(|n| !n.is_empty()) {
            return match self.get_task(name) {
                Some(task) => Self::task_status_dict(&*task),
                None => json!({ "error": format!("Unknown task: {name}") }),
            };
        }

        let mut all = Map::new();
        for task in self.snapshot() {
            all.insert(task.name().to_string(), Self::task_status_dict(&*task));
        }
        Value::Object(all)
    }

    /// Return the registered task by name.
    pub fn get_task(&self, name: &str) -> Option<Arc<dyn BaseTask>> {
        self.tasks.read().get(name).cloned()
    }

    /// Return the registered task by name, downcast to its concrete type.
    pub fn get_task_as<T: BaseTask>(&self, name: &str) -> Option<Arc<T>> {
        self.get_task(name)?.into_any_arc().downcast::<T>().ok()
    }

    /// Register a global status-change listener.
    pub fn on_status_change(&self, callback: StatusCallback) {
        self.status_callbacks.write().push(callback);
    }

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /// Block until a specific task completes.  Returns true if completed.
    pub fn wait_for(&self, name: &str, timeout: Option<f64>) -> bool {
        match self.get_task(name) {
            Some(task) => task.wait(timeout),
            None => true,
        }
    }

    /// Stop all tasks gracefully.
    pub fn shutdown(&self) {
        self.logger.info("[TaskQueue] Shutting down all tasks...");
        let tasks = self.snapshot();
        for task in &tasks {
            task.stop();
        }
        // Give threads a moment to wind down
        for task in &tasks {
            task.wait(Some(5.0));
        }
        self.logger.info("[TaskQueue] All tasks stopped");
    }

    // ------------------------------------------------------------------
    // Internal
    // ------------------------------------------------------------------

    /// The registered tasks in registration order, without holding the map
    /// lock while they run.
    fn snapshot(&self) -> Vec<Arc<dyn BaseTask>> {
        self.tasks.read().values().cloned().collect()
    }

    /// `{"name", "status", "auto", "last_error"}` plus `queued` / `workspace` /
    /// `mcp` for tasks that expose them.
    pub fn task_status_dict(task: &dyn BaseTask) -> Value {
        let mut status = Map::new();
        status.insert("name".to_string(), Value::String(task.name().to_string()));
        status.insert("status".to_string(), Value::String(task.status().as_str().to_string()));
        status.insert("auto".to_string(), Value::Bool(task.auto()));
        status.insert(
            "last_error".to_string(),
            task.last_error().map(Value::String).unwrap_or(Value::Null),
        );
        // Expose queue depth for tasks that support it (e.g. IndexerTask)
        if let Some(queued) = task.queue_size() {
            status.insert("queued".to_string(), json!(queued));
        }
        // Expose background workspace-indexing state (IndexerTask)
        if let Some(workspace) = task.workspace_status() {
            status.insert("workspace".to_string(), workspace);
        }
        // Expose MCP provisioning state (MCPTask)
        if let Some(mcp) = task.mcp_status() {
            status.insert("mcp".to_string(), mcp);
        }
        Value::Object(status)
    }

    /// Relay individual task status changes to global listeners.
    fn on_task_status_change(
        callbacks: &RwLock<Vec<StatusCallback>>,
        task_name: &str,
        status: TaskStatus,
        message: Option<&str>,
    ) {
        let listeners: Vec<StatusCallback> = callbacks.read().clone();
        for cb in listeners {
            let _ = catch_unwind(AssertUnwindSafe(|| cb(task_name, status, message)));
        }
    }
}

/// Python's `str(bool)` (`True` / `False`) for log lines.
fn py_bool(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}
