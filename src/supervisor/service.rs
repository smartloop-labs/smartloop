//! Port of `smartloop/supervisor/service.py`.
//!
//! The supervisor: one port, one workspace registry, a process per project.
//!
//! It owns the client-facing port, the workspace record, the model catalog and
//! credentials, release polling and token refresh — and the children.  It does
//! not own any project's state; that belongs to the child pinned to it.
//! `SLP_SUPERVISOR=0` (or `slp agent start --no-supervised`) falls back to the
//! single shared server.

use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::{Map, Value};

#[allow(unused_imports)]
use crate::config::AppSettings;
use crate::supervisor::pool::ProjectAgentPool;
use crate::supervisor::registry::AgentRegistry;

/// Read the settings at the moment they are needed, not at import.
fn settings_now() -> AppSettings {
    todo!()
}

/// Values of `SLP_SUPERVISOR` that turn supervision *off*.
pub const DISABLED_VALUES: &[&str] = &["0", "false", "no", "off"];

/// Whether this process should supervise rather than serve projects itself.
///
/// On by default; a process pinned to a project is never a supervisor.
pub fn supervision_enabled() -> bool {
    todo!()
}

/// Whether the supervisor should run a model host for its children (always).
pub fn model_host_enabled() -> bool {
    true
}

/// `(project_id, project_dir)` pairs of the current workspace, or `None` when
/// there is no workspace.  Injectable so tests can stand in for
/// `Workspace::load`.
pub type WorkspaceProjects = Arc<dyn Fn() -> Option<Vec<(String, String)>> + Send + Sync>;

/// Owns the registry, the model host, and the pool of project agents.
pub struct Supervisor {
    pub home_dir: String,
    pub registry: Arc<AgentRegistry>,
    pub idle_ttl: f64,
    pub memory_limit_bytes: u64,
    model_host: Mutex<Option<crate::model_host::host::ModelHost>>,
    pool: Mutex<Option<Arc<ProjectAgentPool>>>,
    workspace_projects: WorkspaceProjects,
}

impl Supervisor {
    /// Build a supervisor; `None` arguments come from [`AppSettings`].
    pub fn new(home_dir: Option<String>, idle_ttl: Option<f64>, memory_limit_mb: Option<i64>) -> Self {
        let _ = (home_dir, idle_ttl, memory_limit_mb);
        todo!()
    }

    /// Replace how the workspace's projects are looked up (test seam).
    pub fn set_workspace_projects(&mut self, lookup: WorkspaceProjects) {
        self.workspace_projects = lookup;
    }

    /// The default lookup: `Workspace::load()` read fresh every time.
    pub fn default_workspace_projects() -> WorkspaceProjects {
        todo!()
    }

    /// The pool, once started.
    pub fn pool(&self) -> Option<Arc<ProjectAgentPool>> {
        self.pool.lock().clone()
    }

    /// Where a project's state lives, read fresh from the workspace record.
    pub fn project_dir_for(&self, project_id: &str) -> Option<String> {
        let _ = project_id;
        todo!()
    }

    /// Bring up the model host and the pool, and record ourselves.
    pub fn start(&self, port: u16) {
        let _ = port;
        todo!()
    }

    /// Take back agents that are running but in nobody's registry.
    ///
    /// A stray that answers `/health` is adopted; one that does not is stopped.
    pub fn reclaim_strays(&self) -> Vec<String> {
        todo!()
    }

    /// Run the model host in this process; `None` when it could not start.
    fn start_model_host(&self) -> Option<String> {
        todo!()
    }

    /// Stop the children, then the host. Children first: they use it.
    pub fn stop(&self) {
        todo!()
    }

    /// Stop the agent pinned to `project_id` and forget its backoff.
    pub fn release(&self, project_id: &str) -> bool {
        let _ = project_id;
        todo!()
    }

    /// What `slp agent status` reports: the whole tree, not one process.
    pub fn status(&self) -> Map<String, Value> {
        todo!()
    }
}

/// Whether something is serving the agent contract at `base_url`.
fn answers_health(base_url: &str, timeout: f64) -> bool {
    let _ = (base_url, timeout);
    todo!()
}

/// The process-wide supervisor, when this process is one.
static SUPERVISOR: Lazy<Mutex<Option<Arc<Supervisor>>>> = Lazy::new(|| Mutex::new(None));

/// The running supervisor, or `None` when it was explicitly disabled.
pub fn get_supervisor() -> Option<Arc<Supervisor>> {
    SUPERVISOR.lock().clone()
}

/// Stop `project_id`'s agent, if this process is supervising one.
pub fn release_project(project_id: &str) -> bool {
    let _ = project_id;
    todo!()
}

/// Create and start the process-wide supervisor.
pub fn start_supervisor(port: u16, idle_ttl: Option<f64>) -> Arc<Supervisor> {
    let _ = (port, idle_ttl);
    todo!()
}

/// Stop and forget the process-wide supervisor.
pub fn stop_supervisor() {
    todo!()
}

/// The pool routing should use, or `None` when not supervising.
pub fn get_pool() -> Option<Arc<ProjectAgentPool>> {
    get_supervisor().and_then(|s| s.pool())
}
