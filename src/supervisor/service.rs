//! Port of `smartloop/supervisor/service.py`.
//!
//! The supervisor: one port, one workspace registry, a process per project.
//!
//! What it owns (Tier 1 in the RFC): the client-facing port, the workspace
//! record, the model catalog and credentials, release polling and token
//! refresh — and the children.  What it does not own is any project's state;
//! that belongs to the child fixed to that project, which is the whole point
//! of the split.
//!
//! This is the architecture, not a mode.  `SLP_SUPERVISOR=0` (or `slp agent
//! start --no-supervised`) still falls back to the single shared server, but
//! that is a migration escape hatch to be removed once this has run in the
//! field — not a second topology to keep working indefinitely.

use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::{Map, Value};

use crate::config::AppSettings;
use crate::supervisor::pool::{http_get_status, process_rss, terminate_pid, PoolOptions, ProjectAgentPool};
use crate::supervisor::registry::{
    process_alive, py_int, read_project_agent, AgentRecord, AgentRegistry,
};

/// Logger name of this module.
const LOGGER: &str = "smartloop.supervisor.service";

/// Read the settings at the moment they are needed, not at import.
///
/// Settings frozen when the module was first used would make the agent limits
/// impossible to set from a launcher — and would pin the workspace paths, which
/// had to stay unpinned for a child to be pointed at its own home.
fn settings_now() -> AppSettings {
    AppSettings::new()
}

/// Values of `SLP_SUPERVISOR` that turn supervision *off*.  Supervision is the
/// default, so the environment variable is an opt-out.
pub const DISABLED_VALUES: &[&str] = &["0", "false", "no", "off"];

/// Whether this process should supervise rather than serve projects itself.
///
/// On by default: a project's runaway index job, wedged MCP server or OOM
/// should cost that project and no other, and that is only true when each one
/// has its own process.  `SLP_SUPERVISOR=0` (or `slp agent start
/// --no-supervised`) falls back to the single shared server.
///
/// A process pinned to a project is never a supervisor, whatever the
/// environment says.  Children inherit the environment, so with supervision
/// defaulted on this is what stops a child from trying to supervise itself and
/// spawning a grandchild for the project it already is.
pub fn supervision_enabled() -> bool {
    if crate::workspace::context::pinned_project_id().is_some() {
        return false;
    }
    let value = std::env::var("SLP_SUPERVISOR").unwrap_or_default().trim().to_lowercase();
    !DISABLED_VALUES.contains(&value.as_str())
}

/// Whether the supervisor should run a model host for its children (always).
///
/// The gRPC IPC path is the only supported transport; any opt-out is ignored
/// so the shared model host remains the default and the only routing choice.
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
        let settings = settings_now();
        let home_dir = home_dir.filter(|h| !h.is_empty()).unwrap_or_else(|| settings.home_dir());
        let registry = Arc::new(AgentRegistry::new(home_dir.clone()).load());
        let idle_ttl = idle_ttl.unwrap_or(settings.agent_idle_ttl_s);
        let memory_limit_mb = memory_limit_mb.unwrap_or(settings.agent_memory_limit_mb);
        let memory_limit_bytes = u64::try_from(memory_limit_mb.saturating_mul(1024 * 1024)).unwrap_or(0);
        Self {
            home_dir,
            registry,
            idle_ttl,
            memory_limit_bytes,
            model_host: Mutex::new(None),
            pool: Mutex::new(None),
            workspace_projects: Self::default_workspace_projects(),
        }
    }

    /// Replace how the workspace's projects are looked up (test seam).
    pub fn set_workspace_projects(&mut self, lookup: WorkspaceProjects) {
        self.workspace_projects = lookup;
    }

    /// The default lookup: `Workspace::load()` read fresh every time.
    pub fn default_workspace_projects() -> WorkspaceProjects {
        Arc::new(|| {
            crate::workspace::Workspace::load().map(|workspace| {
                workspace
                    .projects
                    .iter()
                    .map(|project| (project.id.clone(), project.project_dir.clone()))
                    .collect()
            })
        })
    }

    /// The pool, once started.
    pub fn pool(&self) -> Option<Arc<ProjectAgentPool>> {
        self.pool.lock().clone()
    }

    /// Replace the pool (Python assigns `supervisor.pool` directly).
    pub fn set_pool(&self, pool: Option<Arc<ProjectAgentPool>>) {
        *self.pool.lock() = pool;
    }

    // -- project lookup -----------------------------------------------------

    /// Where a project's state lives, read fresh from the workspace record.
    ///
    /// Deliberately not cached: a project created while the supervisor is
    /// running must be routable without a restart, and the workspace record is
    /// cheap to read now that it no longer carries every project's documents.
    pub fn project_dir_for(&self, project_id: &str) -> Option<String> {
        lookup_project_dir(&self.workspace_projects, project_id)
    }

    // -- lifecycle ----------------------------------------------------------

    /// Bring up the model host and the pool, and record ourselves.
    pub fn start(&self, port: u16) {
        self.registry.set_supervisor(i64::from(std::process::id()), port);

        // From here each project's own file belongs to the child pinned to it.
        // Declared before the pool exists rather than after, so a save on the
        // way up cannot republish this process's snapshot over a project a
        // reclaimed agent is already serving.
        crate::workspace::context::delegate_projects(true);

        let address = self.start_model_host();
        let lookup = Arc::clone(&self.workspace_projects);
        let pool = Arc::new(ProjectAgentPool::with_options(
            Arc::clone(&self.registry),
            Arc::new(move |project_id: &str| lookup_project_dir(&lookup, project_id)),
            PoolOptions {
                idle_ttl: self.idle_ttl,
                model_host_address: address,
                memory_limit_bytes: self.memory_limit_bytes,
                ..PoolOptions::default()
            },
        ));
        pool.start_reaper(30.0);
        self.set_pool(Some(pool));

        let adopted: Vec<String> = self.registry.live().iter().map(|r| r.project_id.clone()).collect();
        if !adopted.is_empty() {
            tracing::info!(target: LOGGER, "[supervisor] adopted running agents for {}", adopted.join(", "));
        }
        self.reclaim_strays();
        tracing::info!(target: LOGGER, "[supervisor] ready on port {}", port);
    }

    /// Take back agents that are running but in nobody's registry.
    ///
    /// The registry is how a new supervisor finds the children an old one
    /// left, so anything missing from it is invisible: a stray keeps its
    /// project's task queue and MCP children alive, holds the port recorded
    /// beside the project, and — before the parent watch — never exited.  The
    /// registry goes missing more easily than it sounds: a workspace whose home
    /// was wiped, an agent from a test run against a temporary home, a crash
    /// between spawning a child and recording it.
    ///
    /// A project directory still describes itself (`agent.pid`,
    /// `agent.port`), so that is what is read here.  A stray that answers
    /// `/health` is adopted rather than killed — it is a warm process serving
    /// the right project, and once it is in the registry this supervisor is
    /// the successor its parent watch will find.  One that does not answer is
    /// stopped, and its project's next request starts a fresh one.
    pub fn reclaim_strays(&self) -> Vec<String> {
        let Some(projects) = (self.workspace_projects)() else {
            return Vec::new();
        };

        let mut reclaimed = Vec::new();
        for (project_id, project_dir) in projects {
            let Some((pid, port)) = read_project_agent(&project_dir) else {
                continue;
            };
            if !process_alive(Some(pid)) {
                continue;
            }
            if let Some(known) = self.registry.get(&project_id) {
                if known.pid == pid {
                    continue; // already adopted above
                }
            }

            let record = AgentRecord::new(project_id.clone(), pid, port);
            if answers_health(&record.base_url(), 5.0) {
                tracing::info!(
                    target: LOGGER,
                    "[supervisor] reclaiming stray agent for {} (pid {}, port {})",
                    project_id,
                    pid,
                    port
                );
                self.registry.add(record, Some(&project_dir));
                reclaimed.push(project_id);
            } else {
                tracing::warn!(
                    target: LOGGER,
                    "[supervisor] stray agent for {} (pid {}) is not answering; stopping it",
                    project_id,
                    pid
                );
                terminate_pid(pid, 10.0);
            }
        }
        reclaimed
    }

    /// Run the model host in this process; `None` when it could not start.
    ///
    /// In-process rather than a fourth process: the supervisor is already
    /// long-lived and does no project work, so it is the natural home for the
    /// device, and one less process is one less thing to supervise.  Children
    /// reach it over the socket exactly as they would a separate host, so
    /// splitting it out later changes nothing on their side.
    fn start_model_host(&self) -> Option<String> {
        use crate::model_host::client::{default_socket_path, InProcessModelClient, LocalModelService};
        use crate::model_host::host::ModelHost;

        let configured = std::env::var("SLP_MODEL_HOST").unwrap_or_default().trim().to_string();
        let address = if configured.is_empty() { default_socket_path(&self.home_dir) } else { configured };

        let service: Arc<dyn LocalModelService> = crate::api::services::get_model_service();
        let host = ModelHost::new(Arc::new(InProcessModelClient::new(service)), address);
        match host.start() {
            Ok(bound) => {
                *self.model_host.lock() = Some(host);
                self.registry.set_model_host(Some(i64::from(std::process::id())), Some(&bound));
                Some(bound)
            }
            Err(exc) => {
                // Losing the host is not fatal: children fall back to loading
                // their own models, which is slower and heavier but still
                // correct.
                tracing::warn!(target: LOGGER, "[supervisor] could not start the model host: {}", exc);
                self.registry.set_model_host(None, None);
                None
            }
        }
    }

    /// Stop the children, then the host. Children first: they use it.
    pub fn stop(&self) {
        crate::workspace::context::delegate_projects(false);
        let pool = self.pool.lock().take();
        if let Some(pool) = pool {
            pool.stop_reaper();
            pool.stop_all(10.0);
        }
        let host = self.model_host.lock().take();
        if let Some(host) = host {
            host.stop();
        }
        self.registry.set_model_host(None, None);
        tracing::info!(target: LOGGER, "[supervisor] stopped");
    }

    /// Stop the agent pinned to `project_id` and forget its backoff.
    ///
    /// For a project that is going away — the only lifecycle event that has to
    /// reach the pool.  Everything else the pool stops (idle, over the RSS
    /// ceiling, crashed) is a project that will be started again on its next
    /// request; this one will not, and leaving the child up would have it
    /// serving a project whose directory is about to be removed underneath it.
    ///
    /// Returns whether there was an agent to stop.
    pub fn release(&self, project_id: &str) -> bool {
        let Some(pool) = self.pool() else {
            return false;
        };
        let stopped = pool.stop(project_id, 10.0);
        pool.clear_backoff(project_id);
        stopped
    }

    // -- reporting ----------------------------------------------------------

    /// What `slp agent status` reports: the whole tree, not one process.
    pub fn status(&self) -> Map<String, Value> {
        let agents: Vec<Value> = self
            .registry
            .all()
            .iter()
            .map(|record| {
                let mut entry = Map::new();
                entry.insert("project_id".into(), Value::from(record.project_id.clone()));
                entry.insert("pid".into(), Value::from(record.pid));
                entry.insert("port".into(), Value::from(record.port));
                entry.insert("alive".into(), Value::from(record.alive()));
                entry.insert("idle_seconds".into(), float(round1(record.idle_seconds())));
                entry.insert("rss_bytes".into(), rss(Some(record.pid)));
                Value::Object(entry)
            })
            .collect();

        let mut supervisor = self.registry.supervisor();
        let supervisor_pid = supervisor.get("pid").and_then(py_int);
        supervisor.insert("rss_bytes".into(), rss(supervisor_pid));

        let mut limits = Map::new();
        limits.insert("idle_ttl_seconds".into(), float(self.idle_ttl));
        limits.insert(
            "memory_limit_bytes".into(),
            if self.memory_limit_bytes == 0 { Value::Null } else { Value::from(self.memory_limit_bytes) },
        );

        let mut status = Map::new();
        status.insert("supervisor".into(), Value::Object(supervisor));
        status.insert("model_host".into(), Value::Object(self.registry.model_host()));
        status.insert("agents".into(), Value::Array(agents));
        status.insert("limits".into(), Value::Object(limits));
        status
    }
}

/// `project_id`'s directory according to `lookup`.
fn lookup_project_dir(lookup: &WorkspaceProjects, project_id: &str) -> Option<String> {
    lookup()?.into_iter().find(|(id, _)| id == project_id).map(|(_, dir)| dir)
}

/// Python's `round(value, 1)`.
fn round1(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

/// A float as JSON (`null` when not finite).
fn float(value: f64) -> Value {
    serde_json::Number::from_f64(value).map(Value::Number).unwrap_or(Value::Null)
}

/// Resident size of `pid` as JSON, or `null` when it cannot be read.
///
/// The number the RFC wants measured before deciding whether children are
/// lazily started or pre-warmed: a child carries the server, the SQLite index
/// and the whole service graph even with no model in it.
fn rss(pid: Option<i64>) -> Value {
    process_rss(pid.filter(|p| *p != 0)).map(Value::from).unwrap_or(Value::Null)
}

/// Whether something is serving the agent contract at `base_url`.
fn answers_health(base_url: &str, timeout: f64) -> bool {
    http_get_status(&format!("{base_url}/health"), timeout) == Some(200)
}

/// The process-wide supervisor, when this process is one.
static SUPERVISOR: Lazy<Mutex<Option<Arc<Supervisor>>>> = Lazy::new(|| Mutex::new(None));
/// Serialises creating and stopping the process-wide supervisor.
static SUPERVISOR_LOCK: Mutex<()> = Mutex::new(());

/// The running supervisor, or `None` when it was explicitly disabled.
pub fn get_supervisor() -> Option<Arc<Supervisor>> {
    SUPERVISOR.lock().clone()
}

/// Stop `project_id`'s agent, if this process is supervising one.
///
/// A no-op in a child and under `--no-supervised`, where there is no pool and
/// the process serving the request is the only one there is.
pub fn release_project(project_id: &str) -> bool {
    match get_supervisor() {
        Some(supervisor) => supervisor.release(project_id),
        None => false,
    }
}

/// Create and start the process-wide supervisor.
pub fn start_supervisor(port: u16, idle_ttl: Option<f64>) -> Arc<Supervisor> {
    let _guard = SUPERVISOR_LOCK.lock();
    if let Some(existing) = get_supervisor() {
        return existing;
    }
    let supervisor = Arc::new(Supervisor::new(None, idle_ttl, None));
    *SUPERVISOR.lock() = Some(Arc::clone(&supervisor));
    supervisor.start(port);
    supervisor
}

/// Stop and forget the process-wide supervisor.
pub fn stop_supervisor() {
    let _guard = SUPERVISOR_LOCK.lock();
    let current = get_supervisor();
    if let Some(supervisor) = current {
        supervisor.stop();
        *SUPERVISOR.lock() = None;
    }
}

/// The pool routing should use, or `None` when not supervising.
pub fn get_pool() -> Option<Arc<ProjectAgentPool>> {
    get_supervisor().and_then(|s| s.pool())
}
