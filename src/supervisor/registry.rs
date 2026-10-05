//! Port of `smartloop/supervisor/registry.py`.
//!
//! Where the supervisor records the processes it is responsible for.
//!
//! One `server.pid` and one `server.port` described a workspace when a
//! workspace was one process.  With a process per project there is a set of
//! them, and something has to be able to answer "what is running, on which
//! port, for which project" — for the supervisor's own routing, for
//! `slp agent status`, and for a restart that has to find the children an
//! earlier supervisor left behind.
//!
//! Written in two places, deliberately:
//!
//! * `~/.smartloop/agents.json` — the supervisor's own view, the file it reads
//!   on startup to adopt or reap what a previous run left;
//! * `<project_dir>/agent.pid` and `<project_dir>/agent.port` — beside the
//!   project they belong to, so a project directory is still self-describing.
//!
//! The registry is advisory.  A pid in it is a claim, not a guarantee — the
//! process may have been killed — so every read that matters is checked against
//! the live process ([`process_alive`]).

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{Map, Value};

#[allow(unused_imports)]
use crate::supervisor::parent_watch::process_started_at;

/// Name of the registry file inside the workspace home.
pub const REGISTRY_FILE: &str = "agents.json";
/// Version stamped into `agents.json`.
pub const REGISTRY_VERSION: i64 = 1;

/// Per-project bookkeeping, beside the project it describes.
pub const AGENT_PID_FILE: &str = "agent.pid";
/// Per-project bookkeeping, beside the project it describes.
pub const AGENT_PORT_FILE: &str = "agent.port";

/// Whether `pid` names a live process.
///
/// Asked through sysinfo (psutil) rather than `kill(pid, 0)`, which on Windows
/// would terminate the process it is asking about.  A zombie does not count as
/// alive; a process we may not inspect does.
pub fn process_alive(pid: Option<i64>) -> bool {
    let _ = pid;
    todo!()
}

/// One running project agent, as the supervisor knows it.
///
/// Shared as `Arc<AgentRecord>` between the registry, the pool and callers, so
/// `last_used_at` is interior-mutable (Python mutated the one shared object).
#[derive(Debug)]
pub struct AgentRecord {
    pub project_id: String,
    pub pid: i64,
    pub port: u16,
    pub started_at: f64,
    last_used_at: Mutex<f64>,
}

impl Clone for AgentRecord {
    fn clone(&self) -> Self {
        Self {
            project_id: self.project_id.clone(),
            pid: self.pid,
            port: self.port,
            started_at: self.started_at,
            last_used_at: Mutex::new(*self.last_used_at.lock()),
        }
    }
}

impl AgentRecord {
    /// A record started now, last used now.
    pub fn new(project_id: impl Into<String>, pid: i64, port: u16) -> Self {
        Self::with_times(project_id, pid, port, None, None)
    }

    /// A record with explicit start / last-used times (`None` → now / started_at).
    pub fn with_times(
        project_id: impl Into<String>,
        pid: i64,
        port: u16,
        started_at: Option<f64>,
        last_used_at: Option<f64>,
    ) -> Self {
        let _ = (project_id, pid, port, started_at, last_used_at);
        todo!()
    }

    /// Where the supervisor forwards this project's requests.
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Whether the recorded process is still there.
    pub fn alive(&self) -> bool {
        process_alive(Some(self.pid))
    }

    /// When this agent last served a request (epoch seconds).
    pub fn last_used_at(&self) -> f64 {
        *self.last_used_at.lock()
    }

    /// Overwrite the last-used time (tests age a record this way).
    pub fn set_last_used_at(&self, value: f64) {
        *self.last_used_at.lock() = value;
    }

    /// Seconds since the agent last served a request.
    pub fn idle_seconds(&self) -> f64 {
        todo!()
    }

    /// Record that this agent just served a request.
    pub fn touch(&self) {
        todo!()
    }

    /// The JSON shape stored under `agents.<project_id>` in `agents.json`.
    pub fn to_dict(&self) -> Map<String, Value> {
        todo!()
    }

    /// Parse a stored record; `None` when `pid`/`port` are missing or malformed.
    pub fn from_dict(project_id: &str, data: &Value) -> Option<AgentRecord> {
        let _ = (project_id, data);
        todo!()
    }
}

#[derive(Debug, Default)]
struct RegistryState {
    agents: HashMap<String, Arc<AgentRecord>>,
    supervisor: Map<String, Value>,
    model_host: Map<String, Value>,
}

/// The set of processes this workspace is running, persisted to disk.
///
/// In-memory state is authoritative for the running supervisor; the file is
/// how a *new* supervisor discovers what an old one left behind, and how
/// `slp agent status` reports without talking to anything.
#[derive(Debug)]
pub struct AgentRegistry {
    pub home_dir: String,
    pub path: String,
    state: Mutex<RegistryState>,
}

impl AgentRegistry {
    /// A registry rooted at `home_dir` (`<home_dir>/agents.json`), empty until loaded.
    pub fn new(home_dir: impl Into<String>) -> Self {
        let _ = home_dir;
        todo!()
    }

    /// The record for `project_id`, if any.
    pub fn get(&self, project_id: &str) -> Option<Arc<AgentRecord>> {
        let _ = project_id;
        todo!()
    }

    /// Every record.
    pub fn all(&self) -> Vec<Arc<AgentRecord>> {
        todo!()
    }

    /// Records whose process is actually still there.
    pub fn live(&self) -> Vec<Arc<AgentRecord>> {
        todo!()
    }

    /// A copy of the supervisor entry (`pid`, `port`, `started_at`).
    pub fn supervisor(&self) -> Map<String, Value> {
        todo!()
    }

    /// A copy of the model-host entry (`pid`, `address`).
    pub fn model_host(&self) -> Map<String, Value> {
        todo!()
    }

    /// Add (or replace) a record, persist, and write the per-project files.
    pub fn add(&self, record: AgentRecord, project_dir: Option<&str>) -> Arc<AgentRecord> {
        let _ = (record, project_dir);
        todo!()
    }

    /// Add an already-shared record.
    pub fn add_shared(&self, record: Arc<AgentRecord>, project_dir: Option<&str>) -> Arc<AgentRecord> {
        let _ = (record, project_dir);
        todo!()
    }

    /// Forget a record, persist, and clear the per-project files.
    pub fn remove(&self, project_id: &str, project_dir: Option<&str>) {
        let _ = (project_id, project_dir);
        todo!()
    }

    /// Record the supervisor, with the start time that identifies it.
    pub fn set_supervisor(&self, pid: i64, port: u16) {
        let _ = (pid, port);
        todo!()
    }

    /// Record the model host's pid and address (either may be `None`).
    pub fn set_model_host(&self, pid: Option<i64>, address: Option<&str>) {
        let _ = (pid, address);
        todo!()
    }

    /// Drop records whose process is gone. Returns the ids dropped.
    pub fn reap(&self) -> Vec<String> {
        todo!()
    }

    /// Write the registry atomically; best-effort (logs a warning on failure).
    pub fn save(&self) {
        todo!()
    }

    /// Read what a previous supervisor left, keeping only live processes.
    pub fn load(self) -> Self {
        self.reload();
        self
    }

    /// In-place variant of [`AgentRegistry::load`].
    pub fn reload(&self) -> &Self {
        todo!()
    }
}

/// Record pid/port beside the project, so its directory tells its own story.
pub fn write_project_files(project_dir: &str, record: &AgentRecord) {
    let _ = (project_dir, record);
    todo!()
}

/// Remove `agent.pid` / `agent.port` beside the project.
pub fn clear_project_files(project_dir: &str) {
    let _ = project_dir;
    todo!()
}

/// `(pid, port)` recorded beside a project, or `None`.
pub fn read_project_agent(project_dir: &str) -> Option<(i64, u16)> {
    let _ = project_dir;
    todo!()
}
