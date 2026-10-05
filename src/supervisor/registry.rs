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
//!   project they belong to, so a project directory is still self-describing
//!   and a human debugging one project does not have to parse the whole
//!   registry.
//!
//! The registry is advisory.  A pid in it is a claim, not a guarantee — the
//! process may have been killed — so every read that matters is checked against
//! the live process ([`process_alive`]).

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use indexmap::IndexMap;
use parking_lot::Mutex;
use serde_json::{Map, Value};
use sysinfo::{Pid, Process, ProcessRefreshKind, ProcessStatus, ProcessesToUpdate, System};

use crate::supervisor::parent_watch::process_started_at;

/// Logger name of this module.
const LOGGER: &str = "smartloop.supervisor.registry";

/// Name of the registry file inside the workspace home.
pub const REGISTRY_FILE: &str = "agents.json";
/// Version stamped into `agents.json`.
pub const REGISTRY_VERSION: i64 = 1;

/// Per-project bookkeeping, beside the project it describes.
pub const AGENT_PID_FILE: &str = "agent.pid";
/// Per-project bookkeeping, beside the project it describes.
pub const AGENT_PORT_FILE: &str = "agent.port";

/// Seconds since the epoch, as a float (`time.time()`).
pub(crate) fn now_epoch() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Look one process up through sysinfo (psutil's `Process(pid)`), and hand it
/// to `inspect`.  `None` when there is no such process (or it cannot be seen).
pub(crate) fn with_process<R>(
    pid: i64,
    refresh: ProcessRefreshKind,
    inspect: impl FnOnce(&Process) -> R,
) -> Option<R> {
    if pid <= 0 || pid > i64::from(u32::MAX) {
        return None;
    }
    let sys_pid = Pid::from_u32(pid as u32);
    let mut system = System::new();
    system.refresh_processes_specifics(ProcessesToUpdate::Some(&[sys_pid]), true, refresh);
    system.process(sys_pid).map(inspect)
}

/// Whether the OS says something holds `pid`, asked without signalling it.
///
/// Only used when sysinfo cannot see the process: on POSIX, `kill(pid, 0)`
/// is a pure question, and `EPERM` means the process is there, just not ours
/// (psutil's `AccessDenied`).
#[cfg(unix)]
fn exists_but_hidden(pid: i64) -> bool {
    if pid <= 0 || pid > i64::from(i32::MAX) {
        return false;
    }
    // SAFETY: signal 0 performs only the existence and permission checks.
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    rc != 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
fn exists_but_hidden(_pid: i64) -> bool {
    false
}

/// Whether `pid` names a live process.
///
/// Asked through sysinfo (psutil) rather than the usual `kill(pid, 0)`.  The
/// signal-0 trick is POSIX: on Windows the equivalent would *terminate the
/// process it is asking about* — and this is called from `slp agent status`.
///
/// A zombie does not count as alive.  On POSIX a child nobody has waited on
/// still answers signal 0 after it dies, and treating that as running would
/// have the supervisor keep forwarding requests to a port with nothing behind
/// it.  A process we may not inspect does count: access denied means it is
/// there, just not ours.
pub fn process_alive(pid: Option<i64>) -> bool {
    let pid = match pid {
        Some(pid) if pid > 0 => pid,
        _ => return false,
    };
    match with_process(pid, ProcessRefreshKind::nothing(), |p| p.status()) {
        Some(status) => status != ProcessStatus::Zombie,
        None => exists_but_hidden(pid),
    }
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
        let started_at = started_at.unwrap_or_else(now_epoch);
        let last_used_at = last_used_at.unwrap_or(started_at);
        Self {
            project_id: project_id.into(),
            pid,
            port,
            started_at,
            last_used_at: Mutex::new(last_used_at),
        }
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
        (now_epoch() - self.last_used_at()).max(0.0)
    }

    /// Record that this agent just served a request.
    pub fn touch(&self) {
        *self.last_used_at.lock() = now_epoch();
    }

    /// The JSON shape stored under `agents.<project_id>` in `agents.json`.
    pub fn to_dict(&self) -> Map<String, Value> {
        let mut map = Map::new();
        map.insert("pid".into(), Value::from(self.pid));
        map.insert("port".into(), Value::from(self.port));
        map.insert("started_at".into(), float_value(self.started_at));
        map.insert("last_used_at".into(), float_value(self.last_used_at()));
        map
    }

    /// Parse a stored record; `None` when `pid`/`port` are missing or malformed.
    pub fn from_dict(project_id: &str, data: &Value) -> Option<AgentRecord> {
        let object = data.as_object()?;
        let pid = py_int(object.get("pid")?)?;
        let port = u16::try_from(py_int(object.get("port")?)?).ok()?;
        let started_at = object.get("started_at").and_then(Value::as_f64);
        let last_used_at = object.get("last_used_at").and_then(Value::as_f64);
        Some(AgentRecord::with_times(project_id, pid, port, started_at, last_used_at))
    }
}

/// A float as JSON (`null` for a non-finite value, which JSON cannot carry).
fn float_value(value: f64) -> Value {
    serde_json::Number::from_f64(value).map(Value::Number).unwrap_or(Value::Null)
}

/// Python's `int(value)` over a JSON value: integers, truncated floats,
/// booleans and numeric strings; `None` for anything `int()` would reject.
pub(crate) fn py_int(value: &Value) -> Option<i64> {
    match value {
        Value::Number(n) => n.as_i64().or_else(|| {
            n.as_f64().filter(|f| f.is_finite()).map(|f| f.trunc() as i64)
        }),
        Value::Bool(b) => Some(i64::from(*b)),
        Value::String(s) => parse_py_int(s),
        _ => None,
    }
}

/// Python's `int(text)`: surrounding whitespace, a sign, and `_` between digits.
pub(crate) fn parse_py_int(text: &str) -> Option<i64> {
    let trimmed = text.trim();
    let (sign, digits) = match trimmed.strip_prefix('-') {
        Some(rest) => (-1i64, rest),
        None => (1i64, trimmed.strip_prefix('+').unwrap_or(trimmed)),
    };
    if digits.is_empty()
        || digits.starts_with('_')
        || digits.ends_with('_')
        || digits.contains("__")
        || !digits.chars().all(|c| c.is_ascii_digit() || c == '_')
    {
        return None;
    }
    let cleaned: String = digits.chars().filter(|c| *c != '_').collect();
    cleaned.parse::<i64>().ok().map(|v| sign * v)
}

#[derive(Debug, Default)]
struct RegistryState {
    agents: IndexMap<String, Arc<AgentRecord>>,
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
        let home_dir = home_dir.into();
        let path = Path::new(&home_dir).join(REGISTRY_FILE).to_string_lossy().into_owned();
        Self { home_dir, path, state: Mutex::new(RegistryState::default()) }
    }

    // -- accessors ----------------------------------------------------------

    /// The record for `project_id`, if any.
    pub fn get(&self, project_id: &str) -> Option<Arc<AgentRecord>> {
        self.state.lock().agents.get(project_id).cloned()
    }

    /// Every record.
    pub fn all(&self) -> Vec<Arc<AgentRecord>> {
        self.state.lock().agents.values().cloned().collect()
    }

    /// Records whose process is actually still there.
    pub fn live(&self) -> Vec<Arc<AgentRecord>> {
        self.all().into_iter().filter(|record| record.alive()).collect()
    }

    /// A copy of the supervisor entry (`pid`, `port`, `started_at`).
    pub fn supervisor(&self) -> Map<String, Value> {
        self.state.lock().supervisor.clone()
    }

    /// A copy of the model-host entry (`pid`, `address`).
    pub fn model_host(&self) -> Map<String, Value> {
        self.state.lock().model_host.clone()
    }

    // -- mutation -----------------------------------------------------------

    /// Add (or replace) a record, persist, and write the per-project files.
    pub fn add(&self, record: AgentRecord, project_dir: Option<&str>) -> Arc<AgentRecord> {
        self.add_shared(Arc::new(record), project_dir)
    }

    /// Add an already-shared record.
    pub fn add_shared(&self, record: Arc<AgentRecord>, project_dir: Option<&str>) -> Arc<AgentRecord> {
        {
            let mut state = self.state.lock();
            state.agents.insert(record.project_id.clone(), Arc::clone(&record));
            self.save_state(&state);
        }
        if let Some(dir) = project_dir.filter(|d| !d.is_empty()) {
            write_project_files(dir, &record);
        }
        record
    }

    /// Forget a record, persist, and clear the per-project files.
    pub fn remove(&self, project_id: &str, project_dir: Option<&str>) {
        {
            let mut state = self.state.lock();
            state.agents.shift_remove(project_id);
            self.save_state(&state);
        }
        if let Some(dir) = project_dir.filter(|d| !d.is_empty()) {
            clear_project_files(dir);
        }
    }

    /// Record the supervisor, with the start time that identifies it.
    ///
    /// `started_at` is here for the children: a child watching its parent has
    /// to tell a live supervisor from a recycled pid, and a successor it finds
    /// in this file is a process it has never seen.  Without the start time an
    /// orphan whose pid was reused reads as supervised and stays up forever.
    /// See [`crate::supervisor::parent_watch`].
    pub fn set_supervisor(&self, pid: i64, port: u16) {
        let started_at = process_started_at(Some(pid));
        let mut state = self.state.lock();
        let mut entry = Map::new();
        entry.insert("pid".into(), Value::from(pid));
        entry.insert("port".into(), Value::from(port));
        entry.insert("started_at".into(), started_at.map(float_value).unwrap_or(Value::Null));
        state.supervisor = entry;
        self.save_state(&state);
    }

    /// Record the model host's pid and address (either may be `None`).
    pub fn set_model_host(&self, pid: Option<i64>, address: Option<&str>) {
        let mut state = self.state.lock();
        let mut entry = Map::new();
        entry.insert("pid".into(), pid.map(Value::from).unwrap_or(Value::Null));
        entry.insert("address".into(), address.map(Value::from).unwrap_or(Value::Null));
        state.model_host = entry;
        self.save_state(&state);
    }

    /// Drop records whose process is gone. Returns the ids dropped.
    pub fn reap(&self) -> Vec<String> {
        let mut state = self.state.lock();
        let dead: Vec<String> = state
            .agents
            .iter()
            .filter(|(_, record)| !record.alive())
            .map(|(id, _)| id.clone())
            .collect();
        for project_id in &dead {
            state.agents.shift_remove(project_id);
        }
        if !dead.is_empty() {
            self.save_state(&state);
        }
        dead
    }

    // -- persistence --------------------------------------------------------

    /// Write the registry, atomically.
    ///
    /// Best-effort: the registry is a convenience, and a workspace on a
    /// read-only or full filesystem should still serve requests rather than
    /// fail on bookkeeping.
    pub fn save(&self) {
        let state = self.state.lock();
        self.save_state(&state);
    }

    fn save_state(&self, state: &RegistryState) {
        let mut agents = Map::new();
        for (project_id, record) in &state.agents {
            agents.insert(project_id.clone(), Value::Object(record.to_dict()));
        }
        let mut payload = Map::new();
        payload.insert("version".into(), Value::from(REGISTRY_VERSION));
        payload.insert("supervisor".into(), Value::Object(state.supervisor.clone()));
        payload.insert("model_host".into(), Value::Object(state.model_host.clone()));
        payload.insert("agents".into(), Value::Object(agents));

        if let Err(exc) = self.write_atomically(&Value::Object(payload)) {
            tracing::warn!(target: LOGGER, "[supervisor] could not write {}: {}", self.path, exc);
        }
    }

    fn write_atomically(&self, payload: &Value) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.home_dir)?;
        let tmp = format!("{}.tmp", self.path);
        {
            let mut fh = std::fs::File::create(&tmp)?;
            let text = serde_json::to_string_pretty(payload)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            fh.write_all(text.as_bytes())?;
            fh.flush()?;
            fh.sync_all()?;
        }
        std::fs::rename(&tmp, &self.path)
    }

    /// Read what a previous supervisor left, keeping only live processes.
    pub fn load(self) -> Self {
        self.reload();
        self
    }

    /// In-place variant of [`AgentRegistry::load`].
    ///
    /// Dead records are dropped rather than reported: a pid file outliving its
    /// process is the normal outcome of a crash, and adopting one would have
    /// the supervisor route requests into nothing.
    pub fn reload(&self) -> &Self {
        let data: Value = match std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
        {
            Some(data) => data,
            None => return self,
        };
        let Some(data) = data.as_object() else {
            return self;
        };

        let object_or_empty = |key: &str| -> Map<String, Value> {
            data.get(key).and_then(Value::as_object).cloned().unwrap_or_default()
        };

        let mut state = self.state.lock();
        state.supervisor = object_or_empty("supervisor");
        state.model_host = object_or_empty("model_host");
        state.agents = IndexMap::new();
        for (project_id, raw) in object_or_empty("agents") {
            let raw = if raw.is_null() { Value::Object(Map::new()) } else { raw };
            match AgentRecord::from_dict(&project_id, &raw) {
                Some(record) if record.alive() => {
                    state.agents.insert(project_id, Arc::new(record));
                }
                Some(record) => {
                    tracing::info!(
                        target: LOGGER,
                        "[supervisor] project {}: pid {} is gone, dropping it",
                        project_id,
                        record.pid
                    );
                }
                None => {}
            }
        }
        drop(state);
        self
    }
}

/// Record pid/port beside the project, so its directory tells its own story.
pub fn write_project_files(project_dir: &str, record: &AgentRecord) {
    let result = (|| -> std::io::Result<()> {
        std::fs::create_dir_all(project_dir)?;
        std::fs::write(Path::new(project_dir).join(AGENT_PID_FILE), record.pid.to_string())?;
        std::fs::write(Path::new(project_dir).join(AGENT_PORT_FILE), record.port.to_string())?;
        Ok(())
    })();
    if let Err(exc) = result {
        tracing::warn!(
            target: LOGGER,
            "[supervisor] could not write agent files in {}: {}",
            project_dir,
            exc
        );
    }
}

/// Remove `agent.pid` / `agent.port` beside the project.
pub fn clear_project_files(project_dir: &str) {
    for name in [AGENT_PID_FILE, AGENT_PORT_FILE] {
        let _ = std::fs::remove_file(Path::new(project_dir).join(name));
    }
}

/// `(pid, port)` recorded beside a project, or `None`.
///
/// Used by tooling that has a project directory and no registry — and as the
/// cross-check that the registry and the project agree.
pub fn read_project_agent(project_dir: &str) -> Option<(i64, u16)> {
    let read = |name: &str| std::fs::read_to_string(Path::new(project_dir).join(name)).ok();
    let pid = parse_py_int(&read(AGENT_PID_FILE)?)?;
    let port = u16::try_from(parse_py_int(&read(AGENT_PORT_FILE)?)?).ok()?;
    Some((pid, port))
}
