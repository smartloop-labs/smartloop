//! Port of `smartloop/supervisor/pool.py`.
//!
//! Spawning, watching and reaping the per-project agent processes.
//!
//! The isolation payoff lives here.  When one process served every project, a
//! runaway index job, a wedged MCP server or an OOM took down all of them,
//! because there was one task queue, one process and one `server.pid`.  A
//! child per project turns each of those into one project's problem: the
//! supervisor notices, restarts that child with backoff, and the others never
//! knew.
//!
//! Lifecycle:
//!
//! * **lazy start** — a child is spawned on the first request that names its
//!   project, so an install with ten projects does not pay for ten processes
//!   to answer a question about one;
//! * **idle stop** — a child that has served nothing for
//!   [`DEFAULT_IDLE_TTL_SECONDS`] is stopped, returning its baseline RSS;
//! * **restart on crash** — with exponential backoff, and the backoff reset
//!   once a child has been up long enough to count as healthy.  The policy is
//!   lifted from the single-process foreground server, which has always done
//!   exactly this; here it is applied per child.

use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

use crate::supervisor::parent_watch::{join_with_timeout, parent_environment, process_started_at, StopEvent};
use crate::supervisor::registry::{now_epoch, process_alive, read_project_agent, with_process, AgentRecord, AgentRegistry};
use crate::supervisor::spawn::{popen_detached_devnull, ChildProcess, WaitError};
use crate::utils::timing;

/// Logger name of this module.
const LOGGER: &str = "smartloop.supervisor.pool";

/// How long a child may sit unused before it is stopped.
pub const DEFAULT_IDLE_TTL_SECONDS: f64 = 900.0;

/// How long to wait for a freshly spawned child to answer /health.
pub const DEFAULT_START_TIMEOUT_SECONDS: f64 = 120.0;

/// Restart backoff, matching the single-process policy it is lifted from.
pub const BACKOFF_START: f64 = 1.0;
/// Upper bound of the restart backoff.
pub const BACKOFF_MAX: f64 = 60.0;
/// Uptime after which a crash resets the backoff.
pub const BACKOFF_RESET_AFTER: f64 = 120.0;

/// Per-child resident-size ceiling, in bytes. 0 (the default) means no ceiling.
///
/// A ceiling is the other half of the isolation: a crash is contained by having
/// separate processes, but a project that leaks or over-indexes would still
/// take the machine down with it.  Stopping that one child returns the memory
/// and leaves every other project serving; the next request starts it again.
pub const DEFAULT_MEMORY_LIMIT_BYTES: u64 = 0;

/// The module the child is launched as (`<exe> -m <module>`).
pub const PROJECT_AGENT_MODULE: &str = "smartloop.api.project_agent";

/// A project agent could not be started, or never became healthy.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct ChildStartError(pub String);

/// `project_id -> project_dir` lookup the pool is given.
pub type ProjectDirFor = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;
/// `project_id -> argv` used to launch a child.
pub type CommandFor = Arc<dyn Fn(&str) -> Vec<String> + Send + Sync>;
/// `project_id -> environment` override (tests replace `_environment` this way).
pub type EnvironmentFor = Arc<dyn Fn(&str) -> HashMap<String, String> + Send + Sync>;
/// `pid -> resident bytes` override (tests replace `process_rss` this way).
pub type RssFor = Arc<dyn Fn(Option<i64>) -> Option<u64> + Send + Sync>;
/// Replacement for the `/health` wait (tests replace `_await_health` this way).
/// Receives the project id, the child's handle and the record being started.
pub type AwaitHealthHook =
    Arc<dyn Fn(&str, &Mutex<ChildProcess>, &AgentRecord) -> Result<(), ChildStartError> + Send + Sync>;

/// A child handle shared between the pool's map and a start in progress.
type SharedChild = Arc<Mutex<ChildProcess>>;

/// What one [`ProjectAgentPool::sweep`] did.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct SweepResult {
    pub crashed: Vec<String>,
    pub idled: Vec<String>,
    pub over_limit: Vec<String>,
}

/// Every descendant of `pid` (children, grandchildren, ...), from one snapshot
/// of the process table.
fn descendants(pid: i64) -> Vec<i64> {
    if pid <= 0 {
        return Vec::new();
    }
    let mut system = System::new();
    system.refresh_processes_specifics(ProcessesToUpdate::All, true, ProcessRefreshKind::nothing());
    let mut by_parent: HashMap<Pid, Vec<Pid>> = HashMap::new();
    for (child, process) in system.processes() {
        if let Some(parent) = process.parent() {
            by_parent.entry(parent).or_default().push(*child);
        }
    }
    let mut found = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![Pid::from_u32(pid as u32)];
    while let Some(current) = stack.pop() {
        for child in by_parent.get(&current).into_iter().flatten() {
            if seen.insert(*child) {
                found.push(child.as_u32() as i64);
                stack.push(*child);
            }
        }
    }
    found
}

/// Whether `candidate_pid` is the process we spawned, or one it launched.
///
/// Some interpreters on Windows are launchers: they start the real program as
/// a child and simply wait on it, so the spawned pid names the launcher while
/// the project agent's own pid — the one it publishes — names the grandchild.
/// Direct equality is the common case; walking descendants covers that one,
/// without which a launcher-style child can never pass this check and every
/// cold start times out and retries forever.
fn is_own_pid(launcher_pid: i64, candidate_pid: i64) -> bool {
    if launcher_pid == candidate_pid {
        return true;
    }
    descendants(launcher_pid).contains(&candidate_pid)
}

/// Snapshot of `pid`'s live descendant pids, before anything might kill it.
///
/// Has to be taken before termination, not after: once a process has exited,
/// nobody can be asked what its children were, and the whole point of this is
/// to find a real project agent process still running under a launcher after
/// that launcher is gone.
fn live_children(pid: i64) -> Vec<i64> {
    descendants(pid).into_iter().filter(|child| process_alive(Some(*child))).collect()
}

/// Kill `pid` outright, portably.
fn kill_pid(pid: i64) {
    #[cfg(unix)]
    {
        let _ = crate::supervisor::spawn::send_signal(pid, libc::SIGKILL);
    }
    #[cfg(not(unix))]
    {
        let _ = with_process(pid, ProcessRefreshKind::nothing(), |p| p.kill());
    }
}

/// Ask `pid` to stop, portably (SIGTERM; TerminateProcess on Windows).
fn terminate_signal(pid: i64) -> bool {
    #[cfg(unix)]
    {
        crate::supervisor::spawn::send_signal(pid, libc::SIGTERM).is_ok()
    }
    #[cfg(not(unix))]
    {
        with_process(pid, ProcessRefreshKind::nothing(), |p| p.kill()).unwrap_or(false)
    }
}

/// Wait until `pid` is gone (or a zombie), up to `timeout` seconds.
fn wait_gone(pid: i64, timeout: f64) -> bool {
    let deadline = Instant::now() + Duration::from_secs_f64(timeout.max(0.0));
    loop {
        if !process_alive(Some(pid)) {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Best-effort kill of every process in `pids`, waited on to completion.
///
/// Terminating a handle reaches only the process we hold it for.  On Windows
/// nothing propagates that to a child a launcher-style interpreter started, so
/// without this the real project agent process is orphaned rather than
/// stopped whenever the launcher and the worker are different pids.  The wait
/// matters as much as the kill: without it, a caller checking liveness right
/// after `stop()` returns can still catch the process before it is reaped.
fn kill_all(pids: &[i64]) {
    for pid in pids {
        if !process_alive(Some(*pid)) {
            continue;
        }
        kill_pid(*pid);
        wait_gone(*pid, 5.0);
    }
}

/// The `returncode` as Python formats it (`None` until reaped).
fn format_returncode(code: Option<i32>) -> String {
    code.map_or_else(|| "None".to_string(), |c| c.to_string())
}

/// The set of running project agents, and the policy for keeping them.
///
/// Every method that spawns or stops takes the per-project lock, so two
/// concurrent first-requests for the same cold project start one child rather
/// than two.  Requests to *different* projects do not contend for anything but
/// the brief bookkeeping.
pub struct ProjectAgentPool {
    pub registry: Arc<AgentRegistry>,
    pub project_dir_for: ProjectDirFor,
    pub idle_ttl: f64,
    pub start_timeout: f64,
    pub model_host_address: Option<String>,
    pub memory_limit_bytes: u64,
    /// Replaces [`process_rss`] for the memory ceiling when set (test seam).
    pub process_rss_for: Option<RssFor>,
    command_for: Option<CommandFor>,
    environment_for: Option<EnvironmentFor>,
    await_health_hook: Option<AwaitHealthHook>,
    processes: Mutex<IndexMap<String, SharedChild>>,
    backoff: Mutex<HashMap<String, f64>>,
    locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    stopping: Arc<StopEvent>,
    reaper: Mutex<Option<JoinHandle<()>>>,
}

/// Constructor options for [`ProjectAgentPool`] (Python keyword arguments).
#[derive(Clone)]
pub struct PoolOptions {
    pub idle_ttl: f64,
    pub start_timeout: f64,
    pub model_host_address: Option<String>,
    pub command_for: Option<CommandFor>,
    pub environment_for: Option<EnvironmentFor>,
    pub memory_limit_bytes: u64,
    /// Replacement for the `/health` wait (test seam for `_await_health`).
    pub await_health: Option<AwaitHealthHook>,
}

impl Default for PoolOptions {
    fn default() -> Self {
        Self {
            idle_ttl: DEFAULT_IDLE_TTL_SECONDS,
            start_timeout: DEFAULT_START_TIMEOUT_SECONDS,
            model_host_address: None,
            command_for: None,
            environment_for: None,
            memory_limit_bytes: DEFAULT_MEMORY_LIMIT_BYTES,
            await_health: None,
        }
    }
}

impl ProjectAgentPool {
    /// A pool over `registry`, with defaults for everything else.
    pub fn new(registry: Arc<AgentRegistry>, project_dir_for: ProjectDirFor) -> Self {
        Self::with_options(registry, project_dir_for, PoolOptions::default())
    }

    /// A pool with explicit options.
    pub fn with_options(
        registry: Arc<AgentRegistry>,
        project_dir_for: ProjectDirFor,
        options: PoolOptions,
    ) -> Self {
        Self {
            registry,
            project_dir_for,
            idle_ttl: options.idle_ttl,
            start_timeout: options.start_timeout,
            model_host_address: options.model_host_address,
            memory_limit_bytes: options.memory_limit_bytes,
            process_rss_for: None,
            // How a child is launched.  Overridable so the lifecycle can be
            // exercised against a process that does not load a model.
            command_for: options.command_for,
            environment_for: options.environment_for,
            await_health_hook: options.await_health,
            processes: Mutex::new(IndexMap::new()),
            backoff: Mutex::new(HashMap::new()),
            locks: Mutex::new(HashMap::new()),
            stopping: Arc::new(StopEvent::default()),
            reaper: Mutex::new(None),
        }
    }

    // -- locking ------------------------------------------------------------

    /// One lock per project, so cold starts never serialise across projects.
    fn project_lock(&self, project_id: &str) -> Arc<Mutex<()>> {
        let mut locks = self.locks.lock();
        Arc::clone(locks.entry(project_id.to_string()).or_default())
    }

    fn handle(&self, project_id: &str) -> Option<SharedChild> {
        self.processes.lock().get(project_id).cloned()
    }

    // -- starting -----------------------------------------------------------

    /// The agent for `project_id` if it is genuinely serving, else `None`.
    ///
    /// A child we spawned and have not waited on stays a zombie after it dies,
    /// so the pid alone would say "alive" forever and the supervisor would
    /// keep forwarding requests to a port nothing is listening on.  Polling
    /// the handle we own settles it and reaps the zombie at the same time.  A
    /// record adopted from an earlier supervisor has no handle here, and for
    /// that one the pid is all there is.
    pub fn running(&self, project_id: &str) -> Option<Arc<AgentRecord>> {
        let record = self.registry.get(project_id)?;
        if let Some(process) = self.handle(project_id) {
            // Both signals have to agree, because for a moment after a kill
            // they disagree: the process can already be gone from the process
            // table while `waitpid` has not yet reported it exited.  Trusting
            // the handle alone leaves a window in which the supervisor
            // forwards a request to a port with nothing behind it.
            if process.lock().poll().is_some() {
                return None;
            }
        }
        if record.alive() { Some(record) } else { None }
    }

    /// Return a live agent for `project_id`, starting one if needed (blocking).
    pub fn ensure(&self, project_id: &str) -> Result<Arc<AgentRecord>, ChildStartError> {
        if let Some(record) = self.running(project_id) {
            record.touch();
            return Ok(record);
        }

        let lock = self.project_lock(project_id);
        let _guard = lock.lock();
        // Re-check: another request may have started it while we waited,
        // which is the normal case when a cold project gets two requests at
        // once.
        if let Some(record) = self.running(project_id) {
            record.touch();
            return Ok(record);
        }
        self.start(project_id)
    }

    /// Where a Windows child's raw stdout/stderr go: a crash traceback printed
    /// before its logging is up, beside its `agent.log`.  A file of its own,
    /// because `agent.log` rotates and Windows will not rename a file another
    /// process holds open.
    fn output_path(&self, project_id: &str) -> Option<String> {
        let project_dir = (self.project_dir_for)(project_id).filter(|d| !d.is_empty())?;
        let path = std::path::Path::new(&project_dir);
        if !path.is_dir() {
            return None;
        }
        Some(path.join("agent.out.log").to_string_lossy().into_owned())
    }

    /// Spawn a child and wait for it to answer. Caller holds the project lock.
    fn start(&self, project_id: &str) -> Result<Arc<AgentRecord>, ChildStartError> {
        self.await_backoff(project_id);

        let command = match &self.command_for {
            Some(command_for) => command_for(project_id),
            None => Self::default_command(project_id),
        };
        let env = self.environment(project_id);
        tracing::info!(target: LOGGER, "[supervisor] starting project agent for {}", project_id);
        let _cold_start = timing::phase_with("supervisor.cold_start", serde_json::json!({"project": project_id}));

        // Detach from our console/process group so a Ctrl-C in the terminal
        // reaches the supervisor only — it then stops the children itself, in
        // order, after the model host they use.  Detachment and the descriptor
        // hygiene come from spawn.rs, which uses posix_spawn so no fork() is
        // taken: a fork() while the host's gRPC server is mid-RPC is exactly
        // the class of crash this pool is built to contain.
        let process = popen_detached_devnull(&command, Some(&env), self.output_path(project_id).as_deref())
            .map_err(|exc| {
                ChildStartError(format!("could not spawn an agent for project {project_id}: {exc}"))
            })?;
        let spawned_pid = process.pid();
        let process: SharedChild = Arc::new(Mutex::new(process));

        // Own the handle from the moment it exists.  Recording it only after
        // the child was healthy left every failure in between — a start
        // timeout, an error, a panic through this frame — holding the one
        // reference to a detached process, which is exactly how an agent ends
        // up reparented to init and running for hours.
        self.processes.lock().insert(project_id.to_string(), Arc::clone(&process));

        let project_dir = (self.project_dir_for)(project_id);
        let outcome = catch_unwind(AssertUnwindSafe(|| -> Result<AgentRecord, ChildStartError> {
            let (real_pid, port) = {
                let _t = timing::phase("supervisor.await_port");
                self.await_port(project_id, &process, project_dir.as_deref())?
            };
            let record = AgentRecord::new(project_id, real_pid, port);
            let _t = timing::phase("supervisor.await_health");
            match &self.await_health_hook {
                Some(hook) => hook(project_id, &process, &record)?,
                None => self.await_health(project_id, &process, &record)?,
            }
            Ok(record)
        }));

        let record = match outcome {
            Ok(Ok(record)) => record,
            Ok(Err(error)) => {
                self.processes.lock().shift_remove(project_id);
                self.terminate(&process, 5.0);
                return Err(error);
            }
            Err(panic) => {
                self.processes.lock().shift_remove(project_id);
                self.terminate(&process, 5.0);
                resume_unwind(panic);
            }
        };

        let port = record.port;
        let record = self.registry.add(record, project_dir.as_deref());
        tracing::info!(
            target: LOGGER,
            "[supervisor] project {} ready on port {} (pid {})",
            project_id,
            port,
            spawned_pid
        );
        Ok(record)
    }

    /// How to launch a child: re-exec this binary as a project agent.
    ///
    /// `current_exe()` (or `SLP_EXE`) with `-m smartloop.api.project_agent
    /// --project <id>` — the binary honours `-m` the way `python -m` does,
    /// precisely so it can re-exec itself.
    pub fn default_command(project_id: &str) -> Vec<String> {
        let exe = std::env::var("SLP_EXE")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .or_else(|| {
                std::env::current_exe().ok().map(|p| p.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "slp".to_string());
        vec![
            exe,
            "-m".to_string(),
            PROJECT_AGENT_MODULE.to_string(),
            "--project".to_string(),
            project_id.to_string(),
        ]
    }

    /// The child's environment: ours, plus project id, parent link, model host
    /// and `SLP_SUPERVISOR=0`.  Honours an injected `environment_for`.
    pub fn environment(&self, project_id: &str) -> HashMap<String, String> {
        match &self.environment_for {
            Some(environment_for) => environment_for(project_id),
            None => self.default_environment(project_id),
        }
    }

    /// The real pool environment, ignoring any injected override.
    pub fn default_environment(&self, project_id: &str) -> HashMap<String, String> {
        let mut env: HashMap<String, String> = std::env::vars_os()
            .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
            .collect();
        env.insert("SLP_PROJECT_ID".into(), project_id.to_string());
        // Detaching the child is what keeps Ctrl-C off it; this is what stops
        // that detachment from outliving us.  A child that finds no supervisor
        // behind this pid — and no replacement in the registry — shuts itself
        // down rather than serving a project nobody can reach.
        env.extend(parent_environment());
        if let Some(address) = self.model_host_address.as_deref().filter(|a| !a.is_empty()) {
            env.insert("SLP_MODEL_HOST".into(), address.to_string());
            // The whole point of the host is that the weights are loaded once.
            // A child inherits `preload_model` from the supervisor, which is on
            // by default — leaving it would have every child pull its own copy
            // into the same device.
            env.insert("SLP_API_PRELOAD_MODEL".into(), "0".into());
        }
        // A child must never try to supervise: it serves one project directly.
        // Set explicitly rather than unset — supervision is the default, so an
        // absent variable would mean "on" and the child would spawn a
        // grandchild for the project it already is.
        env.insert("SLP_SUPERVISOR".into(), "0".into());
        env
    }

    /// Wait for the child to publish `(pid, port)` beside its project.
    ///
    /// The child writes `agent.port` beside its project as soon as it has one,
    /// before its server finishes starting, so this resolves well before the
    /// model is loaded.
    ///
    /// Returns `(pid, port)` rather than just the port: with a launcher-style
    /// interpreter the spawned pid names the launcher while the project
    /// agent's own pid — the one it publishes — names the grandchild.  The
    /// caller needs that real pid to track and stop the right process.
    fn await_port(
        &self,
        project_id: &str,
        process: &Mutex<ChildProcess>,
        project_dir: Option<&str>,
    ) -> Result<(i64, u16), ChildStartError> {
        let Some(project_dir) = project_dir.filter(|d| !d.is_empty()) else {
            return Err(ChildStartError(format!(
                "project {project_id} has no directory to read its port from"
            )));
        };
        let deadline = Instant::now() + Duration::from_secs_f64(self.start_timeout.max(0.0));
        while Instant::now() < deadline {
            let (exited, launcher_pid) = {
                let mut child = process.lock();
                (child.poll().is_some(), child.pid())
            };
            if exited {
                return Err(ChildStartError(format!(
                    "agent for project {project_id} exited with code {} before binding a port",
                    format_returncode(process.lock().returncode())
                )));
            }
            if let Some(published) = read_project_agent(project_dir) {
                if is_own_pid(launcher_pid, published.0) {
                    return Ok(published);
                }
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        Self::kill(process);
        Err(ChildStartError(format!(
            "agent for project {project_id} did not publish a port in {:.0}s",
            self.start_timeout
        )))
    }

    /// Wait until the child answers `/health`.
    fn await_health(
        &self,
        project_id: &str,
        process: &Mutex<ChildProcess>,
        record: &AgentRecord,
    ) -> Result<(), ChildStartError> {
        let deadline = Instant::now() + Duration::from_secs_f64(self.start_timeout.max(0.0));
        let url = format!("{}/health", record.base_url());
        while Instant::now() < deadline {
            let exited = process.lock().poll().is_some();
            if exited {
                return Err(ChildStartError(format!(
                    "agent for project {project_id} exited with code {} during startup",
                    format_returncode(process.lock().returncode())
                )));
            }
            if http_get_status(&url, 2.0) == Some(200) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        Self::kill(process);
        Err(ChildStartError(format!(
            "agent for project {project_id} was not healthy within {:.0}s",
            self.start_timeout
        )))
    }

    // -- backoff ------------------------------------------------------------

    /// Hold off restarting a project that just failed.
    ///
    /// Without this, a child that crashes on startup — a corrupt index, an MCP
    /// server that kills the process — would be respawned as fast as requests
    /// arrive.
    fn await_backoff(&self, project_id: &str) {
        let delay = self.backoff.lock().get(project_id).copied().unwrap_or(0.0);
        if delay > 0.0 {
            tracing::warn!(
                target: LOGGER,
                "[supervisor] project {}: waiting {:.0}s before restarting",
                project_id,
                delay
            );
            std::thread::sleep(Duration::from_secs_f64(delay));
        }
    }

    /// Record that a child died, and lengthen or reset its backoff.
    pub fn note_exit(&self, project_id: &str, uptime: f64) {
        let mut backoff = self.backoff.lock();
        if uptime >= BACKOFF_RESET_AFTER {
            backoff.insert(project_id.to_string(), BACKOFF_START);
        } else {
            let current = backoff.get(project_id).copied().unwrap_or(0.0);
            backoff.insert(project_id.to_string(), (current * 2.0).max(BACKOFF_START).min(BACKOFF_MAX));
        }
    }

    /// Forget a project's restart backoff.
    pub fn clear_backoff(&self, project_id: &str) {
        self.backoff.lock().remove(project_id);
    }

    /// The current backoff delay for `project_id`, if any (Python `_backoff.get`).
    pub fn backoff(&self, project_id: &str) -> Option<f64> {
        self.backoff.lock().get(project_id).copied()
    }

    /// Whether the pool holds a process handle for `project_id`.
    pub fn has_process(&self, project_id: &str) -> bool {
        self.processes.lock().contains_key(project_id)
    }

    /// Project ids with a held process handle (Python `_processes` keys).
    pub fn process_ids(&self) -> Vec<String> {
        self.processes.lock().keys().cloned().collect()
    }

    // -- stopping -----------------------------------------------------------

    /// Stop one project's agent. Returns whether there was one to stop.
    pub fn stop(&self, project_id: &str, timeout: f64) -> bool {
        let lock = self.project_lock(project_id);
        let _guard = lock.lock();
        let record = self.registry.get(project_id);
        let process = self.processes.lock().shift_remove(project_id);
        self.registry.remove(project_id, (self.project_dir_for)(project_id).as_deref());
        if let Some(process) = process {
            self.terminate(&process, timeout);
            return true;
        }
        if let Some(record) = record {
            if record.alive() {
                // Adopted from a previous supervisor: we have a pid but no
                // handle, so signal it directly.
                terminate_pid(record.pid, timeout);
                return true;
            }
        }
        false
    }

    /// Stop every child. Called when the supervisor itself shuts down.
    pub fn stop_all(&self, timeout: f64) {
        self.stopping.set();
        let project_ids: Vec<String> = self.registry.all().iter().map(|r| r.project_id.clone()).collect();
        for project_id in project_ids {
            if catch_unwind(AssertUnwindSafe(|| self.stop(&project_id, timeout))).is_err() {
                tracing::error!(target: LOGGER, "[supervisor] failed to stop project {}", project_id);
            }
        }

        // Anything we spawned but never registered — a child that was still
        // starting when the shutdown arrived.  Driven off the handles rather
        // than the registry, because the registry only learns about a child
        // once it is healthy, and a half-started one is just as detached.
        let stragglers: Vec<(String, SharedChild)> = self.processes.lock().drain(..).collect();
        for (project_id, process) in stragglers {
            let pid = process.lock().pid();
            tracing::info!(
                target: LOGGER,
                "[supervisor] stopping unregistered agent for {} (pid {})",
                project_id,
                pid
            );
            if catch_unwind(AssertUnwindSafe(|| self.terminate(&process, timeout))).is_err() {
                tracing::error!(target: LOGGER, "[supervisor] failed to stop pid {}", pid);
            }
        }
    }

    /// SIGTERM, wait, then kill; and kill any launcher descendants.
    fn terminate(&self, process: &Mutex<ChildProcess>, timeout: f64) {
        let pid = process.lock().pid();
        let descendants = live_children(pid);
        if process.lock().poll().is_some() {
            kill_all(&descendants);
            return;
        }
        let result = {
            let mut child = process.lock();
            child.terminate().map_err(WaitError::from).and_then(|_| child.wait(Some(timeout)))
        };
        match result {
            Ok(_) => {}
            Err(WaitError::Timeout(_)) => {
                tracing::warn!(target: LOGGER, "[supervisor] pid {} ignored SIGTERM; killing it", pid);
                Self::kill(process);
            }
            Err(WaitError::Io(_)) => {}
        }
        kill_all(&descendants);
    }

    /// Kill a child and its descendants outright.
    fn kill(process: &Mutex<ChildProcess>) {
        let pid = process.lock().pid();
        let descendants = live_children(pid);
        {
            let mut child = process.lock();
            if child.kill().is_ok() {
                let _ = child.wait(Some(5.0));
            }
        }
        kill_all(&descendants);
    }

    // -- the reaper ---------------------------------------------------------

    /// Run the idle-stop and crash-cleanup sweep in the background.
    pub fn start_reaper(self: &Arc<Self>, interval: f64) {
        let mut reaper = self.reaper.lock();
        if reaper.is_some() {
            return;
        }
        self.stopping.clear();
        // The thread holds the pool weakly, so dropping the last owner is not
        // kept from happening by the reaper alone.
        let pool = Arc::downgrade(self);
        let stopping = Arc::clone(&self.stopping);
        let spawned = std::thread::Builder::new().name("supervisor-reaper".into()).spawn(move || {
            while !stopping.wait(interval) {
                let Some(pool) = pool.upgrade() else {
                    return;
                };
                if catch_unwind(AssertUnwindSafe(|| pool.sweep())).is_err() {
                    tracing::error!(target: LOGGER, "[supervisor] reaper pass failed");
                }
            }
        });
        match spawned {
            Ok(handle) => *reaper = Some(handle),
            Err(exc) => tracing::error!(target: LOGGER, "[supervisor] could not start the reaper: {}", exc),
        }
    }

    /// Stop the background reaper.
    pub fn stop_reaper(&self) {
        self.stopping.set();
        let reaper = self.reaper.lock().take();
        if let Some(handle) = reaper {
            if handle.thread().id() != std::thread::current().id() {
                join_with_timeout(handle, 5.0);
            }
        }
    }

    /// One pass: forget the dead, stop the idle, enforce the RSS ceiling.
    pub fn sweep(&self) -> SweepResult {
        let mut crashed: Vec<String> = Vec::new();

        // Children we spawned: poll the handle, which both settles whether the
        // process is really gone and reaps it if it is.
        let owned: Vec<(String, SharedChild)> =
            self.processes.lock().iter().map(|(id, p)| (id.clone(), Arc::clone(p))).collect();
        for (project_id, process) in owned {
            let (exited, returncode) = {
                let mut child = process.lock();
                let exited = child.poll().is_some();
                (exited, child.returncode())
            };
            if !exited {
                continue;
            }
            let uptime = (now_epoch() - child_started_at(&process.lock())).max(0.0);
            tracing::warn!(
                target: LOGGER,
                "[supervisor] project {}: agent exited with code {}",
                project_id,
                format_returncode(returncode)
            );
            self.processes.lock().shift_remove(&project_id);
            self.registry.remove(&project_id, (self.project_dir_for)(&project_id).as_deref());
            self.note_exit(&project_id, uptime);
            crashed.push(project_id);
        }

        // Records adopted from an earlier supervisor, whose processes are not
        // ours to wait on: the pid is all we have.
        for project_id in self.registry.reap() {
            tracing::warn!(target: LOGGER, "[supervisor] project {}: agent is gone", project_id);
            self.note_exit(&project_id, 0.0);
            crashed.push(project_id);
        }

        let mut idled: Vec<String> = Vec::new();
        if self.idle_ttl > 0.0 {
            for record in self.registry.all() {
                if self.running(&record.project_id).is_none() {
                    continue;
                }
                if record.idle_seconds() >= self.idle_ttl {
                    tracing::info!(
                        target: LOGGER,
                        "[supervisor] project {}: idle for {:.0}s, stopping it",
                        record.project_id,
                        record.idle_seconds()
                    );
                    if self.stop(&record.project_id, 10.0) {
                        idled.push(record.project_id.clone());
                    }
                }
            }
        }

        SweepResult { crashed, idled, over_limit: self.enforce_memory_limit() }
    }

    /// Stop any child that has grown past the ceiling. Returns the ids.
    ///
    /// Stopped rather than killed outright: SIGTERM lets it close its SQLite
    /// handles and llama.cpp contexts cleanly, and the next request for that
    /// project starts a fresh one.  The rest of the workspace never notices,
    /// which is the whole difference from a single process going OOM.
    fn enforce_memory_limit(&self) -> Vec<String> {
        if self.memory_limit_bytes == 0 {
            return Vec::new();
        }

        let mut stopped = Vec::new();
        for record in self.registry.all() {
            if self.running(&record.project_id).is_none() {
                continue;
            }
            let rss = match &self.process_rss_for {
                Some(rss_for) => rss_for(Some(record.pid)),
                None => process_rss(Some(record.pid)),
            };
            let Some(rss) = rss else { continue };
            if rss <= self.memory_limit_bytes {
                continue;
            }
            tracing::warn!(
                target: LOGGER,
                "[supervisor] project {}: {:.0} MB exceeds the {:.0} MB ceiling, stopping it",
                record.project_id,
                rss as f64 / (1024.0 * 1024.0),
                self.memory_limit_bytes as f64 / (1024.0 * 1024.0)
            );
            if self.stop(&record.project_id, 10.0) {
                stopped.push(record.project_id.clone());
            }
        }
        stopped
    }
}

/// The status code of a plain `GET url`, or `None` when nothing answered
/// within `timeout` seconds (`httpx.get(url, timeout=...).status_code`).
///
/// Spoken directly over a socket rather than through an HTTP client so it is
/// safe from any context: the pool's cold starts run on blocking threads that
/// may belong to an async runtime, where a blocking client of its own may not
/// be created or dropped.  Only `http://host:port/path` URLs are probed — the
/// agents' own loopback addresses.
pub(crate) fn http_get_status(url: &str, timeout: f64) -> Option<u16> {
    use std::io::{Read, Write};
    use std::net::{TcpStream, ToSocketAddrs};

    let parsed = url::Url::parse(url).ok()?;
    if parsed.scheme() != "http" {
        return None;
    }
    let host = parsed.host_str()?.to_string();
    let port = parsed.port_or_known_default()?;
    let mut target = parsed.path().to_string();
    if let Some(query) = parsed.query() {
        target.push('?');
        target.push_str(query);
    }
    let timeout = Duration::from_secs_f64(timeout.max(0.001));
    let deadline = Instant::now() + timeout;

    let address = (host.as_str(), port).to_socket_addrs().ok()?.next()?;
    let mut stream = TcpStream::connect_timeout(&address, timeout).ok()?;
    let remaining = |deadline: Instant| deadline.saturating_duration_since(Instant::now()).max(Duration::from_millis(1));
    stream.set_write_timeout(Some(remaining(deadline))).ok()?;
    let request = format!(
        "GET {target} HTTP/1.1\r\nHost: {host}:{port}\r\nAccept: */*\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(request.as_bytes()).ok()?;

    // Read just the status line.
    let mut head = Vec::new();
    let mut buffer = [0u8; 256];
    while !head.windows(2).any(|w| w == b"\r\n") {
        if Instant::now() >= deadline {
            return None;
        }
        stream.set_read_timeout(Some(remaining(deadline))).ok()?;
        let read = stream.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        head.extend_from_slice(&buffer[..read]);
        if head.len() > 8192 {
            break;
        }
    }
    let line = String::from_utf8_lossy(&head);
    let mut parts = line.split_whitespace();
    let version = parts.next()?;
    if !version.starts_with("HTTP/") {
        return None;
    }
    parts.next()?.parse().ok()
}

/// Resident size of `pid` in bytes, or `None` when it cannot be read.
pub fn process_rss(pid: Option<i64>) -> Option<u64> {
    let pid = pid.filter(|p| *p != 0)?;
    with_process(pid, ProcessRefreshKind::nothing().with_memory(), |p| p.memory())
}

/// Best-effort start time, for deciding whether a crash resets the backoff.
fn child_started_at(process: &ChildProcess) -> f64 {
    process_started_at(Some(process.pid())).unwrap_or_else(now_epoch)
}

/// Terminate a process we did not spawn and therefore cannot `wait` on.
///
/// "Ask it to stop, then insist", spelled portably: SIGTERM and a poll for the
/// pid to go, then SIGKILL if it outlives `timeout`.
pub fn terminate_pid(pid: i64, timeout: f64) {
    if !process_alive(Some(pid)) {
        return;
    }
    if terminate_signal(pid) && wait_gone(pid, timeout) {
        return;
    }
    kill_pid(pid);
}
