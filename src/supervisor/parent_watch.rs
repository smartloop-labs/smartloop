//! Port of `smartloop/supervisor/parent_watch.py`.
//!
//! Tie a project agent's lifetime to the supervisor that spawned it.
//!
//! A child is deliberately detached from the supervisor's process group (see
//! the detached spawn in [`crate::supervisor::spawn`]) so a Ctrl-C in the
//! terminal reaches the supervisor alone, which then stops the children
//! itself, in order, after the model host they talk to.  That detachment is
//! also what makes an orphan possible: nothing links a child to its parent
//! afterwards, so a supervisor that is `SIGKILL`-ed, OOM-killed, or lost with
//! its terminal leaves children running.  Each one keeps a project's task
//! queue, its MCP stdio grandchildren and its share of the device, answers on
//! a port recorded in a registry nobody reads any more, and never exits.
//!
//! The link is put back here, from the child's side: the supervisor passes its
//! pid and start time in the environment ([`parent_environment`]) and the child
//! watches them ([`start_parent_watch`]).
//!
//! Watched rather than inherited because inheritance is not portable — the
//! "die with my parent" primitive is `PR_SET_PDEATHSIG` on Linux, `kqueue` on
//! BSD, job objects on Windows — and because a poll can do something
//! inheritance cannot: survive a *planned* handover.  Adopting the children an
//! earlier supervisor left is a feature, so a child whose parent has vanished
//! looks in the registry for a live replacement before giving up, and
//! re-targets onto it when there is one.  Only a child with no parent and no
//! successor after [`DEFAULT_GRACE_SECONDS`] shuts itself down.
//!
//! Start time is checked alongside the pid because pids are recycled: on a busy
//! machine the number a dead supervisor used can belong to something unrelated
//! within minutes, and a child that mistakes a stranger for its parent is an
//! orphan again — just one that looks supervised.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};
use serde_json::Value;
use sysinfo::ProcessRefreshKind;

use crate::supervisor::registry::{process_alive, py_int, with_process, AgentRegistry};

/// Logger name of this module.
const LOGGER: &str = "smartloop.supervisor.parent_watch";

/// The supervisor's pid.
pub const PARENT_PID_ENV: &str = "SLP_PARENT_PID";
/// The supervisor's start time, which disambiguates a recycled pid.
pub const PARENT_STARTED_AT_ENV: &str = "SLP_PARENT_STARTED_AT";

/// How often the child checks that it still has a parent.  Cheap — one
/// process-table read per tick — so the interval is set by how long an orphan
/// may linger, not by what the check costs.
pub const DEFAULT_POLL_SECONDS: f64 = 5.0;

/// How long a child keeps serving after its parent disappears, waiting for a
/// replacement supervisor to come up and adopt it.  Long enough to cover a
/// restart, short enough that an orphan is measured in seconds.
pub const DEFAULT_GRACE_SECONDS: f64 = 30.0;

/// Two `create_time` readings of the same process can differ in the last
/// decimal across platforms; a second of slack is far below pid-reuse latency.
const START_TIME_TOLERANCE: f64 = 1.0;

/// Callback fired when the child decides it is an orphan.
pub type OrphanCallback = Arc<dyn Fn() + Send + Sync>;

/// Python's `repr(float)` closely enough to round-trip (`1.0`, not `1`).
fn float_repr(value: f64) -> String {
    format!("{value:?}")
}

/// The environment a spawning supervisor adds so its child can watch it.
pub fn parent_environment() -> HashMap<String, String> {
    let pid = i64::from(std::process::id());
    let mut env = HashMap::new();
    env.insert(PARENT_PID_ENV.to_string(), pid.to_string());
    if let Some(started_at) = process_started_at(Some(pid)) {
        env.insert(PARENT_STARTED_AT_ENV.to_string(), float_repr(started_at));
    }
    env
}

/// When `pid` started (epoch seconds), or `None` when that cannot be read.
pub fn process_started_at(pid: Option<i64>) -> Option<f64> {
    let pid = pid.filter(|pid| *pid > 0)?;
    with_process(pid, ProcessRefreshKind::nothing(), |p| p.start_time() as f64)
}

/// Whether `pid` is still the process that started at `started_at`.
///
/// Liveness itself is [`process_alive`], which spells the question portably.
/// The start time is the part that matters here: without it a recycled pid
/// reads as a living parent and the child never notices it was orphaned.
///
/// A process that is there but cannot be inspected counts as alive: being
/// unable to tell is not evidence of an orphan.
pub fn parent_is_alive(pid: Option<i64>, started_at: Option<f64>) -> bool {
    if !process_alive(pid) {
        return false;
    }
    let Some(started_at) = started_at else {
        return true;
    };
    match process_started_at(pid) {
        // The process is there but we may not inspect it — being unable to
        // tell is not evidence of an orphan, so keep serving.
        None => true,
        Some(observed) => (observed - started_at).abs() <= START_TIME_TOLERANCE,
    }
}

/// A positive float from env var `name`, else `default` (warns on garbage).
fn configured(name: &str, default: f64) -> f64 {
    let raw = std::env::var(name).unwrap_or_default();
    let raw = raw.trim();
    if raw.is_empty() {
        return default;
    }
    match raw.parse::<f64>() {
        Ok(value) if value > 0.0 => value,
        Ok(_) => default,
        Err(_) => {
            tracing::warn!(
                target: LOGGER,
                "[parent-watch] {}={:?} is not a number; using {:.0}s",
                name,
                raw,
                default
            );
            default
        }
    }
}

/// A settable flag a thread can wait on with a timeout (`threading.Event`).
#[derive(Debug, Default)]
pub(crate) struct StopEvent {
    flag: Mutex<bool>,
    cv: Condvar,
}

impl StopEvent {
    pub(crate) fn set(&self) {
        *self.flag.lock() = true;
        self.cv.notify_all();
    }

    pub(crate) fn clear(&self) {
        *self.flag.lock() = false;
    }

    /// Wait up to `seconds` for the flag; returns whether it is set.
    pub(crate) fn wait(&self, seconds: f64) -> bool {
        let deadline = Instant::now() + Duration::from_secs_f64(seconds.max(0.0));
        let mut flag = self.flag.lock();
        while !*flag {
            if self.cv.wait_until(&mut flag, deadline).timed_out() {
                break;
            }
        }
        *flag
    }
}

/// Join `handle`, giving up (and detaching it) after `timeout` seconds.
pub(crate) fn join_with_timeout(handle: JoinHandle<()>, timeout: f64) {
    let deadline = Instant::now() + Duration::from_secs_f64(timeout.max(0.0));
    while !handle.is_finished() {
        if Instant::now() >= deadline {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let _ = handle.join();
}

/// Polls for the supervisor, and calls back when there is not one.
///
/// Runs on a background thread: the watch must never be the reason a child
/// that *is* supervised stays up, and the callback it fires — an orderly
/// server shutdown — has to be able to stop it.
pub struct ParentWatch {
    pid: Mutex<i64>,
    started_at: Mutex<Option<f64>>,
    pub home_dir: Option<String>,
    pub on_orphaned: Option<OrphanCallback>,
    pub poll_interval: f64,
    pub grace: f64,
    stop: StopEvent,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl std::fmt::Debug for ParentWatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParentWatch")
            .field("pid", &*self.pid.lock())
            .field("poll_interval", &self.poll_interval)
            .field("grace", &self.grace)
            .finish()
    }
}

impl ParentWatch {
    /// A watch on `pid` (not yet started).
    pub fn new(
        pid: i64,
        started_at: Option<f64>,
        home_dir: Option<String>,
        on_orphaned: Option<OrphanCallback>,
        poll_interval: f64,
        grace: f64,
    ) -> Self {
        Self {
            pid: Mutex::new(pid),
            started_at: Mutex::new(started_at),
            home_dir,
            on_orphaned,
            poll_interval,
            grace,
            stop: StopEvent::default(),
            thread: Mutex::new(None),
        }
    }

    /// A watch with the default poll interval and grace.
    pub fn with_defaults(pid: i64) -> Self {
        Self::new(pid, None, None, None, DEFAULT_POLL_SECONDS, DEFAULT_GRACE_SECONDS)
    }

    /// The supervisor pid currently watched (changes on adoption).
    pub fn pid(&self) -> i64 {
        *self.pid.lock()
    }

    /// The start time of the supervisor currently watched.
    pub fn started_at(&self) -> Option<f64> {
        *self.started_at.lock()
    }

    // -- lifecycle ----------------------------------------------------------

    /// Start the polling thread (idempotent) and return the shared watch.
    pub fn start(self) -> Arc<Self> {
        let shared = Arc::new(self);
        shared.start_shared()
    }

    /// Start the polling thread on an already-shared watch (idempotent).
    pub fn start_shared(self: &Arc<Self>) -> Arc<Self> {
        let mut thread = self.thread.lock();
        if thread.is_some() {
            return Arc::clone(self);
        }
        let watch = Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name("parent-watch".into())
            .spawn(move || watch.run());
        match spawned {
            Ok(handle) => {
                *thread = Some(handle);
                tracing::info!(
                    target: LOGGER,
                    "[parent-watch] watching supervisor pid {} (every {:.0}s, {:.0}s grace)",
                    self.pid(),
                    self.poll_interval,
                    self.grace
                );
            }
            Err(exc) => {
                tracing::warn!(target: LOGGER, "[parent-watch] could not start the watch: {}", exc);
            }
        }
        Arc::clone(self)
    }

    /// Stop watching.  Called on an ordinary shutdown, which is not an orphaning.
    pub fn stop(&self, timeout: f64) {
        self.stop.set();
        let thread = self.thread.lock().take();
        if let Some(handle) = thread {
            if handle.thread().id() != std::thread::current().id() {
                join_with_timeout(handle, timeout);
            }
        }
    }

    // -- the loop -----------------------------------------------------------

    /// The poll loop.
    fn run(self: &Arc<Self>) {
        let mut missing_since: Option<Instant> = None;
        while !self.stop.wait(self.poll_interval) {
            if parent_is_alive(Some(self.pid()), self.started_at()) {
                missing_since = None;
                continue;
            }

            if let Some((pid, started_at)) = self.successor() {
                tracing::info!(
                    target: LOGGER,
                    "[parent-watch] supervisor pid {} is gone; adopted by pid {}",
                    self.pid(),
                    pid
                );
                *self.pid.lock() = pid;
                *self.started_at.lock() = started_at;
                missing_since = None;
                continue;
            }

            let now = Instant::now();
            let Some(since) = missing_since else {
                missing_since = Some(now);
                tracing::warn!(
                    target: LOGGER,
                    "[parent-watch] supervisor pid {} is gone; waiting {:.0}s for a replacement \
                     before shutting down",
                    self.pid(),
                    self.grace
                );
                continue;
            };
            if now.duration_since(since).as_secs_f64() >= self.grace {
                tracing::warn!(
                    target: LOGGER,
                    "[parent-watch] no supervisor after {:.0}s; this agent is an orphan, \
                     shutting it down",
                    self.grace
                );
                self.fire();
                return;
            }
        }
    }

    /// A live supervisor that is not the one we were watching, if any.
    ///
    /// Read from the registry rather than passed in, because a replacement
    /// supervisor is by definition a process that did not exist when this
    /// child was spawned.  It records itself before it adopts anybody, so a
    /// pid here means the handover is underway even if this child has not been
    /// routed to yet.
    fn successor(&self) -> Option<(i64, Option<f64>)> {
        let home_dir = self.home_dir.as_deref().filter(|h| !h.is_empty())?;
        // A registry we cannot read is not a successor.
        let supervisor = catch_unwind(|| AgentRegistry::new(home_dir).load().supervisor()).ok()?;
        let pid = py_int(supervisor.get("pid")?)?;
        if pid == self.pid() || pid == i64::from(std::process::id()) {
            return None;
        }
        let started_at = match supervisor.get("started_at") {
            Some(Value::Number(n)) => n.as_f64(),
            _ => None,
        };
        if !parent_is_alive(Some(pid), started_at) {
            return None;
        }
        Some((pid, started_at))
    }

    /// Invoke the orphan callback, logging (not propagating) a panic.
    fn fire(&self) {
        let Some(callback) = self.on_orphaned.as_ref() else {
            return;
        };
        if let Err(panic) = catch_unwind(AssertUnwindSafe(|| callback())) {
            let detail = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            tracing::error!(target: LOGGER, "[parent-watch] orphan shutdown failed: {}", detail);
        }
    }
}

/// Watch the parent named in the environment, if this process has one.
///
/// Returns `None` when [`PARENT_PID_ENV`] is unset — the single shared server
/// and a hand-run project agent were not spawned by a supervisor and have
/// nothing to be orphaned from.
pub fn start_parent_watch(
    on_orphaned: OrphanCallback,
    home_dir: Option<String>,
) -> Option<Arc<ParentWatch>> {
    let raw = std::env::var(PARENT_PID_ENV).unwrap_or_default();
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let Some(pid) = crate::supervisor::registry::parse_py_int(raw) else {
        tracing::warn!(
            target: LOGGER,
            "[parent-watch] {}={:?} is not a pid; not watching",
            PARENT_PID_ENV,
            raw
        );
        return None;
    };
    if pid <= 0 || pid == i64::from(std::process::id()) {
        return None;
    }

    let raw_started = std::env::var(PARENT_STARTED_AT_ENV).unwrap_or_default();
    let raw_started = raw_started.trim();
    let started_at = if raw_started.is_empty() { None } else { raw_started.parse::<f64>().ok() };

    Some(
        ParentWatch::new(
            pid,
            started_at,
            home_dir,
            Some(on_orphaned),
            configured("SLP_PARENT_WATCH_INTERVAL_S", DEFAULT_POLL_SECONDS),
            configured("SLP_PARENT_WATCH_GRACE_S", DEFAULT_GRACE_SECONDS),
        )
        .start(),
    )
}
