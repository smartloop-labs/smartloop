//! Port of `smartloop/supervisor/spawn.py`.
//!
//! Fork-free process spawning for detached children.
//!
//! A `fork()` taken while the supervisor's gRPC server is mid-RPC is not safe,
//! so on POSIX the detached child is placed with `posix_spawn` (new session,
//! stdin on `/dev/null`, every other inherited descriptor closed), never
//! forking the parent.  Platforms that cannot use `posix_spawn` fall back to a
//! `std::process::Command` spawn; on Windows the child runs in a new process
//! group with no console, and its output goes to an optional log file.

use std::collections::HashMap;
use std::io;
use std::process::Child;
use std::time::Duration;

/// Error raised by [`DetachedProcess::wait`] / [`ChildProcess::wait`] when the
/// timeout expires (Python's `subprocess.TimeoutExpired`).
#[derive(Debug, thiserror::Error)]
#[error("Command '{pid}' timed out after {timeout} seconds")]
pub struct TimeoutExpired {
    pub pid: i64,
    pub timeout: f64,
}

/// Errors from waiting on a child.
#[derive(Debug, thiserror::Error)]
pub enum WaitError {
    #[error(transparent)]
    Timeout(#[from] TimeoutExpired),
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// A directory enumerating this process's open file descriptors.
#[cfg(unix)]
fn fd_dir() -> io::Result<&'static str> {
    todo!()
}

/// Whether `fd` is open at this instant.
#[cfg(unix)]
fn still_open(fd: i32) -> bool {
    let _ = fd;
    todo!()
}

/// `posix_spawn` a child in a new session, inheriting stdout/stderr.
///
/// Returns the child pid.  Errors when `posix_spawn` is unavailable or the
/// spawn fails (the caller then falls back to the `std::process` path).
/// `env` of `None` means the current environment.
pub fn spawn_detached(command: &[String], env: Option<&HashMap<String, String>>) -> io::Result<i64> {
    let _ = (command, env);
    todo!()
}

/// The `Popen` surface callers rely on, for a raw spawned pid.
#[derive(Debug)]
pub struct DetachedProcess {
    pub pid: i64,
    pub returncode: Option<i32>,
}

impl DetachedProcess {
    /// Wrap a pid returned by [`spawn_detached`].
    pub fn new(pid: i64) -> Self {
        Self { pid, returncode: None }
    }

    /// The exit code once the child has exited (reaping it), else `None`.
    pub fn poll(&mut self) -> Option<i32> {
        todo!()
    }

    /// Block until the child exits (or `timeout` seconds expire) and reap it.
    pub fn wait(&mut self, timeout: Option<f64>) -> Result<Option<i32>, WaitError> {
        let _ = timeout;
        todo!()
    }

    /// Send SIGTERM.
    pub fn terminate(&self) -> io::Result<()> {
        todo!()
    }

    /// Send SIGKILL.
    pub fn kill(&self) -> io::Result<()> {
        todo!()
    }
}

/// A spawned child: either a `posix_spawn`ed pid or a `std::process::Child`
/// (the `subprocess.Popen` fallback).
#[derive(Debug)]
pub enum ChildProcess {
    Detached(DetachedProcess),
    Popen(PopenProcess),
}

/// The `subprocess.Popen` fallback: a `std::process::Child` plus its cached
/// exit code.
#[derive(Debug)]
pub struct PopenProcess {
    pub child: Child,
    pub returncode: Option<i32>,
}

impl ChildProcess {
    /// The child's pid.
    pub fn pid(&self) -> i64 {
        todo!()
    }

    /// The cached exit code, if the child has been reaped.
    pub fn returncode(&self) -> Option<i32> {
        todo!()
    }

    /// Whether this is the `std::process` fallback (Python `isinstance(proc, Popen)`).
    pub fn is_popen(&self) -> bool {
        matches!(self, ChildProcess::Popen(_))
    }

    /// The exit code once the child has exited, else `None`.
    pub fn poll(&mut self) -> Option<i32> {
        todo!()
    }

    /// Block until exit (or timeout) and reap.
    pub fn wait(&mut self, timeout: Option<f64>) -> Result<Option<i32>, WaitError> {
        let _ = timeout;
        todo!()
    }

    /// Ask the child to stop (SIGTERM / TerminateProcess).
    pub fn terminate(&mut self) -> io::Result<()> {
        todo!()
    }

    /// Kill the child outright.
    pub fn kill(&mut self) -> io::Result<()> {
        todo!()
    }
}

/// Signature of the posix-spawn step, injectable for tests of the fallback.
pub type Spawner = dyn Fn(&[String], Option<&HashMap<String, String>>) -> io::Result<i64>;

/// Spawn `command` detached, falling back to a `std::process` spawn.
///
/// Returns [`ChildProcess::Detached`] when `posix_spawn` succeeds and
/// [`ChildProcess::Popen`] otherwise.  On Windows output goes to `output`
/// (appended), or nowhere when that is `None`.
pub fn popen_detached_devnull(
    command: &[String],
    env: Option<&HashMap<String, String>>,
    output: Option<&str>,
) -> io::Result<ChildProcess> {
    popen_detached_devnull_with(&spawn_detached, command, env, output)
}

/// [`popen_detached_devnull`] with an injectable posix-spawn step (test seam
/// standing in for monkeypatching `spawn_detached`).
pub fn popen_detached_devnull_with(
    spawner: &Spawner,
    command: &[String],
    env: Option<&HashMap<String, String>>,
    output: Option<&str>,
) -> io::Result<ChildProcess> {
    let _ = (spawner, command, env, output, Duration::from_millis(50));
    todo!()
}
