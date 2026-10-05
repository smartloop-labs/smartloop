//! Port of `smartloop/supervisor/spawn.py`.
//!
//! Fork-free process spawning for detached children.
//!
//! A `fork()` taken while the supervisor's gRPC server is mid-RPC is not safe:
//! the parent's other threads may be holding locks at the instant of the fork,
//! and the child — even for the microseconds before it `exec()`s — inherits
//! them held.  This module spawns the detached child through `posix_spawn`,
//! which never forks the parent at all: the kernel places the child image
//! directly.  The child gets its own session, its stdin is `/dev/null`, and
//! every other inherited descriptor (beyond stdout/stderr) is closed in the
//! new image.
//!
//! Platforms that cannot use `posix_spawn` fall back to a
//! `std::process::Command` spawn; on Windows the child runs in a new process
//! group with no console, and its output goes to an optional log file.

use std::collections::HashMap;
use std::io;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

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
#[cfg_attr(target_os = "macos", allow(dead_code))]
fn fd_dir() -> io::Result<&'static str> {
    for candidate in ["/dev/fd", "/proc/self/fd"] {
        if std::path::Path::new(candidate).is_dir() {
            return Ok(candidate);
        }
    }
    Err(io::Error::other("no /dev/fd or /proc/self/fd to enumerate open file descriptors"))
}

/// Whether `fd` is open at this instant.
#[cfg(unix)]
#[cfg_attr(target_os = "macos", allow(dead_code))]
fn still_open(fd: i32) -> bool {
    // SAFETY: F_GETFD only reads the descriptor flags.
    unsafe { libc::fcntl(fd, libc::F_GETFD) != -1 }
}

/// `posix_spawn`'s "new session" flag, where the platform has one.
#[cfg(target_os = "macos")]
const SPAWN_SETSID: libc::c_short = 0x0400;
#[cfg(any(target_os = "linux", target_os = "android"))]
const SPAWN_SETSID: libc::c_short = libc::POSIX_SPAWN_SETSID;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn posix_spawn_file_actions_addinherit_np(
        actions: *mut libc::posix_spawn_file_actions_t,
        fd: libc::c_int,
    ) -> libc::c_int;
}

/// `KEY=VALUE` strings for the child's environment.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
fn environment_block(env: Option<&HashMap<String, String>>) -> io::Result<Vec<std::ffi::CString>> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let to_c = |key: &[u8], value: &[u8]| {
        let mut entry = Vec::with_capacity(key.len() + value.len() + 1);
        entry.extend_from_slice(key);
        entry.push(b'=');
        entry.extend_from_slice(value);
        CString::new(entry).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
    };
    match env {
        Some(env) => env.iter().map(|(k, v)| to_c(k.as_bytes(), v.as_bytes())).collect(),
        None => std::env::vars_os()
            .map(|(k, v)| to_c(k.as_bytes(), v.as_bytes()))
            .collect(),
    }
}

/// Check a `posix_spawn*` return code.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
fn check(rc: libc::c_int) -> io::Result<()> {
    if rc == 0 { Ok(()) } else { Err(io::Error::from_raw_os_error(rc)) }
}

/// `posix_spawn` a child in a new session, inheriting stdout/stderr.
///
/// Comparable to a `Popen(command, env=env, stdin=DEVNULL,
/// start_new_session=True, close_fds=True)` but without ever forking: the
/// child gets its own session, its stdin is `/dev/null`, and every other
/// inherited descriptor is closed in the new image.
///
/// Returns the child pid.  Errors when `posix_spawn` is unavailable or the
/// spawn fails (the caller then falls back to the `std::process` path).
/// `env` of `None` means the current environment.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
pub fn spawn_detached(command: &[String], env: Option<&HashMap<String, String>>) -> io::Result<i64> {
    use std::ffi::CString;
    use std::mem::MaybeUninit;

    if command.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "command must name a program to spawn"));
    }
    let to_c = |s: &String| {
        CString::new(s.as_bytes()).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
    };
    let program = to_c(&command[0])?;
    let args: Vec<CString> = command.iter().map(to_c).collect::<io::Result<_>>()?;
    let envs = environment_block(env)?;
    let mut argv: Vec<*mut libc::c_char> = args.iter().map(|a| a.as_ptr() as *mut _).collect();
    argv.push(std::ptr::null_mut());
    let mut envp: Vec<*mut libc::c_char> = envs.iter().map(|e| e.as_ptr() as *mut _).collect();
    envp.push(std::ptr::null_mut());

    // SAFETY: plain open(2) of a constant path.
    let devnull = unsafe { libc::open(c"/dev/null".as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
    if devnull < 0 {
        return Err(io::Error::last_os_error());
    }

    let mut actions = MaybeUninit::<libc::posix_spawn_file_actions_t>::uninit();
    let mut attr = MaybeUninit::<libc::posix_spawnattr_t>::uninit();
    // SAFETY: every posix_spawn object is initialised before use and destroyed
    // below; the argv/envp arrays outlive the call and are NULL-terminated.
    unsafe {
        let mut actions_ready = false;
        let mut attr_ready = false;
        let result = (|| -> io::Result<i64> {
            check(libc::posix_spawn_file_actions_init(actions.as_mut_ptr()))?;
            actions_ready = true;
            check(libc::posix_spawnattr_init(attr.as_mut_ptr()))?;
            attr_ready = true;

            check(libc::posix_spawn_file_actions_adddup2(actions.as_mut_ptr(), devnull, 0))?;

            #[allow(unused_mut)]
            let mut flags: libc::c_short = SPAWN_SETSID;

            #[cfg(target_os = "macos")]
            {
                // Darwin can close every descriptor not named in the file
                // actions itself, which leaves no enumeration to race against
                // another thread closing a descriptor between the snapshot
                // and the spawn.  stdout/stderr are inherited explicitly.
                flags |= libc::POSIX_SPAWN_CLOEXEC_DEFAULT as libc::c_short;
                check(posix_spawn_file_actions_addinherit_np(actions.as_mut_ptr(), 1))?;
                check(posix_spawn_file_actions_addinherit_np(actions.as_mut_ptr(), 2))?;
            }

            #[cfg(not(target_os = "macos"))]
            {
                // Enumerate the inherited descriptors, then re-verify each is
                // still open a moment before the spawn, closing only what is
                // genuinely inherited, so a stale entry (such as the directory
                // handle the enumeration itself used) cannot abort the spawn.
                let candidates: Vec<i32> = std::fs::read_dir(fd_dir()?)?
                    .filter_map(|entry| entry.ok())
                    .filter_map(|entry| entry.file_name().to_str().and_then(|n| n.parse().ok()))
                    .collect();
                for fd in candidates.into_iter().filter(|fd| *fd > 2 && still_open(*fd)) {
                    check(libc::posix_spawn_file_actions_addclose(actions.as_mut_ptr(), fd))?;
                }
            }

            check(libc::posix_spawnattr_setflags(attr.as_mut_ptr(), flags))?;

            let mut pid: libc::pid_t = 0;
            check(libc::posix_spawnp(
                &mut pid,
                program.as_ptr(),
                actions.as_ptr(),
                attr.as_ptr(),
                argv.as_ptr(),
                envp.as_ptr(),
            ))?;
            Ok(i64::from(pid))
        })();
        if actions_ready {
            libc::posix_spawn_file_actions_destroy(actions.as_mut_ptr());
        }
        if attr_ready {
            libc::posix_spawnattr_destroy(attr.as_mut_ptr());
        }
        libc::close(devnull);
        result
    }
}

/// `posix_spawn` is not used on this platform; the caller falls back.
#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "android")))]
pub fn spawn_detached(command: &[String], env: Option<&HashMap<String, String>>) -> io::Result<i64> {
    let _ = (command, env);
    Err(io::Error::new(io::ErrorKind::Unsupported, "posix_spawn is not available on this platform"))
}

/// Decode a `waitpid` status the way `os.waitstatus_to_exitcode` does:
/// the exit code, or `-signal` for a child killed by a signal.
#[cfg(unix)]
fn waitstatus_to_exitcode(status: libc::c_int) -> i32 {
    if libc::WIFEXITED(status) {
        libc::WEXITSTATUS(status)
    } else if libc::WIFSIGNALED(status) {
        -libc::WTERMSIG(status)
    } else {
        status
    }
}

/// Send `signal` to `pid`.
#[cfg(unix)]
pub(crate) fn send_signal(pid: i64, signal: libc::c_int) -> io::Result<()> {
    if pid <= 0 || pid > i64::from(i32::MAX) {
        return Err(io::Error::from_raw_os_error(libc::ESRCH));
    }
    // SAFETY: kill(2) with a validated pid.
    if unsafe { libc::kill(pid as libc::pid_t, signal) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// The `Popen` surface callers rely on, for a raw spawned pid.
///
/// `posix_spawn` hands back a raw pid rather than a handle object, and nothing
/// here needs a pipe or a handle that keeps the process tied to its parent:
/// the pool polls for exit, signals by pid, and waits.
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
    #[cfg(unix)]
    pub fn poll(&mut self) -> Option<i32> {
        if self.returncode.is_some() {
            return self.returncode;
        }
        let mut status: libc::c_int = 0;
        // SAFETY: non-blocking waitpid on our own child.
        let reaped = unsafe { libc::waitpid(self.pid as libc::pid_t, &mut status, libc::WNOHANG) };
        if reaped > 0 {
            self.returncode = Some(waitstatus_to_exitcode(status));
        }
        // 0: still running; -1 (ECHILD/EINTR): not ours to wait on any more.
        self.returncode
    }

    /// The exit code once the child has exited, else `None`.
    #[cfg(not(unix))]
    pub fn poll(&mut self) -> Option<i32> {
        self.returncode
    }

    /// Block until the child exits (or `timeout` seconds expire) and reap it.
    pub fn wait(&mut self, timeout: Option<f64>) -> Result<Option<i32>, WaitError> {
        if self.returncode.is_none() {
            let deadline = timeout.map(|t| Instant::now() + Duration::from_secs_f64(t.max(0.0)));
            loop {
                if self.poll().is_some() {
                    break;
                }
                if let Some(deadline) = deadline {
                    if Instant::now() >= deadline {
                        return Err(TimeoutExpired { pid: self.pid, timeout: timeout.unwrap_or(0.0) }.into());
                    }
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        Ok(self.returncode)
    }

    /// Send SIGTERM.
    #[cfg(unix)]
    pub fn terminate(&self) -> io::Result<()> {
        send_signal(self.pid, libc::SIGTERM)
    }

    /// Send SIGKILL.
    #[cfg(unix)]
    pub fn kill(&self) -> io::Result<()> {
        send_signal(self.pid, libc::SIGKILL)
    }

    /// Terminate the process.
    #[cfg(not(unix))]
    pub fn terminate(&self) -> io::Result<()> {
        crate::supervisor::pool::terminate_pid(self.pid, 0.0);
        Ok(())
    }

    /// Kill the process.
    #[cfg(not(unix))]
    pub fn kill(&self) -> io::Result<()> {
        self.terminate()
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

impl PopenProcess {
    fn poll(&mut self) -> Option<i32> {
        if self.returncode.is_some() {
            return self.returncode;
        }
        if let Ok(Some(status)) = self.child.try_wait() {
            self.returncode = Some(exit_code(status));
        }
        self.returncode
    }
}

/// An `ExitStatus` as `Popen.returncode` reports it (`-signal` when signalled).
fn exit_code(status: std::process::ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return -signal;
        }
    }
    -1
}

impl ChildProcess {
    /// The child's pid.
    pub fn pid(&self) -> i64 {
        match self {
            ChildProcess::Detached(p) => p.pid,
            ChildProcess::Popen(p) => i64::from(p.child.id()),
        }
    }

    /// The cached exit code, if the child has been reaped.
    pub fn returncode(&self) -> Option<i32> {
        match self {
            ChildProcess::Detached(p) => p.returncode,
            ChildProcess::Popen(p) => p.returncode,
        }
    }

    /// Whether this is the `std::process` fallback (Python `isinstance(proc, Popen)`).
    pub fn is_popen(&self) -> bool {
        matches!(self, ChildProcess::Popen(_))
    }

    /// The exit code once the child has exited, else `None`.
    pub fn poll(&mut self) -> Option<i32> {
        match self {
            ChildProcess::Detached(p) => p.poll(),
            ChildProcess::Popen(p) => p.poll(),
        }
    }

    /// Block until exit (or timeout) and reap.
    pub fn wait(&mut self, timeout: Option<f64>) -> Result<Option<i32>, WaitError> {
        match self {
            ChildProcess::Detached(p) => p.wait(timeout),
            ChildProcess::Popen(p) => {
                if p.returncode.is_none() {
                    match timeout {
                        None => {
                            let status = p.child.wait()?;
                            p.returncode = Some(exit_code(status));
                        }
                        Some(t) => {
                            let deadline = Instant::now() + Duration::from_secs_f64(t.max(0.0));
                            while p.poll().is_none() {
                                if Instant::now() >= deadline {
                                    return Err(TimeoutExpired { pid: i64::from(p.child.id()), timeout: t }.into());
                                }
                                std::thread::sleep(Duration::from_millis(50));
                            }
                        }
                    }
                }
                Ok(p.returncode)
            }
        }
    }

    /// Ask the child to stop (SIGTERM / TerminateProcess).
    pub fn terminate(&mut self) -> io::Result<()> {
        match self {
            ChildProcess::Detached(p) => p.terminate(),
            ChildProcess::Popen(p) => {
                // Like Popen.send_signal: never signal a pid already reaped.
                if p.poll().is_some() {
                    return Ok(());
                }
                #[cfg(unix)]
                {
                    send_signal(i64::from(p.child.id()), libc::SIGTERM)
                }
                #[cfg(not(unix))]
                {
                    p.child.kill()
                }
            }
        }
    }

    /// Kill the child outright.
    pub fn kill(&mut self) -> io::Result<()> {
        match self {
            ChildProcess::Detached(p) => p.kill(),
            ChildProcess::Popen(p) => {
                if p.poll().is_some() {
                    return Ok(());
                }
                p.child.kill()
            }
        }
    }
}

/// Signature of the posix-spawn step, injectable for tests of the fallback.
pub type Spawner = dyn Fn(&[String], Option<&HashMap<String, String>>) -> io::Result<i64>;

/// Spawn `command` detached, falling back to a `std::process` spawn.
///
/// Returns [`ChildProcess::Detached`] when `posix_spawn` succeeds and
/// [`ChildProcess::Popen`] otherwise (stdin is `/dev/null`, own session) — so
/// a platform or runtime that cannot use `posix_spawn` still gets its child.
///
/// On Windows the child does not inherit our stdout/stderr: they are whatever
/// launched the supervisor gave it, and passing some of those through process
/// creation fails.  Its output goes to `output` instead, appended, or nowhere
/// when that is `None`.  It runs without a console window too.
pub fn popen_detached_devnull(
    command: &[String],
    env: Option<&HashMap<String, String>>,
    output: Option<&str>,
) -> io::Result<ChildProcess> {
    popen_detached_devnull_with(&spawn_detached, command, env, output)
}

/// The `std::process::Command` for `command`, with `env` (or ours) and stdin on
/// `/dev/null`.
fn base_command(command: &[String], env: Option<&HashMap<String, String>>) -> io::Result<Command> {
    let program = command
        .first()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "command must name a program to spawn"))?;
    let mut cmd = Command::new(program);
    cmd.args(&command[1..]);
    if let Some(env) = env {
        cmd.env_clear();
        cmd.envs(env);
    }
    cmd.stdin(Stdio::null());
    Ok(cmd)
}

/// [`popen_detached_devnull`] with an injectable posix-spawn step (test seam
/// standing in for monkeypatching `spawn_detached`).
pub fn popen_detached_devnull_with(
    spawner: &Spawner,
    command: &[String],
    env: Option<&HashMap<String, String>>,
    output: Option<&str>,
) -> io::Result<ChildProcess> {
    #[cfg(unix)]
    {
        let _ = output;
        if let Ok(pid) = spawner(command, env) {
            return Ok(ChildProcess::Detached(DetachedProcess::new(pid)));
        }
        let mut cmd = base_command(command, env)?;
        cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());
        {
            use std::os::unix::process::CommandExt;
            // start_new_session=True
            // SAFETY: setsid is async-signal-safe.
            unsafe {
                cmd.pre_exec(|| {
                    if libc::setsid() == -1 {
                        return Err(io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
        let child = cmd.spawn()?;
        Ok(ChildProcess::Popen(PopenProcess { child, returncode: None }))
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = spawner;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;

        let mut cmd = base_command(command, env)?;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
        match output {
            None => {
                cmd.stdout(Stdio::null()).stderr(Stdio::null());
            }
            Some(path) => {
                // The child holds its own copy of the handle; ours can close at once.
                let sink = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
                let err = sink.try_clone()?;
                cmd.stdout(Stdio::from(sink)).stderr(Stdio::from(err));
            }
        }
        let child = cmd.spawn()?;
        Ok(ChildProcess::Popen(PopenProcess { child, returncode: None }))
    }
}
