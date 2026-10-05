//! Local MCP client for subprocess-based (stdio) MCP servers.
//!
//! MCP over stdio uses newline-delimited JSON-RPC 2.0:
//!   - Send a JSON object + `\n` to the process stdin
//!   - Read a JSON object + `\n` from the process stdout
//!
//! Port of `smartloop/mcp/local_client.py`.

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::{Map, Value};

use crate::mcp::client::{
    call_result_parts, jsonrpc_error_message, server_info_from, tools_from, MCPClientError, MCPConnectResult,
    MCPLaunchError, MCPProtocolError, MCPToolCallResult,
};
use crate::mcp::read_only_calls::python_json_dumps;
use crate::paths::expand_home;

/// Python logger name of this module.
const LOGGER: &str = "smartloop.mcp.local_client";

pub const PROTOCOL_VERSION: &str = "2024-11-05";
pub const CLIENT_INFO_NAME: &str = "smartloop-cli";
pub const CLIENT_INFO_VERSION: &str = "1.0.0";

/// Lines of stderr retained per client (see `stderr_chunks`).
pub const STDERR_MAX_LINES: usize = 200;

/// The executable built-in servers run under: `SLP_EXE` when set (tests point
/// it at the built `slp`), else the running binary — the Rust analogue of
/// `sys.executable`.
pub fn runtime_executable() -> String {
    if let Ok(exe) = std::env::var("SLP_EXE") {
        if !exe.is_empty() {
            return exe;
        }
    }
    std::env::current_exe()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "slp".to_string())
}

#[cfg(windows)]
const PATH_SEP: &str = ";";
#[cfg(not(windows))]
const PATH_SEP: &str = ":";

/// `os.path.exists(fn) and os.access(fn, os.X_OK) and not os.path.isdir(fn)`.
fn access_check(path: &Path) -> bool {
    if !path.exists() || path.is_dir() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        match CString::new(path.as_os_str().as_bytes()) {
            // SAFETY: `c` is a valid NUL-terminated path for the duration of the call.
            Ok(c) => unsafe { libc::access(c.as_ptr(), libc::X_OK) == 0 },
            Err(_) => false,
        }
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// `shutil.which(cmd, path=path)`.
pub fn which(cmd: &str, path: Option<&str>) -> Option<String> {
    // A command with a directory part is looked up directly, relative to the
    // current directory, rather than against PATH.
    let has_dir = Path::new(cmd).parent().is_some_and(|p| !p.as_os_str().is_empty());
    #[cfg(windows)]
    let exts: Vec<String> = {
        let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string());
        let mut exts: Vec<String> = pathext.split(';').filter(|e| !e.is_empty()).map(|e| e.to_string()).collect();
        if exts.iter().any(|e| cmd.to_ascii_lowercase().ends_with(&e.to_ascii_lowercase())) {
            exts = vec![String::new()];
        } else {
            exts.insert(0, String::new());
        }
        exts
    };
    #[cfg(not(windows))]
    let exts: Vec<String> = vec![String::new()];

    if has_dir {
        for ext in &exts {
            let candidate = format!("{cmd}{ext}");
            if access_check(Path::new(&candidate)) {
                return Some(candidate);
            }
        }
        return None;
    }

    let path = match path {
        Some(p) => p.to_string(),
        None => std::env::var("PATH").unwrap_or_else(|_| "/bin:/usr/bin".to_string()),
    };
    if path.is_empty() {
        return None;
    }
    let mut seen: Vec<String> = Vec::new();
    for dir in path.split(PATH_SEP) {
        if seen.iter().any(|d| d == dir) {
            continue;
        }
        seen.push(dir.to_string());
        for ext in &exts {
            let name = if dir.is_empty() {
                format!("{cmd}{ext}")
            } else {
                Path::new(dir).join(format!("{cmd}{ext}")).to_string_lossy().into_owned()
            };
            if access_check(Path::new(&name)) {
                return Some(name);
            }
        }
    }
    None
}

/// Python's `Popen.returncode` for an exit status: the code, or the negated
/// signal number for a signal death.
fn returncode(status: &ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            return -sig;
        }
    }
    -1
}

/// Python's `str(float)` for the timeout in messages (`30.0`, `2.5`).
fn py_float(value: f64) -> String {
    if value.fract() == 0.0 && value.is_finite() {
        format!("{value:.1}")
    } else {
        format!("{value}")
    }
}

/// An `OSError` from spawning, worded the way Python's `Popen` reports it
/// (`[Errno 2] No such file or directory: 'uvx'`).
fn spawn_error_message(e: &std::io::Error, program: &str) -> String {
    match e.raw_os_error() {
        Some(code) => {
            let text = e.to_string();
            let strerror = text
                .rsplit_once(" (os error")
                .map(|(head, _)| head.to_string())
                .unwrap_or(text);
            format!("[Errno {code}] {strerror}: '{program}'")
        }
        None => e.to_string(),
    }
}

/// Communicates with a local MCP server via stdin/stdout JSON-RPC.
///
/// `enter()` spawns the child and starts the pipe readers; `exit()` reaps it
/// (idempotent, also run on drop and on `enter()`'s own failure path).
#[derive(Debug)]
pub struct LocalMCPClient {
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub cwd: Option<String>,
    /// Per-request timeout, seconds.
    pub timeout: f64,
    /// The running child, if started.
    pub process: Option<Child>,
    stdin: Option<ChildStdin>,
    request_id: u64,
    initialized: bool,
    /// Lines from the stdout reader; `None` signals EOF.
    stdout_queue: Option<Receiver<Option<String>>>,
    /// Tail-bounded stderr buffer (at most [`STDERR_MAX_LINES`] lines).
    ///
    /// Tail-bounded because `read_stderr_nonblocking` only ever returns the
    /// last few hundred bytes, and the callers that classify a failure match
    /// the *last* traceback line; an unbounded buffer would retain the entire
    /// stream from a chatty launcher.  Append through
    /// [`LocalMCPClient::append_stderr_chunk`] to keep the bound.
    pub stderr_chunks: Arc<Mutex<VecDeque<String>>>,
    reader_stdout: Option<JoinHandle<()>>,
    reader_stderr: Option<JoinHandle<()>>,
}

/// Push onto a bounded deque, dropping from the front (`deque(maxlen=...)`).
fn push_bounded(chunks: &Mutex<VecDeque<String>>, line: String) {
    let mut guard = chunks.lock();
    while guard.len() >= STDERR_MAX_LINES {
        guard.pop_front();
    }
    guard.push_back(line);
}

/// Wait up to `timeout` for a thread to finish, then join it if it did.
fn join_with_timeout(handle: JoinHandle<()>, timeout: Duration) -> Option<JoinHandle<()>> {
    let deadline = Instant::now() + timeout;
    while !handle.is_finished() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    if handle.is_finished() {
        let _ = handle.join();
        None
    } else {
        Some(handle)
    }
}

impl LocalMCPClient {
    /// A client for `command args...` (no env overrides, no cwd, 30 s timeout).
    pub fn new(command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            command: command.into(),
            args,
            env: HashMap::new(),
            cwd: None,
            timeout: 30.0,
            process: None,
            stdin: None,
            request_id: 0,
            initialized: false,
            stdout_queue: None,
            stderr_chunks: Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_MAX_LINES))),
            reader_stdout: None,
            reader_stderr: None,
        }
    }

    /// Extra environment merged over the inherited one.
    pub fn with_env(mut self, env: Option<HashMap<String, String>>) -> Self {
        self.env = env.unwrap_or_default();
        self
    }

    /// Working directory (`~` expanded against the real home).
    pub fn with_cwd(mut self, cwd: Option<&str>) -> Self {
        self.cwd = cwd.filter(|c| !c.is_empty()).map(expand_home);
        self
    }

    pub fn with_timeout(mut self, timeout: f64) -> Self {
        self.timeout = timeout;
        self
    }

    /// Append one stderr line, keeping only the last [`STDERR_MAX_LINES`].
    pub fn append_stderr_chunk(&self, line: impl Into<String>) {
        push_bounded(&self.stderr_chunks, line.into());
    }

    /// Number of stderr lines currently retained.
    pub fn stderr_len(&self) -> usize {
        self.stderr_chunks.lock().len()
    }

    /// The exact program to exec, as a full path where one can be found.
    ///
    /// The configured command and its arguments are run as written — nothing
    /// here rewrites a launcher's invocation.  What it does do is answer
    /// *which* `uvx` (or `npx`, or `python`): the app's `PATH` is not the
    /// shell's, so a command a user verified in a terminal can be missing from
    /// the process that launches it.  Falls through to the bare name when
    /// nothing matches, so the OS produces the error rather than this method
    /// inventing one.
    ///
    /// Built-in `-m smartloop.*` servers run under the current executable
    /// (honouring `SLP_EXE`).  An absolute command is kept as written — except
    /// inside the macOS App Sandbox, where only the app bundle and the
    /// container may be exec'd, so a copy of the same program next to the
    /// `slp` binary or under the framework home is preferred.  Otherwise
    /// `PATH` from `env` wins, then the anchors (the executable's own
    /// directory, then the framework home), then the bare name.
    pub fn resolve_command(&self, env: &HashMap<String, String>) -> String {
        // Built-in MCP servers are launched as `<interpreter> -m smartloop.<...>`.
        // The interpreter path is captured at registration time and persisted
        // per project, so it goes stale across dev/binary runs and version
        // upgrades.  Always run our own modules under the *current* runtime
        // executable, which honours `-m <module>`.
        if self.is_builtin_module() {
            return runtime_executable();
        }

        if Path::new(&self.command).is_absolute() {
            // Outside the sandbox an absolute command wins outright: the app
            // does not silently redirect a path the user configured.  Inside
            // it, the user's own install is refused with EPERM however the
            // path was written, so the copy the app may run is preferred —
            // and with no such copy the path is kept, so the spawn fails
            // honestly rather than running *some other* program.
            if crate::paths::confined_home().is_some() {
                if let Some(basename) = Path::new(&self.command).file_name() {
                    for anchor in Self::resolution_anchors() {
                        let candidate = anchor.join(basename);
                        if access_check(&candidate) {
                            return candidate.to_string_lossy().into_owned();
                        }
                    }
                }
            }
            return self.command.clone();
        }

        if let Some(found) = which(&self.command, env.get("PATH").map(String::as_str)) {
            return found;
        }

        // Entry-point scripts installed alongside the executable (pipx, a
        // venv's bin) — or a server the app ships next to the `slp` binary —
        // are not always on the PATH a GUI process inherits; nor is one a
        // project keeps under the framework home.  A bare name or a relative
        // path is looked up against those anchors, after PATH: they are the
        // fallback, not an override.
        for anchor in Self::resolution_anchors() {
            let candidate = anchor.join(&self.command);
            if access_check(&candidate) {
                return candidate.to_string_lossy().into_owned();
            }
        }

        self.command.clone()
    }

    /// Where a server the app ships with can live: the directory of the
    /// running executable (the app bundle when packaged), then the framework
    /// home (`SLP_HOME`, else `~/.smartloop`).
    fn resolution_anchors() -> Vec<std::path::PathBuf> {
        let mut anchors: Vec<std::path::PathBuf> = Vec::new();
        let exe = runtime_executable();
        if let Some(dir) = Path::new(&exe).parent().filter(|d| !d.as_os_str().is_empty()) {
            anchors.push(dir.to_path_buf());
        }
        if let Ok(settings) = crate::config::AppSettings::try_new() {
            let home = std::path::PathBuf::from(settings.home_dir());
            if !home.as_os_str().is_empty() && !anchors.contains(&home) {
                anchors.push(home);
            }
        }
        anchors
    }

    /// Return an actionable hint for known launcher failures, else `""`.
    ///
    /// `uvx` is only a thin trampoline that execs the real `uv` binary, which
    /// it locates solely as a sibling next to itself (it does not search
    /// PATH).  A host with `uvx` but no adjacent `uv` fails with "Could not
    /// find the `uv` binary".  Point the user at the fix.
    pub fn startup_hint(stderr: &str) -> String {
        if !stderr.is_empty() && stderr.contains("Could not find the `uv` binary") {
            return "\nHint: `uvx` requires the `uv` binary installed alongside it. \
                    Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` \
                    (Windows: `powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"`)."
                .to_string();
        }
        String::new()
    }

    fn is_builtin_module(&self) -> bool {
        self.args.len() >= 2 && self.args[0] == "-m" && self.args[1].starts_with("smartloop.")
    }

    /// Spawn the server and start the pipe readers (`__enter__`).
    pub fn enter(&mut self) -> Result<(), MCPClientError> {
        if self.args.is_empty() {
            return Err(MCPLaunchError(format!(
                "Cannot launch MCP server: command '{}' has no arguments. \
                 Provide at least the module or package name (e.g. args=['-m', 'smartloop.api.mcp.system_server'] \
                 or args=['yahoo-finance-mcp']).",
                self.command
            ))
            .into());
        }

        let mut merged_env: HashMap<String, String> = std::env::vars().collect();
        merged_env.extend(self.env.iter().map(|(k, v)| (k.clone(), v.clone())));
        let resolved_command = self.resolve_command(&merged_env);
        // The exact command line, as it goes to the OS: what a user configured
        // is what runs, and when it does not start this is the line that says
        // which program was actually looked for.
        let mut line = vec![resolved_command.clone()];
        line.extend(self.args.iter().cloned());
        tracing::info!(target: LOGGER, "[mcp-local] launching {}", line.join(" "));
        // (Python also prepended the package root to PYTHONPATH for a built-in
        // `-m smartloop.*` server; the Rust binary carries its own modules, so
        // there is nothing to put on a search path.)

        let mut cmd = Command::new(&resolved_command);
        cmd.args(&self.args)
            .env_clear()
            .envs(&merged_env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(cwd) = &self.cwd {
            cmd.current_dir(cwd);
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| MCPClientError::Other(spawn_error_message(&e, &resolved_command)))?;
        self.stdin = child.stdin.take();
        self.process = Some(child);

        // Anything that fails past this point must still reap the child and
        // close its pipes, or the process and its three pipe FDs stay alive
        // for the whole of the caller's retry ladder.
        if let Err(e) = self.after_spawn() {
            self.exit();
            return Err(e);
        }
        Ok(())
    }

    fn after_spawn(&mut self) -> Result<(), MCPClientError> {
        // For external processes (uvx, npx, …) give the launcher a moment to
        // either start the MCP server or fail visibly.  Built-in modules don't
        // need this guard.
        if !self.is_builtin_module() {
            // Poll up to 2 s in 50 ms steps waiting for the process to either
            // become an MCP server (still alive → break) or exit immediately.
            let deadline = Instant::now() + Duration::from_secs(2);
            while Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(50));
                let process = self.process.as_mut().expect("process spawned");
                let status = process
                    .try_wait()
                    .map_err(|e| MCPClientError::Other(e.to_string()))?;
                let Some(status) = status else {
                    // Still running — assume the MCP server is ready.
                    break;
                };
                // Process exited — read all available stderr now so the
                // caller's hint-retry logic can inspect it.  No reader threads
                // are running yet, so a direct blocking read is safe (the pipe
                // is at EOF).
                let mut stderr = String::new();
                if let Some(pipe) = process.stderr.as_mut() {
                    let mut bytes = Vec::new();
                    if pipe.read_to_end(&mut bytes).is_ok() {
                        stderr = String::from_utf8_lossy(&bytes).into_owned();
                    }
                }
                let code = returncode(&status);
                let trimmed = stderr.trim();
                return Err(MCPLaunchError(format!(
                    "MCP server process exited during startup (exit code {code}).{}{}",
                    Self::startup_hint(&stderr),
                    if trimmed.is_empty() { String::new() } else { format!("\nstderr: {trimmed}") }
                ))
                .into());
            }
        }

        self.start_readers();
        Ok(())
    }

    fn start_readers(&mut self) {
        let Some(process) = self.process.as_mut() else { return };
        let stdout = process.stdout.take();
        let stderr = process.stderr.take();

        let (tx, rx) = mpsc::channel::<Option<String>>();
        self.stdout_queue = Some(rx);
        self.reader_stdout = Some(std::thread::spawn(move || {
            if let Some(stdout) = stdout {
                // MCP stdio is UTF-8 by spec; decode leniently so a stray byte
                // never tears down the stream.
                let mut reader = BufReader::new(stdout);
                let mut buf = Vec::new();
                loop {
                    buf.clear();
                    match reader.read_until(b'\n', &mut buf) {
                        Ok(0) => break, // EOF
                        Ok(_) => {
                            if tx.send(Some(String::from_utf8_lossy(&buf).into_owned())).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::debug!(target: LOGGER, "[mcp-local] stdout reader stopped: {}", e);
                            break;
                        }
                    }
                }
            }
            let _ = tx.send(None); // signal EOF to recv
        }));

        let chunks = Arc::clone(&self.stderr_chunks);
        self.reader_stderr = Some(std::thread::spawn(move || {
            let Some(stderr) = stderr else { return };
            let mut reader = BufReader::new(stderr);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match reader.read_until(b'\n', &mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => push_bounded(&chunks, String::from_utf8_lossy(&buf).into_owned()),
                }
            }
        }));
    }

    /// Reap the child, close its pipes and retire the readers (`__exit__`).
    /// Idempotent.
    pub fn exit(&mut self) {
        // Close stdin first so a well-behaved server sees EOF.
        self.stdin = None;
        if let Some(mut process) = self.process.take() {
            drop(process.stdin.take());
            Self::terminate(&mut process);
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut reaped = false;
            while Instant::now() < deadline {
                match process.try_wait() {
                    Ok(Some(_)) => {
                        reaped = true;
                        break;
                    }
                    Ok(None) => std::thread::sleep(Duration::from_millis(10)),
                    Err(_) => break,
                }
            }
            if !reaped {
                let _ = process.kill();
                let _ = process.wait();
            }
            // Close the pipes explicitly (those the readers did not take).
            drop(process.stdout.take());
            drop(process.stderr.take());
        }
        // Retire the readers so they can't outlive the client; the bounded
        // join is belt-and-braces — a reader still parked on a pipe a
        // grandchild holds open is detached.
        if let Some(handle) = self.reader_stdout.take() {
            let _ = join_with_timeout(handle, Duration::from_secs(1));
        }
        if let Some(handle) = self.reader_stderr.take() {
            let _ = join_with_timeout(handle, Duration::from_secs(1));
        }
        self.stdout_queue = None;
        self.initialized = false;
    }

    /// `Popen.terminate()`: SIGTERM on Unix, `TerminateProcess` elsewhere.
    fn terminate(process: &mut Child) {
        #[cfg(unix)]
        {
            if let Ok(Some(_)) = process.try_wait() {
                return;
            }
            if let Ok(pid) = libc::pid_t::try_from(process.id()) {
                // SAFETY: sending a signal to our own child's pid.
                unsafe {
                    libc::kill(pid, libc::SIGTERM);
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = process.kill();
        }
    }

    fn next_id(&mut self) -> u64 {
        self.request_id += 1;
        self.request_id
    }

    /// Return up to `max_bytes` of the stderr tail collected by the reader.
    ///
    /// Gives the reader a brief moment to flush any pending diagnostic output
    /// so error messages can carry a useful stderr tail.  Keeps the *tail*
    /// rather than the head — for a Python traceback the actual exception is
    /// on the last line, which is what callers classify.
    pub fn read_stderr_nonblocking(&mut self, max_bytes: usize) -> String {
        if let Some(handle) = self.reader_stderr.take() {
            self.reader_stderr = join_with_timeout(handle, Duration::from_millis(500));
        }
        let data: String = self.stderr_chunks.lock().iter().map(String::as_str).collect();
        let trimmed = data.trim();
        let count = trimmed.chars().count();
        trimmed.chars().skip(count.saturating_sub(max_bytes)).collect()
    }

    fn stderr_suffix(&mut self, sep: &str) -> String {
        let stderr = self.read_stderr_nonblocking(500);
        if stderr.is_empty() { String::new() } else { format!("{sep}stderr: {stderr}") }
    }

    fn send(&mut self, message: &Value) -> Result<(), MCPClientError> {
        let Some(process) = self.process.as_mut() else {
            return Err(MCPClientError::Other("StdioMCPClient not started".to_string()));
        };
        // Check if the process has already exited before trying to write.
        if let Ok(Some(status)) = process.try_wait() {
            let code = returncode(&status);
            let suffix = self.stderr_suffix(" ");
            return Err(MCPLaunchError(format!(
                "MCP process exited (code {code}) before message could be sent.{suffix}"
            ))
            .into());
        }
        let line = python_json_dumps(message) + "\n";
        let Some(stdin) = self.stdin.as_mut() else {
            let suffix = self.stderr_suffix(" ");
            return Err(MCPLaunchError(format!(
                "MCP process closed its stdin unexpectedly (broken pipe).{suffix}"
            ))
            .into());
        };
        match stdin.write_all(line.as_bytes()).and_then(|_| stdin.flush()) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {
                let suffix = self.stderr_suffix(" ");
                Err(MCPLaunchError(format!(
                    "MCP process closed its stdin unexpectedly (broken pipe).{suffix}"
                ))
                .into())
            }
            Err(e) => Err(MCPClientError::Other(e.to_string())),
        }
    }

    fn timed_out(&self) -> MCPClientError {
        MCPProtocolError(format!("MCP tool call timed out after {}s", py_float(self.timeout))).into()
    }

    /// Read lines from stdout until a response matching `expected_id` arrives.
    fn recv(&mut self, expected_id: u64) -> Result<Map<String, Value>, MCPClientError> {
        if self.process.is_none() {
            return Err(MCPClientError::Other("StdioMCPClient not started".to_string()));
        }
        let deadline = Instant::now() + Duration::from_secs_f64(self.timeout.max(0.0));
        loop {
            let now = Instant::now();
            if now >= deadline {
                return Err(self.timed_out());
            }
            let remaining = deadline - now;
            let received = match &self.stdout_queue {
                Some(rx) => rx.recv_timeout(remaining),
                None => {
                    // No reader: nothing will ever arrive.
                    std::thread::sleep(remaining);
                    Err(RecvTimeoutError::Timeout)
                }
            };
            let line = match received {
                Ok(Some(line)) => line,
                Err(RecvTimeoutError::Timeout) => return Err(self.timed_out()),
                Ok(None) | Err(RecvTimeoutError::Disconnected) => {
                    // stdout reader hit EOF
                    let exited = self
                        .process
                        .as_mut()
                        .and_then(|p| p.try_wait().ok().flatten())
                        .map(|s| returncode(&s));
                    if let Some(code) = exited {
                        let suffix = self.stderr_suffix(" ");
                        return Err(MCPLaunchError(format!("MCP process exited (code {code}).{suffix}")).into());
                    }
                    let suffix = self.stderr_suffix(" ");
                    return Err(MCPProtocolError(format!("MCP process closed its stdout unexpectedly.{suffix}")).into());
                }
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let msg = match serde_json::from_str::<Value>(line) {
                Ok(Value::Object(msg)) => msg,
                Ok(_) | Err(_) => {
                    tracing::debug!(target: LOGGER, "Non-JSON line from MCP stdio process: {:?}", line);
                    continue;
                }
            };
            // Accept any message that has the right id; one without an id is
            // a server-sent notification — ignore it.
            let matches = match msg.get("id") {
                None | Some(Value::Null) => continue,
                Some(Value::Number(n)) => {
                    n.as_u64() == Some(expected_id) || n.as_f64() == Some(expected_id as f64)
                }
                Some(_) => false,
            };
            if matches {
                return Ok(msg);
            }
        }
    }

    fn request(&mut self, method: &str, params: Option<Value>) -> Result<Map<String, Value>, MCPClientError> {
        let req_id = self.next_id();
        let mut req = Map::new();
        req.insert("jsonrpc".into(), Value::String("2.0".into()));
        req.insert("id".into(), Value::from(req_id));
        req.insert("method".into(), Value::String(method.to_string()));
        if let Some(p) = params {
            req.insert("params".into(), p);
        }
        self.send(&Value::Object(req))?;
        let resp = self.recv(req_id)?;
        if let Some(err) = resp.get("error") {
            return Err(MCPProtocolError(jsonrpc_error_message(err)).into());
        }
        match resp.get("result") {
            None => Ok(Map::new()),
            Some(Value::Object(r)) => Ok(r.clone()),
            Some(other) => Err(MCPClientError::Other(format!(
                "'{}' object has no attribute 'get'",
                crate::mcp::client::json_type_name(other)
            ))),
        }
    }

    fn notify(&mut self, method: &str, params: Option<Value>) -> Result<(), MCPClientError> {
        let mut msg = Map::new();
        msg.insert("jsonrpc".into(), Value::String("2.0".into()));
        msg.insert("method".into(), Value::String(method.to_string()));
        if let Some(p) = params {
            msg.insert("params".into(), p);
        }
        self.send(&Value::Object(msg))
    }

    fn initialize_params() -> Value {
        serde_json::json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": CLIENT_INFO_NAME, "version": CLIENT_INFO_VERSION},
        })
    }

    fn ensure_initialized(&mut self) -> Result<(), MCPClientError> {
        if self.initialized {
            return Ok(());
        }
        self.request("initialize", Some(Self::initialize_params()))?;
        self.notify("notifications/initialized", Some(Value::Object(Map::new())))?;
        self.initialized = true;
        Ok(())
    }

    fn discover(&mut self) -> Result<MCPConnectResult, MCPClientError> {
        let init_result = self.request("initialize", Some(Self::initialize_params()))?;
        let server_info = server_info_from(&init_result)?;
        self.notify("notifications/initialized", Some(Value::Object(Map::new())))?;
        self.initialized = true;

        let tools_result = self.request("tools/list", Some(Value::Object(Map::new())))?;
        let tools = tools_from(&tools_result)?;
        Ok(MCPConnectResult { success: true, tools, server_info, ..Default::default() })
    }

    /// Perform the MCP handshake and return all advertised tools.
    ///
    /// Launch errors propagate; protocol and other errors become an
    /// unsuccessful result.
    pub fn connect_and_discover_tools(&mut self) -> Result<MCPConnectResult, MCPClientError> {
        match self.discover() {
            Ok(result) => Ok(result),
            Err(e @ MCPClientError::Launch(_)) => Err(e),
            Err(MCPClientError::Protocol(e)) => {
                Ok(MCPConnectResult { success: false, error: Some(e.to_string()), ..Default::default() })
            }
            Err(e) => Ok(MCPConnectResult {
                success: false,
                error: Some(format!("Unexpected error: {e}")),
                ..Default::default()
            }),
        }
    }

    /// Call a single tool and return its result.
    pub fn call_tool(&mut self, tool_name: &str, arguments: Option<&Map<String, Value>>) -> MCPToolCallResult {
        let outcome = self.ensure_initialized().and_then(|_| {
            self.request(
                "tools/call",
                Some(serde_json::json!({
                    "name": tool_name,
                    "arguments": Value::Object(arguments.cloned().unwrap_or_default()),
                })),
            )
        });
        match outcome {
            Ok(result) => {
                let (content, is_error) = call_result_parts(&result);
                MCPToolCallResult { success: true, content, is_error, ..Default::default() }
            }
            Err(MCPClientError::Protocol(e)) => {
                tracing::error!(target: LOGGER, "[mcp-local] Tool call '{}' failed: {}", tool_name, e);
                MCPToolCallResult { success: false, error: Some(e.to_string()), ..Default::default() }
            }
            Err(e) => {
                tracing::error!(target: LOGGER, "[mcp-local] Tool call '{}' unexpected error: {}", tool_name, e);
                MCPToolCallResult {
                    success: false,
                    error: Some(format!("Unexpected error: {e}")),
                    ..Default::default()
                }
            }
        }
    }
}

impl Drop for LocalMCPClient {
    fn drop(&mut self) {
        if self.process.is_some() || self.reader_stdout.is_some() || self.reader_stderr.is_some() {
            self.exit();
        }
    }
}
