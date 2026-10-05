//! Local MCP client for subprocess-based (stdio) MCP servers.
//!
//! MCP over stdio uses newline-delimited JSON-RPC 2.0:
//!   - Send a JSON object + `\n` to the process stdin
//!   - Read a JSON object + `\n` from the process stdout
//!
//! Port of `smartloop/mcp/local_client.py`.

use std::collections::{HashMap, VecDeque};
use std::process::{Child, ChildStdin};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::thread::JoinHandle;

use parking_lot::Mutex;
use serde_json::{Map, Value};

#[allow(unused_imports)]
use crate::mcp::client::{
    MCPClientError, MCPConnectResult, MCPLaunchError, MCPProtocolError, MCPServerInfo, MCPToolCallResult,
};
#[allow(unused_imports)]
use crate::paths::expand_home;
#[allow(unused_imports)]
use crate::workspace::MCPToolDefinition;

pub const PROTOCOL_VERSION: &str = "2024-11-05";
pub const CLIENT_INFO_NAME: &str = "smartloop-cli";
pub const CLIENT_INFO_VERSION: &str = "1.0.0";

/// Lines of stderr retained per client.
pub const STDERR_MAX_LINES: usize = 200;

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
    pub stderr_chunks: Arc<Mutex<VecDeque<String>>>,
    reader_stdout: Option<JoinHandle<()>>,
    reader_stderr: Option<JoinHandle<()>>,
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
    pub fn with_cwd(self, cwd: Option<&str>) -> Self {
        todo!()
    }

    pub fn with_timeout(mut self, timeout: f64) -> Self {
        self.timeout = timeout;
        self
    }

    /// The exact program to exec, as a full path where one can be found.
    ///
    /// Built-in `-m smartloop.*` servers run under the current executable
    /// (honouring `SLP_EXE`); absolute commands are kept; otherwise `PATH` from
    /// `env`, then the executable's own directory, then the bare name.
    pub fn resolve_command(&self, env: &HashMap<String, String>) -> String {
        todo!()
    }

    /// Return an actionable hint for known launcher failures, else `""`.
    pub fn startup_hint(stderr: &str) -> String {
        todo!()
    }

    fn is_builtin_module(&self) -> bool {
        todo!()
    }

    /// Spawn the server and start the pipe readers (`__enter__`).
    pub fn enter(&mut self) -> Result<(), MCPClientError> {
        todo!()
    }

    fn start_readers(&mut self) {
        todo!()
    }

    /// Reap the child, close its pipes and retire the readers (`__exit__`).
    /// Idempotent.
    pub fn exit(&mut self) {
        todo!()
    }

    fn next_id(&mut self) -> u64 {
        todo!()
    }

    /// Return up to `max_bytes` of the stderr tail collected by the reader.
    pub fn read_stderr_nonblocking(&mut self, max_bytes: usize) -> String {
        todo!()
    }

    fn send(&mut self, message: &Value) -> Result<(), MCPClientError> {
        todo!()
    }

    /// Read lines from stdout until a response matching `expected_id` arrives.
    fn recv(&mut self, expected_id: u64) -> Result<Map<String, Value>, MCPClientError> {
        todo!()
    }

    fn request(&mut self, method: &str, params: Option<Value>) -> Result<Map<String, Value>, MCPClientError> {
        todo!()
    }

    fn notify(&mut self, method: &str, params: Option<Value>) -> Result<(), MCPClientError> {
        todo!()
    }

    fn ensure_initialized(&mut self) -> Result<(), MCPClientError> {
        todo!()
    }

    /// Perform the MCP handshake and return all advertised tools.
    ///
    /// Launch errors propagate; protocol and other errors become an
    /// unsuccessful result.
    pub fn connect_and_discover_tools(&mut self) -> Result<MCPConnectResult, MCPClientError> {
        todo!()
    }

    /// Call a single tool and return its result.
    pub fn call_tool(&mut self, tool_name: &str, arguments: Option<&Map<String, Value>>) -> MCPToolCallResult {
        todo!()
    }
}

impl Drop for LocalMCPClient {
    fn drop(&mut self) {
        if self.process.is_some() || self.reader_stdout.is_some() || self.reader_stderr.is_some() {
            self.exit();
        }
    }
}
