//! Port of `smartloop/api/mcp/system_server.py`.
//!
//! MCP system tools server – bash, file-read, file-write, grep.
//!
//! Tools fail (rather than return an "Error: ..." string) on failure so the
//! MCP result is flagged `is_error=true`.  The orchestrator then skips the step
//! instead of treating the error text as authoritative tool output, which
//! would otherwise suppress the workspace/document search fallback.
//!
//! Served over stdio by the hidden `slp __system-server` subcommand (Python:
//! `python -m smartloop.api.mcp.system_server`).

#![allow(unused_imports)]

use std::path::Path;

/// Logger name of this module.
pub const LOGGER: &str = "smartloop.api.mcp.system_server";

/// MCP server name (`FastMCP("smartloop")`).
pub const SERVER_NAME: &str = "smartloop";

/// Strip a single ```` ``` ```` fence (with optional language tag) that small
/// models often wrap file content in, returning the inner text.
pub fn extract_fenced_content(text: &str) -> String {
    let _ = text;
    todo!()
}

/// Tool `bash`: execute a bash command (`bash -c`).  Returns
/// `"exit code N\n<stdout+stderr stripped>"`; only a timeout or spawn failure
/// is an error (`"command timed out after {timeout}s"`).
pub async fn bash(command: &str, cwd: Option<&str>, timeout: u64) -> anyhow::Result<String> {
    let _ = (command, cwd, timeout);
    todo!()
}

/// Tool `file-read`: read a file, optionally filtered to lines matching
/// `query` (with context), truncated at 4000 chars.
pub async fn file_read(path: &str, query: Option<&str>) -> anyhow::Result<String> {
    let _ = (path, query);
    todo!()
}

/// Tool `file-write`: write `content` (fence-stripped) to `path`, creating
/// parent directories.
pub async fn file_write(path: &str, content: Option<&str>) -> anyhow::Result<String> {
    let _ = (path, content);
    todo!()
}

/// Tool `grep`: `grep -rn --include <glob> <pattern> .`, first 50 matches.
/// No matches or grep failure is an error.
pub async fn grep(pattern: &str, glob: &str, timeout: u64) -> anyhow::Result<String> {
    let _ = (pattern, glob, timeout);
    todo!()
}

/// The MCP server handler exposing the four tools (`bash`, `file-read`,
/// `file-write`, `grep`) with the same names, argument schemas and defaults
/// (`cwd=None`, `timeout=30`, `query=None`, `glob="*"`).
#[derive(Debug, Clone, Default)]
pub struct SystemServer;

impl SystemServer {
    /// A new server instance.
    pub fn new() -> Self {
        SystemServer
    }
}

/// Serve [`SystemServer`] over stdio until the client disconnects
/// (`mcp.run(transport="stdio")`).
pub async fn run_stdio() -> anyhow::Result<()> {
    todo!()
}
