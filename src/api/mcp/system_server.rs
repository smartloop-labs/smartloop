//! Port of `smartloop/api/mcp/system_server.py`.
//!
//! MCP system tools server – bash, file-read, file-write, grep.
//!
//! Tools fail (rather than return an "Error: ..." string) on failure so the
//! MCP result is flagged `is_error=true`.  The orchestrator then skips the step
//! instead of treating the error text as authoritative tool output, which
//! would otherwise suppress the workspace/document search fallback — e.g.
//! "find my pitch" with a hallucinated path or a grep that misses must fall
//! through to semantic search over the indexed files, not dead-end on
//! "file not found" / "no matches".
//!
//! Served over stdio by the hidden `slp __system-server` subcommand (Python:
//! `python -m smartloop.api.mcp.system_server`).

use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use once_cell::sync::Lazy;
use regex::Regex;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ContentBlock, Implementation, JsonObject, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData as McpError, RoleServer, ServerHandler, ServiceExt};
use serde_json::{json, Value};

/// Logger name of this module.
pub const LOGGER: &str = "smartloop.api.mcp.system_server";

/// MCP server name (`FastMCP("smartloop")`).
pub const SERVER_NAME: &str = "smartloop";

/// Default `timeout` (seconds) of the `bash` and `grep` tools.
pub const DEFAULT_TIMEOUT: u64 = 30;

static FENCE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)```[^\n]*\n(.*?)```").expect("valid regex"));

/// Strip a single ```` ``` ```` fence (with optional language tag) that small
/// models often wrap file content in, returning the inner text.
pub fn extract_fenced_content(text: &str) -> String {
    match FENCE_RE.captures(text) {
        Some(captures) => captures.get(1).map(|m| m.as_str().to_string()).unwrap_or_default(),
        None => text.to_string(),
    }
}

/// Decode process output / file bytes the way Python's text mode does:
/// invalid UTF-8 replaced, and universal newlines (`\r\n` and `\r` → `\n`).
fn decode_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).replace("\r\n", "\n").replace('\r', "\n")
}

/// The first `n` characters of `text` (Python `text[:n]`).
fn char_prefix(text: &str, n: usize) -> &str {
    match text.char_indices().nth(n) {
        Some((index, _)) => &text[..index],
        None => text,
    }
}

/// Run `program args...` capturing stdout/stderr, killing it after
/// `timeout` seconds.  `Ok(None)` means it timed out.
async fn run_with_timeout(
    program: &str,
    args: &[&str],
    cwd: Option<&str>,
    timeout: u64,
) -> anyhow::Result<Option<std::process::Output>> {
    let mut command = tokio::process::Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(cwd) = cwd {
        if !Path::new(cwd).is_dir() {
            anyhow::bail!("[Errno 2] No such file or directory: '{cwd}'");
        }
        command.current_dir(cwd);
    }
    let child = command.spawn().map_err(|e| anyhow::anyhow!("{e}"))?;
    match tokio::time::timeout(Duration::from_secs(timeout), child.wait_with_output()).await {
        Ok(output) => Ok(Some(output.map_err(|e| anyhow::anyhow!("{e}"))?)),
        // Dropping the future drops the child, which kills it (kill_on_drop).
        Err(_) => Ok(None),
    }
}

/// Tool `bash`: execute a bash command (`bash -c`).  Returns
/// `"exit code N\n<stdout+stderr stripped>"`; only a timeout or spawn failure
/// is an error (`"command timed out after {timeout}s"`).
pub async fn bash(command: &str, cwd: Option<&str>, timeout: u64) -> anyhow::Result<String> {
    tracing::info!(
        target: LOGGER,
        "[bash] Executing: {}, cwd={}",
        char_prefix(command, 200),
        cwd.unwrap_or("None")
    );
    // A command that ran to completion is a real result even with a nonzero
    // exit code — its output (e.g. failing tests) belongs in the answer.
    // Only failures to produce a result at all (timeout, spawn error) raise.
    let Some(output) = run_with_timeout("bash", &["-c", command], cwd, timeout).await? else {
        anyhow::bail!("command timed out after {timeout}s");
    };

    let mut text = String::new();
    text.push_str(&decode_text(&output.stdout));
    text.push_str(&decode_text(&output.stderr));
    // Python's returncode is negative for a signal-terminated process.
    let returncode = match output.status.code() {
        Some(code) => code,
        None => {
            #[cfg(unix)]
            {
                use std::os::unix::process::ExitStatusExt;
                output.status.signal().map(|s| -s).unwrap_or(-1)
            }
            #[cfg(not(unix))]
            {
                -1
            }
        }
    };
    Ok(format!("exit code {returncode}\n{}", text.trim()))
}

/// Tool `file-read`: read a file, optionally filtered to lines matching
/// `query` (with context), truncated at 4000 chars.
pub async fn file_read(path: &str, query: Option<&str>) -> anyhow::Result<String> {
    if path.is_empty() {
        anyhow::bail!("no file path provided");
    }
    if !Path::new(path).is_file() {
        anyhow::bail!("file not found: {path}");
    }

    tracing::info!(target: LOGGER, "[file-read] Reading '{}', query='{}'", path, query.unwrap_or(""));

    let content = match tokio::fs::read(path).await {
        Ok(bytes) => decode_text(&bytes),
        Err(e) => anyhow::bail!("error reading file: {e}"),
    };

    if let Some(query) = query.filter(|q| !q.is_empty()) {
        let lines: Vec<&str> = content.split('\n').collect();
        let mut matching: Vec<String> = Vec::new();
        let query_lower = query.to_lowercase();
        for (offset, line) in lines.iter().enumerate() {
            let i = offset + 1;
            if line.to_lowercase().contains(&query_lower) {
                let start = i.saturating_sub(3);
                let end = (i + 2).min(lines.len());
                for j in start..end {
                    let entry = format!("{}: {}", j + 1, lines[j]);
                    if !matching.contains(&entry) {
                        matching.push(entry);
                    }
                }
                if matching.len() > 100 {
                    break;
                }
            }
        }
        if !matching.is_empty() {
            return Ok(format!("File: {path}\nMatching lines for '{query}':\n\n{}", matching.join("\n")));
        }
    }

    let length = content.chars().count();
    if length > 4000 {
        return Ok(format!(
            "File: {path} ({length} chars)\n\n{}\n\n... (truncated)",
            char_prefix(&content, 4000)
        ));
    }
    Ok(format!("File: {path}\n\n{content}"))
}

/// Tool `file-write`: write `content` (fence-stripped) to `path`, creating
/// parent directories.
pub async fn file_write(path: &str, content: Option<&str>) -> anyhow::Result<String> {
    if path.is_empty() {
        anyhow::bail!("no file path provided");
    }
    let Some(content) = content else {
        anyhow::bail!("no content provided");
    };

    let content = extract_fenced_content(content);
    let length = content.chars().count();

    tracing::info!(target: LOGGER, "[file-write] Writing to '{}', content_len={}", path, length);
    let parent = Path::new(path).parent().map(|p| p.to_path_buf()).filter(|p| !p.as_os_str().is_empty());
    let parent = parent.unwrap_or_else(|| Path::new(".").to_path_buf());
    tokio::fs::create_dir_all(&parent).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    match tokio::fs::write(path, content.as_bytes()).await {
        Ok(()) => Ok(format!(
            "Successfully wrote to {path} ({length} chars). IMPORTANT: Reference this file path in your \
             response so the user can locate and download it from the chat log."
        )),
        Err(e) => anyhow::bail!("error writing file: {e}"),
    }
}

/// Tool `grep`: `grep -rn --include <glob> <pattern> .`, first 50 matches.
/// No matches or grep failure is an error.
pub async fn grep(pattern: &str, glob: &str, timeout: u64) -> anyhow::Result<String> {
    if pattern.is_empty() {
        anyhow::bail!("no search pattern provided");
    }

    tracing::info!(target: LOGGER, "[grep] Searching for '{}' (glob={})", pattern, glob);
    let Some(output) = run_with_timeout("grep", &["-rn", "--include", glob, pattern, "."], None, timeout).await?
    else {
        anyhow::bail!("search timed out after {timeout}s");
    };

    // grep exits 0 on match, 1 on no match, >=2 on error (stdout is empty
    // either way — without the returncode check a bad regex would be
    // misreported as a clean miss).
    let code = output.status.code().unwrap_or(2);
    if code > 1 {
        anyhow::bail!("grep failed: {}", decode_text(&output.stderr).trim());
    }

    let stdout = decode_text(&output.stdout);
    let mut text = stdout.trim().to_string();
    if text.is_empty() {
        anyhow::bail!("no matches found for '{pattern}'");
    }
    let lines: Vec<&str> = text.split('\n').collect();
    if lines.len() > 50 {
        text = format!(
            "{}\n\n... ({} total matches, showing first 50)",
            lines[..50].join("\n"),
            lines.len()
        );
    }
    Ok(format!("Search results for '{pattern}':\n\n{text}"))
}

/// A JSON object literal as an `Arc<JsonObject>` (tool schemas).
fn schema(value: Value) -> Arc<JsonObject> {
    match value {
        Value::Object(map) => Arc::new(map),
        _ => Arc::new(JsonObject::new()),
    }
}

/// The `{"result": <str>}` output schema FastMCP derives from `-> str`.
fn output_schema(tool: &str) -> Arc<JsonObject> {
    schema(json!({
        "properties": {"result": {"title": "Result", "type": "string"}},
        "required": ["result"],
        "title": format!("{tool}Output"),
        "type": "object",
    }))
}

/// The tool definitions, with the same names, descriptions, argument schemas
/// and defaults FastMCP derived from the Python signatures.
pub fn tool_definitions() -> Vec<Tool> {
    let mut tools = vec![
        Tool::new(
            "bash",
            "Execute a bash command.\n\n    Args:\n        command: The bash command to execute.\n        \
             cwd: Working directory for the command.\n        timeout: Timeout in seconds.\n    ",
            schema(json!({
                "properties": {
                    "command": {"title": "Command", "type": "string"},
                    "cwd": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null, "title": "Cwd"},
                    "timeout": {"default": 30, "title": "Timeout", "type": "integer"},
                },
                "required": ["command"],
                "title": "bashArguments",
                "type": "object",
            })),
        ),
        Tool::new(
            "file-read",
            "Read a file from the working directory.\n\n    Args:\n        path: File path to read.\n        \
             query: Optional search query to filter matching lines.\n    ",
            schema(json!({
                "properties": {
                    "path": {"title": "Path", "type": "string"},
                    "query": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null, "title": "Query"},
                },
                "required": ["path"],
                "title": "file_readArguments",
                "type": "object",
            })),
        ),
        Tool::new(
            "file-write",
            "Write content to a file in the working directory.\n\n    Args:\n        path: File path to write.\n        \
             content: Content to write.\n    ",
            schema(json!({
                "properties": {
                    "path": {"title": "Path", "type": "string"},
                    "content": {"title": "Content", "type": "string"},
                },
                "required": ["path", "content"],
                "title": "file_writeArguments",
                "type": "object",
            })),
        ),
        Tool::new(
            "grep",
            "Recursively search files in the working directory for a pattern.\n\n    Args:\n        \
             pattern: Text or regex to search for.\n        glob: Optional filename filter (e.g. \"*.py\"). \
             Defaults to all files.\n        timeout: Timeout in seconds.\n    ",
            schema(json!({
                "properties": {
                    "pattern": {"title": "Pattern", "type": "string"},
                    "glob": {"default": "*", "title": "Glob", "type": "string"},
                    "timeout": {"default": 30, "title": "Timeout", "type": "integer"},
                },
                "required": ["pattern"],
                "title": "grepArguments",
                "type": "object",
            })),
        ),
    ];
    for tool in &mut tools {
        let key = tool.name.replace('-', "_");
        tool.output_schema = Some(output_schema(&key));
    }
    tools
}

/// Argument-validation failure, worded like pydantic's (which FastMCP surfaced).
fn validation_error(tool: &str, field: &str, problem: &str) -> anyhow::Error {
    let model = format!("{}Arguments", tool.replace('-', "_"));
    anyhow::anyhow!("1 validation error for {model}\n{field}\n  {problem}")
}

/// A required string argument.
fn required_str<'a>(tool: &str, args: &'a JsonObject, field: &str) -> anyhow::Result<&'a str> {
    match args.get(field) {
        None => Err(validation_error(tool, field, "Field required")),
        Some(Value::String(s)) => Ok(s.as_str()),
        Some(_) => Err(validation_error(tool, field, "Input should be a valid string")),
    }
}

/// An optional (nullable) string argument.
fn optional_str<'a>(tool: &str, args: &'a JsonObject, field: &str) -> anyhow::Result<Option<&'a str>> {
    match args.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.as_str())),
        Some(_) => Err(validation_error(tool, field, "Input should be a valid string")),
    }
}

/// An integer argument with a default (pydantic lax mode also takes an
/// integral float or a numeric string).
fn int_arg(tool: &str, args: &JsonObject, field: &str, default: u64) -> anyhow::Result<u64> {
    let invalid = || validation_error(tool, field, "Input should be a valid integer");
    match args.get(field) {
        None => Ok(default),
        Some(Value::Number(n)) => {
            if let Some(v) = n.as_u64() {
                Ok(v)
            } else if let Some(v) = n.as_i64() {
                Ok(v.max(0) as u64)
            } else if let Some(f) = n.as_f64().filter(|f| f.fract() == 0.0) {
                Ok(f.max(0.0) as u64)
            } else {
                Err(invalid())
            }
        }
        Some(Value::String(s)) => s.trim().parse::<i64>().map(|v| v.max(0) as u64).map_err(|_| invalid()),
        Some(_) => Err(invalid()),
    }
}

/// Dispatch one tool call by name.
pub async fn call_tool_by_name(name: &str, args: &JsonObject) -> anyhow::Result<String> {
    match name {
        "bash" => {
            let command = required_str(name, args, "command")?;
            let cwd = optional_str(name, args, "cwd")?;
            let timeout = int_arg(name, args, "timeout", DEFAULT_TIMEOUT)?;
            bash(command, cwd, timeout).await
        }
        "file-read" => {
            let path = required_str(name, args, "path")?;
            let query = optional_str(name, args, "query")?;
            file_read(path, query).await
        }
        "file-write" => {
            let path = required_str(name, args, "path")?;
            let content = required_str(name, args, "content")?;
            file_write(path, Some(content)).await
        }
        "grep" => {
            let pattern = required_str(name, args, "pattern")?;
            let glob = match args.get("glob") {
                None => "*",
                Some(Value::String(s)) => s.as_str(),
                Some(_) => return Err(validation_error(name, "glob", "Input should be a valid string")),
            };
            let timeout = int_arg(name, args, "timeout", DEFAULT_TIMEOUT)?;
            grep(pattern, glob, timeout).await
        }
        other => anyhow::bail!("Unknown tool: {other}"),
    }
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

    /// Run one tool call and shape the MCP result the way FastMCP did: the
    /// text plus `{"result": text}` structured content on success, an
    /// `is_error` result reading "Error executing tool <name>: <msg>" on
    /// failure.
    pub async fn call(&self, name: &str, args: &JsonObject) -> CallToolResult {
        let known = tool_definitions().iter().any(|t| t.name == name);
        match call_tool_by_name(name, args).await {
            Ok(text) => {
                let mut result = CallToolResult::success(vec![ContentBlock::text(text.clone())]);
                result.structured_content = Some(json!({"result": text}));
                result
            }
            Err(e) if known => {
                CallToolResult::error(vec![ContentBlock::text(format!("Error executing tool {name}: {e}"))])
            }
            Err(e) => CallToolResult::error(vec![ContentBlock::text(e.to_string())]),
        }
    }
}

impl ServerHandler for SystemServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(SERVER_NAME, crate::VERSION))
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(tool_definitions()))
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        tool_definitions().into_iter().find(|t| t.name == name)
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let args = request.arguments.unwrap_or_default();
        Ok(self.call(&request.name, &args).await)
    }
}

/// Serve [`SystemServer`] over stdio until the client disconnects
/// (`mcp.run(transport="stdio")`).
pub async fn run_stdio() -> anyhow::Result<()> {
    // stdout carries the protocol; logs go to stderr, as FastMCP's did.
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .try_init();
    let service = SystemServer::new()
        .serve((tokio::io::stdin(), tokio::io::stdout()))
        .await
        .map_err(|e| anyhow::anyhow!("MCP server initialization failed: {e}"))?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fenced_content_is_unwrapped() {
        assert_eq!(extract_fenced_content("```python\nprint(1)\n```"), "print(1)\n");
        assert_eq!(extract_fenced_content("plain"), "plain");
    }

    #[tokio::test]
    async fn bash_reports_exit_code_and_output() {
        let out = bash("echo hi; exit 3", None, 30).await.unwrap();
        assert_eq!(out, "exit code 3\nhi");
    }

    #[tokio::test]
    async fn bash_times_out() {
        let err = bash("sleep 5", None, 1).await.unwrap_err();
        assert_eq!(err.to_string(), "command timed out after 1s");
    }

    #[tokio::test]
    async fn file_round_trip_and_query() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join("f.txt");
        let path = path.to_str().unwrap();
        let msg = file_write(path, Some("```\na\nNeedle\nc\n```")).await.unwrap();
        assert!(msg.starts_with(&format!("Successfully wrote to {path} (")));
        let read = file_read(path, None).await.unwrap();
        assert_eq!(read, format!("File: {path}\n\na\nNeedle\nc\n"));
        let found = file_read(path, Some("needle")).await.unwrap();
        assert_eq!(found, format!("File: {path}\nMatching lines for 'needle':\n\n1: a\n2: Needle\n3: c\n4: "));
        assert!(file_read("", None).await.is_err());
        assert_eq!(
            file_read("/no/such/file", None).await.unwrap_err().to_string(),
            "file not found: /no/such/file"
        );
    }

    #[tokio::test]
    async fn errors_are_flagged() {
        let result = SystemServer::new().call("file-read", &JsonObject::new()).await;
        assert_eq!(result.is_error, Some(true));
    }
}
