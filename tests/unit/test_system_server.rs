//! Port of `tests/unit/test_system_server.py`.
//!
//! Tests for the smartloop system-tools MCP server (bash/file-read/file-write/
//! grep), invoked the same way the orchestrator invokes it — a real stdio
//! subprocess via `LocalMCPClient`, with `cwd` set to the work dir.
//!
//! The built-in server runs as `<runtime executable> -m
//! smartloop.api.mcp.system_server`; in the tests the runtime executable is
//! the built `slp` binary (`SLP_EXE`), as `sys.executable` is the interpreter
//! in Python's.

#![cfg(unix)]

use std::path::Path;

use serde_json::{json, Map, Value};
use serial_test::serial;

use smartloop::mcp::client::MCPToolCallResult;
use smartloop::mcp::local_client::LocalMCPClient;

use crate::shared::mcp::EnvGuard;

/// A started client over the system server, run in `cwd`.  The guard keeps
/// `SLP_EXE` pointed at the built binary for the test's duration.
fn client(cwd: &Path) -> (LocalMCPClient, EnvGuard) {
    let env = EnvGuard::set(&[("SLP_EXE", Some(env!("CARGO_BIN_EXE_slp")))]);
    let mut c = LocalMCPClient::new(
        env!("CARGO_BIN_EXE_slp"),
        vec!["-m".into(), "smartloop.api.mcp.system_server".into()],
    )
    .with_cwd(Some(&cwd.to_string_lossy()));
    c.enter().expect("the system server starts");
    (c, env)
}

fn args(value: Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap()
}

fn text(result: &MCPToolCallResult) -> String {
    result
        .content
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|b| match b {
            Value::Object(o) => o.get("text").and_then(Value::as_str).unwrap_or("").to_string(),
            other => other.to_string(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
#[serial]
fn test_file_write_then_read_relative_path() {
    // A relative path resolves against the subprocess cwd (the work dir).
    let tmp = tempfile::tempdir().unwrap();
    let (mut c, _env) = client(tmp.path());
    let w = c.call_tool("file-write", Some(&args(json!({"path": "out/report.md", "content": "hello world"}))));
    assert!(w.success, "{w:?}");
    assert_eq!(std::fs::read_to_string(tmp.path().join("out/report.md")).unwrap(), "hello world");

    let r = c.call_tool("file-read", Some(&args(json!({"path": "out/report.md"}))));
    assert!(r.success, "{r:?}");
    assert!(text(&r).contains("hello world"));
    c.exit();
}

#[test]
#[serial]
fn test_file_write_strips_code_fence() {
    // A fenced ``` wrapper the model adds is stripped before write.
    let tmp = tempfile::tempdir().unwrap();
    let fenced = "```markdown\n# Title\n\nbody\n```";
    {
        let (mut c, _env) = client(tmp.path());
        let w = c.call_tool("file-write", Some(&args(json!({"path": "doc.md", "content": fenced}))));
        assert!(w.success, "{w:?}");
        c.exit();
    }
    let written = std::fs::read_to_string(tmp.path().join("doc.md")).unwrap();
    assert_eq!(written, "# Title\n\nbody\n");
    assert!(!written.contains("```"));
}

#[test]
#[serial]
fn test_grep_finds_matches() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("a.py"), "alpha = 1\nNEEDLE here\n").unwrap();
    std::fs::write(tmp.path().join("b.txt"), "no match\n").unwrap();
    let (mut c, _env) = client(tmp.path());
    let r = c.call_tool("grep", Some(&args(json!({"pattern": "NEEDLE"}))));
    assert!(r.success, "{r:?}");
    let out = text(&r);
    assert!(out.contains("a.py"), "{out}");
    assert!(out.contains("NEEDLE"), "{out}");
    c.exit();
}

#[test]
#[serial]
fn test_grep_glob_filter() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("a.py"), "TARGET\n").unwrap();
    std::fs::write(tmp.path().join("b.txt"), "TARGET\n").unwrap();
    let (mut c, _env) = client(tmp.path());
    let r = c.call_tool("grep", Some(&args(json!({"pattern": "TARGET", "glob": "*.py"}))));
    assert!(r.success, "{r:?}");
    let out = text(&r);
    assert!(out.contains("a.py"), "{out}");
    assert!(!out.contains("b.txt"), "{out}");
    c.exit();
}

#[test]
#[serial]
fn test_grep_no_matches_is_error() {
    // A miss raises so the orchestrator skips the step and the workspace
    // search fallback fires, instead of 'No matches' suppressing it.
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("a.py"), "nothing\n").unwrap();
    let (mut c, _env) = client(tmp.path());
    let r = c.call_tool("grep", Some(&args(json!({"pattern": "ZZZ_NOPE"}))));
    assert!(r.is_error, "{r:?}");
    assert!(text(&r).to_lowercase().contains("no matches"));
    c.exit();
}

#[test]
#[serial]
fn test_grep_bad_regex_is_error() {
    // A real grep failure (exit code 2) is an error, not a clean miss.
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("a.py"), "content\n").unwrap();
    let (mut c, _env) = client(tmp.path());
    let r = c.call_tool("grep", Some(&args(json!({"pattern": "["}))));
    assert!(r.is_error, "{r:?}");
    assert!(!text(&r).to_lowercase().contains("no matches"));
    c.exit();
}

#[test]
#[serial]
fn test_file_read_missing_path_is_error() {
    let tmp = tempfile::tempdir().unwrap();
    let (mut c, _env) = client(tmp.path());
    let r = c.call_tool("file-read", Some(&args(json!({"path": "does/not/exist.md"}))));
    assert!(r.is_error, "{r:?}");
    assert!(text(&r).contains("not found"));
    c.exit();
}

#[test]
#[serial]
fn test_file_write_empty_path_is_error() {
    let tmp = tempfile::tempdir().unwrap();
    let (mut c, _env) = client(tmp.path());
    let r = c.call_tool("file-write", Some(&args(json!({"path": "", "content": "hello"}))));
    assert!(r.is_error, "{r:?}");
    c.exit();
}

#[test]
#[serial]
fn test_bash_nonzero_exit_is_normal_output() {
    // A command that ran to completion is a real result even when it fails —
    // its exit code and output belong in the answer.
    let tmp = tempfile::tempdir().unwrap();
    let (mut c, _env) = client(tmp.path());
    let r = c.call_tool("bash", Some(&args(json!({"command": "echo boom >&2; exit 3"}))));
    assert!(r.success, "{r:?}");
    assert!(!r.is_error);
    let out = text(&r);
    assert!(out.contains("exit code 3"), "{out}");
    assert!(out.contains("boom"), "{out}");
    c.exit();
}

#[test]
#[serial]
fn test_bash_timeout_is_error() {
    let tmp = tempfile::tempdir().unwrap();
    let (mut c, _env) = client(tmp.path());
    let r = c.call_tool("bash", Some(&args(json!({"command": "sleep 5", "timeout": 1}))));
    assert!(r.is_error, "{r:?}");
    assert!(text(&r).contains("timed out"));
    c.exit();
}
