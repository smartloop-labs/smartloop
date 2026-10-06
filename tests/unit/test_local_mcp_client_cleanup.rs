//! Port of `tests/unit/test_local_mcp_client_cleanup.py`.
//!
//! LocalMCPClient must reap its child and bound its buffers.
//!
//! Two problems, both on the MCP registration path:
//!
//! * `enter()` fails with `MCPLaunchError` *after* the spawn when the launcher
//!   exits immediately.  Python does not call `__exit__` when `__enter__`
//!   raises, so the child and its three pipe FDs stayed alive for as long as
//!   the exception's traceback held that frame — which is the whole of
//!   `add_local_mcp_server`'s retry ladder, up to two more spawns on top.
//! * The stderr buffer retained the entire stream even though only the last
//!   few hundred bytes are ever read, so a chatty launcher (uv dependency
//!   resolution, tracebacks) piled up unboundedly.
//!
//! Python fakes `Popen` with a mock whose `poll()` reports an exit; here a real
//! child that exits at once with that stderr plays the dead process, and a
//! long sleep the live one.

#![cfg(unix)]

use smartloop::api::services::model_service::ModelService;
use smartloop::mcp::client::MCPClientError;
use smartloop::mcp::local_client::{LocalMCPClient, STDERR_MAX_LINES};

const MCP2_TAIL: &str = "ModuleNotFoundError: No module named 'mcp.server.fastmcp'";

/// A launcher that dies during startup, leaving `boom` + the mcp-2 tail on stderr.
fn dead_launcher() -> LocalMCPClient {
    let script = format!("printf 'boom\\n%s' \"{MCP2_TAIL}\" >&2; exit 1");
    LocalMCPClient::new("/bin/sh", vec!["-c".into(), script])
}

/// Whether `pid` still names a process (a zombie counts — it was not reaped).
fn alive(pid: u32) -> bool {
    // SAFETY: signal 0 only checks that the pid exists.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

// --------------------------------------------------- a failed enter cleans up

#[test]
fn test_child_is_reaped_and_pipes_closed() {
    let mut client = dead_launcher();
    let err = client.enter().expect_err("the launcher exited during startup");
    assert!(matches!(err, MCPClientError::Launch(_)), "expected MCPLaunchError, got {err:?}");

    // `exit()` ran on the failure path: the child is reaped (no zombie left
    // behind) and its pipes are gone with it.
    assert!(client.process.is_none(), "the child was left attached to the client");
    client.exit(); // idempotent
    assert!(client.process.is_none());
}

#[test]
fn test_child_pid_is_gone_after_a_failed_enter() {
    // The reaping, observed from outside: spawn, fail, and the pid is free.
    let mut client = LocalMCPClient::new(
        "/bin/sh",
        vec!["-c".into(), "echo $$ >&2; exit 1".into()],
    );
    let err = client.enter().expect_err("exits during startup").to_string();
    let pid: u32 = err
        .rsplit("stderr: ")
        .next()
        .and_then(|s| s.trim().parse().ok())
        .expect("the child's pid in the stderr tail");
    assert!(!alive(pid), "child {pid} was not reaped");
}

#[test]
fn test_launch_error_still_carries_the_stderr_tail() {
    // Cleanup must not cost us the diagnostic — the retry ladder classifies the
    // failure from the last traceback line.
    let mut client = dead_launcher();
    let err = client.enter().expect_err("the launcher exited during startup");
    let message = err.to_string();
    assert!(message.contains(MCP2_TAIL), "{message}");
    assert!(ModelService::mcp2_incompat_error(&message));
}

// ------------------------------------------------------ the stderr buffer is bounded

#[test]
fn test_buffer_caps_and_keeps_the_tail() {
    let mut client = LocalMCPClient::new("uvx", vec!["demo-mcp".into()]);
    for i in 0..STDERR_MAX_LINES * 10 {
        client.append_stderr_chunk(format!("noise line {i}\n"));
    }
    client.append_stderr_chunk(MCP2_TAIL);

    assert!(client.stderr_len() <= STDERR_MAX_LINES);
    // The classifiers match the *last* line, so the tail must survive.
    assert!(client.read_stderr_nonblocking(500).ends_with(MCP2_TAIL));
    assert!(ModelService::mcp2_incompat_error(&client.read_stderr_nonblocking(500)));
}

// ------------------------------------------------------------ exit is idempotent

#[test]
fn test_exit_can_run_twice() {
    // enter()'s failure path calls exit(), and the caller may call it again —
    // neither must fail.
    let mut client = LocalMCPClient::new("/bin/sleep", vec!["30".into()]);
    client.enter().expect("a long-running child starts");
    assert!(client.process.is_some());
    let pid = client.process.as_ref().unwrap().id();

    client.exit();
    client.exit();
    assert!(client.process.is_none());
    assert!(!alive(pid), "the child outlived the client");
}
