//! Port of `tests/unit/test_spawn.py`.
//!
//! The fork-free detached spawn used for per-project agents.
//!
//! A `fork()` taken while the supervisor's gRPC server is mid-RPC is exactly
//! the accident this module exists to avoid.  The behaviours below are the
//! ones that must survive the swap to `posix_spawn`: the child lands in its own
//! session, its stdin is `/dev/null`, no unrelated descriptors are inherited,
//! and the handle the pool relies on still polls, waits, reaps and signals.

#[cfg(unix)]
use std::time::Instant;

#[cfg(unix)]
use smartloop::supervisor::spawn::{
    popen_detached_devnull, popen_detached_devnull_with, spawn_detached, DetachedProcess, WaitError,
};

#[cfg_attr(not(unix), allow(dead_code))]
fn argv(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| s.to_string()).collect()
}

#[cfg(unix)]
fn sleep_cmd() -> Vec<String> {
    argv(&["sleep", "30"])
}

#[cfg(unix)]
mod test_detached_process_facade {
    use super::*;

    #[test]
    fn test_wait_reaps_and_poll_returns_the_exit_code() {
        let mut proc = DetachedProcess::new(spawn_detached(&argv(&["sh", "-c", "exit 7"]), None).unwrap());

        assert_eq!(proc.wait(Some(10.0)).unwrap(), Some(7));
        assert_eq!(proc.poll(), Some(7));
        assert_eq!(proc.returncode, Some(7));
    }

    #[test]
    fn test_a_wait_timeout_raises_like_popen() {
        let mut proc = DetachedProcess::new(spawn_detached(&sleep_cmd(), None).unwrap());
        let started = Instant::now();
        let result = proc.wait(Some(0.2));
        let elapsed = started.elapsed().as_secs_f64();
        if proc.poll().is_none() {
            proc.kill().unwrap();
        }
        proc.wait(Some(5.0)).unwrap();

        assert!(matches!(result, Err(WaitError::Timeout(_))));
        assert!(elapsed < 10.0);
    }

    #[test]
    fn test_terminate_signals_the_child() {
        let mut proc = DetachedProcess::new(spawn_detached(&sleep_cmd(), None).unwrap());
        proc.terminate().unwrap();
        // No handler installed: SIGTERM kills it, reported as -15.
        let code = proc.wait(Some(5.0));
        if proc.poll().is_none() {
            let _ = proc.kill();
            let _ = proc.wait(Some(5.0));
        }
        assert_eq!(code.unwrap(), Some(-15));
    }
}

#[cfg(unix)]
mod test_spawned_child {
    use super::*;

    #[test]
    fn test_the_child_gets_its_own_session_and_devnull_stdin() {
        let tmp = tempfile::tempdir().unwrap();
        let probe_path = tmp.path().join("probe.json");
        let command = crate::shared::supervisor_stub_command("probe", &probe_path.to_string_lossy());
        let env: std::collections::HashMap<String, String> = std::env::vars().collect();
        let pid = spawn_detached(&command, Some(&env)).unwrap();
        DetachedProcess::new(pid).wait(Some(30.0)).unwrap();

        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&probe_path).unwrap()).unwrap();
        assert_eq!(data["sid"], pid);
        assert_eq!(data["pgid"], pid);
        assert_eq!(data["stdin_devnull"], true);
        // A fork would have copied this test process's whole table: sockets,
        // pipes, the works.  In a spawned child nothing of the parent's
        // survives — 0, 1, 2 (devnull, stdout, stderr) plus at most the
        // probe's own transient descriptors.
        let fds: Vec<i64> = data["fds"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        for fd in [0, 1, 2] {
            assert!(fds.contains(&fd), "{fds:?}");
        }
        assert!(fds.len() <= 6, "{fds:?}");
    }
}

#[cfg(unix)]
mod test_popen_fallback {
    use super::*;

    #[test]
    fn test_it_spawns_a_real_child() {
        let mut proc = popen_detached_devnull(&argv(&["true"]), None, None).unwrap();
        assert!(proc.pid() > 0);
        assert_eq!(proc.wait(Some(10.0)).unwrap(), Some(0));
    }

    /// The scenario this exists for: spawning with gRPC threads active.
    #[test]
    fn test_it_spawns_dozens_while_a_grpc_server_is_live() {
        let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
        let server = runtime.spawn(async move {
            let _ = tonic::transport::Server::builder()
                .add_routes(tonic::service::Routes::default())
                .serve_with_shutdown("127.0.0.1:0".parse().unwrap(), async {
                    let _ = stop_rx.await;
                })
                .await;
        });
        for _ in 0..10 {
            let mut proc = popen_detached_devnull(&argv(&["sleep", "0.05"]), None, None).unwrap();
            assert_eq!(proc.wait(Some(10.0)).unwrap(), Some(0));
        }
        let _ = stop_tx.send(());
        let _ = runtime.block_on(server);
    }

    #[test]
    fn test_it_falls_back_to_subprocess_when_posix_spawn_is_unusable() {
        let no_posix_spawn = |_: &[String], _: Option<&std::collections::HashMap<String, String>>| {
            Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "no posix_spawn"))
        };

        let mut proc = popen_detached_devnull_with(&no_posix_spawn, &argv(&["true"]), None, None).unwrap();

        assert!(proc.is_popen());
        assert_eq!(proc.wait(Some(10.0)).unwrap(), Some(0));
    }
}

/// Windows creation-flag path.
#[cfg(windows)]
mod test_windows_spawn {
    use super::*;
    use smartloop::supervisor::spawn::popen_detached_devnull;

    #[test]
    fn test_the_child_writes_to_its_output_file_not_ours() {
        let tmp = tempfile::tempdir().unwrap();
        let output = tmp.path().join("agent.out.log");
        let mut proc = popen_detached_devnull(
            &argv(&["cmd", "/C", "echo out& echo err 1>&2"]),
            None,
            Some(&output.to_string_lossy()),
        )
        .unwrap();
        assert_eq!(proc.wait(Some(10.0)).unwrap(), Some(0));
        let text = std::fs::read_to_string(&output).unwrap();
        assert_eq!(text.split_whitespace().collect::<Vec<_>>(), vec!["out", "err"]);
    }

    #[test]
    fn test_with_no_output_file_it_still_spawns() {
        let mut proc = popen_detached_devnull(&argv(&["cmd", "/C", "echo x"]), None, None).unwrap();
        assert_eq!(proc.wait(Some(10.0)).unwrap(), Some(0));
    }
}
