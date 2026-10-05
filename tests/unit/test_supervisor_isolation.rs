//! Port of `tests/unit/test_supervisor_isolation.py`.
//!
//! The payoff: one project's problem stays one project's problem.
//!
//! Separate processes contain a *crash*.  A ceiling contains a leak, which
//! would otherwise still take the machine down with every project on it.  And a
//! log per project makes "which of these lines is the project I am debugging"
//! stop being a question.

use std::sync::Arc;
use std::time::Duration;

use serial_test::serial;
use smartloop::supervisor::pool::{process_rss, PoolOptions};
use smartloop::supervisor::registry::process_alive;
use smartloop::supervisor::service::{model_host_enabled, supervision_enabled, Supervisor};

use crate::shared::supervisor_http_status;
use crate::test_supervisor_pool::{make_pool, PoolGuard, Workspace};

/// 2^22 + 1 is above the default pid_max on Linux and macOS, so nothing holds it.
const GONE_PID: i64 = 4194305;

fn pool_with(ws: &Workspace, memory_limit_bytes: u64) -> PoolGuard {
    make_pool(
        ws,
        PoolOptions {
            idle_ttl: 0.0,
            start_timeout: 30.0,
            command_for: Some(ws.stub_command()),
            memory_limit_bytes,
            ..PoolOptions::default()
        },
    )
}

fn healthy(base_url: &str) -> bool {
    supervisor_http_status(&format!("{base_url}/health"), 5.0) == Some(200)
}

/// Sets environment variables for one test and restores them afterwards.
struct EnvGuard(Vec<(String, Option<String>)>);

impl EnvGuard {
    fn new() -> Self {
        Self(Vec::new())
    }

    fn set(&mut self, name: &str, value: &str) {
        self.0.push((name.to_string(), std::env::var(name).ok()));
        // SAFETY: tests touching process-wide environment run serially.
        unsafe { std::env::set_var(name, value) };
    }

    fn remove(&mut self, name: &str) {
        self.0.push((name.to_string(), std::env::var(name).ok()));
        // SAFETY: tests touching process-wide environment run serially.
        unsafe { std::env::remove_var(name) };
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in self.0.iter().rev() {
            // SAFETY: tests touching process-wide environment run serially.
            unsafe {
                match value {
                    Some(v) => std::env::set_var(name, v),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}

mod test_memory_ceilings {
    use super::*;

    #[test]
    fn test_no_ceiling_by_default() {
        let ws = Workspace::new();
        let pool = pool_with(&ws, 0);
        pool.ensure("alpha").unwrap();
        assert!(pool.sweep().over_limit.is_empty());
    }

    #[test]
    fn test_a_child_over_the_ceiling_is_stopped() {
        // One byte: every real process is over it, which is the point — the
        // policy is under test, not the arithmetic of a particular workload.
        let ws = Workspace::new();
        let pool = pool_with(&ws, 1);
        let record = pool.ensure("alpha").unwrap();

        assert_eq!(pool.sweep().over_limit, vec!["alpha"]);

        for _ in 0..100 {
            if !process_alive(Some(record.pid)) {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(pool.registry.get("alpha").is_none());
    }

    #[test]
    fn test_a_child_under_the_ceiling_is_left_alone() {
        let ws = Workspace::new();
        let pool = pool_with(&ws, 64 * 1024 * 1024 * 1024); // 64 GB
        pool.ensure("alpha").unwrap();
        assert!(pool.sweep().over_limit.is_empty());
        assert!(pool.registry.get("alpha").is_some());
    }

    /// The difference from a single process going OOM.
    #[test]
    fn test_stopping_one_over_the_ceiling_leaves_the_others_serving() {
        let ws = Workspace::new();
        let mut pool = pool_with(&ws, 0);
        let alpha = pool.ensure("alpha").unwrap();
        let beta = pool.ensure("beta").unwrap();

        // Put only alpha over the line, by pretending beta is weightless.
        let alpha_pid = alpha.pid;
        pool.configure(|p| {
            p.process_rss_for = Some(Arc::new(move |pid| Some(if pid == Some(alpha_pid) { 10u64.pow(12) } else { 1 })));
            p.memory_limit_bytes = 1024;
        });
        let over_limit = pool.sweep().over_limit;
        pool.configure(|p| p.process_rss_for = None);
        assert_eq!(over_limit, vec!["alpha"]);

        assert!(process_alive(Some(beta.pid)));
        assert!(healthy(&beta.base_url()));
    }

    #[test]
    fn test_the_project_comes_back_on_its_next_request() {
        let ws = Workspace::new();
        let pool = pool_with(&ws, 1);
        let first = pool.ensure("alpha").unwrap();
        pool.sweep();
        pool.clear_backoff("alpha");

        let second = pool.ensure("alpha").unwrap();

        assert_ne!(second.pid, first.pid);
        assert!(healthy(&second.base_url()));
    }

    #[test]
    fn test_rss_of_a_process_that_is_gone_reads_as_none() {
        assert_eq!(process_rss(Some(GONE_PID)), None);
        assert_eq!(process_rss(None), None);
    }

    #[test]
    fn test_rss_of_this_process_is_a_real_number() {
        assert!(process_rss(Some(i64::from(std::process::id()))).unwrap() > 0);
    }
}

mod test_the_settings {
    use super::*;

    #[test]
    #[serial]
    fn test_the_ceiling_comes_from_settings() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().to_string_lossy().into_owned();
        let mut env = EnvGuard::new();
        env.set("SLP_AGENT_MEMORY_LIMIT_MB", "512");
        env.set("SLP_HOME", &home);

        let supervisor = Supervisor::new(Some(home), None, None);

        assert_eq!(supervisor.memory_limit_bytes, 512 * 1024 * 1024);
    }

    #[test]
    #[serial]
    fn test_the_idle_ttl_comes_from_settings() {
        let tmp = tempfile::tempdir().unwrap();
        let mut env = EnvGuard::new();
        env.set("SLP_AGENT_IDLE_TTL_S", "60");

        let supervisor = Supervisor::new(Some(tmp.path().to_string_lossy().into_owned()), None, None);

        assert_eq!(supervisor.idle_ttl, 60.0);
    }

    #[test]
    #[serial]
    fn test_explicit_values_win_over_settings() {
        let tmp = tempfile::tempdir().unwrap();
        let mut env = EnvGuard::new();
        env.set("SLP_AGENT_MEMORY_LIMIT_MB", "512");

        let supervisor = Supervisor::new(Some(tmp.path().to_string_lossy().into_owned()), Some(5.0), Some(16));

        assert_eq!(supervisor.idle_ttl, 5.0);
        assert_eq!(supervisor.memory_limit_bytes, 16 * 1024 * 1024);
    }

    #[test]
    #[serial]
    fn test_no_ceiling_is_reported_as_none() {
        let tmp = tempfile::tempdir().unwrap();
        let supervisor = Supervisor::new(Some(tmp.path().to_string_lossy().into_owned()), None, Some(0));
        assert!(supervisor.status()["limits"]["memory_limit_bytes"].is_null());
    }
}

/// These assert the environment's effect; a pin overrides it entirely.
mod test_the_switches {
    use super::*;
    use smartloop::workspace::context::pin_project;

    /// Unpinned for the test's duration (the autouse `_unpinned` fixture).
    struct Unpinned;

    impl Unpinned {
        fn new() -> Self {
            pin_project(None);
            Unpinned
        }
    }

    impl Drop for Unpinned {
        fn drop(&mut self) {
            pin_project(None);
        }
    }

    #[test]
    #[serial]
    fn test_supervision_is_on_by_default() {
        let _unpinned = Unpinned::new();
        let mut env = EnvGuard::new();
        env.remove("SLP_SUPERVISOR");
        assert!(supervision_enabled());
    }

    #[test]
    #[serial]
    fn test_supervision_can_be_turned_off() {
        let _unpinned = Unpinned::new();
        for value in ["0", "false", "no", "off", "FALSE"] {
            let mut env = EnvGuard::new();
            env.set("SLP_SUPERVISOR", value);
            assert!(!supervision_enabled(), "{value}");
        }
    }

    /// Only an explicit opt-out counts; anything else keeps the default.
    #[test]
    #[serial]
    fn test_a_meaningless_value_leaves_it_on() {
        let _unpinned = Unpinned::new();
        let mut env = EnvGuard::new();
        env.set("SLP_SUPERVISOR", "maybe");
        assert!(supervision_enabled());
    }

    /// A child inherits the environment; the pin is what stops a grandchild.
    #[test]
    #[serial]
    fn test_a_pinned_process_never_supervises() {
        let _unpinned = Unpinned::new();
        let mut env = EnvGuard::new();
        env.set("SLP_SUPERVISOR", "1");
        pin_project(Some("alpha"));

        assert!(!supervision_enabled());
    }

    #[test]
    #[serial]
    fn test_the_model_host_is_on_by_default_when_supervising() {
        let _unpinned = Unpinned::new();
        let mut env = EnvGuard::new();
        env.remove("SLP_MODEL_HOST_DISABLE");
        assert!(model_host_enabled());
    }

    #[test]
    #[serial]
    fn test_the_model_host_can_never_disable_the_grpc_path() {
        let _unpinned = Unpinned::new();
        let mut env = EnvGuard::new();
        env.set("SLP_MODEL_HOST_DISABLE", "1");
        assert!(model_host_enabled());
    }
}

mod test_status_reporting {
    use super::*;

    #[test]
    #[serial]
    fn test_it_reports_every_agent() {
        let ws = Workspace::new();
        let pool = pool_with(&ws, 0);
        let mut supervisor = Supervisor::new(Some(ws.home.clone()), None, None);
        supervisor.set_pool(Some(Arc::clone(&pool.0)));
        supervisor.registry = Arc::clone(&pool.registry);

        pool.ensure("alpha").unwrap();
        pool.ensure("beta").unwrap();

        let status = supervisor.status();

        let agents = status["agents"].as_array().unwrap();
        let mut ids: Vec<&str> = agents.iter().map(|a| a["project_id"].as_str().unwrap()).collect();
        ids.sort();
        assert_eq!(ids, vec!["alpha", "beta"]);
        for agent in agents {
            assert_eq!(agent["alive"], true);
            assert!(agent["rss_bytes"].as_u64().unwrap() > 0);
            assert!(agent["port"].as_u64().unwrap() > 0);
        }
        supervisor.set_pool(None);
    }

    #[test]
    #[serial]
    fn test_an_empty_workspace_reports_no_agents() {
        let tmp = tempfile::tempdir().unwrap();
        let supervisor = Supervisor::new(Some(tmp.path().to_string_lossy().into_owned()), None, None);
        assert!(supervisor.status()["agents"].as_array().unwrap().is_empty());
    }
}

mod test_per_project_logs {
    use smartloop::api::project_agent::{install_project_logging, AGENT_LOG_FILE};
    use smartloop::utils::logging_setup::shutdown_logging;
    use std::time::Duration;

    /// One shared server.log would interleave every project's trace.
    #[test]
    #[serial_test::serial(project_logging)]
    fn test_a_child_writes_its_own_log() {
        let tmp = tempfile::tempdir().unwrap();
        let project_dir = tmp.path().join("alpha");
        std::fs::create_dir_all(&project_dir).unwrap();

        let listener = install_project_logging(&project_dir.to_string_lossy(), "info");
        tracing::warn!(target: "smartloop.test", "a line for alpha only");
        std::thread::sleep(Duration::from_millis(300));
        shutdown_logging(listener.as_ref());

        let log = project_dir.join(AGENT_LOG_FILE);
        assert!(log.exists());
        assert!(std::fs::read_to_string(&log).unwrap().contains("a line for alpha only"));
    }

    #[test]
    #[serial_test::serial(project_logging)]
    fn test_two_projects_write_to_two_files() {
        let tmp = tempfile::tempdir().unwrap();
        for project_id in ["alpha", "beta"] {
            let project_dir = tmp.path().join(project_id);
            std::fs::create_dir_all(&project_dir).unwrap();
            let listener = install_project_logging(&project_dir.to_string_lossy(), "info");
            tracing::warn!(target: "smartloop.test", "line for {}", project_id);
            std::thread::sleep(Duration::from_millis(300));
            shutdown_logging(listener.as_ref());
        }

        let alpha = std::fs::read_to_string(tmp.path().join("alpha").join(AGENT_LOG_FILE)).unwrap();
        let beta = std::fs::read_to_string(tmp.path().join("beta").join(AGENT_LOG_FILE)).unwrap();
        assert!(alpha.contains("line for alpha") && !alpha.contains("line for beta"));
        assert!(beta.contains("line for beta") && !beta.contains("line for alpha"));
    }

    /// A project must still be served when its log file cannot be written.
    #[test]
    #[serial_test::serial(project_logging)]
    fn test_a_log_that_cannot_be_opened_is_not_fatal() {
        let tmp = tempfile::tempdir().unwrap();
        // A path whose parent is a file, so opening the log fails.
        let blocker = tmp.path().join("not-a-dir");
        std::fs::write(&blocker, "").unwrap();

        assert!(install_project_logging(&blocker.join("nested").to_string_lossy(), "info").is_none());
    }
}
