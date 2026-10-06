//! Port of `tests/unit/test_supervisor_reclaim.py`.
//!
//! Agents that are running but in nobody's registry get taken back.
//!
//! The registry is how a supervisor finds the children an earlier one left, so
//! a child missing from it is invisible: nothing routes to it, nothing stops
//! it, and `slp agent status` does not list it.  It still holds the project's
//! port, its task queue and its MCP children.  A project directory describes
//! itself (`agent.pid`, `agent.port`) even when the registry does not, and that
//! is what this reads.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serial_test::serial;
use smartloop::supervisor::registry::{process_alive, AgentRegistry};
use smartloop::supervisor::service::Supervisor;

use crate::shared::{supervisor_http_status, supervisor_reap, supervisor_spawn_stub};

/// A live agent-shaped process beside a project directory (`stray` fixture).
struct Stray {
    tmp: tempfile::TempDir,
    project_dir: String,
    spawned: Vec<std::process::Child>,
}

impl Stray {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let project_dir = tmp.path().join("alpha").to_string_lossy().into_owned();
        Self { tmp, project_dir, spawned: Vec::new() }
    }

    fn home(&self) -> String {
        self.tmp.path().to_string_lossy().into_owned()
    }

    /// Spawn one, and wait for it to publish its port; returns its pid.
    fn spawn(&mut self, healthy: bool) -> i64 {
        let process = supervisor_spawn_stub(if healthy { "agent" } else { "wedged" }, &self.project_dir);
        let pid = i64::from(process.id());
        self.spawned.push(process);
        let port_file = std::path::Path::new(&self.project_dir).join("agent.port");
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if port_file.exists() {
                return pid;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("the stray never published a port");
    }

    fn kill(&mut self, index: usize) {
        let process = &mut self.spawned[index];
        process.kill().unwrap();
        process.wait().unwrap();
    }
}

impl Drop for Stray {
    fn drop(&mut self) {
        for process in &mut self.spawned {
            supervisor_reap(process);
        }
    }
}

/// Points `SLP_HOME` at the test's home for the test's duration.
struct HomeGuard(Option<String>);

impl HomeGuard {
    fn set(home: &str) -> Self {
        let saved = std::env::var("SLP_HOME").ok();
        // SAFETY: tests touching process-wide environment run serially.
        unsafe { std::env::set_var("SLP_HOME", home) };
        Self(saved)
    }
}

impl Drop for HomeGuard {
    fn drop(&mut self) {
        // SAFETY: tests touching process-wide environment run serially.
        unsafe {
            match &self.0 {
                Some(v) => std::env::set_var("SLP_HOME", v),
                None => std::env::remove_var("SLP_HOME"),
            }
        }
    }
}

/// A supervisor over a workspace with one project, and an empty registry.
fn supervisor(stray: &Stray) -> (Supervisor, HomeGuard) {
    let guard = HomeGuard::set(&stray.home());
    let project = ("alpha".to_string(), stray.project_dir.clone());
    let mut made = Supervisor::new(Some(stray.home()), None, None);
    made.set_workspace_projects(Arc::new(move || Some(vec![project.clone()])));
    made.registry = Arc::new(AgentRegistry::new(stray.home()));
    (made, guard)
}

mod test_reclaiming_strays {
    use super::*;

    /// It is a warm process already serving the right project.
    #[test]
    #[serial]
    fn test_a_healthy_stray_is_adopted_rather_than_killed() {
        let mut stray = Stray::new();
        let (supervisor, _home) = supervisor(&stray);
        let pid = stray.spawn(true);

        let reclaimed = supervisor.reclaim_strays();

        assert_eq!(reclaimed, vec!["alpha"]);
        assert_eq!(supervisor.registry.get("alpha").unwrap().pid, pid);
        assert!(process_alive(Some(pid)));
    }

    #[test]
    #[serial]
    fn test_an_adopted_stray_is_routable() {
        let mut stray = Stray::new();
        let (supervisor, _home) = supervisor(&stray);
        stray.spawn(true);

        supervisor.reclaim_strays();

        let record = supervisor.registry.get("alpha").unwrap();
        assert_eq!(supervisor_http_status(&format!("{}/health", record.base_url()), 5.0), Some(200));
    }

    /// Nothing is listening, so there is nothing to adopt.
    #[test]
    #[serial]
    fn test_a_stray_that_does_not_answer_is_stopped() {
        let mut stray = Stray::new();
        let (supervisor, _home) = supervisor(&stray);
        let pid = stray.spawn(false);

        let reclaimed = supervisor.reclaim_strays();

        assert!(reclaimed.is_empty());
        assert!(!process_alive(Some(pid)));
    }

    #[test]
    #[serial]
    fn test_an_agent_the_registry_already_knows_is_left_alone() {
        use smartloop::supervisor::registry::{read_project_agent, AgentRecord};

        let mut stray = Stray::new();
        let (supervisor, _home) = supervisor(&stray);
        let pid = stray.spawn(true);
        let (_, port) = read_project_agent(&stray.project_dir).unwrap();
        supervisor.registry.add(AgentRecord::new("alpha", pid, port), None);

        assert!(supervisor.reclaim_strays().is_empty());
        assert!(process_alive(Some(pid)));
    }

    /// The normal result of a crash — not a stray, just a stale file.
    #[test]
    #[serial]
    fn test_a_pid_file_outliving_its_process_is_ignored() {
        let mut stray = Stray::new();
        let (supervisor, _home) = supervisor(&stray);
        stray.spawn(true);
        stray.kill(0);

        assert!(supervisor.reclaim_strays().is_empty());
        assert!(supervisor.registry.get("alpha").is_none());
    }

    #[test]
    #[serial]
    fn test_a_project_with_no_agent_files_is_ignored() {
        let stray = Stray::new();
        let (supervisor, _home) = supervisor(&stray);
        assert!(supervisor.reclaim_strays().is_empty());
    }

    #[test]
    #[serial]
    fn test_no_workspace_is_not_an_error() {
        let stray = Stray::new();
        let (mut supervisor, _home) = supervisor(&stray);

        supervisor.set_workspace_projects(Arc::new(|| None));

        assert!(supervisor.reclaim_strays().is_empty());
    }
}
