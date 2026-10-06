//! Port of `tests/unit/test_supervisor_pool.py`.
//!
//! Starting, watching and reaping a process per project.
//!
//! Real child processes: the parts worth testing here — publishing a port,
//! going healthy, dying, being restarted with backoff, being stopped when idle
//! — are exactly the parts a mock would assume away.  The children stand in for
//! a project agent without loading a model, which is the only thing about a
//! real one that makes it unusable in a test.

use std::ops::Deref;
use std::sync::Arc;
use std::time::Duration;

use smartloop::supervisor::pool::{
    ChildStartError, CommandFor, PoolOptions, ProjectAgentPool, ProjectDirFor, BACKOFF_MAX, BACKOFF_START,
};
use smartloop::supervisor::registry::{process_alive, AgentRegistry};

use crate::shared::{supervisor_http_status, supervisor_stub_agent};

/// A home with project directories, and the lookup the pool is given.
pub struct Workspace {
    pub tmp: tempfile::TempDir,
    pub home: String,
}

impl Workspace {
    pub fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("ws")).unwrap();
        let home = tmp.path().to_string_lossy().into_owned();
        Self { tmp, home }
    }

    /// `<home>/ws/<project_id>`, except for the project called "unknown".
    pub fn project_dir_for(&self) -> ProjectDirFor {
        let base = self.tmp.path().join("ws");
        Arc::new(move |project_id: &str| {
            (project_id != "unknown").then(|| base.join(project_id).to_string_lossy().into_owned())
        })
    }

    pub fn dir(&self, project_id: &str) -> String {
        (self.project_dir_for())(project_id).unwrap()
    }

    /// The stub agent, told which directory to publish into.
    ///
    /// A project with no directory still gets a command (the pool refuses it
    /// before reading anything); its stub publishes into the scratch home
    /// rather than the test's working directory.
    pub fn stub_command(&self) -> CommandFor {
        let project_dir_for = self.project_dir_for();
        let scratch = self.tmp.path().join("no-directory").to_string_lossy().into_owned();
        Arc::new(move |project_id: &str| {
            supervisor_stub_agent(&project_dir_for(project_id).unwrap_or_else(|| scratch.clone()))
        })
    }

    /// A fixed command for every project.
    pub fn command(parts: &[&str]) -> CommandFor {
        let argv: Vec<String> = parts.iter().map(|s| s.to_string()).collect();
        Arc::new(move |_: &str| argv.clone())
    }

    pub fn registry(&self) -> Arc<AgentRegistry> {
        Arc::new(AgentRegistry::new(self.home.clone()))
    }
}

/// A pool stopped (reaper and children) when the test ends, like the fixture.
pub struct PoolGuard(pub Arc<ProjectAgentPool>);

impl Deref for PoolGuard {
    type Target = ProjectAgentPool;

    fn deref(&self) -> &ProjectAgentPool {
        &self.0
    }
}

impl PoolGuard {
    /// Mutate the pool's settings (Python assigns attributes directly).
    pub fn configure(&mut self, change: impl FnOnce(&mut ProjectAgentPool)) {
        change(Arc::get_mut(&mut self.0).expect("the pool is not shared yet"));
    }
}

impl Drop for PoolGuard {
    fn drop(&mut self) {
        self.0.stop_reaper();
        self.0.stop_all(10.0);
    }
}

pub fn make_pool(ws: &Workspace, options: PoolOptions) -> PoolGuard {
    PoolGuard(Arc::new(ProjectAgentPool::with_options(ws.registry(), ws.project_dir_for(), options)))
}

/// The `pool` fixture: stub children, idle stop off, 30s start timeout.
pub fn pool(ws: &Workspace) -> PoolGuard {
    make_pool(
        ws,
        PoolOptions {
            idle_ttl: 0.0, // off unless a test asks for it
            start_timeout: 30.0,
            command_for: Some(ws.stub_command()),
            ..PoolOptions::default()
        },
    )
}

pub fn kill9(pid: i64) {
    #[cfg(unix)]
    // SAFETY: SIGKILL to a child this test spawned.
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGKILL);
    }
    #[cfg(not(unix))]
    smartloop::supervisor::pool::terminate_pid(pid, 0.0);
}

pub fn wait_dead(pid: i64) {
    for _ in 0..100 {
        if !process_alive(Some(pid)) {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn healthy(base_url: &str) -> bool {
    supervisor_http_status(&format!("{base_url}/health"), 5.0) == Some(200)
}

fn expect_err(result: Result<impl std::fmt::Debug, ChildStartError>, needle: &str) {
    match result {
        Ok(value) => panic!("expected a ChildStartError, got {value:?}"),
        Err(error) => assert!(error.0.contains(needle), "{:?} does not mention {needle:?}", error.0),
    }
}

mod test_starting_on_demand {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_the_first_request_starts_the_project() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let record = pool.ensure("alpha").unwrap();

        assert_eq!(record.project_id, "alpha");
        assert!(process_alive(Some(record.pid)));
        assert!(healthy(&record.base_url()));
    }

    #[test]
    fn test_a_second_request_reuses_the_running_agent() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let first = pool.ensure("alpha").unwrap();
        let second = pool.ensure("alpha").unwrap();

        assert_eq!((second.pid, second.port), (first.pid, first.port));
    }

    #[test]
    fn test_two_projects_get_two_processes() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let alpha = pool.ensure("alpha").unwrap();
        let beta = pool.ensure("beta").unwrap();

        assert_ne!(alpha.pid, beta.pid);
        assert_ne!(alpha.port, beta.port);
    }

    /// Two requests arriving at once on a cold project is the normal case.
    #[test]
    fn test_concurrent_cold_starts_produce_one_child() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let start = std::sync::Barrier::new(4);
        let seen = parking_lot::Mutex::new(Vec::new());
        let errors = parking_lot::Mutex::new(Vec::new());

        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    start.wait();
                    match pool.ensure("alpha") {
                        Ok(record) => seen.lock().push(record.pid),
                        Err(error) => errors.lock().push(error),
                    }
                });
            }
        });

        assert!(errors.lock().is_empty(), "{:?}", errors.lock());
        let mut pids = seen.lock().clone();
        pids.sort();
        pids.dedup();
        assert_eq!(pids.len(), 1);
    }

    #[test]
    fn test_the_child_is_told_which_project_it_is() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let env = pool.environment("alpha");
        assert_eq!(env["SLP_PROJECT_ID"], "alpha");
    }

    /// Unsetting would not do: absent means "on" now, so it must say "0".
    #[test]
    #[serial]
    fn test_a_child_is_told_explicitly_not_to_supervise() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let saved = std::env::var("SLP_SUPERVISOR").ok();
        // SAFETY: serialised with the other tests touching this variable.
        unsafe { std::env::set_var("SLP_SUPERVISOR", "1") };
        let value = pool.environment("alpha")["SLP_SUPERVISOR"].clone();
        restore("SLP_SUPERVISOR", saved);
        assert_eq!(value, "0");
    }

    #[test]
    #[serial]
    fn test_a_child_is_told_not_to_supervise_even_when_nothing_was_set() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let saved = std::env::var("SLP_SUPERVISOR").ok();
        // SAFETY: serialised with the other tests touching this variable.
        unsafe { std::env::remove_var("SLP_SUPERVISOR") };
        let value = pool.environment("alpha")["SLP_SUPERVISOR"].clone();
        restore("SLP_SUPERVISOR", saved);
        assert_eq!(value, "0");
    }

    #[test]
    fn test_the_model_host_is_passed_down() {
        let ws = Workspace::new();
        let pool = make_pool(
            &ws,
            PoolOptions { model_host_address: Some("/tmp/model-host.sock".into()), ..PoolOptions::default() },
        );
        assert_eq!(pool.environment("alpha")["SLP_MODEL_HOST"], "/tmp/model-host.sock");
    }

    /// N copies of the model in one device is what the host exists to avoid.
    #[test]
    fn test_a_child_with_a_host_does_not_load_its_own_weights() {
        let ws = Workspace::new();
        let pool = make_pool(
            &ws,
            PoolOptions { model_host_address: Some("/tmp/model-host.sock".into()), ..PoolOptions::default() },
        );
        assert_eq!(pool.environment("alpha")["SLP_API_PRELOAD_MODEL"], "0");
    }

    /// No host — the --no-supervised style fallback — means local weights.
    #[test]
    #[serial]
    fn test_a_child_without_a_host_loads_its_own() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let saved = std::env::var("SLP_API_PRELOAD_MODEL").ok();
        // SAFETY: serialised with the other tests touching this variable.
        unsafe { std::env::remove_var("SLP_API_PRELOAD_MODEL") };
        let env = pool.environment("alpha");
        restore("SLP_API_PRELOAD_MODEL", saved);
        assert!(!env.contains_key("SLP_MODEL_HOST"));
        assert!(!env.contains_key("SLP_API_PRELOAD_MODEL"));
    }

    #[test]
    fn test_the_default_command_re_execs_this_interpreter() {
        let command = ProjectAgentPool::default_command("alpha");
        let exe = std::env::var("SLP_EXE")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| std::env::current_exe().unwrap().to_string_lossy().into_owned());
        assert_eq!(command[..3], [exe, "-m".to_string(), "smartloop.api.project_agent".to_string()]);
        assert_eq!(command[command.len() - 2..], ["--project".to_string(), "alpha".to_string()]);
    }

    fn restore(name: &str, value: Option<String>) {
        // SAFETY: serialised with the other tests touching this variable.
        unsafe {
            match value {
                Some(v) => std::env::set_var(name, v),
                None => std::env::remove_var(name),
            }
        }
    }
}

mod test_the_registry_tracks_them {
    use super::*;
    use smartloop::supervisor::registry::read_project_agent;

    #[test]
    fn test_a_started_agent_is_recorded() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let record = pool.ensure("alpha").unwrap();
        assert_eq!(pool.registry.get("alpha").unwrap().pid, record.pid);
    }

    #[test]
    fn test_the_project_directory_records_it_too() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let record = pool.ensure("alpha").unwrap();
        let published = read_project_agent(&ws.dir("alpha"));
        assert_eq!(published, Some((record.pid, record.port)));
    }
}

mod test_failing_to_start {
    use super::*;

    #[test]
    fn test_a_project_with_no_directory_cannot_be_started() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        expect_err(pool.ensure("unknown"), "no directory");
    }

    #[test]
    fn test_a_child_that_exits_immediately_is_reported() {
        let ws = Workspace::new();
        let pool = make_pool(
            &ws,
            PoolOptions {
                start_timeout: 10.0,
                command_for: Some(Workspace::command(&["sh", "-c", "exit 3"])),
                ..PoolOptions::default()
            },
        );
        expect_err(pool.ensure("alpha"), "exited with code 3");
    }

    #[test]
    fn test_a_child_that_never_publishes_a_port_times_out() {
        let ws = Workspace::new();
        let pool = make_pool(
            &ws,
            PoolOptions {
                start_timeout: 1.0,
                command_for: Some(Workspace::command(&["sleep", "300"])),
                ..PoolOptions::default()
            },
        );
        expect_err(pool.ensure("alpha"), "did not publish a port");
    }

    #[test]
    fn test_an_unlaunchable_command_is_reported() {
        let ws = Workspace::new();
        let pool = make_pool(
            &ws,
            PoolOptions {
                command_for: Some(Workspace::command(&["/definitely/not/a/binary"])),
                ..PoolOptions::default()
            },
        );
        expect_err(pool.ensure("alpha"), "could not spawn");
    }
}

mod test_stopping {
    use super::*;

    #[test]
    fn test_stopping_ends_the_process() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let record = pool.ensure("alpha").unwrap();

        assert!(pool.stop("alpha", 10.0));

        wait_dead(record.pid);
        assert!(!process_alive(Some(record.pid)));
        assert!(pool.registry.get("alpha").is_none());
    }

    #[test]
    fn test_stopping_something_that_is_not_running_is_not_an_error() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        assert!(!pool.stop("alpha", 10.0));
    }

    #[test]
    fn test_stopping_one_project_leaves_the_others_alone() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let alpha = pool.ensure("alpha").unwrap();
        let beta = pool.ensure("beta").unwrap();

        pool.stop("alpha", 10.0);

        assert!(!process_alive(Some(alpha.pid)) || pool.registry.get("alpha").is_none());
        assert!(process_alive(Some(beta.pid)));
        assert!(healthy(&beta.base_url()));
    }

    #[test]
    fn test_stop_all_ends_every_child() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let records = [pool.ensure("alpha").unwrap(), pool.ensure("beta").unwrap()];

        pool.stop_all(10.0);

        for record in records {
            wait_dead(record.pid);
            assert!(!process_alive(Some(record.pid)));
        }
    }
}

/// The one lifecycle event that has to reach the pool.
///
/// Everything else the pool stops — idle, over the ceiling, crashed — is a
/// project that starts again on its next request.  A deleted one does not, and
/// a child left running would go on serving a directory that is about to be
/// removed underneath it.
mod test_releasing_a_deleted_project {
    use super::*;
    use smartloop::supervisor::service::{release_project, Supervisor};

    fn supervisor(ws: &Workspace, pool: &PoolGuard) -> Supervisor {
        let made = Supervisor::new(Some(ws.home.clone()), None, None);
        made.set_pool(Some(Arc::clone(&pool.0)));
        made
    }

    #[test]
    fn test_releasing_stops_the_agent() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let supervisor = supervisor(&ws, &pool);
        let record = pool.ensure("alpha").unwrap();

        assert!(supervisor.release("alpha"));

        wait_dead(record.pid);
        assert!(!process_alive(Some(record.pid)));
        assert!(pool.registry.get("alpha").is_none());
    }

    #[test]
    fn test_releasing_a_project_with_no_agent_is_not_an_error() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let supervisor = supervisor(&ws, &pool);
        assert!(!supervisor.release("alpha"));
    }

    #[test]
    fn test_releasing_leaves_the_other_projects_serving() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let supervisor = supervisor(&ws, &pool);
        let beta = pool.ensure("beta").unwrap();
        pool.ensure("alpha").unwrap();

        supervisor.release("alpha");

        assert!(process_alive(Some(beta.pid)));
    }

    /// A future project reusing this id must not inherit its penalty.
    #[test]
    fn test_releasing_forgets_the_restart_backoff() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let supervisor = supervisor(&ws, &pool);
        pool.note_exit("alpha", 0.0);
        assert!(pool.backoff("alpha").is_some_and(|delay| delay > 0.0));

        supervisor.release("alpha");

        assert!(pool.backoff("alpha").is_none());
    }

    /// In a child, and under --no-supervised, there is no pool to ask.
    #[test]
    fn test_releasing_without_a_supervisor_is_a_no_op() {
        assert!(!release_project("alpha"));
    }
}

mod test_the_sweep {
    use super::*;

    #[test]
    fn test_an_idle_agent_is_stopped() {
        let ws = Workspace::new();
        let mut pool = pool(&ws);
        pool.configure(|p| p.idle_ttl = 60.0);
        let record = pool.ensure("alpha").unwrap();
        record.set_last_used_at(record.last_used_at() - 120.0);

        assert_eq!(pool.sweep().idled, vec!["alpha"]);
        assert!(pool.registry.get("alpha").is_none());
    }

    #[test]
    fn test_a_busy_agent_is_left_alone() {
        let ws = Workspace::new();
        let mut pool = pool(&ws);
        pool.configure(|p| p.idle_ttl = 60.0);
        pool.ensure("alpha").unwrap();

        assert!(pool.sweep().idled.is_empty());
        assert!(pool.registry.get("alpha").is_some());
    }

    #[test]
    fn test_idle_stop_can_be_switched_off() {
        let ws = Workspace::new();
        let mut pool = pool(&ws);
        pool.configure(|p| p.idle_ttl = 0.0);
        let record = pool.ensure("alpha").unwrap();
        record.set_last_used_at(record.last_used_at() - 100000.0);

        assert!(pool.sweep().idled.is_empty());
    }

    #[test]
    fn test_a_crashed_agent_is_noticed() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let record = pool.ensure("alpha").unwrap();
        kill9(record.pid);
        wait_dead(record.pid);

        assert_eq!(pool.sweep().crashed, vec!["alpha"]);
        assert!(pool.registry.get("alpha").is_none());
    }

    /// The blast radius the split exists to shrink.
    #[test]
    fn test_a_crash_does_not_disturb_another_project() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let alpha = pool.ensure("alpha").unwrap();
        let beta = pool.ensure("beta").unwrap();
        kill9(alpha.pid);
        wait_dead(alpha.pid);

        pool.sweep();

        assert!(process_alive(Some(beta.pid)));
        assert!(healthy(&beta.base_url()));
    }

    #[test]
    fn test_a_crashed_project_starts_again_on_the_next_request() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let first = pool.ensure("alpha").unwrap();
        kill9(first.pid);
        wait_dead(first.pid);
        pool.sweep();
        pool.clear_backoff("alpha"); // skip the wait; the delay is tested below

        let second = pool.ensure("alpha").unwrap();

        assert_ne!(second.pid, first.pid);
        assert!(process_alive(Some(second.pid)));
    }
}

/// A child we spawned and never waited on stays signalable after it dies.
mod test_zombies_are_not_live_agents {
    use super::*;

    #[test]
    fn test_a_killed_child_is_replaced_rather_than_reused() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let first = pool.ensure("alpha").unwrap();
        kill9(first.pid);
        for _ in 0..100 {
            #[cfg(unix)]
            // SAFETY: non-blocking reap of a child of this process.
            unsafe {
                let mut status = 0;
                libc::waitpid(first.pid as libc::pid_t, &mut status, libc::WNOHANG);
            }
            if !process_alive(Some(first.pid)) {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }

        // The pid may still answer signal 0 as a zombie; the pool must decide
        // from the handle it owns, not from the pid, or it would forward this
        // project's requests to a port nothing is listening on.
        let second = pool.ensure("alpha").unwrap();

        assert_ne!(second.pid, first.pid);
        assert!(healthy(&second.base_url()));
    }

    #[test]
    fn test_running_reports_nothing_for_a_dead_child() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        let record = pool.ensure("alpha").unwrap();
        kill9(record.pid);
        for _ in 0..100 {
            if pool.running("alpha").is_none() {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }

        assert!(pool.running("alpha").is_none());
    }
}

mod test_restart_backoff {
    use super::*;

    #[test]
    fn test_the_first_crash_sets_a_delay() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        pool.note_exit("alpha", 1.0);
        assert_eq!(pool.backoff("alpha"), Some(BACKOFF_START));
    }

    #[test]
    fn test_repeated_crashes_back_off_further() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        for _ in 0..4 {
            pool.note_exit("alpha", 1.0);
        }
        assert!(pool.backoff("alpha").unwrap() > BACKOFF_START);
    }

    #[test]
    fn test_the_delay_is_capped() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        for _ in 0..50 {
            pool.note_exit("alpha", 1.0);
        }
        assert_eq!(pool.backoff("alpha"), Some(BACKOFF_MAX));
    }

    /// A crash after two healthy minutes is not a crash loop.
    #[test]
    fn test_a_child_that_ran_a_while_resets_the_backoff() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        for _ in 0..5 {
            pool.note_exit("alpha", 1.0);
        }
        pool.note_exit("alpha", 300.0);
        assert_eq!(pool.backoff("alpha"), Some(BACKOFF_START));
    }

    #[test]
    fn test_one_project_s_backoff_is_its_own() {
        let ws = Workspace::new();
        let pool = pool(&ws);
        pool.note_exit("alpha", 1.0);
        assert!(pool.backoff("beta").is_none());
    }
}

/// A detached child that nobody holds a handle to is an orphan.
///
/// The pool spawns children in their own session so a Ctrl-C in the terminal
/// reaches the supervisor alone.  That is deliberate, and it is also what makes
/// a leak permanent: a process the pool has forgotten cannot be stopped, waited
/// on, or found again through the registry.
mod test_nothing_is_left_behind {
    use super::*;
    use smartloop::supervisor::parent_watch::{process_started_at, PARENT_PID_ENV, PARENT_STARTED_AT_ENV};

    #[test]
    fn test_the_child_is_told_who_its_parent_is() {
        let ws = Workspace::new();
        let pool = pool(&ws);

        let env = pool.environment("alpha");

        let me = i64::from(std::process::id());
        assert_eq!(env[PARENT_PID_ENV], me.to_string());
        let started: f64 = env[PARENT_STARTED_AT_ENV].parse().unwrap();
        let expected = process_started_at(Some(me)).unwrap();
        assert!((started - expected).abs() <= 1e-6 * expected.max(1.0));
    }

    /// It published a port, so it was spawned — and then went silent.
    #[test]
    fn test_a_child_that_never_goes_healthy_is_not_left_running() {
        let ws = Workspace::new();
        let project_dir_for = ws.project_dir_for();
        let silent: CommandFor = Arc::new(move |project_id: &str| {
            crate::shared::supervisor_stub_command("silent", &project_dir_for(project_id).unwrap_or_default())
        });
        let pool = make_pool(
            &ws,
            PoolOptions { start_timeout: 2.0, command_for: Some(silent), ..PoolOptions::default() },
        );

        assert!(pool.ensure("alpha").is_err());

        assert!(pool.process_ids().is_empty());
    }

    /// Ctrl-C through a cold start used to leave the child behind.
    ///
    /// The handle was only recorded once the child was healthy, so anything
    /// raised while waiting for it — including an interrupt, which is not an
    /// ordinary error — dropped the one reference to a detached process.
    #[test]
    fn test_a_start_interrupted_partway_stops_the_child() {
        let ws = Workspace::new();
        let spawned = Arc::new(parking_lot::Mutex::new(Vec::<i64>::new()));
        let seen = Arc::clone(&spawned);
        let pool = make_pool(
            &ws,
            PoolOptions {
                start_timeout: 30.0,
                command_for: Some(ws.stub_command()),
                await_health: Some(Arc::new(move |_project_id, process, _record| {
                    seen.lock().push(process.lock().pid());
                    panic!("KeyboardInterrupt");
                })),
                ..PoolOptions::default()
            },
        );

        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pool.ensure("alpha")));
        assert!(outcome.is_err(), "the interrupt should propagate");

        let spawned = spawned.lock().clone();
        assert!(!spawned.is_empty(), "the child was never spawned");
        assert!(!process_alive(Some(spawned[0])));
        assert!(pool.process_ids().is_empty());
    }

    /// A shutdown that lands mid-start still has to take the child with it.
    #[test]
    fn test_stop_all_reaches_a_child_the_registry_never_learned_about() {
        let ws = Workspace::new();
        let pool = make_pool(
            &ws,
            PoolOptions { command_for: Some(ws.stub_command()), ..PoolOptions::default() },
        );
        let record = pool.ensure("alpha").unwrap();
        // Exactly the state a crash between spawning and recording leaves:
        // a live child, a handle, and nothing in the registry.
        pool.registry.remove("alpha", Some(&ws.dir("alpha")));

        pool.stop_all(10.0);

        assert!(!process_alive(Some(record.pid)));
    }
}
