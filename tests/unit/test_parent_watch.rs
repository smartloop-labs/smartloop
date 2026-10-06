//! Port of `tests/unit/test_parent_watch.py`.
//!
//! A project agent must not outlive the supervisor that spawned it.
//!
//! Children are spawned detached, so the OS does not link them to the parent.
//! Before the parent watch, a supervisor that was killed rather than stopped
//! left its agents running — reparented to init, serving projects nothing
//! routed to, until the machine rebooted.  These tests use real processes for
//! the parent, because the two things that make the watch correct (a pid really
//! being gone, and a *recycled* pid not counting as the parent) are properties
//! of the process table, not of a mock.

use std::sync::Arc;

use serial_test::serial;
use smartloop::supervisor::parent_watch::{
    parent_environment, parent_is_alive, process_started_at, start_parent_watch, OrphanCallback, ParentWatch,
    PARENT_PID_ENV, PARENT_STARTED_AT_ENV,
};
use smartloop::supervisor::registry::AgentRegistry;

use crate::shared::{supervisor_reap, supervisor_sacrificial, SupervisorFlag};

fn my_pid() -> i64 {
    i64::from(std::process::id())
}

/// Sacrificial processes, killed and reaped when the test ends.
struct Sacrificial(Vec<std::process::Child>);

impl Sacrificial {
    fn new() -> Self {
        Self(Vec::new())
    }

    /// Spawn one and return its index.
    fn spawn(&mut self) -> usize {
        self.0.push(supervisor_sacrificial());
        self.0.len() - 1
    }

    fn pid(&self, index: usize) -> i64 {
        i64::from(self.0[index].id())
    }

    fn kill(&mut self, index: usize) {
        let child = &mut self.0[index];
        child.kill().unwrap();
        child.wait().unwrap();
    }
}

impl Drop for Sacrificial {
    fn drop(&mut self) {
        for child in &mut self.0 {
            supervisor_reap(child);
        }
    }
}

fn setter(flag: &Arc<SupervisorFlag>) -> OrphanCallback {
    let flag = Arc::clone(flag);
    Arc::new(move || flag.set())
}

fn watch(pid: i64, home_dir: Option<String>, flag: &Arc<SupervisorFlag>, grace: f64) -> Arc<ParentWatch> {
    ParentWatch::new(pid, process_started_at(Some(pid)), home_dir, Some(setter(flag)), 0.05, grace).start()
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * b.abs().max(1.0)
}

mod test_parent_identity {
    use super::*;

    #[test]
    fn test_the_environment_carries_the_pid_and_the_start_time() {
        let env = parent_environment();

        assert_eq!(env[PARENT_PID_ENV], my_pid().to_string());
        let started: f64 = env[PARENT_STARTED_AT_ENV].parse().unwrap();
        assert!(approx(started, process_started_at(Some(my_pid())).unwrap()));
    }

    #[test]
    fn test_a_live_parent_reads_as_alive() {
        let mut procs = Sacrificial::new();
        let p = procs.spawn();
        let pid = procs.pid(p);

        assert!(parent_is_alive(Some(pid), process_started_at(Some(pid))));
    }

    #[test]
    fn test_a_dead_parent_reads_as_gone() {
        let mut procs = Sacrificial::new();
        let p = procs.spawn();
        let pid = procs.pid(p);
        let started_at = process_started_at(Some(pid));
        procs.kill(p);

        assert!(!parent_is_alive(Some(pid), started_at));
    }

    /// The check that keeps an orphan from adopting a stranger.
    ///
    /// Pids are reused, so on a busy machine the number a dead supervisor held
    /// can belong to something else within minutes.  Matching on the pid alone
    /// would read that as a living parent and the child would stay up for good.
    #[test]
    fn test_a_recycled_pid_is_not_the_parent() {
        let alive_but_unrelated = my_pid();
        let long_ago = process_started_at(Some(alive_but_unrelated)).unwrap() - 3600.0;

        assert!(!parent_is_alive(Some(alive_but_unrelated), Some(long_ago)));
    }

    /// Being unable to tell is not evidence of an orphan.
    #[test]
    fn test_an_unknown_start_time_does_not_condemn_a_live_parent() {
        assert!(parent_is_alive(Some(my_pid()), None));
    }
}

mod test_the_watch {
    use super::*;

    #[test]
    fn test_it_does_nothing_while_the_parent_is_alive() {
        let mut procs = Sacrificial::new();
        let p = procs.spawn();
        let fired = SupervisorFlag::new();
        let watch = watch(procs.pid(p), None, &fired, 0.1);

        assert!(!fired.wait(1.0));
        watch.stop(5.0);
    }

    #[test]
    fn test_it_fires_once_the_parent_is_gone() {
        let mut procs = Sacrificial::new();
        let p = procs.spawn();
        let fired = SupervisorFlag::new();
        let watch = watch(procs.pid(p), None, &fired, 0.2);

        procs.kill(p);

        assert!(fired.wait(10.0));
        watch.stop(5.0);
    }

    /// A supervisor restart is a handover, not an orphaning.
    #[test]
    fn test_it_waits_out_the_grace_period_first() {
        let mut procs = Sacrificial::new();
        let p = procs.spawn();
        let fired = SupervisorFlag::new();
        let watch = watch(procs.pid(p), None, &fired, 1.5);

        procs.kill(p);

        assert!(!fired.wait(0.7));
        assert!(fired.wait(10.0));
        watch.stop(5.0);
    }

    /// An ordinary shutdown must not be reported as an orphaning.
    #[test]
    fn test_stopping_the_watch_stops_the_shutdown() {
        let mut procs = Sacrificial::new();
        let p = procs.spawn();
        let fired = SupervisorFlag::new();
        let watch = watch(procs.pid(p), None, &fired, 0.1);

        watch.stop(5.0);
        procs.kill(p);

        assert!(!fired.wait(1.0));
    }

    #[test]
    fn test_a_failing_callback_does_not_take_the_thread_down_silently() {
        let mut procs = Sacrificial::new();
        let p = procs.spawn();
        let called = SupervisorFlag::new();
        let flag = Arc::clone(&called);
        let boom: OrphanCallback = Arc::new(move || {
            flag.set();
            panic!("shutdown failed");
        });
        let pid = procs.pid(p);
        let watch = ParentWatch::new(pid, process_started_at(Some(pid)), None, Some(boom), 0.05, 0.1).start();
        procs.kill(p);

        assert!(called.wait(10.0));
        watch.stop(5.0);
    }
}

/// A replacement supervisor adopts the children, so the watch re-targets.
mod test_handover {
    use super::*;

    #[test]
    fn test_a_live_successor_in_the_registry_keeps_the_child_up() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().to_string_lossy().into_owned();
        let mut procs = Sacrificial::new();
        let parent = procs.spawn();
        let successor = procs.spawn();
        let registry = AgentRegistry::new(home.clone());
        registry.set_supervisor(procs.pid(successor), 38540);

        let fired = SupervisorFlag::new();
        let watch = watch(procs.pid(parent), Some(home), &fired, 0.2);
        procs.kill(parent);

        assert!(!fired.wait(1.5));
        assert_eq!(watch.pid(), procs.pid(successor), "the watch should re-target");
        watch.stop(5.0);
    }

    #[test]
    fn test_a_dead_successor_is_not_a_successor() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().to_string_lossy().into_owned();
        let mut procs = Sacrificial::new();
        let parent = procs.spawn();
        let successor = procs.spawn();
        AgentRegistry::new(home.clone()).set_supervisor(procs.pid(successor), 38540);
        let parent_pid = procs.pid(parent);
        procs.kill(parent);
        procs.kill(successor);
        let parent_started = process_started_at(Some(parent_pid));

        let fired = SupervisorFlag::new();
        let watch =
            ParentWatch::new(parent_pid, parent_started, Some(home), Some(setter(&fired)), 0.05, 0.2).start();

        assert!(fired.wait(10.0));
        watch.stop(5.0);
    }

    /// The registry records a start time for exactly this case.
    #[test]
    fn test_a_recycled_successor_pid_is_not_a_successor() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().to_string_lossy().into_owned();
        let mut procs = Sacrificial::new();
        let parent = procs.spawn();
        let registry = AgentRegistry::new(home.clone());
        registry.set_supervisor(my_pid(), 38540);
        // Rewrite the recorded start time as if the pid had been reused.
        let path = tmp.path().join("agents.json");
        let mut data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        data["supervisor"]["started_at"] =
            serde_json::json!(process_started_at(Some(my_pid())).unwrap() - 3600.0);
        std::fs::write(&path, data.to_string()).unwrap();

        let fired = SupervisorFlag::new();
        let watch = watch(procs.pid(parent), Some(home), &fired, 0.2);
        procs.kill(parent);

        assert!(fired.wait(10.0));
        watch.stop(5.0);
    }

    #[test]
    fn test_an_unreadable_registry_is_not_a_successor() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().to_string_lossy().into_owned();
        let mut procs = Sacrificial::new();
        let parent = procs.spawn();
        std::fs::write(tmp.path().join("agents.json"), "{ not json").unwrap();

        let fired = SupervisorFlag::new();
        let watch = watch(procs.pid(parent), Some(home), &fired, 0.2);
        procs.kill(parent);

        assert!(fired.wait(10.0));
        watch.stop(5.0);
    }
}

/// Saves and restores the parent-watch environment variables.
struct EnvGuard(Vec<(&'static str, Option<String>)>);

impl EnvGuard {
    fn new() -> Self {
        let names = [
            PARENT_PID_ENV,
            PARENT_STARTED_AT_ENV,
            "SLP_PARENT_WATCH_INTERVAL_S",
            "SLP_PARENT_WATCH_GRACE_S",
        ];
        Self(names.iter().map(|n| (*n, std::env::var(n).ok())).collect())
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in &self.0 {
            // SAFETY: the tests touching these variables run serially.
            unsafe {
                match value {
                    Some(v) => std::env::set_var(name, v),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}

fn set_env(name: &str, value: &str) {
    // SAFETY: the tests touching these variables run serially.
    unsafe { std::env::set_var(name, value) }
}

fn noop() -> OrphanCallback {
    Arc::new(|| {})
}

mod test_opting_out {
    use super::*;

    /// The shared server, and a hand-run agent, have nothing to watch.
    #[test]
    #[serial(parent_watch_env)]
    fn test_a_process_with_no_parent_is_not_watched() {
        let _env = EnvGuard::new();
        // SAFETY: serialised with every other test touching this variable.
        unsafe { std::env::remove_var(PARENT_PID_ENV) };

        assert!(start_parent_watch(noop(), None).is_none());
    }

    #[test]
    #[serial(parent_watch_env)]
    fn test_a_nonsense_parent_pid_is_not_watched() {
        let _env = EnvGuard::new();
        set_env(PARENT_PID_ENV, "not-a-pid");

        assert!(start_parent_watch(noop(), None).is_none());
    }

    #[test]
    #[serial(parent_watch_env)]
    fn test_a_process_is_not_its_own_parent() {
        let _env = EnvGuard::new();
        set_env(PARENT_PID_ENV, &my_pid().to_string());

        assert!(start_parent_watch(noop(), None).is_none());
    }

    #[test]
    #[serial(parent_watch_env)]
    fn test_the_interval_and_grace_are_configurable() {
        let _env = EnvGuard::new();
        let mut procs = Sacrificial::new();
        let p = procs.spawn();
        let pid = procs.pid(p);
        set_env(PARENT_PID_ENV, &pid.to_string());
        set_env(PARENT_STARTED_AT_ENV, &format!("{:?}", process_started_at(Some(pid)).unwrap()));
        set_env("SLP_PARENT_WATCH_INTERVAL_S", "0.25");
        set_env("SLP_PARENT_WATCH_GRACE_S", "2.5");

        let watch = start_parent_watch(noop(), None).unwrap();

        assert_eq!((watch.poll_interval, watch.grace), (0.25, 2.5));
        watch.stop(5.0);
    }
}
