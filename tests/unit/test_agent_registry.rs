//! Port of `tests/unit/test_agent_registry.py`.
//!
//! What is running, for a workspace that is now several processes.
//!
//! `server.pid` and `server.port` described a workspace when a workspace was
//! one process. With a process per project there is a set of them, and the
//! registry is what can answer which project is on which port — for routing,
//! for `slp agent status`, and for a supervisor adopting what an earlier one
//! left.

use smartloop::supervisor::parent_watch::process_started_at;
use smartloop::supervisor::registry::{
    process_alive, read_project_agent, AgentRecord, AgentRegistry, AGENT_PID_FILE, AGENT_PORT_FILE,
};

/// 2^22 + 1 is above the default pid_max on Linux and macOS, so nothing holds it.
const GONE_PID: i64 = 4194305;

fn my_pid() -> i64 {
    i64::from(std::process::id())
}

fn registry(tmp: &tempfile::TempDir) -> AgentRegistry {
    AgentRegistry::new(tmp.path().to_string_lossy().into_owned())
}

fn record(project_id: &str, pid: Option<i64>, port: u16) -> AgentRecord {
    AgentRecord::new(project_id, pid.unwrap_or_else(my_pid), port)
}

fn alpha() -> AgentRecord {
    record("alpha", None, 41000)
}

mod test_liveness {
    use super::*;

    #[test]
    fn test_this_process_is_alive() {
        assert!(process_alive(Some(my_pid())));
    }

    #[test]
    fn test_a_pid_that_is_gone_is_not() {
        assert!(!process_alive(Some(GONE_PID)));
    }

    #[test]
    fn test_a_missing_pid_is_not_alive() {
        for pid in [None, Some(0), Some(-1)] {
            assert!(!process_alive(pid), "{pid:?}");
        }
    }
}

/// `kill(pid, 0)` is POSIX-only as a *question*.
///
/// On Windows the equivalent maps onto TerminateProcess — so the conventional
/// check would kill the agent every time `slp agent status` asked whether it
/// was running.
mod test_liveness_does_not_kill_what_it_asks_about {
    use super::*;

    #[test]
    fn test_it_does_not_signal_the_process() {
        // The question is asked of the process table; it must leave the
        // process it asks about exactly as it was.
        let mut child = shared::supervisor_sacrificial();
        let pid = i64::from(child.id());
        for _ in 0..20 {
            assert!(process_alive(Some(pid)));
        }
        assert!(child.try_wait().unwrap().is_none(), "asking must not end the process");
        assert!(!process_alive(Some(GONE_PID)));
        shared::supervisor_reap(&mut child);
    }

    /// A child nobody waited on still answers signal 0 after it dies.
    #[test]
    fn test_a_zombie_does_not_count_as_alive() {
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let pid = i64::from(child.id());
        for _ in 0..200 {
            if !process_alive(Some(pid)) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let alive = process_alive(Some(pid));
        child.wait().unwrap();
        assert!(!alive);
    }

    use crate::shared;
}

mod test_records {
    use super::*;

    #[test]
    fn test_it_knows_where_to_forward() {
        assert_eq!(record("alpha", None, 41234).base_url(), "http://127.0.0.1:41234");
    }

    #[test]
    fn test_a_fresh_record_is_not_idle() {
        assert!(alpha().idle_seconds() < 1.0);
    }

    #[test]
    fn test_touching_resets_the_idle_clock() {
        let record = alpha();
        record.set_last_used_at(record.last_used_at() - 500.0);
        assert!(record.idle_seconds() > 400.0);
        record.touch();
        assert!(record.idle_seconds() < 1.0);
    }

    #[test]
    fn test_it_round_trips() {
        let original = alpha();
        let restored =
            AgentRecord::from_dict("alpha", &serde_json::Value::Object(original.to_dict())).unwrap();
        assert_eq!(restored.pid, original.pid);
        assert_eq!(restored.port, original.port);
    }

    #[test]
    fn test_a_malformed_record_is_rejected_not_raised() {
        assert!(AgentRecord::from_dict("alpha", &serde_json::json!({"pid": "not-a-pid"})).is_none());
        assert!(AgentRecord::from_dict("alpha", &serde_json::json!({})).is_none());
    }
}

mod test_the_registry {
    use super::*;

    #[test]
    fn test_an_added_agent_can_be_found() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = registry(&tmp);
        registry.add(alpha(), None);
        assert_eq!(registry.get("alpha").unwrap().port, 41000);
    }

    #[test]
    fn test_an_unknown_project_has_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(registry(&tmp).get("nope").is_none());
    }

    #[test]
    fn test_live_excludes_the_dead() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = registry(&tmp);
        registry.add(record("alpha", None, 41000), None);
        registry.add(record("beta", Some(GONE_PID), 41000), None);
        let live: Vec<String> = registry.live().iter().map(|r| r.project_id.clone()).collect();
        assert_eq!(live, vec!["alpha"]);
    }

    #[test]
    fn test_reaping_drops_the_dead_and_keeps_the_rest() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = registry(&tmp);
        registry.add(record("alpha", None, 41000), None);
        registry.add(record("beta", Some(GONE_PID), 41000), None);

        assert_eq!(registry.reap(), vec!["beta"]);
        let all: Vec<String> = registry.all().iter().map(|r| r.project_id.clone()).collect();
        assert_eq!(all, vec!["alpha"]);
    }

    #[test]
    fn test_removing_forgets_it() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = registry(&tmp);
        registry.add(alpha(), None);
        registry.remove("alpha", None);
        assert!(registry.get("alpha").is_none());
    }
}

mod test_persistence {
    use super::*;

    #[test]
    fn test_it_survives_a_restart() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = registry(&tmp);
        registry.set_supervisor(my_pid(), 38540);
        registry.set_model_host(Some(my_pid()), Some("/tmp/model-host.sock"));
        registry.add(alpha(), None);

        let reloaded = AgentRegistry::new(tmp.path().to_string_lossy().into_owned()).load();

        assert_eq!(reloaded.supervisor()["pid"], serde_json::json!(my_pid()));
        assert_eq!(reloaded.supervisor()["port"], serde_json::json!(38540));
        assert_eq!(reloaded.model_host()["address"], serde_json::json!("/tmp/model-host.sock"));
        assert_eq!(reloaded.get("alpha").unwrap().port, 41000);
    }

    /// What a child reads to tell a successor from a recycled pid.
    ///
    /// A restarted supervisor adopts the children of the old one, so a child
    /// whose parent has gone looks here for a replacement.  A bare pid is not
    /// enough to identify one: reuse would hand the child a stranger to live
    /// behind forever.
    #[test]
    fn test_the_supervisor_record_carries_its_start_time() {
        let tmp = tempfile::tempdir().unwrap();
        registry(&tmp).set_supervisor(my_pid(), 38540);

        let reloaded = AgentRegistry::new(tmp.path().to_string_lossy().into_owned()).load();

        let recorded = reloaded.supervisor()["started_at"].as_f64().unwrap();
        let expected = process_started_at(Some(my_pid())).unwrap();
        assert!((recorded - expected).abs() <= 1e-6 * expected.abs().max(1.0));
    }

    /// A pid file outliving its process is the normal result of a crash.
    #[test]
    fn test_a_new_supervisor_does_not_adopt_dead_children() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = registry(&tmp);
        registry.add(record("alpha", None, 41000), None);
        registry.add(record("beta", Some(GONE_PID), 41000), None);

        let reloaded = AgentRegistry::new(tmp.path().to_string_lossy().into_owned()).load();

        assert!(reloaded.get("alpha").is_some());
        assert!(reloaded.get("beta").is_none());
    }

    #[test]
    fn test_no_registry_file_is_not_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let fresh = tmp.path().join("fresh");
        assert!(AgentRegistry::new(fresh.to_string_lossy().into_owned()).load().all().is_empty());
    }

    #[test]
    fn test_a_corrupt_registry_is_not_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("agents.json"), "{not json").unwrap();
        assert!(registry(&tmp).load().all().is_empty());
    }

    #[test]
    fn test_the_file_is_readable_json() {
        let tmp = tempfile::tempdir().unwrap();
        registry(&tmp).add(alpha(), None);
        let saved: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(tmp.path().join("agents.json")).unwrap()).unwrap();
        assert_eq!(saved["version"], 1);
        assert_eq!(saved["agents"]["alpha"]["port"], 41000);
    }
}

/// A project directory should still tell its own story.
mod test_the_projects_own_files {
    use super::*;

    #[test]
    fn test_pid_and_port_land_beside_the_project() {
        let tmp = tempfile::tempdir().unwrap();
        let project_dir = tmp.path().join("ws").join("alpha");
        let dir = project_dir.to_string_lossy().into_owned();
        registry(&tmp).add(alpha(), Some(&dir));

        assert_eq!(std::fs::read_to_string(project_dir.join(AGENT_PID_FILE)).unwrap(), my_pid().to_string());
        assert_eq!(std::fs::read_to_string(project_dir.join(AGENT_PORT_FILE)).unwrap(), "41000");
        assert_eq!(read_project_agent(&dir), Some((my_pid(), 41000)));
    }

    #[test]
    fn test_they_are_cleared_when_the_agent_goes() {
        let tmp = tempfile::tempdir().unwrap();
        let project_dir = tmp.path().join("ws").join("alpha");
        let dir = project_dir.to_string_lossy().into_owned();
        let registry = registry(&tmp);
        registry.add(alpha(), Some(&dir));
        registry.remove("alpha", Some(&dir));

        assert!(!project_dir.join(AGENT_PID_FILE).exists());
        assert_eq!(read_project_agent(&dir), None);
    }

    #[test]
    fn test_a_project_with_no_files_reads_as_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(read_project_agent(&tmp.path().to_string_lossy()), None);
    }

    #[test]
    fn test_garbage_in_the_files_reads_as_none() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(AGENT_PID_FILE), "not-a-pid").unwrap();
        std::fs::write(tmp.path().join(AGENT_PORT_FILE), "nope").unwrap();
        assert_eq!(read_project_agent(&tmp.path().to_string_lossy()), None);
    }
}
