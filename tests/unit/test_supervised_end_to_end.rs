//! Port of `tests/unit/test_supervised_end_to_end.py`.
//!
//! A supervisor, real project agent processes, and the client contract intact.
//!
//! Everything here runs for real: the supervisor binds a port, spawns the
//! `slp` binary's project agent (`slp -m smartloop.api.project_agent`) as
//! separate OS processes, and requests are answered by the child that owns the
//! project.  What it does not do is load a model — `SLP_API_PRELOAD_MODEL=0`
//! keeps the device out of it, which is the one thing about a live agent a
//! test cannot have.
//!
//! This is the phase's claim under test: same port, same responses, one
//! process per project.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use smartloop::supervisor::parent_watch::{parent_environment, process_started_at};
use smartloop::supervisor::pool::{PoolOptions, ProjectAgentPool};
use smartloop::supervisor::registry::{process_alive, read_project_agent, AgentRegistry};

use crate::shared::{supervisor_http_get, supervisor_http_status};

const SLP: &str = env!("CARGO_BIN_EXE_slp");

/// A workspace with two projects and no models (`home` fixture).
struct Home {
    tmp: tempfile::TempDir,
}

impl Home {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let workspace_id = "ws-e2e";
        let workspace_dir = tmp.path().join(workspace_id);
        let mut projects = Vec::new();
        for project_id in ["alpha", "beta"] {
            let project_dir = workspace_dir.join(project_id);
            std::fs::create_dir_all(&project_dir).unwrap();
            let mut name = project_id.to_string();
            name[..1].make_ascii_uppercase();
            projects.push(json!({
                "id": project_id,
                "name": name,
                "project_dir": project_dir.to_string_lossy(),
                "current": project_id == "alpha",
                "system": false,
            }));
        }
        let config = json!({
            "id": "",
            "workspace": {
                "id": workspace_id,
                "workspace_dir": workspace_dir.to_string_lossy(),
                "models_dir": workspace_dir.join("models").to_string_lossy(),
                "models": [],
                "projects": projects,
            },
        });
        std::fs::write(tmp.path().join("config.json"), config.to_string()).unwrap();
        Self { tmp }
    }

    fn path(&self) -> PathBuf {
        self.tmp.path().to_path_buf()
    }

    fn project_dir(&self, project_id: &str) -> PathBuf {
        self.tmp.path().join("ws-e2e").join(project_id)
    }

    /// Environment for a real project agent: this workspace, no model load.
    fn child_env(&self) -> HashMap<String, String> {
        let mut env: HashMap<String, String> = std::env::vars().collect();
        env.insert("SLP_HOME".into(), self.tmp.path().to_string_lossy().into_owned());
        env.insert("SLP_API_PRELOAD_MODEL".into(), "0".into());
        env.insert("SLP_LOG_LEVEL".into(), "warning".into());
        env.remove("SLP_SUPERVISOR");
        env.remove("SLP_MODEL_HOST");
        env
    }
}

/// The `pool` fixture: real agents, stopped when the test ends.
struct E2ePool(Arc<ProjectAgentPool>);

impl std::ops::Deref for E2ePool {
    type Target = ProjectAgentPool;

    fn deref(&self) -> &ProjectAgentPool {
        &self.0
    }
}

impl Drop for E2ePool {
    fn drop(&mut self) {
        self.0.stop_reaper();
        self.0.stop_all(10.0);
    }
}

fn pool(home: &Home) -> E2ePool {
    let base = home.path().join("ws-e2e");
    let child_env = home.child_env();
    E2ePool(Arc::new(ProjectAgentPool::with_options(
        Arc::new(AgentRegistry::new(home.path().to_string_lossy().into_owned())),
        Arc::new(move |project_id: &str| {
            let directory = base.join(project_id);
            directory.exists().then(|| directory.to_string_lossy().into_owned())
        }),
        PoolOptions {
            idle_ttl: 0.0,
            start_timeout: 120.0,
            command_for: Some(Arc::new(|project_id: &str| {
                [SLP, "-m", "smartloop.api.project_agent", "--project", project_id, "--log-level", "warning"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect()
            })),
            environment_for: Some(Arc::new(move |project_id: &str| {
                let mut env = child_env.clone();
                env.insert("SLP_PROJECT_ID".into(), project_id.to_string());
                // Keep the parent link the real pool adds.  Replacing the
                // whole environment used to drop it, which left these children
                // — real, detached processes — with nothing tying them to the
                // test run that spawned them.
                env.extend(parent_environment());
                // Notice fast: the default 5s/30s is tuned for surviving a
                // supervisor restart, and a test that asserts on orphan cleanup
                // should not wait out a real handover window.
                env.insert("SLP_PARENT_WATCH_INTERVAL_S".into(), "0.5".into());
                env.insert("SLP_PARENT_WATCH_GRACE_S".into(), "1".into());
                env
            })),
            ..PoolOptions::default()
        },
    )))
}

fn kill9(pid: i64) {
    #[cfg(unix)]
    // SAFETY: SIGKILL to an agent this test spawned.
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGKILL);
    }
    #[cfg(not(unix))]
    smartloop::supervisor::pool::terminate_pid(pid, 0.0);
}

mod test_real_project_agents {
    use super::*;

    #[test]
    fn test_an_agent_starts_and_answers() {
        let home = Home::new();
        let pool = pool(&home);
        let record = pool.ensure("alpha").unwrap();

        let (status, body) = supervisor_http_get(&format!("{}/health", record.base_url()), 30.0).unwrap();

        assert_eq!(status, 200);
        assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["status"], "healthy");
    }

    #[test]
    fn test_each_project_gets_its_own_process() {
        let home = Home::new();
        let pool = pool(&home);
        let alpha = pool.ensure("alpha").unwrap();
        let beta = pool.ensure("beta").unwrap();

        assert_ne!(alpha.pid, beta.pid);
        assert!(process_alive(Some(alpha.pid)) && process_alive(Some(beta.pid)));
    }

    /// The pointer is gone: the child *is* the project, fixed at spawn.
    #[test]
    fn test_an_agent_serves_the_project_it_was_spawned_for() {
        let home = Home::new();
        let pool = pool(&home);
        let alpha = pool.ensure("alpha").unwrap();
        let beta = pool.ensure("beta").unwrap();

        let alpha_docs = supervisor_http_status(&format!("{}/v1/projects/alpha/documents", alpha.base_url()), 30.0);
        let beta_docs = supervisor_http_status(&format!("{}/v1/projects/beta/documents", beta.base_url()), 30.0);

        assert_eq!(alpha_docs, Some(200));
        assert_eq!(beta_docs, Some(200));
    }

    /// `alpha` is the selected project; `beta` still gets served.
    ///
    /// A single server had to move the selection to answer for beta, which is
    /// what made two concurrent projects race.  A child is pinned to beta at
    /// spawn and never touches the flag — which still reads `alpha`, because
    /// the selection is the UI's and nothing here changed it.
    #[test]
    fn test_the_unselected_project_gets_an_agent_of_its_own() {
        let home = Home::new();
        let pool = pool(&home);
        let beta = pool.ensure("beta").unwrap();

        assert_eq!(supervisor_http_status(&format!("{}/health", beta.base_url()), 30.0), Some(200));

        let (_, body) = supervisor_http_get(&format!("{}/v1/projects", beta.base_url()), 30.0).unwrap();
        let projects: Value = serde_json::from_str(&body).unwrap();
        let selected = projects["projects"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|p| p["current"].as_bool().unwrap_or(false))
            .cloned();
        assert!(selected.is_some_and(|p| p["id"] == "alpha"));
    }

    /// The blast radius: one project's crash is one project's problem.
    #[test]
    fn test_killing_one_agent_leaves_the_other_serving() {
        let home = Home::new();
        let pool = pool(&home);
        let alpha = pool.ensure("alpha").unwrap();
        let beta = pool.ensure("beta").unwrap();

        kill9(alpha.pid);
        for _ in 0..200 {
            if pool.running("alpha").is_none() {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }

        assert_eq!(supervisor_http_status(&format!("{}/health", beta.base_url()), 30.0), Some(200));
    }

    #[test]
    fn test_a_crashed_project_comes_back_on_the_next_request() {
        let home = Home::new();
        let pool = pool(&home);
        let first = pool.ensure("alpha").unwrap();
        kill9(first.pid);
        for _ in 0..200 {
            if pool.running("alpha").is_none() {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        pool.sweep();
        pool.clear_backoff("alpha");

        let second = pool.ensure("alpha").unwrap();

        assert_ne!(second.pid, first.pid);
        assert_eq!(supervisor_http_status(&format!("{}/health", second.base_url()), 30.0), Some(200));
    }

    #[test]
    fn test_the_agent_publishes_its_pid_and_port() {
        let home = Home::new();
        let pool = pool(&home);
        let record = pool.ensure("alpha").unwrap();

        let published = read_project_agent(&home.project_dir("alpha").to_string_lossy());
        assert_eq!(published, Some((record.pid, record.port)));
    }

    #[test]
    fn test_the_agent_binds_only_loopback() {
        let home = Home::new();
        let pool = pool(&home);
        let record = pool.ensure("alpha").unwrap();
        assert!(record.base_url().starts_with("http://127.0.0.1:"));
    }
}

/// Same port, same responses — the client should not be able to tell.
mod test_the_client_contract_is_unchanged {
    use super::*;
    use axum::extract::Path;
    use axum::routing::get;
    use axum::{Json, Router};
    use smartloop::supervisor::routing::{AgentPool, ProjectRoutingMiddleware};

    /// A supervisor app on a real port, routing to the real children.
    async fn supervisor(pool: &E2ePool) -> (String, tokio::task::JoinHandle<()>) {
        async fn health() -> Json<Value> {
            Json(json!({"status": "healthy", "served_by": "supervisor"}))
        }
        async fn documents(Path(_project_id): Path<String>) -> Json<Value> {
            Json(json!({"served_by": "supervisor"})) // never reached
        }

        let shared: Arc<dyn AgentPool> = Arc::clone(&pool.0) as Arc<dyn AgentPool>;
        let app = Arc::new(ProjectRoutingMiddleware::new(Arc::new(move || Some(Arc::clone(&shared))))).layer(
            Router::new()
                .route("/health", get(health))
                .route("/v1/projects/{project_id}/documents", get(documents)),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://127.0.0.1:{port}"), task)
    }

    fn client() -> reqwest::Client {
        reqwest::Client::builder().no_proxy().timeout(Duration::from_secs(60)).build().unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_a_project_request_is_answered_by_its_child() {
        let home = Home::new();
        let pool = pool(&home);
        let (base, task) = supervisor(&pool).await;

        let response = client().get(format!("{base}/v1/projects/alpha/documents")).send().await.unwrap();

        assert_eq!(response.status().as_u16(), 200);
        let body: Value = response.json().await.unwrap();
        assert_ne!(body.get("served_by"), Some(&json!("supervisor")));
        assert!(pool.registry.get("alpha").is_some());
        task.abort();
        tokio::task::spawn_blocking(move || drop(pool)).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_the_first_request_is_what_starts_the_project() {
        let home = Home::new();
        let pool = pool(&home);
        let (base, task) = supervisor(&pool).await;
        assert!(pool.registry.get("beta").is_none());

        let _ = client().get(format!("{base}/v1/projects/beta/documents")).send().await;

        assert!(pool.registry.get("beta").is_some());
        task.abort();
        tokio::task::spawn_blocking(move || drop(pool)).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_workspace_level_requests_never_start_anything() {
        let home = Home::new();
        let pool = pool(&home);
        let (base, task) = supervisor(&pool).await;

        let _ = client().get(format!("{base}/health")).send().await;

        assert!(pool.registry.all().is_empty());
        task.abort();
        tokio::task::spawn_blocking(move || drop(pool)).await.unwrap();
    }
}

/// A real agent, a real parent, and the parent taken away.
///
/// The regression this exists for was observable in `ps`: agents reparented to
/// init, alive for hours after the run that spawned them was interrupted, each
/// still holding its project's task queue and MCP children.  Children are
/// spawned detached — on purpose, so a Ctrl-C reaches the supervisor alone —
/// which is why the link has to be watched from the child rather than
/// inherited from the OS.
mod test_orphaned_agents {
    use super::*;
    use smartloop::supervisor::parent_watch::{PARENT_PID_ENV, PARENT_STARTED_AT_ENV};

    fn spawn_agent(home: &Home, parent_pid: i64, parent_started_at: Option<f64>) -> std::process::Child {
        let mut env = home.child_env();
        env.insert("SLP_PROJECT_ID".into(), "alpha".into());
        env.insert(PARENT_PID_ENV.into(), parent_pid.to_string());
        env.insert(
            PARENT_STARTED_AT_ENV.into(),
            parent_started_at.map(|s| format!("{s:?}")).unwrap_or_else(|| "None".into()),
        );
        env.insert("SLP_PARENT_WATCH_INTERVAL_S".into(), "0.5".into());
        env.insert("SLP_PARENT_WATCH_GRACE_S".into(), "1".into());
        let mut command = std::process::Command::new(SLP);
        command
            .args(["-m", "smartloop.api.project_agent", "--project", "alpha", "--log-level", "warning"])
            .env_clear()
            .envs(env)
            .stdin(std::process::Stdio::null());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        command.spawn().unwrap()
    }

    fn await_serving(agent: &mut std::process::Child, home: &Home, timeout: f64) -> u16 {
        let project_dir = home.project_dir("alpha").to_string_lossy().into_owned();
        let deadline = Instant::now() + Duration::from_secs_f64(timeout);
        while Instant::now() < deadline {
            assert!(agent.try_wait().unwrap().is_none(), "the agent exited during startup");
            if let Some((pid, port)) = read_project_agent(&project_dir) {
                if pid == i64::from(agent.id())
                    && supervisor_http_status(&format!("http://127.0.0.1:{port}/health"), 5.0) == Some(200)
                {
                    return port;
                }
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!("the agent never became healthy");
    }

    fn wait_exit(child: &mut std::process::Child, timeout: f64) -> Option<std::process::ExitStatus> {
        let deadline = Instant::now() + Duration::from_secs_f64(timeout);
        while Instant::now() < deadline {
            if let Some(status) = child.try_wait().unwrap() {
                return Some(status);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        None
    }

    struct Reaped(Vec<std::process::Child>);

    impl Drop for Reaped {
        fn drop(&mut self) {
            for process in &mut self.0 {
                let _ = process.kill();
                let _ = process.wait();
            }
        }
    }

    #[test]
    fn test_an_agent_exits_when_its_supervisor_is_killed() {
        let home = Home::new();
        let mut processes = Reaped(vec![crate::shared::supervisor_sacrificial()]);
        let parent_pid = i64::from(processes.0[0].id());
        let started_at = process_started_at(Some(parent_pid));
        processes.0.push(spawn_agent(&home, parent_pid, started_at));
        await_serving(&mut processes.0[1], &home, 120.0);

        // SIGKILL, not terminate: a supervisor that shuts down cleanly stops
        // its children itself.  The orphan is what happens when it never gets
        // the chance.
        processes.0[0].kill().unwrap();
        processes.0[0].wait().unwrap();

        assert!(wait_exit(&mut processes.0[1], 90.0).is_some());
        assert!(!process_alive(Some(i64::from(processes.0[1].id()))));
    }

    /// The watch must not be a liveness bug of its own.
    #[test]
    fn test_an_agent_keeps_serving_while_its_supervisor_lives() {
        let home = Home::new();
        let mut processes = Reaped(vec![crate::shared::supervisor_sacrificial()]);
        let parent_pid = i64::from(processes.0[0].id());
        let started_at = process_started_at(Some(parent_pid));
        processes.0.push(spawn_agent(&home, parent_pid, started_at));
        let port = await_serving(&mut processes.0[1], &home, 120.0);

        std::thread::sleep(Duration::from_secs(3)); // several poll intervals and grace periods

        assert!(processes.0[1].try_wait().unwrap().is_none());
        assert_eq!(supervisor_http_status(&format!("http://127.0.0.1:{port}/health"), 10.0), Some(200));
    }
}
