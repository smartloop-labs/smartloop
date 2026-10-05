//! Port of `tests/unit/test_project_context.py`.
//!
//! The project a request works on is scoped to that request, not to the
//! process.
//!
//! Before this, `project_id` was honoured by flipping a `current` flag on a
//! workspace-wide list. Two chats on different projects therefore raced: the
//! second moved the flag while the first was still resolving documents, skills
//! and MCP servers through it. These tests pin the behaviour that replaced it.

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use parking_lot::RwLock;
use serial_test::serial;

use smartloop::workspace::context::{
    ProjectBoundExecutor, ProjectContext, active_project_id, bind, pin_project, pinned_project_id,
    use_project,
};
use smartloop::workspace::workspace::{Project, Workspace};

use crate::shared::workspace_support::{LogCapture, ProcessScopeReset};

fn project(pid: &str, current: bool) -> Project {
    Project {
        id: pid.to_string(),
        project_dir: format!("/tmp/{pid}"),
        current,
        name: Some(pid.to_string()),
        ..Default::default()
    }
}

/// The `workspace` fixture.
fn workspace() -> Workspace {
    Workspace::new("ws-1").with_workspace_dir("/tmp/ws-1").with_projects(vec![
        project("alpha", true),
        project("beta", false),
        project("gamma", false),
    ])
}

fn current_id(ws: &Workspace) -> String {
    ws.current_project().expect("a project resolves").id.clone()
}

mod the_binding_is_scoped {
    use super::*;

    #[test]
    #[serial]
    fn test_nothing_is_bound_by_default() {
        let _reset = ProcessScopeReset::new();
        assert_eq!(active_project_id(), None);
    }

    #[test]
    #[serial]
    fn test_a_binding_lasts_only_for_its_block() {
        let _reset = ProcessScopeReset::new();
        {
            let _bound = use_project(Some("beta"));
            assert_eq!(active_project_id().as_deref(), Some("beta"));
        }
        assert_eq!(active_project_id(), None);
    }

    #[test]
    #[serial]
    fn test_bindings_nest() {
        let _reset = ProcessScopeReset::new();
        let _beta = use_project(Some("beta"));
        {
            let _gamma = use_project(Some("gamma"));
            assert_eq!(active_project_id().as_deref(), Some("gamma"));
        }
        assert_eq!(active_project_id().as_deref(), Some("beta"));
    }

    /// A request that named no project must keep falling through.
    #[test]
    #[serial]
    fn test_binding_to_nothing_is_a_no_op() {
        let _reset = ProcessScopeReset::new();
        let _beta = use_project(Some("beta"));
        let _none = use_project(None);
        assert_eq!(active_project_id().as_deref(), Some("beta"));
    }
}

mod which_project_resolves {
    use super::*;

    #[test]
    #[serial]
    fn test_the_bound_project_wins_over_the_selected_one() {
        let _reset = ProcessScopeReset::new();
        let ws = workspace();
        assert_eq!(current_id(&ws), "alpha"); // the UI's selection
        let _beta = use_project(Some("beta"));
        assert_eq!(current_id(&ws), "beta");
    }

    /// The CLI, the supervisor and clients that send no project_id.
    #[test]
    #[serial]
    fn test_an_unbound_reader_still_gets_the_selection() {
        let _reset = ProcessScopeReset::new();
        let ws = workspace();
        assert_eq!(current_id(&ws), "alpha");
    }

    #[test]
    #[serial]
    fn test_a_binding_to_an_unknown_project_falls_back_rather_than_failing() {
        let _reset = ProcessScopeReset::new();
        let ws = workspace();
        let caplog = LogCapture::start();
        {
            let _bound = use_project(Some("not-a-project"));
            assert_eq!(current_id(&ws), "alpha");
        }
        assert!(caplog.text().contains("not in this workspace"));
    }

    /// A per-project agent child fixes its project once, at spawn.
    #[test]
    #[serial]
    fn test_a_pinned_process_resolves_its_own_project() {
        let _reset = ProcessScopeReset::new();
        let ws = workspace();
        pin_project(Some("gamma"));
        assert_eq!(pinned_project_id().as_deref(), Some("gamma"));
        assert_eq!(current_id(&ws), "gamma");
    }

    #[test]
    #[serial]
    fn test_a_request_binding_still_wins_over_the_pin() {
        let _reset = ProcessScopeReset::new();
        let ws = workspace();
        pin_project(Some("gamma"));
        let _beta = use_project(Some("beta"));
        assert_eq!(current_id(&ws), "beta");
    }
}

/// The bug this replaced: request B moved the pointer out from under A.
mod concurrent_requests_do_not_race_each_other {
    use super::*;

    #[test]
    #[serial]
    fn test_two_threads_hold_different_projects_at_once() {
        let _reset = ProcessScopeReset::new();
        let ws = Arc::new(workspace());
        let both_bound = Arc::new(Barrier::new(2));

        let handles: Vec<_> = ["beta", "gamma"]
            .into_iter()
            .map(|pid| {
                let ws = Arc::clone(&ws);
                let both_bound = Arc::clone(&both_bound);
                thread::spawn(move || {
                    let _bound = use_project(Some(pid));
                    // Both bindings are live before either reads, so a shared
                    // pointer would have one thread answering for the other.
                    both_bound.wait();
                    let mut seen = Vec::new();
                    for _ in 0..20 {
                        seen.push(current_id(&ws));
                        thread::sleep(Duration::from_millis(1));
                    }
                    (pid, seen)
                })
            })
            .collect();

        for handle in handles {
            let (pid, seen) = handle.join().unwrap();
            assert_eq!(seen, vec![pid.to_string(); 20]);
        }
    }

    #[test]
    #[serial]
    fn test_a_second_request_does_not_disturb_the_first() {
        let _reset = ProcessScopeReset::new();
        let ws = Arc::new(workspace());
        let _beta = use_project(Some("beta"));

        let other_ws = Arc::clone(&ws);
        let other = thread::spawn(move || {
            let _gamma = use_project(Some("gamma"));
            assert_eq!(current_id(&other_ws), "gamma");
        });
        other.join().unwrap();

        assert_eq!(current_id(&ws), "beta");
    }
}

/// An executor hop does not copy the context; the chat path takes several.
mod carrying_the_binding_into_threads {
    use super::*;

    #[test]
    #[serial]
    fn test_bind_carries_the_project_into_a_worker() {
        let _reset = ProcessScopeReset::new();
        let ws = Arc::new(workspace());
        let read_ws = Arc::clone(&ws);
        let read = move || current_id(&read_ws);

        let result = {
            let _beta = use_project(Some("beta"));
            thread::spawn(bind(read)).join().unwrap()
        };

        assert_eq!(result, "beta");
    }

    /// Shows what bind() is for — the same hop without it.
    #[test]
    #[serial]
    fn test_an_unbound_hop_would_have_read_the_selection() {
        let _reset = ProcessScopeReset::new();
        let ws = Arc::new(workspace());
        let read_ws = Arc::clone(&ws);
        let read = move || current_id(&read_ws);

        let result = {
            let _beta = use_project(Some("beta"));
            thread::spawn(read).join().unwrap()
        };

        assert_eq!(result, "alpha");
    }

    #[test]
    #[serial]
    fn test_bind_leaves_the_worker_thread_clean() {
        let _reset = ProcessScopeReset::new();
        let wrapped = {
            let _beta = use_project(Some("beta"));
            bind(|| ())
        };

        let after = thread::spawn(move || {
            wrapped();
            active_project_id()
        })
        .join()
        .unwrap();
        assert_eq!(after, None);
    }

    /// One pool, two projects in flight — the orchestrator's fan-out.
    #[test]
    #[serial]
    fn test_a_pool_carries_each_submitter_s_own_project() {
        let _reset = ProcessScopeReset::new();
        let ws = Arc::new(workspace());
        let pool = ProjectBoundExecutor::new(4, None);

        let submit = |pool: &ProjectBoundExecutor| {
            (0..8)
                .map(|_| {
                    let ws = Arc::clone(&ws);
                    pool.submit(move || current_id(&ws))
                })
                .collect::<Vec<_>>()
        };
        let beta_futures = {
            let _beta = use_project(Some("beta"));
            submit(&pool)
        };
        let gamma_futures = {
            let _gamma = use_project(Some("gamma"));
            submit(&pool)
        };

        let collect = |futures: Vec<smartloop::workspace::context::PoolFuture<String>>| {
            futures
                .into_iter()
                .map(|f| f.result_timeout(Duration::from_secs(10)).expect("in time").unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(collect(beta_futures), vec!["beta".to_string(); 8]);
        assert_eq!(collect(gamma_futures), vec!["gamma".to_string(); 8]);
        pool.shutdown(true);
    }

    /// Copying rather than entering a Context — the fan-out is parallel.
    #[test]
    #[serial]
    fn test_the_same_binding_can_be_live_in_many_workers_at_once() {
        let _reset = ProcessScopeReset::new();
        let ws = Arc::new(workspace());
        let pool = ProjectBoundExecutor::new(4, None);
        let started = Arc::new(Barrier::new(4));

        let futures: Vec<_> = {
            let _beta = use_project(Some("beta"));
            (0..4)
                .map(|_| {
                    let ws = Arc::clone(&ws);
                    let started = Arc::clone(&started);
                    pool.submit(move || {
                        started.wait(); // all four inside the binding simultaneously
                        current_id(&ws)
                    })
                })
                .collect()
        };
        let results: Vec<String> = futures
            .into_iter()
            .map(|f| f.result_timeout(Duration::from_secs(15)).expect("in time").unwrap())
            .collect();
        assert_eq!(results, vec!["beta".to_string(); 4]);
        pool.shutdown(true);
    }
}

/// The last place that still moved the selection to do its work.
///
/// `add_document` / `add_document_async` used to set `current` on the
/// addressed project and clear it on every other one, purely so that
/// `GenericModel.add_document` — which resolves through
/// `Workspace.current_project` — would write into the right place.  In a
/// per-project child that write cannot persist (`current` belongs to the
/// workspace record), and under the single-server fallback it *did* persist:
/// uploading a document silently moved the project the UI was looking at.
mod adding_a_document_binds_rather_than_flipping {
    use super::*;

    use std::collections::HashMap;

    use parking_lot::Mutex;
    use smartloop::api::services::model_service::ModelService;
    use smartloop::llms::base_llm::BaseLLM;
    use smartloop::model::downloader::slp_downloader::{SharedLogger, SharedWorkspace};
    use smartloop::model::model_manager::Model;
    use smartloop::model::slp_model::{
        DocumentProgressCallback, DocumentSource, SLPModel, TrainProgressCallback,
    };
    use smartloop::types::ModelType;
    use smartloop::utils::log_utils::LoggerLike;
    use smartloop::workspace::types::Document;

    use crate::shared::workspace_support::EnvGuard;

    struct NullLogger;

    impl LoggerLike for NullLogger {
        fn info(&self, _msg: &str) {}
        fn warning(&self, _msg: &str) {}
        fn debug(&self, _msg: &str) {}
        fn error(&self, _msg: &str) {}
    }

    /// Resolves its target exactly how the real one does.
    struct SlpModelStub {
        workspace: SharedWorkspace,
        logger: SharedLogger,
        seen: Arc<Mutex<HashMap<String, Option<String>>>>,
    }

    impl SLPModel for SlpModelStub {
        fn workspace(&self) -> &SharedWorkspace {
            &self.workspace
        }
        fn logger(&self) -> &SharedLogger {
            &self.logger
        }
        fn init(&self) -> anyhow::Result<()> {
            Ok(())
        }
        fn add_document(
            &self,
            _source: DocumentSource,
            _progress_callback: Option<DocumentProgressCallback>,
            _cwd: Option<&str>,
            _generate_filename: bool,
        ) -> anyhow::Result<Vec<Document>> {
            let ws = self.workspace.read();
            let project = ws.current_project();
            self.seen
                .lock()
                .insert("project_id".to_string(), project.map(|p| p.id.clone()));
            let project_dir = project.map(|p| p.project_dir.clone()).unwrap_or_default();
            Ok(vec![Document {
                id: "d1".to_string(),
                path: format!("{project_dir}/d1.md"),
                ..Default::default()
            }])
        }
        fn add_skill(&self, _name: &str, _content: &str) -> anyhow::Result<()> {
            Ok(())
        }
        fn status(&self) -> anyhow::Result<()> {
            Ok(())
        }
        fn build(&self, _progress_callback: Option<TrainProgressCallback>) -> anyhow::Result<()> {
            Ok(())
        }
        fn train(&self, _progress_callback: Option<TrainProgressCallback>) -> anyhow::Result<()> {
            Ok(())
        }
        fn model_type(&self) -> ModelType {
            ModelType::Text
        }
        fn model_id(&self) -> String {
            "test/model".to_string()
        }
        fn get_instance(&self) -> anyhow::Result<Arc<dyn BaseLLM>> {
            anyhow::bail!("no LLM in this test")
        }
    }

    /// The `service` fixture: a bare `ModelService` whose model resolves
    /// documents through the shared workspace.  Persistence is pointed at a
    /// temp home so nothing reaches the real one.
    struct Fixture {
        service: Arc<ModelService>,
        workspace: SharedWorkspace,
        seen: Arc<Mutex<HashMap<String, Option<String>>>>,
        _home: tempfile::TempDir,
        _env: EnvGuard,
        _reset: ProcessScopeReset,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            Workspace::set_config_file(None);
        }
    }

    fn service() -> Fixture {
        let reset = ProcessScopeReset::new();
        let home = tempfile::tempdir().unwrap();
        let env = EnvGuard::set("SLP_HOME", home.path());
        Workspace::set_config_file(Some(home.path().join("config.json").to_string_lossy().into_owned()));

        let mut ws = workspace();
        for p in &mut ws.projects {
            p.project_dir = home.path().join(&p.id).to_string_lossy().into_owned();
        }
        let shared: SharedWorkspace = Arc::new(RwLock::new(ws));
        let seen = Arc::new(Mutex::new(HashMap::new()));
        let slp = SlpModelStub {
            workspace: Arc::clone(&shared),
            logger: Arc::new(NullLogger),
            seen: Arc::clone(&seen),
        };
        let made = ModelService::new_bare();
        *made.model.write() = Some(Arc::new(Model::new("m", Arc::clone(&shared), Some(Arc::new(slp)))));
        Fixture { service: made, workspace: shared, seen, _home: home, _env: env, _reset: reset }
    }

    fn seen_project(fx: &Fixture) -> Option<String> {
        fx.seen.lock().get("project_id").cloned().flatten()
    }

    fn selected(fx: &Fixture) -> Vec<String> {
        fx.workspace.read().projects.iter().filter(|p| p.current).map(|p| p.id.clone()).collect()
    }

    /// Even though a different project is the one marked `current`.
    #[test]
    #[serial]
    fn test_the_addressed_project_receives_the_document() {
        let fx = service();
        assert_eq!(current_id(&fx.workspace.read()), "alpha");

        fx.service.add_document_async("/tmp/f.md", Some("beta"), None, false).unwrap();

        assert_eq!(seen_project(&fx).as_deref(), Some("beta"));
    }

    #[test]
    #[serial]
    fn test_the_selection_does_not_move() {
        let fx = service();
        fx.service.add_document_async("/tmp/f.md", Some("beta"), None, false).unwrap();

        assert_eq!(selected(&fx), vec!["alpha"]);
    }

    #[test]
    #[serial]
    fn test_the_binding_does_not_outlive_the_call() {
        let fx = service();
        fx.service.add_document_async("/tmp/f.md", Some("beta"), None, false).unwrap();

        assert_eq!(active_project_id(), None);
    }

    /// A child is pinned to gamma; an explicit project_id still wins.
    #[test]
    #[serial]
    fn test_it_works_the_same_in_a_pinned_child() {
        let fx = service();
        pin_project(Some("gamma"));

        fx.service.add_document_async("/tmp/f.md", Some("beta"), None, false).unwrap();

        assert_eq!(seen_project(&fx).as_deref(), Some("beta"));
        assert_eq!(selected(&fx), vec!["alpha"]);
    }

    #[test]
    #[serial]
    fn test_an_unnamed_project_falls_back_to_the_selection() {
        let fx = service();
        fx.service.add_document_async("/tmp/f.md", None, None, false).unwrap();

        assert_eq!(seen_project(&fx).as_deref(), Some("alpha"));
    }

    #[test]
    #[serial]
    fn test_the_synchronous_path_binds_too() {
        let fx = service();
        fx.service.add_document("/tmp/f.md", Some("beta"), None).unwrap();

        assert_eq!(seen_project(&fx).as_deref(), Some("beta"));
        assert_eq!(selected(&fx), vec!["alpha"]);
    }
}

mod project_context {
    use super::*;

    #[test]
    #[serial]
    fn test_it_resolves_from_whatever_is_bound() {
        let _reset = ProcessScopeReset::new();
        let ws = Arc::new(RwLock::new(workspace()));
        let ctx = {
            let _beta = use_project(Some("beta"));
            ProjectContext::resolve(Some(Arc::clone(&ws))).expect("bound")
        };
        assert_eq!(ctx.project_id, "beta");
        assert_eq!(ctx.project().unwrap().id, "beta");
        assert_eq!(ctx.project_dir().as_deref(), Some("/tmp/beta"));
    }

    #[test]
    #[serial]
    fn test_unbound_resolves_to_nothing() {
        let _reset = ProcessScopeReset::new();
        let ws = Arc::new(RwLock::new(workspace()));
        assert!(ProjectContext::resolve(Some(ws)).is_none());
    }

    #[test]
    #[serial]
    fn test_it_can_re_establish_itself_as_the_binding() {
        let _reset = ProcessScopeReset::new();
        let ws = Arc::new(RwLock::new(workspace()));
        let ctx = ProjectContext::new(Arc::clone(&ws), "gamma");
        let _bound = ctx.bound();
        assert_eq!(current_id(&ws.read()), "gamma");
    }

    #[test]
    #[serial]
    fn test_lookup_is_late_so_a_reload_is_picked_up() {
        let _reset = ProcessScopeReset::new();
        let ws = Arc::new(RwLock::new(workspace()));
        let ctx = ProjectContext::new(Arc::clone(&ws), "beta");
        // Distinguishable from the original, so the lookup provably reads the
        // replacement rather than a copy taken at construction.
        let replacement = Project { description: Some("replacement".to_string()), ..project("beta", false) };
        ws.write().projects = vec![replacement.clone()];
        assert_eq!(ctx.project(), Some(replacement));
    }
}
