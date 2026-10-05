//! Port of `tests/unit/test_project_services_are_project_work.py`.
//!
//! Installing a project's MCP servers and skills belongs to that project's
//! agent.
//!
//! `setup_project_services` is step 6 of `/v1/init` (step 8 of
//! `/v1/bootstrap`).  It runs `setup_project_tools` — which registers the
//! system MCP server and writes the default skills — then marks the project
//! `system=true` to record that the work is done, and saves.
//!
//! That save claims no project.  `/v1/init` names no project in its path, so
//! under supervision it is answered by the supervisor, where `save()` writes
//! only the workspace record.  Both halves of the result were therefore
//! discarded: the registered servers, and the flag that would have stopped it
//! happening again.  The flag going with them is what made it re-run on every
//! init, spawning the same MCP stdio children over and over in a process that
//! does not own the project they belong to — and resolving *which* project
//! through the `current` flag, so it could pick one a child was serving.
//!
//! Nothing is lost by skipping it there.  `create_project` installs the tools
//! once under its own claim, and the project's agent runs
//! `setup_project_tools` for the project it is pinned to when it loads a model
//! — where the save actually persists.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use futures::StreamExt;
use parking_lot::Mutex;
use serial_test::serial;
use smartloop::api::routers::init::{setup_project_services, ProjectServicesHost, SseStream};
use smartloop::workspace::context::{delegate_projects, pin_project};
use smartloop::workspace::workspace::Project;

/// The `_clear_process_scope` autouse fixture: the pin and the delegation flag
/// are process-global; never leak one.
struct ClearProcessScope;

impl Drop for ClearProcessScope {
    fn drop(&mut self) {
        pin_project(None);
        delegate_projects(false);
    }
}

/// A model service whose current project still needs its services (the
/// `MagicMock` of the Python test).
///
/// `system` is the "already set up" marker, so `false` is the state that makes
/// the body run at all.
struct Svc {
    current_project: Mutex<Option<Project>>,
    /// Make reading `current_project` fail the test (the `PropertyMock` with an
    /// `AssertionError` side effect).
    current_project_raises: bool,
    setup_project_tools_calls: AtomicUsize,
    save_calls: AtomicUsize,
}

impl Svc {
    fn new(system: bool) -> Self {
        let project = Project {
            id: "alpha".to_string(),
            name: Some("alpha".to_string()),
            system,
            ..Default::default()
        };
        Self {
            current_project: Mutex::new(Some(project)),
            current_project_raises: false,
            setup_project_tools_calls: AtomicUsize::new(0),
            save_calls: AtomicUsize::new(0),
        }
    }

    fn current_system(&self) -> Option<bool> {
        self.current_project.lock().as_ref().map(|p| p.system)
    }
}

impl ProjectServicesHost for Svc {
    fn current_project(&self) -> Option<Project> {
        if self.current_project_raises {
            panic!("resolved the current project under delegation");
        }
        self.current_project.lock().clone()
    }

    fn setup_project_tools(&self, _project: &Project) -> anyhow::Result<()> {
        self.setup_project_tools_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn mark_current_project_system_and_save(&self) -> anyhow::Result<()> {
        if let Some(project) = self.current_project.lock().as_mut() {
            project.system = true;
        }
        self.save_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

async fn drain(stream: SseStream) -> Vec<String> {
    stream.map(|event| event.expect("the setup step raised")).collect().await
}

/// `TestOnlyAProjectsOwnProcessSetsItUp`
mod test_only_a_projects_own_process_sets_it_up {
    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_a_delegating_supervisor_installs_nothing() {
        let _scope = ClearProcessScope;
        let svc = Arc::new(Svc::new(false));
        delegate_projects(true);

        let events = drain(setup_project_services(Arc::clone(&svc))).await;

        assert!(events.is_empty());
        assert_eq!(svc.setup_project_tools_calls.load(Ordering::SeqCst), 0);
    }

    /// The save is the half that would have been discarded anyway.
    #[tokio::test]
    #[serial]
    async fn test_a_delegating_supervisor_writes_nothing() {
        let _scope = ClearProcessScope;
        let svc = Arc::new(Svc::new(false));
        delegate_projects(true);

        drain(setup_project_services(Arc::clone(&svc))).await;

        assert_eq!(svc.save_calls.load(Ordering::SeqCst), 0);
        assert_eq!(svc.current_system(), Some(false));
    }

    /// `SLP_SUPERVISOR=0`: unpinned and undelegated, as it always was.
    #[tokio::test]
    #[serial]
    async fn test_the_single_server_fallback_still_sets_up() {
        let _scope = ClearProcessScope;
        let svc = Arc::new(Svc::new(false));

        let events = drain(setup_project_services(Arc::clone(&svc))).await;

        assert_eq!(svc.setup_project_tools_calls.load(Ordering::SeqCst), 1);
        assert_eq!(svc.save_calls.load(Ordering::SeqCst), 1);
        assert_eq!(svc.current_system(), Some(true));
        assert_eq!(events.len(), 2);
    }
}

/// `TestTheGuardIsNarrow`
mod test_the_guard_is_narrow {
    use super::*;

    /// The pre-existing early return is unchanged by the new one.
    #[tokio::test]
    #[serial]
    async fn test_an_already_set_up_project_is_still_skipped() {
        let _scope = ClearProcessScope;
        let svc = Arc::new(Svc::new(true));

        let events = drain(setup_project_services(Arc::clone(&svc))).await;

        assert!(events.is_empty());
        assert_eq!(svc.setup_project_tools_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    #[serial]
    async fn test_a_project_less_workspace_is_still_skipped() {
        let _scope = ClearProcessScope;
        let svc = Arc::new(Svc::new(false));
        *svc.current_project.lock() = None;

        let events = drain(setup_project_services(Arc::clone(&svc))).await;

        assert!(events.is_empty());
        assert_eq!(svc.setup_project_tools_calls.load(Ordering::SeqCst), 0);
    }

    /// A supervisor must not resolve `current_project` to decide this.
    ///
    /// That attribute is the `current` flag — the thing the per-project split
    /// exists to stop consulting — so reading it here would pick out some
    /// project a child is serving before concluding it has no business
    /// touching one.  The read is made to panic, so the order is asserted
    /// rather than assumed.
    #[tokio::test]
    #[serial]
    async fn test_delegation_is_checked_before_the_workspace_is_read() {
        let _scope = ClearProcessScope;
        let mut svc = Svc::new(false);
        svc.current_project_raises = true;
        let svc = Arc::new(svc);
        delegate_projects(true);

        assert!(drain(setup_project_services(svc)).await.is_empty());
    }
}
