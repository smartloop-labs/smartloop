//! Port of `tests/unit/test_project_tools_are_claimed.py`.
//!
//! A new project's tools are installed under the claim that lets them persist.
//!
//! `create_project` runs in the supervisor and claims the project for each of
//! its `save()` calls with `own_project`; a delegating supervisor writes no
//! project file without one.  `setup_project_tools` used to sit outside every
//! claim, yet it saves (installing the System Tools server), so under
//! delegation a project came up with no tools registered.  What is asserted
//! here is the claim being *active around the call*.

use std::sync::Arc;

use parking_lot::Mutex;

use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::workspace::Workspace;
use smartloop::workspace::context::{delegate_projects, owned_project_id, pin_project};

use serial_test::serial;

use crate::shared::model_service_b::{global_lock, service_with_workspace};

/// The pin and the delegation flag are process-global; never leak one.
struct ProcessScope {
    _guard: parking_lot::MutexGuard<'static, ()>,
}

impl Drop for ProcessScope {
    fn drop(&mut self) {
        // Runs while a failed test unwinds too; a reset that itself fails
        // must not turn that failure into an abort of the whole binary.
        let _ = std::panic::catch_unwind(|| {
            pin_project(None);
            delegate_projects(false);
            Workspace::set_config_file(None);
        });
    }
}

/// A model service whose workspace is real enough to make a directory in.
///
/// `create_project` joins `workspace_dir` into a path and appends to
/// `projects`; the tool install is recorded rather than performed.
struct Fixture {
    svc: Arc<ModelService>,
    /// The claim in force each time `setup_project_tools` was called, with
    /// the project id it was called for.
    claims: Arc<Mutex<Vec<(String, Option<String>)>>>,
    _tmp: tempfile::TempDir,
    // Dropped last: releases the process-global state after everything else.
    _scope: ProcessScope,
}

fn fixture() -> Fixture {
    let scope = ProcessScope { _guard: global_lock() };
    let tmp = tempfile::tempdir().expect("tempdir");
    let workspace_dir = tmp.path().to_string_lossy().into_owned();
    Workspace::set_config_file(Some(tmp.path().join("config.json").to_string_lossy().into_owned()));
    let svc = service_with_workspace(&workspace_dir, vec![], vec![]);

    let claims: Arc<Mutex<Vec<(String, Option<String>)>>> = Arc::new(Mutex::new(Vec::new()));
    let seen = claims.clone();
    let mut hooks = ModelServiceHooks::default();
    hooks.setup_project_tools = Some(Arc::new(move |project_id: &str| {
        seen.lock().push((project_id.to_string(), owned_project_id()));
        Ok(())
    }));
    svc.set_hooks(hooks);
    Fixture { svc, claims, _tmp: tmp, _scope: scope }
}

fn created_id(svc: &ModelService) -> String {
    let model = svc.model.read().clone().expect("model");
    let workspace = model.workspace.read();
    workspace.projects[0].id.clone()
}

#[test]
#[serial]
fn test_the_install_runs_under_a_claim_on_the_new_project() {
    let f = fixture();
    delegate_projects(true);

    f.svc.create_project("alpha", None, false, None, None).expect("project created");

    let created = created_id(&f.svc);
    let claims: Vec<Option<String>> = f.claims.lock().iter().map(|(_, claim)| claim.clone()).collect();
    assert_eq!(claims, vec![Some(created)]);
}

/// It stays as narrow as the operation, like the saves around it.
#[test]
#[serial]
fn test_the_claim_does_not_outlive_the_install() {
    let f = fixture();
    delegate_projects(true);

    f.svc.create_project("alpha", None, false, None, None).expect("project created");

    assert_eq!(owned_project_id(), None);
}

/// `SLP_SUPERVISOR=0`: the claim is redundant, the install is not.
///
/// Unpinned and undelegated, every project is already writable, so the claim
/// changes nothing — but wrapping the call must not have stopped it.
#[test]
#[serial]
fn test_the_single_server_fallback_still_installs() {
    let f = fixture();

    f.svc.create_project("alpha", None, false, None, None).expect("project created");

    let created = created_id(&f.svc);
    let calls: Vec<String> = f.claims.lock().iter().map(|(id, _)| id.clone()).collect();
    assert_eq!(calls, vec![created]);
}
