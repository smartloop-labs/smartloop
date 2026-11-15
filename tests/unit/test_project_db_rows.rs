//! Each project in config.json has a row in the workspace DB's `projects`
//! table: written on create, deleted on delete.

use std::sync::Arc;

use serial_test::serial;

use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::platform::workspace_db_for;
use smartloop::workspace::Workspace;

use crate::shared::model_service_b::{global_lock, service_with_workspace};

struct Fixture {
    svc: Arc<ModelService>,
    _tmp: tempfile::TempDir,
    // Dropped last: releases the process-global state after everything else.
    _guard: parking_lot::MutexGuard<'static, ()>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::panic::catch_unwind(|| Workspace::set_config_file(None));
    }
}

/// A model service over a workspace in a temp dir, with the tool install
/// stubbed out.
fn fixture() -> Fixture {
    let guard = global_lock();
    let tmp = tempfile::tempdir().expect("tempdir");
    Workspace::set_config_file(Some(tmp.path().join("config.json").to_string_lossy().into_owned()));
    let svc = service_with_workspace(&tmp.path().to_string_lossy(), vec![], vec![]);
    let mut hooks = ModelServiceHooks::default();
    hooks.setup_project_tools = Some(Arc::new(|_: &str| Ok(())));
    svc.set_hooks(hooks);
    Fixture { svc, _tmp: tmp, _guard: guard }
}

/// Project ids recorded in the fixture workspace's DB.
fn recorded(svc: &ModelService) -> Vec<String> {
    let model = svc.model.read().clone().expect("model");
    let workspace = model.workspace.read();
    let mut ids = workspace_db_for(&workspace).unwrap().projects().ids(&workspace.id).unwrap();
    ids.sort();
    ids
}

fn created_id(created: &serde_json::Value) -> String {
    created["id"].as_str().expect("project id").to_string()
}

#[test]
#[serial]
fn test_a_created_project_is_recorded() {
    let f = fixture();
    let created = f.svc.create_project("alpha", None, false, None, None, None).unwrap();
    assert_eq!(recorded(&f.svc), [created_id(&created)]);
}

#[test]
#[serial]
fn test_the_system_project_is_recorded() {
    let f = fixture();
    let created = f.svc.create_project("general_chat", None, true, None, None, None).unwrap();
    assert_eq!(recorded(&f.svc), [created_id(&created)]);
}

#[test]
#[serial]
fn test_a_deleted_project_loses_its_row() {
    let f = fixture();
    let kept = f.svc.create_project("alpha", None, false, None, None, None).unwrap();
    let deleted = f.svc.create_project("beta", None, false, None, None, None).unwrap();
    f.svc.delete_project(&created_id(&deleted)).unwrap();
    assert_eq!(recorded(&f.svc), [created_id(&kept)]);
}

#[test]
#[serial]
fn test_the_system_project_keeps_its_row() {
    let f = fixture();
    let system = f.svc.create_project("general_chat", None, true, None, None, None).unwrap();
    assert!(f.svc.delete_project(&created_id(&system)).is_err());
    assert_eq!(recorded(&f.svc), [created_id(&system)]);
}
