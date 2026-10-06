//! Port of `tests/test_model.py`.

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod shared;

use serial_test::serial;
use smartloop::config::model_config::ModelConfig;
use smartloop::model::model_manager::ModelManager;
use smartloop::model_host::client::{ModelHostError, RemoteModelClient};
use smartloop::workspace::workspace::{Project, Workspace};

use crate::shared::model_fakes::TestLogger;

/// Sets `SLP_HOME` for the test and restores the previous value on drop,
/// resetting the workspace singleton on both sides (the `try/finally`).
struct HomeGuard {
    previous: Option<String>,
}

impl HomeGuard {
    fn set(home: &str) -> Self {
        let previous = std::env::var("SLP_HOME").ok();
        // SAFETY: every test that touches the environment is `#[serial]`.
        unsafe { std::env::set_var("SLP_HOME", home) };
        Workspace::reset();
        HomeGuard { previous }
    }
}

impl Drop for HomeGuard {
    fn drop(&mut self) {
        Workspace::reset();
        // SAFETY: every test that touches the environment is `#[serial]`.
        unsafe {
            match &self.previous {
                Some(value) => std::env::set_var("SLP_HOME", value),
                None => std::env::remove_var("SLP_HOME"),
            }
        }
    }
}

#[test]
#[serial]
fn test_stale_project_id_is_cleared() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("tmp_home");
    std::fs::create_dir_all(&home).unwrap();
    let _home = HomeGuard::set(&home.to_string_lossy());

    let ws_dir = home.join("workspace-1");
    std::fs::create_dir_all(&ws_dir).unwrap();
    let mut workspace = Workspace::new("workspace-1");
    workspace.workspace_dir = ws_dir.to_string_lossy().into_owned();
    workspace.models_dir = ws_dir.join("models").to_string_lossy().into_owned();
    workspace.projects = Vec::new();

    let project = Project {
        id: "real-project".to_string(),
        project_dir: ws_dir.join("real-project").to_string_lossy().into_owned(),
        current: true,
        name: Some("real".to_string()),
        ..Default::default()
    };
    std::fs::create_dir_all(&project.project_dir).unwrap();
    workspace.projects = vec![project];
    workspace.save().unwrap();

    let mut config = ModelConfig { project_id: Some("ghost-project".to_string()), ..ModelConfig::new("sl-mini") };
    workspace.add_model("sl-mini", "smartloop-ai/slp-orchestra-mini", Some(config.clone()), true);
    workspace.save().unwrap();

    let cleaned = ModelManager::purge_stale_project_id(&mut workspace, &mut config, &TestLogger);

    assert_eq!(cleaned.project_id, None);
    assert_eq!(config.project_id, None);
    assert_eq!(workspace.get_model(Some("sl-mini")).unwrap().config.unwrap().project_id, None);
}

#[test]
fn test_unbound_model_host_port_is_rejected() {
    let client = RemoteModelClient::new("127.0.0.1:0");
    // Every RPC opens the channel first (`_connect`), and an unbound
    // placeholder address must be refused there rather than dialled.
    let result: Result<_, ModelHostError> = client.encode_texts_with_timeout(&["x".to_string()], None, None);
    assert!(result.is_err());
}
