//! Port of `tests/unit/test_bootstrap_is_workspace_work.py`.
//!
//! Downloading the shared orchestrator weights belongs to the workspace owner.
//!
//! `bootstrap_orchestrator_model` pulls the orchestrator GGUF and records it as
//! downloaded in `config.json`.  Both halves are workspace-level: the weights
//! sit in one shared `models_dir`, and a per-project agent never writes the
//! workspace record, so a child running this would fetch its own copy of the
//! weights and then lose the flag at save, doing it all again on the next
//! start.
//!
//! Children usually never reach it, because the pool sets
//! `SLP_API_PRELOAD_MODEL=0` — but only when there is a model host to reach,
//! so with the host disabled or failed they inherit the supervisor's preload
//! setting and arrive here after all.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use serde_json::{json, Value};
use serial_test::serial;

use smartloop::api::main::{bootstrap_orchestrator_model_with, BootstrapError, OrchestratorDownloader};
use smartloop::workspace::context::pin_project;
use smartloop::workspace::workspace::{SharedWorkspace, Workspace};

use crate::shared::api_core::ProjectPinReset;

/// The patched `SLPArtifactDownloader`: its bootstrap source lists
/// `[{"name": "sl-mini"}]` and `bootstrap_download` records its calls.
#[derive(Default)]
struct Downloader {
    bootstrap_download_calls: Mutex<Vec<String>>,
    constructed: Mutex<usize>,
}

struct DownloaderHandle(Arc<Downloader>);

impl OrchestratorDownloader for DownloaderHandle {
    fn list_bootstrap_models(&self) -> anyhow::Result<Vec<Value>> {
        Ok(vec![json!({"name": "sl-mini"})])
    }

    fn bootstrap_download(&self, model_name: &str) -> anyhow::Result<PathBuf> {
        self.0.bootstrap_download_calls.lock().push(model_name.to_string());
        Ok(PathBuf::new())
    }
}

/// The `MagicMock` model service: counts `_ensure_workspace` calls and hands
/// back `workspace`.
struct Service {
    ensure_workspace_calls: Mutex<usize>,
    workspace: SharedWorkspace,
}

impl Service {
    fn new(workspace: SharedWorkspace) -> Self {
        Service { ensure_workspace_calls: Mutex::new(0), workspace }
    }
}

/// Run the bootstrap with the stand-ins.
fn bootstrap(service: &Service, downloader: &Arc<Downloader>) -> Result<(), BootstrapError> {
    bootstrap_orchestrator_model_with(
        || {
            *service.ensure_workspace_calls.lock() += 1;
            Ok(Arc::clone(&service.workspace))
        },
        |_workspace| {
            *downloader.constructed.lock() += 1;
            Box::new(DownloaderHandle(Arc::clone(downloader))) as Box<dyn OrchestratorDownloader>
        },
    )
}

/// A real workspace whose `config.json` lives in a temp directory, so the save
/// the owner does is observable and touches nothing real.
struct TempWorkspace {
    _dir: tempfile::TempDir,
    config_file: PathBuf,
    workspace: SharedWorkspace,
}

impl TempWorkspace {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let config_file = dir.path().join("config.json");
        Workspace::set_config_file(Some(config_file.to_string_lossy().into_owned()));
        let mut workspace = Workspace::new("ws-test");
        workspace.workspace_dir = dir.path().join("ws-test").to_string_lossy().into_owned();
        workspace.models_dir = dir.path().join("ws-test").join("models").to_string_lossy().into_owned();
        TempWorkspace { _dir: dir, config_file, workspace: Arc::new(RwLock::new(workspace)) }
    }

    fn saved(&self) -> bool {
        Path::new(&self.config_file).exists()
    }
}

impl Drop for TempWorkspace {
    fn drop(&mut self) {
        Workspace::set_config_file(None);
    }
}

mod test_only_the_workspace_owner_bootstraps {
    use super::*;

    #[test]
    #[serial]
    fn test_a_pinned_child_does_not_download() {
        let _pin = ProjectPinReset::new();
        let downloader = Arc::new(Downloader::default());
        pin_project(Some("alpha"));
        let service = Service::new(Arc::new(RwLock::new(Workspace::new("ws"))));

        let result = bootstrap(&service, &downloader);

        match result {
            Err(BootstrapError::FileNotFound(message)) => assert!(message.contains("does not own"), "{message}"),
            other => panic!("expected FileNotFound, got {other:?}"),
        }
        assert!(downloader.bootstrap_download_calls.lock().is_empty());
    }

    #[test]
    #[serial]
    fn test_a_pinned_child_writes_nothing_to_the_workspace() {
        let _pin = ProjectPinReset::new();
        let downloader = Arc::new(Downloader::default());
        pin_project(Some("alpha"));
        let service = Service::new(Arc::new(RwLock::new(Workspace::new("ws"))));

        let result = bootstrap(&service, &downloader);

        assert!(matches!(result, Err(BootstrapError::FileNotFound(_))));
        assert_eq!(*service.ensure_workspace_calls.lock(), 0);
    }

    #[test]
    #[serial]
    fn test_the_workspace_owner_still_bootstraps() {
        let _pin = ProjectPinReset::new();
        let downloader = Arc::new(Downloader::default());
        let workspace = TempWorkspace::new();
        let service = Service::new(Arc::clone(&workspace.workspace));

        bootstrap(&service, &downloader).unwrap();

        assert_eq!(*downloader.bootstrap_download_calls.lock(), vec!["sl-mini".to_string()]);
        assert!(workspace.saved(), "the workspace record was not saved");
    }

    #[test]
    #[serial]
    fn test_the_owner_records_the_model_as_downloaded() {
        let _pin = ProjectPinReset::new();
        let downloader = Arc::new(Downloader::default());
        let workspace = TempWorkspace::new();
        let service = Service::new(Arc::clone(&workspace.workspace));

        bootstrap(&service, &downloader).unwrap();

        let model = workspace.workspace.read().get_model(Some("sl-mini")).expect("registered");
        assert!(model.downloaded);
    }
}
