//! Port of `smartloop/tasks/downloader_task.py`.
//!
//! Downloader Task — downloads model artifacts in a background thread.
//!
//! auto=true: starts automatically when submitted.  Downloads GGUF files
//! first for fastest time-to-usable-model, then remaining artifacts.

use std::any::Any;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::model::downloader::artifact_downloader::SLPArtifactDownloader;
use crate::model::downloader::slp_downloader::ProgressCallback;
use crate::tasks::base_task::{BaseTask, TaskCore, TaskKwargs, TaskLogger, TaskOptions, TaskResult};
use crate::workspace::workspace::SharedWorkspace;

/// Background task that downloads model artifacts.
///
/// Downloads GGUF first so the model can be loaded while the remaining
/// training artifacts are still downloading.
pub struct DownloaderTask {
    core: TaskCore,
    pub workspace: SharedWorkspace,
    pub downloader: Arc<SLPArtifactDownloader>,
    pub model_id: String,
    pub progress_callback: Option<ProgressCallback>,
    pub bootstrap: bool,
    gguf_ready: AtomicBool,
}

impl DownloaderTask {
    /// Task name `"downloader"`, `auto=true`.
    pub fn new(
        workspace: SharedWorkspace,
        downloader: Arc<SLPArtifactDownloader>,
        logger: TaskLogger,
        model_id: impl Into<String>,
        progress_callback: Option<ProgressCallback>,
        bootstrap: bool,
        options: TaskOptions,
    ) -> Self {
        let _ = (workspace, downloader, logger, model_id.into(), progress_callback, bootstrap, options);
        todo!()
    }

    /// True once the GGUF file has been downloaded.
    pub fn gguf_ready(&self) -> bool {
        let _ = &self.gguf_ready;
        todo!()
    }

    /// Check whether a GGUF file already exists in `<models_dir>/<model_id>`.
    fn check_gguf_exists(&self) -> bool {
        todo!()
    }
}

impl BaseTask for DownloaderTask {
    fn core(&self) -> &TaskCore {
        &self.core
    }

    /// Download GGUF model artifacts (bootstrap endpoint when `bootstrap`),
    /// then flag the workspace model as downloaded.  `PermissionError` is
    /// propagated (non-retryable); other failures become a failed result.
    fn execute(&self, kwargs: &TaskKwargs) -> anyhow::Result<TaskResult> {
        let _ = kwargs;
        todo!()
    }

    /// Skip if the GGUF already exists.
    fn should_run(&self) -> bool {
        !self.check_gguf_exists()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn into_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }
}
