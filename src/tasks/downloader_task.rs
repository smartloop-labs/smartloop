//! Port of `smartloop/tasks/downloader_task.py`.
//!
//! Downloader Task — downloads model artifacts in a background thread.
//!
//! auto=true: starts automatically when submitted.  Downloads GGUF files
//! first for fastest time-to-usable-model, then remaining artifacts.

use std::any::Any;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::json;

use crate::model::downloader::artifact_downloader::SLPArtifactDownloader;
use crate::model::downloader::slp_downloader::ProgressCallback;
use crate::tasks::base_task::{
    is_permission_exception, BaseTask, TaskCore, TaskKwargs, TaskLogger, TaskOptions, TaskResult,
};
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
        Self {
            core: TaskCore::new("downloader", logger, true, options),
            workspace,
            downloader,
            model_id: model_id.into(),
            progress_callback,
            bootstrap,
            gguf_ready: AtomicBool::new(false),
        }
    }

    /// True once the GGUF file has been downloaded.
    pub fn gguf_ready(&self) -> bool {
        self.gguf_ready.load(Ordering::SeqCst)
    }

    /// Check whether a GGUF file already exists in `<models_dir>/<model_id>`.
    fn check_gguf_exists(&self) -> bool {
        let models_dir = self.workspace.read().models_dir.clone();
        let model_dir = Path::new(&models_dir).join(&self.model_id);
        if !model_dir.is_dir() {
            return false;
        }
        match std::fs::read_dir(&model_dir) {
            Ok(entries) => entries
                .flatten()
                .any(|e| e.file_name().to_string_lossy().ends_with(".gguf")),
            Err(_) => false,
        }
    }

    /// The download itself; any error is returned to [`BaseTask::execute`].
    fn download(&self) -> anyhow::Result<()> {
        let logger = &self.core.logger;
        logger.info(&format!("[downloader] Downloading GGUF for {}", self.model_id));

        if !self.check_gguf_exists() {
            if self.bootstrap {
                self.downloader
                    .bootstrap_download(None, self.progress_callback.clone())?;
            } else {
                self.downloader
                    .download_with_artifacts(&self.model_id, self.progress_callback.clone(), None)?;
            }
        }

        self.gguf_ready.store(true, Ordering::SeqCst);
        logger.info(&format!("[downloader] GGUF ready for {}", self.model_id));

        // Mark the just-downloaded model as downloaded so on-demand
        // large-model selection (best_model_for / largest_downloaded_model,
        // which filter on `downloaded`) sees it immediately — without needing
        // a project switch to reload the workspace from disk.  Look it up by
        // model_id (the model that was actually downloaded); the no-argument
        // default-model lookup left a newly downloaded non-active model
        // flagged not-downloaded and unusable until reload.
        {
            let mut workspace = self.workspace.write();
            if let Some(ws_model) = workspace.models.iter_mut().find(|m| m.model_id == self.model_id) {
                if !ws_model.downloaded {
                    ws_model.downloaded = true;
                }
            }
        }

        logger.info(&format!("[downloader] All artifacts downloaded for {}", self.model_id));
        Ok(())
    }
}

impl BaseTask for DownloaderTask {
    fn core(&self) -> &TaskCore {
        &self.core
    }

    /// Download GGUF model artifacts (bootstrap endpoint when `bootstrap`),
    /// then flag the workspace model as downloaded.  `PermissionError` is
    /// propagated (non-retryable); other failures become a failed result.
    fn execute(&self, _kwargs: &TaskKwargs) -> anyhow::Result<TaskResult> {
        match self.download() {
            Ok(()) => Ok(TaskResult::ok(json!({ "model_id": self.model_id }))),
            // Non-retryable — re-raise so BaseTask skips retries
            Err(exc) if is_permission_exception(&exc) => Err(exc),
            Err(exc) => {
                self.core.logger.error(&format!("[downloader] Download failed: {exc}"));
                Ok(TaskResult::failure(exc.to_string()))
            }
        }
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
