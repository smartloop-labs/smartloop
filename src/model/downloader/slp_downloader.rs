//! Port of `smartloop/model/downloader/slp_downloader.py`.
//!
//! Progress types shared by every downloader, the per-file → aggregate
//! progress roll-up, and the [`SLPDownloader`] base trait.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::utils::log_utils::LoggerLike;

/// Progress info emitted during model download.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DownloadProgress {
    pub filename: String,
    pub downloaded: u64,
    pub total: u64,
    /// `"downloading"` (default) or `"completed"`.
    pub status: String,
}

impl DownloadProgress {
    /// Build a progress event; Python's default `status` is `"downloading"`.
    pub fn new(filename: impl Into<String>, downloaded: u64, total: u64, status: impl Into<String>) -> Self {
        Self { filename: filename.into(), downloaded, total, status: status.into() }
    }

    /// `{"filename", "downloaded", "total", "status"}` as JSON.
    pub fn to_dict(&self) -> serde_json::Value {
        serde_json::json!({
            "filename": self.filename,
            "downloaded": self.downloaded,
            "total": self.total,
            "status": self.status,
        })
    }
}

/// Type alias for progress callbacks (`Callable[[DownloadProgress], None]`).
pub type ProgressCallback = Arc<dyn Fn(DownloadProgress) + Send + Sync>;

/// Roll up per-file progress into a single 0->100% download bar.
///
/// A multi-file model downloads its files in parallel, and each file reports
/// its own cumulative `downloaded`/`total`.  This wrapper sums the per-file
/// byte counts against the known grand total and emits one aggregate
/// [`DownloadProgress`] under a single stable `label`.
///
/// Thread-safe: the parallel download workers all call the same instance.
pub struct AggregateProgress {
    total: u64,
    callback: Option<ProgressCallback>,
    label: String,
    per_file: Mutex<HashMap<String, u64>>,
}

impl AggregateProgress {
    /// New aggregator over `total_bytes`, forwarding to `callback` as `label`.
    pub fn new(total_bytes: u64, callback: Option<ProgressCallback>, label: impl Into<String>) -> Self {
        Self { total: total_bytes, callback, label: label.into(), per_file: Mutex::new(HashMap::new()) }
    }

    /// Record one per-file event and emit the rolled-up event (`__call__`).
    ///
    /// Per-file counts only ever move forward so the bar never dips.
    pub fn call(&self, progress: DownloadProgress) {
        let Some(callback) = &self.callback else {
            return;
        };
        let mut downloaded = {
            let mut per_file = self.per_file.lock();
            // Per-file counts only ever move forward — ignore any regression
            // from a resumed/restarted file so the rolled-up bar never dips.
            let prev = per_file.get(&progress.filename).copied().unwrap_or(0);
            per_file.insert(progress.filename.clone(), prev.max(progress.downloaded));
            per_file.values().sum::<u64>()
        };
        if self.total > 0 {
            downloaded = downloaded.min(self.total);
        }
        callback(DownloadProgress::new(self.label.clone(), downloaded, self.total, "downloading"));
    }

    /// Emit a terminal 100% `completed` event for the whole download.
    pub fn finish(&self) {
        let Some(callback) = &self.callback else {
            return;
        };
        callback(DownloadProgress::new(self.label.clone(), self.total, self.total, "completed"));
    }

    /// Wrap a shared aggregator as a [`ProgressCallback`].
    pub fn as_callback(self: &Arc<Self>) -> ProgressCallback {
        let this = Arc::clone(self);
        Arc::new(move |progress| this.call(progress))
    }
}

/// Shared logger handle passed to downloaders and models (`logging.Logger`).
pub type SharedLogger = Arc<dyn LoggerLike>;

/// Shared, mutable workspace handle (re-exported from the workspace module).
pub use crate::workspace::workspace::SharedWorkspace;

/// Base class for model downloaders.
pub trait SLPDownloader: Send + Sync {
    /// Download `model_id` into the workspace's models directory and return
    /// the local directory.
    fn download(&self, model_id: &str, progress_callback: Option<ProgressCallback>) -> anyhow::Result<PathBuf>;
}

/// Which download entry point a [`DownloadStubs`] hook intercepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownloadStubCall {
    /// `download(model_id, ...)` on any downloader.
    Download { model_id: String, models_dir: Option<PathBuf> },
    /// `SLPArtifactDownloader::bootstrap_download(model_name, ...)`.
    BootstrapDownload { model_name: Option<String>, models_dir: Option<PathBuf> },
}

/// A process-wide download interceptor.
pub type DownloadStubFn = Arc<dyn Fn(DownloadStubCall) -> anyhow::Result<PathBuf> + Send + Sync>;

/// A process-wide `resolve_artifacts` interceptor.
pub type ResolveStubFn =
    Arc<dyn Fn(&str) -> anyhow::Result<crate::model::downloader::artifact_source::ArtifactMap> + Send + Sync>;

/// Test seam replacing the Python tests' class-level monkeypatching of
/// `SLPArtifactDownloader.download` / `.bootstrap_download` /
/// `.resolve_artifacts` and `SLPDownloader.download`: when installed, every
/// downloader consults these before touching the network or disk.
#[derive(Clone, Default)]
pub struct DownloadStubs {
    pub download: Option<DownloadStubFn>,
    pub bootstrap_download: Option<DownloadStubFn>,
    pub resolve_artifacts: Option<ResolveStubFn>,
}

static DOWNLOAD_STUBS: once_cell::sync::Lazy<parking_lot::RwLock<Option<DownloadStubs>>> =
    once_cell::sync::Lazy::new(|| parking_lot::RwLock::new(None));

/// Install (or with `None`, remove) the process-wide download stubs.  Tests only.
pub fn set_download_stubs(stubs: Option<DownloadStubs>) {
    *DOWNLOAD_STUBS.write() = stubs;
}

/// The currently installed download stubs, if any.
pub fn download_stubs() -> Option<DownloadStubs> {
    DOWNLOAD_STUBS.read().clone()
}
