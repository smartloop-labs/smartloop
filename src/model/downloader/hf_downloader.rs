//! Port of `smartloop/model/downloader/hf_downloader.py`.
//!
//! Downloads a model repository from the HuggingFace Hub with byte-level
//! progress.  `huggingface_hub` has no Rust equivalent, so the two calls it
//! made (`HfApi.list_repo_tree(recursive=True)` and `hf_hub_download`) are
//! hand-ported over reqwest against the Hub's public HTTP API.

use std::path::PathBuf;

use crate::model::downloader::slp_downloader::{
    AggregateProgress, DownloadProgress, ProgressCallback, SLPDownloader, SharedLogger, SharedWorkspace,
};

/// Parallel file downloads (`min(4, cpu_count)`).
pub fn max_workers() -> usize {
    todo!()
}

/// Seconds between progress polls of the on-disk file sizes.
pub const POLL_INTERVAL: f64 = 0.5;

/// Base URL of the HuggingFace Hub.
pub const HF_ENDPOINT: &str = "https://huggingface.co";

/// One file entry of a repository tree (`RepoFile` with a known size).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub struct RepoFile {
    pub path: String,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(rename = "type", default)]
    pub kind: String,
}

/// List every entry of `repo_id`'s tree recursively (`HfApi.list_repo_tree`).
pub fn list_repo_tree(repo_id: &str, token: Option<&str>) -> anyhow::Result<Vec<RepoFile>> {
    let _ = (repo_id, token);
    todo!()
}

/// Download one file of `repo_id` into `local_dir/<filename>`, writing to a
/// `.incomplete` sidecar first (`hf_hub_download(local_dir=...)`).
pub fn hf_hub_download(repo_id: &str, filename: &str, local_dir: &std::path::Path, token: Option<&str>) -> anyhow::Result<PathBuf> {
    let _ = (repo_id, filename, local_dir, token);
    todo!()
}

/// HuggingFace Hub downloader.
pub struct HFDownloader {
    pub workspace: SharedWorkspace,
    pub logger: SharedLogger,
}

impl HFDownloader {
    /// New downloader for `workspace`.
    pub fn new(workspace: SharedWorkspace, logger: SharedLogger) -> Self {
        Self { workspace, logger }
    }
}

impl SLPDownloader for HFDownloader {
    /// Download model from HuggingFace Hub with byte-level progress.
    fn download(&self, model_id: &str, progress_callback: Option<ProgressCallback>) -> anyhow::Result<PathBuf> {
        let _ = (model_id, progress_callback);
        let _: Option<(AggregateProgress, DownloadProgress)> = None;
        todo!()
    }
}
