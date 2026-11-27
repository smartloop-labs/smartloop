//! Port of `smartloop/model/downloader/__init__.py`.

pub mod artifact_downloader;
pub mod artifact_source;
pub mod cancellation;
pub mod digest;
pub mod hf_downloader;
pub mod inflight;
pub mod ranges;
pub mod resume;
pub mod slp_downloader;

pub use artifact_downloader::SLPArtifactDownloader;
pub use artifact_source::{ArtifactSource, SLPBootstrapSource, SLPModelSource};
pub use cancellation::{CancelToken, DownloadCancelled, DownloadInProgress};
pub use hf_downloader::HFDownloader;
pub use slp_downloader::{DownloadProgress, ProgressCallback, SLPDownloader};
