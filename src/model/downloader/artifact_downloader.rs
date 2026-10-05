//! Port of `smartloop/model/downloader/artifact_downloader.py`.
//!
//! Downloads model artifacts from signed URLs provided by an
//! [`ArtifactSource`]: parallel files, parallel Range chunks for large files,
//! retry with exponential backoff + jitter and resume, MD5 verification and an
//! atomic `.part` → final publish.  Errors keep their *kind* (transient vs.
//! permanent, disk-full, cancelled) so the API layer can choose a status.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use crate::config::AppSettings;
#[allow(unused_imports)]
use crate::constants::{canonical_model_name, display_model_name, platform_model_name};
use crate::model::downloader::artifact_source::{
    ArtifactMap, ArtifactSource, BootstrapArtifactSource, SLPBootstrapSource, SLPModelSource,
};
use crate::model::downloader::cancellation::{CancelToken, DownloadCancelled};
#[allow(unused_imports)]
use crate::model::downloader::inflight::downloading;
use crate::model::downloader::slp_downloader::{
    AggregateProgress, DownloadProgress, ProgressCallback, SLPDownloader, SharedLogger, SharedWorkspace,
};
use crate::platform::SLPPlatformClient;

/// Total concurrent connections budget (`min(32, cpu_count * 2)`).
pub fn io_thread_budget() -> usize {
    todo!()
}

/// Parallel file downloads (`min(4, cpu_count)`; capped: I/O-bound).
pub fn max_workers() -> usize {
    todo!()
}

/// Parallel chunk downloads per file (`max(2, io_thread_budget / max_workers)`).
pub fn chunk_workers() -> usize {
    todo!()
}

/// 8 MB read buffer for simple downloads (and MD5 reads).
pub const CHUNK_SIZE: usize = 8 * 1024 * 1024;
/// 1 MB read buffer within each chunk.
pub const RANGE_READ_SIZE: usize = 1024 * 1024;
/// Files larger than this (100 MB) use chunked download.
pub const CHUNKED_THRESHOLD: u64 = 100 * 1024 * 1024;

/// Retry policy for transient network failures during download.  Sized so the
/// cumulative backoff window (~1+2+4+8+16+30+30 ≈ 90s) bridges a typical 1–2
/// minute network blip.
pub const MAX_DOWNLOAD_ATTEMPTS: u32 = 8;
/// Seconds; first backoff.
pub const RETRY_BASE_DELAY: f64 = 1.0;
/// Seconds; backoff cap.
pub const RETRY_MAX_DELAY: f64 = 30.0;
/// Transient HTTP statuses.
pub const RETRYABLE_STATUS: [u16; 6] = [408, 429, 500, 502, 503, 504];

/// Per-operation timeout.  The read timeout is deliberately short so a stalled
/// stream is detected in ~1 minute and retried; it bounds the idle gap between
/// received bytes, not the total transfer time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamTimeout {
    pub connect: Duration,
    pub read: Duration,
    pub write: Duration,
    pub pool: Duration,
}

/// `httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0)`.
pub const STREAM_TIMEOUT: StreamTimeout = StreamTimeout {
    connect: Duration::from_secs(30),
    read: Duration::from_secs(60),
    write: Duration::from_secs(30),
    pool: Duration::from_secs(30),
};

/// How many times to re-download a file whose bytes fail MD5 verification.
pub const MAX_VERIFY_ATTEMPTS: u32 = 3;

/// Statuses that mean the signed URL went stale, not that the caller is
/// unentitled (GCS: 403 `ExpiredToken`, 401 for an unparsable signature).
pub const STALE_URL_STATUS: [u16; 2] = [401, 403];

/// Message for a 401 on the transfer itself.
pub const UNAUTHORIZED_MSG: &str = "Download rejected (401 Unauthorized). \
The download URL may have expired or the resource requires authentication.";

/// Kind of transport failure — mirrors the `httpx.TransportError` subclasses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportErrorKind {
    Connect,
    Read,
    Write,
    RemoteProtocol,
    ConnectTimeout,
    ReadTimeout,
    WriteTimeout,
    PoolTimeout,
    Other,
}

/// An HTTP-layer failure: either a bad status (`httpx.HTTPStatusError`) or a
/// transport failure (`httpx.TransportError`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HttpError {
    #[error("{message}")]
    Status { status_code: u16, message: String },
    #[error("{message}")]
    Transport { kind: TransportErrorKind, message: String },
}

impl HttpError {
    /// Convenience constructor for a status error.
    pub fn status(status_code: u16, message: impl Into<String>) -> Self {
        HttpError::Status { status_code, message: message.into() }
    }

    /// Convenience constructor for a transport error.
    pub fn transport(kind: TransportErrorKind, message: impl Into<String>) -> Self {
        HttpError::Transport { kind, message: message.into() }
    }

    /// Map a reqwest error onto the httpx-style classification.
    pub fn from_reqwest(err: &reqwest::Error) -> Self {
        let _ = err;
        todo!()
    }
}

/// A signed URL was rejected mid-transfer, so a fresh one may well work.
///
/// A connection-class error (not a permission error) because that is the
/// action it implies: ask for new URLs, not new credentials.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct StaleDownloadURLError(pub String);

/// Everything that can escape a download, keeping the Python exception kind.
#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    /// The user asked to stop (`DownloadCancelled`).
    #[error(transparent)]
    Cancelled(#[from] DownloadCancelled),
    /// `httpx.HTTPStatusError` / `httpx.TransportError`.
    #[error(transparent)]
    Http(#[from] HttpError),
    /// A stale signed URL (`StaleDownloadURLError`, a `ConnectionError`).
    #[error(transparent)]
    StaleUrl(#[from] StaleDownloadURLError),
    /// A classified transient failure (`ConnectionError`).
    #[error("{message}")]
    Connection {
        message: String,
        #[source]
        cause: Option<Box<DownloadError>>,
    },
    /// A classified permanent failure (`RuntimeError`).
    #[error("{message}")]
    Runtime {
        message: String,
        #[source]
        cause: Option<Box<DownloadError>>,
    },
    /// Local filesystem failure (`OSError`, incl. `PermissionError`, ENOSPC).
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Bad input (`ValueError`).
    #[error("{0}")]
    Value(String),
    /// Failure raised by an artifact source / the platform (e.g.
    /// `ModelAccessError`, `PermissionError`, `ConnectionError`); downcast the
    /// inner error to recover its type.
    #[error(transparent)]
    Source(anyhow::Error),
}

impl DownloadError {
    /// `ConnectionError(message)` with an optional cause.
    pub fn connection(message: impl Into<String>, cause: Option<DownloadError>) -> Self {
        DownloadError::Connection { message: message.into(), cause: cause.map(Box::new) }
    }

    /// `RuntimeError(message)` with an optional cause.
    pub fn runtime(message: impl Into<String>, cause: Option<DownloadError>) -> Self {
        DownloadError::Runtime { message: message.into(), cause: cause.map(Box::new) }
    }

    /// True for the Python `ConnectionError` family (Connection, StaleUrl).
    pub fn is_connection_error(&self) -> bool {
        todo!()
    }

    /// True for a local `PermissionError` (IO permission denied).
    pub fn is_permission_error(&self) -> bool {
        todo!()
    }

    /// True for `DownloadCancelled`.
    pub fn is_cancelled(&self) -> bool {
        matches!(self, DownloadError::Cancelled(_))
    }

    /// The classified cause (`__cause__`), if any.
    pub fn cause(&self) -> Option<&DownloadError> {
        todo!()
    }
}

/// Result alias for downloader operations.
pub type DownloadResult<T> = Result<T, DownloadError>;

/// True when `path` is a vision mmproj / CLIP projector GGUF.
pub fn is_mmproj(path: &str) -> bool {
    let _ = path;
    todo!()
}

/// Pick which GGUFs to download from a model's artifact map.
///
/// Keeps the quantized main weights (e.g. `*-Q4_K_M.gguf`, preferred over the
/// raw F16) plus the mmproj / CLIP projector.  The main weights are never
/// dropped when the model ships no quantized variant.
pub fn select_gguf_artifacts(artifacts: &ArtifactMap) -> ArtifactMap {
    let _ = artifacts;
    todo!()
}

/// True when `exc` — or the failure it wraps — is an out-of-space IO error.
pub fn is_disk_full(exc: &DownloadError) -> bool {
    let _ = exc;
    todo!()
}

/// True for transient failures worth retrying (network blips, 5xx, 429).
pub fn is_retryable_error(exc: &DownloadError) -> bool {
    let _ = exc;
    todo!()
}

/// True when trying again — now or later — could plausibly succeed.
///
/// Wider than [`is_retryable_error`]: includes a stale signed URL, a full
/// disk and any connection-class error; never a cancel.
pub fn is_transient_error(exc: &DownloadError) -> bool {
    let _ = exc;
    todo!()
}

/// An error that reads as `message` but still says whether to try again:
/// `Connection` when transient, `Runtime` otherwise.
pub fn classified(message: impl Into<String>, transient: bool) -> DownloadError {
    let _ = (message.into(), transient);
    todo!()
}

/// Collapse per-file failures into one error that keeps their kind.
///
/// A cancel wins outright; otherwise transient only when *every* file failed
/// transiently, with " — no space left on device" appended (and the disk-full
/// error kept as the cause) when that is what happened.
pub fn aggregate_error(failed: Vec<(String, DownloadError)>) -> DownloadError {
    let _ = failed;
    todo!()
}

/// Exponential backoff with 50–100% jitter for a 1-based attempt number.
pub fn backoff_delay(attempt: u32) -> f64 {
    let _ = attempt;
    todo!()
}

/// Case-insensitive response headers.
pub type Headers = BTreeMap<String, String>;

/// Response to a HEAD probe.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HeadResponse {
    pub status_code: u16,
    /// Header names lower-cased.
    pub headers: Headers,
}

/// A streaming GET response (the body of `with httpx.stream(...) as resp`).
/// Dropping it closes the connection.
pub trait StreamResponse: Send {
    /// HTTP status code.
    fn status_code(&self) -> u16;
    /// `Err(HttpError::Status)` for a 4xx/5xx status.
    fn raise_for_status(&self) -> Result<(), HttpError>;
    /// Next body chunk of up to `chunk_size` bytes; `Ok(None)` at end of body.
    fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<Vec<u8>>, HttpError>;
}

/// The HTTP operations the downloader performs — a seam so tests can script
/// responses (the Python tests monkeypatch `httpx.head` / `httpx.stream`).
pub trait DownloadHttp: Send + Sync {
    /// `httpx.head(url, follow_redirects=True, timeout=30)`.
    fn head(&self, url: &str) -> Result<HeadResponse, HttpError>;
    /// `httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=timeout)`.
    fn stream(&self, url: &str, headers: &Headers, timeout: StreamTimeout) -> Result<Box<dyn StreamResponse>, HttpError>;
}

/// Default [`DownloadHttp`] backed by blocking reqwest.
#[derive(Debug, Clone, Default)]
pub struct ReqwestDownloadHttp;

impl DownloadHttp for ReqwestDownloadHttp {
    fn head(&self, url: &str) -> Result<HeadResponse, HttpError> {
        let _ = url;
        todo!()
    }

    fn stream(&self, url: &str, headers: &Headers, timeout: StreamTimeout) -> Result<Box<dyn StreamResponse>, HttpError> {
        let _ = (url, headers, timeout);
        todo!()
    }
}

/// Test seam replacing `_download_artifacts` (Python tests assign a function).
pub type DownloadArtifactsHook = Arc<dyn Fn(&ArtifactMap, &Path) -> DownloadResult<PathBuf> + Send + Sync>;

/// Test seam observing directory creation (Python tests patch `os.makedirs`).
pub type MakedirsHook = Arc<dyn Fn(&Path) + Send + Sync>;

/// Download model artifacts from URLs provided by an ArtifactSource.
///
/// Fetches signed URLs via artifact sources, then downloads all files in
/// parallel with MD5 verification and progress reporting.
pub struct SLPArtifactDownloader {
    workspace: Option<SharedWorkspace>,
    models_dir_override: Option<PathBuf>,
    pub logger: SharedLogger,
    /// The token a stop request will raise.
    pub cancel_token: CancelToken,
    pub settings: Option<AppSettings>,
    model_source: Arc<dyn ArtifactSource>,
    bootstrap_source: Arc<dyn BootstrapArtifactSource>,
    http: Arc<dyn DownloadHttp>,
    download_artifacts_hook: Mutex<Option<DownloadArtifactsHook>>,
    makedirs_hook: Mutex<Option<MakedirsHook>>,
}

impl SLPArtifactDownloader {
    /// Build a downloader for `workspace`.  `cancel` defaults to a fresh token,
    /// so every check point is unconditional.  Reads `AppSettings` and builds
    /// the platform-backed model / bootstrap sources.
    pub fn new(workspace: SharedWorkspace, logger: SharedLogger, cancel: Option<CancelToken>) -> Self {
        let _ = (workspace, logger, cancel);
        let _: Option<(SLPPlatformClient, SLPModelSource, SLPBootstrapSource)> = None;
        todo!()
    }

    /// Construct without a workspace or network (the Python tests'
    /// `SLPArtifactDownloader.__new__` + attribute assignment).  Sources
    /// default to ones that error if used; HTTP defaults to reqwest.
    pub fn for_tests(models_dir: Option<PathBuf>, logger: SharedLogger, cancel: Option<CancelToken>) -> Self {
        let _ = (models_dir, logger, cancel);
        todo!()
    }

    /// Replace the authenticated model source (`_model_source`).
    pub fn with_model_source(mut self, source: Arc<dyn ArtifactSource>) -> Self {
        self.model_source = source;
        self
    }

    /// Replace the bootstrap source (`_bootstrap_source`).
    pub fn with_bootstrap_source(mut self, source: Arc<dyn BootstrapArtifactSource>) -> Self {
        self.bootstrap_source = source;
        self
    }

    /// Replace the HTTP transport.
    pub fn with_http(mut self, http: Arc<dyn DownloadHttp>) -> Self {
        self.http = http;
        self
    }

    /// Tests only: replace `_download_artifacts`.
    pub fn set_download_artifacts_hook(&self, hook: Option<DownloadArtifactsHook>) {
        *self.download_artifacts_hook.lock() = hook;
    }

    /// Tests only: observe the model-directory `makedirs`.
    pub fn set_makedirs_hook(&self, hook: Option<MakedirsHook>) {
        *self.makedirs_hook.lock() = hook;
    }

    /// The workspace's models directory (or the test override).
    pub fn models_dir(&self) -> PathBuf {
        let _ = (&self.workspace, &self.models_dir_override);
        todo!()
    }

    /// Return the hex-encoded MD5 digest of a local file, checking `cancel`
    /// per read (`"verification cancelled"`).
    pub fn compute_local_md5(filepath: &Path, cancel: Option<&CancelToken>) -> DownloadResult<String> {
        let _ = (filepath, cancel);
        todo!()
    }

    /// Extract the hex-encoded MD5 from the `ETag` header (32 hex chars, quotes
    /// stripped, lower-cased); `None` otherwise.  Header lookup is
    /// case-insensitive.
    pub fn parse_remote_md5(headers: &Headers) -> Option<String> {
        let _ = headers;
        todo!()
    }

    /// Call `attempt_fn(attempt_number)` with retries on transient errors.
    ///
    /// Checks the cancel token before each attempt; permission and cancel
    /// errors propagate immediately; backoff waits on the cancel token.
    pub fn run_with_retries<T>(
        &self,
        description: &str,
        attempt_fn: &mut dyn FnMut(u32) -> DownloadResult<T>,
    ) -> DownloadResult<T> {
        let _ = (description, attempt_fn);
        todo!()
    }

    /// HEAD request → `(size, md5, accepts_ranges)`; falls back to
    /// `(0, None, false)` when the probe ultimately fails (but propagates a
    /// cancel).
    pub fn get_remote_file_info(&self, url: &str) -> DownloadResult<(u64, Option<String>, bool)> {
        let _ = url;
        todo!()
    }

    /// Check, download, and verify a set of artifacts.  Shared by bootstrap
    /// and download.
    pub fn download_artifacts(
        &self,
        artifacts: &ArtifactMap,
        local_dir: &Path,
        progress_callback: Option<ProgressCallback>,
        log_prefix: &str,
        log_dir: &Path,
        label: &str,
    ) -> DownloadResult<PathBuf> {
        let _ = (artifacts, local_dir, progress_callback, log_prefix, log_dir, label);
        let _: Option<AggregateProgress> = None;
        todo!()
    }

    /// Download a single file, then verify and atomically publish it.
    ///
    /// Bytes go to a `.part` sidecar; a corrupt download is re-fetched up to
    /// [`MAX_VERIFY_ATTEMPTS`] times before failing with
    /// `"<name>: MD5 verification failed after 3 attempts"`.
    #[allow(clippy::too_many_arguments)]
    pub fn download_file(
        &self,
        filename: &str,
        url: &str,
        local_dir: &Path,
        size: u64,
        position: usize,
        progress_callback: Option<ProgressCallback>,
        supports_range: bool,
        remote_md5: Option<&str>,
    ) -> DownloadResult<String> {
        let _ = (filename, url, local_dir, size, position, progress_callback, supports_range, remote_md5);
        todo!()
    }

    /// Download a file in a single request, retrying transient failures and
    /// resuming via Range when supported.
    pub fn download_file_simple(
        &self,
        flat_name: &str,
        url: &str,
        dest: &Path,
        size: u64,
        progress_callback: Option<ProgressCallback>,
        supports_range: bool,
    ) -> DownloadResult<()> {
        let _ = (flat_name, url, dest, size, progress_callback, supports_range);
        let _: Option<DownloadProgress> = None;
        todo!()
    }

    /// Download a byte range to its file offset, retrying transient failures
    /// and resuming from within the range; `downloaded_counter` only ever
    /// advances by newly written bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn download_chunk(
        &self,
        url: &str,
        dest: &Path,
        start: u64,
        end: u64,
        flat_name: &str,
        total_size: u64,
        downloaded_counter: &Mutex<u64>,
        progress_callback: Option<ProgressCallback>,
    ) -> DownloadResult<()> {
        let _ = (url, dest, start, end, flat_name, total_size, downloaded_counter, progress_callback);
        todo!()
    }

    /// Download a large file using parallel Range requests.
    pub fn download_file_chunked(
        &self,
        flat_name: &str,
        url: &str,
        dest: &Path,
        size: u64,
        progress_callback: Option<ProgressCallback>,
    ) -> DownloadResult<()> {
        let _ = (flat_name, url, dest, size, progress_callback);
        todo!()
    }

    /// Download GGUF model via the unauthenticated `/v1/bootstrap/models`
    /// endpoint (first-run scenarios).  Downloads `model_name`'s artifacts, or
    /// the first model returned when `None`.
    pub fn bootstrap_download(
        &self,
        model_name: Option<&str>,
        progress_callback: Option<ProgressCallback>,
    ) -> DownloadResult<PathBuf> {
        let _ = (model_name, progress_callback);
        todo!()
    }

    /// Lookup the HuggingFace model_id from a short model name (falls back to
    /// `model_name`), e.g. `"sl-mini"` → `"smartloop-ai/slp-orchestra-mini"`.
    pub fn resolve_model_id(model_name: &str) -> String {
        let _ = model_name;
        todo!()
    }

    /// Reverse-lookup the canonical short model name from a model_id (falls
    /// back to `model_id`).
    pub fn resolve_model_name(model_id: &str) -> String {
        let _ = model_id;
        todo!()
    }

    /// Ask the platform for this model's signed GGUF URLs (the entitlement
    /// step), filtered through [`select_gguf_artifacts`].
    pub fn resolve_artifacts(&self, model_id: &str) -> DownloadResult<ArtifactMap> {
        let _ = model_id;
        todo!()
    }

    /// Download GGUF model via `/v1/models/{model_name}/download`, storing
    /// files under the model_id path.  `artifacts` accepts an already-resolved
    /// map from [`Self::resolve_artifacts`].
    pub fn download_with_artifacts(
        &self,
        model_id: &str,
        progress_callback: Option<ProgressCallback>,
        artifacts: Option<ArtifactMap>,
    ) -> DownloadResult<PathBuf> {
        let _ = (model_id, progress_callback, artifacts);
        todo!()
    }
}

impl SLPDownloader for SLPArtifactDownloader {
    fn download(&self, model_id: &str, progress_callback: Option<ProgressCallback>) -> anyhow::Result<PathBuf> {
        Ok(self.download_with_artifacts(model_id, progress_callback, None)?)
    }
}
