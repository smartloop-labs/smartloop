//! Port of `smartloop/model/downloader/artifact_downloader.py`.
//!
//! Downloads model artifacts from signed URLs provided by an
//! [`ArtifactSource`]: parallel files, parallel Range chunks for large files,
//! retry with exponential backoff + jitter and resume, MD5 verification and an
//! atomic `.part` → final publish.  Errors keep their *kind* (transient vs.
//! permanent, disk-full, cancelled) so the API layer can choose a status.

use std::collections::{BTreeMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use md5::{Digest, Md5};
use once_cell::sync::OnceCell;
use parking_lot::Mutex;

use crate::config::{AppSettings, SUPPORTED_MODELS};
use crate::constants::{canonical_model_name, display_model_name, platform_model_name};
use crate::model::downloader::artifact_source::{
    ArtifactMap, ArtifactSource, BootstrapArtifactSource, SLPBootstrapSource, SLPModelSource, available_names,
    entry_name,
};
use crate::model::downloader::cancellation::{CancelToken, DownloadCancelled};
use crate::model::downloader::inflight::downloading;
use crate::model::downloader::slp_downloader::{
    AggregateProgress, DownloadProgress, DownloadStubCall, ProgressCallback, SLPDownloader, SharedLogger,
    SharedWorkspace, download_stubs,
};
use crate::platform::{ModelAccessError, PlatformError, SLPPlatformClient};

/// `os.cpu_count() or 4`.
fn cpu_count() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// Total concurrent connections budget (`min(32, cpu_count * 2)`).
pub fn io_thread_budget() -> usize {
    32.min(cpu_count() * 2)
}

/// Parallel file downloads (`min(4, cpu_count)`; capped: I/O-bound).
pub fn max_workers() -> usize {
    4.min(cpu_count())
}

/// Parallel chunk downloads per file (`max(2, io_thread_budget / max_workers)`).
pub fn chunk_workers() -> usize {
    2.max(io_thread_budget() / max_workers())
}

/// 8 MB read buffer for simple downloads (and MD5 reads).
pub const CHUNK_SIZE: usize = 8 * 1024 * 1024;
/// 1 MB read buffer within each chunk.
pub const RANGE_READ_SIZE: usize = 1024 * 1024;
/// Files larger than this (100 MB) use chunked download.
pub const CHUNKED_THRESHOLD: u64 = 100 * 1024 * 1024;

/// Retry policy for transient network failures during download.  A dropped
/// connection mid-stream on a multi-GB GGUF would otherwise fail the whole
/// download; instead we retry with exponential backoff + jitter and resume
/// from where we left off.  The attempt count is sized so the cumulative
/// backoff window (~1+2+4+8+16+30+30 ≈ 90s) bridges a typical 1–2 minute
/// network blip — during an outage, connection attempts fail almost instantly,
/// so it's the backoff, not the attempt count alone, that spans the gap.
pub const MAX_DOWNLOAD_ATTEMPTS: u32 = 8;
/// Seconds; first backoff.
pub const RETRY_BASE_DELAY: f64 = 1.0;
/// Seconds; backoff cap.
pub const RETRY_MAX_DELAY: f64 = 30.0;
/// Transient HTTP statuses.
pub const RETRYABLE_STATUS: [u16; 6] = [408, 429, 500, 502, 503, 504];

/// Per-operation timeout.  The read timeout is deliberately short so a stalled
/// stream (e.g. Wi-Fi dropped mid-download) is detected in ~1 minute and
/// retried, instead of blocking on a dead socket for the whole read window.  A
/// large file is still allowed to take arbitrarily long overall — this bounds
/// the idle gap between received bytes, not the total transfer time.
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

/// Timeout of the HEAD probe (`httpx.head(..., timeout=30)`).
pub const HEAD_TIMEOUT: Duration = Duration::from_secs(30);

/// How many times to re-download a file whose bytes fail MD5 verification
/// (silent corruption that completed without a network error).
pub const MAX_VERIFY_ATTEMPTS: u32 = 3;

/// Statuses that mean the signed URL went stale, not that the caller is
/// unentitled.  Entitlement is settled before the transfer starts (see
/// [`SLPArtifactDownloader::resolve_artifacts`]), so an auth refusal *here* is
/// about the URL rather than the account: GCS answers 403
/// `<Code>ExpiredToken</Code>` for a signature past `X-Goog-Expires` and 401
/// for one it cannot parse.  Telling the user to sign in would be a dead end —
/// they already are, or did not need to be.
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

/// An HTTP-layer failure: a bad status (`httpx.HTTPStatusError`), a transport
/// failure (`httpx.TransportError`), or a request that could not be made at
/// all (`httpx.TooManyRedirects` / `httpx.InvalidURL` — neither is a
/// transport error, so neither is retried).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HttpError {
    #[error("{message}")]
    Status { status_code: u16, message: String },
    #[error("{message}")]
    Transport { kind: TransportErrorKind, message: String },
    #[error("{message}")]
    Request { message: String },
}

/// `err` followed by every source in its chain, `": "`-separated.
fn error_chain_message(err: &(dyn std::error::Error + 'static)) -> String {
    let mut message = err.to_string();
    let mut source = err.source();
    while let Some(cause) = source {
        let text = cause.to_string();
        if !message.contains(&text) {
            message.push_str(": ");
            message.push_str(&text);
        }
        source = cause.source();
    }
    message
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
        let message = error_chain_message(err);
        if err.is_timeout() {
            let kind = if err.is_connect() { TransportErrorKind::ConnectTimeout } else { TransportErrorKind::ReadTimeout };
            return HttpError::transport(kind, message);
        }
        if err.is_connect() {
            return HttpError::transport(TransportErrorKind::Connect, message);
        }
        if let Some(status) = err.status() {
            return HttpError::status(status.as_u16(), message);
        }
        if err.is_redirect() || err.is_builder() {
            return HttpError::Request { message };
        }
        if err.is_body() || err.is_decode() {
            return HttpError::transport(TransportErrorKind::Read, message);
        }
        if err.is_request() {
            return HttpError::transport(TransportErrorKind::RemoteProtocol, message);
        }
        HttpError::transport(TransportErrorKind::Other, message)
    }

    /// Map an IO error raised while reading a response body.
    pub fn from_body_io(err: &std::io::Error) -> Self {
        let kind = if err.kind() == std::io::ErrorKind::TimedOut {
            TransportErrorKind::ReadTimeout
        } else {
            TransportErrorKind::Read
        };
        HttpError::transport(kind, error_chain_message(err))
    }

    /// The HTTP status of a status error.
    pub fn status_code(&self) -> Option<u16> {
        match self {
            HttpError::Status { status_code, .. } => Some(*status_code),
            _ => None,
        }
    }
}

/// A signed URL was rejected mid-transfer, so a fresh one may well work.
///
/// A connection-class error (not a permission error) because that is the
/// action it implies: the account was already cleared before the stream
/// opened, and the fix is to ask for new URLs, not new credentials.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct StaleDownloadURLError(pub String);

/// Everything that can escape a download, keeping the Python exception kind.
#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    /// The user asked to stop (`DownloadCancelled`).
    #[error(transparent)]
    Cancelled(#[from] DownloadCancelled),
    /// The user asked to stop while a transient failure was being retried
    /// (`raise DownloadCancelled(...) from exc`): still a cancel, with the
    /// failure it interrupted kept as its cause.
    #[error("{cancelled}")]
    CancelledBy {
        cancelled: DownloadCancelled,
        #[source]
        cause: Box<DownloadError>,
    },
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

    /// Wrap an error raised by an artifact source.  An error that already is a
    /// [`DownloadError`] (e.g. a bootstrap `ValueError`) is unwrapped so its
    /// kind survives the trip through `anyhow`.
    pub fn from_source(err: anyhow::Error) -> Self {
        let err = match err.downcast::<DownloadError>() {
            Ok(inner) => return inner,
            Err(err) => err,
        };
        match err.downcast::<DownloadCancelled>() {
            Ok(cancelled) => DownloadError::Cancelled(cancelled),
            Err(err) => DownloadError::Source(err),
        }
    }

    /// Convert into an `anyhow::Error`.  A [`DownloadError::Source`] hands
    /// back the original error and a plain cancel becomes a bare
    /// [`DownloadCancelled`], so callers can downcast either directly; every
    /// other kind stays a [`DownloadError`] (see [`cancelled_in`] for a
    /// lookup that finds a cancel in any of these shapes).
    pub fn into_anyhow(self) -> anyhow::Error {
        match self {
            DownloadError::Source(err) => err,
            DownloadError::Cancelled(cancelled) => anyhow::Error::new(cancelled),
            other => anyhow::Error::new(other),
        }
    }

    /// Attach `cause` (`exc.__cause__ = cause`) to a classified error; other
    /// kinds are returned unchanged.
    pub fn with_cause(self, cause: DownloadError) -> Self {
        match self {
            DownloadError::Connection { message, .. } => DownloadError::connection(message, Some(cause)),
            DownloadError::Runtime { message, .. } => DownloadError::runtime(message, Some(cause)),
            DownloadError::Cancelled(cancelled) | DownloadError::CancelledBy { cancelled, .. } => {
                DownloadError::CancelledBy { cancelled, cause: Box::new(cause) }
            }
            other => other,
        }
    }

    /// True for the Python `ConnectionError` family: classified transient
    /// failures, stale URLs, connection-class OS errors and platform
    /// `ConnectionError`s.
    pub fn is_connection_error(&self) -> bool {
        match self {
            DownloadError::Connection { .. } | DownloadError::StaleUrl(_) => true,
            DownloadError::Io(err) => is_connection_io(err),
            DownloadError::Source(err) => err.chain().any(|cause| {
                cause.downcast_ref::<PlatformError>().is_some_and(PlatformError::is_connection)
                    || cause.downcast_ref::<StaleDownloadURLError>().is_some()
                    || cause.downcast_ref::<std::io::Error>().is_some_and(is_connection_io)
            }),
            _ => false,
        }
    }

    /// True for a `PermissionError`: a local permission denial, or a platform
    /// refusal (`PermissionError` / `ModelAccessError`).
    pub fn is_permission_error(&self) -> bool {
        match self {
            DownloadError::Io(err) => err.kind() == std::io::ErrorKind::PermissionDenied,
            DownloadError::Source(err) => {
                if let Some(platform) = err.downcast_ref::<PlatformError>() {
                    return platform.is_permission();
                }
                if err.downcast_ref::<ModelAccessError>().is_some() {
                    return true;
                }
                err.downcast_ref::<std::io::Error>()
                    .is_some_and(|io| io.kind() == std::io::ErrorKind::PermissionDenied)
            }
            _ => false,
        }
    }

    /// True for `DownloadCancelled`.
    pub fn is_cancelled(&self) -> bool {
        matches!(self, DownloadError::Cancelled(_) | DownloadError::CancelledBy { .. })
    }

    /// The `DownloadCancelled` this error is, if it is one.
    pub fn as_cancelled(&self) -> Option<&DownloadCancelled> {
        match self {
            DownloadError::Cancelled(cancelled) | DownloadError::CancelledBy { cancelled, .. } => Some(cancelled),
            _ => None,
        }
    }

    /// The classified cause (`__cause__`), if any.
    pub fn cause(&self) -> Option<&DownloadError> {
        match self {
            DownloadError::Connection { cause, .. } | DownloadError::Runtime { cause, .. } => cause.as_deref(),
            DownloadError::CancelledBy { cause, .. } => Some(cause),
            _ => None,
        }
    }

    /// The HTTP status the API should answer when the error itself carries one
    /// (`getattr(exc, "status_code", None)`): 499 for a cancel, the platform
    /// status for a platform refusal.
    pub fn status_code(&self) -> Option<u16> {
        match self {
            DownloadError::Cancelled(c) | DownloadError::CancelledBy { cancelled: c, .. } => Some(c.status_code()),
            DownloadError::Source(err) => {
                if let Some(platform) = err.downcast_ref::<PlatformError>() {
                    return platform.status_code();
                }
                err.downcast_ref::<ModelAccessError>().map(|e| e.status_code)
            }
            _ => None,
        }
    }
}

/// The [`DownloadCancelled`] an `anyhow::Error` carries, whether as a bare
/// cancel or as a [`DownloadError`] cancel anywhere in its chain — the
/// `isinstance(exc, DownloadCancelled)` test for errors that crossed an
/// `anyhow` boundary (e.g. [`SLPDownloader::download`]).
pub fn cancelled_in(err: &anyhow::Error) -> Option<DownloadCancelled> {
    err.chain().find_map(|cause| {
        cause
            .downcast_ref::<DownloadCancelled>()
            .cloned()
            .or_else(|| cause.downcast_ref::<DownloadError>().and_then(DownloadError::as_cancelled).cloned())
    })
}

/// Python's `ConnectionError` subclasses among OS errors.
fn is_connection_io(err: &std::io::Error) -> bool {
    use std::io::ErrorKind::*;
    matches!(err.kind(), ConnectionRefused | ConnectionReset | ConnectionAborted | BrokenPipe)
}

/// Result alias for downloader operations.
pub type DownloadResult<T> = Result<T, DownloadError>;

/// `os.path.basename` for a POSIX-style artifact name.
fn basename(path: &str) -> &str {
    let sep = |c: char| c == '/' || (cfg!(windows) && c == '\\');
    path.rsplit(sep).next().unwrap_or(path)
}

/// True when `path` is a vision mmproj / CLIP projector GGUF.
pub fn is_mmproj(path: &str) -> bool {
    let name = basename(path).to_lowercase();
    name.contains("mmproj") || name.contains("clip")
}

/// Pick which GGUFs to download from a model's artifact map.
///
/// Keeps the quantized main weights (e.g. `*-Q4_K_M.gguf`, preferred over the
/// raw F16) plus the mmproj / CLIP projector (vision models need it).  The
/// mmproj is never counted as the main weights, and the main weights are never
/// dropped when the model ships no quantized variant (the bug being fixed:
/// `{**quantized, **mmproj}` collapsed to mmproj-only for F16-only models,
/// leaving no loadable main GGUF).
pub fn select_gguf_artifacts(artifacts: &ArtifactMap) -> ArtifactMap {
    let gguf: Vec<(&String, &String)> = artifacts.iter().filter(|(k, _)| basename(k).ends_with(".gguf")).collect();
    let mmproj: Vec<(&String, &String)> = gguf.iter().copied().filter(|(k, _)| is_mmproj(k)).collect();
    let main: Vec<(&String, &String)> = gguf.iter().copied().filter(|(k, _)| !is_mmproj(k)).collect();
    let quantized: Vec<(&String, &String)> =
        main.iter().copied().filter(|(k, _)| basename(k).to_uppercase().contains("-Q")).collect();
    let weights = if quantized.is_empty() { main } else { quantized };
    let mut selected = ArtifactMap::new();
    for (k, v) in weights.into_iter().chain(mmproj) {
        selected.insert(k.clone(), v.clone());
    }
    selected
}

/// True when `exc` — or the failure it wraps — is an out-of-space IO error.
///
/// The chunked path wraps the transfer's IO error in a classified connection
/// error before it reaches [`aggregate_error`] (see [`classified`]), so the
/// errno may live on the cause rather than on the error itself — follow the
/// chain instead of reading one frame.
pub fn is_disk_full(exc: &DownloadError) -> bool {
    let enospc = |err: &std::io::Error| err.raw_os_error() == Some(libc::ENOSPC);
    let mut current = Some(exc);
    while let Some(err) = current {
        match err {
            DownloadError::Io(io) if enospc(io) => return true,
            DownloadError::Source(source) => {
                if source.chain().any(|c| c.downcast_ref::<std::io::Error>().is_some_and(enospc)) {
                    return true;
                }
            }
            _ => {}
        }
        current = err.cause();
    }
    false
}

/// True for transient failures worth retrying (network blips, 5xx, 429).
pub fn is_retryable_error(exc: &DownloadError) -> bool {
    match exc {
        DownloadError::Http(HttpError::Status { status_code, .. }) => RETRYABLE_STATUS.contains(status_code),
        // TransportError covers connect/read/write timeouts, connection resets,
        // and RemoteProtocolError (server closed the stream / incomplete read).
        DownloadError::Http(HttpError::Transport { .. }) => true,
        _ => false,
    }
}

/// True when trying again — now or later — could plausibly succeed.
///
/// Wider than [`is_retryable_error`], which decides whether *this* call
/// retries immediately.  This decides how the failure is described to whoever
/// asked for the download, and a stale signed URL belongs here even though
/// retrying the same URL is pointless: re-requesting the download works.
///
/// A full disk is the point of this function: retrying the same URL *is*
/// pointless, nobody can run out of space forever, and some space just got
/// freed.  Classification lives on this side, not [`is_retryable_error`], so
/// the transfer itself still fails fast — a disk that cannot accept bytes must
/// not grind through [`MAX_DOWNLOAD_ATTEMPTS`] with backoff.
pub fn is_transient_error(exc: &DownloadError) -> bool {
    if exc.is_cancelled() {
        // Asked for, so nothing about it is worth retrying or describing as a
        // blip.  Stated here as well as at every raise site because this is
        // the function that decides what the client is told, and "transient"
        // would have the router answer 502 for a button the user just pressed.
        return false;
    }
    if is_disk_full(exc) {
        return true;
    }
    if is_retryable_error(exc) {
        return true;
    }
    if let DownloadError::Http(HttpError::Status { status_code, .. }) = exc {
        return STALE_URL_STATUS.contains(status_code);
    }
    exc.is_connection_error()
}

/// An error that reads as `message` but still says whether to try again:
/// `Connection` when transient, `Runtime` otherwise.
///
/// The API router turns whatever escapes the download onto an HTTP status for
/// its own client, and the only thing that status has to convey is whether
/// trying again could help.  Re-raising as a bare runtime error — which every
/// layer here used to do — answered "500, something broke" for a dropped
/// socket and a corrupt file alike.
pub fn classified(message: impl Into<String>, transient: bool) -> DownloadError {
    if transient { DownloadError::connection(message, None) } else { DownloadError::runtime(message, None) }
}

/// Collapse per-file failures into one error that keeps their kind.
///
/// Transient only when *every* file failed transiently: one unfixable file
/// means the download stays broken, so promising a retry would be a lie.
///
/// Names a disk-full when that is what happened.  The previous message listed
/// nothing but filenames, so an out-of-disk failure read exactly like a
/// dropped socket, and the one piece of information that has an action — "no
/// space left on device" — never reached the client.
pub fn aggregate_error(mut failed: Vec<(String, DownloadError)>) -> DownloadError {
    if let Some(index) = failed.iter().position(|(_, exc)| exc.is_cancelled()) {
        // Failures come back in whatever order they land, so a cancel and a
        // sibling's dropped socket in the same instant could otherwise be
        // reported either way round.  The cancel is the true account of why
        // the batch stopped, and it is the one the stream has to see to answer
        // `cancelled` instead of `error`.
        return failed.swap_remove(index).1;
    }

    let names: Vec<&str> = failed.iter().map(|(name, _)| name.as_str()).collect();
    let mut message = format!("Failed to download {} file(s): {}", failed.len(), names.join(", "));
    let disk_full = failed.iter().position(|(_, exc)| is_disk_full(exc));
    if disk_full.is_some() {
        message.push_str(" — no space left on device");
    }
    let result = classified(message, failed.iter().all(|(_, exc)| is_transient_error(exc)));
    match disk_full {
        // Keep the errno reachable: `status_for` walks the cause chain to
        // answer 507 rather than 502, and the classified remap above would
        // otherwise discard the only frame that still has it.
        Some(index) => result.with_cause(failed.swap_remove(index).1),
        None => result,
    }
}

/// Exponential backoff with 50–100% jitter for a 1-based attempt number.
pub fn backoff_delay(attempt: u32) -> f64 {
    let exponent = i32::try_from(attempt).unwrap_or(i32::MAX).saturating_sub(1);
    let delay = (RETRY_BASE_DELAY * 2f64.powi(exponent)).min(RETRY_MAX_DELAY);
    delay * (0.5 + rand::random::<f64>() * 0.5)
}

/// Case-insensitive response headers.
pub type Headers = BTreeMap<String, String>;

/// Case-insensitive header lookup (`httpx.Headers.get`).
pub fn header_get<'a>(headers: &'a Headers, name: &str) -> Option<&'a str> {
    headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
}

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

/// Shared HEAD client (never dropped, so safe to have been built anywhere).
static HEAD_CLIENT: OnceCell<reqwest::blocking::Client> = OnceCell::new();
/// Shared streaming client for [`STREAM_TIMEOUT`].
static STREAM_CLIENT: OnceCell<reqwest::blocking::Client> = OnceCell::new();

fn build_client(connect: Duration, io: Duration) -> Result<reqwest::blocking::Client, HttpError> {
    // The blocking client's `timeout` bounds each read of the body (and the
    // wait for the response head), which is httpx's per-operation read timeout.
    reqwest::blocking::Client::builder()
        .connect_timeout(connect)
        .timeout(io)
        .redirect(reqwest::redirect::Policy::limited(20))
        .build()
        .map_err(|e| HttpError::from_reqwest(&e))
}

fn stream_client(timeout: StreamTimeout) -> Result<reqwest::blocking::Client, HttpError> {
    if timeout == STREAM_TIMEOUT {
        return STREAM_CLIENT.get_or_try_init(|| build_client(timeout.connect, timeout.read)).cloned();
    }
    build_client(timeout.connect, timeout.read)
}

/// httpx-style message for a failing status.
fn status_message(status: reqwest::StatusCode, url: &str) -> String {
    let kind = match status.as_u16() {
        100..=199 => "Informational response",
        300..=399 => "Redirect response",
        400..=499 => "Client error",
        500..=599 => "Server error",
        _ => "Invalid status code",
    };
    format!(
        "{kind} '{} {}' for url '{url}'\nFor more information check: \
         https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/{}",
        status.as_u16(),
        status.canonical_reason().unwrap_or(""),
        status.as_u16()
    )
}

struct ReqwestStream {
    response: reqwest::blocking::Response,
    url: String,
}

impl StreamResponse for ReqwestStream {
    fn status_code(&self) -> u16 {
        self.response.status().as_u16()
    }

    fn raise_for_status(&self) -> Result<(), HttpError> {
        let status = self.response.status();
        if status.is_client_error() || status.is_server_error() {
            return Err(HttpError::status(status.as_u16(), status_message(status, &self.url)));
        }
        Ok(())
    }

    fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<Vec<u8>>, HttpError> {
        let mut buf = vec![0u8; chunk_size.max(1)];
        let mut filled = 0usize;
        while filled < buf.len() {
            match self.response.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(HttpError::from_body_io(&e)),
            }
        }
        if filled == 0 {
            return Ok(None);
        }
        buf.truncate(filled);
        Ok(Some(buf))
    }
}

impl DownloadHttp for ReqwestDownloadHttp {
    fn head(&self, url: &str) -> Result<HeadResponse, HttpError> {
        let client = HEAD_CLIENT.get_or_try_init(|| build_client(HEAD_TIMEOUT, HEAD_TIMEOUT))?;
        let response = client.head(url).send().map_err(|e| HttpError::from_reqwest(&e))?;
        let headers = response
            .headers()
            .iter()
            .map(|(k, v)| (k.as_str().to_lowercase(), String::from_utf8_lossy(v.as_bytes()).into_owned()))
            .collect();
        Ok(HeadResponse { status_code: response.status().as_u16(), headers })
    }

    fn stream(&self, url: &str, headers: &Headers, timeout: StreamTimeout) -> Result<Box<dyn StreamResponse>, HttpError> {
        let client = stream_client(timeout)?;
        let mut request = client.get(url);
        for (name, value) in headers {
            request = request.header(name.as_str(), value.as_str());
        }
        let response = request.send().map_err(|e| HttpError::from_reqwest(&e))?;
        Ok(Box::new(ReqwestStream { response, url: url.to_string() }))
    }
}

/// Test seam replacing `_download_artifacts` (Python tests assign a function).
pub type DownloadArtifactsHook = Arc<dyn Fn(&ArtifactMap, &Path) -> DownloadResult<PathBuf> + Send + Sync>;

/// Test seam observing directory creation (Python tests patch `os.makedirs`).
pub type MakedirsHook = Arc<dyn Fn(&Path) + Send + Sync>;

/// Test seam replacing `_download_chunk(start, end)` inside the chunk pool
/// (Python tests monkeypatch `SLPArtifactDownloader._download_chunk`).
pub type DownloadChunkHook = Arc<dyn Fn(u64, u64) -> DownloadResult<()> + Send + Sync>;

/// An artifact source for a downloader built without a platform client: any
/// use is an error.
struct UnconfiguredSource;

impl ArtifactSource for UnconfiguredSource {
    fn get_artifacts(&self, model_name: &str) -> anyhow::Result<ArtifactMap> {
        anyhow::bail!("no artifact source configured (asked for '{model_name}')")
    }
}

impl BootstrapArtifactSource for UnconfiguredSource {
    fn list_models(&self) -> anyhow::Result<Vec<serde_json::Value>> {
        anyhow::bail!("no bootstrap source configured")
    }
}

/// Run `work` over `jobs` on up to `workers` threads, handing each result to
/// `handle` as it completes (`as_completed`).  `handle` returning `Some(err)`
/// stops consuming and returns that error; as with leaving a
/// `ThreadPoolExecutor` block by raising, the remaining work still runs to
/// completion before this returns.
fn run_pool<J, R, F, H>(workers: usize, jobs: Vec<J>, work: F, mut handle: H) -> Option<DownloadError>
where
    J: Send,
    R: Send,
    F: Fn(J) -> R + Sync,
    H: FnMut(R) -> Option<DownloadError>,
{
    if jobs.is_empty() {
        return None;
    }
    let threads = workers.max(1).min(jobs.len());
    let queue = Mutex::new(jobs.into_iter().collect::<VecDeque<J>>());
    let (tx, rx) = mpsc::channel::<R>();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let tx = tx.clone();
            let queue = &queue;
            let work = &work;
            scope.spawn(move || {
                loop {
                    let job = queue.lock().pop_front();
                    let Some(job) = job else { break };
                    // A receiver that stopped listening is not a reason to
                    // abandon queued work (the executor would have run it).
                    let _ = tx.send(work(job));
                }
            });
        }
        drop(tx);
        for result in rx.iter() {
            if let Some(err) = handle(result) {
                return Some(err);
            }
        }
        None
    })
}

/// Download model artifacts from URLs provided by an ArtifactSource.
///
/// Fetches signed URLs via artifact sources, then downloads all files in
/// parallel with MD5 verification and progress reporting.
pub struct SLPArtifactDownloader {
    workspace: Option<SharedWorkspace>,
    models_dir_override: Option<PathBuf>,
    pub logger: SharedLogger,
    /// The token a stop request will raise.  Replaceable after construction
    /// (`downloader.cancel_token = token` in Python): `/v1/init` claims the
    /// registry entry on its download thread and hands the token over there.
    cancel_token: parking_lot::RwLock<CancelToken>,
    pub settings: Option<AppSettings>,
    model_source: Arc<dyn ArtifactSource>,
    bootstrap_source: Arc<dyn BootstrapArtifactSource>,
    http: Arc<dyn DownloadHttp>,
    download_artifacts_hook: Mutex<Option<DownloadArtifactsHook>>,
    makedirs_hook: Mutex<Option<MakedirsHook>>,
    download_chunk_hook: Mutex<Option<DownloadChunkHook>>,
    chunk_workers_override: Mutex<Option<usize>>,
}

impl SLPArtifactDownloader {
    /// Build a downloader for `workspace`.  `cancel` is the token a stop
    /// request will raise.
    ///
    /// On the instance rather than threaded through eight signatures: every
    /// call site builds its own downloader (`/v1/init`, `/v1/bootstrap`, the
    /// orchestrator bootstrap, `GenericModel::init`), so there is no sharing
    /// hazard.  Defaulting to a fresh token rather than to `None` means every
    /// check point is unconditional — a downloader nobody can cancel simply
    /// holds a flag nobody sets.
    pub fn new(workspace: SharedWorkspace, logger: SharedLogger, cancel: Option<CancelToken>) -> Self {
        let settings = AppSettings::new();
        let platform_client = Arc::new(SLPPlatformClient::new(&settings.slp_api_base));
        Self {
            workspace: Some(workspace),
            models_dir_override: None,
            logger,
            cancel_token: parking_lot::RwLock::new(cancel.unwrap_or_default()),
            settings: Some(settings),
            model_source: Arc::new(SLPModelSource::new(platform_client.clone())),
            bootstrap_source: Arc::new(SLPBootstrapSource::new(platform_client)),
            http: Arc::new(ReqwestDownloadHttp),
            download_artifacts_hook: Mutex::new(None),
            makedirs_hook: Mutex::new(None),
            download_chunk_hook: Mutex::new(None),
            chunk_workers_override: Mutex::new(None),
        }
    }

    /// Construct without a workspace or network (the Python tests'
    /// `SLPArtifactDownloader.__new__` + attribute assignment).  Sources
    /// default to ones that error if used; HTTP defaults to reqwest.
    pub fn for_tests(models_dir: Option<PathBuf>, logger: SharedLogger, cancel: Option<CancelToken>) -> Self {
        Self {
            workspace: None,
            models_dir_override: models_dir,
            logger,
            cancel_token: parking_lot::RwLock::new(cancel.unwrap_or_default()),
            settings: None,
            model_source: Arc::new(UnconfiguredSource),
            bootstrap_source: Arc::new(UnconfiguredSource),
            http: Arc::new(ReqwestDownloadHttp),
            download_artifacts_hook: Mutex::new(None),
            makedirs_hook: Mutex::new(None),
            download_chunk_hook: Mutex::new(None),
            chunk_workers_override: Mutex::new(None),
        }
    }

    /// The token a stop request will raise (a handle to the same flag).
    pub fn cancel_token(&self) -> CancelToken {
        self.cancel_token.read().clone()
    }

    /// Replace the token every later check consults.
    pub fn set_cancel_token(&self, token: CancelToken) {
        *self.cancel_token.write() = token;
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

    /// The bootstrap source (`_bootstrap_source`), read by the API's
    /// orchestrator bootstrap.
    pub fn bootstrap_source(&self) -> &Arc<dyn BootstrapArtifactSource> {
        &self.bootstrap_source
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

    /// Tests only: replace each `_download_chunk` call of the chunk pool.
    pub fn set_download_chunk_hook(&self, hook: Option<DownloadChunkHook>) {
        *self.download_chunk_hook.lock() = hook;
    }

    /// Tests only: override `CHUNK_WORKERS` (`None` restores the default).
    pub fn set_chunk_workers(&self, workers: Option<usize>) {
        *self.chunk_workers_override.lock() = workers;
    }

    /// The workspace's models directory (or the test override).
    pub fn models_dir(&self) -> PathBuf {
        if let Some(dir) = &self.models_dir_override {
            return dir.clone();
        }
        self.workspace.as_ref().map(|ws| PathBuf::from(&ws.read().models_dir)).unwrap_or_default()
    }

    /// Return the hex-encoded MD5 digest of a local file.
    ///
    /// Checked per read because this runs on a finished multi-GB file: a 4 GB
    /// digest is seconds of work with no network in it, and a cancel arriving
    /// during one would otherwise wait it out with nothing to show for it.
    pub fn compute_local_md5(filepath: &Path, cancel: Option<&CancelToken>) -> DownloadResult<String> {
        let mut md5 = Md5::new();
        let mut file = File::open(filepath)?;
        let mut buf = vec![0u8; CHUNK_SIZE];
        loop {
            let n = match file.read(&mut buf) {
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            };
            if n == 0 {
                break;
            }
            if let Some(cancel) = cancel {
                cancel.check("verification")?;
            }
            md5.update(&buf[..n]);
        }
        Ok(hex::encode(md5.finalize()))
    }

    /// Extract the hex-encoded MD5 from the `ETag` response header.
    ///
    /// For non-composite GCS objects the ETag is the hex-encoded MD5 wrapped
    /// in quotes (32 hex chars).  Header lookup is case-insensitive.
    pub fn parse_remote_md5(headers: &Headers) -> Option<String> {
        let etag = header_get(headers, "ETag").unwrap_or("").trim_matches(|c| c == '"' || c == ' ');
        if etag.chars().count() == 32 && python_int_base16_ok(etag) {
            return Some(etag.to_lowercase());
        }
        None
    }

    /// Call `attempt_fn(attempt_number)` with retries on transient errors.
    ///
    /// `attempt_fn` must be resumable/idempotent: it is re-invoked from scratch
    /// on each retry and is expected to pick up where the previous attempt
    /// left off (e.g. by inspecting bytes already written).  Permission errors
    /// (401) and non-transient errors propagate immediately.
    pub fn run_with_retries<T>(
        &self,
        description: &str,
        attempt_fn: &mut dyn FnMut(u32) -> DownloadResult<T>,
    ) -> DownloadResult<T> {
        for attempt in 1..=MAX_DOWNLOAD_ATTEMPTS {
            self.cancel_token().check(description)?;
            let exc = match attempt_fn(attempt) {
                Ok(value) => return Ok(value),
                Err(exc) => exc,
            };
            if exc.is_permission_error() || exc.is_cancelled() {
                return Err(exc);
            }
            if !is_retryable_error(&exc) || attempt == MAX_DOWNLOAD_ATTEMPTS {
                return Err(exc);
            }
            let delay = backoff_delay(attempt);
            self.logger.warning(&format!(
                "[retry] {description} failed (attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS}): {exc}. Retrying in {delay:.1}s"
            ));
            // The one substitution that makes a cancel prompt during an
            // outage.  Backoff reaches 30s and the window is ~90s in total, so
            // a plain sleep here is where a stop request would spend most of
            // its time waiting on a transfer that is not even transferring.
            if self.cancel_token().wait(delay) {
                return Err(DownloadError::CancelledBy {
                    cancelled: DownloadCancelled(format!("{description} cancelled")),
                    cause: Box::new(exc),
                });
            }
        }
        // Unreachable: the loop either returns or raises on the final attempt.
        Err(DownloadError::runtime(format!("{description} exhausted retries"), None))
    }

    /// HEAD request to get file size, MD5 hash, and Range support.
    ///
    /// Retries transient failures; falls back to `(0, None, false)` when the
    /// probe ultimately fails (callers then download single-stream,
    /// unverified) — but propagates a cancel.
    pub fn get_remote_file_info(&self, url: &str) -> DownloadResult<(u64, Option<String>, bool)> {
        let mut attempt = |_attempt_no: u32| -> DownloadResult<(u64, Option<String>, bool)> {
            let resp = self.http.head(url)?;
            let size = match header_get(&resp.headers, "Content-Length") {
                Some(cl) if !cl.is_empty() => cl.trim().parse::<u64>().map_err(|_| {
                    DownloadError::Value(format!("invalid literal for int() with base 10: '{cl}'"))
                })?,
                _ => 0,
            };
            let md5 = Self::parse_remote_md5(&resp.headers);
            let accept_ranges = header_get(&resp.headers, "Accept-Ranges").unwrap_or("").to_lowercase() == "bytes";
            Ok((size, md5, accept_ranges))
        };
        match self.run_with_retries("HEAD probe", &mut attempt) {
            Ok(info) => Ok(info),
            // Above the blanket fallback below, which is the whole reason this
            // arm exists: swallowed here, a cancel would be reported as "no
            // size, no digest, no Range support" and the file would go on to
            // download single-stream and unverified.
            Err(exc) if exc.is_cancelled() => Err(exc),
            Err(_) => Ok((0, None, false)),
        }
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
        struct Pending {
            filename: String,
            url: String,
            size: u64,
            supports_range: bool,
            remote_md5: Option<String>,
        }
        let mut pending: Vec<Pending> = Vec::new();
        let mut skipped = 0usize;

        for (filename, url) in artifacts {
            self.cancel_token().check("download")?;
            let flat_name = basename(filename);
            let local_path = local_dir.join(flat_name);
            let (size, remote_md5, accept_ranges) = self.get_remote_file_info(url)?;

            if local_path.is_file() {
                if let Some(remote) = &remote_md5 {
                    let local_md5 = Self::compute_local_md5(&local_path, None)?;
                    if &local_md5 == remote {
                        skipped += 1;
                        continue;
                    }
                } else if size > 0 && std::fs::metadata(&local_path)?.len() == size {
                    skipped += 1;
                    continue;
                }
            }

            pending.push(Pending {
                filename: filename.clone(),
                url: url.clone(),
                size,
                supports_range: accept_ranges,
                remote_md5,
            });
        }

        if skipped > 0 {
            self.logger.info(&format!("[ok] {skipped} file(s) already up-to-date, skipped"));
        }

        if pending.is_empty() {
            self.logger.info(&format!("[ok] All files already downloaded at: {}", log_dir.display()));
            return Ok(local_dir.to_path_buf());
        }

        let total_bytes: u64 = pending.iter().map(|p| p.size).filter(|s| *s > 0).sum();
        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        self.logger.info(&format!("Downloading {} file(s), {total_mb:.1} MB total", pending.len()));

        // Roll the parallel per-file progress up into a single 0->100% bar so
        // the client sees one download, not the files fighting over the meter.
        let aggregate = Arc::new(AggregateProgress::new(total_bytes, progress_callback, label));
        let aggregate_callback = aggregate.as_callback();

        let mut failed: Vec<(String, DownloadError)> = Vec::new();
        let jobs: Vec<(usize, Pending)> = pending.into_iter().enumerate().collect();
        let raised = run_pool(
            max_workers(),
            jobs,
            |(position, p)| {
                let result = self.download_file(
                    &p.filename,
                    &p.url,
                    local_dir,
                    p.size,
                    position,
                    Some(aggregate_callback.clone()),
                    p.supports_range,
                    p.remote_md5.as_deref(),
                );
                (p.filename, result)
            },
            |(filename, result)| match result {
                Ok(_) => None,
                // A local filesystem refusal (unwritable models dir, a sandbox
                // denial) — nothing about it improves by letting the other
                // files finish first.
                Err(e) if e.is_permission_error() => Some(e),
                // Beside the refusal above rather than below with the
                // failures: collected into `failed`, a cancel would be
                // collapsed by `aggregate_error` into "failed to download 2
                // file(s)" and the stream would answer with an error frame for
                // something the user asked for.
                Err(e) if e.is_cancelled() => Some(e),
                Err(e) => {
                    self.logger.error(&format!("[error] {filename}: {e}"));
                    failed.push((filename, e));
                    None
                }
            },
        );
        if let Some(err) = raised {
            return Err(err);
        }

        if !failed.is_empty() {
            return Err(aggregate_error(failed));
        }

        // Before the terminal 100% event, not after: a cancel that lands while
        // the last worker is unwinding would otherwise tell the client the
        // download completed, and the stream would go on to mark the model
        // downloaded.
        self.cancel_token().check("download")?;
        aggregate.finish();
        self.logger.info(&format!("[ok] {log_prefix} downloaded to: {}", log_dir.display()));
        Ok(local_dir.to_path_buf())
    }

    /// Download a single file, then verify and atomically publish it.
    ///
    /// Bytes are written to a `.part` sidecar so the final path only ever
    /// appears once the download is complete (and MD5-verified, when the
    /// server provides a digest).  A corrupt download — bytes that finished
    /// without a network error but don't match the expected MD5 — is
    /// re-downloaded from scratch up to [`MAX_VERIFY_ATTEMPTS`] times before
    /// failing with `"<name>: MD5 verification failed after 3 attempts"`.
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
        let _ = position;
        let flat_name = basename(filename);

        // On entry, so a worker the pool starts after the flag was raised
        // exits before opening a socket.
        self.cancel_token().check(flat_name)?;

        let dest = local_dir.join(flat_name);
        let part = PathBuf::from(format!("{}.part", dest.display()));
        let size_mb = size as f64 / (1024.0 * 1024.0);

        for verify_attempt in 1..=MAX_VERIFY_ATTEMPTS {
            self.cancel_token().check(flat_name)?;
            // Start each verification attempt from a clean slate so we never
            // resume onto bytes a previous corrupt attempt wrote.
            if part.exists() {
                std::fs::remove_file(&part)?;
            }

            if supports_range && size >= CHUNKED_THRESHOLD {
                let workers = self.effective_chunk_workers();
                let chunk_size_mb = size as f64 / workers as f64 / (1024.0 * 1024.0);
                self.logger.info(&format!(
                    "Downloading {flat_name} ({size_mb:.1} MB) using {workers} parallel chunks of {chunk_size_mb:.1} MB each"
                ));
                self.download_file_chunked(flat_name, url, &part, size, progress_callback.clone())?;
            } else {
                let reason = if !supports_range { "no Range support" } else { "below threshold" };
                self.logger.info(&format!("Downloading {flat_name} ({size_mb:.1} MB) single-stream ({reason})"));
                self.download_file_simple(flat_name, url, &part, size, progress_callback.clone(), supports_range)?;
            }

            // No server-provided digest → can't verify; the atomic rename
            // below still guarantees we never publish a truncated file.
            let Some(expected) = remote_md5 else {
                break;
            };

            let local_md5 = Self::compute_local_md5(&part, Some(&self.cancel_token()))?;
            if local_md5 == expected {
                break;
            }

            self.logger.warning(&format!(
                "[verify] {flat_name} MD5 mismatch (attempt {verify_attempt}/{MAX_VERIFY_ATTEMPTS}): \
                 expected {expected}, got {local_md5}"
            ));
            if verify_attempt == MAX_VERIFY_ATTEMPTS {
                if part.exists() {
                    std::fs::remove_file(&part)?;
                }
                return Err(DownloadError::runtime(
                    format!("{flat_name}: MD5 verification failed after {MAX_VERIFY_ATTEMPTS} attempts"),
                    None,
                ));
            }
        }

        // Atomically publish: the final path appears only now, fully written.
        std::fs::rename(&part, &dest)?;

        if let Some(callback) = &progress_callback {
            callback(DownloadProgress::new(flat_name, size, size, "completed"));
        }

        Ok(filename.to_string())
    }

    /// Download a file in a single request, retrying transient failures.
    ///
    /// When the server supports Range requests, a retry resumes from the bytes
    /// already written rather than restarting from zero.  Otherwise each retry
    /// re-downloads the whole file.
    pub fn download_file_simple(
        &self,
        flat_name: &str,
        url: &str,
        dest: &Path,
        size: u64,
        progress_callback: Option<ProgressCallback>,
        supports_range: bool,
    ) -> DownloadResult<()> {
        let mut attempt = |attempt_no: u32| -> DownloadResult<()> {
            // Resume only from bytes written by a PRIOR attempt in this call —
            // the first attempt always starts clean, so a pre-existing corrupt
            // file (which is why it's being re-downloaded) is overwritten, not
            // trusted.
            let mut resume_from = 0u64;
            if attempt_no > 1 && supports_range && dest.exists() {
                resume_from = std::fs::metadata(dest)?.len();
                if size > 0 && resume_from >= size {
                    return Ok(()); // this attempt's predecessor already finished the file
                }
            }
            let mut headers = Headers::new();
            if resume_from > 0 {
                headers.insert("Range".to_string(), format!("bytes={resume_from}-"));
            }
            let mut resp = self.http.stream(url, &headers, STREAM_TIMEOUT)?;
            if resp.status_code() == 401 {
                return Err(StaleDownloadURLError(UNAUTHORIZED_MSG.to_string()).into());
            }
            // If we asked to resume but the server ignored it (sent 200, not
            // 206), it's streaming the whole file — rewrite from the start.
            if resume_from > 0 && resp.status_code() != 206 {
                resume_from = 0;
            }
            resp.raise_for_status()?;
            let mut downloaded = resume_from;
            let mut file = if resume_from > 0 {
                OpenOptions::new().append(true).create(true).open(dest)?
            } else {
                File::create(dest)?
            };
            while let Some(chunk) = resp.next_chunk(CHUNK_SIZE)? {
                // Returning from inside the stream is what closes the socket:
                // dropping `resp` drops the connection rather than leaving it
                // to the 60s read timeout.
                self.cancel_token().check(flat_name)?;
                file.write_all(&chunk)?;
                downloaded += chunk.len() as u64;
                if let Some(callback) = &progress_callback {
                    callback(DownloadProgress::new(flat_name, downloaded, size, "downloading"));
                }
            }
            Ok(())
        };

        self.run_with_retries(&format!("download {flat_name}"), &mut attempt)
    }

    /// Download a byte range to its file offset, retrying transient failures.
    ///
    /// A retry resumes from within the range: the file is pre-allocated (so
    /// its on-disk size can't tell us how far we got), so we track the bytes
    /// written for this chunk locally and resume from `start + written`.  The
    /// shared progress counter only ever advances by newly written bytes, so a
    /// retry never double-counts.
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
        let mut written = 0u64; // bytes of THIS chunk already flushed to disk

        // On entry, so a range the pool starts after the flag was raised exits
        // before opening a socket.  Every model worth downloading is over
        // CHUNKED_THRESHOLD, so this pool is where a cancel usually lands.
        self.cancel_token().check(flat_name)?;

        let mut attempt = |_attempt_no: u32| -> DownloadResult<()> {
            let cur_start = start + written;
            if cur_start > end {
                return Ok(()); // chunk already fully written by a prior attempt
            }
            let mut headers = Headers::new();
            headers.insert("Range".to_string(), format!("bytes={cur_start}-{end}"));
            let mut resp = self.http.stream(url, &headers, STREAM_TIMEOUT)?;
            if resp.status_code() == 401 {
                return Err(StaleDownloadURLError(UNAUTHORIZED_MSG.to_string()).into());
            }
            resp.raise_for_status()?;
            let mut file = OpenOptions::new().read(true).write(true).open(dest)?;
            file.seek(SeekFrom::Start(cur_start))?;
            while let Some(chunk) = resp.next_chunk(RANGE_READ_SIZE)? {
                self.cancel_token().check(flat_name)?;
                file.write_all(&chunk)?;
                written += chunk.len() as u64;
                if let Some(callback) = &progress_callback {
                    let current = {
                        let mut counter = downloaded_counter.lock();
                        *counter += chunk.len() as u64;
                        *counter
                    };
                    callback(DownloadProgress::new(flat_name, current, total_size, "downloading"));
                }
            }
            Ok(())
        };

        self.run_with_retries(&format!("chunk {start}-{end} of {flat_name}"), &mut attempt)
    }

    /// `CHUNK_WORKERS`, or the test override.
    fn effective_chunk_workers(&self) -> usize {
        self.chunk_workers_override.lock().unwrap_or_else(chunk_workers).max(1)
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
        // Pre-allocate the file.
        {
            let mut file = File::create(dest)?;
            if size == 0 {
                return Err(std::io::Error::from(std::io::ErrorKind::InvalidInput).into());
            }
            file.seek(SeekFrom::Start(size - 1))?;
            file.write_all(&[0])?;
        }

        // Split the file into exactly CHUNK_WORKERS equal segments.
        let workers = self.effective_chunk_workers() as u64;
        let chunk_size = size.div_ceil(workers);
        let mut chunks: Vec<(u64, u64)> = Vec::new();
        let mut offset = 0u64;
        while offset < size {
            let end = (offset + chunk_size - 1).min(size - 1);
            chunks.push((offset, end));
            offset = end + 1;
        }

        let downloaded_counter = Mutex::new(0u64); // shared across threads
        let hook = self.download_chunk_hook.lock().clone();

        let raised = run_pool(
            workers as usize,
            chunks,
            |(start, end)| {
                let result = match &hook {
                    Some(hook) => hook(start, end),
                    None => self.download_chunk(
                        url,
                        dest,
                        start,
                        end,
                        flat_name,
                        size,
                        &downloaded_counter,
                        progress_callback.clone(),
                    ),
                };
                ((start, end), result)
            },
            |((start, end), result)| match result {
                Ok(()) => None,
                // Above the wrapper below, which would turn it into a
                // connection or runtime error and lose the one thing the
                // stream needs to know about it.
                Err(e) if e.is_cancelled() => Some(e),
                // Keep the cause's kind: every model worth downloading is over
                // CHUNKED_THRESHOLD, so this is the path a real dropped
                // connection takes, and flattening it to a runtime error here
                // would undo the classification the outer layers preserve.
                Err(e) => {
                    let transient = is_transient_error(&e);
                    Some(
                        classified(
                            format!("Chunk download failed for {flat_name} (bytes {start}-{end}): {e}"),
                            transient,
                        )
                        .with_cause(e),
                    )
                }
            },
        );
        match raised {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// Run the `_download_artifacts` test hook when installed, else the real
    /// [`Self::download_artifacts`].
    fn run_download_artifacts(
        &self,
        artifacts: &ArtifactMap,
        local_dir: &Path,
        progress_callback: Option<ProgressCallback>,
        log_prefix: &str,
        label: &str,
    ) -> DownloadResult<PathBuf> {
        let hook = self.download_artifacts_hook.lock().clone();
        if let Some(hook) = hook {
            return hook(artifacts, local_dir);
        }
        self.download_artifacts(artifacts, local_dir, progress_callback, log_prefix, local_dir, label)
    }

    /// `os.makedirs(local_dir, exist_ok=True)`, observed by the test hook.
    fn makedirs(&self, local_dir: &Path) -> DownloadResult<()> {
        let hook = self.makedirs_hook.lock().clone();
        if let Some(hook) = hook {
            hook(local_dir);
        }
        std::fs::create_dir_all(local_dir)?;
        Ok(())
    }

    /// Download GGUF model via the unauthenticated `/v1/bootstrap/models`
    /// endpoint.
    ///
    /// Intended for first-run scenarios (Electron installer, brew install)
    /// where the user has no developer token yet.  Downloads only the GGUF
    /// file so the model is immediately ready for inference without building.
    ///
    /// If `model_name` is provided, only that model's artifacts are
    /// downloaded.  Otherwise downloads the first model returned by the
    /// endpoint.
    pub fn bootstrap_download(
        &self,
        model_name: Option<&str>,
        progress_callback: Option<ProgressCallback>,
    ) -> DownloadResult<PathBuf> {
        if let Some(stub) = download_stubs().and_then(|s| s.bootstrap_download) {
            return stub(DownloadStubCall::BootstrapDownload {
                model_name: model_name.map(str::to_string),
                models_dir: None,
            })
            .map_err(DownloadError::from_source);
        }

        let models = self.bootstrap_source.list_models().map_err(DownloadError::from_source)?;
        let target = match model_name.filter(|name| !name.is_empty()) {
            Some(name) => {
                // The bootstrap endpoint publishes upstream names, which can
                // lag a local rename — match on the published name for the
                // request.
                let wanted = platform_model_name(name);
                match models.iter().find(|m| entry_name(m) == Some(wanted.as_str())) {
                    Some(target) => target,
                    None => {
                        return Err(DownloadError::Value(format!(
                            "Model '{}' not found in bootstrap. Available: {}",
                            display_model_name(name),
                            available_names(&models)
                        )));
                    }
                }
            }
            None => models.first().ok_or_else(|| DownloadError::Value("list index out of range".to_string()))?,
        };

        let published_name = entry_name(target).unwrap_or("unknown").to_string();
        let resolved_name = canonical_model_name(&published_name);
        let model_id = Self::resolve_model_id(&resolved_name);
        let artifacts = self.bootstrap_source.get_artifacts(&published_name).map_err(DownloadError::from_source)?;

        let artifacts = select_gguf_artifacts(&artifacts);

        if artifacts.is_empty() {
            return Err(DownloadError::Value(format!(
                "No GGUF artifact found for bootstrap model '{resolved_name}'"
            )));
        }

        let local_dir = self.models_dir().join(&model_id);

        // Marked from the makedirs, not from download_artifacts: the directory
        // exists — and is ours — before the first byte lands, and a disk sweep
        // landing in that gap would delete it out from under us.
        let _mark = downloading(&local_dir);
        self.makedirs(&local_dir)?;

        self.logger.info(&format!("Bootstrap downloading model: {}", display_model_name(&resolved_name)));

        self.run_download_artifacts(
            &artifacts,
            &local_dir,
            progress_callback,
            "Bootstrap model",
            &display_model_name(&resolved_name),
        )
    }

    /// Lookup the HuggingFace model_id from a short model name.
    ///
    /// E.g. `"sl-mini"` → `"smartloop-ai/slp-orchestra-mini"`.  Legacy short
    /// names resolve too (`SUPPORTED_MODELS` canonicalizes its keys).  Falls
    /// back to `model_name` if no mapping is found.
    pub fn resolve_model_id(model_name: &str) -> String {
        match SUPPORTED_MODELS.get(model_name) {
            Some(model_type) => (model_type.model_id)(),
            None => model_name.to_string(),
        }
    }

    /// Reverse-lookup the canonical short model name from a HuggingFace
    /// model_id.
    ///
    /// E.g. `"smartloop-ai/slp-orchestra-mini"` → `"sl-mini"`.  Returns the
    /// canonical name, NOT the name the platform publishes — callers hitting
    /// the API must pass it through [`platform_model_name`] first.  Falls back
    /// to `model_id` if no mapping is found.
    pub fn resolve_model_name(model_id: &str) -> String {
        for (name, model_type) in SUPPORTED_MODELS.items() {
            if (model_type.model_id)() == model_id {
                return name.to_string();
            }
        }
        model_id.to_string()
    }

    /// Ask the platform for this model's signed GGUF URLs.
    ///
    /// This is the step that decides entitlement, and it is separated from
    /// [`Self::download_with_artifacts`] so a caller can settle "may this
    /// account have these weights?" *before* committing to a long-running
    /// transfer — the API router runs it ahead of opening its progress stream,
    /// so a refusal is an HTTP status rather than an error buried in a stream
    /// that already returned 200.
    ///
    /// The URLs are minted per call and the route is rate-limited, so the
    /// result is passed on to the download rather than being requested twice.
    /// Fails with the platform's `ModelAccessError` / `PermissionError` when
    /// the account is not entitled and `ConnectionError` for transient
    /// failures (inside [`DownloadError::Source`]).
    pub fn resolve_artifacts(&self, model_id: &str) -> DownloadResult<ArtifactMap> {
        if let Some(stub) = download_stubs().and_then(|s| s.resolve_artifacts) {
            return stub(model_id).map_err(DownloadError::from_source);
        }
        let model_name = Self::resolve_model_name(model_id);
        // The catalog is keyed by the name the platform publishes, which can
        // lag a local rename (sl-mini is still served as slp-orchestra-mini).
        let artifacts = self
            .model_source
            .get_artifacts(&platform_model_name(&model_name))
            .map_err(DownloadError::from_source)?;
        Ok(select_gguf_artifacts(&artifacts))
    }

    /// Download GGUF model via the `/v1/models/{model_name}/download` endpoint.
    ///
    /// Accepts the HuggingFace-style `model_id` (e.g. `"google/gemma-3-1b-it"`)
    /// to match the base-class interface, resolves it to the short model name
    /// for the API call, and stores files under the model_id path.
    ///
    /// `artifacts` accepts an already-resolved map from
    /// [`Self::resolve_artifacts`]; callers that have one pass it so the signed
    /// URLs are minted once.  Omitting it resolves them here, which is what the
    /// CLI and TUI do.
    ///
    /// No credential check runs here: free models are downloadable
    /// anonymously, so gating on a token locally would deny a download the
    /// platform would have served.  The platform is the only authority on
    /// access — it answers 403 (`ModelAccessError`) for a model the caller's
    /// plan doesn't grant.
    pub fn download_with_artifacts(
        &self,
        model_id: &str,
        progress_callback: Option<ProgressCallback>,
        artifacts: Option<ArtifactMap>,
    ) -> DownloadResult<PathBuf> {
        if let Some(stub) = download_stubs().and_then(|s| s.download) {
            return stub(DownloadStubCall::Download { model_id: model_id.to_string(), models_dir: None })
                .map_err(DownloadError::from_source);
        }

        let model_name = Self::resolve_model_name(model_id);

        self.logger.info(&format!("Downloading model: {}", display_model_name(&model_name)));
        println!();

        let artifacts = match artifacts {
            Some(artifacts) => artifacts,
            None => self.resolve_artifacts(model_id)?,
        };

        // After the entitlement call, not before: a refused download should
        // not leave an empty model directory behind for the workspace to trip
        // over.
        let local_dir = self.models_dir().join(model_id);

        // See bootstrap_download: the mark has to cover the makedirs, not just
        // the transfer, or a sweep can delete the directory between the two.
        let _mark = downloading(&local_dir);
        self.makedirs(&local_dir)?;

        if artifacts.is_empty() {
            self.logger.info("[ok] No GGUF files to download");
            return Ok(local_dir);
        }

        self.run_download_artifacts(&artifacts, &local_dir, progress_callback, "Model", &display_model_name(&model_name))
    }
}

/// Whether Python's `int(s, 16)` accepts `s`: optional surrounding
/// whitespace, an optional sign, an optional `0x` prefix, and hex digits with
/// single underscores between them (one may follow the prefix).
fn python_int_base16_ok(s: &str) -> bool {
    let s = s.trim();
    let s = s.strip_prefix(['+', '-']).unwrap_or(s);
    let (s, prefixed) = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(rest) => (rest, true),
        None => (s, false),
    };
    let s = if prefixed { s.strip_prefix('_').unwrap_or(s) } else { s };
    if s.is_empty() || s.starts_with('_') || s.ends_with('_') || s.contains("__") {
        return false;
    }
    s.chars().all(|c| c == '_' || c.is_ascii_hexdigit())
}

impl SLPDownloader for SLPArtifactDownloader {
    fn download(&self, model_id: &str, progress_callback: Option<ProgressCallback>) -> anyhow::Result<PathBuf> {
        self.download_with_artifacts(model_id, progress_callback, None).map_err(DownloadError::into_anyhow)
    }
}
