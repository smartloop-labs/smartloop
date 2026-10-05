//! Port of `smartloop/model/downloader/cancellation.py`.
//!
//! Asking a download in progress to stop, and finding the one to ask.
//!
//! A transfer runs on worker threads, and that work cannot be cancelled once a
//! worker has picked it up.  So stopping is cooperative: the byte loops consult
//! a flag, and this module is where the flag lives and how a second request
//! finds it.
//!
//! Two registries cover a download, answering different questions.
//! [`crate::model::downloader::inflight`] is keyed by directory and asks "is it
//! safe to delete this path", which needs containment in both directions and
//! must tolerate two transfers into one tree.  This one is keyed by `model_id`
//! and asks "which download did the user just click stop on", which needs
//! identity.  Neither collapses into the other.
//!
//! Process-local, for the same reason `inflight` is: a token is meaningful
//! exactly as long as the thread it can interrupt, and a row that outlived the
//! process would offer a cancel button for a download that is not running.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use once_cell::sync::Lazy;
use parking_lot::{Condvar, Mutex};

/// The caller asked this download to stop.  Not a failure.
///
/// Deliberately its own error type rather than a connection / IO / runtime
/// error: a cancel must not be classified "try again" (502), must not fail the
/// batch as an unwritable disk would, and must not read as 503 "busy".
/// `status_code` is 499, nginx's "client closed request".
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct DownloadCancelled(pub String);

impl DownloadCancelled {
    /// HTTP status the API layer answers for a cancelled download.
    pub const STATUS_CODE: u16 = 499;

    /// Build a cancellation error with the given message.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// HTTP status for this error (`499`).
    pub fn status_code(&self) -> u16 {
        Self::STATUS_CODE
    }
}

/// This model is already downloading, so a second transfer was refused.
///
/// A `RuntimeError` subclass in Python; maps to HTTP 409.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct DownloadInProgress(pub String);

impl DownloadInProgress {
    /// HTTP status the API layer answers for a refused concurrent download.
    pub const STATUS_CODE: u16 = 409;

    /// Build the error with the given message.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// HTTP status for this error (`409`).
    pub fn status_code(&self) -> u16 {
        Self::STATUS_CODE
    }
}

/// Port of `threading.Event`: a shareable boolean flag that threads can wait on.
///
/// Cloning yields a handle to the *same* event.
#[derive(Debug, Clone, Default)]
pub struct Event {
    inner: Arc<(Mutex<bool>, Condvar)>,
}

impl Event {
    /// A new, unset event.
    pub fn new() -> Self {
        todo!()
    }

    /// Set the flag and wake every waiter.
    pub fn set(&self) {
        todo!()
    }

    /// Reset the flag to false.
    pub fn clear(&self) {
        todo!()
    }

    /// Whether the flag is set.
    pub fn is_set(&self) -> bool {
        todo!()
    }

    /// Block until the flag is set or `timeout` seconds pass; returns the flag.
    pub fn wait(&self, timeout: f64) -> bool {
        let _ = timeout;
        todo!()
    }

    /// Block until the flag is set or `timeout` passes; returns the flag.
    pub fn wait_for(&self, timeout: Duration) -> bool {
        let _ = timeout;
        todo!()
    }

    /// True when both handles refer to the same underlying event.
    pub fn ptr_eq(&self, other: &Event) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

/// One download's stop flag, plus the two ways it gets consulted.
///
/// [`CancelToken::check`] keeps every call site in the byte loops one line, and
/// [`CancelToken::wait`] is a drop-in for a sleep, so the downloader's retry
/// backoff is interruptible.  Cloning yields a handle to the same token;
/// identity is compared with [`CancelToken::ptr_eq`] (or `==`).
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    event: Event,
    /// Set once the transfer's threads have stopped *and* its bytes have been
    /// discarded — not merely when the flag was raised.
    pub done: Event,
    /// Test seam (the Python tests monkeypatch `CancelToken.wait`): when set,
    /// [`CancelToken::wait`] returns the cancelled state immediately instead of
    /// sleeping.
    instant_wait: Arc<AtomicBool>,
}

impl CancelToken {
    /// A fresh, uncancelled token.
    pub fn new() -> Self {
        todo!()
    }

    /// Ask this download to stop.  Idempotent.
    pub fn cancel(&self) {
        todo!()
    }

    /// Whether a stop has been requested.
    pub fn cancelled(&self) -> bool {
        todo!()
    }

    /// Return `Err(DownloadCancelled("<what> cancelled"))` if a stop has been
    /// requested.  Python's default `what` is `"download"`.
    pub fn check(&self, what: &str) -> Result<(), DownloadCancelled> {
        let _ = what;
        todo!()
    }

    /// Sleep for `timeout` seconds, returning early and `true` if cancelled.
    pub fn wait(&self, timeout: f64) -> bool {
        let _ = timeout;
        todo!()
    }

    /// Tests only: make [`CancelToken::wait`] return immediately with the
    /// token's cancelled state (mirrors the Python `_no_sleep` fixture).
    pub fn set_instant_wait(&self, instant: bool) {
        let _ = instant;
        todo!()
    }

    /// True when both handles refer to the same token.
    pub fn ptr_eq(&self, other: &CancelToken) -> bool {
        self.event.ptr_eq(&other.event)
    }
}

impl PartialEq for CancelToken {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other)
    }
}

impl Eq for CancelToken {}

/// `model_id` -> the token for the transfer writing that model's directory.
static ACTIVE: Lazy<Mutex<HashMap<String, CancelToken>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// How often [`begin`] re-checks for a departing entry, in seconds.
pub const POLL_INTERVAL: f64 = 0.05;

/// Default grace [`begin`] waits for an existing entry to leave, in seconds.
pub const DEFAULT_BEGIN_WAIT: f64 = 10.0;

/// Claim `model_id` for a new download, or fail with [`DownloadInProgress`].
///
/// Waits up to [`DEFAULT_BEGIN_WAIT`] seconds for an existing entry to leave
/// before refusing (see [`begin_with_wait`]).
pub fn begin(model_id: &str) -> Result<CancelToken, DownloadInProgress> {
    begin_with_wait(model_id, DEFAULT_BEGIN_WAIT)
}

/// Claim `model_id`, waiting up to `wait` seconds for an existing entry to
/// leave before refusing with `DownloadInProgress("Already downloading: <id>")`.
///
/// That grace is what makes cancel-then-immediately-re-enable work: the entry
/// is held until the cancelled transfer's bytes are gone (see [`end`]).
pub fn begin_with_wait(model_id: &str, wait: f64) -> Result<CancelToken, DownloadInProgress> {
    let _ = (model_id, wait, &*ACTIVE);
    todo!()
}

/// The token for `model_id`'s running download, if there is one.
pub fn find(model_id: &str) -> Option<CancelToken> {
    let _ = model_id;
    todo!()
}

/// Signal `model_id`'s download to stop, returning its token.
///
/// `None` means nothing was running — a plain outcome, not an error, because
/// the client races by construction.
pub fn request_cancel(model_id: &str) -> Option<CancelToken> {
    let _ = model_id;
    todo!()
}

/// Release `model_id`, but only if `token` is still the one registered, then
/// set `token.done`.
///
/// The identity check stops a slow wind-down from unregistering a newer
/// download.  Callers release *after* the bytes are discarded.
pub fn end(model_id: &str, token: &CancelToken) {
    let _ = (model_id, token);
    todo!()
}

/// The `model_id`s with a download registered (sorted), for status and tests.
pub fn active_keys() -> Vec<String> {
    todo!()
}

/// Drop every registration.  Tests only.
pub fn reset_for_tests() {
    todo!()
}
