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
//! That holds today because `/v1/init` and `DELETE /v1/downloads/…` are both
//! workspace-level, so `supervisor::routing::project_to_proxy` leaves them with
//! the process that owns the workspace.  A route that grew a `?project_id=`
//! would be proxied to a child and would signal an empty registry.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use parking_lot::{Condvar, Mutex};

/// The caller asked this download to stop.  Not a failure.
///
/// Deliberately its own error type rather than a connection / IO / runtime
/// error, and that is load-bearing in three directions at once:
///
/// * not a connection error — `artifact_downloader::is_transient_error` would
///   classify a cancel "try again" and the router would answer 502;
/// * not an IO / permission error — `download_artifacts` re-raises those
///   immediately to fail the whole batch fast, which is right for an
///   unwritable disk and wrong as a description of a cancel;
/// * not a runtime error — `storage` and the download router map that to 503
///   "busy, try later".
///
/// `status_code` is 499, nginx's "client closed request": no standard code says
/// "you asked me to stop and I did", and 499 is the one clients already read
/// as a deliberate abort rather than a fault.
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
/// Nothing serialized downloads on `model_id` before this: two calls wrote the
/// same directory and each deleted the other's `.part` on its way in
/// (`download_file` unlinks a pre-existing sidecar rather than resuming onto
/// it), so both made progress and neither finished.
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

/// Convert a Python-style seconds value into a `Duration` (negative / NaN →
/// zero, overflow / infinity → `Duration::MAX`).
fn seconds(timeout: f64) -> Duration {
    if timeout.is_nan() || timeout <= 0.0 {
        Duration::ZERO
    } else {
        Duration::try_from_secs_f64(timeout).unwrap_or(Duration::MAX)
    }
}

impl Event {
    /// A new, unset event.
    pub fn new() -> Self {
        Self { inner: Arc::new((Mutex::new(false), Condvar::new())) }
    }

    /// Set the flag and wake every waiter.
    pub fn set(&self) {
        let (lock, cond) = &*self.inner;
        *lock.lock() = true;
        cond.notify_all();
    }

    /// Reset the flag to false.
    pub fn clear(&self) {
        *self.inner.0.lock() = false;
    }

    /// Whether the flag is set.
    pub fn is_set(&self) -> bool {
        *self.inner.0.lock()
    }

    /// Block until the flag is set or `timeout` seconds pass; returns the flag.
    pub fn wait(&self, timeout: f64) -> bool {
        self.wait_for(seconds(timeout))
    }

    /// Block until the flag is set or `timeout` passes; returns the flag.
    pub fn wait_for(&self, timeout: Duration) -> bool {
        let (lock, cond) = &*self.inner;
        let mut flag = lock.lock();
        match Instant::now().checked_add(timeout) {
            Some(deadline) => {
                while !*flag {
                    if cond.wait_until(&mut flag, deadline).timed_out() {
                        break;
                    }
                }
            }
            None => {
                while !*flag {
                    cond.wait(&mut flag);
                }
            }
        }
        *flag
    }

    /// True when both handles refer to the same underlying event.
    pub fn ptr_eq(&self, other: &Event) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

/// One download's stop flag, plus the two ways it gets consulted.
///
/// A wrapper rather than a bare [`Event`] for two reasons.
/// [`CancelToken::check`] keeps every call site in the byte loops one line,
/// which is what makes it affordable to put one in each of them.  And
/// [`CancelToken::wait`] is a drop-in for a sleep, so the downloader's retry
/// backoff becomes interruptible by substitution rather than by restructuring —
/// without it a cancel arriving early in a 30s backoff waits the whole backoff
/// out.  Cloning yields a handle to the same token; identity is compared with
/// [`CancelToken::ptr_eq`] (or `==`).
#[derive(Debug, Clone)]
pub struct CancelToken {
    event: Event,
    /// Set once the transfer's threads have stopped *and* its bytes have been
    /// discarded — not merely when the flag was raised.  The route offers a
    /// bounded wait on this so a caller can ask for "stopped" in the strong
    /// sense; tests use it to avoid sleeping on a guess.
    pub done: Event,
    /// Test seam (the Python tests monkeypatch `CancelToken.wait`): when set,
    /// [`CancelToken::wait`] returns the cancelled state immediately instead of
    /// sleeping.
    instant_wait: Arc<AtomicBool>,
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancelToken {
    /// A fresh, uncancelled token.
    pub fn new() -> Self {
        Self { event: Event::new(), done: Event::new(), instant_wait: Arc::new(AtomicBool::new(false)) }
    }

    /// Ask this download to stop.  Idempotent.
    pub fn cancel(&self) {
        self.event.set();
    }

    /// Whether a stop has been requested.
    pub fn cancelled(&self) -> bool {
        self.event.is_set()
    }

    /// Return `Err(DownloadCancelled("<what> cancelled"))` if a stop has been
    /// requested.  Python's default `what` is `"download"`.
    pub fn check(&self, what: &str) -> Result<(), DownloadCancelled> {
        if self.event.is_set() {
            return Err(DownloadCancelled(format!("{what} cancelled")));
        }
        Ok(())
    }

    /// Sleep for `timeout` seconds, returning early and `true` if cancelled.
    pub fn wait(&self, timeout: f64) -> bool {
        if self.instant_wait.load(Ordering::SeqCst) {
            return self.cancelled();
        }
        self.event.wait(timeout)
    }

    /// Tests only: make [`CancelToken::wait`] return immediately with the
    /// token's cancelled state (mirrors the Python `_no_sleep` fixture).
    pub fn set_instant_wait(&self, instant: bool) {
        self.instant_wait.store(instant, Ordering::SeqCst);
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

/// How often [`begin`] re-checks for a departing entry, in seconds.  Short
/// enough that a hand-off costs nothing perceptible, long enough not to spin.
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
/// is held until the cancelled transfer's bytes are gone (see [`end`]), and a
/// user who clicks stop and then start again is well inside it.  Without the
/// wait, the second click would be answered "already downloading" for a
/// download that is in the act of deleting itself.
pub fn begin_with_wait(model_id: &str, wait: f64) -> Result<CancelToken, DownloadInProgress> {
    let deadline = Instant::now().checked_add(seconds(wait));
    loop {
        {
            let mut active = ACTIVE.lock();
            if !active.contains_key(model_id) {
                let token = CancelToken::new();
                active.insert(model_id.to_string(), token.clone());
                return Ok(token);
            }
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(DownloadInProgress(format!("Already downloading: {model_id}")));
        }
        std::thread::sleep(seconds(POLL_INTERVAL));
    }
}

/// The token for `model_id`'s running download, if there is one.
pub fn find(model_id: &str) -> Option<CancelToken> {
    ACTIVE.lock().get(model_id).cloned()
}

/// Signal `model_id`'s download to stop, returning its token.
///
/// `None` means nothing was running — which the route reports as a plain
/// outcome rather than an error, because the client races by construction: a
/// progress frame arrives, the user clicks, and the transfer finishes before
/// the request lands.
pub fn request_cancel(model_id: &str) -> Option<CancelToken> {
    let token = find(model_id);
    if let Some(token) = &token {
        token.cancel();
    }
    token
}

/// Release `model_id`, but only if `token` is still the one registered, then
/// set `token.done`.
///
/// The identity check is what stops a slow wind-down from unregistering a
/// newer download: a cancelled transfer can be inside a 60s socket read while
/// its replacement has already claimed the key, and an unconditional removal
/// there would leave the live download with no token for the next cancel to
/// find.
///
/// Callers release *after* the bytes are discarded, not when the threads stop.
/// Holding it across the cleanup is what keeps a second download from creating
/// the directory into a tree that is about to be removed.
pub fn end(model_id: &str, token: &CancelToken) {
    {
        let mut active = ACTIVE.lock();
        if active.get(model_id).is_some_and(|t| t.ptr_eq(token)) {
            active.remove(model_id);
        }
    }
    token.done.set();
}

/// The `model_id`s with a download registered (sorted), for status and tests.
pub fn active_keys() -> Vec<String> {
    let mut keys: Vec<String> = ACTIVE.lock().keys().cloned().collect();
    keys.sort();
    keys
}

/// Drop every registration.  Tests only.
pub fn reset_for_tests() {
    ACTIVE.lock().clear();
}
