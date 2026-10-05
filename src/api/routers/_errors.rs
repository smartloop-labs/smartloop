//! Port of `smartloop/api/routers/_errors.py`.
//!
//! Turning a framework exception back into the HTTP status it stands for.
//!
//! Shared because the alternative is each router deciding for itself, and the
//! decisions drifted: `PermissionError` means four different things depending on
//! who raised it, and a router that reads it as one of them answers "sign in" to a
//! full disk.
//!
//! Also home to [`HTTPException`], the Rust stand-in for FastAPI's
//! `HTTPException`: it renders as `{"detail": ...}` with the given status.

use std::error::Error as StdError;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::Value;

#[allow(unused_imports)]
use crate::model::downloader::artifact_downloader::DownloadError;
#[allow(unused_imports)]
use crate::platform::{ModelAccessError, PlatformError};

/// `errno.ENOSPC` — no space left on device.
pub const ENOSPC: i32 = 28;

/// FastAPI's `HTTPException`: an error answered as `{"detail": <detail>}` with
/// `status_code`.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{status_code}: {detail}")]
pub struct HTTPException {
    /// HTTP status to answer with.
    pub status_code: u16,
    /// The `detail` payload (usually a string).
    pub detail: Value,
}

impl HTTPException {
    /// An exception whose detail is the string *detail*.
    pub fn new(status_code: u16, detail: impl Into<String>) -> Self {
        Self { status_code, detail: Value::String(detail.into()) }
    }

    /// An exception with an arbitrary JSON detail.
    pub fn with_detail(status_code: u16, detail: Value) -> Self {
        Self { status_code, detail }
    }
}

impl IntoResponse for HTTPException {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status_code)
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, axum::Json(serde_json::json!({ "detail": self.detail }))).into_response()
    }
}

/// True when *exc* — or the failure it wraps — is an out-of-space OSError.
///
/// The download path wraps a raw ENOSPC in a classified `ConnectionError`
/// (keeping the cause as its source), so the errno is not always on the error
/// itself — follow the chain before declaring it a network failure.
pub fn is_disk_full(exc: &(dyn StdError + 'static)) -> bool {
    let _ = exc;
    todo!()
}

/// Classify *exc* as the HTTP status it would have been.
///
/// Platform errors carry the upstream status and are answered with it, so the
/// client never has to match on message text.
///
/// Only [`ModelAccessError`] earns a 403.  `PermissionError` used to, which was
/// too wide by half: an unwritable models directory arrived as "not on your
/// plan".  Disk full is 507; a `ConnectionError` is 502; anything else is 500.
pub fn status_for(exc: &(dyn StdError + 'static)) -> u16 {
    let _ = exc;
    todo!()
}
