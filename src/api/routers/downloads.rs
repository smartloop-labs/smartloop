//! Port of `smartloop/api/routers/downloads.py`.
//!
//! Model downloads as something a client can address while they run.
//!
//! Only the cancel lives here today — a download is still *started* by
//! `POST /v1/init`, which does nine other things besides.  The route is shaped
//! and named for where it is going rather than for where it came from, so the
//! desktop app's stop button is written once: when downloads become their own
//! resource, `POST`/`GET` join this router and this handler does not move.
//!
//! Workspace-level on purpose.  `supervisor::routing::project_to_proxy`
//! forwards a request that names a project in its path *or its query string*,
//! and the cancel registry is process-local — a proxied cancel would signal a
//! child's empty registry while the parent held the running transfer.  So this
//! path names no project, and must not grow a `project_id` parameter.

#![allow(unused_imports)]

use std::sync::Arc;

use axum::extract::{Path, Query};
use axum::routing::delete;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::routers::_errors::HTTPException;
use crate::api::services::{get_model_service, ModelService};
use crate::config::SUPPORTED_MODELS;
use crate::constants::canonical_model_name;
use crate::model::downloader::cancellation;

/// Python logger name (`logging.getLogger(__name__)`).
pub const LOG_TARGET: &str = "smartloop.api.routers.downloads";

/// Upper bound on the optional synchronous wait.  Long enough to cover the
/// usual case — threads stop at the next 1-8MB chunk boundary and the tree is
/// removed — without letting a caller hold a worker for the 60s a stalled
/// socket can take (`STREAM_TIMEOUT.read`).
pub const MAX_WAIT_MS: i64 = 5000;

/// Query string of `DELETE /v1/downloads/{model_name}`.
///
/// `wait_ms` must be `>= 0` (FastAPI `Query(ge=0)` → 422 otherwise).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct CancelDownloadQuery {
    /// Block up to this long for the bytes to be gone.
    #[serde(default)]
    pub wait_ms: Option<i64>,
}

/// The `/v1` router carrying the downloads routes (`tags=["downloads"]`).
pub fn router() -> Router {
    todo!()
}

/// The `model_id` naming `model_name`'s directory, or a `ValueError`
/// (`Unknown model: <name>`).
///
/// The workspace record first, because that is what the download is actually
/// writing under; `SUPPORTED_MODELS` covers a cancel that races the
/// registration in `/v1/init`.
pub fn resolve_model_id(model_name: &str, model_service: &ModelService) -> anyhow::Result<String> {
    let _ = (model_name, model_service);
    todo!()
}

/// `DELETE /v1/downloads/{model_name}` — axum handler.
///
/// Validates `wait_ms` (422 in FastAPI's validation shape when negative or
/// not an integer) and delegates to [`cancel_download`].
pub async fn cancel_download_handler(
    Path(model_name): Path<String>,
    Query(query): Query<CancelDownloadQuery>,
) -> Result<Json<Value>, HTTPException> {
    let _ = (model_name, query);
    todo!()
}

/// Stop the download of `model_name` and discard what it wrote.
///
/// Signals and returns.  It does not delete anything itself and it does not
/// wait by default: cleanup belongs to the download thread, which is the only
/// one that knows every worker has stopped.
///
/// A download that is already finished, or was never running, is a 200 with
/// `cancelled: false` (`status: "not_found"`) rather than a 404.  `wait_ms`
/// waits (bounded by [`MAX_WAIT_MS`]) for the transfer's threads to stop *and*
/// its bytes to be discarded, and reports `stopped` when both have happened;
/// otherwise `cancelling`.
///
/// Errors: `ValueError` → 400, `RuntimeError` → 503, anything else → 500.
pub async fn cancel_download(
    model_name: &str,
    wait_ms: i64,
    model_service: Arc<ModelService>,
) -> Result<Value, HTTPException> {
    let _ = (model_name, wait_ms, model_service);
    todo!()
}
