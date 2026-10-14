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

use std::sync::Arc;

use axum::extract::{Path, RawQuery};
use axum::routing::delete;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::api::routers::_errors::HTTPException;
use crate::api::routers::models::{py_error_kind_of, run_blocking, validation_error, PyErrorKind};
use crate::api::services::model_service::ModelServiceError;
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

impl CancelDownloadQuery {
    /// Parse and validate a raw query string the way FastAPI validates
    /// `wait_ms: int = Query(default=0, ge=0)`: a non-integer or a negative
    /// value is a 422 in FastAPI's validation shape.  The last occurrence of
    /// a repeated parameter wins, as in Starlette.
    pub fn parse(raw: Option<&str>) -> Result<Self, HTTPException> {
        let mut wait_ms: Option<String> = None;
        if let Some(raw) = raw {
            for (k, v) in url::form_urlencoded::parse(raw.as_bytes()) {
                if k == "wait_ms" {
                    wait_ms = Some(v.into_owned());
                }
            }
        }
        let Some(raw_value) = wait_ms else {
            return Ok(Self { wait_ms: None });
        };
        let parsed = parse_py_int(&raw_value).ok_or_else(|| {
            validation_error(
                "int_parsing",
                json!(["query", "wait_ms"]),
                "Input should be a valid integer, unable to parse string as an integer",
                Value::String(raw_value.clone()),
            )
        })?;
        if parsed < 0 {
            let mut err = validation_error(
                "greater_than_equal",
                json!(["query", "wait_ms"]),
                "Input should be greater than or equal to 0",
                Value::String(raw_value),
            );
            err.detail[0]["ctx"] = json!({ "ge": 0 });
            return Err(err);
        }
        Ok(Self { wait_ms: Some(parsed) })
    }
}

/// Pydantic's lax string → int: surrounding whitespace, an optional sign,
/// digits with `_` separators.
fn parse_py_int(raw: &str) -> Option<i64> {
    let s = raw.trim();
    let (sign, digits) = match s.strip_prefix('-') {
        Some(rest) => (-1i64, rest),
        None => (1i64, s.strip_prefix('+').unwrap_or(s)),
    };
    if digits.is_empty()
        || digits.starts_with('_')
        || digits.ends_with('_')
        || digits.contains("__")
        || !digits.chars().all(|c| c.is_ascii_digit() || c == '_')
    {
        return None;
    }
    let cleaned: String = digits.chars().filter(|c| *c != '_').collect();
    cleaned.parse::<i64>().ok().map(|v| sign * v)
}

/// The `/v1` router carrying the downloads routes (`tags=["downloads"]`).
pub fn router() -> Router {
    Router::new().route("/v1/downloads/{model_name}", delete(cancel_download_handler))
}

/// The `model_id` naming `model_name`'s directory, or a `ValueError`
/// (`Unknown model: <name>`).
///
/// The workspace record first, because that is what the download is actually
/// writing under; `SUPPORTED_MODELS` covers a cancel that races the
/// registration in `/v1/init`.
pub fn resolve_model_id(model_name: &str, model_service: &ModelService) -> anyhow::Result<String> {
    let workspace = model_service.ensure_workspace(None)?;
    if let Some(entry) = workspace.read().get_model(Some(model_name)) {
        if !entry.model_id.is_empty() {
            return Ok(entry.model_id);
        }
    }

    let Some(model_cls) = SUPPORTED_MODELS.get(model_name) else {
        return Err(ModelServiceError::Value(format!("Unknown model: {model_name}")).into());
    };
    Ok((model_cls.model_id)())
}

/// `DELETE /v1/downloads/{model_name}` — axum handler.
///
/// Validates `wait_ms` (422 in FastAPI's validation shape when negative or
/// not an integer) and delegates to [`cancel_download`].
#[utoipa::path(
    delete,
    path = "/v1/downloads/{model_name}",
    tag = "models",
    params(("model_name" = String, Path, description = "Model name"), ("wait_ms" = Option<i64>, Query, description = "Milliseconds to wait for the download to stop")),
    summary = "Cancel a model download",
    description = "",
    responses((status = 200, description = "OK", body = serde_json::Value), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn cancel_download_handler(
    Path(model_name): Path<String>,
    RawQuery(query): RawQuery,
) -> Result<Json<Value>, HTTPException> {
    let query = CancelDownloadQuery::parse(query.as_deref())?;
    cancel_download(&model_name, query.wait_ms.unwrap_or(0), get_model_service())
        .await
        .map(Json)
}

/// Stop the download of `model_name` and discard what it wrote.
///
/// Signals and returns.  It does not delete anything itself and it does not
/// wait by default: cleanup belongs to the download thread, which is the only
/// one that knows every worker has stopped, and a route that unlinked while
/// the transfer was mid-write would race an open file.
///
/// A download that is already finished, or was never running, is a 200 with
/// `cancelled: false` (`status: "not_found"`) rather than a 404.  The client
/// races by construction — a progress frame arrives, the user clicks, the
/// transfer completes before the request lands — and answering 404 would push
/// the picker into an error path for an ordinary outcome.
///
/// `wait_ms` trades that promptness for certainty: it waits (bounded by
/// [`MAX_WAIT_MS`]) for the transfer's threads to stop *and* its bytes to be
/// discarded, and reports `stopped` when both have happened; otherwise
/// `cancelling`.
///
/// Errors: `ValueError` → 400, `RuntimeError` → 503, anything else → 500.
pub async fn cancel_download(
    model_name: &str,
    wait_ms: i64,
    model_service: Arc<ModelService>,
) -> Result<Value, HTTPException> {
    let fail = |e: anyhow::Error| match py_error_kind_of(&e) {
        PyErrorKind::Value => HTTPException::new(400, e.to_string()),
        PyErrorKind::Runtime => HTTPException::new(503, e.to_string()),
        _ => {
            tracing::error!(
                target: LOG_TARGET,
                "Error cancelling download for {model_name}: {e:#}"
            );
            HTTPException::new(500, e.to_string())
        }
    };

    // An older client may still address the model by its previous name.
    let name = canonical_model_name(model_name);
    let model_id = resolve_model_id(&name, &model_service).map_err(fail)?;

    let Some(token) = cancellation::request_cancel(&model_id) else {
        return Ok(json!({
            "cancelled": false,
            "status": "not_found",
            "model_name": name,
            "model_id": model_id,
        }));
    };

    let mut status = "cancelling";
    if wait_ms != 0 {
        let timeout = wait_ms.min(MAX_WAIT_MS) as f64 / 1000.0;
        let done = token.done.clone();
        if run_blocking(move || done.wait(timeout)).await {
            status = "stopped";
        }
    }

    tracing::info!(target: LOG_TARGET, "[downloads] Cancel requested for {name} ({status})");
    Ok(json!({
        "cancelled": true,
        "status": status,
        "model_name": name,
        "model_id": model_id,
    }))
}
