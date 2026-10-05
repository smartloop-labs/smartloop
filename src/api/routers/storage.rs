//! Port of `smartloop/api/routers/storage.py`.
//!
//! Storage usage and cleanup router.
//!
//! Reports what the SLP home directory is spending disk on and reclaims the parts
//! that are safe to lose.  The desktop app used to do this itself in Electron JS;
//! these endpoints exist so the framework — which owns the workspace record and
//! the loaded models — makes those decisions instead.

use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[allow(unused_imports)]
use crate::api::services::{get_model_service, storage_service, ModelService};

use super::_errors::HTTPException;

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.storage";

/// Categories to reclaim.  Nothing is deleted unless asked for explicitly.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct CleanupRequest {
    /// Delete downloaded models that aren't in use.
    #[serde(default)]
    pub models: bool,
    /// Restrict the model sweep to these ids; omit for every removable model.
    #[serde(default)]
    pub model_ids: Option<Vec<String>>,
    /// Delete framework installs other than the running one.
    #[serde(default)]
    pub versions: bool,
    /// Empty server.log and delete its rotated backups.
    #[serde(default)]
    pub logs: bool,
    /// Empty the regenerable cache directories.
    #[serde(default)]
    pub cache: bool,
    /// Delete project directories the workspace no longer references.
    #[serde(default)]
    pub orphan_projects: bool,
    /// Report what would be removed without removing it.
    #[serde(default)]
    pub dry_run: bool,
}

/// Routes under `/v1` (tag `storage`).
pub fn router() -> Router {
    Router::new()
        .route("/v1/storage/usage", get(get_storage_usage))
        .route("/v1/storage/cleanup", post(cleanup_storage))
}

/// `GET /v1/storage/usage` — disk usage under the SLP home directory, by category.
pub async fn get_storage_usage() -> Result<Json<Value>, HTTPException> {
    todo!()
}

/// `POST /v1/storage/cleanup` — reclaim disk space in the requested categories.
///
/// Returns what was removed, what failed, and a fresh usage report, so the
/// caller doesn't need a follow-up request to refresh its view.
pub async fn cleanup_storage(Json(request): Json<CleanupRequest>) -> Result<Json<Value>, HTTPException> {
    let _ = request;
    todo!()
}
