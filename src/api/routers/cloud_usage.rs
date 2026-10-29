//! Cloud model allowance, relayed from the platform.
//!
//! Remote models are metered in credits per account per month. The desktop
//! app hands its tokens to this service and keeps none, so it reads the
//! account's allowance through here: `GET /v1/chat/quota` passes the hosted
//! answer through unchanged.
//!
//! Failures keep the hosted meaning: 401 is signed out, 403 is an account
//! without cloud access (the UI hides the meter), 502 is the platform
//! unreachable.

use axum::routing::get;
use axum::{Json, Router};

use crate::platform::client::{JsonObject, PlatformError, CLOUD_USAGE_TIMEOUT};
use crate::platform::get_platform_service;

use super::_errors::{run_blocking, unhandled_error, HTTPException};

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.cloud_usage";

/// Routes under `/v1` (tag `cloud-usage`).
pub fn router() -> Router {
    Router::new().route("/v1/chat/quota", get(get_cloud_quota))
}

/// The platform's failure as this router's status.
fn platform_error(e: PlatformError) -> HTTPException {
    if e.is_permission() {
        match e.status_code() {
            Some(403) => HTTPException::new(403, "Cloud models are not available on this account"),
            _ => HTTPException::new(401, "Not authenticated"),
        }
    } else if e.is_connection() {
        HTTPException::new(502, "Platform API unavailable")
    } else {
        tracing::error!(target: LOGGER, "Unhandled platform error: {}", e);
        unhandled_error()
    }
}

/// `GET /v1/chat/quota` — the account's cloud model allowance this period.
#[utoipa::path(
    get,
    path = "/v1/chat/quota",
    tag = "cloud-usage",
    summary = "Get the account's cloud model allowance",
    description = "Relays the hosted quota: `percent_used`, `percent_available`, `in_flight`, `requests`, `prompt_tokens`, `completion_tokens`, `period`, `resets_at`, `plan`. Credit amounts (`limit`, `used`, `remaining`, `reserved`) are null unless the account may see them; `percent_used` is null for unmetered accounts.",
    responses((status = 200, description = "OK", body = serde_json::Value), (status = 401, description = "Not signed in"), (status = 403, description = "No cloud access"), (status = 502, description = "Platform API unavailable"), (status = 500, description = "Server error"))
)]
pub async fn get_cloud_quota() -> Result<Json<JsonObject>, HTTPException> {
    run_blocking(|| get_platform_service().client().cloud_quota(CLOUD_USAGE_TIMEOUT))
        .await?
        .map(Json)
        .map_err(platform_error)
}
