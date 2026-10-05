//! Port of `smartloop/api/routers/releases.py`.
//!
//! Release update-check router.
//!
//! Exposes the current published Electron app release for a channel, cached
//! locally from the cloud platform API.  The client (Electron) polls this to show
//! a "new version available" notification; stage 2 will point electron-updater at
//! the returned `feed_url`.

use axum::extract::Query;
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::config::app_settings::AppSettings;
#[allow(unused_imports)]
use crate::platform::{get_platform_service, semver, PlatformService};

use super::_errors::HTTPException;

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.releases";

/// One downloadable artifact of a release.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ReleaseArtifact {
    pub platform: String,
    pub arch: String,
    pub kind: String,
    pub url: String,
    pub sha512: String,
    pub size: i64,
}

/// A published release.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ReleaseInfo {
    pub version: String,
    pub name: String,
    #[serde(default)]
    pub notes: String,
    pub channel: String,
    #[serde(default)]
    pub published_at: Option<String>,
    #[serde(default)]
    pub artifacts: Vec<ReleaseArtifact>,
    #[serde(default)]
    pub min_version: Option<String>,
}

/// Response of `GET /v1/releases/latest`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct LatestReleaseResponse {
    #[serde(default)]
    pub update_available: Option<bool>,
    #[serde(default)]
    pub update_required: bool,
    #[serde(default)]
    pub release: Option<ReleaseInfo>,
    pub feed_url: String,
    pub channel: String,
}

/// Query of `GET /v1/releases/latest`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct LatestReleaseQuery {
    #[serde(default)]
    pub current_version: Option<String>,
    #[serde(default)]
    pub channel: Option<String>,
    #[serde(default)]
    pub force: bool,
}

/// Routes under `/v1` (tag `releases`).
pub fn router() -> Router {
    Router::new().route("/v1/releases/latest", get(latest_release))
}

/// `GET /v1/releases/latest` — the latest published release for *channel*
/// (cached locally).
///
/// When *current_version* is supplied, `update_available` / `update_required`
/// are computed locally by semver comparison, mirroring the cloud contract.
/// `feed_url` is derived from `SLP_API_BASE`.  `force=true` bypasses the local
/// cache TTL.  400 for an unknown channel, 502 when the platform is unavailable.
pub async fn latest_release(
    Query(query): Query<LatestReleaseQuery>,
) -> Result<Json<LatestReleaseResponse>, HTTPException> {
    let _ = query;
    todo!()
}
