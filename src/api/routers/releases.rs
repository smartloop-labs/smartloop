//! Port of `smartloop/api/routers/releases.py`.
//!
//! Release update-check router.
//!
//! Exposes the current published Electron app release for a channel, cached
//! locally from the cloud platform API.  The client (Electron) polls this to show
//! a "new version available" notification; stage 2 will point electron-updater at
//! the returned `feed_url`.

use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::app_settings::AppSettings;
use crate::platform::{get_platform_service, semver};

use super::_errors::{
    deserialize_py_bool, py_repr_str, run_blocking, unhandled_error, HTTPException, ValidatedQuery,
};

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
    #[serde(default, deserialize_with = "deserialize_py_bool")]
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
/// are computed locally by semver comparison, mirroring the cloud
/// `/releases/latest` contract.  `feed_url` is derived from `SLP_API_BASE` so it
/// tracks dev/prod automatically.
///
/// Pass `force=true` (used by an explicit "check for updates" action) to bypass
/// the local cache TTL and re-poll the cloud immediately.  400 for an unknown
/// channel, 502 when the platform is unavailable.
pub async fn latest_release(
    ValidatedQuery(query): ValidatedQuery<LatestReleaseQuery>,
) -> Result<Json<LatestReleaseResponse>, HTTPException> {
    // The platform service and settings are synchronous; keep them off the
    // async runtime.
    run_blocking(move || latest_release_sync(query)).await?.map(Json)
}

/// The body of [`latest_release`].
pub fn latest_release_sync(query: LatestReleaseQuery) -> Result<LatestReleaseResponse, HTTPException> {
    let settings = AppSettings::new();
    let channels = settings.release_channels();
    let channel = match query.channel.filter(|c| !c.is_empty()) {
        Some(channel) => channel,
        None => match channels.first() {
            Some(first) => first.clone(),
            None => return Err(unhandled_error()),
        },
    };
    if !channels.contains(&channel) {
        return Err(HTTPException::new(400, format!("Unknown channel {}", py_repr_str(&channel))));
    }

    let svc = get_platform_service();
    let wrapper = match svc.get_latest_release(&channel, query.force) {
        Ok(wrapper) => wrapper,
        Err(e) if e.is_connection() => {
            return Err(HTTPException::new(502, "Platform API unavailable"));
        }
        Err(e) => {
            tracing::error!(target: LOGGER, "Unhandled error fetching the latest release: {}", e);
            return Err(unhandled_error());
        }
    };

    let feed_url = format!(
        "{}/v1/releases/feed/{}",
        settings.slp_api_base.trim_end_matches('/'),
        channel
    );

    // PlatformService returns the cloud wrapper {update_available, update_required, release};
    // the inner `release` may be None when no published release exists for the channel.
    let release = match wrapper.get("release") {
        Some(Value::Object(release)) if !release.is_empty() => release.clone(),
        _ => {
            return Ok(LatestReleaseResponse {
                feed_url,
                channel,
                ..Default::default()
            });
        }
    };

    let info: ReleaseInfo = serde_json::from_value(Value::Object(release)).map_err(|e| {
        tracing::error!(target: LOGGER, "Invalid release payload from the platform API: {}", e);
        unhandled_error()
    })?;

    let mut update_available = None;
    let mut update_required = false;
    if let Some(current_version) = query.current_version.as_deref().filter(|v| !v.is_empty()) {
        update_available = Some(semver::is_newer(&info.version, current_version));
        if let Some(min_version) = info.min_version.as_deref().filter(|v| !v.is_empty()) {
            update_required = semver::is_newer(min_version, current_version);
        }
    }

    Ok(LatestReleaseResponse {
        update_available,
        update_required,
        release: Some(info),
        feed_url,
        channel,
    })
}
