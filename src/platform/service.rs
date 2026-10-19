//! Port of `smartloop/platform/service.py`.
//!
//! Cache-through service for Smartloop platform API calls.
//!
//! Sits between the API routers and the platform client + local DB.  On
//! success, writes responses to the local DB.  On connection failure, falls
//! back to cached data.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde_json::Value;

use crate::platform::client::{JsonObject, PlatformApi, PlatformError, PlatformResult};
#[allow(unused_imports)]
use crate::platform::client::SLPPlatformClient;
use crate::platform::workspace_db::WorkspaceDB;

/// Python logger name for this module.
pub const LOGGER: &str = "smartloop.platform.service";

// TTLs in seconds
pub const USER_TTL: f64 = 5.0 * 60.0; // 5 minutes
pub const MODELS_TTL: f64 = 15.0 * 60.0; // 15 minutes
pub const RELEASES_TTL: f64 = 60.0 * 60.0; // 1 hour
/// 1 hour — the curated catalog changes rarely.
pub const MCP_CATALOG_TTL: f64 = 60.0 * 60.0;

/// A wall-clock source returning seconds since the epoch.
pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

static CLOCK: Lazy<RwLock<Option<Clock>>> = Lazy::new(|| RwLock::new(None));

/// The real wall-clock time in seconds since the epoch, ignoring any override.
pub fn system_now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// Current wall-clock time in seconds (Python `time.time()`), honouring an
/// override installed with [`set_clock`].  Used by this module and by
/// [`WorkspaceDB`] for `fetched_at` timestamps.
pub fn now() -> f64 {
    let clock = CLOCK.read().clone();
    match clock {
        Some(clock) => clock(),
        None => system_now(),
    }
}

/// Install (or with `None`, remove) a clock override — the seam tests use in
/// place of monkeypatching `time.time`.
pub fn set_clock(clock: Option<Clock>) {
    *CLOCK.write() = clock;
}

/// A local cache failure as a platform error (Python lets it propagate).
fn cache_error(e: anyhow::Error) -> PlatformError {
    PlatformError::Other(format!("{e:#}"))
}

/// Try cloud first, fall back to cache on connection error.
///
/// If a fresh cache entry exists (within *ttl*), return it without hitting the
/// cloud.  Otherwise fetch from cloud, update cache, and return.  On a
/// `ConnectionError`, return stale cache if available.
///
/// When *force* is true, the TTL short-circuit is skipped and the cloud is
/// always queried (used to refresh per-user state on login).  Cache write
/// failures are logged, not raised.
pub fn fetch_or_cache<T>(
    ttl: f64,
    get_cached: impl Fn() -> anyhow::Result<Option<T>>,
    get_timestamp: impl Fn() -> anyhow::Result<Option<f64>>,
    fetch_remote: impl FnOnce() -> PlatformResult<T>,
    save_cache: impl FnOnce(&T) -> anyhow::Result<()>,
    force: bool,
) -> PlatformResult<T> {
    // Check if cache is fresh enough
    if !force {
        let ts = get_timestamp().map_err(cache_error)?;
        if let Some(ts) = ts {
            if now() - ts < ttl {
                if let Some(cached) = get_cached().map_err(cache_error)? {
                    return Ok(cached);
                }
            }
        }
    }

    // Try cloud
    let data = match fetch_remote() {
        Ok(data) => data,
        Err(err) if err.is_connection() => {
            // Offline — serve stale cache if we've ever fetched successfully
            if get_timestamp().map_err(cache_error)?.is_some() {
                if let Some(cached) = get_cached().map_err(cache_error)? {
                    tracing::info!(target: "smartloop.platform.service", "Platform API unreachable, serving cached data");
                    return Ok(cached);
                }
            }
            return Err(err);
        }
        Err(err) => return Err(err),
    };

    // Cloud succeeded — update cache
    if let Err(e) = save_cache(&data) {
        tracing::warn!(target: "smartloop.platform.service", "Failed to write cache: {:#}", e);
    }

    Ok(data)
}

/// Coordinates between the cloud platform client and local DB cache.
///
/// When `db` is `None` the service degrades to a pure proxy — used in the
/// brief window before a workspace exists on disk and there is nowhere to
/// persist the cache yet.
#[derive(Clone)]
pub struct PlatformService {
    client: Arc<dyn PlatformApi>,
    db: Option<Arc<WorkspaceDB>>,
}

impl PlatformService {
    /// New service over *client* with optional cache *db*.
    pub fn new(client: Arc<dyn PlatformApi>, db: Option<Arc<WorkspaceDB>>) -> Self {
        Self { client, db }
    }

    /// The underlying platform client.
    pub fn client(&self) -> &Arc<dyn PlatformApi> {
        &self.client
    }

    /// The cache DB, if any.
    pub fn db(&self) -> Option<&Arc<WorkspaceDB>> {
        self.db.as_ref()
    }

    /// Return the current user's profile (cached or fresh).
    pub fn get_current_user(&self, force: bool) -> PlatformResult<JsonObject> {
        let Some(db) = &self.db else {
            return self.client.get_current_user();
        };
        fetch_or_cache(
            USER_TTL,
            || db.get_user(),
            || db.user_fetched_at(),
            || self.client.get_current_user(),
            |data| db.save_user(data),
            force,
        )
    }

    /// Update user profile on the cloud and refresh cache.
    pub fn update_current_user(&self, payload: &JsonObject) -> PlatformResult<JsonObject> {
        let data = self.client.update_current_user(payload)?;
        if let Some(db) = &self.db {
            if let Err(e) = db.save_user(&data) {
                tracing::warn!(target: "smartloop.platform.service", "Failed to cache user update: {:#}", e);
            }
        }
        Ok(data)
    }

    /// Permanently delete the signed-in account on the cloud.
    ///
    /// The cached profile is left alone, as logout leaves it: the credentials
    /// that authorised the call are cleared by the caller, and nothing reads
    /// a profile while signed out.
    pub fn delete_current_user(&self) -> PlatformResult<()> {
        self.client.delete_current_user()
    }

    /// Return the model catalog (cached or fresh).
    pub fn list_models(&self, force: bool) -> PlatformResult<Vec<JsonObject>> {
        let Some(db) = &self.db else {
            return self.client.list_models();
        };
        fetch_or_cache(
            MODELS_TTL,
            || db.list_models().map(Some),
            || db.models_fetched_at(),
            || self.client.list_models(),
            |models| db.save_models(models),
            force,
        )
    }

    /// Return the curated MCP server catalog (cached or fresh).
    ///
    /// Cached so the Connections list still renders — and a curated install
    /// can still resolve its slug — when the platform API is unreachable or
    /// the token has gone stale.  Neither state implies the user is offline:
    /// they can still reach a third-party MCP server and complete OAuth.
    pub fn list_mcp_catalog(&self, force: bool) -> PlatformResult<Vec<JsonObject>> {
        let Some(db) = &self.db else {
            return self.client.list_mcp_catalog();
        };
        fetch_or_cache(
            MCP_CATALOG_TTL,
            || db.list_mcp_catalog().map(Some),
            || db.mcp_catalog_fetched_at(),
            || self.client.list_mcp_catalog(),
            |entries| db.save_mcp_catalog(entries),
            force,
        )
    }

    /// Return the latest-release envelope for *channel* (cached or fresh).
    ///
    /// Mirrors the cloud `LatestReleaseResponse` shape `{update_available,
    /// update_required, release}`.  We poll the cloud WITHOUT
    /// `current_version` and cache only the inner `release` (so one cache
    /// entry serves every client); `update_available` / `update_required` are
    /// left null/false here and computed per-request by the caller.
    /// `release` is `null` when the channel has no published release.
    ///
    /// Set *force* to bypass the TTL and re-poll the cloud — used by the
    /// client's explicit "check for updates" action, mirroring
    /// [`Self::get_current_user`].
    ///
    /// NOTE: this does NOT reuse [`fetch_or_cache`].  That helper treats a
    /// cached value of `None` as "nothing cached", but for releases `None` is
    /// a *legitimate* cached value meaning "this channel has no published
    /// release yet" (e.g. an unreleased `beta`).  Conflating the two would
    /// make the TTL fast-path and the offline fallback misfire for empty
    /// channels — re-polling the cloud on every request, and returning an
    /// error offline for a state we already know.  So we discriminate on the
    /// fetch *timestamp* (present iff we've ever successfully polled), which
    /// the DB records even when the release itself is `None`.
    pub fn get_latest_release(&self, channel: &str, force: bool) -> PlatformResult<JsonObject> {
        let fetch_release = || -> PlatformResult<Option<JsonObject>> {
            let wrapper = self.client.get_latest_release(channel)?.unwrap_or_default();
            match wrapper.get("release") {
                None | Some(Value::Null) => Ok(None),
                Some(Value::Object(release)) => Ok(Some(release.clone())),
                Some(other) => Err(PlatformError::Value(format!(
                    "Unexpected release payload for channel {channel:?}: {other}"
                ))),
            }
        };

        let Some(db) = &self.db else {
            return Ok(envelope(fetch_release()?));
        };

        let fetched_at = db.release_fetched_at(channel).map_err(cache_error)?;

        // Fresh cache within TTL — serve without hitting the cloud. Works even
        // when the cached release is None (channel has nothing published).
        if let Some(fetched_at) = fetched_at {
            if !force && now() - fetched_at < RELEASES_TTL {
                return Ok(envelope(db.get_release(channel).map_err(cache_error)?));
            }
        }

        let release = match fetch_release() {
            Ok(release) => release,
            Err(err) if err.is_connection() => {
                // Offline — serve whatever we last learned, including a
                // known-empty channel.  Only re-raise if we've genuinely never
                // fetched.
                if fetched_at.is_some() {
                    tracing::info!(
                        target: "smartloop.platform.service",
                        "Platform API unreachable, serving cached release for '{}'",
                        channel
                    );
                    return Ok(envelope(db.get_release(channel).map_err(cache_error)?));
                }
                return Err(err);
            }
            Err(err) => return Err(err),
        };

        if let Err(e) = db.save_release(channel, release.as_ref()) {
            tracing::warn!(target: "smartloop.platform.service", "Failed to write release cache: {:#}", e);
        }
        Ok(envelope(release))
    }
}

/// `{"update_available": null, "update_required": false, "release": release}`.
fn envelope(release: Option<JsonObject>) -> JsonObject {
    let mut out = JsonObject::new();
    out.insert("update_available".into(), Value::Null);
    out.insert("update_required".into(), Value::Bool(false));
    out.insert("release".into(), release.map(Value::Object).unwrap_or(Value::Null));
    out
}
