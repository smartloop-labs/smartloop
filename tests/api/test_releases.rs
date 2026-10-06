//! Port of `tests/api/test_releases.py`.
//!
//! API tests for the `/v1/releases` router.
//!
//! The Python tests replaced the whole `PlatformService` with a `MagicMock`
//! and asserted the `(channel, force=...)` it was called with.  Here a real
//! `PlatformService` sits over a scripted platform client, so the channel is
//! asserted on the client call, and `force` by its observable effect: with a
//! fresh cache entry, an unforced request is served from the cache without
//! reaching the client, while a forced one re-polls it.

use std::sync::Arc;

use serde_json::{json, Value};
use smartloop::platform::client::JsonObject;
use smartloop::platform::WorkspaceDB;

use crate::common::core_support::{obj, platform_service_with_db, PlatformFixture};

/// Shaped like the cloud ReleaseRead, including fields the local ReleaseInfo
/// drops.
fn release() -> Value {
    json!({
        "id": "abc123",
        "version": "1.0.15",
        "name": "Release 1.0.15",
        "notes": "shiny",
        "channel": "stable",
        "status": "published",
        "published_at": "2026-07-01T00:00:00Z",
        "artifacts": [
            {"platform": "darwin", "arch": "arm64", "kind": "zip",
             "url": "https://cdn/x.zip", "sha512": "abc", "size": 123},
        ],
        "min_version": "1.0.10",
        "created_at": "2026-06-01T00:00:00Z",
        "updated_at": "2026-06-02T00:00:00Z",
    })
}

/// [`release`] with `overrides` merged on top (`{**_RELEASE, ...}`).
fn release_with(overrides: Value) -> Value {
    let mut base = release();
    for (k, v) in overrides.as_object().expect("an object").clone() {
        base[k] = v;
    }
    base
}

/// Wrap an inner release dict in the cloud LatestReleaseResponse envelope
/// that `PlatformService::get_latest_release` actually returns.
fn wrap(release: Value) -> Option<JsonObject> {
    Some(obj(json!({"update_available": null, "update_required": false, "release": release})))
}

/// A cache-backed fixture whose `channel` already holds a freshly fetched
/// `cached` release.  Keep the returned temp dir alive with the fixture.
fn with_fresh_cache(channel: &str, cached: Value) -> (tempfile::TempDir, PlatformFixture) {
    let dir = tempfile::tempdir().expect("temp dir for the workspace DB");
    let db = WorkspaceDB::new(dir.path().join("workspace.db")).expect("workspace DB");
    db.save_release(channel, Some(&obj(cached))).expect("seed the release cache");
    let fixture = platform_service_with_db(Some(Arc::new(db)));
    (dir, fixture)
}

mod test_latest_release {
    use serial_test::serial;
    use smartloop::platform::PlatformError;

    use super::*;
    use crate::common::core_support::{get, platform_service};

    #[tokio::test]
    #[serial]
    async fn test_no_current_version_returns_release_and_feed_url() {
        let platform_service = platform_service();
        platform_service.set_latest_release(Ok(wrap(release())));

        let resp = get("/v1/releases/latest").await;
        assert_eq!(resp.status_code(), 200);
        let body = resp.json();
        assert_eq!(body["update_available"], Value::Null);
        assert_eq!(body["update_required"], false);
        assert_eq!(body["channel"], "stable");
        assert!(body["feed_url"].as_str().unwrap().ends_with("/v1/releases/feed/stable"));
        assert_eq!(body["release"]["version"], "1.0.15");
        // extra cloud fields are dropped by the local response model
        assert!(body["release"].get("id").is_none());
        assert!(body["release"].get("status").is_none());
    }

    #[tokio::test]
    #[serial]
    async fn test_older_client_gets_update_available_and_required() {
        let platform_service = platform_service();
        platform_service.set_latest_release(Ok(wrap(release())));

        let resp = get("/v1/releases/latest?current_version=0.0.1").await;
        let body = resp.json();
        assert_eq!(body["update_available"], true);
        assert_eq!(body["update_required"], true); // min_version 1.0.10 > 0.0.1
    }

    #[tokio::test]
    #[serial]
    async fn test_current_client_gets_no_update() {
        let platform_service = platform_service();
        platform_service.set_latest_release(Ok(wrap(release())));

        let resp = get("/v1/releases/latest?current_version=1.0.15").await;
        let body = resp.json();
        assert_eq!(body["update_available"], false);
        assert_eq!(body["update_required"], false);
    }

    #[tokio::test]
    #[serial]
    async fn test_beta_channel_switch() {
        let platform_service = platform_service();
        platform_service.set_latest_release(Ok(wrap(release_with(json!({"channel": "beta"})))));

        let resp = get("/v1/releases/latest?channel=beta").await;
        let body = resp.json();
        assert_eq!(body["channel"], "beta");
        assert!(body["feed_url"].as_str().unwrap().ends_with("/v1/releases/feed/beta"));
        let calls = platform_service.calls_to("get_latest_release");
        assert_eq!(calls.last().map(|c| c.args.clone()), Some(vec![json!("beta")]));
    }

    #[tokio::test]
    #[serial]
    async fn test_unknown_channel_rejected() {
        let platform_service = platform_service();
        let resp = get("/v1/releases/latest?channel=nope").await;
        assert_eq!(resp.status_code(), 400);
        assert!(platform_service.calls_to("get_latest_release").is_empty());
    }

    #[tokio::test]
    #[serial]
    async fn test_default_does_not_force() {
        // A fresh cached stable release that differs from what the cloud
        // would answer: an unforced request is served from it.
        let (_dir, platform_service) = with_fresh_cache("stable", release_with(json!({"version": "1.0.14"})));
        platform_service.set_latest_release(Ok(wrap(release())));

        let resp = get("/v1/releases/latest").await;
        assert_eq!(resp.status_code(), 200);
        assert_eq!(resp.json()["channel"], "stable");
        assert_eq!(resp.json()["release"]["version"], "1.0.14");
        assert!(
            platform_service.calls_to("get_latest_release").is_empty(),
            "an unforced request must not bypass a fresh cache"
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_force_bypasses_cache() {
        // A "check for updates" click sends force=true, which re-polls the
        // cloud immediately instead of waiting out the local cache TTL.
        let (_dir, platform_service) =
            with_fresh_cache("beta", release_with(json!({"version": "1.0.14", "channel": "beta"})));
        platform_service.set_latest_release(Ok(wrap(release_with(json!({"channel": "beta"})))));

        let resp = get("/v1/releases/latest?force=true&channel=beta").await;
        assert_eq!(resp.status_code(), 200);
        let calls = platform_service.calls_to("get_latest_release");
        assert_eq!(calls.last().map(|c| c.args.clone()), Some(vec![json!("beta")]));
        assert_eq!(resp.json()["release"]["version"], "1.0.15");
    }

    #[tokio::test]
    #[serial]
    async fn test_no_published_release() {
        // Cloud API's null-release response: wrapper present, inner release is None
        let platform_service = platform_service();
        platform_service.set_latest_release(Ok(wrap(Value::Null)));

        let resp = get("/v1/releases/latest").await;
        assert_eq!(resp.status_code(), 200);
        let body = resp.json();
        assert_eq!(body["release"], Value::Null);
        assert_eq!(body["update_available"], Value::Null);
        assert!(body["feed_url"].as_str().unwrap().ends_with("/v1/releases/feed/stable"));
    }

    #[tokio::test]
    #[serial]
    async fn test_wrapper_missing_release_key_returns_no_release() {
        // Defensive: if the cloud response ever comes back without a `release`
        // key, treat it as "nothing published" rather than 500ing.
        let platform_service = platform_service();
        platform_service.set_latest_release(Ok(Some(obj(json!({
            "update_available": null,
            "update_required": false,
        })))));

        let resp = get("/v1/releases/latest").await;
        assert_eq!(resp.status_code(), 200);
        assert_eq!(resp.json()["release"], Value::Null);
    }

    #[tokio::test]
    #[serial]
    async fn test_platform_unavailable_returns_502() {
        let platform_service = platform_service();
        platform_service.set_latest_release(Err(PlatformError::connection("down")));

        let resp = get("/v1/releases/latest").await;
        assert_eq!(resp.status_code(), 502);
    }
}
