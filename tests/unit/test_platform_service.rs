//! Port of `tests/unit/test_platform_service.py`.
//!
//! `#[serial]` throughout: several tests shift the platform clock, which every
//! TTL check reads.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Value};
use serial_test::serial;
use smartloop::platform::client::{JsonObject, PlatformApi, PlatformError, PlatformResult};
use smartloop::platform::service::{PlatformService, RELEASES_TTL, USER_TTL};
use smartloop::platform::workspace_db::WorkspaceDB;

use crate::shared::platform_support::{advance_clock, obj};

fn user() -> JsonObject {
    obj(json!({
        "email": "chris@example.com",
        "name": "Chris",
        "is_premium": false,
        "is_paid": true,
        "number_of_tokens_processed": 10,
        "default_temperature": 0.3,
        "default_memory": true,
    }))
}

fn models() -> Vec<JsonObject> {
    vec![obj(json!({
        "model_id": "google/gemma-3-1b-it", "name": "gemma3-1b", "size": "1B",
        "description": "Gemma 3 1B", "access_level": "public", "is_accessible": true,
    }))]
}

fn release() -> JsonObject {
    obj(json!({"version": "1.0.15", "name": "v", "notes": "", "channel": "stable",
               "artifacts": [], "min_version": null}))
}

/// The cloud client returns the full LatestReleaseResponse wrapper.
fn release_wrapper() -> JsonObject {
    obj(json!({"update_available": null, "update_required": false, "release": release()}))
}

/// A channel that exists but has nothing published yet (e.g. an unreleased
/// `beta`): the cloud returns a wrapper whose `release` is None.  `None` is a
/// real cached value here, not a cache miss — these tests guard the sentinel fix.
fn empty_wrapper() -> JsonObject {
    obj(json!({"update_available": null, "update_required": false, "release": null}))
}

/// Minimal stand-in for SLPPlatformClient.
#[derive(Default)]
struct FakeClient {
    user: Mutex<Option<JsonObject>>,
    models: Option<Vec<JsonObject>>,
    release_wrapper: Option<JsonObject>,
    fail: AtomicBool,
    call_count: Mutex<HashMap<&'static str, usize>>,
}

impl FakeClient {
    fn new(
        user: Option<JsonObject>,
        models: Option<Vec<JsonObject>>,
        release_wrapper: Option<JsonObject>,
        fail: bool,
    ) -> Arc<Self> {
        Arc::new(FakeClient {
            user: Mutex::new(user),
            models,
            release_wrapper,
            fail: AtomicBool::new(fail),
            call_count: Mutex::new(HashMap::new()),
        })
    }

    fn with_user(u: JsonObject) -> Arc<Self> {
        Self::new(Some(u), None, None, false)
    }

    fn with_models(m: Vec<JsonObject>) -> Arc<Self> {
        Self::new(None, Some(m), None, false)
    }

    fn with_release(w: JsonObject) -> Arc<Self> {
        Self::new(None, None, Some(w), false)
    }

    fn failing() -> Arc<Self> {
        Self::new(None, None, None, true)
    }

    fn calls(&self, name: &str) -> usize {
        self.call_count.lock().get(name).copied().unwrap_or(0)
    }

    fn hit(&self, name: &'static str) -> PlatformResult<()> {
        *self.call_count.lock().entry(name).or_default() += 1;
        if self.fail.load(Ordering::SeqCst) {
            return Err(PlatformError::connection("offline"));
        }
        Ok(())
    }
}

impl PlatformApi for FakeClient {
    fn get_latest_release(&self, _channel: &str) -> PlatformResult<Option<JsonObject>> {
        self.hit("release")?;
        Ok(self.release_wrapper.clone())
    }

    fn get_current_user(&self) -> PlatformResult<JsonObject> {
        self.hit("user")?;
        Ok(self.user.lock().clone().expect("no user configured"))
    }

    fn update_current_user(&self, payload: &JsonObject) -> PlatformResult<JsonObject> {
        self.hit("update")?;
        let mut merged = self.user.lock().clone().unwrap_or_default();
        for (k, v) in payload {
            merged.insert(k.clone(), v.clone());
        }
        *self.user.lock() = Some(merged.clone());
        Ok(merged)
    }

    fn list_models(&self) -> PlatformResult<Vec<JsonObject>> {
        self.hit("models")?;
        Ok(self.models.clone().expect("no models configured"))
    }
}

/// A WorkspaceDB backed by a temp file (the `db` fixture).
fn db() -> (tempfile::TempDir, Arc<WorkspaceDB>) {
    let tmp = tempfile::tempdir().unwrap();
    let db = WorkspaceDB::new(tmp.path().join("workspace.db")).unwrap();
    (tmp, Arc::new(db))
}

fn service(client: &Arc<FakeClient>, db: Option<&Arc<WorkspaceDB>>) -> PlatformService {
    PlatformService::new(client.clone(), db.cloned())
}

mod test_get_current_user {
    use super::*;

    #[test]
    #[serial]
    fn test_fetches_from_cloud_on_first_call() {
        let (_tmp, db) = db();
        let client = FakeClient::with_user(user());
        let svc = service(&client, Some(&db));

        let result = svc.get_current_user(false).unwrap();
        assert_eq!(result["email"], json!("chris@example.com"));
        assert_eq!(client.calls("user"), 1);
    }

    #[test]
    #[serial]
    fn test_returns_cached_within_ttl() {
        let (_tmp, db) = db();
        let client = FakeClient::with_user(user());
        let svc = service(&client, Some(&db));

        svc.get_current_user(false).unwrap();
        svc.get_current_user(false).unwrap();
        assert_eq!(client.calls("user"), 1); // only one cloud call
    }

    #[test]
    #[serial]
    fn test_falls_back_to_cache_when_offline() {
        let (_tmp, db) = db();
        // Seed the cache
        db.save_user(&user()).unwrap();

        let client = FakeClient::failing();
        let svc = service(&client, Some(&db));

        let result = svc.get_current_user(false).unwrap();
        assert_eq!(result["email"], json!("chris@example.com"));
    }

    #[test]
    #[serial]
    fn test_raises_when_offline_and_no_cache() {
        let (_tmp, db) = db();
        let client = FakeClient::failing();
        let svc = service(&client, Some(&db));

        let err = svc.get_current_user(false).unwrap_err();
        assert!(err.is_connection());
    }

    #[test]
    #[serial]
    fn test_refetches_after_ttl() {
        let (_tmp, db) = db();
        let client = FakeClient::with_user(user());
        let svc = service(&client, Some(&db));

        svc.get_current_user(false).unwrap();
        assert_eq!(client.calls("user"), 1);

        // Simulate time passing beyond TTL
        let _clock = advance_clock(USER_TTL + 1.0);

        svc.get_current_user(false).unwrap();
        assert_eq!(client.calls("user"), 2);
    }
}

mod test_update_current_user {
    use super::*;

    #[test]
    #[serial]
    fn test_updates_cloud_and_cache() {
        let (_tmp, db) = db();
        let client = FakeClient::with_user(user());
        let svc = service(&client, Some(&db));

        let result = svc.update_current_user(&obj(json!({"default_temperature": 0.9}))).unwrap();
        assert_eq!(result["default_temperature"], json!(0.9));

        // Cache should be updated
        let cached = db.get_user().unwrap().unwrap();
        assert_eq!(cached["default_temperature"], json!(0.9));
    }

    #[test]
    #[serial]
    fn test_raises_when_offline() {
        let (_tmp, db) = db();
        let client = FakeClient::failing();
        let svc = service(&client, Some(&db));

        let err = svc.update_current_user(&obj(json!({"default_temperature": 0.9}))).unwrap_err();
        assert!(err.is_connection());
    }
}

/// When no workspace exists yet, the service runs as a pure proxy.
mod test_no_db {
    use super::*;

    #[test]
    #[serial]
    fn test_get_user_proxies_to_cloud() {
        let client = FakeClient::with_user(user());
        let svc = service(&client, None);
        assert_eq!(svc.get_current_user(false).unwrap(), user());
        // Hits cloud every time — no cache.
        svc.get_current_user(false).unwrap();
        assert_eq!(client.calls("user"), 2);
    }

    #[test]
    #[serial]
    fn test_list_models_proxies_to_cloud() {
        let client = FakeClient::with_models(models());
        let svc = service(&client, None);
        assert_eq!(svc.list_models(false).unwrap(), models());
        svc.list_models(false).unwrap();
        assert_eq!(client.calls("models"), 2);
    }

    #[test]
    #[serial]
    fn test_get_latest_release_proxies_to_cloud() {
        let client = FakeClient::with_release(release_wrapper());
        let svc = service(&client, None);
        // returns the LatestReleaseResponse envelope
        assert_eq!(svc.get_latest_release("stable", false).unwrap(), release_wrapper());
        svc.get_latest_release("stable", false).unwrap();
        assert_eq!(client.calls("release"), 2);
    }

    #[test]
    #[serial]
    fn test_update_proxies_without_caching() {
        let client = FakeClient::with_user(user());
        let svc = service(&client, None);
        let result = svc.update_current_user(&obj(json!({"default_temperature": 0.5}))).unwrap();
        assert_eq!(result["default_temperature"], json!(0.5));
    }
}

mod test_list_models {
    use super::*;

    #[test]
    #[serial]
    fn test_listed_models_hydrate_the_spec_registry() {
        use smartloop::model::spec::MODEL_SPECS;

        let row = obj(json!({
            "name": "spec-hydration-probe", "model_id": "org/spec-hydration-probe",
            "param_size_b": 7.0, "capabilities": ["general", "vision"],
        }));
        let client = FakeClient::with_models(vec![row]);
        let svc = service(&client, None);
        svc.list_models(false).unwrap();

        let spec = MODEL_SPECS.get("spec-hydration-probe").expect("hydrated from the catalog");
        assert_eq!(spec.model_id, "org/spec-hydration-probe");
        assert_eq!(spec.param_size_b, 7.0);
        // "general" said the model holds a text conversation: `text`.
        assert_eq!(spec.capability_values(), ["text", "vision"]);
    }

    #[test]
    #[serial]
    fn test_fetches_from_cloud_on_first_call() {
        let (_tmp, db) = db();
        let client = FakeClient::with_models(models());
        let svc = service(&client, Some(&db));

        let result = svc.list_models(false).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["model_id"], json!("google/gemma-3-1b-it"));
        assert_eq!(client.calls("models"), 1);
    }

    #[test]
    #[serial]
    fn test_returns_cached_within_ttl() {
        let (_tmp, db) = db();
        let client = FakeClient::with_models(models());
        let svc = service(&client, Some(&db));

        svc.list_models(false).unwrap();
        svc.list_models(false).unwrap();
        assert_eq!(client.calls("models"), 1);
    }

    #[test]
    #[serial]
    fn test_falls_back_to_cache_when_offline() {
        let (_tmp, db) = db();
        db.save_models(&models()).unwrap();

        let client = FakeClient::failing();
        let svc = service(&client, Some(&db));

        let result = svc.list_models(false).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    #[serial]
    fn test_raises_when_offline_and_no_cache() {
        let (_tmp, db) = db();
        let client = FakeClient::failing();
        let svc = service(&client, Some(&db));

        let err = svc.list_models(false).unwrap_err();
        assert!(err.is_connection());
    }
}

mod test_get_latest_release {
    use super::*;

    #[test]
    #[serial]
    fn test_returns_envelope_and_caches_the_release_field() {
        let (_tmp, db) = db();
        let client = FakeClient::with_release(release_wrapper());
        let svc = service(&client, Some(&db));

        let result = svc.get_latest_release("stable", false).unwrap();
        assert_eq!(result, release_wrapper());
        assert_eq!(client.calls("release"), 1);
        // only the inner release is cached (client-agnostic)
        assert_eq!(db.get_release("stable").unwrap(), Some(release()));
    }

    #[test]
    #[serial]
    fn test_returns_cached_within_ttl() {
        let (_tmp, db) = db();
        let client = FakeClient::with_release(release_wrapper());
        let svc = service(&client, Some(&db));

        svc.get_latest_release("stable", false).unwrap();
        svc.get_latest_release("stable", false).unwrap();
        assert_eq!(client.calls("release"), 1);
    }

    #[test]
    #[serial]
    fn test_force_bypasses_ttl() {
        let (_tmp, db) = db();
        let client = FakeClient::with_release(release_wrapper());
        let svc = service(&client, Some(&db));

        svc.get_latest_release("stable", false).unwrap();
        svc.get_latest_release("stable", true).unwrap();
        assert_eq!(client.calls("release"), 2);
    }

    #[test]
    #[serial]
    fn test_falls_back_to_cache_when_offline() {
        let (_tmp, db) = db();
        db.save_release("stable", Some(&release())).unwrap();

        let client = FakeClient::failing();
        let svc = service(&client, Some(&db));

        assert_eq!(svc.get_latest_release("stable", false).unwrap()["release"], Value::Object(release()));
    }

    #[test]
    #[serial]
    fn test_refetches_after_ttl() {
        let (_tmp, db) = db();
        let client = FakeClient::with_release(release_wrapper());
        let svc = service(&client, Some(&db));

        svc.get_latest_release("stable", false).unwrap();
        assert_eq!(client.calls("release"), 1);

        let _clock = advance_clock(RELEASES_TTL + 1.0);

        svc.get_latest_release("stable", false).unwrap();
        assert_eq!(client.calls("release"), 2);
    }

    #[test]
    #[serial]
    fn test_raises_when_offline_and_never_fetched() {
        let (_tmp, db) = db();
        let client = FakeClient::failing();
        let svc = service(&client, Some(&db));

        let err = svc.get_latest_release("beta", false).unwrap_err();
        assert!(err.is_connection());
    }
}

mod test_get_latest_release_empty_channel {
    use super::*;

    #[test]
    #[serial]
    fn test_empty_release_is_cached_within_ttl() {
        let (_tmp, db) = db();
        let client = FakeClient::with_release(empty_wrapper());
        let svc = service(&client, Some(&db));

        let first = svc.get_latest_release("beta", false).unwrap();
        let second = svc.get_latest_release("beta", false).unwrap();

        assert_eq!(first["release"], Value::Null);
        assert_eq!(second["release"], Value::Null);
        // The TTL fast-path must apply even though the release is None — no
        // redundant second cloud call for an empty channel.
        assert_eq!(client.calls("release"), 1);
    }

    #[test]
    #[serial]
    fn test_empty_release_served_offline_after_being_fetched() {
        let (_tmp, db) = db();
        let client = FakeClient::with_release(empty_wrapper());
        let svc = service(&client, Some(&db));

        // First poll learns "nothing published" and records a timestamp.
        svc.get_latest_release("beta", false).unwrap();

        // Now go offline and force past the TTL: we should serve the
        // known-empty state rather than raising, because we HAVE fetched before.
        client.fail.store(true, Ordering::SeqCst);
        let result = svc.get_latest_release("beta", true).unwrap();
        assert_eq!(result["release"], Value::Null);
    }

    #[test]
    #[serial]
    fn test_refetches_empty_channel_after_ttl() {
        let (_tmp, db) = db();
        let client = FakeClient::with_release(empty_wrapper());
        let svc = service(&client, Some(&db));

        svc.get_latest_release("beta", false).unwrap();
        assert_eq!(client.calls("release"), 1);

        let _clock = advance_clock(RELEASES_TTL + 1.0);

        svc.get_latest_release("beta", false).unwrap();
        assert_eq!(client.calls("release"), 2);
    }
}
