//! Port of `tests/unit/test_mcp_catalog_cache.py`.
//!
//! The curated MCP catalog is cached in workspace_db.
//!
//! Without a cache the Connections list 502s and a curated install silently
//! degrades to LLM enrichment whenever the platform API is unreachable or the
//! token has gone stale — neither of which means the user is offline.  They can
//! still reach a third-party MCP server and complete OAuth.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::json;
use serial_test::serial;
use smartloop::platform::client::{JsonObject, PlatformApi, PlatformError, PlatformResult};
use smartloop::platform::service::PlatformService;
use smartloop::platform::workspace_db::WorkspaceDB;

use crate::shared::platform_support::obj;

fn entry() -> JsonObject {
    obj(json!({
        "slug": "linear",
        "display_name": "Linear",
        "description": "Issue tracking for software teams",
        "hosted_by": "external",
        "remote_config": {"server_url": "https://mcp.linear.app/mcp", "auth_type": "oauth"},
        "tags": ["issues", "planning"],
        "category": "Productivity",
    }))
}

/// The `db` fixture.
fn db() -> (tempfile::TempDir, Arc<WorkspaceDB>) {
    let tmp = tempfile::tempdir().unwrap();
    let db = WorkspaceDB::new(tmp.path().join("workspace.db")).unwrap();
    (tmp, Arc::new(db))
}

/// `MagicMock()` client whose `list_mcp_catalog` answers are scripted
/// (`return_value` / `side_effect`) and counted.
struct MockClient {
    list_mcp_catalog: Mutex<PlatformResult<Vec<JsonObject>>>,
    call_count: AtomicUsize,
}

impl MockClient {
    fn returning(entries: Vec<JsonObject>) -> Arc<Self> {
        Arc::new(MockClient { list_mcp_catalog: Mutex::new(Ok(entries)), call_count: AtomicUsize::new(0) })
    }

    fn failing(err: PlatformError) -> Arc<Self> {
        Arc::new(MockClient { list_mcp_catalog: Mutex::new(Err(err)), call_count: AtomicUsize::new(0) })
    }

    fn set_side_effect(&self, err: PlatformError) {
        *self.list_mcp_catalog.lock() = Err(err);
    }
}

impl PlatformApi for MockClient {
    fn list_mcp_catalog(&self) -> PlatformResult<Vec<JsonObject>> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        self.list_mcp_catalog.lock().clone()
    }
}

mod test_workspace_db_round_trip {
    use super::*;

    /// Entries are stored as JSON blobs precisely so nesting survives.
    #[test]
    #[serial]
    fn test_nested_config_and_tags_survive() {
        let (_tmp, db) = db();
        db.save_mcp_catalog(&[entry()]).unwrap();
        let got = db.list_mcp_catalog().unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["remote_config"]["server_url"], json!("https://mcp.linear.app/mcp"));
        assert_eq!(got[0]["tags"], json!(["issues", "planning"]));
    }

    #[test]
    #[serial]
    fn test_empty_before_first_fetch() {
        let (_tmp, db) = db();
        assert_eq!(db.list_mcp_catalog().unwrap(), Vec::<JsonObject>::new());
        assert!(db.mcp_catalog_fetched_at().unwrap().is_none());
    }

    #[test]
    #[serial]
    fn test_timestamp_set_after_save() {
        let (_tmp, db) = db();
        db.save_mcp_catalog(&[entry()]).unwrap();
        assert!(db.mcp_catalog_fetched_at().unwrap().is_some());
    }

    #[test]
    #[serial]
    fn test_save_replaces_rather_than_appends() {
        let (_tmp, db) = db();
        db.save_mcp_catalog(&[entry()]).unwrap();
        let mut notion = entry();
        notion.insert("slug".into(), json!("notion"));
        notion.insert("display_name".into(), json!("Notion"));
        db.save_mcp_catalog(&[notion]).unwrap();
        let slugs: Vec<_> = db.list_mcp_catalog().unwrap().iter().map(|e| e["slug"].clone()).collect();
        assert_eq!(slugs, vec![json!("notion")]);
    }

    #[test]
    #[serial]
    fn test_entries_without_a_slug_are_skipped() {
        let (_tmp, db) = db();
        db.save_mcp_catalog(&[entry(), obj(json!({"display_name": "Broken"}))]).unwrap();
        let slugs: Vec<_> = db.list_mcp_catalog().unwrap().iter().map(|e| e["slug"].clone()).collect();
        assert_eq!(slugs, vec![json!("linear")]);
    }
}

mod test_serve_stale_when_platform_is_down {
    use super::*;

    #[test]
    #[serial]
    fn test_cache_serves_when_client_cannot_connect() {
        let (_tmp, db) = db();
        let client = MockClient::returning(vec![entry()]);
        let svc = PlatformService::new(client.clone(), Some(db));

        assert_eq!(svc.list_mcp_catalog(false).unwrap(), vec![entry()]); // populates cache

        client.set_side_effect(PlatformError::connection("platform down"));
        assert_eq!(svc.list_mcp_catalog(true).unwrap(), vec![entry()]); // still served
    }

    #[test]
    #[serial]
    fn test_fresh_cache_avoids_a_network_call() {
        let (_tmp, db) = db();
        let client = MockClient::returning(vec![entry()]);
        let svc = PlatformService::new(client.clone(), Some(db));

        svc.list_mcp_catalog(false).unwrap();
        svc.list_mcp_catalog(false).unwrap();
        assert_eq!(client.call_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    #[serial]
    fn test_no_db_falls_back_to_the_client() {
        let client = MockClient::returning(vec![entry()]);
        assert_eq!(PlatformService::new(client, None).list_mcp_catalog(false).unwrap(), vec![entry()]);
    }
}

/// Resolution must never raise — a failed lookup falls back to the
/// client-supplied URL and lets enrichment run, so an install is never blocked
/// by the catalog.
mod test_resolve_catalog_entry {
    use super::*;
    use smartloop::api::routers::mcp_catalog as module;
    use smartloop::platform::set_platform_service_override;

    /// Removes the `get_platform_service` override when dropped.
    struct PatchedService;

    impl Drop for PatchedService {
        fn drop(&mut self) {
            set_platform_service_override(None);
        }
    }

    /// `monkeypatch.setattr(mod, "get_platform_service", lambda: svc)` where
    /// `svc.list_mcp_catalog` is scripted by *client*.
    fn patch_service(client: Arc<MockClient>) -> PatchedService {
        set_platform_service_override(Some(Arc::new(PlatformService::new(client, None))));
        PatchedService
    }

    #[test]
    #[serial]
    fn test_resolves_known_slug() {
        let _svc = patch_service(MockClient::returning(vec![entry()]));
        let entry = module::resolve_catalog_entry("linear");
        assert!(entry.is_some());
        let entry = entry.unwrap();
        assert_eq!(entry.description.as_deref(), Some("Issue tracking for software teams"));
        assert_eq!(entry.tags, vec!["issues".to_string(), "planning".to_string()]);
        assert_eq!(entry.server_type.as_deref(), Some("remote"));
    }

    #[test]
    #[serial]
    fn test_unknown_slug_returns_none() {
        let _svc = patch_service(MockClient::returning(vec![entry()]));
        assert!(module::resolve_catalog_entry("nope").is_none());
    }

    #[test]
    #[serial]
    fn test_lookup_failure_returns_none_instead_of_raising() {
        let _svc = patch_service(MockClient::failing(PlatformError::connection("down")));
        assert!(module::resolve_catalog_entry("linear").is_none());
    }

    #[test]
    #[serial]
    fn test_empty_slug_short_circuits() {
        let _svc = patch_service(MockClient::returning(vec![entry()]));
        assert!(module::resolve_catalog_entry("").is_none());
    }

    /// Dispatch keys off the resolved type, so this is what lets a future
    /// local listing work with no client release.
    #[test]
    #[serial]
    fn test_local_entry_derives_local_server_type() {
        let local = obj(json!({
            "slug": "weather",
            "display_name": "Weather",
            "hosted_by": "local",
            "local_config": {"command": "uvx", "args": ["weather-mcp"]},
            "tags": ["weather"],
        }));
        let _svc = patch_service(MockClient::returning(vec![local]));
        let entry = module::resolve_catalog_entry("weather").unwrap();
        assert_eq!(entry.server_type.as_deref(), Some("local"));
        assert_eq!(entry.local_config.as_ref().unwrap().command, "uvx");
    }
}
