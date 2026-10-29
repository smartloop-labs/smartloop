//! Port of `tests/unit/test_platform_client.py`.
//!
//! Every test runs inside the conftest fixtures ([`autouse`]) against a local
//! canned-response server ([`StubServer`], the `stub_server` fixture).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Value};
use serial_test::serial;
use smartloop::auth::credential_store;
use smartloop::platform::client::{
    Headers, JsonObject, PlatformError, CLOUD_USAGE_TIMEOUT, SEARCH_STATUS_TIMEOUT, SEARCH_TIMEOUT, SLPPlatformClient,
};
use smartloop::workspace::context::pin_project;

use crate::shared::platform_support::{autouse, obj, StubServer};

const UNREACHABLE: &str = "http://127.0.0.1:1";

fn headers(pairs: &[(&str, &str)]) -> Headers {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

/// `SLPPlatformClient(base_url, auth_headers_fn=lambda: <pairs>)`.
fn client_with_auth(base_url: &str, pairs: &'static [(&'static str, &'static str)]) -> SLPPlatformClient {
    SLPPlatformClient::with_hooks(base_url, Some(Arc::new(move || headers(pairs))), None)
}

fn objects(value: Value) -> Vec<JsonObject> {
    value.as_array().unwrap().iter().map(|v| obj(v.clone())).collect()
}

fn artifacts(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

/// `pytest.raises(ConnectionError, match=...)`.
#[track_caller]
fn assert_connection(err: &PlatformError, needle: &str) {
    assert!(err.is_connection(), "expected ConnectionError, got {err:?}");
    assert!(err.message().contains(needle), "{:?} does not mention {needle:?}", err.message());
}

/// `pytest.raises(PermissionError, match=...)`.
#[track_caller]
fn assert_permission(err: &PlatformError, needle: &str) {
    assert!(err.is_permission(), "expected PermissionError, got {err:?}");
    assert!(err.message().contains(needle), "{:?} does not mention {needle:?}", err.message());
}

/// `pytest.raises(ValueError, match=...)`.
#[track_caller]
fn assert_value_error(err: &PlatformError, needle: &str) {
    assert!(matches!(err, PlatformError::Value(_)), "expected ValueError, got {err:?}");
    assert!(err.message().contains(needle), "{:?} does not mention {needle:?}", err.message());
}

mod test_list_bootstrap_models {
    use super::*;

    #[test]
    #[serial]
    fn test_returns_models() {
        let _c = autouse();
        let server = StubServer::start();
        let models = json!([{"name": "slp-orchestra-mini", "model_id": "google/gemma-3-1b-it"}]);
        server.route("/v1/bootstrap/models", 200, models.clone());

        let client = SLPPlatformClient::new(server.base_url());
        let result = client.list_bootstrap_models().unwrap();
        assert_eq!(result, objects(models));
    }

    #[test]
    #[serial]
    fn test_empty_response_raises() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/bootstrap/models", 200, json!([]));

        let client = SLPPlatformClient::new(server.base_url());
        let err = client.list_bootstrap_models().unwrap_err();
        assert_value_error(&err, "no models");
    }

    #[test]
    #[serial]
    fn test_404_raises_connection_error() {
        let _c = autouse();
        let server = StubServer::start(); // no route registered

        let client = SLPPlatformClient::new(server.base_url());
        let err = client.list_bootstrap_models().unwrap_err();
        assert_connection(&err, "not found");
    }
}

mod test_list_models {
    use super::*;

    #[test]
    #[serial]
    fn test_returns_models_with_auth() {
        let _c = autouse();
        let server = StubServer::start();
        let models = json!([{"name": "slp-orchestra-mini", "model_id": "google/gemma-3-1b-it"}]);
        server.route("/v1/models", 200, models.clone());

        let client = client_with_auth(server.base_url(), &[("Authorization", "Bearer jwt")]);
        assert_eq!(client.list_models().unwrap(), objects(models));
    }

    #[test]
    #[serial]
    fn test_returns_models_without_auth() {
        let _c = autouse();
        let server = StubServer::start();
        let models = json!([{"name": "slp-orchestra-mini", "model_id": "google/gemma-3-1b-it"}]);
        server.route("/v1/models", 200, models.clone());

        let client = client_with_auth(server.base_url(), &[]);
        assert_eq!(client.list_models().unwrap(), objects(models));
    }

    /// When the first request (with auth) gets 401, retry without auth succeeds.
    #[test]
    #[serial]
    fn test_retries_without_auth_on_401() {
        let _c = autouse();
        let server = StubServer::start();
        let models = json!([{"name": "slp-orchestra-mini", "model_id": "google/gemma-3-1b-it"}]);
        // 401 on the first call, 200 on every one after it.
        server.route_seq("/v1/models", vec![(401, json!({"detail": "Unauthorized"})), (200, models.clone())]);

        let client = client_with_auth(server.base_url(), &[("Authorization", "Bearer expired")]);
        assert_eq!(client.list_models().unwrap(), objects(models));
        assert_eq!(server.requests().len(), 2);
    }

    #[test]
    #[serial]
    fn test_connection_error_on_non_401() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/models", 500, json!({"detail": "Internal Server Error"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.list_models().unwrap_err();
        assert_connection(&err, "API error");
    }

    #[test]
    #[serial]
    fn test_connection_error_on_unreachable() {
        let _c = autouse();
        let client = client_with_auth(UNREACHABLE, &[]);
        let err = client.list_models().unwrap_err();
        assert!(err.is_connection(), "{err:?}");
    }
}

mod test_get_model_artifacts {
    use super::*;

    #[test]
    #[serial]
    fn test_returns_artifacts() {
        let _c = autouse();
        let server = StubServer::start();
        server.route(
            "/v1/models/slp-orchestra-mini/download",
            200,
            json!({"artifacts": {"model.gguf": "https://storage/model.gguf"}}),
        );

        let client = client_with_auth(server.base_url(), &[("X-Api-Key", "test")]);
        let result = client.get_model_artifacts("slp-orchestra-mini").unwrap();
        assert_eq!(result, artifacts(&[("model.gguf", "https://storage/model.gguf")]));
    }

    #[test]
    #[serial]
    fn test_empty_artifacts_raises() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/models/slp-orchestra-mini/download", 200, json!({"artifacts": {}}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.get_model_artifacts("slp-orchestra-mini").unwrap_err();
        assert_value_error(&err, "no artifacts");
    }

    #[test]
    #[serial]
    fn test_401_raises_permission_error() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/models/slp-orchestra-mini/download", 401, json!({"error": "unauthorized"}));

        let client = client_with_auth(server.base_url(), &[]);
        // Reaches an end user in the app, so it says what they can do about it
        // rather than naming an env var they have never set.
        let err = client.get_model_artifacts("slp-orchestra-mini").unwrap_err();
        assert_permission(&err, "Sign in");
    }

    #[test]
    #[serial]
    fn test_404_raises_permission_error() {
        let _c = autouse();
        let server = StubServer::start();

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.get_model_artifacts("nonexistent").unwrap_err();
        assert_permission(&err, "not found");
    }

    /// 403 is "not on your plan", not "your token is bad" — the two lead to
    /// different UI (upgrade vs. sign in), so they must not collapse into one
    /// exception.
    #[test]
    #[serial]
    fn test_403_raises_model_access_error() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/models/gemma3-12b/download", 403, json!({"detail": "forbidden"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.get_model_artifacts("gemma3-12b").unwrap_err();
        assert!(matches!(err, PlatformError::ModelAccess(_)), "{err:?}");
        assert!(err.message().contains("not available on your current plan"));
    }

    /// A free model is served to an anonymous caller; the client must not
    /// require credentials to ask.
    #[test]
    #[serial]
    fn test_free_model_downloads_without_auth_headers() {
        let _c = autouse();
        let server = StubServer::start();
        server.route(
            "/v1/models/gemma3-4b/download",
            200,
            json!({"artifacts": {"model.gguf": "https://storage/model.gguf"}}),
        );

        let client = client_with_auth(server.base_url(), &[]);
        assert_eq!(
            client.get_model_artifacts("gemma3-4b").unwrap(),
            artifacts(&[("model.gguf", "https://storage/model.gguf")])
        );
    }

    /// Rate limiting is transient, so it must not be typed as a plan problem.
    ///
    /// The download route is rate-limited per IP, which anonymous downloads now
    /// reach.  Classing a 429 as a PermissionError would tell the user to
    /// upgrade when what they need is to wait.
    #[test]
    #[serial]
    fn test_429_raises_connection_error() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/models/slp-orchestra-mini/download", 429, json!({"detail": "rate limited"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.get_model_artifacts("slp-orchestra-mini").unwrap_err();
        assert!(err.is_connection(), "{err:?}");
        assert!(!err.is_permission());
        assert_eq!(err.status_code(), Some(429));
    }

    #[test]
    #[serial]
    fn test_server_error_raises_connection_error() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/models/slp-orchestra-mini/download", 500, json!({"detail": "boom"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.get_model_artifacts("slp-orchestra-mini").unwrap_err();
        assert!(err.is_connection(), "{err:?}");
        assert!(!err.is_permission());
        assert_eq!(err.status_code(), Some(500));
    }

    /// Callers classify by status, not by matching message text.
    ///
    /// /v1/init reads this back off the error to choose its own HTTP status,
    /// and a message is a bad channel for that: it is written for a person and
    /// it gets reworded.
    #[test]
    #[serial]
    fn test_errors_carry_the_upstream_status() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/models/gemma3-12b/download", 403, json!({"detail": "upgrade"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.get_model_artifacts("gemma3-12b").unwrap_err();
        assert!(matches!(err, PlatformError::ModelAccess(_)), "{err:?}");
        assert_eq!(err.status_code(), Some(403));
    }
}

mod test_get_latest_release {
    use super::*;

    #[test]
    #[serial]
    fn test_returns_wrapper() {
        let _c = autouse();
        let server = StubServer::start();
        let wrapper = json!({
            "update_available": null,
            "update_required": false,
            "release": {"version": "1.0.15", "name": "v", "channel": "stable"},
        });
        server.route("/v1/releases/latest", 200, wrapper.clone());

        let client = client_with_auth(server.base_url(), &[]);
        assert_eq!(client.get_latest_release("stable").unwrap(), Some(obj(wrapper)));
        assert_eq!(server.requests()[0].target, "/v1/releases/latest?channel=stable");
    }

    #[test]
    #[serial]
    fn test_null_release_returned() {
        let _c = autouse();
        let server = StubServer::start();
        let wrapper = json!({"update_available": null, "update_required": false, "release": null});
        server.route("/v1/releases/latest", 200, wrapper);

        let client = client_with_auth(server.base_url(), &[]);
        assert_eq!(client.get_latest_release("beta").unwrap().unwrap()["release"], Value::Null);
    }

    #[test]
    #[serial]
    fn test_error_raises_connection_error() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/releases/latest", 500, json!({"detail": "boom"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.get_latest_release("stable").unwrap_err();
        assert_connection(&err, "API error");
    }

    #[test]
    #[serial]
    fn test_unreachable_raises_connection_error() {
        let _c = autouse();
        let client = client_with_auth(UNREACHABLE, &[]);
        assert!(client.get_latest_release("stable").unwrap_err().is_connection());
    }
}

/// 401/403 must stay distinguishable from every other failure: the first means
/// search is off for the account, the rest are transient.
mod test_web_search {
    use super::*;

    #[test]
    #[serial]
    fn test_returns_links() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/websearch", 200, json!(["http://a", "http://b"]));

        let client = client_with_auth(server.base_url(), &[("Authorization", "Bearer jwt")]);
        assert_eq!(client.web_search("bikes", 5, SEARCH_TIMEOUT).unwrap(), vec!["http://a", "http://b"]);
        assert_eq!(server.requests()[0].target, "/v1/websearch?q=bikes&count=5");
    }

    #[test]
    #[serial]
    fn test_drops_non_string_and_blank_entries() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/websearch", 200, json!(["http://a", "", null, 3, "  "]));

        let client = client_with_auth(server.base_url(), &[]);
        assert_eq!(client.web_search("bikes", 5, SEARCH_TIMEOUT).unwrap(), vec!["http://a"]);
    }

    #[test]
    #[serial]
    fn test_non_list_body_returns_empty() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/websearch", 200, json!({"results": []}));

        let client = client_with_auth(server.base_url(), &[]);
        assert_eq!(client.web_search("bikes", 5, SEARCH_TIMEOUT).unwrap(), Vec::<String>::new());
    }

    fn check_auth_failure_raises_permission_error(status: u16) {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/websearch", status, json!({"detail": "nope"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.web_search("bikes", 5, SEARCH_TIMEOUT).unwrap_err();
        assert_permission(&err, &status.to_string());
    }

    #[test]
    #[serial]
    fn test_auth_failure_raises_permission_error_401() {
        check_auth_failure_raises_permission_error(401);
    }

    #[test]
    #[serial]
    fn test_auth_failure_raises_permission_error_403() {
        check_auth_failure_raises_permission_error(403);
    }

    /// 429 means the month's searches are used up — stop searching, don't
    /// retry it as if the endpoint had blipped.
    #[test]
    #[serial]
    fn test_spent_allowance_raises_permission_error_not_connection_error() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/websearch", 429, json!({"detail": "limit reached"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.web_search("bikes", 5, SEARCH_TIMEOUT).unwrap_err();
        assert_permission(&err, "429");
    }

    #[test]
    #[serial]
    fn test_server_error_raises_connection_error() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/websearch", 500, json!({"detail": "boom"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.web_search("bikes", 5, SEARCH_TIMEOUT).unwrap_err();
        assert_connection(&err, "API error");
    }

    #[test]
    #[serial]
    fn test_unreachable_raises_connection_error() {
        let _c = autouse();
        let client = client_with_auth(UNREACHABLE, &[]);
        assert!(client.web_search("bikes", 5, SEARCH_TIMEOUT).unwrap_err().is_connection());
    }
}

mod test_web_search_status {
    use super::*;

    #[test]
    #[serial]
    fn test_enabled() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/websearch/status", 200, json!({"enabled": true}));

        let client = client_with_auth(server.base_url(), &[]);
        assert!(client.web_search_status(SEARCH_STATUS_TIMEOUT).unwrap());
    }

    #[test]
    #[serial]
    fn test_disabled() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/websearch/status", 200, json!({"enabled": false}));

        let client = client_with_auth(server.base_url(), &[]);
        assert!(!client.web_search_status(SEARCH_STATUS_TIMEOUT).unwrap());
    }

    fn check_auth_failure_raises_permission_error(status: u16) {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/websearch/status", status, json!({"detail": "nope"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.web_search_status(SEARCH_STATUS_TIMEOUT).unwrap_err();
        assert_permission(&err, &status.to_string());
    }

    #[test]
    #[serial]
    fn test_auth_failure_raises_permission_error_401() {
        check_auth_failure_raises_permission_error(401);
    }

    #[test]
    #[serial]
    fn test_auth_failure_raises_permission_error_403() {
        check_auth_failure_raises_permission_error(403);
    }

    #[test]
    #[serial]
    fn test_detail_returns_the_whole_body() {
        let _c = autouse();
        let server = StubServer::start();
        let body = json!({
            "enabled": true,
            "provider": "serper",
            "reason": null,
            "quota": {"limit": 5, "used": 2, "remaining": 3, "period": "2026-10",
                      "resets_at": "2026-11-01T00:00:00Z", "plan": "free"}
        });
        server.route("/v1/websearch/status", 200, body.clone());

        let client = client_with_auth(server.base_url(), &[]);
        let detail = client.web_search_status_detail(SEARCH_STATUS_TIMEOUT).unwrap();
        assert_eq!(serde_json::Value::Object(detail), body);
    }

    #[test]
    #[serial]
    fn test_detail_auth_failure_raises_permission_error() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/websearch/status", 401, json!({"detail": "nope"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.web_search_status_detail(SEARCH_STATUS_TIMEOUT).unwrap_err();
        assert_permission(&err, "401");
    }

    #[test]
    #[serial]
    fn test_server_error_raises_connection_error() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/websearch/status", 500, json!({"detail": "boom"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.web_search_status(SEARCH_STATUS_TIMEOUT).unwrap_err();
        assert_connection(&err, "API error");
    }
}

mod test_cloud_usage {
    use super::*;

    #[test]
    #[serial]
    fn test_quota_returns_the_whole_body() {
        let _c = autouse();
        let server = StubServer::start();
        let body = json!({
            "limit": null, "used": null, "remaining": null, "percent_used": 42.5,
            "percent_available": 50.0, "in_flight": 1, "requests": 12,
            "prompt_tokens": 9000, "completion_tokens": 3000, "period": "2026-10",
            "resets_at": "2026-11-01T00:00:00Z", "plan": "paid"
        });
        server.route("/v1/chat/quota", 200, body.clone());

        let client = client_with_auth(server.base_url(), &[]);
        let quota = client.cloud_quota(CLOUD_USAGE_TIMEOUT).unwrap();
        assert_eq!(Value::Object(quota), body);
    }

    fn check_refusal_keeps_its_status(status: u16) {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/chat/quota", status, json!({"detail": "nope"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.cloud_quota(CLOUD_USAGE_TIMEOUT).unwrap_err();
        assert!(err.is_permission());
        assert_eq!(err.status_code(), Some(status));
    }

    #[test]
    #[serial]
    fn test_signed_out_is_permission_401() {
        check_refusal_keeps_its_status(401);
    }

    #[test]
    #[serial]
    fn test_no_cloud_access_is_permission_403() {
        check_refusal_keeps_its_status(403);
    }

    #[test]
    #[serial]
    fn test_server_error_raises_connection_error() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/chat/quota", 500, json!({"detail": "boom"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.cloud_quota(CLOUD_USAGE_TIMEOUT).unwrap_err();
        assert_connection(&err, "API error");
    }
}

mod test_request_headers {
    use super::*;

    /// The account access log would otherwise show the HTTP library for every
    /// call — including the unauthenticated ones.
    #[test]
    #[serial]
    fn test_user_agent_is_sent_with_and_without_auth() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/users/me", 200, json!({"email": "a@b.c"}));
        server.route("/v1/bootstrap/models", 200, json!([{"name": "sl-mini"}]));

        let client = client_with_auth(server.base_url(), &[("Authorization", "Bearer jwt")]);
        client.get_current_user().unwrap();
        client.list_bootstrap_models().unwrap();

        let seen: Vec<_> = server.requests().into_iter().filter(|r| r.method == "GET").collect();
        assert_eq!(seen.len(), 2);
        assert!(seen.iter().all(|h| h.header("User-Agent").unwrap_or("").starts_with("smartloop/")));
        assert_eq!(seen[0].header("Authorization"), Some("Bearer jwt"));
        assert!(seen[1].header("Authorization").is_none(), "bootstrap is deliberately unauthenticated");
    }
}

mod test_get_current_user {
    use super::*;

    #[test]
    #[serial]
    fn test_returns_user_profile() {
        let _c = autouse();
        let server = StubServer::start();
        let user_data = json!({"email": "chris@example.com", "name": "Chris"});
        server.route("/v1/users/me", 200, user_data.clone());

        let client = client_with_auth(server.base_url(), &[("Authorization", "Bearer jwt")]);
        let result = client.get_current_user().unwrap();
        assert_eq!(result, obj(user_data));
    }

    #[test]
    #[serial]
    fn test_401_raises_permission_error() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/users/me", 401, json!({"detail": "Unauthorized"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.get_current_user().unwrap_err();
        assert_permission(&err, "expired or invalid");
    }

    /// A 500 is the platform having a bad moment, not a bad credential.
    ///
    /// It used to arrive as `PermissionError` — the endpoint's error class was
    /// applied to every non-2xx — which sent it down the expired-token path:
    /// the cache-through service refuses to serve stale data for it, and the
    /// refresh agent clears both tokens on it.  A platform blip logged the
    /// user out.
    #[test]
    #[serial]
    fn test_a_server_error_is_transient_not_a_verdict() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/users/me", 500, json!({"detail": "Internal Server Error"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.get_current_user().unwrap_err();
        assert_connection(&err, "API error");
        assert!(!err.is_permission());
        assert_eq!(err.status_code(), Some(500));
    }

    #[test]
    #[serial]
    fn test_a_rate_limit_is_transient_too() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/users/me", 429, json!({"detail": "Too Many Requests"}));

        let client = client_with_auth(server.base_url(), &[]);
        assert!(client.get_current_user().unwrap_err().is_connection());
    }
}

/// Every authenticated call redeems the refresh token once on a 401.
///
/// This used to live at a single call site — `_resolve_user`, behind
/// `/v1/auth/status` — so a model download answered "Sign in to download this
/// model" and web search quietly reported itself off, both while a working
/// refresh token sat on disk.  It belongs to the client, where the 401
/// actually arrives and every endpoint gets it.
mod test_refresh_on_401 {
    use super::*;

    /// A client whose refresh swaps the token the next request will send;
    /// each refresh records the token it replaced in *refreshes*.
    fn client(base_url: &str, refreshes: &Arc<Mutex<Vec<String>>>) -> SLPPlatformClient {
        let state = Arc::new(Mutex::new("expired".to_string()));
        let auth_state = state.clone();
        let auth = Arc::new(move || headers(&[("Authorization", &format!("Bearer {}", auth_state.lock()))]));
        let refreshes = refreshes.clone();
        let refresh = Arc::new(move || {
            let mut token = state.lock();
            refreshes.lock().push(token.clone());
            *token = "fresh".to_string();
            true
        });
        SLPPlatformClient::with_hooks(base_url, Some(auth), Some(refresh))
    }

    #[test]
    #[serial]
    fn test_a_401_is_retried_with_the_refreshed_token() {
        let _c = autouse();
        let server = StubServer::start();
        let user = json!({"email": "chris@example.com", "name": "Chris"});
        server.route_seq("/v1/users/me", vec![(401, json!({"detail": "Unauthorized"})), (200, user.clone())]);
        let refreshes = Arc::new(Mutex::new(Vec::new()));

        let client = client(server.base_url(), &refreshes);
        assert_eq!(client.get_current_user().unwrap(), obj(user));

        assert_eq!(refreshes.lock().len(), 1);
        let requests = server.requests();
        assert_eq!(requests.len(), 2);
        // The retry re-read auth_headers_fn rather than reusing the prepared
        // headers — otherwise it would resend the token that just failed.
        assert_eq!(requests[0].header("Authorization"), Some("Bearer expired"));
        assert_eq!(requests[1].header("Authorization"), Some("Bearer fresh"));
    }

    /// A plan refusal is about the account, not the credential.
    ///
    /// `ModelAccessError` is a permission error, so keying the retry on the
    /// error type would re-mint the same identity and ask again for a model
    /// the plan still does not grant.  Keyed on the status, it cannot.
    #[test]
    #[serial]
    fn test_a_403_is_not_a_stale_token() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/models/gemma3-12b/download", 403, json!({"detail": "forbidden"}));
        let refreshes = Arc::new(Mutex::new(Vec::new()));

        let client = client(server.base_url(), &refreshes);
        let err = client.get_model_artifacts("gemma3-12b").unwrap_err();
        assert!(matches!(err, PlatformError::ModelAccess(_)), "{err:?}");

        assert!(refreshes.lock().is_empty());
        assert_eq!(server.requests().len(), 1);
    }

    #[test]
    #[serial]
    fn test_nothing_to_refresh_with_means_no_retry() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/models/sl-mini/download", 401, json!({"detail": "Unauthorized"}));

        let client =
            SLPPlatformClient::with_hooks(server.base_url(), Some(Arc::new(Headers::new)), Some(Arc::new(|| false)));
        let err = client.get_model_artifacts("sl-mini").unwrap_err();
        assert_permission(&err, "Sign in");

        assert_eq!(server.requests().len(), 1);
    }

    /// A token the platform keeps refusing must not become a request loop.
    #[test]
    #[serial]
    fn test_the_retry_happens_once() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/users/me", 401, json!({"detail": "Unauthorized"}));
        let refreshes = Arc::new(Mutex::new(Vec::new()));

        let client = client(server.base_url(), &refreshes);
        assert!(client.get_current_user().unwrap_err().is_permission());

        assert_eq!(refreshes.lock().len(), 1);
        assert_eq!(server.requests().len(), 2);
    }

    /// The default hook redeems against this client, not `AppSettings`.
    ///
    /// Resolving the base URL from settings instead means a client built for
    /// any other host sends its refresh token somewhere it does not belong.
    #[test]
    #[serial]
    fn test_the_refresh_goes_to_the_host_that_refused() {
        let _c = autouse();
        let server = StubServer::start();
        server.route_seq(
            "/v1/users/me",
            vec![(401, json!({"detail": "Unauthorized"})), (200, json!({"email": "c@e.com"}))],
        );
        server.route("/v1/auth/jwt/refresh", 200, json!({"access_token": "fresh"}));

        credential_store::set_access_token("expired");
        credential_store::set_refresh_token("rt");
        let client = SLPPlatformClient::new(server.base_url()); // no refresh_fn — the default hook
        assert_eq!(client.get_current_user().unwrap(), obj(json!({"email": "c@e.com"})));

        let paths: Vec<_> = server.requests().into_iter().map(|r| r.path).collect();
        assert_eq!(paths, vec!["/v1/users/me", "/v1/auth/jwt/refresh", "/v1/users/me"]);
        assert_eq!(credential_store::get_access_token().as_deref(), Some("fresh"));
    }

    /// `list_models` degrades to anonymous, but only as a last resort.
    ///
    /// Going straight to the anonymous retry left a signed-in user looking at
    /// a free-tier catalog — every entry `is_accessible` false — with no
    /// signal that their token had merely expired.
    #[test]
    #[serial]
    fn test_the_catalog_refreshes_before_falling_back_to_anonymous() {
        let _c = autouse();
        let server = StubServer::start();
        let models = json!([{"name": "sl-mini"}]);
        server.route_seq("/v1/models", vec![(401, json!({"detail": "Unauthorized"})), (200, models.clone())]);
        let refreshes = Arc::new(Mutex::new(Vec::new()));

        let client = client(server.base_url(), &refreshes);
        assert_eq!(client.list_models().unwrap(), objects(models));

        assert_eq!(refreshes.lock().len(), 1);
        assert_eq!(server.requests()[1].header("Authorization"), Some("Bearer fresh"));
    }
}

/// The refresh grant itself — the one call that must never be misread.
///
/// Its failures are acted on destructively: `refresh_access_token_now` clears
/// both credentials on a permission error.
mod test_refresh_access_token {
    use super::*;

    const PATH: &str = "/v1/auth/jwt/refresh";

    #[test]
    #[serial]
    fn test_it_posts_the_token_unauthenticated() {
        let _c = autouse();
        let server = StubServer::start();
        server.route(PATH, 200, json!({"access_token": "new", "refresh_token": "rotated"}));

        let client = client_with_auth(server.base_url(), &[("Authorization", "Bearer dead")]);
        let data = client.refresh_access_token("old-rt").unwrap();

        assert_eq!(data, obj(json!({"access_token": "new", "refresh_token": "rotated"})));
        let sent = &server.requests()[0];
        assert_eq!(sent.method, "POST");
        assert_eq!(sent.json(), json!({"refresh_token": "old-rt"}));
        // Sending the dead access token alongside would invite the endpoint to
        // judge it rather than the refresh token.
        assert!(sent.header("Authorization").is_none());
    }

    #[test]
    #[serial]
    fn test_a_401_is_a_dead_refresh_token() {
        let _c = autouse();
        let server = StubServer::start();
        server.route(PATH, 401, json!({"detail": "Unauthorized"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.refresh_access_token("old-rt").unwrap_err();
        assert_permission(&err, "expired or revoked");
    }

    /// The regression this whole review started from.
    ///
    /// Every non-2xx took the endpoint's `PermissionError`, and the refresh
    /// agent clears *both* tokens on that — so one bad minute on the platform
    /// logged every user out permanently, with no way back but a re-login.
    fn check_a_transient_failure_is_not_a_dead_refresh_token(status: u16) {
        let _c = autouse();
        let server = StubServer::start();
        server.route(PATH, status, json!({"detail": "nope"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.refresh_access_token("old-rt").unwrap_err();
        assert!(err.is_connection(), "{err:?}");
        assert!(!err.is_permission());
    }

    #[test]
    #[serial]
    fn test_a_transient_failure_is_not_a_dead_refresh_token_429() {
        check_a_transient_failure_is_not_a_dead_refresh_token(429);
    }

    #[test]
    #[serial]
    fn test_a_transient_failure_is_not_a_dead_refresh_token_500() {
        check_a_transient_failure_is_not_a_dead_refresh_token(500);
    }

    #[test]
    #[serial]
    fn test_a_transient_failure_is_not_a_dead_refresh_token_502() {
        check_a_transient_failure_is_not_a_dead_refresh_token(502);
    }

    #[test]
    #[serial]
    fn test_a_transient_failure_is_not_a_dead_refresh_token_503() {
        check_a_transient_failure_is_not_a_dead_refresh_token(503);
    }

    #[test]
    #[serial]
    fn test_a_reply_without_an_access_token_raises() {
        let _c = autouse();
        let server = StubServer::start();
        server.route(PATH, 200, json!({"token_type": "Bearer"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.refresh_access_token("old-rt").unwrap_err();
        assert_value_error(&err, "missing access_token");
    }
}

/// `authed` retries when the credentials *changed*, not when the refresh
/// reported success.  The two come apart in both directions.
mod test_the_retry_follows_the_credential {
    use super::*;

    /// Redeeming a dead refresh token clears both OAuth credentials — and
    /// leaves a working developer token behind.
    ///
    /// Keyed on the refresh's return value that read as failure, so the
    /// request was abandoned with "Sign in to download this model" for a user
    /// who was signed in, and the identical next request succeeded.
    #[test]
    #[serial]
    fn test_a_dead_oauth_token_falls_back_to_the_developer_token() {
        let _c = autouse();
        let server = StubServer::start();
        server.route_seq(
            "/v1/models/sl-mini/download",
            vec![
                (401, json!({"detail": "Unauthorized"})),
                (200, json!({"artifacts": {"m.gguf": "https://blob/m.gguf"}})),
            ],
        );
        server.route("/v1/auth/jwt/refresh", 401, json!({"detail": "revoked"}));
        credential_store::set_access_token("expired");
        credential_store::set_refresh_token("revoked-rt");
        credential_store::set_token("sk-dev-good");

        let client = SLPPlatformClient::new(server.base_url());
        assert_eq!(client.get_model_artifacts("sl-mini").unwrap(), artifacts(&[("m.gguf", "https://blob/m.gguf")]));

        assert_eq!(credential_store::get_access_token(), None);
        assert_eq!(credential_store::get_refresh_token(), None);
        let sent: Vec<_> = server.requests().into_iter().filter(|r| r.path.ends_with("/download")).collect();
        assert_eq!(sent[0].header("Authorization"), Some("Bearer expired"));
        assert_eq!(sent[1].header("X-Api-Key"), Some("sk-dev-good"));
    }

    /// A refresh that reports success without changing what we send would
    /// only buy a guaranteed second 401.
    #[test]
    #[serial]
    fn test_an_unchanged_credential_is_not_resent() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/users/me", 401, json!({"detail": "Unauthorized"}));

        let client = SLPPlatformClient::with_hooks(
            server.base_url(),
            Some(Arc::new(|| headers(&[("Authorization", "Bearer same")]))),
            Some(Arc::new(|| true)),
        );
        assert!(client.get_current_user().unwrap_err().is_permission());

        assert_eq!(server.requests().len(), 1);
    }
}

mod test_telemetry_cannot_log_you_out {
    use super::*;

    /// `push_token_usage` is fire-and-forget and swallows every failure.
    ///
    /// Routed through `authed` it could redeem the refresh token, and a
    /// refresh endpoint answering 401 clears both credentials — so a telemetry
    /// POST could log the user out with nothing visible but a log line.
    /// Records flush in batches, so it would do it once per queued record.
    #[test]
    #[serial]
    fn test_a_401_on_usage_does_not_redeem_the_refresh_token() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/usage/tokens", 401, json!({"detail": "Unauthorized"}));
        credential_store::set_access_token("expired");
        credential_store::set_refresh_token("rt");

        SLPPlatformClient::new(server.base_url())
            .push_token_usage(&obj(json!({"tokens": 1})), smartloop::platform::client::PUSH_TOKEN_USAGE_TIMEOUT);

        assert!(!server.requests().iter().any(|r| r.path.contains("refresh")));
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("rt"));
    }
}

/// A pinned process answers its own 401 rather than spending the token.
///
/// `authed` asks its `refresh_fn` and then compares credentials.  In an agent
/// child the hook declines — redeeming a single-use refresh token is the
/// workspace owner's job — so the comparison finds nothing changed and the
/// original 401 stands.  What must not happen is the child reaching the
/// refresh endpoint at all.
mod test_a_project_agent_does_not_redeem {
    use super::*;

    #[test]
    #[serial]
    fn test_a_401_in_a_pinned_process_is_returned_as_is() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/users/me", 401, json!({"detail": "Unauthorized"}));
        credential_store::set_access_token("expired");
        credential_store::set_refresh_token("rt");
        pin_project(Some("alpha"));

        let client = SLPPlatformClient::new(server.base_url());
        assert!(client.get_current_user().unwrap_err().is_permission());

        let requests = server.requests();
        assert_eq!(requests.len(), 1, "the 401 was retried");
        assert!(!requests.iter().any(|r| r.path.contains("refresh")));
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("rt"));
    }

    /// Declining still recovers, because nothing is cached.
    ///
    /// `auth_headers_fn` here stands in for the credential file: the second
    /// read sees what the supervisor wrote during the round trip.  `authed`
    /// re-reads after the 401, finds the headers changed, and retries with the
    /// new token — without this process redeeming anything.
    #[test]
    #[serial]
    fn test_a_pinned_process_picks_up_a_token_the_owner_rotated() {
        let _c = autouse();
        let server = StubServer::start();
        let user = json!({"email": "chris@example.com", "name": "Chris"});
        server.route_seq("/v1/users/me", vec![(401, json!({"detail": "Unauthorized"})), (200, user.clone())]);
        pin_project(Some("alpha"));

        let reads = Arc::new(AtomicUsize::new(0));
        let auth = {
            let reads = reads.clone();
            Arc::new(move || {
                let n = reads.fetch_add(1, Ordering::SeqCst) + 1;
                let token = if n == 1 { "expired" } else { "rotated-by-the-owner" };
                headers(&[("Authorization", &format!("Bearer {token}"))])
            })
        };

        let client = SLPPlatformClient::with_hooks(server.base_url(), Some(auth), None);
        assert_eq!(client.get_current_user().unwrap(), obj(user));

        let requests = server.requests();
        let sent: Vec<_> = requests.iter().map(|r| r.header("Authorization").map(str::to_string)).collect();
        assert_eq!(
            sent,
            vec![Some("Bearer expired".to_string()), Some("Bearer rotated-by-the-owner".to_string())]
        );
        assert!(!requests.iter().any(|r| r.path.contains("refresh")));
    }

    /// The chat path is how a child reaches the platform most often.
    ///
    /// A 401 here becomes "search is off for this account", which is the right
    /// answer from inside a child: it cannot tell a stale token from a plan
    /// without search.  What it must not do is spend the refresh token to
    /// find out.
    #[test]
    #[serial]
    fn test_web_search_in_a_child_reports_off_without_redeeming() {
        let _c = autouse();
        let server = StubServer::start();
        server.route("/v1/websearch/status", 401, json!({"detail": "Unauthorized"}));
        credential_store::set_access_token("expired");
        credential_store::set_refresh_token("rt");
        pin_project(Some("alpha"));

        let err = SLPPlatformClient::new(server.base_url()).web_search_status(SEARCH_STATUS_TIMEOUT).unwrap_err();
        assert!(err.is_permission());

        assert!(!server.requests().iter().any(|r| r.path.contains("refresh")));
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("rt"));
    }
}

/// The platform's `detail` is what the desktop shows the user, so each refusal
/// keeps it — and its status — rather than being reworded here.
mod test_verify_apple_transaction {
    use super::*;

    const PATH: &str = "/v1/payments/apple/verify";

    #[test]
    #[serial]
    fn test_posts_the_transaction_and_returns_the_answer() {
        let _c = autouse();
        let server = StubServer::start();
        let body = json!({"is_paid": true, "subscription_status": "active", "current_period_end": null});
        server.route(PATH, 200, body.clone());

        let client = client_with_auth(server.base_url(), &[("Authorization", "Bearer jwt")]);
        let result = client.verify_apple_transaction("2000000123").unwrap();

        assert_eq!(result, obj(body));
        let sent = &server.requests()[0];
        assert_eq!(sent.method, "POST");
        assert_eq!(sent.json(), json!({"transaction_id": "2000000123"}));
        assert_eq!(sent.header("Authorization"), Some("Bearer jwt"));
    }

    #[test]
    #[serial]
    fn test_401_raises_permission_error() {
        let _c = autouse();
        let server = StubServer::start();
        server.route(PATH, 401, json!({"detail": "Unauthorized"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.verify_apple_transaction("tx").unwrap_err();
        assert_permission(&err, "expired or invalid");
        assert_eq!(err.status_code(), Some(401));
    }

    fn check_refusal_keeps_detail(status: u16, detail: &str) {
        let _c = autouse();
        let server = StubServer::start();
        server.route(PATH, status, json!({"detail": detail}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.verify_apple_transaction("tx").unwrap_err();
        assert_permission(&err, detail);
        assert_eq!(err.message(), detail);
        assert_eq!(err.status_code(), Some(status));
    }

    #[test]
    #[serial]
    fn test_400_invalid_transaction_keeps_detail() {
        check_refusal_keeps_detail(400, "Invalid transaction");
    }

    #[test]
    #[serial]
    fn test_409_already_linked_keeps_detail() {
        check_refusal_keeps_detail(409, "Transaction is already linked to another account");
    }

    /// Not configured is the platform's state, not a verdict about this
    /// caller, so it stays transient — with its detail.
    #[test]
    #[serial]
    fn test_503_is_a_connection_error_with_detail() {
        let _c = autouse();
        let server = StubServer::start();
        server.route(PATH, 503, json!({"detail": "Apple IAP is not configured"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.verify_apple_transaction("tx").unwrap_err();
        assert_connection(&err, "Apple IAP is not configured");
        assert_eq!(err.status_code(), Some(503));
    }

    #[test]
    #[serial]
    fn test_a_refusal_without_detail_falls_back_to_the_reason_phrase() {
        let _c = autouse();
        let server = StubServer::start();
        server.route(PATH, 409, json!({"detail": [{"msg": "not a string"}]}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.verify_apple_transaction("tx").unwrap_err();
        assert_permission(&err, "API error (409): Conflict");
    }

    #[test]
    #[serial]
    fn test_unreachable_raises_connection_error() {
        let _c = autouse();
        let client = client_with_auth(UNREACHABLE, &[]);
        let err = client.verify_apple_transaction("tx").unwrap_err();
        assert_connection(&err, "Connection error");
        assert_eq!(err.status_code(), None);
    }
}

mod test_create_checkout_session {
    use super::*;

    const PATH: &str = "/v1/payments/create-checkout-session";

    #[test]
    #[serial]
    fn test_posts_and_returns_the_checkout_url() {
        let _c = autouse();
        let server = StubServer::start();
        let body = json!({"checkout_url": "https://checkout.stripe.com/c/pay/cs_test"});
        server.route(PATH, 200, body.clone());

        let client = client_with_auth(server.base_url(), &[("Authorization", "Bearer jwt")]);
        assert_eq!(client.create_checkout_session().unwrap(), obj(body));
        let sent = &server.requests()[0];
        assert_eq!(sent.method, "POST");
        assert_eq!(sent.header("Authorization"), Some("Bearer jwt"));
        // Google's load balancer answers a bodiless POST with 411.
        assert_eq!(sent.json(), json!({}));
        assert_eq!(sent.header("Content-Length"), Some("2"));
    }

    #[test]
    #[serial]
    fn test_404_stripe_disabled_is_a_permission_error_with_status() {
        let _c = autouse();
        let server = StubServer::start();
        server.route(PATH, 404, json!({"detail": "Not Found"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.create_checkout_session().unwrap_err();
        assert_permission(&err, "Not Found");
        assert_eq!(err.status_code(), Some(404));
    }

    #[test]
    #[serial]
    fn test_400_stripe_error_keeps_detail() {
        let _c = autouse();
        let server = StubServer::start();
        server.route(PATH, 400, json!({"detail": "Stripe error: No such price"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.create_checkout_session().unwrap_err();
        assert_eq!(err.message(), "Stripe error: No such price");
        assert_eq!(err.status_code(), Some(400));
    }

    #[test]
    #[serial]
    fn test_401_raises_permission_error() {
        let _c = autouse();
        let server = StubServer::start();
        server.route(PATH, 401, json!({"detail": "Unauthorized"}));

        let client = client_with_auth(server.base_url(), &[]);
        let err = client.create_checkout_session().unwrap_err();
        assert_permission(&err, "expired or invalid");
    }
}
