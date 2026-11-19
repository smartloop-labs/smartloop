//! Port of `tests/api/test_auth.py`.
//!
//! Tests for the /v1/auth API router.

use std::sync::Arc;

use serde_json::{json, Value};
use serial_test::serial;

use smartloop::api::routers::auth::{
    self, AuthPlatformService, UserInfo,
};
use smartloop::auth::credential_store;
use smartloop::platform::client::{JsonObject, PlatformResult};
use smartloop::platform::PlatformError;

use crate::common::misc_client::{delete, get, post_json, with_query};
use crate::shared::misc_isolation::IsolatedHome;

/// A platform service standing in for the module-level `get_platform_service`
/// (the Python `MagicMock`): `get_current_user` answers with *get_user*,
/// `list_models` with an empty catalog.
struct FakePlatform {
    get_user: Box<dyn Fn() -> PlatformResult<JsonObject> + Send + Sync>,
}

impl FakePlatform {
    fn answering(get_user: impl Fn() -> PlatformResult<JsonObject> + Send + Sync + 'static) -> Arc<Self> {
        Arc::new(FakePlatform { get_user: Box::new(get_user) })
    }

    /// The `MagicMock()` default: every call succeeds with nothing useful.
    fn quiet() -> Arc<Self> {
        Self::answering(|| Ok(JsonObject::new()))
    }
}

impl AuthPlatformService for FakePlatform {
    fn get_current_user(&self, _force: bool) -> PlatformResult<JsonObject> {
        (self.get_user)()
    }

    fn list_models(&self, _force: bool) -> PlatformResult<Vec<JsonObject>> {
        Ok(Vec::new())
    }
}

/// The autouse fixtures of the Python module: an isolated credential store,
/// `_no_network` (the router's platform service is a fake) and
/// `_no_platform_call` (`_resolve_user` answers `None`).
struct AuthFixture {
    _home: IsolatedHome,
}

impl AuthFixture {
    fn new() -> Self {
        let home = IsolatedHome::new();
        auth::set_platform_service_override(Some(FakePlatform::quiet()));
        no_platform_call_start();
        AuthFixture { _home: home }
    }
}

impl Drop for AuthFixture {
    fn drop(&mut self) {
        auth::set_resolve_user_override(None);
        auth::set_platform_service_override(None);
    }
}

/// `_no_platform_call.start()`: `_resolve_user` patched to return `None`.
fn no_platform_call_start() {
    auth::set_resolve_user_override(Some(Arc::new(|| None)));
}

/// `_no_platform_call.stop()`.
fn no_platform_call_stop() {
    auth::set_resolve_user_override(None);
}

/// A platform 401 as the client actually raises it, status attached.
fn unauthorized() -> PlatformError {
    PlatformError::Permission { message: "Authentication failed".to_string(), status_code: Some(401) }
}

fn client() -> axum::Router {
    auth::router()
}

mod test_auth_status {
    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_no_credentials() {
        let _fx = AuthFixture::new();
        let resp = get(&client(), "/v1/auth/status").await;
        assert_eq!(resp.status, 200);
        let data = resp.json();
        assert_eq!(data["authenticated"], json!(false));
        assert_eq!(data["method"], Value::Null);
        assert_eq!(data["has_developer_token"], json!(false));
        assert_eq!(data["has_access_token"], json!(false));
        assert_eq!(data["user"], Value::Null);
    }

    #[tokio::test]
    #[serial]
    async fn test_developer_token_only() {
        let _fx = AuthFixture::new();
        credential_store::set_token("sk-dev-123");
        let data = get(&client(), "/v1/auth/status").await.json();
        assert_eq!(data["authenticated"], json!(true));
        assert_eq!(data["method"], json!("developer_token"));
        assert_eq!(data["has_developer_token"], json!(true));
        assert_eq!(data["has_access_token"], json!(false));
    }

    #[tokio::test]
    #[serial]
    async fn test_access_token_only() {
        let _fx = AuthFixture::new();
        credential_store::set_access_token("jwt-abc");
        let data = get(&client(), "/v1/auth/status").await.json();
        assert_eq!(data["authenticated"], json!(true));
        assert_eq!(data["method"], json!("access_token"));
        assert_eq!(data["has_developer_token"], json!(false));
        assert_eq!(data["has_access_token"], json!(true));
    }

    #[tokio::test]
    #[serial]
    async fn test_both_tokens_prefers_access() {
        let _fx = AuthFixture::new();
        credential_store::set_token("sk-dev-123");
        credential_store::set_access_token("jwt-abc");
        let data = get(&client(), "/v1/auth/status").await.json();
        assert_eq!(data["method"], json!("access_token"));
        assert_eq!(data["has_developer_token"], json!(true));
        assert_eq!(data["has_access_token"], json!(true));
    }

    #[tokio::test]
    #[serial]
    async fn test_status_includes_user_info() {
        let _fx = AuthFixture::new();
        credential_store::set_access_token("jwt-abc");
        let user = UserInfo { email: "chris@example.com".to_string(), name: Some("Chris".to_string()) };
        auth::set_resolve_user_override(Some(Arc::new(move || Some(user.clone()))));
        let data = get(&client(), "/v1/auth/status").await.json();
        no_platform_call_start();
        assert_eq!(data["user"], json!({"email": "chris@example.com", "name": "Chris"}));
    }

    #[tokio::test]
    #[serial]
    async fn test_status_user_null_on_platform_failure() {
        let _fx = AuthFixture::new();
        credential_store::set_access_token("jwt-abc");
        let data = get(&client(), "/v1/auth/status").await.json();
        assert_eq!(data["authenticated"], json!(true));
        assert_eq!(data["user"], Value::Null);
    }
}

/// What `_resolve_user` still owns after the refresh moved into the client.
///
/// Redeeming the refresh token used to happen here, which is why this was the
/// only path in the app that recovered from an expired token.  It now happens
/// in `SLPPlatformClient::authed`, so a permission error arriving here has
/// already survived a refresh attempt.  What is left is the disposal: drop the
/// dead access token so later requests fall back to the developer token, and
/// leave a *transient* failure alone.
mod test_expired_access_token {
    use super::*;

    /// Stop the `_no_platform_call` patch so the real `_resolve_user` runs,
    /// with the platform service answering `get_current_user` with *get_user*.
    ///
    /// Patched at the *service* boundary, not the client: the service caches
    /// the user with a TTL, so patching the client would leave a warm cache to
    /// satisfy later calls.
    async fn status(get_user: impl Fn() -> PlatformResult<JsonObject> + Send + Sync + 'static) -> Value {
        no_platform_call_stop();
        auth::set_platform_service_override(Some(FakePlatform::answering(get_user)));
        let data = get(&client(), "/v1/auth/status").await.json();
        auth::set_platform_service_override(Some(FakePlatform::quiet()));
        no_platform_call_start();
        data
    }

    #[tokio::test]
    #[serial]
    async fn test_a_dead_access_token_is_dropped_for_the_developer_token() {
        let _fx = AuthFixture::new();
        credential_store::set_access_token("expired-jwt");
        credential_store::set_token("sk-dev-valid");

        let data = status(|| Err(unauthorized())).await;

        assert_eq!(credential_store::get_access_token(), None);
        assert_eq!(data["authenticated"], json!(true));
        assert_eq!(data["method"], json!("developer_token"));
    }

    /// Only the refresh *endpoint* gets to decide the refresh token is dead.
    ///
    /// A 401 from `/v1/users/me` says nothing about it, and the refresh agent
    /// has already cleared it if its own call was refused.
    #[tokio::test]
    #[serial]
    async fn test_the_refresh_token_is_not_collateral() {
        let _fx = AuthFixture::new();
        credential_store::set_access_token("expired-jwt");
        credential_store::set_refresh_token("still-good-rt");

        status(|| Err(unauthorized())).await;

        assert_eq!(credential_store::get_access_token(), None);
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("still-good-rt"));
    }

    /// 403 and 404 arrive here as permission errors too — they are not the
    /// token's fault, and an API deploy that moves a route should not log
    /// everyone out of the desktop client.
    #[tokio::test]
    #[serial]
    async fn test_only_a_401_discards_the_token() {
        let _fx = AuthFixture::new();
        for status_code in [403u16, 404] {
            credential_store::set_access_token("good-jwt");
            let data = status(move || {
                Err(PlatformError::Permission {
                    message: format!("API error ({status_code})"),
                    status_code: Some(status_code),
                })
            })
            .await;

            assert_eq!(credential_store::get_access_token().as_deref(), Some("good-jwt"), "{status_code}");
            assert_eq!(data["user"], Value::Null);
        }
    }

    /// A platform blip arrives as a connection error, not a permission error.
    ///
    /// It used to arrive as the latter — every non-2xx took the endpoint's
    /// error class — and this path would then throw away a perfectly good
    /// access token because the platform had a bad minute.
    #[tokio::test]
    #[serial]
    async fn test_an_outage_is_not_a_logout() {
        let _fx = AuthFixture::new();
        credential_store::set_access_token("good-jwt");

        let data = status(|| Err(PlatformError::connection("platform unreachable"))).await;

        assert_eq!(credential_store::get_access_token().as_deref(), Some("good-jwt"));
        assert_eq!(data["authenticated"], json!(true));
        assert_eq!(data["user"], Value::Null);
    }
}

mod test_set_token {
    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_set_developer_token_default() {
        let _fx = AuthFixture::new();
        let resp = post_json(&client(), "/v1/auth/token", &json!({"token": "sk-dev-123"})).await;
        assert_eq!(resp.status, 200);
        assert_eq!(resp.json()["has_developer_token"], json!(true));
        assert_eq!(credential_store::get_token().as_deref(), Some("sk-dev-123"));
    }

    #[tokio::test]
    #[serial]
    async fn test_set_access_token() {
        let _fx = AuthFixture::new();
        let resp = post_json(
            &client(),
            "/v1/auth/token",
            &json!({"token": "jwt-abc", "type": "access_token"}),
        )
        .await;
        assert_eq!(resp.status, 200);
        assert_eq!(resp.json()["has_access_token"], json!(true));
        assert_eq!(credential_store::get_access_token().as_deref(), Some("jwt-abc"));
    }

    #[tokio::test]
    #[serial]
    async fn test_missing_token_returns_422() {
        let _fx = AuthFixture::new();
        let resp = post_json(&client(), "/v1/auth/token", &json!({})).await;
        assert_eq!(resp.status, 422);
    }
}

mod test_clear_token {
    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_clear_all_default() {
        let _fx = AuthFixture::new();
        credential_store::set_token("sk-dev-123");
        credential_store::set_access_token("jwt-abc");
        credential_store::set_refresh_token("rt-xyz");

        let resp = delete(&client(), "/v1/auth/token").await;
        assert_eq!(resp.status, 200);
        let data = resp.json();
        assert_eq!(data["authenticated"], json!(false));
        assert_eq!(credential_store::get_token(), None);
        assert_eq!(credential_store::get_access_token(), None);
        assert_eq!(credential_store::get_refresh_token(), None);
    }

    #[tokio::test]
    #[serial]
    async fn test_clear_developer_only() {
        let _fx = AuthFixture::new();
        credential_store::set_token("sk-dev-123");
        credential_store::set_access_token("jwt-abc");

        let resp = delete(&client(), "/v1/auth/token?type=developer_token").await;
        assert_eq!(resp.status, 200);
        let data = resp.json();
        assert_eq!(data["has_developer_token"], json!(false));
        assert_eq!(data["has_access_token"], json!(true));
    }

    #[tokio::test]
    #[serial]
    async fn test_clear_access_only() {
        let _fx = AuthFixture::new();
        credential_store::set_token("sk-dev-123");
        credential_store::set_access_token("jwt-abc");
        credential_store::set_refresh_token("rt-xyz");

        let resp = delete(&client(), "/v1/auth/token?type=access_token").await;
        assert_eq!(resp.status, 200);
        let data = resp.json();
        assert_eq!(data["has_developer_token"], json!(true));
        assert_eq!(data["has_access_token"], json!(false));
        assert_eq!(credential_store::get_refresh_token(), None);
    }
}

mod test_oauth_callback {
    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_callback_with_access_token_only() {
        let _fx = AuthFixture::new();
        let resp = post_json(&client(), "/v1/auth/callback", &json!({"access_token": "jwt-from-webapp"})).await;
        assert_eq!(resp.status, 200);
        let data = resp.json();
        assert_eq!(data["has_access_token"], json!(true));
        assert_eq!(credential_store::get_access_token().as_deref(), Some("jwt-from-webapp"));
        assert_eq!(credential_store::get_refresh_token(), None);
    }

    #[tokio::test]
    #[serial]
    async fn test_callback_with_refresh_token() {
        let _fx = AuthFixture::new();
        let resp = post_json(
            &client(),
            "/v1/auth/callback",
            &json!({"access_token": "jwt-from-webapp", "refresh_token": "rt-from-webapp"}),
        )
        .await;
        assert_eq!(resp.status, 200);
        assert_eq!(credential_store::get_access_token().as_deref(), Some("jwt-from-webapp"));
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("rt-from-webapp"));
    }

    #[tokio::test]
    #[serial]
    async fn test_callback_missing_access_token_returns_422() {
        let _fx = AuthFixture::new();
        let resp = post_json(&client(), "/v1/auth/callback", &json!({})).await;
        assert_eq!(resp.status, 422);
    }

    #[tokio::test]
    #[serial]
    async fn test_callback_overwrites_existing() {
        let _fx = AuthFixture::new();
        credential_store::set_access_token("old-jwt");
        let resp = post_json(&client(), "/v1/auth/callback", &json!({"access_token": "new-jwt"})).await;
        assert_eq!(resp.status, 200);
        assert_eq!(credential_store::get_access_token().as_deref(), Some("new-jwt"));
    }

    /// Safari blocks the HTTPS→HTTP fetch, so the webapp navigates instead.
    ///
    /// Untested until now, which is uncomfortable for the one login path a
    /// whole browser depends on.
    #[tokio::test]
    #[serial]
    async fn test_the_safari_fallback_stores_both_and_redirects() {
        let _fx = AuthFixture::new();
        let uri = with_query(
            "/v1/auth/callback",
            &[("access_token", "jwt-via-get"), ("refresh_token", "rt-via-get")],
        );
        let resp = get(&client(), &uri).await;

        assert_eq!(resp.status, 307);
        assert!(resp.header("location").unwrap_or_default().contains("mode=desktop"));
        assert_eq!(credential_store::get_access_token().as_deref(), Some("jwt-via-get"));
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("rt-via-get"));
    }

    #[tokio::test]
    #[serial]
    async fn test_the_safari_fallback_needs_an_access_token() {
        let _fx = AuthFixture::new();
        let resp = get(&client(), "/v1/auth/callback").await;
        assert_eq!(resp.status, 422);
    }
}

/// Whether hosted search will serve is cached for a minute, per account.
///
/// Signing in with that cache holding the logged-out "no" left web search
/// skipping itself for the first minute of the session — the orchestrator asks
/// the gate before it runs the query, so the whole step disappeared.
/// `reset_status_cache` existed for exactly this and nothing called it.
mod test_signing_in_clears_the_search_gate {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use smartloop::search::providers::{self, WebSearchCallError, WebSearchClient};

    use super::*;

    /// A search client whose status endpoint answers *status*, counting calls.
    struct StatusClient {
        status: Result<bool, String>,
        calls: Arc<AtomicUsize>,
    }

    impl WebSearchClient for StatusClient {
        fn web_search(&self, _query: &str, _count: usize) -> Result<Vec<String>, WebSearchCallError> {
            Ok(Vec::new())
        }

        fn web_search_status(&self) -> Result<bool, WebSearchCallError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.status.clone().map_err(WebSearchCallError::Permission)
        }
    }

    fn install_status(status: Result<bool, String>) -> Arc<AtomicUsize> {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        providers::set_platform_client_factory(Some(Arc::new(move || {
            Box::new(StatusClient { status: status.clone(), calls: counter.clone() }) as Box<dyn WebSearchClient>
        })));
        calls
    }

    /// Leave the gate's cache holding a "no".
    fn gate_says_no() {
        providers::reset_status_cache();
        install_status(Err("not signed in".to_string()));
        credential_store::set_token("sk-dev"); // so the gate gets past has_credentials
        assert!(!providers::hosted_search_available(None));
        providers::set_platform_client_factory(None);
        credential_store::clear_token();
    }

    /// `providers._status_cache is None`, observed through the gate: with a
    /// signed-in account and a status endpoint that now says yes, an emptied
    /// cache asks the endpoint (and answers yes), while a surviving "no" would
    /// answer no without asking.
    fn status_cache_is_empty() -> bool {
        let calls = install_status(Ok(true));
        providers::set_credentials_check(Some(Arc::new(|| true)));
        let available = providers::hosted_search_available(None);
        providers::set_credentials_check(None);
        providers::set_platform_client_factory(None);
        providers::reset_status_cache();
        available && calls.load(Ordering::SeqCst) == 1
    }

    #[tokio::test]
    #[serial]
    async fn test_a_login_reopens_the_gate() {
        let _fx = AuthFixture::new();
        gate_says_no();

        post_json(&client(), "/v1/auth/callback", &json!({"access_token": "jwt-fresh"})).await;

        assert!(status_cache_is_empty(), "the stale 'search is off' survived the login");
    }

    #[tokio::test]
    #[serial]
    async fn test_a_logout_reopens_it_too() {
        let _fx = AuthFixture::new();
        credential_store::set_access_token("jwt");
        gate_says_no();

        delete(&client(), "/v1/auth/token").await;

        assert!(status_cache_is_empty());
    }
}

mod test_device_login {
    use super::*;
    use std::collections::VecDeque;
    use std::time::{Duration, Instant};

    use parking_lot::Mutex;
    use smartloop::platform::client::DevicePoll;

    /// The platform's side of a device code sign-in: the code it hands out,
    /// then one scripted answer per poll (pending once the script runs out).
    struct DeviceFake {
        code: PlatformResult<JsonObject>,
        polls: Mutex<VecDeque<PlatformResult<DevicePoll>>>,
    }

    impl DeviceFake {
        fn install(polls: Vec<PlatformResult<DevicePoll>>) -> Arc<Self> {
            Self::install_with(Ok(code_reply()), polls)
        }

        fn install_with(code: PlatformResult<JsonObject>, polls: Vec<PlatformResult<DevicePoll>>) -> Arc<Self> {
            let fake = Arc::new(DeviceFake { code, polls: Mutex::new(polls.into()) });
            auth::set_platform_service_override(Some(fake.clone()));
            fake
        }

        fn then(&self, poll: PlatformResult<DevicePoll>) {
            self.polls.lock().push_back(poll);
        }
    }

    impl AuthPlatformService for DeviceFake {
        fn get_current_user(&self, _force: bool) -> PlatformResult<JsonObject> {
            Ok(JsonObject::new())
        }

        fn list_models(&self, _force: bool) -> PlatformResult<Vec<JsonObject>> {
            Ok(Vec::new())
        }

        fn request_device_code(&self, _client_id: &str) -> PlatformResult<JsonObject> {
            self.code.clone()
        }

        fn poll_device_token(&self, _device_code: &str, _client_id: &str) -> PlatformResult<DevicePoll> {
            // The fake answers at once; keep a pending poller from spinning.
            std::thread::sleep(Duration::from_millis(5));
            self.polls.lock().pop_front().unwrap_or(Ok(DevicePoll::Pending))
        }
    }

    /// What the platform answers a device code request with, polling at once.
    fn code_reply() -> JsonObject {
        let value = json!({
            "device_code": "secret-device-code",
            "user_code": "BCDF-GHJK",
            "verification_uri": "https://app.smartloop.ai/device",
            "verification_uri_complete": "https://app.smartloop.ai/device?code=BCDF-GHJK",
            "expires_in": 600,
            "interval": 0,
        });
        value.as_object().unwrap().clone()
    }

    fn tokens() -> DevicePoll {
        let value = json!({"access_token": "jwt-from-device", "refresh_token": "rt-from-device"});
        DevicePoll::Tokens(value.as_object().unwrap().clone())
    }

    /// The device sign-in fixture: auth isolation, and no poller outliving the test.
    struct DeviceFixture {
        _auth: AuthFixture,
    }

    impl DeviceFixture {
        fn new() -> Self {
            DeviceFixture { _auth: AuthFixture::new() }
        }
    }

    impl Drop for DeviceFixture {
        fn drop(&mut self) {
            auth::cancel_device_login();
        }
    }

    async fn start() -> Value {
        let resp = post_json(&client(), "/v1/auth/device", &json!({"client_id": "docent-code"})).await;
        assert_eq!(resp.status, 200);
        resp.json().clone()
    }

    /// `GET /v1/auth/device` once it stops being pending.
    async fn settled() -> Value {
        let started = Instant::now();
        loop {
            let data = get(&client(), "/v1/auth/device").await.json().clone();
            if data["status"] != json!("pending") || started.elapsed() > Duration::from_secs(5) {
                return data;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    #[tokio::test]
    #[serial]
    async fn test_approval_stores_the_tokens() {
        let _fx = DeviceFixture::new();
        DeviceFake::install(vec![Ok(DevicePoll::Pending), Ok(tokens())]);

        let data = start().await;
        assert_eq!(data["status"], json!("pending"));
        assert_eq!(data["user_code"], json!("BCDF-GHJK"));
        assert_eq!(data["verification_uri_complete"], json!("https://app.smartloop.ai/device?code=BCDF-GHJK"));
        assert!(!data.to_string().contains("secret-device-code"), "the device code stays in the agent");

        assert_eq!(settled().await["status"], json!("approved"));
        assert_eq!(credential_store::get_access_token().as_deref(), Some("jwt-from-device"));
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("rt-from-device"));
    }

    #[tokio::test]
    #[serial]
    async fn test_a_platform_blip_keeps_polling() {
        let _fx = DeviceFixture::new();
        DeviceFake::install(vec![Err(PlatformError::connection("down")), Ok(tokens())]);

        start().await;
        assert_eq!(settled().await["status"], json!("approved"));
        assert_eq!(credential_store::get_access_token().as_deref(), Some("jwt-from-device"));
    }

    #[tokio::test]
    #[serial]
    async fn test_denied_and_expired_store_nothing() {
        let _fx = DeviceFixture::new();

        DeviceFake::install(vec![Ok(DevicePoll::Denied)]);
        start().await;
        assert_eq!(settled().await["status"], json!("denied"));

        DeviceFake::install(vec![Ok(DevicePoll::Expired)]);
        start().await;
        assert_eq!(settled().await["status"], json!("expired"));

        assert_eq!(credential_store::get_access_token(), None);
    }

    #[tokio::test]
    #[serial]
    async fn test_a_refused_poll_fails_with_the_reason() {
        let _fx = DeviceFixture::new();
        DeviceFake::install(vec![Err(PlatformError::Permission {
            message: "Unknown device_code.".to_string(),
            status_code: Some(400),
        })]);

        start().await;
        let data = settled().await;
        assert_eq!(data["status"], json!("failed"));
        assert_eq!(data["message"], json!("Unknown device_code."));
    }

    #[tokio::test]
    #[serial]
    async fn test_an_unknown_client_is_refused() {
        let _fx = DeviceFixture::new();
        DeviceFake::install_with(
            Err(PlatformError::Permission { message: "Unknown client_id.".to_string(), status_code: Some(401) }),
            vec![],
        );

        let resp = post_json(&client(), "/v1/auth/device", &json!({"client_id": "nobody"})).await;
        assert_eq!(resp.status, 400);
        assert_eq!(get(&client(), "/v1/auth/device").await.json()["status"], json!("none"));
    }

    #[tokio::test]
    #[serial]
    async fn test_signing_out_cancels_a_pending_sign_in() {
        let _fx = DeviceFixture::new();
        let fake = DeviceFake::install(vec![]);

        start().await;
        delete(&client(), "/v1/auth/token").await;
        assert_eq!(get(&client(), "/v1/auth/device").await.json()["status"], json!("none"));

        // Approved after the sign-out: the cancelled poller must not store it.
        fake.then(Ok(tokens()));
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(credential_store::get_access_token(), None);
    }

    #[tokio::test]
    #[serial]
    async fn test_missing_client_id_returns_422() {
        let _fx = DeviceFixture::new();
        let resp = post_json(&client(), "/v1/auth/device", &json!({})).await;
        assert_eq!(resp.status, 422);
    }
}
