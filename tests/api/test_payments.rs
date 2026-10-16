//! Tests for the `/v1/payments` API router.

mod test_verify_apple {
    use serde_json::{json, Value};
    use serial_test::serial;
    use smartloop::platform::PlatformError;

    use crate::common::core_support::{obj, platform_service, post};

    const PATH: &str = "/v1/payments/apple/verify";

    fn verified() -> Value {
        json!({
            "is_paid": true,
            "subscription_status": "active",
            "current_period_end": "2026-11-16T00:00:00Z",
        })
    }

    #[tokio::test]
    #[serial]
    async fn test_relays_the_platform_answer() {
        let platform_service = platform_service();
        platform_service.set_verify_apple_transaction(Ok(obj(verified())));
        let resp = post(PATH, json!({"transaction_id": "2000000123456789"})).await;

        assert_eq!(resp.status_code(), 200);
        assert_eq!(resp.json(), &verified());
        let calls = platform_service.calls_to("verify_apple_transaction");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].args, vec![json!("2000000123456789")]);
    }

    /// Pro models unlock without a restart: the cached profile (`is_paid`) and
    /// catalog (`is_accessible`) are re-fetched past their TTL.
    #[tokio::test]
    #[serial]
    async fn test_success_refreshes_the_cached_user_and_catalog() {
        let platform_service = platform_service();
        platform_service.set_verify_apple_transaction(Ok(obj(verified())));
        platform_service.set_current_user(Ok(obj(json!({"email": "chris@example.com", "is_paid": true}))));
        platform_service.set_list_models(Ok(vec![]));
        let resp = post(PATH, json!({"transaction_id": "tx"})).await;

        assert_eq!(resp.status_code(), 200);
        assert_eq!(platform_service.calls_to("get_current_user").len(), 1);
        assert_eq!(platform_service.calls_to("list_models").len(), 1);
    }

    /// The refresh is best-effort: the purchase went through, so a failed
    /// refetch must not turn it into an error.
    #[tokio::test]
    #[serial]
    async fn test_a_failed_refresh_does_not_fail_the_purchase() {
        let platform_service = platform_service();
        platform_service.set_verify_apple_transaction(Ok(obj(verified())));
        platform_service.set_current_user(Err(PlatformError::connection("Connection refused")));
        platform_service.set_list_models(Err(PlatformError::connection("Connection refused")));
        let resp = post(PATH, json!({"transaction_id": "tx"})).await;

        assert_eq!(resp.status_code(), 200);
        assert_eq!(resp.json()["is_paid"], true);
    }

    #[tokio::test]
    #[serial]
    async fn test_a_refusal_does_not_refresh_anything() {
        let platform_service = platform_service();
        platform_service.set_verify_apple_transaction(Err(PlatformError::Permission {
            message: "Invalid transaction".into(),
            status_code: Some(400),
        }));
        post(PATH, json!({"transaction_id": "tx"})).await;

        assert!(platform_service.calls_to("get_current_user").is_empty());
        assert!(platform_service.calls_to("list_models").is_empty());
    }

    #[tokio::test]
    #[serial]
    async fn test_401_when_the_token_is_rejected() {
        let platform_service = platform_service();
        platform_service.set_verify_apple_transaction(Err(PlatformError::Permission {
            message: "Authentication failed. Token may be expired or invalid.".into(),
            status_code: Some(401),
        }));
        let resp = post(PATH, json!({"transaction_id": "tx"})).await;

        assert_eq!(resp.status_code(), 401);
        assert_eq!(resp.json()["detail"], "Not authenticated");
    }

    #[tokio::test]
    #[serial]
    async fn test_401_when_not_signed_in() {
        let platform_service = platform_service();
        platform_service.set_verify_apple_transaction(Err(PlatformError::permission("No credentials")));
        let resp = post(PATH, json!({"transaction_id": "tx"})).await;

        assert_eq!(resp.status_code(), 401);
        assert_eq!(resp.json()["detail"], "Not authenticated");
    }

    async fn check_status_and_detail_pass_through(error: PlatformError, status: u16, detail: &str) {
        let platform_service = platform_service();
        platform_service.set_verify_apple_transaction(Err(error));
        let resp = post(PATH, json!({"transaction_id": "tx"})).await;

        assert_eq!(resp.status_code(), status);
        assert_eq!(resp.json()["detail"], detail);
    }

    #[tokio::test]
    #[serial]
    async fn test_400_invalid_transaction_keeps_the_platform_detail() {
        let detail = "Transaction could not be verified with Apple";
        let error = PlatformError::Permission { message: detail.into(), status_code: Some(400) };
        check_status_and_detail_pass_through(error, 400, detail).await;
    }

    #[tokio::test]
    #[serial]
    async fn test_409_already_linked_keeps_the_platform_detail() {
        let detail = "This purchase is already linked to another Smartloop account";
        let error = PlatformError::Permission { message: detail.into(), status_code: Some(409) };
        check_status_and_detail_pass_through(error, 409, detail).await;
    }

    #[tokio::test]
    #[serial]
    async fn test_503_not_configured_keeps_the_platform_detail() {
        let detail = "Apple In-App Purchase is not configured";
        let error = PlatformError::Connection { message: detail.into(), status_code: Some(503) };
        check_status_and_detail_pass_through(error, 503, detail).await;
    }

    #[tokio::test]
    #[serial]
    async fn test_502_when_platform_unreachable() {
        let error = PlatformError::connection("Connection error: refused");
        check_status_and_detail_pass_through(error, 502, "Platform API unavailable").await;
    }

    #[tokio::test]
    #[serial]
    async fn test_502_on_another_server_error() {
        let error = PlatformError::Connection { message: "API error (500)".into(), status_code: Some(500) };
        check_status_and_detail_pass_through(error, 502, "Platform API unavailable").await;
    }

    #[tokio::test]
    #[serial]
    async fn test_422_when_transaction_id_is_blank() {
        let platform_service = platform_service();
        let resp = post(PATH, json!({"transaction_id": "  "})).await;

        assert_eq!(resp.status_code(), 422);
        assert!(platform_service.calls_to("verify_apple_transaction").is_empty());
    }

    #[tokio::test]
    #[serial]
    async fn test_422_when_transaction_id_is_missing() {
        let platform_service = platform_service();
        let resp = post(PATH, json!({})).await;

        assert_eq!(resp.status_code(), 422);
        assert!(platform_service.calls_to("verify_apple_transaction").is_empty());
    }
}

mod test_checkout {
    use serde_json::json;
    use serial_test::serial;
    use smartloop::platform::PlatformError;

    use crate::common::core_support::{obj, platform_service, post};

    const PATH: &str = "/v1/payments/checkout";

    #[tokio::test]
    #[serial]
    async fn test_relays_the_checkout_url() {
        let platform_service = platform_service();
        let body = json!({"checkout_url": "https://checkout.stripe.com/c/pay/cs_test_123"});
        platform_service.set_checkout_session(Ok(obj(body.clone())));
        let resp = post(PATH, json!({})).await;

        assert_eq!(resp.status_code(), 200);
        assert_eq!(resp.json(), &body);
        assert_eq!(platform_service.calls_to("create_checkout_session").len(), 1);
    }

    async fn check(error: PlatformError, status: u16, detail: &str) {
        let platform_service = platform_service();
        platform_service.set_checkout_session(Err(error));
        let resp = post(PATH, json!({})).await;

        assert_eq!(resp.status_code(), status);
        assert_eq!(resp.json()["detail"], detail);
    }

    /// The desktop hides the upgrade option on a 404, so it must stay a 404.
    #[tokio::test]
    #[serial]
    async fn test_404_when_stripe_is_disabled() {
        let error = PlatformError::Permission { message: "Not Found".into(), status_code: Some(404) };
        check(error, 404, "Not Found").await;
    }

    #[tokio::test]
    #[serial]
    async fn test_400_stripe_error_keeps_the_platform_detail() {
        let detail = "Stripe error: No such price";
        let error = PlatformError::Permission { message: detail.into(), status_code: Some(400) };
        check(error, 400, detail).await;
    }

    #[tokio::test]
    #[serial]
    async fn test_401_when_the_token_is_rejected() {
        let error = PlatformError::Permission { message: "expired".into(), status_code: Some(401) };
        check(error, 401, "Not authenticated").await;
    }

    #[tokio::test]
    #[serial]
    async fn test_502_when_platform_unreachable() {
        check(PlatformError::connection("Connection error: refused"), 502, "Platform API unavailable").await;
    }
}

mod test_refresh {
    use std::sync::Arc;

    use serde_json::{json, Value};
    use serial_test::serial;
    use smartloop::platform::{PlatformError, WorkspaceDB};

    use crate::common::core_support::{obj, platform_service, platform_service_with_db, post, PlatformFixture};

    const PATH: &str = "/v1/payments/refresh";

    fn user(is_paid: bool) -> Value {
        json!({"email": "chris@example.com", "name": "Chris", "is_paid": is_paid})
    }

    /// A cache-backed fixture whose cached profile says `cached_paid`, fresh
    /// enough that `GET /v1/users/me` would serve it without asking.
    fn with_cached_user(cached_paid: bool) -> (tempfile::TempDir, PlatformFixture) {
        let dir = tempfile::tempdir().expect("temp dir for the workspace DB");
        let db = WorkspaceDB::new(dir.path().join("workspace.db")).expect("workspace DB");
        db.save_user(&obj(user(cached_paid))).expect("seed the user cache");
        (dir, platform_service_with_db(Some(Arc::new(db))))
    }

    /// The point of the route: the cache is fresh and says unpaid, the cloud
    /// says paid, and the cloud wins — then the cache agrees.
    #[tokio::test]
    #[serial]
    async fn test_returns_the_cloud_profile_not_the_cache() {
        let (_dir, fixture) = with_cached_user(false);
        fixture.set_current_user(Ok(obj(user(true))));
        fixture.set_list_models(Ok(vec![]));
        let resp = post(PATH, json!({})).await;

        assert_eq!(resp.status_code(), 200);
        let data = resp.json();
        assert_eq!(data["email"], "chris@example.com");
        assert_eq!(data["is_paid"], true);
        assert_eq!(data["default_temperature"], 0.3);
        assert_eq!(fixture.calls_to("get_current_user").len(), 1);
        let cached = fixture.service.db().unwrap().get_user().unwrap().unwrap();
        assert_eq!(cached["is_paid"], true);
    }

    #[tokio::test]
    #[serial]
    async fn test_a_plan_change_refreshes_the_catalog() {
        let (_dir, fixture) = with_cached_user(false);
        fixture.set_current_user(Ok(obj(user(true))));
        fixture.set_list_models(Ok(vec![]));
        post(PATH, json!({})).await;

        assert_eq!(fixture.calls_to("list_models").len(), 1);
    }

    /// Polled every few seconds: an unchanged plan costs one call, not two.
    #[tokio::test]
    #[serial]
    async fn test_an_unchanged_plan_skips_the_catalog() {
        let (_dir, fixture) = with_cached_user(false);
        fixture.set_current_user(Ok(obj(user(false))));
        let resp = post(PATH, json!({})).await;

        assert_eq!(resp.status_code(), 200);
        assert_eq!(resp.json()["is_paid"], false);
        assert!(fixture.calls_to("list_models").is_empty());
    }

    #[tokio::test]
    #[serial]
    async fn test_a_failed_catalog_refresh_still_returns_the_profile() {
        let (_dir, fixture) = with_cached_user(false);
        fixture.set_current_user(Ok(obj(user(true))));
        fixture.set_list_models(Err(PlatformError::connection("Connection refused")));
        let resp = post(PATH, json!({})).await;

        assert_eq!(resp.status_code(), 200);
        assert_eq!(resp.json()["is_paid"], true);
    }

    #[tokio::test]
    #[serial]
    async fn test_401_when_not_signed_in() {
        let platform_service = platform_service();
        platform_service.set_current_user(Err(PlatformError::permission("Not authenticated")));
        let resp = post(PATH, json!({})).await;

        assert_eq!(resp.status_code(), 401);
        assert_eq!(resp.json()["detail"], "Not authenticated");
    }

    /// Never the stale cache: an offline poll must not report the old plan as
    /// if it were the answer.
    #[tokio::test]
    #[serial]
    async fn test_502_when_unreachable_even_with_a_cache() {
        let (_dir, fixture) = with_cached_user(false);
        fixture.set_current_user(Err(PlatformError::connection("Connection refused")));
        let resp = post(PATH, json!({})).await;

        assert_eq!(resp.status_code(), 502);
        assert_eq!(resp.json()["detail"], "Platform API unavailable");
    }
}
