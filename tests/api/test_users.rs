//! Port of `tests/api/test_users.py`.
//!
//! Tests for the `/v1/users` API router.

mod test_get_me {
    use serde_json::json;
    use serial_test::serial;
    use smartloop::platform::PlatformError;

    use crate::common::core_support::{get, obj, platform_service};

    #[tokio::test]
    #[serial]
    async fn test_returns_full_profile() {
        let platform_service = platform_service();
        platform_service.set_current_user(Ok(obj(json!({
            "email": "chris@example.com",
            "name": "Chris",
            "is_premium": true,
            "is_paid": true,
            "number_of_tokens_processed": 5000,
            "default_temperature": 0.5,
            "default_memory": true,
        }))));
        let resp = get("/v1/users/me").await;

        assert_eq!(resp.status_code(), 200);
        let data = resp.json();
        assert_eq!(data["email"], "chris@example.com");
        assert_eq!(data["name"], "Chris");
        assert_eq!(data["is_premium"], true);
        assert_eq!(data["is_paid"], true);
        assert_eq!(data["number_of_tokens_processed"], 5000);
        assert_eq!(data["default_temperature"], 0.5);
        assert!(data.get("selected_model").is_none());
    }

    #[tokio::test]
    #[serial]
    async fn test_optional_fields_default() {
        let platform_service = platform_service();
        platform_service.set_current_user(Ok(obj(json!({"email": "chris@example.com"}))));
        let resp = get("/v1/users/me").await;

        assert_eq!(resp.status_code(), 200);
        let data = resp.json();
        assert_eq!(data["email"], "chris@example.com");
        // `data["name"] is None`: present and null, not absent.
        assert_eq!(data.get("name"), Some(&serde_json::Value::Null));
        assert_eq!(data["is_premium"], false);
        assert_eq!(data["is_paid"], false);
        assert_eq!(data["number_of_tokens_processed"], 0);
        assert_eq!(data["default_temperature"], 0.3);
        assert_eq!(data["default_memory"], true);
    }

    #[tokio::test]
    #[serial]
    async fn test_401_when_not_authenticated() {
        let platform_service = platform_service();
        platform_service.set_current_user(Err(PlatformError::permission("Not authenticated")));
        let resp = get("/v1/users/me").await;

        assert_eq!(resp.status_code(), 401);
        assert_eq!(resp.json()["detail"], "Not authenticated");
    }

    #[tokio::test]
    #[serial]
    async fn test_502_when_platform_unavailable() {
        let platform_service = platform_service();
        platform_service.set_current_user(Err(PlatformError::connection("Connection refused")));
        let resp = get("/v1/users/me").await;

        assert_eq!(resp.status_code(), 502);
        assert_eq!(resp.json()["detail"], "Platform API unavailable");
    }
}

mod test_update_me {
    use serde_json::json;
    use serial_test::serial;
    use smartloop::platform::PlatformError;

    use crate::common::core_support::{obj, patch, platform_service};

    #[tokio::test]
    #[serial]
    async fn test_updates_default_temperature() {
        let platform_service = platform_service();
        platform_service.set_update_current_user(Ok(obj(json!({
            "email": "chris@example.com",
            "name": "Chris",
            "default_temperature": 0.9,
        }))));
        let resp = patch("/v1/users/me", json!({"default_temperature": 0.9})).await;

        assert_eq!(resp.status_code(), 200);
        assert_eq!(resp.json()["default_temperature"], 0.9);
    }

    #[tokio::test]
    #[serial]
    async fn test_422_when_no_fields() {
        // The `client` fixture depends on `platform_service`, so the override
        // is installed here too.
        let _platform_service = platform_service();
        let resp = patch("/v1/users/me", json!({})).await;
        assert_eq!(resp.status_code(), 422);
    }

    #[tokio::test]
    #[serial]
    async fn test_401_when_not_authenticated() {
        let platform_service = platform_service();
        platform_service.set_update_current_user(Err(PlatformError::permission("Not authenticated")));
        let resp = patch("/v1/users/me", json!({"default_temperature": 0.4})).await;

        assert_eq!(resp.status_code(), 401);
    }
}

mod test_delete_me {
    use axum::http::Method;
    use serial_test::serial;
    use smartloop::auth::credential_store;
    use smartloop::platform::PlatformError;

    use crate::common::core_support::{platform_service, send};

    #[tokio::test]
    #[serial]
    async fn test_deletes_and_signs_out() {
        let platform_service = platform_service();
        platform_service.set_delete_current_user(Ok(()));
        // Signing out re-reads the catalog anonymously.
        platform_service.set_list_models(Ok(vec![]));
        credential_store::set_tokens("access-token", Some("refresh-token"));
        credential_store::set_token("developer-token");

        let resp = send(Method::DELETE, "/v1/users/me", None).await;

        assert_eq!(resp.status_code(), 204);
        assert_eq!(platform_service.calls_to("delete_current_user").len(), 1);
        // Every credential belonged to the deleted account.
        assert_eq!(credential_store::get_access_token(), None);
        assert_eq!(credential_store::get_refresh_token(), None);
        assert_eq!(credential_store::get_token(), None);
    }

    #[tokio::test]
    #[serial]
    async fn test_keeps_credentials_when_platform_fails() {
        let platform_service = platform_service();
        platform_service.set_delete_current_user(Err(PlatformError::connection("Connection error")));
        credential_store::set_tokens("access-token", None);

        let resp = send(Method::DELETE, "/v1/users/me", None).await;

        assert_eq!(resp.status_code(), 502);
        // The account still exists, so the session must too.
        assert_eq!(credential_store::get_access_token().as_deref(), Some("access-token"));
    }

    #[tokio::test]
    #[serial]
    async fn test_401_when_not_authenticated() {
        let platform_service = platform_service();
        platform_service.set_delete_current_user(Err(PlatformError::permission("Not authenticated")));
        let resp = send(Method::DELETE, "/v1/users/me", None).await;

        assert_eq!(resp.status_code(), 401);
    }
}
