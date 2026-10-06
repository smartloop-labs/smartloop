//! Port of `tests/api/test_mcp_catalog.py`.
//!
//! Tests for the `/v1/mcp-catalog` API router.
//!
//! The router reads through `PlatformService`, so the catalog stays available
//! when the platform API is down.  These drive the shared `platform_service`
//! override rather than the raw client.

mod test_list_mcp_catalog {
    use serde_json::{json, Value};
    use serial_test::serial;
    use smartloop::platform::PlatformError;

    use crate::common::core_support::{get, obj, platform_service};

    fn entries(value: Value) -> Vec<smartloop::platform::client::JsonObject> {
        value.as_array().expect("a JSON array").iter().cloned().map(obj).collect()
    }

    #[tokio::test]
    #[serial]
    async fn test_returns_catalog_list() {
        let platform_service = platform_service();
        let mock_data = json!([
            {
                "slug": "linear",
                "display_name": "Linear",
                "description": "Project management",
                "server_type": "remote",
                "hosted_by": "external",
                "remote_config": {
                    "server_url": "https://mcp.linear.app/sse",
                    "auth_type": "oauth",
                },
                "icon_url": "https://linear.app/favicon.ico",
                "category": "Project Management",
                "tags": ["project-management"],
                "sort_order": 0,
                "catalog_version": 1,
            },
            {
                "slug": "yahoo-finance",
                "display_name": "Yahoo Finance",
                "description": "Stock data",
                "server_type": "local",
                "hosted_by": "local",
                "local_config": {
                    "command": "uvx",
                    "args": ["yahoo-finance-mcp"],
                    "env": {},
                },
                "category": "Finance",
                "tags": ["finance"],
                "sort_order": 10,
                "catalog_version": 1,
            },
        ]);
        platform_service.set_list_mcp_catalog(Ok(entries(mock_data)));
        let resp = get("/v1/mcp-catalog").await;

        assert_eq!(resp.status_code(), 200);
        let data = resp.json().as_array().expect("a JSON array");
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["slug"], "linear");
        assert_eq!(data[0]["server_type"], "remote");
        assert_eq!(data[0]["hosted_by"], "external");
        assert_eq!(data[0]["remote_config"]["server_url"], "https://mcp.linear.app/sse");
        assert_eq!(data[1]["slug"], "yahoo-finance");
        assert_eq!(data[1]["hosted_by"], "local");
        assert_eq!(data[1]["local_config"]["command"], "uvx");
    }

    /// The platform catalog does not send server_type; the router derives it
    /// from hosted_by / config presence rather than 500-ing on validation.
    #[tokio::test]
    #[serial]
    async fn test_derives_server_type_when_platform_omits_it() {
        let platform_service = platform_service();
        let mock_data = json!([
            {
                "slug": "notion",
                "display_name": "Notion",
                "hosted_by": "external",
                "remote_config": {
                    "server_url": "https://mcp.notion.com/mcp",
                    "auth_type": "oauth",
                },
                "sort_order": 0,
                "catalog_version": 1,
            },
            {
                "slug": "yahoo-finance",
                "display_name": "Yahoo Finance",
                "hosted_by": "local",
                "local_config": {"command": "uvx", "args": ["yahoo-finance-mcp"]},
                "sort_order": 10,
                "catalog_version": 1,
            },
        ]);
        platform_service.set_list_mcp_catalog(Ok(entries(mock_data)));
        let resp = get("/v1/mcp-catalog").await;

        assert_eq!(resp.status_code(), 200);
        let data = resp.json();
        assert_eq!(data[0]["server_type"], "remote");
        assert_eq!(data[1]["server_type"], "local");
    }

    #[tokio::test]
    #[serial]
    async fn test_returns_empty_list() {
        let platform_service = platform_service();
        platform_service.set_list_mcp_catalog(Ok(vec![]));
        let resp = get("/v1/mcp-catalog").await;

        assert_eq!(resp.status_code(), 200);
        assert_eq!(resp.json(), &json!([]));
    }

    #[tokio::test]
    #[serial]
    async fn test_502_when_platform_unavailable() {
        let platform_service = platform_service();
        platform_service.set_list_mcp_catalog(Err(PlatformError::connection("Connection refused")));
        let resp = get("/v1/mcp-catalog").await;

        assert_eq!(resp.status_code(), 502);
        assert_eq!(resp.json()["detail"], "Platform API unavailable");
    }

    #[tokio::test]
    #[serial]
    async fn test_401_when_unauthorized() {
        let platform_service = platform_service();
        platform_service.set_list_mcp_catalog(Err(PlatformError::permission("Authentication failed.")));
        let resp = get("/v1/mcp-catalog").await;

        assert_eq!(resp.status_code(), 401);
        let detail = resp.json()["detail"].as_str().expect("a string detail");
        assert!(detail.contains("Authentication failed"), "detail was {detail:?}");
    }
}
