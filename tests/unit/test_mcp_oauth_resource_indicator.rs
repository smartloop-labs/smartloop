//! Port of `tests/unit/test_mcp_oauth_resource_indicator.py`.
//!
//! RFC 8707 resource indicators in the MCP OAuth flow.
//!
//! MCP auth (spec rev 2025-06-18) requires the client to name the MCP server it
//! wants a token for, in both the authorization request and every token
//! request.  Servers that enforce it — monday.com rejects the authorize call
//! outright with "invalid request: resource parameter required" — are
//! unreachable without this, and a refresh that drops the indicator is
//! rejected even though the original grant succeeded.

use std::sync::Arc;

use serde_json::{json, Map, Value};
use serial_test::serial;

use smartloop::mcp::oauth::MCPOAuthService;

use crate::shared::mcp::{ok_json, FakeHttp};

/// Records POST bodies so tests can assert on what was actually sent.
fn recording_client(json_data: Option<Value>) -> Arc<FakeHttp> {
    let body = json_data.unwrap_or_else(|| json!({"access_token": "at", "token_type": "Bearer"}));
    FakeHttp::always(ok_json(body))
}

fn first_post(fake: &FakeHttp) -> (String, Map<String, Value>) {
    fake.posts().into_iter().next().expect("a POST was made")
}

// ------------------------------------------------------------ canonical resource

#[test]
fn test_path_is_significant() {
    // /mcp and /sse are distinct resources on the same host.
    assert_eq!(MCPOAuthService::canonical_resource("https://mcp.monday.com/mcp"), "https://mcp.monday.com/mcp");
    assert_eq!(MCPOAuthService::canonical_resource("https://mcp.monday.com/sse"), "https://mcp.monday.com/sse");
}

#[test]
fn test_scheme_and_host_are_lowercased() {
    assert_eq!(MCPOAuthService::canonical_resource("HTTPS://MCP.Monday.COM/mcp"), "https://mcp.monday.com/mcp");
}

#[test]
fn test_fragment_is_dropped() {
    // RFC 8707 §2 forbids a fragment in a resource indicator.
    assert_eq!(
        MCPOAuthService::canonical_resource("https://mcp.example.com/mcp#frag"),
        "https://mcp.example.com/mcp"
    );
}

#[test]
fn test_bare_host_has_no_trailing_slash() {
    assert_eq!(MCPOAuthService::canonical_resource("https://mcp.example.com/"), "https://mcp.example.com");
}

#[test]
fn test_query_is_preserved() {
    assert_eq!(
        MCPOAuthService::canonical_resource("https://mcp.example.com/mcp?v=2"),
        "https://mcp.example.com/mcp?v=2"
    );
}

// --------------------------------------------------------- authorization request

#[test]
fn test_resource_is_sent_when_given() {
    let url = MCPOAuthService::new(None).build_authorization_url(
        "https://auth.monday.com/oauth2/authorize",
        "cid",
        "http://localhost:38540/v1/mcp/oauth/callback",
        "st",
        "cc",
        None,
        Some("https://mcp.monday.com/mcp"),
    );
    assert!(url.contains("resource=https%3A%2F%2Fmcp.monday.com%2Fmcp"), "{url}");
}

#[test]
fn test_resource_is_omitted_when_absent() {
    let url = MCPOAuthService::new(None).build_authorization_url(
        "https://auth.example.com/authorize",
        "cid",
        "http://localhost/cb",
        "st",
        "cc",
        None,
        None,
    );
    assert!(!url.contains("resource="), "{url}");
}

// --------------------------------------------------------------- token requests

#[test]
fn test_code_exchange_sends_resource() {
    let client = recording_client(None);
    MCPOAuthService::with_transport(client.clone())
        .exchange_code_for_tokens(
            "https://auth.monday.com/oauth_ms/oauth/token",
            "code",
            "http://localhost/cb",
            "cid",
            "cv",
            None,
            Some("https://mcp.monday.com/mcp"),
        )
        .unwrap();
    let (_, body) = first_post(&client);
    assert_eq!(body["resource"], "https://mcp.monday.com/mcp");
}

#[test]
fn test_refresh_sends_resource() {
    let client = recording_client(None);
    MCPOAuthService::with_transport(client.clone())
        .refresh_access_token(
            "https://auth.monday.com/oauth_ms/oauth/token",
            "rt",
            "cid",
            None,
            Some("https://mcp.monday.com/mcp"),
        )
        .unwrap();
    let (_, body) = first_post(&client);
    assert_eq!(body["resource"], "https://mcp.monday.com/mcp");
}

#[test]
fn test_refresh_omits_resource_when_absent() {
    // Pre-existing servers registered without an indicator keep working.
    let client = recording_client(None);
    MCPOAuthService::with_transport(client.clone())
        .refresh_access_token("https://auth.example.com/token", "rt", "cid", None, None)
        .unwrap();
    let (_, body) = first_post(&client);
    assert!(!body.contains_key("resource"));
}

// ------------------------------------------------------- stored-server refresh

/// `MCPToolCaller.refresh_access_token` runs off the persisted config.
mod stored_server_refresh {
    use super::*;

    use smartloop::agents::mcp::mcp_tool_caller::reset_refresh_state;
    use smartloop::mcp::client::HttpTransportOverride;
    use smartloop::workspace::workspace::{
        MCPAuthType, MCPOAuthCredentials, MCPServer, MCPServerType, RemoteMCPServerConfig,
    };

    use crate::shared::mcp::tool_caller;

    fn server(resource: Option<&str>) -> MCPServer {
        MCPServer {
            id: "s1".into(),
            name: "monday".into(),
            server_type: MCPServerType::Remote,
            remote_config: Some(RemoteMCPServerConfig {
                server_url: "https://mcp.monday.com/mcp".into(),
                auth_type: MCPAuthType::Oauth,
                token_endpoint: Some("https://auth.monday.com/oauth_ms/oauth/token".into()),
                resource: resource.map(str::to_string),
                ..Default::default()
            }),
            oauth_credentials: Some(MCPOAuthCredentials {
                client_id: "cid".into(),
                refresh_token: Some("rt".into()),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn refresh(mut server: MCPServer) -> (Option<String>, Vec<(String, Map<String, Value>)>) {
        reset_refresh_state();
        // The stored-config refresh runs through MCPOAuthService, which opens
        // its own client: the process-wide transport is what is replaced
        // (Python: `patch("httpx.Client", return_value=client)`).
        let client = recording_client(None);
        let _transport = HttpTransportOverride::install(client.clone());
        let caller = tool_caller();
        let token = caller.refresh_access_token(&mut server, None);
        reset_refresh_state();
        (token, client.posts())
    }

    #[test]
    #[serial(mcp_global)]
    fn test_persisted_resource_is_replayed() {
        let (token, posts) = refresh(server(Some("https://mcp.monday.com/mcp")));
        assert_eq!(token.as_deref(), Some("at"));
        assert_eq!(posts[0].1["resource"], "https://mcp.monday.com/mcp");
    }

    #[test]
    #[serial(mcp_global)]
    fn test_server_without_resource_refreshes_unchanged() {
        let (token, posts) = refresh(server(None));
        assert_eq!(token.as_deref(), Some("at"));
        assert!(!posts[0].1.contains_key("resource"));
    }
}

// ------------------------------------------------------ register flow plumbing

/// The register endpoint must derive an indicator and carry it to the callback.
mod register_flow_plumbing {
    use super::*;

    use smartloop::api::model::mcp::MCPRegisterRequest;

    use crate::shared::mcp::{register_via_oauth, reset_router_state, OAuthRegistration};

    fn register(prm_resource: Option<&str>, server_url: &str) -> OAuthRegistration {
        let mut prm = json!({"authorization_servers": ["https://auth.monday.com/mcp"]});
        if let Some(resource) = prm_resource {
            prm["resource"] = json!(resource);
        }
        let out = register_via_oauth(
            Some(prm),
            json!({
                "issuer": "https://auth.monday.com/mcp",
                "authorization_endpoint": "https://auth.monday.com/oauth2/authorize",
                "token_endpoint": "https://auth.monday.com/oauth_ms/oauth/token",
                "registration_endpoint": "https://auth.monday.com/oauth_ms/oauth/register",
            }),
            MCPRegisterRequest {
                server_url: Some(server_url.to_string()),
                server_type: "remote".into(),
                ..Default::default()
            },
            None,
        );
        reset_router_state();
        out
    }

    #[test]
    #[serial(mcp_global)]
    fn test_prm_resource_wins() {
        let out = register(Some("https://mcp.monday.com/mcp"), "https://mcp.monday.com/mcp");
        let resp = out.response.expect("the OAuth branch answers");
        assert_eq!(out.authorize["resource"], "https://mcp.monday.com/mcp");
        assert_eq!(out.session["resource"], "https://mcp.monday.com/mcp");
        assert_eq!(resp.resource.as_deref(), Some("https://mcp.monday.com/mcp"));
    }

    #[test]
    #[serial(mcp_global)]
    fn test_falls_back_to_canonical_server_url() {
        // A server publishing no PRM `resource` still gets an indicator.
        let out = register(None, "https://mcp.example.com/mcp");
        assert_eq!(out.authorize["resource"], "https://mcp.example.com/mcp");
        assert_eq!(out.session["resource"], "https://mcp.example.com/mcp");
    }
}

// ------------------------------------------------ post-auth discovery diagnostics

/// A grant can succeed and tool discovery still fail.  The callback used to
/// report every such case as "MCP server has no tools available", which hid
/// the only thing that identifies the cause — Atlassian's catalog URL pointed
/// at the retired `/v1/sse` transport, and the 401/dead-endpoint detail was
/// dropped.
mod post_auth_discovery_diagnostics {
    use super::*;

    use smartloop::api::routers::mcp_servers::OAuthCallbackQuery;
    use smartloop::api::services::model_service::ModelService;
    use smartloop::mcp::client::MCPConnectResult;

    use crate::shared::mcp::{reset_router_state, run_callback, token_response};

    fn callback(connect_result: MCPConnectResult) -> (u16, String) {
        let session = json!({
            "project_id": "p1",
            "server_url": "https://mcp.atlassian.com/v1/sse",
            "state": "st",
            "code_verifier": "cv",
            "client_id": "cid",
            "client_secret": null,
            "token_endpoint": "https://mcp.atlassian.com/v1/token",
            "authorization_endpoint": "https://mcp.atlassian.com/v1/authorize",
            "resource": "https://mcp.atlassian.com/v1/sse",
            "callback_url": "http://localhost:38540/v1/mcp/oauth/callback",
            "name": "Atlassian",
            "return_url": null,
            "description": null,
            "tags": null,
            "catalog_slug": "atlassian",
        });
        let (resp, _) = run_callback(
            Some(session),
            OAuthCallbackQuery { code: Some("c".into()), state: Some("st".into()), ..Default::default() },
            connect_result,
            token_response(None, None, None),
            ModelService::new_bare(),
        );
        reset_router_state();
        (resp.status_code(), resp.body().to_string())
    }

    #[test]
    #[serial(mcp_global)]
    fn test_handshake_failure_reports_the_transport_error() {
        let (status, body) = callback(MCPConnectResult {
            success: false,
            error: Some("HTTP error: 404".into()),
            ..Default::default()
        });
        assert_eq!(status, 400);
        assert!(body.contains("HTTP error: 404"), "{body}");
        assert!(body.contains("https://mcp.atlassian.com/v1/sse"), "{body}");
    }

    #[test]
    #[serial(mcp_global)]
    fn test_empty_tool_list_says_so_rather_than_blaming_the_transport() {
        // Handshake fine, nothing offered — most often a grant with no scopes.
        let (status, body) = callback(MCPConnectResult { success: true, tools: vec![], ..Default::default() });
        assert_eq!(status, 400);
        assert!(body.contains("no tools"), "{body}");
        assert!(body.contains("scopes"), "{body}");
    }

    #[test]
    #[serial(mcp_global)]
    fn test_server_url_is_escaped_into_the_page() {
        let (_status, body) = callback(MCPConnectResult {
            success: false,
            error: Some("<script>x</script>".into()),
            ..Default::default()
        });
        assert!(!body.contains("<script>"), "{body}");
        assert!(body.contains("&lt;script&gt;"), "{body}");
    }
}
