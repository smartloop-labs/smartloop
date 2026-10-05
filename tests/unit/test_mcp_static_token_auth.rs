//! Port of `tests/unit/test_mcp_static_token_auth.py`.
//!
//! A server registered from a URL carrying its own token authenticates its
//! calls.
//!
//! Registering `http://host/v1/mcp?token=abc` splits the credential off the URL
//! and stores it on the remote config; there are no OAuth credentials because
//! none were ever negotiated.  Discovery has always read that field, so such a
//! server lists its tools and looks healthy — and then every call to one of
//! those tools went out with no Authorization header and came back 401.

use smartloop::agents::mcp::mcp_tool_caller::MCPToolCaller;
use smartloop::api::routers::mcp_servers;
use smartloop::api::services::model_service::ModelService;
use smartloop::workspace::workspace::{MCPOAuthCredentials, MCPServer, RemoteMCPServerConfig};
use wiremock::matchers::any;
use wiremock::{Mock, MockServer, ResponseTemplate};

fn server(token: Option<&str>, oauth_token: Option<&str>) -> MCPServer {
    MCPServer {
        name: "local-61185".to_string(),
        remote_config: Some(RemoteMCPServerConfig {
            server_url: "http://127.0.0.1:61185/v1/mcp".to_string(),
            token: token.map(str::to_string),
            ..Default::default()
        }),
        oauth_credentials: oauth_token.map(|t| MCPOAuthCredentials {
            access_token: Some(t.to_string()),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn test_static_token_on_remote_config_is_used() {
    assert_eq!(MCPToolCaller::get_access_token(&server(Some("abc"), None)).as_deref(), Some("abc"));
}

#[test]
fn test_oauth_credentials_still_win() {
    let server = server(Some("static"), Some("fresh"));
    assert_eq!(MCPToolCaller::get_access_token(&server).as_deref(), Some("fresh"));
}

#[test]
fn test_no_credential_anywhere_is_none() {
    assert_eq!(MCPToolCaller::get_access_token(&server(None, None)), None);
}

/// "/info" belongs on the path — appended raw it lands inside the query.
#[tokio::test]
async fn test_info_probe_keeps_the_query_string() {
    // The probe is not served: whatever it reaches answers with an error.
    let probe = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(404))
        .mount(&probe)
        .await;

    let result = mcp_servers::try_register_smartloop_mcp_server(
        &format!("{}/v1/mcp?token=abc", probe.uri()),
        ModelService::new_bare(),
        "p",
    )
    .await;

    // An unreachable probe is "not a Smartloop server", not an error.
    assert!(matches!(result, Ok(None)));
    let seen: Vec<String> = probe
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .map(|r| r.url.to_string())
        .collect();
    assert_eq!(seen, vec![format!("{}/v1/mcp/info?token=abc", probe.uri())]);
}
