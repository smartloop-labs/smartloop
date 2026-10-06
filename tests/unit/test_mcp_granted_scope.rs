//! Port of `tests/unit/test_mcp_granted_scope.py`.
//!
//! What the grant came back with, kept rather than discarded.
//!
//! RFC 6749 §5.1 lets the token response name the scope actually granted,
//! which need not be the scope asked for — a server may narrow the set and say
//! so.  The value was parsed into `TokenResponse` and then dropped by every
//! caller, so a down-scoped grant was invisible until a tool call failed, and
//! the page shown when a connection produced no tools could only guess that
//! scopes were why.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use serial_test::serial;

use smartloop::agents::mcp::mcp_tool_caller::reset_refresh_state;
use smartloop::api::routers::mcp_servers::{OAuthCallbackQuery, OAuthCallbackResponse};
use smartloop::mcp::client::{HttpTransportOverride, MCPConnectResult};
use smartloop::workspace::workspace::{
    MCPAuthType, MCPOAuthCredentials, MCPServer, MCPServerType, RemoteMCPServerConfig,
};

use crate::shared::mcp::{
    ok_json, reset_router_state, run_callback, token_response, tool, tool_caller, FakeHttp, ServiceEnv,
};

/// `test_mcp_credential_persistence._server(**creds)`.
fn server(scope: Option<&str>) -> MCPServer {
    MCPServer {
        id: "s1".into(),
        name: "monday".into(),
        server_type: MCPServerType::Remote,
        remote_config: Some(RemoteMCPServerConfig {
            server_url: "https://mcp.monday.com/mcp".into(),
            auth_type: MCPAuthType::Oauth,
            token_endpoint: Some("https://auth.monday.com/oauth_ms/oauth/token".into()),
            ..Default::default()
        }),
        oauth_credentials: Some(MCPOAuthCredentials {
            client_id: "cid".into(),
            access_token: Some("expired".into()),
            refresh_token: Some("old-rt".into()),
            scope: scope.map(str::to_string),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Refresh `server` against a token endpoint answering `payload`
/// (`test_mcp_credential_persistence._reply`).
fn refresh_with_reply(server: &mut MCPServer, payload: Value) {
    reset_refresh_state();
    let _transport = HttpTransportOverride::install(FakeHttp::always(ok_json(payload)));
    tool_caller().refresh_access_token(server, None);
    reset_refresh_state();
}

// ------------------------------------------------------ the credential carries it

#[test]
fn test_the_model_records_a_granted_scope() {
    let creds = MCPOAuthCredentials {
        client_id: "cid".into(),
        scope: Some("profile email".into()),
        ..Default::default()
    };
    assert_eq!(creds.scope.as_deref(), Some("profile email"));
}

#[test]
fn test_it_defaults_to_nothing_for_credentials_written_before_it_existed() {
    assert_eq!(MCPOAuthCredentials { client_id: "cid".into(), ..Default::default() }.scope, None);
    // And on disk: a record written before the field existed reads back as None.
    let creds: MCPOAuthCredentials = serde_json::from_value(json!({"client_id": "cid"})).unwrap();
    assert_eq!(creds.scope, None);
}

// ------------------------------------------------------- a refresh keeps it current

#[test]
#[serial]
#[serial(mcp_global)]
fn test_a_narrowed_refresh_is_recorded() {
    // §6 lets a server narrow on refresh; the stored scope must follow.
    let mut server = server(Some("profile email offline_access"));
    refresh_with_reply(
        &mut server,
        json!({
            "access_token": "new", "token_type": "Bearer",
            "expires_in": 3600, "scope": "profile offline_access",
        }),
    );
    assert_eq!(server.oauth_credentials.unwrap().scope.as_deref(), Some("profile offline_access"));
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_a_refresh_that_names_no_scope_leaves_the_grant_alone() {
    // Omitting it means "unchanged", not "nothing".
    let mut server = server(Some("profile email offline_access"));
    refresh_with_reply(&mut server, json!({"access_token": "new", "token_type": "Bearer", "expires_in": 3600}));
    assert_eq!(server.oauth_credentials.unwrap().scope.as_deref(), Some("profile email offline_access"));
}

// ---------------------------------------------------------------- the callback

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

/// Run the redirect callback to completion; returns the response and the
/// connection the callback stored (Python: what `add_remote_mcp_server` was
/// called with).
fn complete(connect_result: MCPConnectResult, granted_scope: Option<&str>) -> (OAuthCallbackResponse, Option<MCPServer>) {
    let env = ServiceEnv::new("p1");
    let session = json!({
        "created_at": now(),
        "project_id": "p1",
        "server_url": "https://mcp.getclera.com/",
        "state": "st",
        "code_verifier": "cv",
        "code_challenge": "chal",
        "scope": "profile email offline_access",
        "client_id": "cid",
        "client_secret": null,
        "token_endpoint": "https://clerk.getclera.com/oauth/token",
        "authorization_endpoint": "https://clerk.getclera.com/oauth/authorize",
        "resource": "https://mcp.getclera.com",
        "callback_url": "http://localhost:38540/v1/mcp/oauth/callback",
        "name": "Clera",
        "return_url": null,
        "description": null,
        "tags": null,
        "catalog_slug": null,
    });
    let (resp, _) = run_callback(
        Some(session),
        OAuthCallbackQuery { code: Some("c".into()), state: Some("st".into()), ..Default::default() },
        connect_result,
        token_response(Some("rt"), Some(3600), granted_scope),
        env.model_service.clone(),
    );
    reset_router_state();
    let saved = env.servers().into_iter().next();
    (resp, saved)
}

fn one_tool() -> MCPConnectResult {
    MCPConnectResult { success: true, tools: vec![tool("t", Some(json!({})))], ..Default::default() }
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_the_granted_scope_reaches_the_record() {
    let (_, saved) = complete(one_tool(), Some("profile email"));
    let saved = saved.expect("the connection was stored");
    assert_eq!(saved.oauth_credentials.unwrap().scope.as_deref(), Some("profile email"));
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_a_grant_naming_no_scope_persists_nothing() {
    let (_, saved) = complete(one_tool(), None);
    let saved = saved.expect("the connection was stored");
    assert_eq!(saved.oauth_credentials.unwrap().scope, None);
}

// ------------------------------------------------- the empty tool page stops guessing

#[test]
#[serial]
#[serial(mcp_global)]
fn test_it_names_what_was_actually_granted() {
    let (resp, _) = complete(MCPConnectResult { success: true, tools: vec![], ..Default::default() }, Some("profile email"));
    assert_eq!(resp.status_code(), 400);
    assert!(resp.body().contains("profile email"), "{}", resp.body());
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_it_says_so_when_the_grant_named_no_scopes() {
    let (resp, _) = complete(MCPConnectResult { success: true, tools: vec![], ..Default::default() }, None);
    assert!(resp.body().contains("no scopes"), "{}", resp.body());
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_a_transport_failure_still_reports_the_transport() {
    // A dead endpoint is not a scope problem and must not read as one.
    let (resp, _) = complete(
        MCPConnectResult { success: false, error: Some("HTTP error: 404".into()), ..Default::default() },
        Some("profile"),
    );
    let body = resp.body();
    assert!(body.contains("HTTP error: 404"), "{body}");
    assert!(!body.contains("granted"), "{body}");
}
