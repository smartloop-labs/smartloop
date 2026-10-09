//! Port of `tests/unit/test_mcp_dead_grant.py`.
//!
//! A connection that can no longer authenticate stops reporting itself
//! healthy.
//!
//! `status`/`last_error` were written only by the provisioning pass, so a
//! runtime auth failure changed nothing: a connection whose grant the provider
//! had retired kept reporting `ready` with its full tool list forever, and
//! `build_available_servers` never consulted `status` anyway, so the planner
//! went on offering tools that could only 401 — into a failure path that
//! swallows the error.  To the user that reads as "that app has nothing in it".
//!
//! Only a refusal that re-minting cannot recover from counts.  A network blip
//! or a 5xx must leave the connection alone; failing it over one bad
//! round-trip would cost someone a working connection.
//!
//! Python patched `httpx.Client` so the token endpoint's `post` raised; here
//! the token endpoint is a local wiremock server answering with the same
//! refusal (or an address nothing listens on, for the unreachable case).

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::json;
use serial_test::serial;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

use smartloop::agents::mcp::argument_extractor::MCPArgumentExtractor;
use smartloop::agents::mcp::mcp_tool_caller::{reset_refresh_state, MCPToolCaller, PersistFn, ServerChange};
use smartloop::agents::mcp::result_mapper::MCPResultMapper;
use smartloop::agents::orchestration::step_executor::StepExecutor;
use smartloop::workspace::workspace::{
    MCPAuthType, MCPOAuthCredentials, MCPServer, MCPServerStatus, MCPServerType, MCPToolDefinition,
    RemoteMCPServerConfig,
};

use crate::shared::orch_exec::StubLLM;

const TOKEN_ENDPOINT: &str = "https://auth.monday.com/token";

fn server_with(refresh_token: Option<&str>, token: Option<&str>, token_endpoint: &str) -> MCPServer {
    MCPServer {
        id: "s1".into(),
        name: "monday".into(),
        server_type: MCPServerType::Remote,
        remote_config: Some(RemoteMCPServerConfig {
            server_url: "https://mcp.monday.com/mcp".into(),
            auth_type: if refresh_token.is_some() { MCPAuthType::Oauth } else { MCPAuthType::Token },
            token_endpoint: Some(token_endpoint.into()),
            token: token.map(str::to_string),
            ..Default::default()
        }),
        oauth_credentials: refresh_token.map(|rt| MCPOAuthCredentials {
            client_id: "cid".into(),
            refresh_token: Some(rt.into()),
            ..Default::default()
        }),
        tools: vec![MCPToolDefinition {
            name: "get_board".into(),
            description: Some(String::new()),
            input_schema: Some(serde_json::Map::new()),
        }],
        ..Default::default()
    }
}

/// `_server()` — an OAuth connection holding refresh token `rt`.
fn server(token_endpoint: &str) -> MCPServer {
    server_with(Some("rt"), None, token_endpoint)
}

fn caller(persist: Option<PersistFn>) -> MCPToolCaller {
    MCPToolCaller::new(
        Arc::new(MCPArgumentExtractor::new(StubLLM::shared(""))),
        Arc::new(Mutex::new(MCPResultMapper::new())),
        persist,
    )
}

/// A persist hook that records what each call wrote (`saves.append(1)`).
fn counting_persist() -> (PersistFn, Arc<Mutex<Vec<ServerChange>>>) {
    let saves = Arc::new(Mutex::new(Vec::new()));
    let sink = saves.clone();
    let persist: PersistFn = Arc::new(move |_server: &MCPServer, change: ServerChange| {
        sink.lock().push(change);
        Ok(())
    });
    (persist, saves)
}

/// A token endpoint refusing every redemption with `code` (no `error` key
/// when `None`) and HTTP `status`.  Returns the runtime driving the mock (keep
/// it alive) and the endpoint URL.
fn refusing_endpoint(code: Option<&str>, status: u16) -> (tokio::runtime::Runtime, MockServer, String) {
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let body = match code {
        Some(c) => json!({"error": c}),
        None => json!({}),
    };
    let mock = rt.block_on(async {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status).set_body_json(body))
            .mount(&mock)
            .await;
        mock
    });
    let url = format!("{}/token", mock.uri());
    (rt, mock, url)
}

/// Forget MCP redemption locks and cooldowns around each test (conftest's
/// autouse `_reset_mcp_refresh_state`).
struct ResetRefreshState;

impl ResetRefreshState {
    fn new() -> Self {
        reset_refresh_state();
        Self
    }
}

impl Drop for ResetRefreshState {
    fn drop(&mut self) {
        reset_refresh_state();
    }
}

// ---------------------------------------------------------------------------
// TestOnlyADeadGrantFailsTheConnection
// ---------------------------------------------------------------------------

fn a_retired_grant_fails_it(code: &str) {
    let _reset = ResetRefreshState::new();
    let (_rt, _mock, url) = refusing_endpoint(Some(code), 400);
    let mut server = server(&url);
    let (persist, saves) = counting_persist();
    let caller = caller(Some(persist));

    assert_eq!(caller.refresh_access_token(&mut server, None), None);

    assert_eq!(server.status, MCPServerStatus::Failed);
    assert!(server.last_error.as_deref().unwrap_or("").contains("Reconnect"));
    assert_eq!(*saves.lock(), vec![ServerChange::Status]);
}

#[test]
#[serial]
fn test_a_retired_grant_fails_it_invalid_grant() {
    a_retired_grant_fails_it("invalid_grant");
}

#[test]
#[serial]
fn test_a_retired_grant_fails_it_invalid_client() {
    a_retired_grant_fails_it("invalid_client");
}

/// A 5xx is the provider having a bad day, not the grant expiring.
#[test]
#[serial]
fn test_a_server_error_leaves_it_alone() {
    let _reset = ResetRefreshState::new();
    let (_rt, _mock, url) = refusing_endpoint(Some("server_error"), 503);
    let mut server = server(&url);
    let caller = caller(None);

    assert_eq!(caller.refresh_access_token(&mut server, None), None);

    assert_eq!(server.status, MCPServerStatus::Ready);
    assert_eq!(server.last_error, None);
}

#[test]
#[serial]
fn test_an_unreachable_endpoint_leaves_it_alone() {
    let _reset = ResetRefreshState::new();
    // Nothing listens on port 1: the redemption fails to connect.
    let mut server = server("http://127.0.0.1:1/token");
    let caller = caller(None);

    assert_eq!(caller.refresh_access_token(&mut server, None), None);

    assert_eq!(server.status, MCPServerStatus::Ready);
}

#[test]
#[serial]
fn test_a_refusal_with_no_error_code_leaves_it_alone() {
    let _reset = ResetRefreshState::new();
    let (_rt, _mock, url) = refusing_endpoint(None, 400);
    let mut server = server(&url);
    let caller = caller(None);

    assert_eq!(caller.refresh_access_token(&mut server, None), None);

    assert_eq!(server.status, MCPServerStatus::Ready);
}

#[test]
#[serial]
fn test_the_same_failure_twice_is_written_once() {
    let _reset = ResetRefreshState::new();
    let (_rt, _mock, url) = refusing_endpoint(Some("invalid_grant"), 400);
    let mut server = server(&url);
    let (persist, saves) = counting_persist();
    let caller = caller(Some(persist));

    caller.refresh_access_token(&mut server, None);
    caller.refresh_access_token(&mut server, None);

    assert_eq!(*saves.lock(), vec![ServerChange::Status]);
}

// ---------------------------------------------------------------------------
// TestARejectedPastedCredentialFailsIt
// There is no refresh to fall back on, so the 401 is the verdict.
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn test_a_pasted_token_that_stopped_working_is_reported() {
    let _reset = ResetRefreshState::new();
    let mut server = server_with(None, Some("ghp_xxx"), TOKEN_ENDPOINT);
    let caller = caller(None);

    assert_eq!(caller.refresh_access_token(&mut server, None), None);

    assert_eq!(server.status, MCPServerStatus::Failed);
    assert!(server.last_error.as_deref().unwrap_or("").contains("Enter a new token"));
}

/// Nothing to re-enter: it needs re-authorizing, not a new paste.
#[test]
#[serial]
fn test_an_oauth_connection_with_no_refresh_token_is_left_alone() {
    let _reset = ResetRefreshState::new();
    let mut server = server_with(None, None, TOKEN_ENDPOINT);
    let caller = caller(None);

    assert_eq!(caller.refresh_access_token(&mut server, None), None);

    assert_eq!(server.status, MCPServerStatus::Ready);
}

// ---------------------------------------------------------------------------
// TestThePlannerStopsOfferingIt
// ---------------------------------------------------------------------------

#[test]
fn test_a_failed_connection_is_not_offered() {
    let mut server = server(TOKEN_ENDPOINT);
    server.status = MCPServerStatus::Failed;
    server.last_error = Some("This connection's authorization has expired.".into());

    let available = StepExecutor::build_available_servers(&[server]);

    assert!(available.is_empty());
}

#[test]
fn test_a_healthy_connection_still_is() {
    let available = StepExecutor::build_available_servers(&[server(TOKEN_ENDPOINT)]);

    let names: Vec<&str> = available.iter().map(|s| s["name"].as_str().unwrap()).collect();
    assert_eq!(names, vec!["monday"]);
}
