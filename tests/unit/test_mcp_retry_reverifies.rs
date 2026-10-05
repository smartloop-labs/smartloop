//! Port of `tests/unit/test_mcp_retry_reverifies.py`.
//!
//! Re-checking a connection has to actually re-check it.
//!
//! `POST /v1/projects/{id}/mcp/{server_id}/refresh` is the only recovery the UI
//! offers.  It cleared `status` and `last_error` up front and queued a
//! provisioning pass — and that pass only discovers when a server has no
//! tools.  A connection whose authorization died keeps every tool discovery
//! ever found, so the pass skipped discovery and set it READY again, having
//! made no network call at all.  The user pressed the one button there was and
//! was told it was fixed.
//!
//! The other half: discovery runs on the stored credentials, so it ages out
//! like any call.  Renewing has to be tried before a connection is called
//! broken, or fixing the false green just produces a false red.
//!
//! Python drives `ModelService`'s methods over a `MagicMock` service with the
//! collaborators mocked.  Here the real service runs over an in-memory
//! workspace, and "discovery was called" is "the MCP endpoint saw the
//! handshake" on the fake HTTP layer.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use serial_test::serial;

use smartloop::agents::mcp::mcp_tool_caller::reset_refresh_state;
use smartloop::mcp::client::{
    HttpBody, HttpResponse, HttpTransport, HttpTransportOverride, MCPAuthExpiredError, MCPClient, MCPClientError,
};
use smartloop::workspace::workspace::MCPServerStatus;
use smartloop::workspace::{MCPServer, MCPToolDefinition};

use crate::shared::mcp::{ok_json, oauth_server, tool, wait_for, FakeHttp, ServiceEnv};

const SERVER_URL: &str = "https://mcp.notion.com/mcp";
const TOKEN_ENDPOINT: &str = "https://t";
const EXPIRED: &str = "This connection's authorization has expired. Reconnect it to keep using it.";

fn search_tool(name: &str) -> MCPToolDefinition {
    tool(name, Some(json!({})))
}

/// A remote connection that failed on auth but kept its catalog.
fn server(status: MCPServerStatus, tools: Option<Vec<MCPToolDefinition>>, token: &str) -> MCPServer {
    let mut s = oauth_server("s1", "notion", SERVER_URL, Some(TOKEN_ENDPOINT), Some(token), Some("rt"));
    s.status = status;
    s.last_error = Some(EXPIRED.to_string());
    s.tools = tools.unwrap_or_else(|| vec![search_tool("search")]);
    s.oauth_credentials.as_mut().unwrap().client_secret = Some("s".into());
    s
}

/// The JSON-RPC method a POST carried.
fn rpc_method(body: Option<&HttpBody>) -> String {
    match body {
        Some(HttpBody::Json(v)) => v.get("method").and_then(Value::as_str).unwrap_or("").to_string(),
        _ => String::new(),
    }
}

/// POSTs to the MCP endpoint carrying `method`, with their Authorization header.
fn handshakes(fake: &FakeHttp, method: &str) -> Vec<Option<String>> {
    fake.calls
        .lock()
        .iter()
        .filter(|c| c.method == "POST" && c.url == SERVER_URL && rpc_method(c.body.as_ref()) == method)
        .map(|c| c.headers.iter().find(|(k, _)| k == "Authorization").map(|(_, v)| v.clone()))
        .collect()
}

/// An MCP server that answers a handshake with `tools`.
fn mcp_answer(method: &str, tools: &[&str]) -> HttpResponse {
    match method {
        "initialize" => ok_json(json!({"jsonrpc": "2.0", "id": "1", "result": {"serverInfo": {"name": "notion"}}})),
        "tools/list" => ok_json(json!({
            "jsonrpc": "2.0", "id": "2",
            "result": {"tools": tools.iter().map(|t| json!({"name": t})).collect::<Vec<_>>()},
        })),
        _ => HttpResponse::new(202, ""),
    }
}

// --------------------------------------------- a retry drops the catalog it cannot trust

#[test]
#[serial]
#[serial(mcp_global)]
fn test_the_cached_tools_are_cleared() {
    let env = ServiceEnv::new("p1");
    env.add_server(server(MCPServerStatus::Failed, None, "stale"));
    env.record_provisioning();

    env.model_service.retry_mcp_server("s1", Some("p1")).expect("retried");

    let stored = &env.servers()[0];
    assert!(stored.tools.is_empty());
    assert_eq!(stored.status, MCPServerStatus::Pending);
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_the_pass_is_queued() {
    let env = ServiceEnv::new("p1");
    env.add_server(server(MCPServerStatus::Failed, None, "stale"));
    let queued = env.record_provisioning();

    env.model_service.retry_mcp_server("s1", Some("p1")).expect("retried");

    assert!(wait_for(Duration::from_secs(5), || !queued.lock().is_empty()), "nothing was queued");
    assert_eq!(*queued.lock(), vec![("p1".to_string(), "s1".to_string())]);
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_provisioning_then_discovers_rather_than_trusting_the_cache() {
    // The two halves together: after a retry, the pass must go to the wire.
    let env = ServiceEnv::new("p1");
    env.add_server(server(MCPServerStatus::Failed, None, "stale"));
    env.record_provisioning();
    env.model_service.retry_mcp_server("s1", Some("p1")).expect("retried");

    env.model_service.provision_mcp_server("p1", "s1").expect("a pass ran");

    assert!(!handshakes(&env.http, "initialize").is_empty(), "the pass trusted the cached catalog");
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_without_a_retry_a_known_catalog_is_still_not_rediscovered() {
    // The guard earns its keep on the registration path — leave it there.
    let env = ServiceEnv::new("p1");
    env.add_server(server(MCPServerStatus::Pending, None, "stale"));

    env.model_service.provision_mcp_server("p1", "s1").expect("a pass ran");

    assert!(handshakes(&env.http, "tools/list").is_empty(), "a known catalog was rediscovered");
}

// ------------------------------------------------- discovery renews before giving up

#[test]
#[serial]
#[serial(mcp_global)]
fn test_an_expired_token_is_renewed_and_discovery_retried() {
    reset_refresh_state();
    let env = ServiceEnv::new("p1");
    let mut srv = server(MCPServerStatus::Failed, Some(vec![]), "stale");
    env.add_server(srv.clone());

    // The first handshake is refused (the token aged out); after the renewal
    // the server answers.
    let mcp_posts = Arc::new(AtomicUsize::new(0));
    let seen = mcp_posts.clone();
    let fake = FakeHttp::new(move |method, url, body| {
        if url == TOKEN_ENDPOINT {
            return Ok(ok_json(json!({"access_token": "fresh", "token_type": "Bearer", "expires_in": 3600})));
        }
        if method == "POST" && url == SERVER_URL {
            if seen.fetch_add(1, Ordering::SeqCst) == 0 {
                return Ok(HttpResponse::new(401, "").with_header("www-authenticate", "Bearer"));
            }
            return Ok(mcp_answer(&rpc_method(body), &["search"]));
        }
        Ok(HttpResponse::new(404, ""))
    });
    let _transport = HttpTransportOverride::install(fake.clone());

    env.model_service.discover_mcp_tools(&mut srv).expect("discovery succeeded after renewal");

    assert_eq!(fake.posts().iter().filter(|(u, _)| u == TOKEN_ENDPOINT).count(), 1);
    assert_eq!(srv.tools.iter().map(|t| t.name.clone()).collect::<Vec<_>>(), vec!["search".to_string()]);
    // The second attempt must carry the renewed credential, not the stale one.
    let lists = handshakes(&fake, "tools/list");
    assert_eq!(lists.last().cloned().flatten().as_deref(), Some("Bearer fresh"));
    reset_refresh_state();
}

/// Whether `err` is an `MCPAuthExpiredError`.
fn is_auth_expired(err: &anyhow::Error) -> bool {
    err.downcast_ref::<MCPAuthExpiredError>().is_some()
        || matches!(err.downcast_ref::<MCPClientError>(), Some(MCPClientError::AuthExpired(_)))
}

/// An MCP endpoint that always 401s, and a token endpoint that has retired
/// the grant.
fn dead_grant() -> Arc<FakeHttp> {
    FakeHttp::new(|_, url, _| {
        if url == TOKEN_ENDPOINT {
            return Ok(HttpResponse::json(400, &json!({"error": "invalid_grant"})));
        }
        if url == SERVER_URL {
            return Ok(HttpResponse::new(401, ""));
        }
        Ok(HttpResponse::new(404, ""))
    })
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_a_retired_grant_keeps_its_own_diagnosis() {
    // Not overwritten by whatever the wire said.
    reset_refresh_state();
    let env = ServiceEnv::new("p1");
    let mut srv = server(MCPServerStatus::Ready, Some(vec![]), "stale");
    srv.last_error = None;
    env.add_server(srv.clone());
    let _transport = HttpTransportOverride::install(dead_grant());

    let err = env.model_service.discover_mcp_tools(&mut srv).expect_err("the grant is dead");

    assert!(is_auth_expired(&err), "expected MCPAuthExpiredError, got {err:?}");
    assert_eq!(srv.status, MCPServerStatus::Failed);
    reset_refresh_state();
}

#[test]
#[serial]
#[serial(mcp_global)]
fn test_the_provisioning_pass_preserves_that_message() {
    reset_refresh_state();
    let env = ServiceEnv::new("p1");
    env.add_server(server(MCPServerStatus::Failed, Some(vec![]), "stale"));
    let _transport = HttpTransportOverride::install(dead_grant());

    let result = env.model_service.provision_mcp_server("p1", "s1").expect("a pass ran");

    assert_eq!(result["status"], "failed");
    let stored = &env.servers()[0];
    assert!(stored.last_error.as_deref().unwrap_or("").contains("Reconnect it"), "{:?}", stored.last_error);
    assert_eq!(stored.status, MCPServerStatus::Failed);
    reset_refresh_state();
}

// ------------------------------------------- the connect result carries the auth failure

/// Decided from the status, as the tool-call path already does.
fn result_for(transport: Arc<dyn HttpTransport>) -> smartloop::mcp::client::MCPConnectResult {
    let mut client = MCPClient::with_transport(transport);
    client.connect_and_discover_tools("https://mcp.example.com/mcp", None, "Bearer")
}

#[test]
fn test_a_401_sets_auth_expired() {
    let result = result_for(FakeHttp::always(HttpResponse::new(401, "nope")));
    assert!(!result.success);
    assert!(result.auth_expired);
}

#[test]
fn test_a_403_does_not() {
    // Insufficient permission is not something a new token fixes.
    let result = result_for(FakeHttp::always(HttpResponse::new(403, "nope")));
    assert!(!result.auth_expired);
}

#[test]
fn test_a_transport_error_does_not() {
    let result = result_for(FakeHttp::new(|_, _, _| Err("connection refused".to_string())));
    assert!(!result.auth_expired);
}
