//! Port of `tests/unit/test_explorer_token_refresh.py`.
//!
//! Every MCP call path renews an expired token, through one shared helper.
//!
//! `Explorer::call_mcp_tool` — the main MCP answering path since indexing was
//! replaced by live tool calls — called the tool with whatever token was
//! stored and returned `""` on failure, where every failure is a deliberate
//! silent skip.  So a connected app whose token had simply aged out looked to
//! retrieval like an app with nothing in it, while the same app answered a
//! planned execute step fine because *that* path retried.
//!
//! The retry now lives in `MCPToolCaller::call_with_refresh`, so the six call
//! paths cannot drift apart again — which they had: two guarded on a server
//! URL, one retried with the wrong tool name, and this one had no retry at all.
//!
//! Python replaces `caller.call_tool` and `caller.refresh_access_token` with
//! mocks.  Here the transport is replaced through the caller's call-tool hook
//! and the refresh runs for real against a fake token endpoint, so "refresh
//! returned X" is "the token endpoint minted X" and "refresh returned None" is
//! "the token endpoint refused".

use std::sync::Arc;

use serde_json::{json, Map, Value};
use serial_test::serial;

use smartloop::agents::mcp::mcp_tool_caller::{reset_refresh_state, MCPToolCaller};
use smartloop::mcp::client::MCPToolCallResult;
use smartloop::workspace::MCPServer;

use crate::shared::mcp::{
    explorer, in_order, oauth_server, stub_transport, text_result, token_endpoint, token_posts, tool_caller,
    unauthorized, FakeResolver,
};

const TOKEN_ENDPOINT: &str = "https://example/token";

fn server() -> MCPServer {
    oauth_server("s1", "monday", "https://example/mcp", Some(TOKEN_ENDPOINT), Some("expired"), Some("rt"))
}

fn no_args() -> Map<String, Value> {
    Map::new()
}

fn first_text(result: &MCPToolCallResult) -> String {
    result.content.as_ref().unwrap()[0]["text"].as_str().unwrap().to_string()
}

// ------------------------------------------------------- the helper renews once

#[test]
#[serial(mcp_global)]
fn test_the_call_is_retried_on_the_fresh_token() {
    reset_refresh_state();
    let (_endpoint, _guard) = token_endpoint(200, "fresh");
    let caller = tool_caller();
    let calls = stub_transport(&caller, in_order(vec![unauthorized(), text_result("Sprint board")]));

    let mut server = server();
    let (result, refreshed) = caller.call_with_refresh(&mut server, "get_board", Some(&no_args()), None);

    assert!(refreshed);
    assert_eq!(first_text(&result), "Sprint board");
    assert_eq!(calls.lock()[1].3.as_deref(), Some("fresh"));
    reset_refresh_state();
}

#[test]
#[serial(mcp_global)]
fn test_the_failed_token_is_offered_to_the_refresh() {
    // So a redemption that lost the race can return the winner's token.
    //
    // Python asserts `refresh_access_token(..., stale_token="expired")`.  The
    // Rust refresh is reached through the real method, so what is observable
    // is its consequence: the token the call failed with is still the stored
    // one, so the redemption goes ahead — once, with the stored refresh token.
    reset_refresh_state();
    let (endpoint, _guard) = token_endpoint(503, "");
    let caller = tool_caller();
    let calls = stub_transport(&caller, in_order(vec![unauthorized()]));

    let mut server = server();
    caller.call_with_refresh(&mut server, "get_board", Some(&no_args()), None);

    assert_eq!(calls.lock()[0].3.as_deref(), Some("expired"));
    let posts = token_posts(&endpoint, TOKEN_ENDPOINT);
    assert_eq!(posts.len(), 1);
    assert_eq!(posts[0]["refresh_token"], "rt");
    reset_refresh_state();
}

#[test]
#[serial(mcp_global)]
fn test_nothing_to_refresh_with_is_still_a_silent_skip() {
    reset_refresh_state();
    let (_endpoint, _guard) = token_endpoint(503, "");
    let caller = tool_caller();
    let calls = stub_transport(&caller, in_order(vec![unauthorized()]));

    let mut server = server();
    let (result, refreshed) = caller.call_with_refresh(&mut server, "get_board", Some(&no_args()), None);

    assert!(!result.success);
    assert!(!refreshed);
    assert_eq!(calls.lock().len(), 1);
    reset_refresh_state();
}

#[test]
#[serial(mcp_global)]
fn test_an_ordinary_failure_does_not_spend_a_refresh_token() {
    // Only a real 401 sets `auth_expired`; a tool error is not one.
    reset_refresh_state();
    let (endpoint, _guard) = token_endpoint(200, "fresh");
    let caller = tool_caller();
    stub_transport(
        &caller,
        in_order(vec![MCPToolCallResult {
            success: false,
            error: Some("board 401 not found".into()),
            ..Default::default()
        }]),
    );

    let mut server = server();
    let (_result, refreshed) = caller.call_with_refresh(&mut server, "get_board", Some(&no_args()), None);

    assert!(!refreshed);
    assert_eq!(endpoint.call_count(), 0, "a refresh token was spent on a tool error");
    reset_refresh_state();
}

// ----------------------------------------------------- the collect path uses it

/// The path that had no retry at all now goes through the helper.
fn explorer_over(caller: MCPToolCaller) -> smartloop::agents::mcp::explorer::Explorer {
    let resolver = Arc::new(FakeResolver { answer: Box::new(|_| None), calls: Default::default() });
    explorer(Arc::new(caller), resolver, Arc::new(smartloop::mcp::call_plans::follow_up_calls))
}

#[test]
#[serial(mcp_global)]
fn test_an_expired_token_still_returns_the_records() {
    reset_refresh_state();
    let (endpoint, _guard) = token_endpoint(200, "fresh");
    let caller = tool_caller();
    stub_transport(&caller, in_order(vec![unauthorized(), text_result("Sprint board")]));

    let ex = explorer_over(caller);
    let args = json!({"id": "1"}).as_object().cloned().unwrap();
    let text = ex.call_mcp_tool(&server(), "get_board", &args, "what is on my board");

    assert!(text.contains("Sprint board"), "{text}");
    assert_eq!(token_posts(&endpoint, TOKEN_ENDPOINT).len(), 1);
    reset_refresh_state();
}

#[test]
#[serial(mcp_global)]
fn test_a_dead_grant_is_still_a_silent_skip() {
    reset_refresh_state();
    let (_endpoint, _guard) = token_endpoint(503, "");
    let caller = tool_caller();
    stub_transport(&caller, in_order(vec![unauthorized()]));

    let ex = explorer_over(caller);
    let args = json!({"id": "1"}).as_object().cloned().unwrap();
    let text = ex.call_mcp_tool(&server(), "get_board", &args, "what is on my board");

    assert_eq!(text, "");
    reset_refresh_state();
}

#[test]
#[serial(mcp_global)]
fn test_the_follow_up_fetch_retries_with_its_own_tool() {
    // It was retried with the *listing* tool's name, not the detail tool's.
    reset_refresh_state();
    let (_endpoint, _guard) = token_endpoint(200, "fresh");
    let caller = tool_caller();
    let calls = stub_transport(&caller, in_order(vec![unauthorized(), text_result("the body")]));

    let resolver = Arc::new(FakeResolver { answer: Box::new(|_| None), calls: Default::default() });
    // `patch("smartloop.mcp.call_plans.follow_up_calls", return_value=[...])`.
    let ex = explorer(
        Arc::new(caller),
        resolver,
        Arc::new(|_, _, _| vec![("get_item".to_string(), json!({"id": "7"}).as_object().cloned().unwrap())]),
    );
    let text = ex.fetch_record_details(&server(), "list_items", "payload", None);

    assert!(text.contains("the body"), "{text}");
    assert_eq!(calls.lock()[1].1, "get_item");
    reset_refresh_state();
}
