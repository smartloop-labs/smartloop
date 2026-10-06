//! Port of `tests/unit/test_mcp_auth_expired.py`.
//!
//! A token has expired when the server said 401, not when the word appears.
//!
//! Every MCP call site decided a token had expired by asking `"401" in
//! result.error` — a substring test against a message that embeds the server's
//! own response body.  A tool returning a document that mentions 401 spent a
//! refresh token on nothing and retried a call that had not failed for auth
//! reasons at all.
//!
//! `auth_expired` is set by the transport from the HTTP status instead.
//!
//! Python patched `MCPClient._ensure_session` to raise the connection error;
//! here the session handshake itself meets a server answering that status.

use smartloop::mcp::client::{HttpResponse, MCPClient, MCPToolCallResult};

use crate::shared::mcp::FakeHttp;

#[test]
fn test_a_body_mentioning_401_is_not_an_auth_failure() {
    let result = MCPToolCallResult {
        success: false,
        error: Some("Tool call failed: the upstream returned error code 401 in its payload".into()),
        ..Default::default()
    };
    assert!(!result.auth_expired);
}

#[test]
fn test_the_transport_flags_a_real_401() {
    let mut client = MCPClient::with_transport(FakeHttp::always(HttpResponse::new(401, "")));
    let result = client.call_tool("https://example/mcp", "whoami", None, None, "Bearer");

    assert!(!result.success);
    assert!(result.auth_expired);
}

#[test]
fn test_a_403_is_not_an_expired_token() {
    // Insufficient permissions is about the grant, not the token's age.
    let mut client = MCPClient::with_transport(FakeHttp::always(HttpResponse::new(403, "")));
    let result = client.call_tool("https://example/mcp", "whoami", None, None, "Bearer");

    assert!(!result.success);
    assert!(!result.auth_expired);
}
