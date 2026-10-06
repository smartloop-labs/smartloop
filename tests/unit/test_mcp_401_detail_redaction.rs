//! Port of `tests/unit/test_mcp_401_detail_redaction.py`.
//!
//! The 401 detail must not carry the token that was refused.
//!
//! It embedded `Authorization sent: <first 30 characters>`.  "Bearer " is
//! seven of those, so twenty-three characters of a live bearer token travelled
//! in a string that is logged at error level *and* stored as
//! `server.last_error` — which `GET /v1/projects/{id}/mcp` returns to the
//! client and the UI renders.  A short token went out whole.

use serde_json::json;

use smartloop::mcp::client::{HttpResponse, MCPClient, MCPClientError};

use crate::shared::mcp::FakeHttp;

const TOKEN: &str = "sl_live_0123456789abcdef";

fn client_answering_401(body: &str) -> MCPClient {
    MCPClient::with_transport(FakeHttp::always(HttpResponse::new(401, body)))
}

fn detail(body: &str, token: Option<&str>) -> String {
    let mut client = client_answering_401(body);
    let err = client
        .send_request("https://example/mcp", "tools/call", Some(json!({})), token, "Bearer")
        .expect_err("a 401 must raise");
    assert!(matches!(err, MCPClientError::Connection(_)), "expected MCPConnectionError, got {err:?}");
    err.to_string()
}

#[test]
fn test_no_part_of_the_token_appears() {
    let detail = detail("", Some(TOKEN));
    assert!(!detail.contains(TOKEN));
    // The prefix the old code would have emitted, too.
    assert!(!detail.contains(&TOKEN[..23]));
    assert!(!detail.contains("0123456789"));
}

#[test]
fn test_it_still_says_a_credential_was_sent() {
    // The diagnostic value is presence and shape, not the secret.
    let detail = detail("", Some(TOKEN));
    assert!(detail.contains("Bearer"));
    assert!(detail.contains(&format!("{} chars", TOKEN.len())));
}

#[test]
fn test_it_says_when_none_was_sent() {
    assert!(detail("", None).contains("Authorization sent: (none)"));
}

#[test]
fn test_an_empty_body_does_not_widen_the_budget() {
    // _friendly_discovery_error truncates at 180 chars; with an empty body the
    // whole preview used to fit, which is how the fragment reached the API.
    let detail = detail("", Some(TOKEN));
    let head: String = detail.chars().take(180).collect();
    assert!(!head.contains(&TOKEN[..10]));
}
