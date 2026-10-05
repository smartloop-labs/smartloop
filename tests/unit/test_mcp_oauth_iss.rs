//! Port of `tests/unit/test_mcp_oauth_iss.py`.
//!
//! RFC 9207: the redirect has to come from the server we sent the user to.
//!
//! A client that talks to more than one authorization server can be walked
//! into a mix-up.  The attacker starts a flow against a server it controls,
//! hands a code issued there to this callback, and the code is redeemed at the
//! *honest* server's token endpoint — the one named in the pending session.
//! The `iss` parameter is what tells the two apart, and Clerk already sends it.
//!
//! Absence matters as much as mismatch: a server that advertises the parameter
//! and then omits it has either been impersonated or had the parameter
//! stripped, so treating "no value" as "nothing to check" would hand over the
//! bypass for free.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use serial_test::serial;

use smartloop::api::routers::mcp_servers::{OAuthCallbackQuery, OAuthCallbackResponse, TokenExchangeParams};
use smartloop::mcp::client::MCPConnectResult;
use smartloop::mcp::oauth::{MCPOAuthService, OAuthMetadata};

use crate::shared::mcp::{ok_json, reset_router_state, run_callback, token_response, tool, FakeHttp, ServiceEnv};

const ISSUER: &str = "https://clerk.getclera.com";

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

fn session(issuer: Option<&str>, advertised: bool) -> Value {
    json!({
        "created_at": now(),
        "project_id": "p1",
        "server_url": "https://mcp.getclera.com/",
        "state": "st",
        "code_verifier": "cv",
        "code_challenge": "chal",
        "scope": "profile email offline_access",
        "client_id": "cid",
        "client_secret": null,
        "token_endpoint": format!("{ISSUER}/oauth/token"),
        "authorization_endpoint": format!("{ISSUER}/oauth/authorize"),
        "issuer": issuer,
        "issuer_parameter_supported": advertised,
        "resource": "https://mcp.getclera.com",
        "callback_url": "http://localhost:38540/v1/mcp/oauth/callback",
        "name": "Clera",
        "return_url": null,
        "description": null,
        "tags": null,
        "catalog_slug": null,
    })
}

fn params(code: Option<&str>, state: &str, iss: Option<&str>, error: Option<&str>) -> OAuthCallbackQuery {
    OAuthCallbackQuery {
        code: code.map(str::to_string),
        state: Some(state.to_string()),
        iss: iss.map(str::to_string),
        error: error.map(str::to_string),
        ..Default::default()
    }
}

/// Deliver a redirect and report both the page and whether the code was spent.
fn callback(session: Option<Value>, query: OAuthCallbackQuery) -> (OAuthCallbackResponse, Vec<TokenExchangeParams>) {
    let env = ServiceEnv::new("p1");
    let out = run_callback(
        session,
        query,
        MCPConnectResult { success: true, tools: vec![tool("t", Some(json!({})))], ..Default::default() },
        token_response(Some("rt"), Some(3600), Some("profile")),
        env.model_service.clone(),
    );
    reset_router_state();
    out
}

// ----------------------------------------------------------- the issuer is checked

#[test]
#[serial(mcp_global)]
fn test_a_matching_issuer_proceeds() {
    let (resp, exchanged) = callback(Some(session(Some(ISSUER), true)), params(Some("c"), "st", Some(ISSUER), None));
    assert_eq!(resp.status_code(), 200, "{}", resp.body());
    assert_eq!(exchanged.len(), 1);
}

#[test]
#[serial(mcp_global)]
fn test_a_mismatched_issuer_never_spends_the_code() {
    // The whole point: an attacker's code must not reach the honest server.
    let (resp, exchanged) = callback(
        Some(session(Some(ISSUER), true)),
        params(Some("c"), "st", Some("https://evil.example.com"), None),
    );
    assert_eq!(resp.status_code(), 400);
    assert!(exchanged.is_empty(), "the authorization code was presented anyway");
}

#[test]
#[serial(mcp_global)]
fn test_the_rejection_names_both_issuers() {
    let (resp, _) = callback(
        Some(session(Some(ISSUER), true)),
        params(Some("c"), "st", Some("https://evil.example.com"), None),
    );
    let body = resp.body();
    assert!(body.contains("clerk.getclera.com"), "{body}");
    assert!(body.contains("evil.example.com"), "{body}");
}

#[test]
#[serial(mcp_global)]
fn test_a_missing_issuer_is_refused_when_the_server_promised_one() {
    let (resp, exchanged) = callback(Some(session(Some(ISSUER), true)), params(Some("c"), "st", None, None));
    assert_eq!(resp.status_code(), 400);
    assert!(exchanged.is_empty());
}

#[test]
#[serial(mcp_global)]
fn test_a_missing_issuer_is_fine_when_none_was_promised() {
    let (resp, exchanged) = callback(Some(session(Some(ISSUER), false)), params(Some("c"), "st", None, None));
    assert_eq!(resp.status_code(), 200, "{}", resp.body());
    assert_eq!(exchanged.len(), 1);
}

#[test]
#[serial(mcp_global)]
fn test_metadata_with_no_issuer_at_all_does_not_block_the_flow() {
    // Pre-RFC-9207 servers stay reachable.
    let (resp, exchanged) = callback(Some(session(None, false)), params(Some("c"), "st", None, None));
    assert_eq!(resp.status_code(), 200, "{}", resp.body());
    assert_eq!(exchanged.len(), 1);
}

#[test]
#[serial(mcp_global)]
fn test_a_server_that_promises_an_issuer_but_publishes_none_is_refused() {
    let (resp, exchanged) = callback(Some(session(None, true)), params(Some("c"), "st", Some(ISSUER), None));
    assert_eq!(resp.status_code(), 400);
    assert!(exchanged.is_empty());
}

// ------------------------------------------------------ it runs before everything else

#[test]
#[serial(mcp_global)]
fn test_an_error_response_is_checked_too() {
    // Otherwise the invalid_scope retry reissues on an attacker's say-so.
    let (resp, _) = callback(
        Some(session(Some(ISSUER), true)),
        params(None, "st", Some("https://evil.example.com"), Some("invalid_scope")),
    );
    assert_eq!(resp.status_code(), 400);
    // Not the 302 the retry would have produced.
    assert!(resp.body().contains("did not come from the authorization server"), "{}", resp.body());
}

#[test]
#[serial(mcp_global)]
fn test_a_genuine_error_response_still_retries() {
    let (resp, _) = callback(Some(session(Some(ISSUER), true)), params(None, "st", Some(ISSUER), Some("invalid_scope")));
    assert_eq!(resp.status_code(), 302);
}

#[test]
#[serial(mcp_global)]
fn test_an_unknown_state_is_left_to_the_session_check() {
    let (resp, exchanged) = callback(None, params(Some("c"), "st", Some(ISSUER), None));
    assert_eq!(resp.status_code(), 400);
    assert!(resp.body().contains("Session expired"), "{}", resp.body());
    assert!(exchanged.is_empty());
}

// ------------------------------------------------------------- metadata parsing

#[test]
fn test_the_flag_is_read_from_the_document() {
    let fake = FakeHttp::always(ok_json(json!({
        "issuer": ISSUER,
        "authorization_endpoint": format!("{ISSUER}/oauth/authorize"),
        "token_endpoint": format!("{ISSUER}/oauth/token"),
        "authorization_response_iss_parameter_supported": true,
    })));
    let mut svc = MCPOAuthService::new(None);
    svc.set_transport(Some(fake));
    let meta = svc.discover_oauth_metadata(ISSUER).unwrap();
    assert!(meta.authorization_response_iss_parameter_supported);
    assert_eq!(meta.issuer.as_deref(), Some(ISSUER));
}

#[test]
fn test_it_defaults_to_false_when_absent() {
    assert!(!OAuthMetadata::default().authorization_response_iss_parameter_supported);
}
