//! Port of `tests/unit/test_mcp_oauth_error_reporting.py`.
//!
//! What the user is told when a provider refuses authorization.
//!
//! FastAPI binds only the query parameters a route declares.  The callbacks
//! declared `code`, `state` and `error` and nothing else, so the rest of the
//! RFC 6749 §4.1.2.1 error response — `error_description` and `error_uri` —
//! arrived on the redirect and was dropped on the floor.  `error` alone comes
//! from a fixed enum, so the page could only ever say `invalid_scope`: the
//! sentence naming *which* scope went in the bin, and nothing was logged
//! either, because the branch returns before any logging.

use serde_json::json;
use serial_test::serial;

use smartloop::api::routers::mcp_servers::{mcp_oauth_callback, OAuthCallbackQuery};
use smartloop::api::services::model_service::ModelService;
use smartloop::mcp::oauth::{token_error, TokenExchangeError};

use crate::shared::mcp::{block_on, capture_logs, reset_router_state};

fn query(
    error: Option<&str>,
    error_description: Option<&str>,
    error_uri: Option<&str>,
) -> OAuthCallbackQuery {
    OAuthCallbackQuery {
        error: error.map(str::to_string),
        error_description: error_description.map(str::to_string),
        error_uri: error_uri.map(str::to_string),
        ..Default::default()
    }
}

/// Deliver an error redirect; `(status, body)`.
fn redirect(params: OAuthCallbackQuery) -> (u16, String) {
    reset_router_state();
    let resp = block_on(mcp_oauth_callback(params, ModelService::new_bare()));
    (resp.status_code(), resp.body().to_string())
}

// ------------------------------------------------------ authorization error page

#[test]
#[serial(mcp_global)]
fn test_description_reaches_the_page() {
    // The Clera failure: the enum says nothing, the description says 'openid'.
    let (status, body) = redirect(query(
        Some("invalid_scope"),
        Some(
            "The requested scope is invalid, unknown, or malformed. The \
             OAuth 2.0 Client is not allowed to request scope 'openid'.",
        ),
        None,
    ));
    assert_eq!(status, 400);
    assert!(body.contains("invalid_scope"), "{body}");
    assert!(body.contains("not allowed to request scope &#x27;openid&#x27;"), "{body}");
}

#[test]
#[serial(mcp_global)]
fn test_error_alone_still_renders() {
    let (status, body) = redirect(query(Some("access_denied"), None, None));
    assert_eq!(status, 400);
    assert!(body.contains("access_denied"), "{body}");
}

#[test]
#[serial(mcp_global)]
fn test_http_error_uri_becomes_a_link() {
    let (_, body) = redirect(query(Some("invalid_scope"), None, Some("https://clerk.com/docs/oauth/scoped-access")));
    assert!(body.contains(r#"<a href="https://clerk.com/docs/oauth/scoped-access""#), "{body}");
}

#[test]
#[serial(mcp_global)]
fn test_non_http_error_uri_is_never_linked() {
    // `error_uri` arrives from a server we discovered, not one we configured.
    let (_, body) = redirect(query(Some("invalid_scope"), None, Some("javascript:alert(1)")));
    assert!(!body.contains("<a href"), "{body}");
    assert!(body.contains("javascript:alert(1)"), "{body}");
}

#[test]
#[serial(mcp_global)]
fn test_description_is_escaped() {
    let (_, body) = redirect(query(Some("invalid_scope"), Some("<script>alert(1)</script>"), None));
    assert!(!body.contains("<script>"), "{body}");
    assert!(body.contains("&lt;script&gt;"), "{body}");
}

#[test]
#[serial(mcp_global)]
fn test_the_refusal_is_logged() {
    let (_, logged) = capture_logs(tracing::Level::WARN, || {
        redirect(query(
            Some("invalid_scope"),
            Some("not allowed to request scope 'openid'"),
            Some("https://example.com/why"),
        ))
    });
    assert!(logged.contains("smartloop.api.routers.mcp_servers"), "{logged}");
    assert!(logged.contains("invalid_scope"), "{logged}");
    assert!(logged.contains("openid"), "{logged}");
    assert!(logged.contains("https://example.com/why"), "{logged}");
}

// ------------------------------------------------------ token endpoint error body

/// RFC 6749 §5.2 carries the same three fields on the token endpoint.
fn error(payload: serde_json::Value, status: u16) -> TokenExchangeError {
    token_error(status, &payload.to_string(), "Token exchange failed")
}

#[test]
fn test_description_and_uri_are_both_reported() {
    let err = error(
        json!({
            "error": "invalid_scope",
            "error_description": "scope 'openid' is not permitted",
            "error_uri": "https://example.com/why",
        }),
        400,
    );
    assert_eq!(err.error_code.as_deref(), Some("invalid_scope"));
    assert!(err.to_string().contains("scope 'openid' is not permitted"));
    assert!(err.to_string().contains("https://example.com/why"));
}

#[test]
fn test_the_error_code_still_drives_dead_grant_detection() {
    assert!(error(json!({"error": "invalid_grant"}), 400).grant_is_dead());
    assert!(!error(json!({"error": "invalid_scope"}), 400).grant_is_dead());
}

#[test]
fn test_a_body_that_is_not_json_leaves_the_status_line_as_the_story() {
    let err = token_error(500, "<html>nope</html>", "Token refresh failed");
    assert_eq!(err.error_code, None);
    assert!(err.to_string().contains("Token refresh failed: HTTP 500"));
}

#[test]
fn test_a_bare_error_code_is_used_as_its_own_description() {
    let err = error(json!({"error": "invalid_client"}), 400);
    assert!(err.to_string().contains("invalid_client"));
}
