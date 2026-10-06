//! Port of `tests/unit/test_mcp_oauth_scopes.py`.
//!
//! Which scopes the authorization request asks for.
//!
//! The flow used to join every scope the *authorization server* advertised and
//! ask for all of them, consulting the protected resource's list only when the
//! server published none.  That is backwards twice over.  RFC 9728 §2 defines
//! the PRM list as the scopes "used in authorization requests to request
//! access to this protected resource", while RFC 8414's is a catalog of
//! everything the server knows across every resource it fronts — and §7.2 asks
//! clients for "as limited a scope as possible" either way.
//!
//! Clera is the case that exposed it: the resource wants three scopes, Clerk
//! advertises seven, and a DCR-registered client is not permitted `openid`.  So
//! the authorize call was refused outright with `invalid_scope` on a
//! connection that needed nothing unusual.
//!
//! Python records the keyword arguments `build_authorization_url` was called
//! with; here the real builder runs and its URL's query is what is read —
//! `scope=None` is a URL without a `scope` parameter.

use serde_json::{json, Value};
use serial_test::serial;

use smartloop::api::model::mcp::MCPRegisterRequest;

use crate::shared::mcp::{register_via_oauth, reset_router_state, OAuthRegistration};

// What the two servers actually publish, fetched from them while diagnosing this.
const CLERA_PRM_SCOPES: &[&str] = &["profile", "email", "offline_access"];
const CLERK_AS_SCOPES: &[&str] = &[
    "openid", "profile", "email", "public_metadata", "private_metadata", "offline_access", "user:org:read",
];

/// Drive the register endpoint down its OAuth path and capture the authorize
/// call.  `prm = None` publishes no protected-resource metadata at all.
fn register(prm: Option<Value>, as_scopes: &[&str]) -> OAuthRegistration {
    let out = register_via_oauth(
        prm,
        json!({
            "issuer": "https://clerk.getclera.com",
            "authorization_endpoint": "https://clerk.getclera.com/oauth/authorize",
            "token_endpoint": "https://clerk.getclera.com/oauth/token",
            "registration_endpoint": "https://clerk.getclera.com/oauth/register",
            "scopes_supported": as_scopes,
        }),
        MCPRegisterRequest {
            server_url: Some("https://mcp.getclera.com/".into()),
            server_type: "remote".into(),
            ..Default::default()
        },
        None,
    );
    reset_router_state();
    assert!(out.response.is_ok(), "register failed: {:?}", out.response.as_ref().err());
    out
}

fn clera_prm() -> Value {
    json!({
        "resource": "https://mcp.getclera.com/",
        "authorization_servers": ["https://clerk.getclera.com"],
        "scopes_supported": CLERA_PRM_SCOPES,
    })
}

/// The `scope` the authorization request carried (`None` when absent).
fn requested_scope(out: &OAuthRegistration) -> Option<String> {
    out.authorize.get("scope").and_then(Value::as_str).map(str::to_string)
}

// ------------------------------------------------------------- scope selection

#[test]
#[serial(mcp_global)]
fn test_the_resource_list_is_what_is_asked_for() {
    // The Clera case, end to end: three scopes requested, not Clerk's seven.
    let out = register(Some(clera_prm()), CLERK_AS_SCOPES);
    assert_eq!(requested_scope(&out).as_deref(), Some("profile email offline_access"));
}

#[test]
#[serial(mcp_global)]
fn test_a_scope_only_the_server_advertises_is_never_requested() {
    // `openid` is in Clerk's catalog and forbidden to DCR clients.
    let out = register(Some(clera_prm()), CLERK_AS_SCOPES);
    let scope = requested_scope(&out).expect("a scope was requested");
    for catalog_only in ["openid", "private_metadata", "public_metadata", "user:org:read"] {
        assert!(!scope.contains(catalog_only), "{catalog_only} in {scope}");
    }
}

#[test]
#[serial(mcp_global)]
fn test_a_resource_declaring_nothing_asks_for_nothing() {
    // No parameter at all, so the server applies its own default.
    let out = register(
        Some(json!({
            "resource": "https://mcp.example.com/mcp",
            "authorization_servers": ["https://as.example.com"],
            "scopes_supported": [],
        })),
        CLERK_AS_SCOPES,
    );
    assert_eq!(requested_scope(&out), None);
}

#[test]
#[serial(mcp_global)]
fn test_no_resource_metadata_at_all_asks_for_nothing() {
    let out = register(None, CLERK_AS_SCOPES);
    assert_eq!(requested_scope(&out), None);
}

#[test]
#[serial(mcp_global)]
fn test_the_server_catalog_is_not_a_fallback() {
    // Even with no PRM scopes and a rich catalog, we do not help ourselves.
    let out = register(
        Some(json!({
            "resource": "https://mcp.example.com/mcp",
            "authorization_servers": ["https://as.example.com"],
        })),
        &["read", "write", "admin"],
    );
    assert_eq!(requested_scope(&out), None);
}

// ------------------------------------------------- the session carries the request

/// The callback needs both to reissue an authorization the server refused.
#[test]
#[serial(mcp_global)]
fn test_scope_and_challenge_survive_the_redirect() {
    let out = register(Some(clera_prm()), CLERK_AS_SCOPES);
    assert_eq!(out.session["scope"], "profile email offline_access");
    assert_eq!(out.session["code_challenge"], out.authorize["code_challenge"]);
}

#[test]
#[serial(mcp_global)]
fn test_an_empty_scope_is_recorded_as_none() {
    let out = register(None, CLERK_AS_SCOPES);
    assert_eq!(out.session["scope"], Value::Null);
}
