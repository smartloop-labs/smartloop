//! Port of `tests/unit/test_mcp_oauth_discovery.py`.
//!
//! Tests for MCP OAuth metadata discovery (RFC 9728 / RFC 8414).
//!
//! These exercise the discovery URL handling that unblocks path-scoped OAuth
//! servers such as Trello (https://mcp.trello.com/v1).  A fake HTTP client is
//! injected via `MCPOAuthService::with_transport` (Python:
//! `MCPOAuthService(http_client=...)`) so no real network calls are made.

use std::sync::Arc;

use serde_json::json;

use smartloop::mcp::client::{HttpResponse, MCPClient};
use smartloop::mcp::oauth::{MCPOAuthError, MCPOAuthService};

use crate::shared::mcp::{ok_json, FakeHttp};

/// Maps an exact URL -> response; unmapped URLs answer 404.  The fake records
/// every requested URL so tests can assert ordering.
fn service(routes: Vec<(&str, HttpResponse)>) -> (MCPOAuthService, Arc<FakeHttp>) {
    let fake = FakeHttp::routes(routes);
    (MCPOAuthService::with_transport(fake.clone()), fake)
}

// ------------------------------------------------------------ PRM discovery

#[test]
fn test_advertised_metadata_url_is_tried_first() {
    let advertised = "https://mcp.trello.com/v1/custom-prm";
    let (svc, fake) = service(vec![(
        advertised,
        ok_json(json!({
            "resource": "https://mcp.trello.com/v1",
            "authorization_servers": ["https://auth.atlassian.com/abc123"],
        })),
    )]);
    let prm = svc
        .discover_protected_resource_metadata("https://mcp.trello.com/v1", Some(advertised))
        .unwrap();
    assert_eq!(prm.authorization_servers, vec!["https://auth.atlassian.com/abc123".to_string()]);
    // The advertised URL must be attempted before any guessed well-known URL.
    assert_eq!(fake.get_calls()[0], advertised);
}

#[test]
fn test_base_503_falls_through_to_path_scoped_200() {
    // Trello's base PRM URL persistently 503s; the path-scoped one returns 200.
    let base = "https://mcp.trello.com/.well-known/oauth-protected-resource";
    let scoped = "https://mcp.trello.com/.well-known/oauth-protected-resource/v1";
    let (svc, fake) = service(vec![
        (base, HttpResponse::new(503, "")),
        (
            scoped,
            ok_json(json!({
                "authorization_servers": ["https://auth.atlassian.com/abc123"],
                "scopes_supported": ["read", "write"],
            })),
        ),
    ]);
    let prm = svc.discover_protected_resource_metadata("https://mcp.trello.com/v1", None).unwrap();
    assert_eq!(prm.authorization_servers, vec!["https://auth.atlassian.com/abc123".to_string()]);
    // The 503 did not abort discovery.
    assert!(fake.get_calls().iter().any(|u| u == base));
}

#[test]
fn test_metadata_url_failure_falls_back_to_guessed() {
    let advertised = "https://mcp.trello.com/v1/custom-prm";
    let base = "https://mcp.trello.com/.well-known/oauth-protected-resource";
    let (svc, _fake) = service(vec![
        (advertised, HttpResponse::new(503, "")),
        (base, ok_json(json!({"authorization_servers": ["https://as.example.com"]}))),
    ]);
    let prm = svc
        .discover_protected_resource_metadata("https://mcp.trello.com/v1", Some(advertised))
        .unwrap();
    assert_eq!(prm.authorization_servers, vec!["https://as.example.com".to_string()]);
}

#[test]
fn test_all_endpoints_fail_raises() {
    // Everything 404s.
    let (svc, _fake) = service(vec![]);
    let err = svc.discover_protected_resource_metadata("https://mcp.trello.com/v1", None).unwrap_err();
    assert!(
        matches!(err, MCPOAuthError::ResourceMetadataDiscovery(_)),
        "expected ResourceMetadataDiscoveryError, got {err:?}"
    );
}

#[test]
fn test_a_transport_error_falls_through_too() {
    // The fake's routes may also raise (Python maps a URL to an exception):
    // a connection error on one candidate must not abort discovery either.
    let base = "https://mcp.trello.com/.well-known/oauth-protected-resource";
    let scoped = "https://mcp.trello.com/.well-known/oauth-protected-resource/v1";
    let fake = FakeHttp::new(move |_, url, _| {
        if url == base {
            Err("connection refused".to_string())
        } else if url == scoped {
            Ok(ok_json(json!({"authorization_servers": ["https://as.example.com"]})))
        } else {
            Ok(HttpResponse::new(404, ""))
        }
    });
    let svc = MCPOAuthService::with_transport(fake.clone());
    let prm = svc.discover_protected_resource_metadata("https://mcp.trello.com/v1", None).unwrap();
    assert_eq!(prm.authorization_servers, vec!["https://as.example.com".to_string()]);
    assert_eq!(fake.get_calls(), vec![base.to_string(), scoped.to_string()]);
}

// ------------------------------------------------------ OAuth metadata discovery

#[test]
fn test_path_append_form_resolves_for_atlassian() {
    let issuer = "https://auth.atlassian.com/abc123";
    let working = format!("{issuer}/.well-known/oauth-authorization-server");
    let (svc, fake) = service(vec![
        // Both well-known-inserted forms fail for Atlassian.
        ("https://auth.atlassian.com/.well-known/oauth-authorization-server", HttpResponse::new(404, "")),
        (
            "https://auth.atlassian.com/.well-known/oauth-authorization-server/abc123",
            HttpResponse::new(404, ""),
        ),
        (
            working.as_str(),
            ok_json(json!({
                "issuer": issuer,
                "authorization_endpoint": format!("{issuer}/authorize"),
                "token_endpoint": format!("{issuer}/token"),
                "registration_endpoint": format!("{issuer}/register"),
            })),
        ),
    ]);
    let meta = svc.discover_oauth_metadata(issuer).unwrap();
    assert_eq!(meta.registration_endpoint.as_deref(), Some(format!("{issuer}/register").as_str()));
    assert!(fake.get_calls().contains(&working));
}

#[test]
fn test_bare_url_does_not_shadow_path_scoped_dcr_metadata() {
    // Atlassian's real shape: the bare URL 200s without a registration_endpoint
    // while the path-scoped URL carries DCR.  We must return the DCR-capable doc.
    let issuer = "https://auth.atlassian.com/abc123";
    let host_insert_with_path = "https://auth.atlassian.com/.well-known/oauth-authorization-server/abc123";
    let bare = "https://auth.atlassian.com/.well-known/oauth-authorization-server";
    let (svc, _fake) = service(vec![
        (
            bare,
            ok_json(json!({
                "issuer": "https://auth.atlassian.com",
                "authorization_endpoint": "https://auth.atlassian.com/authorize",
                "token_endpoint": "https://auth.atlassian.com/token",
                // no registration_endpoint
            })),
        ),
        (
            host_insert_with_path,
            ok_json(json!({
                "issuer": issuer,
                "authorization_endpoint": format!("{issuer}/authorize"),
                "token_endpoint": format!("{issuer}/token"),
                "registration_endpoint": format!("{issuer}/dcr/register"),
            })),
        ),
    ]);
    let meta = svc.discover_oauth_metadata(issuer).unwrap();
    assert_eq!(meta.registration_endpoint.as_deref(), Some(format!("{issuer}/dcr/register").as_str()));
}

#[test]
fn test_no_path_issuer_resolves_via_host_insert() {
    // Regression guard: issuers without a path still resolve at the base URL.
    let base = "https://mcp.linear.app/.well-known/oauth-authorization-server";
    let (svc, _fake) = service(vec![(
        base,
        ok_json(json!({
            "issuer": "https://mcp.linear.app",
            "authorization_endpoint": "https://mcp.linear.app/authorize",
            "token_endpoint": "https://mcp.linear.app/token",
            "registration_endpoint": "https://mcp.linear.app/register",
        })),
    )]);
    let meta = svc.discover_oauth_metadata("https://mcp.linear.app").unwrap();
    assert_eq!(meta.registration_endpoint.as_deref(), Some("https://mcp.linear.app/register"));
}

#[test]
fn test_no_path_issuer_only_emits_host_insert_url() {
    let svc = MCPOAuthService::new(None);
    let urls = svc.get_metadata_urls_to_try("https://mcp.linear.app");
    assert_eq!(urls, vec!["https://mcp.linear.app/.well-known/oauth-authorization-server".to_string()]);
}

#[test]
fn test_path_issuer_emits_all_candidate_forms() {
    let svc = MCPOAuthService::new(None);
    let urls = svc.get_metadata_urls_to_try("https://auth.atlassian.com/abc123");
    // Path-bearing forms first; the bare (path-stripped) URL is a last resort.
    assert_eq!(
        urls,
        vec![
            "https://auth.atlassian.com/.well-known/oauth-authorization-server/abc123".to_string(),
            "https://auth.atlassian.com/abc123/.well-known/oauth-authorization-server".to_string(),
            "https://auth.atlassian.com/abc123/.well-known/openid-configuration".to_string(),
            "https://auth.atlassian.com/.well-known/oauth-authorization-server".to_string(),
        ]
    );
}

/// `TestOAuthMetadataDiscovery` (a module only because the test name is
/// shared with `TestPRMDiscovery`).
mod oauth_metadata_discovery {
    use super::*;

    #[test]
    fn test_all_endpoints_fail_raises() {
        let (svc, _fake) = service(vec![]);
        let err = svc.discover_oauth_metadata("https://auth.atlassian.com/abc123").unwrap_err();
        assert!(matches!(err, MCPOAuthError::MetadataDiscovery(_)), "expected MetadataDiscoveryError, got {err:?}");
    }
}

// ------------------------------------------------ resource_metadata header parsing

#[test]
fn test_extracts_resource_metadata_url() {
    let header = r#"Bearer resource_metadata="https://mcp.trello.com/v1/prm", resource="https://mcp.trello.com/v1""#;
    assert_eq!(
        MCPClient::parse_resource_metadata_from_www_authenticate(header).as_deref(),
        Some("https://mcp.trello.com/v1/prm")
    );
}

#[test]
fn test_does_not_confuse_with_plain_resource_field() {
    let header = r#"Bearer resource="https://mcp.trello.com/v1""#;
    assert_eq!(MCPClient::parse_resource_metadata_from_www_authenticate(header), None);
    // And the plain-resource parser must not grab the metadata field.
    let header2 = r#"Bearer resource_metadata="https://mcp.trello.com/v1/prm""#;
    assert_eq!(MCPClient::parse_resource_from_www_authenticate(header2), None);
}

#[test]
fn test_unquoted_form() {
    // Mirrors the existing resource= parser: bare value, trailing comma stripped.
    assert_eq!(
        MCPClient::parse_resource_metadata_from_www_authenticate("Bearer resource_metadata=https://x/prm").as_deref(),
        Some("https://x/prm")
    );
    assert_eq!(
        MCPClient::parse_resource_metadata_from_www_authenticate("Bearer resource_metadata=https://x/prm,").as_deref(),
        Some("https://x/prm")
    );
}

#[test]
fn test_empty_header_returns_none() {
    assert_eq!(MCPClient::parse_resource_metadata_from_www_authenticate(""), None);
}
