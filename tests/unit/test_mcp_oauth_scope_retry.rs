//! Port of `tests/unit/test_mcp_oauth_scope_retry.py`.
//!
//! Reissuing an authorization the server refused on scope.
//!
//! The PRM list is what RFC 9728 §2 says to ask for, but it describes the
//! resource and not the client: a client registered through DCR can be
//! permitted less than the resource advertises.  When that happens the error
//! names at most one offending scope, so there is nothing to negotiate against
//! — the one useful retry asks for none and lets the server apply the default
//! for the client it registered.

use std::collections::HashMap;

use serde_json::{json, Map, Value};
use serial_test::serial;

use smartloop::api::routers::mcp_servers::{
    self as router_mod, OAuthCallbackQuery, OAuthCallbackResponse, OAuthSession,
};
use smartloop::api::services::ModelService;

/// A pending session as `/mcp/oauth/start` leaves it, with *overrides* applied.
fn session(overrides: Value) -> OAuthSession {
    let mut base: Map<String, Value> = match json!({
        "created_at": 0.0,
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
    }) {
        Value::Object(map) => map,
        _ => unreachable!(),
    };
    if let Value::Object(overrides) = overrides {
        for (k, v) in overrides {
            base.insert(k, v);
        }
    }
    base
}

/// Call the callback with an authorization *error* for *state*.
async fn callback(state: Option<&str>, error: &str) -> OAuthCallbackResponse {
    let query = OAuthCallbackQuery {
        state: state.map(str::to_owned),
        error: Some(error.to_string()),
        ..Default::default()
    };
    router_mod::mcp_oauth_callback(query, ModelService::new_bare()).await
}

/// Deliver an authorization error to the callback and return the response,
/// with the pending session left under `"st"` afterwards.
async fn refuse(error: &str, session: Option<OAuthSession>, state: Option<&str>) -> (OAuthCallbackResponse, Option<OAuthSession>) {
    router_mod::PENDING_OAUTH_SESSIONS.lock().clear();
    router_mod::COMPLETED_OAUTH_RESPONSES.lock().clear();
    if let Some(session) = session {
        router_mod::PENDING_OAUTH_SESSIONS.lock().insert("st".to_string(), session);
    }
    let resp = callback(state, error).await;
    let left = router_mod::PENDING_OAUTH_SESSIONS.lock().get("st").cloned();
    (resp, left)
}

/// `parse_qs(urlparse(target).query)`.
fn parse_qs(target: &str) -> HashMap<String, Vec<String>> {
    let url = url::Url::parse(target).expect("redirect target is a URL");
    let mut params: HashMap<String, Vec<String>> = HashMap::new();
    for (k, v) in url.query_pairs() {
        params.entry(k.into_owned()).or_default().push(v.into_owned());
    }
    params
}

fn strs(values: &[&str]) -> Vec<String> {
    values.iter().map(|s| s.to_string()).collect()
}

mod test_retry {
    use super::*;

    #[tokio::test]
    #[serial(mcp_global)]
    async fn test_a_refused_scope_is_reissued_without_one() {
        let (resp, _) = refuse("invalid_scope", Some(session(json!({}))), Some("st")).await;
        assert_eq!(resp.status_code(), 302);
        let target = resp.location().expect("a redirect").to_string();
        assert!(target.starts_with("https://clerk.getclera.com/oauth/authorize"), "{target}");
        assert!(!parse_qs(&target).contains_key("scope"));
    }

    #[tokio::test]
    #[serial(mcp_global)]
    async fn test_the_reissue_keeps_everything_else_the_grant_needs() {
        let (resp, _) = refuse("invalid_scope", Some(session(json!({}))), Some("st")).await;
        let params = parse_qs(resp.location().expect("a redirect"));
        assert_eq!(params["state"], strs(&["st"]));
        assert_eq!(params["client_id"], strs(&["cid"]));
        assert_eq!(params["code_challenge"], strs(&["chal"]));
        assert_eq!(params["code_challenge_method"], strs(&["S256"]));
        // RFC 8707 — dropping the indicator would make the retry fail differently.
        assert_eq!(params["resource"], strs(&["https://mcp.getclera.com"]));
    }

    #[tokio::test]
    #[serial(mcp_global)]
    async fn test_the_session_records_what_the_retry_asked_for() {
        let (_, session) = refuse("invalid_scope", Some(session(json!({}))), Some("st")).await;
        let session = session.expect("the session stays pending for the retry");
        assert_eq!(session["scope_retry"], json!(true));
        assert_eq!(session["scope"], Value::Null);
    }

    /// Otherwise a server refusing the empty request bounces the browser forever.
    #[tokio::test]
    #[serial(mcp_global)]
    async fn test_a_second_refusal_gives_up() {
        let (first, _) = refuse("invalid_scope", Some(session(json!({}))), Some("st")).await;
        assert_eq!(first.status_code(), 302);

        // Same session, now carrying the retry mark the first refusal left.
        let second = callback(Some("st"), "invalid_scope").await;
        assert_eq!(second.status_code(), 400);
        assert!(second.body().contains("invalid_scope"));
    }
}

mod test_when_not_to_retry {
    use super::*;

    #[tokio::test]
    #[serial(mcp_global)]
    async fn test_another_error_is_reported_not_retried() {
        let (resp, _) = refuse("access_denied", Some(session(json!({}))), Some("st")).await;
        assert_eq!(resp.status_code(), 400);
        assert!(resp.body().contains("access_denied"));
    }

    /// The request already named no scopes; asking again changes nothing.
    #[tokio::test]
    #[serial(mcp_global)]
    async fn test_nothing_to_drop_means_nothing_to_retry() {
        let (resp, _) = refuse("invalid_scope", Some(session(json!({"scope": null}))), Some("st")).await;
        assert_eq!(resp.status_code(), 400);
    }

    /// A client driving the flow itself owns its own PKCE pair.
    #[tokio::test]
    #[serial(mcp_global)]
    async fn test_a_session_without_a_challenge_is_not_rebuilt() {
        let (resp, _) = refuse("invalid_scope", Some(session(json!({"code_challenge": null}))), Some("st")).await;
        assert_eq!(resp.status_code(), 400);
    }

    #[tokio::test]
    #[serial(mcp_global)]
    async fn test_an_unknown_state_is_reported() {
        let (resp, _) = refuse("invalid_scope", None, Some("st")).await;
        assert_eq!(resp.status_code(), 400);
    }

    #[tokio::test]
    #[serial(mcp_global)]
    async fn test_an_error_without_state_is_reported() {
        let (resp, _) = refuse("invalid_scope", Some(session(json!({}))), None).await;
        assert_eq!(resp.status_code(), 400);
    }
}
