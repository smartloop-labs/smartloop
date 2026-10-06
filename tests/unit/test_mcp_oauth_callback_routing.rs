//! Port of `tests/unit/test_mcp_oauth_callback_routing.py`.
//!
//! The OAuth redirect has to come back to the process that can save it.
//!
//! Registration is project-routed, so the pending session — client_secret,
//! code_verifier, the curated metadata — is in the child's memory, and only the
//! child's `save()` writes that project's `project.json`.  The redirect target
//! was built from `request.base_url` and carried no project, which made it the
//! one MCP route the supervisor answered itself: it looked up a session it did
//! not have, and on the path where it did have one it would have written
//! nothing.

// Every test here holds both serial keys: the default one (the process-scope
// globals `pin_project` / `delegate_projects` and env vars shared with the
// workspace tests) and `mcp_global` (the router's pending OAuth sessions and
// hooks, shared with the other OAuth tests that also use the state "st").
use serial_test::serial;

use smartloop::api::routers::mcp_servers as module;
use smartloop::supervisor::routing::{RequestScope, project_to_proxy};
use smartloop::workspace::context::{delegate_projects, own_project, pin_project};

use crate::shared::workspace_support::{EnvGuard, ProcessScopeReset};

fn scope(path: &str, method: &str) -> RequestScope {
    RequestScope {
        method: method.to_string(),
        path: path.to_string(),
        query_string: Vec::new(),
        headers: Vec::new(),
    }
}

mod the_redirect_target_is_addressed_to_the_supervisor {
    use super::*;

    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_it_names_the_project() {
        let _reset = ProcessScopeReset::new();
        let url = module::default_callback_url("p1");
        assert!(url.ends_with("/v1/projects/p1/mcp/oauth/callback"));
    }

    /// A child serves on an ephemeral port that is reaped when idle.
    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_it_uses_the_public_port_not_the_childs() {
        let _reset = ProcessScopeReset::new();
        // `AppSettings()` patched to `api_port=38540`.
        let _port = EnvGuard::set("SLP_API_PORT", "38540");
        assert_eq!(
            module::default_callback_url("p1"),
            "http://localhost:38540/v1/projects/p1/mcp/oauth/callback"
        );
    }

    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_the_supervisor_forwards_it_to_the_owning_child() {
        let _reset = ProcessScopeReset::new();
        let _port = EnvGuard::set("SLP_API_PORT", "38540");
        let url = module::default_callback_url("p1");
        let path = url.split("localhost:38540").nth(1).expect("the public port").to_string();
        assert_eq!(project_to_proxy(&scope(&path, "GET")).as_deref(), Some("p1"));
    }

    /// Which is why nothing generates it any more.
    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_the_unscoped_path_still_stops_at_the_supervisor() {
        let _reset = ProcessScopeReset::new();
        assert_eq!(project_to_proxy(&scope("/v1/mcp/oauth/callback", "GET")), None);
    }
}

/// `can_persist_project` mirrors what `Workspace::save()` will do.
///
/// Python patched `pinned_project_id` / `projects_delegated` /
/// `owned_project_id` on the context module; here the real process state is
/// set instead, which is what those accessors read.
mod who_may_finish_the_flow {
    use super::*;

    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_the_child_pinned_to_the_project_may() {
        let _reset = ProcessScopeReset::new();
        pin_project(Some("p1"));
        assert!(module::can_persist_project(Some("p1")));
    }

    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_a_child_pinned_to_another_project_may_not() {
        let _reset = ProcessScopeReset::new();
        pin_project(Some("p2"));
        assert!(!module::can_persist_project(Some("p1")));
    }

    /// It writes the workspace record and no project file at all.
    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_a_delegating_supervisor_may_not() {
        let _reset = ProcessScopeReset::new();
        delegate_projects(true);
        assert!(!module::can_persist_project(Some("p1")));
    }

    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_a_supervisor_that_claimed_the_project_may() {
        let _reset = ProcessScopeReset::new();
        delegate_projects(true);
        let _owned = own_project(Some("p1"));
        assert!(module::can_persist_project(Some("p1")));
    }

    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_an_unsupervised_server_may() {
        let _reset = ProcessScopeReset::new();
        assert!(module::can_persist_project(Some("p1")));
    }

    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_a_session_with_no_project_may_not() {
        let _reset = ProcessScopeReset::new();
        assert!(!module::can_persist_project(None));
    }
}

/// An authorization code is single-use: failing after the exchange leaves the
/// user with nothing to retry.
mod the_callback_refuses_rather_than_spend_the_code {
    use super::*;

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::{Map, Value, json};
    use smartloop::api::routers::mcp_servers::{
        McpRouteHooks, OAuthCallbackQuery, OAuthCallbackResponse,
    };
    use smartloop::api::services::model_service::ModelService;
    use smartloop::mcp::oauth::MCPOAuthError;

    /// Removes the exchange hook again when the test ends.
    struct HooksReset;

    impl Drop for HooksReset {
        fn drop(&mut self) {
            module::set_hooks(McpRouteHooks::default());
            module::PENDING_OAUTH_SESSIONS.lock().clear();
            module::COMPLETED_OAUTH_RESPONSES.lock().clear();
        }
    }

    fn setup_method() {
        module::PENDING_OAUTH_SESSIONS.lock().clear();
        module::COMPLETED_OAUTH_RESPONSES.lock().clear();
    }

    fn now() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }

    async fn call(
        routed_project_id: Option<&str>,
        session_project_id: &str,
        delegated: bool,
    ) -> (OAuthCallbackResponse, usize) {
        let session = json!({
            "created_at": now(), "project_id": session_project_id,
            "server_url": "https://mcp.notion.com/mcp", "state": "st",
            "code_verifier": "v", "client_id": "c", "client_secret": "s",
            "token_endpoint": "https://t", "authorization_endpoint": "https://a",
            "resource": null, "callback_url": "https://cb", "name": "notion",
            "return_url": null, "description": null, "tags": null,
            "catalog_slug": null,
        });
        let session: Map<String, Value> = session.as_object().cloned().unwrap();
        module::PENDING_OAUTH_SESSIONS.lock().insert("st".to_string(), session);

        pin_project(None);
        delegate_projects(delegated);

        let exchanged = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&exchanged);
        module::set_hooks(McpRouteHooks {
            exchange_oauth_code: Some(Arc::new(move |_params| {
                counter.fetch_add(1, Ordering::SeqCst);
                Err(MCPOAuthError::Other("the exchange must not run".to_string()))
            })),
            ..Default::default()
        });

        let query = OAuthCallbackQuery {
            code: Some("abc".to_string()),
            state: Some("st".to_string()),
            error: None,
            ..Default::default()
        };
        let response =
            module::complete_oauth_redirect(query, ModelService::new_bare(), routed_project_id).await;
        let calls = exchanged.load(Ordering::SeqCst);
        (response, calls)
    }

    #[tokio::test]
    #[serial]
    #[serial(mcp_global)]
    async fn test_a_delegating_supervisor_refuses_before_exchanging() {
        let _reset = ProcessScopeReset::new();
        let _hooks = HooksReset;
        setup_method();
        let (response, exchanged) = call(None, "p1", true).await;
        assert_eq!(response.status_code(), 500);
        assert_eq!(exchanged, 0, "the code was exchanged");
    }

    /// The redirect named one project; the session belongs to another.
    #[tokio::test]
    #[serial]
    #[serial(mcp_global)]
    async fn test_a_mismatched_route_refuses_before_exchanging() {
        let _reset = ProcessScopeReset::new();
        let _hooks = HooksReset;
        setup_method();
        let (response, exchanged) = call(Some("p2"), "p1", false).await;
        assert_eq!(response.status_code(), 500);
        assert_eq!(exchanged, 0, "the code was exchanged");
    }
}
