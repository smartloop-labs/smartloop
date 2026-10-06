//! Port of `tests/unit/test_mcp_register_off_event_loop.py`.
//!
//! MCP registration must not run on the event loop.
//!
//! `register_mcp_server` is `async`, but everything under it blocks: a child
//! process launch with a 2s startup poll, up to three launch attempts through
//! the uvx/mcp<2 retry ladder, then local inference and an embedding pass.
//! Left on the runtime thread that stalls every other request — and every open
//! SSE chat stream — for the better part of a minute.

use std::sync::Arc;
use std::thread::ThreadId;

use parking_lot::Mutex;
use serde_json::{json, Value};

use smartloop::api::model::mcp::MCPRegisterRequest;
use smartloop::api::routers::mcp_servers::HTTPException;
use smartloop::api::routers::mcp_servers::register_mcp_server;
use smartloop::api::services::model_service::{ModelServiceError, ModelServiceHooks};
use smartloop::api::services::ModelService;
use smartloop::mcp::client::MCPDiscoveryError;

fn result() -> Value {
    json!({
        "id": "srv-1", "name": "demo", "enabled": true, "tools": [],
        "server_type": "local", "args": [],
    })
}

/// A model service whose `add_local_mcp_server` is `side_effect`.
fn service(
    side_effect: impl Fn(&str, Option<&str>) -> anyhow::Result<Value> + Send + Sync + 'static,
) -> Arc<ModelService> {
    let svc = ModelService::new_bare();
    svc.set_hooks(ModelServiceHooks { add_local_mcp_server: Some(Arc::new(side_effect)), ..Default::default() });
    svc
}

async fn register(svc: Arc<ModelService>, request: MCPRegisterRequest) -> Result<(), HTTPException> {
    register_mcp_server("proj-1", request, svc).await.map(|_| ())
}

fn local_request() -> MCPRegisterRequest {
    MCPRegisterRequest {
        server_type: "local".to_string(),
        command: Some("uvx".to_string()),
        args: Some(vec!["demo-mcp".to_string()]),
        ..Default::default()
    }
}

mod test_registration_runs_off_the_loop {
    use super::*;

    #[tokio::test]
    async fn test_blocking_service_call_runs_on_a_worker_thread() {
        let seen: Arc<Mutex<Vec<ThreadId>>> = Arc::default();
        let record = Arc::clone(&seen);
        let svc = service(move |_command, _project_id| {
            record.lock().push(std::thread::current().id());
            Ok(result())
        });
        // The thread driving the request — the "event loop" (a
        // current-thread runtime under #[tokio::test]).
        let loop_thread = std::thread::current().id();

        let _ = register(svc, local_request()).await;

        let seen = seen.lock();
        assert!(!seen.is_empty(), "add_local_mcp_server was never called");
        assert_ne!(seen[0], loop_thread);
    }

    /// The error has to survive the hop to the worker, or a bad package would
    /// surface as a 500 instead of the actionable 422.
    #[tokio::test]
    async fn test_discovery_failure_still_maps_to_422() {
        let svc = service(|_, _| Err(MCPDiscoveryError("no such package".to_string()).into()));

        let exc = register(svc, local_request()).await.expect_err("an HTTPException");
        assert_eq!(exc.status_code, 422);
        assert!(exc.detail.to_string().contains("no such package"), "{:?}", exc.detail);
    }

    #[tokio::test]
    async fn test_value_error_still_maps_to_400() {
        let svc = service(|_, _| Err(ModelServiceError::Value("bad args".to_string()).into()));

        let exc = register(svc, local_request()).await.expect_err("an HTTPException");
        assert_eq!(exc.status_code, 400);
    }
}
