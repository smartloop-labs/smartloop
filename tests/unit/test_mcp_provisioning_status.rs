//! Port of `tests/unit/test_mcp_provisioning_status.py`.
//!
//! What a client polls to know a connection is ready to use.
//!
//! Registering an MCP server returns before it is usable: tool discovery runs
//! on the MCP task.  Until it finishes the connection can answer nothing — so
//! the API has to say so.  The block is still called `indexing` on the wire,
//! because that is the field clients read; what it reports is provisioning.

use std::sync::Arc;

use serde_json::{Map, Value};

use smartloop::api::services::model_service::{ModelService, ModelServiceError, ModelServiceHooks};
use smartloop::workspace::{MCPServer, MCPServerStatus, MCPToolDefinition};

use crate::shared::model_service_b::{project, service_with_workspace};

fn server() -> MCPServer {
    MCPServer {
        id: "srv1".into(),
        name: "notion".into(),
        catalog_slug: Some("notion".into()),
        tools: vec![MCPToolDefinition {
            name: "search".into(),
            description: Some("d".into()),
            input_schema: Some(Map::new()),
            ..Default::default()
        }],
        ..MCPServer::default()
    }
}

#[test]
fn test_a_server_still_provisioning_is_in_progress() {
    let state = ModelService::mcp_provisioning_state(&MCPServer { status: MCPServerStatus::Pending, ..server() });
    assert_eq!(state["state"], "provisioning");
    assert_eq!(state["in_progress"], Value::Bool(true));
}

#[test]
fn test_a_discovered_server_is_ready() {
    let state = ModelService::mcp_provisioning_state(&server());
    assert_eq!(state["state"], "ready");
    assert_eq!(state["in_progress"], Value::Bool(false));
    assert_eq!(state["error"], Value::Null);
}

#[test]
fn test_a_failed_provisioning_carries_its_reason() {
    let state = ModelService::mcp_provisioning_state(&MCPServer {
        status: MCPServerStatus::Failed,
        last_error: Some("401 from server".into()),
        ..server()
    });
    assert_eq!(state["state"], "failed");
    assert_eq!(state["in_progress"], Value::Bool(false));
    assert_eq!(state["error"], "401 from server");
}

/// The tools discovery found are the catalog a request is resolved against,
/// so "ready with 0 tools" is a state worth being able to see.
#[test]
fn test_the_tool_count_is_reported() {
    assert_eq!(ModelService::mcp_provisioning_state(&server())["tools"], 1);
    assert_eq!(ModelService::mcp_provisioning_state(&MCPServer { tools: vec![], ..server() })["tools"], 0);
}

/// A service whose project `p1` holds `servers`.
fn svc(servers: Vec<MCPServer>) -> Arc<ModelService> {
    let mut p1 = project("p1", "/tmp/p1", true);
    p1.name = Some("proj".into());
    p1.mcp_servers = servers;
    let svc = service_with_workspace("/tmp", vec![p1.clone()], vec![]);
    let mut hooks = ModelServiceHooks::default();
    hooks.get_project = Some(Arc::new(move |_id: &str| Ok(p1.clone())));
    svc.set_hooks(hooks);
    svc
}

#[test]
fn test_the_server_dict_carries_its_indexing_block() {
    let svc = svc(vec![MCPServer { status: MCPServerStatus::Pending, ..server() }]);
    let result = svc.mcp_server_status("srv1", Some("p1")).unwrap();
    assert_eq!(result["id"], "srv1");
    assert_eq!(result["indexing"]["state"], "provisioning");
    assert_eq!(result["indexing"]["in_progress"], Value::Bool(true));
}

#[test]
fn test_an_unknown_server_is_a_not_found() {
    let svc = svc(vec![server()]);
    let err = svc.mcp_server_status("nope", Some("p1")).expect_err("expected a ValueError the router maps to 404");
    assert!(matches!(err.downcast_ref::<ModelServiceError>(), Some(ModelServiceError::Value(_))));
    assert!(err.to_string().contains("nope"));
}

/// Otherwise watching a freshly registered connection costs one request per
/// server on every poll.
#[test]
fn test_the_list_carries_the_same_block() {
    let svc = svc(vec![server()]);
    let servers = svc.list_mcp_servers_with_status("p1").unwrap();
    let states: Vec<&Value> = servers.iter().map(|s| &s["indexing"]["state"]).collect();
    assert_eq!(states, vec!["ready"]);
}

#[test]
fn test_system_servers_stay_out_of_the_list() {
    let svc = svc(vec![server(), MCPServer { id: "sys".into(), name: "smartloop".into(), system: true, ..server() }]);
    let servers = svc.list_mcp_servers_with_status("p1").unwrap();
    let ids: Vec<&Value> = servers.iter().map(|s| &s["id"]).collect();
    assert_eq!(ids, vec!["srv1"]);
}
