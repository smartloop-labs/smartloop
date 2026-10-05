//! Port of `tests/unit/test_install_system_servers.py`.
//!
//! Regression tests: `install_system_tools_server` must find an existing
//! record by its stable id, not its display name.
//!
//! Records saved under an earlier name (e.g. "Smartloop" before the constant
//! was unified to "smartloop") were never matched by the name-based lookup, so
//! their tool list stayed stale — a project created before grep was added kept
//! advertising a server without it, and tool resolution failed with
//! "Tool 'grep' not found on server 'Smartloop'".
//!
//! The Python tests drove the methods with a mock `self`; here the service
//! holds a real workspace with the project, and the `save_model` seam counts
//! `self._model.save()` calls.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;

use smartloop::agents::planner::{FETCH_SERVER_ID, SYSTEM_TOOLS_SERVER_ID, SYSTEM_TOOLS_SERVER_NAME};
use smartloop::api::config::api_settings;
use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::workspace::{MCPServer, MCPServerConfig, MCPServerType, MCPToolDefinition, Project, Workspace};

use crate::shared::model_service_a::{attach_workspace, bare_service};

/// A service holding `project` in its workspace, and its save counter.
fn svc(mcp_servers: Vec<MCPServer>) -> (Arc<ModelService>, Arc<AtomicUsize>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let project = Project {
        id: "p1".to_string(),
        project_dir: dir.path().to_string_lossy().into_owned(),
        current: true,
        name: Some("p1".to_string()),
        mcp_servers,
        ..Default::default()
    };
    let svc = bare_service();
    let mut workspace = Workspace::new("ws");
    workspace.projects = vec![project];
    attach_workspace(&svc, workspace);
    // The resident model is the default one, so `_ensure_model` is a no-op.
    *svc.model_name.write() = Some(api_settings().default_model);

    let saves = Arc::new(AtomicUsize::new(0));
    let counter = saves.clone();
    svc.set_hooks(ModelServiceHooks {
        save_model: Some(Arc::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })),
        ..Default::default()
    });
    (svc, saves, dir)
}

fn server(server_id: &str, name: &str, tool_names: &[&str]) -> MCPServer {
    MCPServer {
        id: server_id.to_string(),
        name: name.to_string(),
        server_type: MCPServerType::Local,
        config: Some(MCPServerConfig {
            command: "python".to_string(),
            args: vec!["-m".to_string(), "x".to_string()],
            ..Default::default()
        }),
        enabled: true,
        system: true,
        tools: tool_names
            .iter()
            .map(|n| MCPToolDefinition {
                name: n.to_string(),
                description: Some(n.to_string()),
                input_schema: json!({"type": "object"}).as_object().cloned(),
            })
            .collect(),
        ..Default::default()
    }
}

fn system_tool_names() -> Vec<String> {
    ModelService::system_tool_defs().into_iter().map(|t| t.name).collect()
}

fn servers(svc: &ModelService) -> Vec<MCPServer> {
    let ws = svc.workspace_handle().expect("workspace");
    let guard = ws.read();
    guard.project("p1").expect("p1").mcp_servers.clone()
}

#[test]
fn test_stale_renamed_record_is_refreshed_not_duplicated() {
    // pre-grep snapshot
    let stale = server(SYSTEM_TOOLS_SERVER_ID, "Smartloop", &["bash", "file-read", "file-write", "explore"]);
    let (svc, saves, _dir) = svc(vec![stale]);

    svc.install_system_tools_server("p1").expect("install");

    let servers = servers(&svc);
    assert_eq!(servers.len(), 1);
    assert_eq!(servers[0].name, SYSTEM_TOOLS_SERVER_NAME);
    assert!(servers[0].tools.iter().any(|t| t.name == "grep"));
    assert_eq!(saves.load(Ordering::SeqCst), 1);
}

#[test]
fn test_current_record_is_left_untouched() {
    let names = system_tool_names();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let current = server(SYSTEM_TOOLS_SERVER_ID, SYSTEM_TOOLS_SERVER_NAME, &names);
    let (svc, saves, _dir) = svc(vec![current]);

    svc.install_system_tools_server("p1").expect("install");

    assert_eq!(servers(&svc).len(), 1);
    assert_eq!(saves.load(Ordering::SeqCst), 0);
}

#[test]
fn test_missing_record_is_registered() {
    let (svc, saves, _dir) = svc(vec![]);

    svc.install_system_tools_server("p1").expect("install");

    let servers = servers(&svc);
    assert_eq!(servers.len(), 1);
    let installed = &servers[0];
    assert_eq!(installed.id, SYSTEM_TOOLS_SERVER_ID);
    assert_eq!(installed.name, SYSTEM_TOOLS_SERVER_NAME);
    assert!(installed.tools.iter().any(|t| t.name == "grep"));
    assert_eq!(saves.load(Ordering::SeqCst), 1);
}

#[test]
fn test_legacy_fetch_server_is_removed() {
    // Fetching moved from an MCP server to the in-process FetchAgent, so a
    // previously installed fetch server must be dropped from the project.
    let stale = server(FETCH_SERVER_ID, "fetch", &["fetch"]);
    let names = system_tool_names();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let system = server(SYSTEM_TOOLS_SERVER_ID, SYSTEM_TOOLS_SERVER_NAME, &names);
    let (svc, saves, _dir) = svc(vec![stale, system]);

    svc.remove_fetch_server("p1").expect("remove");

    let ids: Vec<String> = servers(&svc).into_iter().map(|s| s.id).collect();
    assert_eq!(ids, vec![SYSTEM_TOOLS_SERVER_ID.to_string()]);
    assert_eq!(saves.load(Ordering::SeqCst), 1);
}

#[test]
fn test_remove_fetch_server_noop_when_absent() {
    let (svc, saves, _dir) = svc(vec![]);

    svc.remove_fetch_server("p1").expect("remove");

    assert!(servers(&svc).is_empty());
    assert_eq!(saves.load(Ordering::SeqCst), 0);
}

#[test]
fn test_sweep_legacy_fetch_server_removes_and_persists() {
    // `remove_fetch_server` only runs for the actively loaded project, so wild
    // projects that are merely listed keep their stale `__fetch__` entry. The
    // sweep on `get_project` cleans any project on access.
    let stale = server(FETCH_SERVER_ID, "fetch", &["fetch"]);
    let other = server("server-a", "Server A", &["do_thing"]);
    let (svc, saves, _dir) = svc(vec![stale, other]);

    svc.sweep_legacy_fetch_server("p1");

    let ids: Vec<String> = servers(&svc).into_iter().map(|s| s.id).collect();
    assert_eq!(ids, vec!["server-a".to_string()]);
    assert_eq!(saves.load(Ordering::SeqCst), 1);

    // Idempotent: a second sweep neither removes nor saves again.
    svc.sweep_legacy_fetch_server("p1");
    let ids: Vec<String> = servers(&svc).into_iter().map(|s| s.id).collect();
    assert_eq!(ids, vec!["server-a".to_string()]);
    assert_eq!(saves.load(Ordering::SeqCst), 1);
}

#[test]
fn test_sweep_legacy_fetch_server_noop_when_absent() {
    let (svc, saves, _dir) = svc(vec![]);

    svc.sweep_legacy_fetch_server("p1");

    assert!(servers(&svc).is_empty());
    assert_eq!(saves.load(Ordering::SeqCst), 0);
}
