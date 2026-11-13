//! A freshly registered MCP server must be usable by the next prompt as soon
//! as its tools are known.
//!
//! Provisioning discovers a server's tools and then enriches its metadata with
//! one or two LLM calls on the shared model host — seconds, and longer while a
//! chat is generating.  Every chat turn opens with `workspace.reload()` and
//! then offers the orchestrator only servers that have tools.  While the
//! discovered tools stayed on the provisioning pass's private copy until the
//! enrichment finished, a prompt sent right after registering reloaded a record
//! with no tools and left the new connection out of the turn.
//!
//! These tests stand in for that prompt: a reload in the middle of the
//! enrichment, then the same eligibility check the chat path makes.

use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::json;
use serial_test::serial;

use smartloop::api::config::api_settings;
use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::workspace::workspace::{
    MCPServer, MCPServerConfig, MCPServerStatus, MCPServerType, MCPToolDefinition, Project, Workspace,
    set_settings_home_dir,
};

use crate::shared::model_service_a::{attach_workspace, bare_service};
use crate::shared::workspace_support::{EnvGuard, ProcessScopeReset};

fn join(base: &str, part: &str) -> String {
    Path::new(base).join(part).to_string_lossy().into_owned()
}

/// A temp framework home with the config path pointed at it.
struct Home {
    root: String,
    _dir: tempfile::TempDir,
    _env: Vec<EnvGuard>,
    _reset: ProcessScopeReset,
}

impl Home {
    fn new() -> Self {
        let reset = ProcessScopeReset::new();
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_string_lossy().into_owned();
        let env = vec![
            EnvGuard::set("SLP_HOME", dir.path()),
            EnvGuard::remove("SLP_CONFIG_FILE"),
            EnvGuard::remove("SLP_LEGACY_CONFIG"),
        ];
        Workspace::set_config_file(Some(join(&root, "config.json")));
        set_settings_home_dir(Some(root.clone()));
        Self { root, _dir: dir, _env: env, _reset: reset }
    }
}

impl Drop for Home {
    fn drop(&mut self) {
        Workspace::set_config_file(None);
        set_settings_home_dir(None);
    }
}

fn time_tools() -> Vec<MCPToolDefinition> {
    ["get_current_time", "convert_time"]
        .iter()
        .map(|name| MCPToolDefinition {
            name: name.to_string(),
            description: Some("time".to_string()),
            input_schema: json!({"type": "object"}).as_object().cloned(),
            ..Default::default()
        })
        .collect()
}

/// A service over a saved workspace holding project `p1` with one server
/// just registered (`pending`, no tools yet).
fn service_with_pending_server(home: &Home) -> Arc<ModelService> {
    let workspace_dir = join(&home.root, "w1");
    let mut workspace = Workspace::new("w1")
        .with_workspace_dir(workspace_dir.clone())
        .with_models_dir(join(&workspace_dir, "models"))
        .with_projects(vec![Project {
            id: "p1".to_string(),
            name: Some("proj".to_string()),
            current: true,
            project_dir: join(&workspace_dir, "p1"),
            mcp_servers: vec![MCPServer {
                id: "s1".to_string(),
                name: "time".to_string(),
                server_type: MCPServerType::Local,
                config: Some(MCPServerConfig {
                    command: "uvx".to_string(),
                    args: vec!["mcp-server-time".to_string()],
                    ..Default::default()
                }),
                enabled: true,
                status: MCPServerStatus::Pending,
                ..Default::default()
            }],
            ..Default::default()
        }]);
    workspace.save().unwrap();
    // What a running process holds: the record as loaded from disk.
    workspace.reload();
    let svc = bare_service();
    attach_workspace(&svc, workspace);
    *svc.model_name.write() = Some(api_settings().default_model);
    svc
}

/// The servers a chat turn offers the orchestrator (`orchestrate`'s
/// `eligible_servers`), read after the reload every turn opens with.
fn servers_a_prompt_would_see(svc: &ModelService) -> Vec<(String, usize)> {
    let ws = svc.workspace_handle().expect("workspace");
    ws.write().reload();
    let guard = ws.read();
    guard
        .project("p1")
        .expect("p1")
        .mcp_servers
        .iter()
        .filter(|s| s.enabled && !s.tools.is_empty())
        .map(|s| (s.name.clone(), s.tools.len()))
        .collect()
}

#[test]
#[serial]
fn test_a_prompt_sent_while_the_new_server_is_being_enriched_sees_its_tools() {
    let home = Home::new();
    let svc = service_with_pending_server(&home);

    let seen_mid_enrichment: Arc<Mutex<Option<Vec<(String, usize)>>>> = Arc::default();
    let weak = Arc::downgrade(&svc);
    let record = seen_mid_enrichment.clone();
    svc.set_hooks(ModelServiceHooks {
        discover_mcp_tools: Some(Arc::new(|server: &mut MCPServer| {
            server.tools = time_tools();
            Ok(())
        })),
        enrich_mcp_server: Some(Arc::new(move |server: &mut MCPServer| {
            // The user prompts while the LLM is still describing the server.
            let svc = weak.upgrade().expect("service alive");
            *record.lock() = Some(servers_a_prompt_would_see(&svc));
            server.description = Some("Current time and timezone conversion".to_string());
        })),
        ..Default::default()
    });

    let result = svc.provision_mcp_server("p1", "s1").unwrap();
    assert_eq!(result["status"], "ready");

    assert_eq!(
        seen_mid_enrichment.lock().clone(),
        Some(vec![("time".to_string(), 2)]),
        "a prompt sent during enrichment must already be offered the new server's tools",
    );
}

#[test]
#[serial]
fn test_a_reload_during_enrichment_does_not_lose_the_finished_pass() {
    let home = Home::new();
    let svc = service_with_pending_server(&home);

    let weak = Arc::downgrade(&svc);
    svc.set_hooks(ModelServiceHooks {
        discover_mcp_tools: Some(Arc::new(|server: &mut MCPServer| {
            server.tools = time_tools();
            Ok(())
        })),
        enrich_mcp_server: Some(Arc::new(move |server: &mut MCPServer| {
            let svc = weak.upgrade().expect("service alive");
            let _ = servers_a_prompt_would_see(&svc);
            server.description = Some("Current time and timezone conversion".to_string());
        })),
        ..Default::default()
    });

    svc.provision_mcp_server("p1", "s1").unwrap();

    // In memory, after the next prompt's reload...
    assert_eq!(servers_a_prompt_would_see(&svc), vec![("time".to_string(), 2)]);
    // ...and on disk, for the next process.
    let on_disk = Workspace::load().expect("saved workspace");
    let server = &on_disk.project("p1").expect("p1").mcp_servers[0];
    assert_eq!(server.status, MCPServerStatus::Ready);
    assert_eq!(server.tools.len(), 2);
    assert_eq!(server.description.as_deref(), Some("Current time and timezone conversion"));
}
