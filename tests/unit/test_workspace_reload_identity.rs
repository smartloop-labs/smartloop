//! Port of `tests/unit/test_workspace_reload_identity.py`.
//!
//! `reload()` must refresh the records callers hold, not replace them.
//!
//! A chat request opens with `Workspace::reload()`.  Background passes — MCP
//! provisioning, document indexing — act on a `Project` or a record inside one
//! across that reload and write to it afterwards.  While reload rebound
//! `projects` to a freshly-parsed list, those writes landed on a record the
//! workspace no longer contained and the next `save()` serialized the list that
//! had replaced them.
//!
//! Python asserts object identity (`is`); here a record keeps its identity when
//! it is updated where it lies — the same address in the workspace before and
//! after — rather than being swapped for a freshly-parsed copy.

use std::fs;
use std::path::Path;

use serde_json::{Map, Value};
use serial_test::serial;

use smartloop::workspace::workspace::{
    Document, MCPServer, MCPServerStatus, MCPServerType, Project, RemoteMCPServerConfig, Workspace,
    project_state_path, set_settings_home_dir,
};

use crate::shared::workspace_support::{EnvGuard, ProcessScopeReset};

fn join(base: &str, part: &str) -> String {
    Path::new(base).join(part).to_string_lossy().into_owned()
}

fn workspace(home: &str, servers: Vec<MCPServer>, documents: Vec<Document>) -> Workspace {
    Workspace::new("w1")
        .with_workspace_dir(join(home, "w1"))
        .with_models_dir(join(&join(home, "w1"), "models"))
        .with_projects(vec![Project {
            id: "p1".to_string(),
            name: Some("proj".to_string()),
            current: true,
            project_dir: join(&join(home, "w1"), "p1"),
            mcp_servers: servers,
            documents,
            ..Default::default()
        }])
}

fn server(id: &str, name: &str, account: Option<&str>) -> MCPServer {
    MCPServer {
        id: id.to_string(),
        name: name.to_string(),
        server_type: MCPServerType::Remote,
        remote_config: Some(RemoteMCPServerConfig {
            server_url: "https://mcp.notion.com/mcp".to_string(),
            ..Default::default()
        }),
        account: account.map(str::to_string),
        ..Default::default()
    }
}

fn default_server() -> MCPServer {
    server("s1", "notion", None)
}

/// The `home` fixture: a temp framework home with the config path pointed at
/// it (Python patched `Workspace.config_file` and `workspace._settings`).
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
            EnvGuard::remove("SLP_HOME"),
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

/// Stand in for another process writing this project's own file.
fn rewrite_on_disk(home: &str, mutate: impl FnOnce(&mut Map<String, Value>)) {
    let path = project_state_path(&join(&join(home, "w1"), "p1"));
    let mut data: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    mutate(data["project"].as_object_mut().unwrap());
    fs::write(&path, serde_json::to_string(&data).unwrap()).unwrap();
}

mod the_objects_survive_a_reload {
    use super::*;

    #[test]
    #[serial]
    fn test_the_project_is_the_same_object() {
        let home = Home::new();
        let mut ws = workspace(&home.root, vec![default_server()], vec![]);
        ws.save().unwrap();
        let held: *const Project = &ws.projects[0];

        ws.reload();

        assert!(std::ptr::eq(&ws.projects[0], held));
    }

    #[test]
    #[serial]
    fn test_the_server_is_the_same_object() {
        let home = Home::new();
        let mut ws = workspace(&home.root, vec![default_server()], vec![]);
        ws.save().unwrap();
        let held: *const MCPServer = &ws.projects[0].mcp_servers[0];

        ws.reload();

        assert!(std::ptr::eq(&ws.projects[0].mcp_servers[0], held));
    }

    #[test]
    #[serial]
    fn test_a_document_is_the_same_object() {
        let home = Home::new();
        let document = Document { id: "d1".to_string(), path: "/tmp/a.pdf".to_string(), ..Default::default() };
        let mut ws = workspace(&home.root, vec![], vec![document]);
        ws.save().unwrap();
        let held: *const Document = &ws.projects[0].documents[0];

        ws.reload();

        assert!(std::ptr::eq(&ws.projects[0].documents[0], held));
    }

    /// The bug in its original shape: a pass mutates what it is holding.
    #[test]
    #[serial]
    fn test_a_write_after_a_reload_reaches_the_disk() {
        let home = Home::new();
        let mut ws = workspace(&home.root, vec![default_server()], vec![]);
        ws.save().unwrap();

        ws.reload(); // a chat request, mid-pass
        let held = &mut ws.projects[0].mcp_servers[0];
        held.account = Some("chris@smartloop.ai".to_string());
        held.status = MCPServerStatus::Ready;
        ws.save().unwrap();

        let reloaded = Workspace::load().unwrap();
        assert_eq!(reloaded.projects[0].mcp_servers[0].account.as_deref(), Some("chris@smartloop.ai"));
    }
}

/// In-place must not mean stale: reload exists to pick up other writers.
mod it_still_reads_from_disk {
    use super::*;

    #[test]
    #[serial]
    fn test_a_field_another_process_wrote_is_visible_on_the_held_object() {
        let home = Home::new();
        let mut ws = workspace(&home.root, vec![server("s1", "notion", None)], vec![]);
        ws.save().unwrap();
        let held: *const MCPServer = &ws.projects[0].mcp_servers[0];

        rewrite_on_disk(&home.root, |project| {
            project["mcp_servers"][0]["account"] = Value::String("someone@else.com".to_string());
        });
        ws.reload();

        assert!(std::ptr::eq(&ws.projects[0].mcp_servers[0], held));
        assert_eq!(ws.projects[0].mcp_servers[0].account.as_deref(), Some("someone@else.com"));
    }

    #[test]
    #[serial]
    fn test_a_server_added_elsewhere_appears() {
        let home = Home::new();
        let mut ws = workspace(&home.root, vec![default_server()], vec![]);
        ws.save().unwrap();

        rewrite_on_disk(&home.root, |project| {
            let mut second = project["mcp_servers"][0].clone();
            second["id"] = Value::String("s2".to_string());
            second["name"] = Value::String("linear".to_string());
            project["mcp_servers"].as_array_mut().unwrap().push(second);
        });
        ws.reload();

        assert_eq!(ws.projects[0].mcp_servers.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), vec!["s1", "s2"]);
    }

    #[test]
    #[serial]
    fn test_a_server_deleted_elsewhere_is_dropped() {
        let home = Home::new();
        let mut ws = workspace(&home.root, vec![default_server(), server("s2", "linear", None)], vec![]);
        ws.save().unwrap();

        rewrite_on_disk(&home.root, |project| {
            project["mcp_servers"].as_array_mut().unwrap().retain(|s| s["id"] != "s1");
        });
        ws.reload();

        assert_eq!(ws.projects[0].mcp_servers.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), vec!["s2"]);
    }

    #[test]
    #[serial]
    fn test_disk_order_wins() {
        let home = Home::new();
        let mut ws = workspace(&home.root, vec![default_server(), server("s2", "linear", None)], vec![]);
        ws.save().unwrap();

        rewrite_on_disk(&home.root, |project| {
            project["mcp_servers"].as_array_mut().unwrap().reverse();
        });
        ws.reload();

        assert_eq!(ws.projects[0].mcp_servers.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), vec!["s2", "s1"]);
    }
}

/// The same promise on the narrower, mtime-gated path.
mod refresh_project_keeps_them_too {
    use super::*;

    #[test]
    #[serial]
    fn test_the_server_survives_a_refresh() {
        let home = Home::new();
        let mut ws = workspace(&home.root, vec![server("s1", "notion", None)], vec![]);
        ws.save().unwrap();
        let held: *const MCPServer = &ws.projects[0].mcp_servers[0];

        rewrite_on_disk(&home.root, |project| {
            project["mcp_servers"][0]["account"] = Value::String("chris@smartloop.ai".to_string());
        });

        assert!(ws.refresh_project("p1"));
        assert!(std::ptr::eq(&ws.projects[0].mcp_servers[0], held));
        assert_eq!(ws.projects[0].mcp_servers[0].account.as_deref(), Some("chris@smartloop.ai"));
    }
}
