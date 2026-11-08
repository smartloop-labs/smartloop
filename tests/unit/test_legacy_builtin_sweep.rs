//! Self-healing of what smartloop used to install on a project itself.
//!
//! Smartloop no longer ships built-in MCP servers: fetching is the in-process
//! FetchAgent, and running commands, writing files and saving skills belong
//! to the harness.  Projects created before still carry the persisted
//! `__fetch__`, `__system_tools__` and `__skills__` entries and the default
//! skills written for them; `sweep_legacy_builtins` (run on every
//! `get_project`) drops them.
//!
//! The service holds a real workspace with the project, and the `save_model`
//! seam counts `self._model.save()` calls.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;

use smartloop::agents::planner::FETCH_SERVER_ID;
use smartloop::api::config::api_settings;
use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::workspace::{
    MCPServer, MCPServerConfig, MCPServerType, MCPToolDefinition, Project, Skill, Workspace,
};

use crate::shared::model_service_a::{attach_workspace, bare_service};

const SYSTEM_TOOLS_ID: &str = "__system_tools__";
const SKILLS_ID: &str = "__skills__";

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

fn server_ids(svc: &ModelService) -> Vec<String> {
    let ws = svc.workspace_handle().expect("workspace");
    let guard = ws.read();
    guard.project("p1").expect("p1").mcp_servers.iter().map(|s| s.id.clone()).collect()
}

fn skills_dir(dir: &tempfile::TempDir) -> std::path::PathBuf {
    dir.path().join("skills")
}

fn write_skill(dir: &tempfile::TempDir, name: &str, content: &str) {
    let skill = Skill {
        id: name.to_string(),
        name: name.to_string(),
        description: Some(format!("{name} skill")),
        content: content.to_string(),
        system: false,
    };
    ModelService::write_skill_md_in(&skills_dir(dir), &skill).expect("write skill");
}

#[test]
fn test_sweep_drops_every_legacy_builtin_server_and_persists_once() {
    let (svc, saves, _dir) = svc(vec![
        server(FETCH_SERVER_ID, "fetch", &["fetch"]),
        server(SYSTEM_TOOLS_ID, "smartloop", &["bash", "file-read", "file-write", "grep", "explore"]),
        server(SKILLS_ID, "Skills", &["save-skill"]),
        server("server-a", "Server A", &["do_thing"]),
    ]);

    svc.sweep_legacy_builtins("p1");

    assert_eq!(server_ids(&svc), vec!["server-a".to_string()]);
    assert_eq!(saves.load(Ordering::SeqCst), 1);

    // Idempotent: a second sweep neither removes nor saves again.
    svc.sweep_legacy_builtins("p1");
    assert_eq!(server_ids(&svc), vec!["server-a".to_string()]);
    assert_eq!(saves.load(Ordering::SeqCst), 1);
}

#[test]
fn test_sweep_is_a_noop_without_legacy_servers() {
    let (svc, saves, _dir) = svc(vec![server("server-a", "Server A", &["do_thing"])]);

    svc.sweep_legacy_builtins("p1");

    assert_eq!(server_ids(&svc), vec!["server-a".to_string()]);
    assert_eq!(saves.load(Ordering::SeqCst), 0);
}

#[test]
fn test_get_project_sweeps_on_access() {
    let (svc, _saves, _dir) = svc(vec![
        server(SYSTEM_TOOLS_ID, "smartloop", &["bash"]),
        server("server-a", "Server A", &["do_thing"]),
    ]);

    let project = svc.get_project("p1").expect("project");

    let ids: Vec<&str> = project.mcp_servers.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, vec!["server-a"]);
}

#[test]
fn test_sweep_removes_the_default_skills_smartloop_wrote() {
    let (svc, _saves, dir) = svc(vec![]);
    write_skill(&dir, "file-read", "read the file at the given path and return its content");
    write_skill(&dir, "file-write", "write content to a file at the given path");
    write_skill(&dir, "bash", "execute the shell command on the host machine");
    write_skill(&dir, "fetch-url", "fetch a url");
    write_skill(&dir, "__system_mcp__", "system tools");
    std::fs::write(skills_dir(&dir).join("__system_mcp__.md"), "system tools").expect("write");
    write_skill(&dir, "good-morning", "say good morning");

    svc.sweep_legacy_builtins("p1");

    let left: Vec<String> = std::fs::read_dir(skills_dir(&dir))
        .expect("skills dir")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(left, vec!["good-morning".to_string()]);
}

#[test]
fn test_sweep_keeps_a_user_skill_that_reuses_a_default_name() {
    let (svc, _saves, dir) = svc(vec![]);
    // The harness (or the user) wrote its own `bash` skill: not ours to delete.
    write_skill(&dir, "bash", "Run the project's test suite with cargo test.");

    svc.sweep_legacy_builtins("p1");

    assert!(Path::new(&skills_dir(&dir).join("bash").join("SKILL.md")).is_file());
}

#[test]
fn test_setup_project_tools_installs_nothing() {
    let (svc, _saves, dir) = svc(vec![server(SKILLS_ID, "Skills", &["save-skill"])]);

    svc.setup_project_tools("p1").expect("setup");

    assert!(server_ids(&svc).is_empty());
    assert!(!skills_dir(&dir).join("file-read").exists());
    assert!(!skills_dir(&dir).join("bash").exists());
}
