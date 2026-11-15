//! Port of `tests/unit/test_project_io_e2e.py`.
//!
//! End-to-end tests for project export/import.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde_json::Value;
use serial_test::serial;

use smartloop::api::services::project_io::{self, export_project, import_project, ProjectIoError};
use smartloop::config::app_settings::AppSettings;
use smartloop::workspace::workspace::{self as ws_mod, MCPAuthType, MCPServerType};
use smartloop::workspace::{
    Document, DocumentStatus, MCPOAuthCredentials, MCPServer, MCPServerConfig, Project,
    RemoteMCPServerConfig, Skill, Workspace,
};

/// The `workspace_env` fixture: isolated workspace directories and a seeded
/// `config.json`.
struct WorkspaceEnv {
    _tmp: tempfile::TempDir,
    home_dir: String,
    #[allow(dead_code)]
    work_dir: String,
    workspace: Workspace,
    project: Project,
    session_id: String,
    config_file: String,
}

fn workspace_env() -> WorkspaceEnv {
    let tmp = tempfile::tempdir().unwrap();
    let tmp_path = tmp.path();
    let home_dir = tmp_path.join("dot_smartloop");
    let work_dir = tmp_path.join("usercode"); // the user's own folder (no framework state)
    fs::create_dir(&home_dir).unwrap();
    fs::create_dir(&work_dir).unwrap();

    let workspace_id = uuid::Uuid::new_v4().to_string();
    let workspace_dir = home_dir.join(&workspace_id);
    fs::create_dir(&workspace_dir).unwrap();

    // Create a project with skills, documents, and MCP servers
    let project_id = uuid::Uuid::new_v4().to_string();
    let project_dir = workspace_dir.join(&project_id);
    fs::create_dir(&project_dir).unwrap();

    // Put a sample document file in the project dir
    fs::create_dir(project_dir.join("docs")).unwrap();
    fs::write(project_dir.join("docs").join("readme.txt"), "hello world").unwrap();

    // Skills live under project_dir, keyed by project id — not by name and
    // not under any client-supplied path.
    let project_name = "my-test-project";
    let skill_dir = project_dir.join("skills").join("greet");
    fs::create_dir_all(&skill_dir).unwrap();
    fs::write(skill_dir.join("SKILL.md"), "---\nname: greet\n---\nSay hello").unwrap();

    // Create conversation history
    let conv_dir = home_dir.join("conversations");
    fs::create_dir(&conv_dir).unwrap();
    let session_id = uuid::Uuid::new_v4().to_string();
    fs::write(
        conv_dir.join(format!("{session_id}.jsonl")),
        "{\"role\":\"user\",\"content\":\"Hello\"}\n{\"role\":\"assistant\",\"content\":\"Hi there!\"}\n",
    )
    .unwrap();

    let project = Project {
        id: project_id.clone(),
        project_dir: project_dir.to_string_lossy().into_owned(),
        current: true,
        name: Some(project_name.to_string()),
        description: Some("A test project".to_string()),
        documents: vec![Document {
            id: "doc1".into(),
            path: "/some/path.pdf".into(),
            status: DocumentStatus::Indexed,
            summary: Some("A doc".into()),
            ..Default::default()
        }],
        skills: vec![Skill {
            id: "sk1".into(),
            name: "greet".into(),
            description: Some("Greet skill".into()),
            content: "Say hello".into(),
            system: false,
        }],
        mcp_servers: vec![
            MCPServer {
                id: "mcp1".into(),
                name: "local-server".into(),
                config: Some(MCPServerConfig {
                    command: "npx".into(),
                    args: vec!["@test/server".into()],
                    ..Default::default()
                }),
                ..Default::default()
            },
            MCPServer {
                id: "mcp2".into(),
                name: "remote-server".into(),
                server_type: MCPServerType::Remote,
                remote_config: Some(RemoteMCPServerConfig {
                    server_url: "https://mcp.example.com".into(),
                    auth_type: MCPAuthType::Oauth,
                    token: Some("secret-token".into()),
                    ..Default::default()
                }),
                oauth_credentials: Some(MCPOAuthCredentials {
                    client_id: "cid".into(),
                    client_secret: Some("csecret".into()),
                    access_token: Some("at".into()),
                    refresh_token: Some("rt".into()),
                    ..Default::default()
                }),
                ..Default::default()
            },
        ],
        sessions: vec![session_id.clone()],
        ..Default::default()
    };

    let mut workspace = Workspace::new(workspace_id);
    workspace.workspace_dir = workspace_dir.to_string_lossy().into_owned();
    workspace.models_dir = workspace_dir.join("models").to_string_lossy().into_owned();
    workspace.projects = vec![project.clone()];

    let config_file = home_dir.join("config.json");
    let mut ws_data = serde_json::to_value(&workspace).unwrap();
    ws_data.as_object_mut().unwrap().remove("current_project");
    let data = serde_json::json!({"id": "", "workspace": ws_data});
    fs::write(&config_file, serde_json::to_string_pretty(&data).unwrap()).unwrap();

    WorkspaceEnv {
        home_dir: home_dir.to_string_lossy().into_owned(),
        work_dir: work_dir.to_string_lossy().into_owned(),
        workspace,
        project,
        session_id,
        config_file: config_file.to_string_lossy().into_owned(),
        _tmp: tmp,
    }
}

/// The stack of patches pointing the framework at the temp home; undone on drop.
///
/// Need to patch the settings in both project_io and workspace modules so that
/// `Workspace::load()` re-resolves paths against the temp dir.
/// Restores every patch on drop, `SLP_HOME` included: the workspace reads
/// `SLP_HOME` before the settings' home (as Python's `_home_dir` does), so
/// the env var has to name the fixture home too, not the process-wide
/// isolated one.
struct Patches {
    saved_slp_home: Option<std::ffi::OsString>,
}

fn patch_settings(env: &WorkspaceEnv) -> Patches {
    let saved_slp_home = std::env::var_os("SLP_HOME");
    // SAFETY: every test in this file is #[serial].
    unsafe { std::env::set_var("SLP_HOME", &env.home_dir) };
    let mut settings = AppSettings::new();
    settings.slp_home = Some(env.home_dir.clone());
    project_io::set_settings(Some(settings));
    ws_mod::set_settings_home_dir(Some(env.home_dir.clone()));
    Workspace::set_config_file(Some(env.config_file.clone()));
    Patches { saved_slp_home }
}

impl Drop for Patches {
    fn drop(&mut self) {
        project_io::set_settings(None);
        ws_mod::set_settings_home_dir(None);
        Workspace::set_config_file(None);
        Workspace::reset();
        // SAFETY: every test in this file is #[serial].
        unsafe {
            match self.saved_slp_home.take() {
                Some(home) => std::env::set_var("SLP_HOME", home),
                None => std::env::remove_var("SLP_HOME"),
            }
        }
    }
}

/// Member names of a zip archive, in archive order.
fn zip_names(path: &Path) -> Vec<String> {
    let mut archive = zip::ZipArchive::new(fs::File::open(path).unwrap()).unwrap();
    (0..archive.len()).map(|i| archive.by_index(i).unwrap().name().to_string()).collect()
}

/// Read one member of a zip archive as text.
fn zip_read(path: &Path, name: &str) -> String {
    let mut archive = zip::ZipArchive::new(fs::File::open(path).unwrap()).unwrap();
    let mut member = archive.by_name(name).unwrap();
    let mut out = String::new();
    member.read_to_string(&mut out).unwrap();
    out
}

fn read_config(env: &WorkspaceEnv) -> Value {
    serde_json::from_str(&fs::read_to_string(&env.config_file).unwrap()).unwrap()
}

// ── TestExportProject ───────────────────────────────────────────────────

#[test]
#[serial]
fn test_export_creates_valid_zip() {
    // Export produces a zip with manifest.json, workspace/, and conversations/.
    let env = workspace_env();
    let project = &env.project;
    let _patches = patch_settings(&env);

    let zip_path = export_project(&project.id).unwrap();
    assert!(zip_path.exists());

    let names = zip_names(&zip_path);

    // manifest.json present
    assert!(names.contains(&"manifest.json".to_string()));

    // workspace files present
    assert!(names.iter().any(|n| n.starts_with("workspace/")));
    assert!(names.contains(&"workspace/docs/readme.txt".to_string()));

    // Skills ride inside the state tree, not a separate one.
    assert!(names.contains(&"workspace/skills/greet/SKILL.md".to_string()));
    assert!(!names.iter().any(|n| n.starts_with("project/")));

    // conversation history present
    let conv_name = format!("conversations/{}.jsonl", env.session_id);
    assert!(names.contains(&conv_name));
    let conv_content = zip_read(&zip_path, &conv_name);
    assert!(conv_content.contains("\"Hello\""));
    assert!(conv_content.contains("\"Hi there!\""));

    // Check manifest content
    let manifest: Value = serde_json::from_str(&zip_read(&zip_path, "manifest.json")).unwrap();
    assert_eq!(manifest["version"], "1.0");
    assert!(manifest.get("exported_at").is_some());
    assert_eq!(manifest["project"]["id"], project.id.as_str());
    assert_eq!(manifest["project"]["name"], "my-test-project");
    // Skills are exported as files, not in the manifest
    assert!(manifest["project"].get("skills").is_none());
    assert_eq!(manifest["project"]["documents"].as_array().unwrap().len(), 1);
    assert!(manifest["project"]["sessions"]
        .as_array()
        .unwrap()
        .contains(&Value::String(env.session_id.clone())));

    // OAuth credentials stripped
    let mcp_servers = manifest["project"]["mcp_servers"].as_array().unwrap();
    let remote_srv = mcp_servers.iter().find(|s| s["name"] == "remote-server").unwrap();
    assert!(remote_srv.get("oauth_credentials").map(Value::is_null).unwrap_or(true));
    assert!(remote_srv["remote_config"].get("token").map(Value::is_null).unwrap_or(true));

    fs::remove_file(&zip_path).unwrap();
}

#[test]
#[serial]
fn test_export_nonexistent_project_raises() {
    // Exporting a non-existent project raises FileNotFoundError.
    let env = workspace_env();
    let _patches = patch_settings(&env);
    let err = export_project("nonexistent-id").unwrap_err();
    assert!(matches!(err, ProjectIoError::FileNotFoundError(_)), "{err:?}");
}

// ── TestImportProject ───────────────────────────────────────────────────

#[test]
#[serial]
fn test_import_creates_project() {
    // Import restores project metadata, files, and conversations.
    let env = workspace_env();
    let _patches = patch_settings(&env);

    let zip_path = export_project(&env.project.id).unwrap();

    let result = import_project(&zip_path, None, None).unwrap();

    // New ID generated
    assert_ne!(result["id"], env.project.id.as_str());
    assert_eq!(result["name"], "my-test-project");
    // cwd is machine-local, so an imported project starts without one.
    assert_eq!(result["cwd"], "");
    assert_eq!(result["description"], "A test project");
    assert_eq!(result["current"], true);

    // Documents preserve original status (vector DB is copied with project)
    let documents = result["documents"].as_array().unwrap();
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0]["status"], "indexed");

    // Skills preserved
    let skills = result["skills"].as_array().unwrap();
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0]["name"], "greet");

    // Workspace files copied
    let new_id = result["id"].as_str().unwrap();
    let new_project_dir = Path::new(&env.workspace.workspace_dir).join(new_id);
    assert!(new_project_dir.join("docs").join("readme.txt").is_file());

    // Skills landed in the new project's state dir.
    assert!(new_project_dir.join("skills").join("greet").join("SKILL.md").is_file());

    // Conversation history restored
    let conv_file = Path::new(&env.home_dir).join("conversations").join(format!("{}.jsonl", env.session_id));
    assert!(conv_file.is_file());

    // Config updated
    let config = read_config(&env);
    let ws = &config["workspace"];
    assert!(ws["projects"].as_array().unwrap().iter().any(|p| p["id"] == new_id));

    // The new project is recorded in the workspace DB.
    let db = smartloop::platform::workspace_db_for(&env.workspace).unwrap();
    assert!(db.projects().ids(&env.workspace.id).unwrap().iter().any(|id| id == new_id));

    fs::remove_file(&zip_path).unwrap();
}

#[test]
#[serial]
fn test_import_with_rename() {
    // Import with a name override renames the project.
    let env = workspace_env();
    let _patches = patch_settings(&env);

    let zip_path = export_project(&env.project.id).unwrap();

    let result = import_project(&zip_path, Some("renamed-project"), None).unwrap();

    assert_eq!(result["name"], "renamed-project");

    // Renaming changes the label only — state stays keyed by project id.
    let new_project_dir = Path::new(&env.workspace.workspace_dir).join(result["id"].as_str().unwrap());
    assert!(new_project_dir.join("skills").join("greet").join("SKILL.md").is_file());
    assert_eq!(result["skills"][0]["name"], "greet");

    fs::remove_file(&zip_path).unwrap();
}

#[test]
#[serial]
fn test_import_corrupt_zip_raises() {
    // Importing a corrupt zip raises ValueError.
    let env = workspace_env();
    let _patches = patch_settings(&env);

    let mut bad = tempfile::Builder::new().suffix(".zip").tempfile().unwrap();
    bad.write_all(b"not a zip file").unwrap();
    bad.flush().unwrap();

    match import_project(bad.path(), None, None) {
        Err(ProjectIoError::ValueError(msg)) => assert!(msg.contains("not a valid zip"), "{msg}"),
        other => panic!("expected ValueError, got {other:?}"),
    }
}

#[test]
#[serial]
fn test_import_missing_manifest_raises() {
    // Importing a zip without manifest.json raises ValueError.
    let env = workspace_env();
    let _patches = patch_settings(&env);

    let bad = tempfile::Builder::new().suffix(".zip").tempfile().unwrap();
    {
        let mut zf = zip::ZipWriter::new(fs::File::create(bad.path()).unwrap());
        zf.start_file("random.txt", zip::write::SimpleFileOptions::default()).unwrap();
        zf.write_all(b"hello").unwrap();
        zf.finish().unwrap();
    }

    match import_project(bad.path(), None, None) {
        Err(ProjectIoError::ValueError(msg)) => assert!(msg.contains("missing manifest"), "{msg}"),
        other => panic!("expected ValueError, got {other:?}"),
    }
}

// ── TestImportingIntoTheCallersWorkspace ────────────────────────────────
//
// Where an import lands when the caller already holds a workspace.
//
// `import_project` used to build its own `Workspace.load()`, save, and
// return.  The caller's instance — the one the API server serves every
// request from — never learned the project existed.  Every later read
// resolved it to whichever project was selected, and the first save from
// that instance wrote a project list the imported one was missing from,
// taking its `config.json` entry with it while the directory, the documents
// and the index stayed on disk.

#[test]
#[serial]
fn test_the_project_lands_in_the_instance_the_caller_passed() {
    let env = workspace_env();
    let _patches = patch_settings(&env);

    let mut live = Workspace::load().unwrap();
    assert_eq!(live.projects.len(), 1);

    let zip_path = export_project(&env.project.id).unwrap();
    let result = import_project(&zip_path, None, Some(&mut live)).unwrap();

    let new_id = result["id"].as_str().unwrap();
    assert!(
        live.projects.iter().any(|p| p.id == new_id),
        "the imported project is missing from the instance it was imported into"
    );
    assert_eq!(live.projects.len(), 2);
    let _ = fs::remove_file(&zip_path);
}

#[test]
#[serial]
fn test_a_later_save_keeps_the_imported_project() {
    // The regression: the caller saves, and the import survives it.
    //
    // This is the failure as it actually happened — nothing here touches the
    // import again.  The caller simply persists something unrelated (a
    // selection change) afterwards, which is enough to drop a project the
    // instance was never told about.
    let env = workspace_env();
    let _patches = patch_settings(&env);

    let mut live = Workspace::load().unwrap();
    let zip_path = export_project(&env.project.id).unwrap();
    let imported = import_project(&zip_path, None, Some(&mut live)).unwrap();

    // Something unrelated moves the selection and persists it.
    live.projects[0].current = true;
    live.save().unwrap();

    let on_disk = read_config(&env);
    let ids: Vec<&str> = on_disk["workspace"]["projects"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["id"].as_str().unwrap())
        .collect();
    assert!(
        ids.contains(&imported["id"].as_str().unwrap()),
        "a save from the caller's workspace erased the imported project"
    );
    assert_eq!(ids.len(), 2);
    let _ = fs::remove_file(&zip_path);
}

#[test]
#[serial]
fn test_no_workspace_still_loads_one() {
    // The CLI holds no instance, so the default is unchanged.
    let env = workspace_env();
    let _patches = patch_settings(&env);

    let zip_path = export_project(&env.project.id).unwrap();
    let result = import_project(&zip_path, None, None).unwrap();
    let on_disk = read_config(&env);
    let ids: Vec<&Value> = on_disk["workspace"]["projects"].as_array().unwrap().iter().map(|p| &p["id"]).collect();
    assert!(ids.contains(&&result["id"]));
    let _ = fs::remove_file(&zip_path);
}

// ── TestExportImportRoundTrip ───────────────────────────────────────────

#[test]
#[serial]
fn test_full_cycle() {
    // Export → import produces a new project with matching data but new ID.
    let env = workspace_env();
    let original = env.project.clone();
    let _patches = patch_settings(&env);

    let zip_path = export_project(&original.id).unwrap();
    let result = import_project(&zip_path, None, None).unwrap();

    assert_ne!(result["id"], original.id.as_str());
    assert_eq!(result["name"], original.name.clone().unwrap().as_str());
    // cwd does not survive a round trip — it is a machine-local working
    // folder, not project state.
    assert_eq!(result["cwd"], "");
    assert_eq!(result["description"], original.description.clone().unwrap().as_str());
    assert_eq!(result["skills"].as_array().unwrap().len(), original.skills.len());
    assert_eq!(result["documents"].as_array().unwrap().len(), original.documents.len());

    // Document statuses preserved (vector DB copied with project)
    for doc in result["documents"].as_array().unwrap() {
        assert_eq!(doc["status"], "indexed");
    }

    fs::remove_file(&zip_path).unwrap();
}

// ── TestLegacyArchiveImport ─────────────────────────────────────────────
//
// Archives from builds that stored skills in the user-space folder.
//
// Those zips carry a separate `project/` tree alongside `workspace/`.  Export
// no longer writes one, but import must still fold it into the new state
// directory or every previously-exported project loses its skills.

#[test]
#[serial]
fn test_legacy_project_tree_is_folded_into_state_dir() {
    let env = workspace_env();
    let _patches = patch_settings(&env);

    let zip_path = export_project(&env.project.id).unwrap();

    // Rewrite the archive into the old shape: skills under project/.
    let legacy: PathBuf = zip_path.with_file_name("legacy.zip");
    {
        let mut src = zip::ZipArchive::new(fs::File::open(&zip_path).unwrap()).unwrap();
        let mut dst = zip::ZipWriter::new(fs::File::create(&legacy).unwrap());
        for i in 0..src.len() {
            let mut member = src.by_index(i).unwrap();
            let item = member.name().to_string();
            let mut data = Vec::new();
            member.read_to_end(&mut data).unwrap();
            let target = if item == "workspace/skills/greet/SKILL.md" {
                "project/skills/greet/SKILL.md".to_string()
            } else {
                item
            };
            dst.start_file(target, zip::write::SimpleFileOptions::default()).unwrap();
            dst.write_all(&data).unwrap();
        }
        dst.finish().unwrap();
    }

    let result = import_project(&legacy, None, None).unwrap();

    let skills = result["skills"].as_array().unwrap();
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0]["name"], "greet");
    let new_project_dir = Path::new(&env.workspace.workspace_dir).join(result["id"].as_str().unwrap());
    assert!(new_project_dir.join("skills").join("greet").join("SKILL.md").is_file());

    fs::remove_file(&zip_path).unwrap();
    fs::remove_file(&legacy).unwrap();
}
