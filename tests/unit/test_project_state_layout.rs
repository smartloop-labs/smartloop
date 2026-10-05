//! Port of `tests/unit/test_project_state_layout.py`.
//!
//! Skills and documents live under project_dir, never a client-supplied path.
//!
//! The old layout anchored them to `project.cwd` (defaulting to
//! `~/smartloop/<name>`), an absolute user-space path.  That path does not
//! follow the framework's home, so under the macOS App Sandbox — where `~` is
//! redirected to the app container — the app could read one workspace and
//! write another.  `project_dir` derives from `AppSettings.home_dir`, so it
//! always travels with the framework.

use std::fs;
use std::path::Path;

use serial_test::serial;

use smartloop::workspace::migrations::migrate_project_state;
use smartloop::workspace::workspace::{Project, Workspace};

use crate::shared::workspace_support::{EnvGuard, ProcessScopeReset};

fn s(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn join(base: &str, part: &str) -> String {
    Path::new(base).join(part).to_string_lossy().into_owned()
}

fn project(tmp_path: &Path, cwd: Option<&Path>, name: &str) -> Project {
    let state = tmp_path.join("state").join("p1");
    fs::create_dir_all(&state).unwrap();
    Project {
        id: "p1".to_string(),
        name: Some(name.to_string()),
        project_dir: s(&state),
        current: true,
        cwd: cwd.map(s),
        ..Default::default()
    }
}

fn write_skill(skills_dir: impl AsRef<Path>, name: &str, body: &str) {
    let d = skills_dir.as_ref().join(name);
    fs::create_dir_all(&d).unwrap();
    fs::write(d.join("SKILL.md"), format!("---\nname: {name}\ndescription: d-{name}\n---\n{body}")).unwrap();
}

fn skill_names(project: &Project) -> Vec<String> {
    project.get_skills().into_iter().map(|s| s.name).collect()
}

/// The Python `_WS` stand-in: a workspace holding just `projects`.
fn ws(projects: Vec<Project>) -> Workspace {
    Workspace::new("ws").with_projects(projects)
}

// --------------------------------------------------------------------------
// layout
// --------------------------------------------------------------------------

#[test]
fn test_state_dirs_hang_off_project_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let p = project(tmp.path(), Some(&tmp.path().join("usercode")), "proj");
    assert_eq!(p.skills_dir(), join(&p.project_dir, "skills"));
    assert_eq!(p.documents_dir(), join(&p.project_dir, "documents"));
}

/// A client-supplied cwd must not influence where state is stored.
#[test]
fn test_state_dirs_ignore_cwd_entirely() {
    let tmp = tempfile::tempdir().unwrap();
    let a = project(tmp.path(), Some(&tmp.path().join("one")), "proj");
    let b = project(tmp.path(), Some(&tmp.path().join("two")), "proj");
    assert_eq!(a.skills_dir(), b.skills_dir());
    assert_eq!(a.documents_dir(), b.documents_dir());
}

/// A project with no working folder is still fully usable.
#[test]
fn test_state_dirs_survive_a_missing_cwd() {
    let tmp = tempfile::tempdir().unwrap();
    let p = project(tmp.path(), None, "proj");
    assert_eq!(p.work_dir(), None);
    assert!(p.skills_dir().starts_with(&p.project_dir));
}

#[test]
#[serial]
fn test_work_dir_is_the_users_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let p = project(tmp.path(), Some(&tmp.path().join("usercode")), "proj");
    assert_eq!(p.work_dir(), Some(s(&tmp.path().join("usercode"))));
}

#[test]
#[serial]
fn test_work_dir_expands_tilde() {
    let tmp = tempfile::tempdir().unwrap();
    let _home = EnvGuard::set("HOME", tmp.path().join("home"));
    let p = Project {
        id: "p".to_string(),
        name: Some("n".to_string()),
        project_dir: s(tmp.path()),
        current: true,
        cwd: Some("~/repo".to_string()),
        ..Default::default()
    };
    assert_eq!(p.work_dir(), Some(s(&tmp.path().join("home").join("repo"))));
}

/// `hasattr(p, "visible_dir")` / `hasattr(p, "folder_name")`: the fields do
/// not exist on the struct (a compile-time fact), so the runtime check is that
/// the serialized record carries neither.
#[test]
fn test_visible_dir_concept_is_gone() {
    let tmp = tempfile::tempdir().unwrap();
    let p = project(tmp.path(), None, "proj");
    let dumped = serde_json::to_value(&p).unwrap();
    assert!(dumped.get("visible_dir").is_none());
    assert!(dumped.get("folder_name").is_none());
}

#[test]
fn test_get_skills_reads_from_project_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let p = project(tmp.path(), Some(&tmp.path().join("usercode")), "proj");
    write_skill(p.skills_dir(), "greet", "BODY");
    // A stale copy in the old user-space location must be ignored.
    fs::create_dir_all(tmp.path().join("usercode")).unwrap();
    write_skill(tmp.path().join("usercode").join("skills"), "stale", "BODY");

    assert_eq!(skill_names(&p), vec!["greet"]);
}

// --------------------------------------------------------------------------
// migration
// --------------------------------------------------------------------------

#[test]
#[serial]
fn test_migration_moves_legacy_skills_and_documents() {
    let tmp = tempfile::tempdir().unwrap();
    let legacy = tmp.path().join("usercode");
    let p = project(tmp.path(), Some(&legacy), "proj");
    write_skill(legacy.join("skills"), "greet", "BODY");
    fs::create_dir_all(legacy.join("documents")).unwrap();
    fs::write(legacy.join("documents").join("notes.txt"), "hello").unwrap();

    let moved = migrate_project_state(Some(&ws(vec![p.clone()])));

    let counts = &moved["p1"];
    assert_eq!(counts.len(), 2);
    assert_eq!(counts["skills"], 1);
    assert_eq!(counts["documents"], 1);
    assert_eq!(skill_names(&p), vec!["greet"]);
    assert!(Path::new(&join(&p.documents_dir(), "notes.txt")).is_file());
}

#[test]
#[serial]
fn test_migration_is_idempotent() {
    let tmp = tempfile::tempdir().unwrap();
    let legacy = tmp.path().join("usercode");
    let p = project(tmp.path(), Some(&legacy), "proj");
    write_skill(legacy.join("skills"), "greet", "BODY");

    let first = migrate_project_state(Some(&ws(vec![p.clone()])));
    let second = migrate_project_state(Some(&ws(vec![p.clone()])));

    assert_eq!(first["p1"]["skills"], 1);
    assert!(second.is_empty());
}

/// An already-migrated skill wins over the stale legacy copy.
#[test]
#[serial]
fn test_migration_never_clobbers_newer_state() {
    let tmp = tempfile::tempdir().unwrap();
    let legacy = tmp.path().join("usercode");
    let p = project(tmp.path(), Some(&legacy), "proj");
    write_skill(p.skills_dir(), "greet", "NEW");
    write_skill(legacy.join("skills"), "greet", "OLD");

    migrate_project_state(Some(&ws(vec![p.clone()])));

    assert!(p.get_skills()[0].content.contains("NEW"));
}

/// Recoverable by hand if the move goes wrong.
#[test]
#[serial]
fn test_migration_leaves_the_source_in_place() {
    let tmp = tempfile::tempdir().unwrap();
    let legacy = tmp.path().join("usercode");
    let p = project(tmp.path(), Some(&legacy), "proj");
    write_skill(legacy.join("skills"), "greet", "BODY");

    migrate_project_state(Some(&ws(vec![p.clone()])));

    assert!(legacy.join("skills").join("greet").join("SKILL.md").is_file());
}

/// Projects predating persisted cwd lived at ~/smartloop/<name>.
#[test]
#[serial]
fn test_migration_picks_up_legacy_name_based_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    let _home = EnvGuard::set("HOME", &home);
    let p = project(tmp.path(), None, "general_chat");
    write_skill(home.join("smartloop").join("general_chat").join("skills"), "greet", "BODY");

    let moved = migrate_project_state(Some(&ws(vec![p.clone()])));

    assert_eq!(moved["p1"]["skills"], 1);
    assert_eq!(skill_names(&p), vec!["greet"]);
}

/// The sandbox case: the old path exists but is denied. Must not fail.
#[cfg(unix)]
#[test]
#[serial]
fn test_migration_survives_an_unreadable_legacy_path() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::tempdir().unwrap();
    let legacy = tmp.path().join("denied");
    fs::create_dir(&legacy).unwrap();
    write_skill(legacy.join("skills"), "greet", "BODY");
    let p = project(tmp.path(), Some(&legacy), "proj");
    fs::set_permissions(&legacy, fs::Permissions::from_mode(0o000)).unwrap();

    let result = std::panic::catch_unwind(|| migrate_project_state(Some(&ws(vec![p.clone()]))));

    fs::set_permissions(&legacy, fs::Permissions::from_mode(0o755)).unwrap();
    assert!(result.expect("migration must not fail").is_empty());
}

#[test]
#[serial]
fn test_migration_handles_a_missing_legacy_path() {
    let tmp = tempfile::tempdir().unwrap();
    let p = project(tmp.path(), Some(&tmp.path().join("does-not-exist")), "proj");
    assert!(migrate_project_state(Some(&ws(vec![p]))).is_empty());
}

#[test]
fn test_migration_tolerates_no_workspace() {
    assert!(migrate_project_state(None).is_empty());
}

#[test]
#[serial]
fn test_migration_skips_project_without_state_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let p = Project {
        id: "x".to_string(),
        name: Some("x".to_string()),
        project_dir: String::new(),
        current: true,
        cwd: Some(s(tmp.path())),
        ..Default::default()
    };
    assert!(migrate_project_state(Some(&ws(vec![p]))).is_empty());
}

// --------------------------------------------------------------------------
// ingest destination
// --------------------------------------------------------------------------

/// Documents must be *created* under project_dir, not created then moved.
///
/// The regression: `add_document` preferred `<cwd>/documents` and only fell
/// back to `documents_dir`, so an uploaded file landed in the user's folder —
/// outside the framework home, where it cannot follow SLP_HOME or the App
/// Sandbox container.
mod test_document_ingest_destination {
    use super::*;

    use std::path::PathBuf;
    use std::sync::Arc;

    use parking_lot::RwLock;
    use smartloop::config::supported_models::ModelClass;
    use smartloop::model::downloader::slp_downloader::{ProgressCallback, SLPDownloader, SharedWorkspace};
    use smartloop::model::generic::GenericModel;
    use smartloop::model::slp_model::{DocumentSource, SLPModel};
    use smartloop::types::{ModelCapability, ModelType};
    use smartloop::utils::log_utils::LoggerLike;

    /// `m.logger = MagicMock()`.
    struct NullLogger;

    impl LoggerLike for NullLogger {
        fn info(&self, _msg: &str) {}
        fn warning(&self, _msg: &str) {}
        fn debug(&self, _msg: &str) {}
        fn error(&self, _msg: &str) {}
    }

    /// Only the ingest path is under test; nothing is ever downloaded.
    struct NoDownloader;

    impl SLPDownloader for NoDownloader {
        fn download(&self, model_id: &str, _progress_callback: Option<ProgressCallback>) -> anyhow::Result<PathBuf> {
            anyhow::bail!("no download expected for {model_id}")
        }
    }

    fn stub_model_id() -> String {
        "test/model".to_string()
    }
    fn stub_model_version() -> i64 {
        0
    }
    fn stub_param_size() -> f64 {
        0.0
    }
    fn stub_capabilities() -> Vec<ModelCapability> {
        vec![ModelCapability::General]
    }
    fn stub_thinking_default() -> bool {
        false
    }
    fn stub_vision_handler() -> Option<String> {
        None
    }

    /// The concrete stand-in: `GenericModel` is abstract in Python; only the
    /// ingest path is under test.
    fn model(tmp_path: &Path, cwd: Option<&Path>) -> (GenericModel, Project) {
        let state = tmp_path.join("state").join("p1");
        fs::create_dir_all(&state).unwrap();
        let project = Project {
            id: "p1".to_string(),
            name: Some("proj".to_string()),
            project_dir: s(&state),
            current: true,
            cwd: cwd.map(s),
            ..Default::default()
        };
        // `m.workspace.current_project = project`.
        let workspace: SharedWorkspace =
            Arc::new(RwLock::new(Workspace::new("ws").with_projects(vec![project.clone()])));
        let class = ModelClass {
            class_name: "_Concrete",
            model_id: stub_model_id,
            model_version: stub_model_version,
            param_size_in_b: stub_param_size,
            default_capabilities: stub_capabilities,
            thinking_default: stub_thinking_default,
            vision_chat_handler: stub_vision_handler,
        };
        let m = GenericModel::new(workspace, Arc::new(NoDownloader), Arc::new(NullLogger), class, ModelType::Text);
        (m, project)
    }

    #[test]
    #[serial]
    fn test_ingest_writes_under_project_dir() {
        let _reset = ProcessScopeReset::new();
        let tmp = tempfile::tempdir().unwrap();
        // Anything persisted along the way stays inside the temp dir.
        let _home = EnvGuard::set("SLP_HOME", tmp.path().join("slp_home"));
        let src = tmp.path().join("Invoice.pdf");
        fs::write(&src, b"%PDF-1.4 fake").unwrap();
        let user_folder = tmp.path().join("usercode");
        fs::create_dir(&user_folder).unwrap();
        let (m, project) = model(tmp.path(), Some(&user_folder));

        m.add_document(DocumentSource::Path(src), None, Some(&s(&user_folder)), false).unwrap();

        assert!(Path::new(&join(&project.documents_dir(), "Invoice.pdf")).is_file());
    }

    #[test]
    #[serial]
    fn test_ingest_never_touches_the_user_folder() {
        let _reset = ProcessScopeReset::new();
        let tmp = tempfile::tempdir().unwrap();
        // Anything persisted along the way stays inside the temp dir.
        let _home = EnvGuard::set("SLP_HOME", tmp.path().join("slp_home"));
        let src = tmp.path().join("Invoice.pdf");
        fs::write(&src, b"%PDF-1.4 fake").unwrap();
        let user_folder = tmp.path().join("usercode");
        fs::create_dir(&user_folder).unwrap();
        let (m, _) = model(tmp.path(), Some(&user_folder));

        m.add_document(DocumentSource::Path(src), None, Some(&s(&user_folder)), false).unwrap();

        assert!(!user_folder.join("documents").exists());
    }

    /// Even an explicit cwd argument must not move project state.
    #[test]
    #[serial]
    fn test_request_cwd_cannot_redirect_storage() {
        let _reset = ProcessScopeReset::new();
        let tmp = tempfile::tempdir().unwrap();
        // Anything persisted along the way stays inside the temp dir.
        let _home = EnvGuard::set("SLP_HOME", tmp.path().join("slp_home"));
        let src = tmp.path().join("Invoice.pdf");
        fs::write(&src, b"%PDF-1.4 fake").unwrap();
        let elsewhere = tmp.path().join("elsewhere");
        fs::create_dir(&elsewhere).unwrap();
        let (m, project) = model(tmp.path(), None);

        m.add_document(DocumentSource::Path(src), None, Some(&s(&elsewhere)), false).unwrap();

        assert!(Path::new(&join(&project.documents_dir(), "Invoice.pdf")).is_file());
        assert!(!elsewhere.join("documents").exists());
    }
}

// --------------------------------------------------------------------------
// self-containment: copy <home_dir> anywhere and it runs as-is
// --------------------------------------------------------------------------

/// config.json stores absolute paths; none of them may be trusted.
///
/// Copy the framework home to another machine, user, or sandbox container and
/// every persisted path is stale.  `Workspace::load` must re-derive them.
mod test_relocatable_framework {
    use super::*;

    use serde_json::{Value, json};
    use smartloop::workspace::workspace::set_settings_home_dir;

    /// Write a config.json describing one project, plus its files on disk.
    fn seed(home: &Path, doc_name: &str) -> std::path::PathBuf {
        let (ws_id, proj_id) = ("ws-1", "proj-1");
        let proj_dir = home.join(ws_id).join(proj_id);
        fs::create_dir_all(proj_dir.join("skills").join("greet")).unwrap();
        fs::write(proj_dir.join("skills").join("greet").join("SKILL.md"), "---\nname: greet\n---\nSay hello").unwrap();
        fs::create_dir_all(proj_dir.join("documents")).unwrap();
        fs::write(proj_dir.join("documents").join(doc_name), "data").unwrap();
        let cfg = json!({
            "id": "",
            "workspace": {
                "id": ws_id,
                "workspace_dir": format!("/OLD/HOME/{ws_id}"),
                "models_dir": format!("/OLD/HOME/{ws_id}/models"),
                "projects": [{
                    "id": proj_id,
                    "project_dir": format!("/OLD/HOME/{ws_id}/{proj_id}"),
                    "current": true,
                    "name": "proj",
                    "cwd": "/OLD/USER/FOLDER/that/is/gone",
                    "documents": [{
                        "id": "d1",
                        "path": format!("/OLD/HOME/{ws_id}/{proj_id}/documents/{doc_name}"),
                        "name": doc_name,
                        "status": "indexed",
                    }],
                }],
            },
        });
        fs::write(home.join("config.json"), serde_json::to_string(&cfg).unwrap()).unwrap();
        proj_dir
    }

    /// `_settings.slp_home = home` and `Workspace.config_file = home/config.json`
    /// for the duration of one `Workspace.load()`.
    fn load(home: &Path) -> Workspace {
        let _reset = ProcessScopeReset::new();
        let _env = [
            EnvGuard::remove("SLP_HOME"),
            EnvGuard::remove("SLP_CONFIG_FILE"),
            EnvGuard::remove("SLP_LEGACY_CONFIG"),
        ];
        set_settings_home_dir(Some(s(home)));
        Workspace::set_config_file(Some(s(&home.join("config.json"))));
        let ws = Workspace::load();
        Workspace::set_config_file(None);
        set_settings_home_dir(None);
        ws.expect("a workspace on disk")
    }

    fn rewrite_config(home: &Path, edit: impl FnOnce(&mut Value)) {
        let path = home.join("config.json");
        let mut cfg: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        edit(&mut cfg);
        fs::write(&path, serde_json::to_string(&cfg).unwrap()).unwrap();
    }

    #[test]
    #[serial]
    fn test_project_dir_is_rebased_onto_the_new_home() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("copied_home");
        fs::create_dir(&home).unwrap();
        let proj_dir = seed(&home, "Invoice.pdf");

        let ws = load(&home);

        assert_eq!(ws.projects[0].project_dir, s(&proj_dir));
    }

    #[test]
    #[serial]
    fn test_skills_are_found_after_relocation() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("copied_home");
        fs::create_dir(&home).unwrap();
        seed(&home, "Invoice.pdf");

        let ws = load(&home);

        assert_eq!(skill_names(&ws.projects[0]), vec!["greet"]);
    }

    #[test]
    #[serial]
    fn test_document_paths_are_rebased() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("copied_home");
        fs::create_dir(&home).unwrap();
        let proj_dir = seed(&home, "Invoice.pdf");

        let ws = load(&home);

        let doc = &ws.projects[0].documents[0];
        assert_eq!(doc.path, s(&proj_dir.join("documents").join("Invoice.pdf")));
        assert!(Path::new(&doc.path).is_file());
    }

    /// The stale client path survives in the config but locates nothing.
    #[test]
    #[serial]
    fn test_a_dead_user_cwd_does_not_break_anything() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("copied_home");
        fs::create_dir(&home).unwrap();
        seed(&home, "Invoice.pdf");

        let ws = load(&home);
        let project = &ws.projects[0];

        assert!(!Path::new(&project.work_dir().unwrap()).exists());
        assert_eq!(skill_names(project), vec!["greet"]);
        assert!(project.skills_dir().starts_with(&s(&home)));
        assert!(project.documents_dir().starts_with(&s(&home)));
    }

    /// After a migration both copies exist; the framework's must win.
    ///
    /// Otherwise the pointer stays on the user-space copy and the install
    /// keeps depending on a directory outside <home_dir>.
    #[test]
    #[serial]
    fn test_framework_copy_wins_over_a_still_valid_outside_path() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("copied_home");
        fs::create_dir(&home).unwrap();
        let proj_dir = seed(&home, "Invoice.pdf");
        let external = tmp.path().join("leftover").join("Invoice.pdf");
        fs::create_dir(external.parent().unwrap()).unwrap();
        fs::write(&external, "stale copy").unwrap();

        rewrite_config(&home, |cfg| {
            cfg["workspace"]["projects"][0]["documents"][0]["path"] = json!(s(&external));
        });

        let ws = load(&home);

        assert_eq!(ws.projects[0].documents[0].path, s(&proj_dir.join("documents").join("Invoice.pdf")));
    }

    /// A document with no counterpart in the project is left untouched.
    #[test]
    #[serial]
    fn test_unknown_document_keeps_its_path() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("copied_home");
        fs::create_dir(&home).unwrap();
        seed(&home, "Invoice.pdf");
        let external = tmp.path().join("elsewhere.pdf");
        fs::write(&external, "x").unwrap();

        rewrite_config(&home, |cfg| {
            let doc = &mut cfg["workspace"]["projects"][0]["documents"][0];
            doc["path"] = json!(s(&external));
            doc["name"] = json!("elsewhere.pdf");
        });

        let ws = load(&home);

        assert_eq!(ws.projects[0].documents[0].path, s(&external));
    }
}

/// Skills come from the project's own dir, via both readers.
///
/// Two readers exist — `Project::get_skills` (chat/orchestrator) and
/// `ModelService::read_skills_from_disk` (API/UI).  They must agree, and
/// neither may leak another project's skills.
mod test_skills_are_read_per_project {
    use super::*;

    use smartloop::api::services::model_service::ModelService;
    use smartloop::workspace::types::Skill;

    fn proj(tmp_path: &Path, pid: &str, names: &[&str]) -> Project {
        let p = Project {
            id: pid.to_string(),
            name: Some(pid.to_string()),
            current: false,
            project_dir: s(&tmp_path.join("state").join(pid)),
            cwd: Some("/some/client/path/that/is/ignored".to_string()),
            ..Default::default()
        };
        for n in names {
            let d = Path::new(&p.skills_dir()).join(n);
            fs::create_dir_all(&d).unwrap();
            fs::write(d.join("SKILL.md"), format!("---\nname: {n}\ndescription: d\n---\nbody")).unwrap();
        }
        p
    }

    fn reader_names(p: &Project) -> Vec<String> {
        ModelService::read_skills_from_disk(p).into_iter().map(|s| s.name).collect()
    }

    #[test]
    fn test_each_project_sees_only_its_own() {
        let tmp = tempfile::tempdir().unwrap();
        let a = proj(tmp.path(), "a", &["fetch-url", "good-morning"]);
        let b = proj(tmp.path(), "b", &["other"]);

        assert_eq!(skill_names(&a), vec!["fetch-url", "good-morning"]);
        assert_eq!(skill_names(&b), vec!["other"]);
    }

    #[test]
    fn test_both_readers_agree() {
        let tmp = tempfile::tempdir().unwrap();
        let p = proj(tmp.path(), "a", &["fetch-url", "good-morning"]);

        assert_eq!(skill_names(&p), reader_names(&p));
    }

    /// No cache anywhere — the next read picks it up.
    #[test]
    fn test_a_skill_created_now_is_visible_immediately() {
        let tmp = tempfile::tempdir().unwrap();
        let p = proj(tmp.path(), "a", &[]);
        assert!(p.get_skills().is_empty());

        ModelService::write_skill_md(
            &p,
            &Skill {
                id: "s".to_string(),
                name: "fetch-url".to_string(),
                description: Some("d".to_string()),
                content: "body".to_string(),
                system: false,
            },
        )
        .unwrap();

        assert_eq!(skill_names(&p), vec!["fetch-url"]);
        assert_eq!(reader_names(&p), vec!["fetch-url"]);
    }

    #[test]
    fn test_writes_land_in_the_project_dir_not_the_client_path() {
        let tmp = tempfile::tempdir().unwrap();
        let p = proj(tmp.path(), "a", &[]);
        ModelService::write_skill_md(
            &p,
            &Skill {
                id: "s".to_string(),
                name: "fetch-url".to_string(),
                content: "body".to_string(),
                ..Default::default()
            },
        )
        .unwrap();

        assert!(Path::new(&p.skills_dir()).join("fetch-url").join("SKILL.md").is_file());
        assert!(!Path::new("/some/client/path/that/is/ignored").exists());
    }
}
