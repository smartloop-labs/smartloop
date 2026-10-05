//! Port of `tests/unit/test_project_state_split.py`.
//!
//! A project's state lives in the project's directory, not in one shared file.
//!
//! `config.json` used to hold every project. That made a save a
//! whole-workspace snapshot: two writers each holding a `Workspace` would have
//! the last one silently discard the other project's newly indexed documents
//! and MCP registrations — the write-clobber hazard that had to go before two
//! processes could run side by side.

use std::fs;
use std::path::Path;

use serde_json::{Value, json};
use serial_test::serial;

use smartloop::workspace::context::{delegate_projects, own_project, pin_project};
use smartloop::workspace::migrations::{self, MONOLITH_BACKUP_SUFFIX, SplitResult, split_project_state};
use smartloop::workspace::workspace::{
    MCPServer, Project, Workspace, project_state_path, read_project_state,
};

use crate::shared::workspace_support::{EnvGuard, LogCapture, TempHome};

fn join(base: &str, part: &str) -> String {
    Path::new(base).join(part).to_string_lossy().into_owned()
}

fn workspace_with(home: &TempHome, project_ids: &[&str]) -> Workspace {
    let ws_dir = home.join("ws-1");
    Workspace::new("ws-1")
        .with_workspace_dir(ws_dir.clone())
        .with_models_dir(join(&ws_dir, "models"))
        .with_projects(
            project_ids
                .iter()
                .map(|pid| Project {
                    id: pid.to_string(),
                    project_dir: join(&ws_dir, pid),
                    current: *pid == project_ids[0],
                    name: Some(title(pid)),
                    sessions: vec![format!("{pid}-session")],
                    mcp_servers: vec![MCPServer {
                        id: format!("{pid}-srv"),
                        name: format!("{pid} server"),
                        ..Default::default()
                    }],
                    ..Default::default()
                })
                .collect(),
        )
}

fn workspace(home: &TempHome) -> Workspace {
    workspace_with(home, &["alpha", "beta"])
}

/// `str.title()` for a single lowercase word.
fn title(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

fn config(home: &TempHome) -> Value {
    serde_json::from_str(&fs::read_to_string(home.path().join("config.json")).unwrap()).unwrap()
}

fn state_of(home: &TempHome, project_id: &str) -> serde_json::Map<String, Value> {
    read_project_state(&join(&home.join("ws-1"), project_id)).expect("a project state")
}

fn load() -> Workspace {
    Workspace::load().expect("a workspace on disk")
}

fn current_id(ws: &Workspace) -> String {
    ws.current_project().expect("a selected project").id.clone()
}

mod what_ends_up_where {
    use super::*;

    #[test]
    #[serial]
    fn test_config_json_keeps_only_a_thin_index() {
        let home = TempHome::new();
        workspace(&home).save().unwrap();

        let entry = config(&home)["workspace"]["projects"][0].clone();
        assert_eq!(entry["id"], "alpha");
        assert_eq!(entry["current"], true);
        assert!(entry["project_dir"].as_str().unwrap().ends_with(&join("ws-1", "alpha")));
        for bulk in ["documents", "skills", "mcp_servers", "sessions", "model_settings", "web_search_enabled"] {
            assert!(entry.get(bulk).is_none(), "{bulk} leaked into the index");
        }
    }

    #[test]
    #[serial]
    fn test_the_project_keeps_its_own_full_record() {
        let home = TempHome::new();
        workspace(&home).save().unwrap();

        let state = state_of(&home, "alpha");
        assert_eq!(state["mcp_servers"][0]["name"], "alpha server");
        assert_eq!(state["sessions"], json!(["alpha-session"]));
        // The whole record, so the directory is self-describing.
        assert_eq!(state["id"], "alpha");
        assert_eq!(state["name"], "Alpha");
    }

    #[test]
    #[serial]
    fn test_the_file_is_stamped_with_a_version() {
        let home = TempHome::new();
        workspace(&home).save().unwrap();

        let raw: Value = serde_json::from_str(
            &fs::read_to_string(project_state_path(&join(&home.join("ws-1"), "alpha"))).unwrap(),
        )
        .unwrap();
        assert_eq!(raw["version"], 1);
    }

    #[test]
    #[serial]
    fn test_workspace_level_state_stays_in_config_json() {
        let home = TempHome::new();
        let mut ws = workspace(&home);
        ws.add_model("sl-mini", "smartloop-ai/sl-mini", None, true);
        ws.save().unwrap();

        let saved = config(&home)["workspace"].clone();
        assert_eq!(saved["id"], "ws-1");
        let names: Vec<&str> =
            saved["models"].as_array().unwrap().iter().map(|m| m["model_name"].as_str().unwrap()).collect();
        assert_eq!(names, vec!["sl-mini"]);
    }
}

mod it_comes_back_the_same {
    use super::*;

    #[test]
    #[serial]
    fn test_a_saved_workspace_round_trips() {
        let home = TempHome::new();
        workspace(&home).save().unwrap();

        let reloaded = load();

        assert_eq!(reloaded.projects.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(), vec!["alpha", "beta"]);
        let alpha = &reloaded.projects[0];
        assert_eq!(alpha.name.as_deref(), Some("Alpha"));
        assert_eq!(alpha.sessions, vec!["alpha-session"]);
        assert_eq!(alpha.mcp_servers[0].name, "alpha server");
        assert_eq!(current_id(&reloaded), "alpha");
    }

    /// project.json carries `current` too; config.json must still win.
    #[test]
    #[serial]
    fn test_the_selection_is_the_workspace_s_to_own() {
        let home = TempHome::new();
        let mut ws = workspace(&home);
        ws.save().unwrap();

        // Move the selection and rewrite only the workspace record, the way a
        // supervisor would.
        for project in &mut ws.projects {
            project.current = project.id == "beta";
        }
        ws.save().unwrap();

        assert_eq!(current_id(&load()), "beta");
    }

    /// A half-migrated workspace must not lose the projects not yet split.
    #[test]
    #[serial]
    fn test_a_project_with_no_file_of_its_own_still_loads() {
        let home = TempHome::new();
        workspace(&home).save().unwrap();
        fs::remove_file(project_state_path(&join(&home.join("ws-1"), "beta"))).unwrap();

        let reloaded = load();

        assert_eq!(reloaded.projects.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(), vec!["alpha", "beta"]);
        assert_eq!(reloaded.projects[0].sessions, vec!["alpha-session"]);
        assert_eq!(reloaded.projects[1].name.as_deref(), Some("Beta")); // from the thin index
    }

    #[test]
    #[serial]
    fn test_a_corrupt_project_file_does_not_take_the_workspace_down() {
        let home = TempHome::new();
        workspace(&home).save().unwrap();
        let path = project_state_path(&join(&home.join("ws-1"), "beta"));
        fs::write(&path, "{not json").unwrap();

        let reloaded = load();

        assert_eq!(reloaded.projects.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(), vec!["alpha", "beta"]);
        assert_eq!(reloaded.projects[0].mcp_servers[0].name, "alpha server");
    }
}

/// The clobber the split exists to remove.
mod one_writer_per_project {
    use super::*;

    #[test]
    #[serial]
    fn test_two_workspaces_saving_do_not_discard_each_other_s_project() {
        let home = TempHome::new();
        workspace(&home).save().unwrap();

        // Two processes, each with its own full copy of the workspace — the
        // exact shape that used to lose data.
        let (mut one, mut two) = (load(), load());
        one.projects[0].sessions.push("added-by-one".to_string());
        two.projects[1].sessions.push("added-by-two".to_string());

        pin_project(Some("alpha"));
        one.save().unwrap();
        pin_project(Some("beta"));
        two.save().unwrap();
        pin_project(None);

        let reloaded = load();
        assert_eq!(reloaded.projects[0].sessions, vec!["alpha-session", "added-by-one"]);
        assert_eq!(reloaded.projects[1].sessions, vec!["beta-session", "added-by-two"]);
    }

    #[test]
    #[serial]
    fn test_a_pinned_process_writes_only_its_own_project() {
        let home = TempHome::new();
        workspace(&home).save().unwrap();
        let before = fs::read_to_string(home.path().join("config.json")).unwrap();

        let mut ws = load();
        pin_project(Some("beta"));
        ws.projects[0].sessions.push("should-not-persist".to_string());
        ws.projects[1].sessions.push("should-persist".to_string());
        ws.save().unwrap();

        assert_eq!(fs::read_to_string(home.path().join("config.json")).unwrap(), before); // untouched
        let alpha = state_of(&home, "alpha");
        let beta = state_of(&home, "beta");
        assert_eq!(alpha["sessions"], json!(["alpha-session"]));
        assert_eq!(beta["sessions"], json!(["beta-session", "should-persist"]));
    }

    #[test]
    #[serial]
    fn test_a_pin_to_an_unknown_project_writes_nothing() {
        let home = TempHome::new();
        workspace(&home).save().unwrap();
        let before = fs::read_to_string(home.path().join("config.json")).unwrap();

        let caplog = LogCapture::start();
        let ws = load();
        pin_project(Some("not-a-project"));
        ws.save().unwrap();

        assert_eq!(fs::read_to_string(home.path().join("config.json")).unwrap(), before);
        assert!(caplog.text().contains("not in this workspace"));
    }
}

/// The clobber, one tier up.
///
/// A supervisor holds one `Workspace` for the life of the process and saves it
/// whenever anything workspace-level changes.  Writing every project's file
/// from that snapshot would republish it over the documents and MCP servers
/// the children have indexed since — the same loss the split removed between
/// two peers, reappearing between a parent and its children.
mod the_supervisor_does_not_write_what_it_delegated {
    use super::*;

    #[test]
    #[serial]
    fn test_it_writes_the_workspace_record_and_nothing_else() {
        let home = TempHome::new();
        workspace(&home).save().unwrap();

        let mut supervisor = load();
        delegate_projects(true);

        // A child registers a server while the supervisor holds its snapshot.
        let mut child = load();
        pin_project(Some("alpha"));
        child.projects[0].mcp_servers.push(MCPServer {
            id: "new".to_string(),
            name: "added by the child".to_string(),
            ..Default::default()
        });
        child.save().unwrap();
        pin_project(None);

        // Anything workspace-level then makes the supervisor save.
        supervisor.projects[1].current = true;
        supervisor.projects[0].current = false;
        supervisor.save().unwrap();

        let reloaded = load();
        assert_eq!(
            reloaded.projects[0].mcp_servers.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            vec!["alpha server", "added by the child"]
        );
        assert_eq!(current_id(&reloaded), "beta"); // the record was written
    }

    /// Create, import and delete write a file no child owns yet.
    #[test]
    #[serial]
    fn test_a_claimed_project_is_still_written() {
        let home = TempHome::new();
        workspace(&home).save().unwrap();

        let mut ws = load();
        delegate_projects(true);
        ws.projects[0].sessions.push("claimed".to_string());
        ws.projects[1].sessions.push("not-claimed".to_string());
        {
            let _owned = own_project(Some("alpha"));
            ws.save().unwrap();
        }

        let reloaded = load();
        assert_eq!(reloaded.projects[0].sessions, vec!["alpha-session", "claimed"]);
        assert_eq!(reloaded.projects[1].sessions, vec!["beta-session"]);
    }

    /// The single-server fallback and the CLI own every project there is.
    #[test]
    #[serial]
    fn test_an_unsupervised_process_still_writes_everything() {
        let home = TempHome::new();
        workspace(&home).save().unwrap();

        let mut ws = load();
        ws.projects[0].sessions.push("added".to_string());
        ws.projects[1].sessions.push("added".to_string());
        ws.save().unwrap();

        let reloaded = load();
        assert_eq!(reloaded.projects[0].sessions, vec!["alpha-session", "added"]);
        assert_eq!(reloaded.projects[1].sessions, vec!["beta-session", "added"]);
    }
}

mod migrating_off_the_monolith {
    use super::*;

    /// `ws.model_dump(mode="json")` minus the computed `current_project`.
    fn dump(ws: &Workspace) -> Value {
        serde_json::to_value(ws).unwrap()
    }

    fn write_monolith(home: &TempHome) -> Workspace {
        let ws = workspace(home);
        let document = json!({"id": "", "workspace": dump(&ws)});
        fs::write(home.path().join("config.json"), serde_json::to_string_pretty(&document).unwrap()).unwrap();
        ws
    }

    #[test]
    #[serial]
    fn test_it_splits_an_existing_monolith() {
        let home = TempHome::new();
        write_monolith(&home);
        let mut ws = load();

        let result = split_project_state(Some(&mut ws));

        let mut migrated = result.migrated.clone();
        migrated.sort();
        assert_eq!(migrated, vec!["alpha", "beta"]);
        assert_eq!(state_of(&home, "alpha")["mcp_servers"][0]["name"], "alpha server");
        assert!(config(&home)["workspace"]["projects"][0].get("mcp_servers").is_none());
    }

    #[test]
    #[serial]
    fn test_nothing_is_lost_in_the_split() {
        let home = TempHome::new();
        write_monolith(&home);
        let mut before = load();

        split_project_state(Some(&mut before));
        Workspace::reset();
        let after = load();

        let dumped = |ws: &Workspace| -> Vec<Value> {
            ws.projects.iter().map(|p| serde_json::to_value(p).unwrap()).collect()
        };
        assert_eq!(dumped(&after), dumped(&before));
    }

    #[test]
    #[serial]
    fn test_the_pre_split_config_is_kept_for_a_rollback() {
        let home = TempHome::new();
        write_monolith(&home);
        let original = fs::read_to_string(home.path().join("config.json")).unwrap();

        let result = split_project_state(Some(&mut load()));

        let expected = format!("{}{}", home.join("config.json"), MONOLITH_BACKUP_SUFFIX);
        assert_eq!(result.backup.as_deref(), Some(expected.as_str()));
        assert_eq!(fs::read_to_string(result.backup.unwrap()).unwrap(), original);
    }

    #[test]
    #[serial]
    fn test_a_second_run_changes_nothing() {
        let home = TempHome::new();
        write_monolith(&home);
        split_project_state(Some(&mut load()));
        Workspace::reset();

        let result = split_project_state(Some(&mut load()));

        assert!(result.migrated.is_empty());
    }

    #[test]
    #[serial]
    fn test_a_second_run_does_not_overwrite_the_backup() {
        let home = TempHome::new();
        write_monolith(&home);
        let original = fs::read_to_string(home.path().join("config.json")).unwrap();
        split_project_state(Some(&mut load()));
        Workspace::reset();

        // Force another pass by removing one project's file.
        fs::remove_file(project_state_path(&join(&home.join("ws-1"), "beta"))).unwrap();
        split_project_state(Some(&mut load()));

        let backup = format!("{}{}", home.join("config.json"), MONOLITH_BACKUP_SUFFIX);
        assert_eq!(fs::read_to_string(backup).unwrap(), original);
    }

    #[test]
    #[serial]
    fn test_an_empty_workspace_is_not_an_error() {
        let home = TempHome::new();
        let mut ws = Workspace::new("ws-1").with_workspace_dir(home.join("ws-1"));
        assert_eq!(split_project_state(Some(&mut ws)), SplitResult { migrated: vec![], backup: None });
    }

    #[test]
    #[serial]
    fn test_no_workspace_is_not_an_error() {
        let _home = TempHome::new();
        assert_eq!(split_project_state(None), SplitResult { migrated: vec![], backup: None });
    }
}

mod the_rollback_switch {
    use super::*;

    #[test]
    #[serial]
    fn test_legacy_mode_writes_the_single_document_again() {
        let home = TempHome::new();
        let _legacy = EnvGuard::set("SLP_LEGACY_CONFIG", "1");

        workspace(&home).save().unwrap();

        let entry = config(&home)["workspace"]["projects"][0].clone();
        assert_eq!(entry["mcp_servers"][0]["name"], "alpha server");
        assert!(!Path::new(&project_state_path(&join(&home.join("ws-1"), "alpha"))).exists());
    }

    #[test]
    #[serial]
    fn test_legacy_mode_ignores_project_files_already_written() {
        let home = TempHome::new();
        workspace(&home).save().unwrap(); // split layout on disk
        // Restore the pre-split document the way a rollback would.
        let ws = workspace(&home);
        let mut dumped = serde_json::to_value(&ws).unwrap();
        dumped["projects"][0]["sessions"] = json!(["from-the-backup"]);
        fs::write(
            home.path().join("config.json"),
            serde_json::to_string(&json!({"id": "", "workspace": dumped})).unwrap(),
        )
        .unwrap();

        let _legacy = EnvGuard::set("SLP_LEGACY_CONFIG", "1");
        let reloaded = load();

        assert_eq!(reloaded.projects[0].sessions, vec!["from-the-backup"]);
    }

    #[test]
    #[serial]
    fn test_legacy_mode_skips_the_migration_entirely() {
        let home = TempHome::new();
        let _legacy = EnvGuard::set("SLP_LEGACY_CONFIG", "1");
        let mut ws = workspace(&home);

        assert!(split_project_state(Some(&mut ws)).migrated.is_empty());
    }
}

mod paths_resolve_late {
    use super::*;

    #[test]
    #[serial]
    fn test_config_file_follows_the_home_it_is_asked_in() {
        let home = TempHome::new();
        assert_eq!(Workspace::config_file(), home.join("config.json"));

        let tmp = tempfile::tempdir().unwrap();
        let other = tmp.path().join("elsewhere");
        let _other_home = EnvGuard::set("SLP_HOME", &other);
        assert_eq!(Workspace::config_file(), other.join("config.json").to_string_lossy());
    }

    #[test]
    #[serial]
    fn test_an_explicit_override_wins() {
        let _home = TempHome::new();
        let tmp = tempfile::tempdir().unwrap();
        let pinned = tmp.path().join("pinned.json");
        let _override = EnvGuard::set("SLP_CONFIG_FILE", &pinned);
        assert_eq!(Workspace::config_file(), pinned.to_string_lossy());
    }

    /// Tests override it by assignment; the resolution must not block that.
    #[test]
    #[serial]
    fn test_it_is_still_patchable() {
        let _home = TempHome::new();
        Workspace::set_config_file(Some("/somewhere/else.json".to_string()));
        assert_eq!(Workspace::config_file(), "/somewhere/else.json");
        Workspace::set_config_file(None);
    }
}

mod needs_split_detection {
    use super::*;

    #[test]
    #[serial]
    fn test_a_project_without_a_file_needs_one() {
        let home = TempHome::new();
        assert!(migrations::needs_split(&workspace(&home)));
    }

    #[test]
    #[serial]
    fn test_a_fully_split_workspace_does_not() {
        let home = TempHome::new();
        let ws = workspace(&home);
        ws.save().unwrap();
        assert!(!migrations::needs_split(&ws));
    }

    #[test]
    #[serial]
    fn test_a_project_with_no_directory_is_skipped() {
        let home = TempHome::new();
        let mut ws = workspace(&home);
        for project in &mut ws.projects {
            project.project_dir = String::new();
        }
        assert!(!migrations::needs_split(&ws));
    }
}
