//! Port of `tests/unit/test_project_state_refresh.py`.
//!
//! What a process holding a project sees after its owner writes to it.
//!
//! Under supervision a project's own state is written by the child pinned to
//! it, while the supervisor holds a copy taken when it first learned the
//! project existed.  The model host runs in the supervisor and resolves model
//! selection from that copy, so nothing invalidating it means a model disabled
//! in a project keeps answering for that project — observed live: the picker
//! showed one model, the host loaded the one that had just been turned off,
//! and only a restart reconciled them.

use std::fs;
use std::path::Path;
use std::time::Duration;

use serde_json::{Map, Value, json};
use serial_test::serial;

use smartloop::workspace::context::pin_project;
use smartloop::workspace::workspace::{Project, ProjectModel, Workspace, project_state_path};

use crate::shared::workspace_support::TempHome;

fn join(base: &str, part: &str) -> String {
    Path::new(base).join(part).to_string_lossy().into_owned()
}

fn title(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

fn workspace(home: &TempHome) -> Workspace {
    let ws_dir = home.join("ws-1");
    let ws = Workspace::new("ws-1")
        .with_workspace_dir(ws_dir.clone())
        .with_models_dir(join(&ws_dir, "models"))
        .with_projects(
            ["alpha", "beta"]
                .iter()
                .map(|pid| Project {
                    id: pid.to_string(),
                    project_dir: join(&ws_dir, pid),
                    current: *pid == "alpha",
                    name: Some(title(pid)),
                    model_settings: vec![
                        ProjectModel { enabled: true, ..ProjectModel::new("gemma4-e4b") },
                        ProjectModel { enabled: true, ..ProjectModel::new("ministral3-3b") },
                    ],
                    ..Default::default()
                })
                .collect(),
        );
    for project in &ws.projects {
        fs::create_dir_all(&project.project_dir).unwrap();
    }
    ws.save().unwrap();
    ws
}

/// Simulate the pinned child persisting its own project.json.
fn child_writes(home: &TempHome, project_id: &str, changes: Value) {
    let path = project_state_path(&join(&home.join("ws-1"), project_id));
    let mut doc: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    let project: &mut Map<String, Value> = doc["project"].as_object_mut().unwrap();
    for (key, value) in changes.as_object().unwrap() {
        project.insert(key.clone(), value.clone());
    }
    let before = fs::metadata(&path).unwrap().modified().unwrap();
    fs::write(&path, serde_json::to_string(&doc).unwrap()).unwrap();
    // Guarantee a distinct mtime even on a coarse-grained filesystem clock.
    let file = fs::File::options().write(true).open(&path).unwrap();
    let written = file.metadata().unwrap().modified().unwrap();
    file.set_modified(written.max(before) + Duration::from_millis(1)).unwrap();
}

fn load() -> Workspace {
    Workspace::load().expect("a workspace on disk")
}

mod picking_up_the_owners_write {
    use super::*;

    #[test]
    #[serial]
    fn test_a_disabled_model_is_seen() {
        let home = TempHome::new();
        let mut ws = workspace(&home);
        assert!(ws.effective_model_enabled("gemma4-e4b", Some(&ws.projects[0])));

        child_writes(
            &home,
            "alpha",
            json!({"model_settings": [
                {"model_name": "gemma4-e4b", "enabled": false},
                {"model_name": "ministral3-3b", "enabled": true},
            ]}),
        );

        assert!(ws.refresh_project("alpha"));
        assert!(!ws.effective_model_enabled("gemma4-e4b", Some(&ws.projects[0])));
        assert!(ws.effective_model_enabled("ministral3-3b", Some(&ws.projects[0])));
    }

    /// A caller already holding the Project must see the new state.
    ///
    /// `current_project` hands out the record itself, and the host resolves
    /// through it while the request runs.  Replacing the list entry would
    /// leave that caller reading a detached copy.
    #[test]
    #[serial]
    fn test_the_project_object_is_updated_in_place() {
        let home = TempHome::new();
        let mut ws = workspace(&home);
        let held: *const Project = &ws.projects[0];

        child_writes(&home, "alpha", json!({"model_settings": [{"model_name": "gemma4-e4b", "enabled": false}]}));
        ws.refresh_project("alpha");

        assert!(std::ptr::eq(held, &ws.projects[0]));
        let held = &ws.projects[0];
        assert_eq!(
            held.model_settings.iter().map(|m| (m.model_name.as_str(), m.enabled)).collect::<Vec<_>>(),
            vec![("gemma4-e4b", false)]
        );
    }

    /// The host calls this per frame; the usual cost must be one stat.
    #[test]
    #[serial]
    fn test_an_unchanged_file_is_not_re_read() {
        let home = TempHome::new();
        let mut ws = workspace(&home);
        assert!(ws.refresh_project("alpha")); // first look, never seen
        assert!(!ws.refresh_project("alpha"));
        assert!(!ws.refresh_project("alpha"));
    }

    /// `current` is workspace-owned; a project's own file cannot claim it.
    #[test]
    #[serial]
    fn test_the_selection_stays_the_workspaces_answer() {
        let home = TempHome::new();
        let mut ws = workspace(&home);
        assert!(ws.projects[0].current && !ws.projects[1].current);

        child_writes(&home, "beta", json!({"current": true, "name": "Renamed"}));
        ws.refresh_project("beta");

        assert!(!ws.projects[1].current, "a child claimed the selection");
        assert_eq!(ws.projects[1].name.as_deref(), Some("Renamed"), "its own fields still arrive");
    }

    #[test]
    #[serial]
    fn test_an_unknown_project_is_a_no_op() {
        let home = TempHome::new();
        let mut ws = workspace(&home);
        assert!(!ws.refresh_project("nope"));
        assert!(!ws.refresh_project(""));
    }

    #[test]
    #[serial]
    fn test_an_unreadable_file_does_not_raise() {
        let home = TempHome::new();
        let mut ws = workspace(&home);
        let path = project_state_path(&ws.projects[0].project_dir);
        fs::write(&path, "{not json").unwrap();
        assert!(!ws.refresh_project("alpha"));
        assert_eq!(ws.projects[0].name.as_deref(), Some("Alpha"), "the copy we have survives");
    }

    /// The mtime map is per instance.
    ///
    /// A shared one lets whichever instance looked first mark the file seen,
    /// and the other then skips a read it genuinely needs.
    #[test]
    #[serial]
    fn test_two_instances_keep_their_own_vintage() {
        let home = TempHome::new();
        let mut ws = workspace(&home);
        let mut other = load();
        child_writes(&home, "alpha", json!({"name": "Renamed"}));

        assert!(ws.refresh_project("alpha"));
        assert!(other.refresh_project("alpha"));
    }
}

/// The mirror: the workspace owner's write, reaching a running child.
///
/// Weights are workspace-wide — one shared `models_dir`, one registry in
/// `config.json`, written only by whoever owns that file.  A child holds the
/// workspace it loaded at spawn, and `ensure_workspace` re-reads from disk
/// only while no model pipeline exists, so a model downloaded meanwhile was
/// invisible to it: missing from `GET /v1/projects/<id>/models` and a 404 from
/// the `PATCH` that would enable it, both answered by the child.
mod seeing_a_model_registered_after_the_child_started {
    use super::*;

    /// A workspace with one model already registered, as on disk.
    fn seeded(home: &TempHome) -> Workspace {
        let mut ws = workspace(home);
        ws.add_model("gemma4-e4b", "google/gemma-4-e4b-it", None, true);
        ws.save().unwrap();
        ws
    }

    /// The process that owns config.json downloads and registers a model.
    fn owner_registers(model_name: &str) {
        let mut owner = load();
        owner.add_model(model_name, &format!("vendor/{model_name}"), None, true);
        owner.save().unwrap();
    }

    #[test]
    #[serial]
    fn test_the_new_model_arrives() {
        let home = TempHome::new();
        let mut ws = seeded(&home);
        assert!(ws.refresh_models()); // first look
        assert_eq!(ws.models.iter().map(|m| m.model_name.as_str()).collect::<Vec<_>>(), vec!["gemma4-e4b"]);

        owner_registers("ministral3-3b");

        assert!(ws.refresh_models());
        assert!(ws.get_model(Some("ministral3-3b")).is_some());
    }

    /// Nothing persisted is trusted as a location — same rule as load().
    #[test]
    #[serial]
    fn test_the_weights_are_found_under_this_home() {
        let home = TempHome::new();
        let mut ws = seeded(&home);
        ws.refresh_models();
        owner_registers("ministral3-3b");
        ws.refresh_models();

        assert_eq!(ws.get_model(Some("ministral3-3b")).unwrap().models_dir, ws.models_dir);
    }

    #[test]
    #[serial]
    fn test_an_unchanged_record_is_not_re_read() {
        let home = TempHome::new();
        let mut ws = seeded(&home);
        assert!(ws.refresh_models());
        assert!(!ws.refresh_models());
    }

    /// Only the registry is taken.
    ///
    /// A child holds project state its siblings own; adopting a sibling's
    /// record from the shared file would undo the split that keeps one process
    /// from overwriting another's work.
    #[test]
    #[serial]
    fn test_the_project_list_is_left_alone() {
        let home = TempHome::new();
        let mut ws = seeded(&home);
        ws.refresh_models();
        let before: Vec<String> = ws.projects.iter().map(|p| p.id.clone()).collect();
        let held: *const Project = &ws.projects[0];

        let mut owner = load();
        owner.projects[0].name = Some("RenamedByTheOwner".to_string());
        owner.save().unwrap();

        ws.refresh_models();
        assert_eq!(ws.projects.iter().map(|p| p.id.clone()).collect::<Vec<_>>(), before);
        assert!(std::ptr::eq(held, &ws.projects[0]));
        assert_eq!(ws.projects[0].name.as_deref(), Some("Alpha"));
    }

    #[test]
    #[serial]
    fn test_a_malformed_record_keeps_what_we_have() {
        let home = TempHome::new();
        let mut ws = seeded(&home);
        ws.refresh_models();
        fs::write(Workspace::config_file(), "{not json").unwrap();

        assert!(!ws.refresh_models());
        assert_eq!(ws.models.iter().map(|m| m.model_name.as_str()).collect::<Vec<_>>(), vec!["gemma4-e4b"]);
    }
}

/// Only a child re-reads it.
///
/// The owner must not re-read its own record out from under mutations it has
/// not saved yet — `/v1/init` registers a model well before it persists one,
/// and a refresh triggered by anything else would drop the registration.
mod who_refreshes_the_registry {
    use super::*;

    use std::sync::Arc;

    use parking_lot::RwLock;
    use smartloop::api::services::model_service::ModelService;
    use smartloop::config::app_settings::AppSettings;
    use smartloop::model::model_manager::Model;
    use smartloop::workspace::workspace::SharedWorkspace;

    /// A bare service whose model pipeline is "initialized" over `workspace`
    /// (so `ensure_model` has nothing to do and `ensure_workspace` returns it).
    fn service(workspace: &SharedWorkspace) -> Arc<ModelService> {
        let made = ModelService::new_bare();
        *made.model.write() = Some(Arc::new(Model::new("m", Arc::clone(workspace), None)));
        *made.model_name.write() = Some(AppSettings::new().default_model);
        made
    }

    #[test]
    #[serial]
    fn test_a_child_picks_up_the_download() {
        let home = TempHome::new();
        let mut ws = workspace(&home);
        ws.add_model("gemma4-e4b", "google/gemma-4-e4b-it", None, true);
        ws.save().unwrap();
        let shared: SharedWorkspace = Arc::new(RwLock::new(ws));
        let service = service(&shared);
        pin_project(Some("alpha"));
        service.ensure_workspace(None).unwrap();

        let mut owner = load(); // the owner, unpinned in its own process
        pin_project(None);
        owner.add_model("ministral3-3b", "vendor/ministral3-3b", None, true);
        owner.save().unwrap();
        pin_project(Some("alpha"));

        let workspace = service.ensure_workspace(None).unwrap();
        assert!(workspace.read().get_model(Some("ministral3-3b")).is_some());
    }

    /// The /v1/init shape: register, then something else writes the file.
    #[test]
    #[serial]
    fn test_the_owner_keeps_its_unsaved_registration() {
        let home = TempHome::new();
        let mut ws = workspace(&home);
        ws.add_model("gemma4-e4b", "google/gemma-4-e4b-it", None, true);
        ws.save().unwrap();
        let shared: SharedWorkspace = Arc::new(RwLock::new(ws));
        let service = service(&shared); // unpinned: this process owns the record
        shared.write().add_model("ministral3-3b", "vendor/ministral3-3b", None, true); // not saved yet

        // Someone else touches config.json in the meantime.
        let other = load();
        other.save().unwrap();

        let workspace = service.ensure_workspace(None).unwrap();
        assert!(
            workspace.read().get_model(Some("ministral3-3b")).is_some(),
            "the owner re-read its own record and lost an unsaved registration"
        );
    }
}
